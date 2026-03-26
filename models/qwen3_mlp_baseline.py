from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor


DEFAULT_MODEL_PATH = str(
    Path(__file__).resolve().parents[1] / "checkpoints" / "Qwen3-VL-2B-Instruct"
)
DEFAULT_PROMPT = (
    "You are an autonomous driving expert. Predict the future ego trajectory from the front camera image."
)


def masked_l2_loss(pred_traj: torch.Tensor, target_traj: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    per_step = ((pred_traj - target_traj) ** 2).sum(dim=-1)
    mask = target_mask.float()
    denom = mask.sum().clamp_min(1.0)
    return (per_step * mask).sum() / denom


class TrajectoryMLPHead(nn.Module):
    def __init__(self, hidden_size: int, future_steps: int, use_layernorm: bool = False):
        super().__init__()
        self.future_steps = future_steps
        layers = []
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_size))
        layers.extend(
            [
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, future_steps * 2),
            ]
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.layers(hidden).view(hidden.shape[0], self.future_steps, 2)


class QwenTrajectoryCollator:
    def __init__(self, processor, prompt: str = DEFAULT_PROMPT):
        self.processor = processor
        self.prompt = prompt

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        messages = []
        for sample in samples:
            image_path = sample["image_paths"]["CAM_FRONT"]
            messages.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_path},
                            {"type": "text", "text": self.prompt},
                        ],
                    }
                ]
            )

        texts = [
            self.processor.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            )
            for message in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        model_inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        batch = dict(model_inputs)
        batch["target_traj"] = torch.from_numpy(
            np.stack([sample["ego_future_traj"] for sample in samples], axis=0)
        ).float()
        batch["target_mask"] = torch.from_numpy(
            np.stack([sample["ego_future_mask"] for sample in samples], axis=0)
        ).float()
        batch["frame_token"] = [sample["frame_token"] for sample in samples]
        batch["frame_idx"] = [sample["frame_idx"] for sample in samples]
        batch["future_gt_boxes"] = [sample["future_gt_boxes"] for sample in samples]
        return batch


class Qwen3MLPTrajectoryModel(nn.Module):
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_PATH,
        future_steps: int = 12,
        use_layernorm: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.future_steps = future_steps
        self.use_layernorm = use_layernorm
        self.processor = AutoProcessor.from_pretrained(model_name)

        preferred_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        base_model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            dtype=preferred_dtype if torch.cuda.is_available() else torch.float32,
        )
        base_model.config.use_cache = False
        for param in base_model.parameters():
            param.requires_grad = False

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(base_model, lora_config)
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            self.model.enable_input_require_grads()

        self._unfreeze_visual_adaptors()

        hidden_size = int(self.model.config.text_config.hidden_size)
        self.head = TrajectoryMLPHead(
            hidden_size=hidden_size,
            future_steps=future_steps,
            use_layernorm=use_layernorm,
        )
        self.head.float()

    def _unfreeze_visual_adaptors(self):
        for name, param in self.model.named_parameters():
            if name.startswith("base_model.model.model.visual.merger") or name.startswith(
                "base_model.model.model.visual.deepstack_merger_list"
            ):
                param.requires_grad = True

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        model_inputs = {
            key: value
            for key, value in batch.items()
            if key in {"input_ids", "attention_mask", "mm_token_type_ids", "pixel_values", "image_grid_thw"}
        }
        outputs = self.model(
            **model_inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        hidden = outputs.hidden_states[-1]
        last_indices = batch["attention_mask"].sum(dim=-1) - 1
        features = hidden[torch.arange(hidden.shape[0], device=hidden.device), last_indices]
        pred_offsets = self.head(features.float())
        pred_traj = torch.cumsum(pred_offsets, dim=1)

        result = {
            "pred_offsets": pred_offsets,
            "pred_traj": pred_traj,
        }
        if "target_traj" in batch and "target_mask" in batch:
            result["loss"] = masked_l2_loss(pred_traj, batch["target_traj"], batch["target_mask"])
        return result

    def get_collator(self, prompt: str = DEFAULT_PROMPT) -> QwenTrajectoryCollator:
        return QwenTrajectoryCollator(self.processor, prompt=prompt)

    def get_model_config(self) -> Dict[str, Any]:
        return {
            "future_steps": self.future_steps,
            "use_layernorm": self.use_layernorm,
        }

    def get_trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        trainable = {}
        trainable_names = {name for name, param in self.named_parameters() if param.requires_grad}
        state_dict = self.state_dict()
        for name, tensor in state_dict.items():
            if name in trainable_names:
                trainable[name] = tensor.detach().cpu()
        return trainable

    def load_trainable_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.load_state_dict(state_dict, strict=False)
