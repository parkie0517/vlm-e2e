"""Qwen3-VL CoT + Diffusion Head for trajectory prediction.

Pipeline:
  Training:
    image + GT perception Q/A + motion prompt → VLM predicts command word → CE loss
    VLM hidden state → diffusion head(GT command, anchors) → L2 loss vs GT trajectory

  Eval (3-pass):
    Pass 1: image + perception Q → VLM generates perception answer
    Pass 2: image + perception Q/A + motion prompt → VLM generates command word
            + extract hidden state
    Pass 3: diffusion head(hidden_state, command, anchors) → trajectory (12, 2)
"""
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

from models.diffusion_head import TrajectoryDiffusionHead
from utils.trajectory_tokenizer import deterministic_perception_index


DEFAULT_MODEL_PATH = str(
    Path(__file__).resolve().parents[1] / "checkpoints" / "Qwen3-VL-2B-Instruct"
)
DEFAULT_PERCEPTION_TEMPLATE = "Perception question: {question}\nPerception answer: {answer}"
DEFAULT_MOTION_PROMPT = (
    "You are an autonomous driving expert. Based on the front camera image and the "
    "perception context, predict the high-level driving command. "
    "Answer with one word: right, left, or straight."
)

COMMAND_WORDS = ["right", "left", "straight"]
COMMAND_TO_INDEX = {w: i for i, w in enumerate(COMMAND_WORDS)}


# ---------------------------------------------------------------------------
# Training collator
# ---------------------------------------------------------------------------

class CoTDiffusionTrainCollator:
    """Builds training batches for the CoT + diffusion pipeline.

    For each sample:
      - Randomly picks one perception Q/A pair (GT answer)
      - Builds prompt: perception context + motion prompt
      - Target text: command word (right/left/straight)
      - Passes through GT trajectory, command index, masks for diffusion head
    """

    def __init__(
        self,
        processor,
        perception_template: str = DEFAULT_PERCEPTION_TEMPLATE,
        motion_prompt: str = DEFAULT_MOTION_PROMPT,
    ):
        self.processor = processor
        self.perception_template = perception_template
        self.motion_prompt = motion_prompt

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompt_messages = []
        full_texts = []
        prompt_texts = []

        for sample in samples:
            qa = random.choice(sample["qa"]["perception"])
            context = self.perception_template.format(question=qa["Q"], answer=qa["A"])
            prompt = f"{context}\n\n{self.motion_prompt}"
            prompt_messages.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": sample["image_paths"]["CAM_FRONT"]},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
            )

        prompt_texts = [
            self.processor.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            )
            for message in prompt_messages
        ]
        # Target: command word
        command_texts = [COMMAND_WORDS[sample["command"]] for sample in samples]
        full_texts = [f"{pt}{ct}" for pt, ct in zip(prompt_texts, command_texts)]

        image_inputs, video_inputs = process_vision_info(prompt_messages)
        prompt_inputs = self.processor(
            text=prompt_texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        full_inputs = self.processor(
            text=full_texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Label masking: only supervise the command word tokens
        labels = full_inputs["input_ids"].clone()
        labels[full_inputs["attention_mask"] == 0] = -100
        prompt_lengths = prompt_inputs["attention_mask"].sum(dim=-1)
        for batch_idx, prompt_len in enumerate(prompt_lengths.tolist()):
            labels[batch_idx, :prompt_len] = -100

        batch = dict(full_inputs)
        batch["labels"] = labels
        batch["target_traj"] = torch.from_numpy(
            np.stack([s["ego_future_traj"] for s in samples], axis=0)
        ).float()
        batch["target_mask"] = torch.from_numpy(
            np.stack([s["ego_future_mask"] for s in samples], axis=0)
        ).float()
        batch["command"] = torch.tensor([s["command"] for s in samples], dtype=torch.long)
        batch["future_gt_boxes"] = [s["future_gt_boxes"] for s in samples]
        batch["frame_token"] = [s["frame_token"] for s in samples]
        batch["frame_idx"] = [s["frame_idx"] for s in samples]
        return batch


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class Qwen3CoTDiffusionModel(nn.Module):
    def __init__(
        self,
        anchor_path: str,
        model_name: str = DEFAULT_MODEL_PATH,
        future_steps: int = 12,
        num_modes: int = 6,
        vlm_proj_dim: int = 256,
        diffusion_step_embed_dim: int = 256,
        down_dims: tuple = (128, 256),
        t_trunc: int = 40,
        num_inference_steps: int = 2,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.future_steps = future_steps
        self.vlm_proj_dim = vlm_proj_dim

        # --- VLM setup ---
        self.processor = AutoProcessor.from_pretrained(model_name)

        preferred_dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )
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
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(base_model, lora_config)
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            self.model.enable_input_require_grads()

        self._unfreeze_visual_adaptors()

        # --- VLM → Diffusion projector ---
        hidden_size = int(self.model.config.text_config.hidden_size)
        self.vlm_projector = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, vlm_proj_dim),
            nn.GELU(),
        )
        self.vlm_projector.float()

        # --- Diffusion head ---
        self.diffusion_head = TrajectoryDiffusionHead(
            anchor_path=anchor_path,
            future_steps=future_steps,
            num_modes=num_modes,
            global_cond_dim=vlm_proj_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            t_trunc=t_trunc,
            num_inference_steps=num_inference_steps,
        )
        self.diffusion_head.float()

    def _unfreeze_visual_adaptors(self):
        for name, param in self.model.named_parameters():
            if name.startswith("base_model.model.model.visual.merger") or name.startswith(
                "base_model.model.model.visual.deepstack_merger_list"
            ):
                param.requires_grad = True

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Training forward: VLM CE loss + diffusion L2 loss.

        Avoids passing labels to the model (which allocates full vocab logits).
        Instead we extract hidden states and compute CE on command tokens only.
        """
        model_inputs = {
            k: v for k, v in batch.items()
            if k in {"input_ids", "attention_mask", "mm_token_type_ids", "pixel_values", "image_grid_thw"}
        }
        outputs = self.model(
            **model_inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

        # --- VLM CE loss on command tokens only (memory efficient) ---
        hidden = outputs.hidden_states[-1]  # (B, seq_len, H)
        labels = batch["labels"]  # -100 everywhere except command word tokens
        # Find supervised positions: where labels != -100
        # Compute logits only at those positions to save memory
        supervised_mask = labels != -100  # (B, seq_len)
        if supervised_mask.any():
            # Shift: predict next token, so use hidden at position t to predict label at t
            # In causal LM: logit[t] predicts token[t+1], but labels are already aligned
            # by HF convention (labels[t] is the target for logits[t])
            lm_head = self.model.get_output_embeddings()
            # Gather hidden states at supervised positions
            sup_hidden = hidden[supervised_mask]  # (N, H)
            sup_logits = lm_head(sup_hidden)  # (N, vocab_size)
            # Shift labels: for causal LM, logits at pos t predict token at pos t+1
            shift_labels = labels[:, 1:]  # (B, seq_len-1)
            shift_mask = shift_labels != -100
            shift_hidden = hidden[:, :-1][shift_mask]  # (N, H)
            shift_logits = lm_head(shift_hidden)  # (N, vocab)
            shift_targets = shift_labels[shift_mask]  # (N,)
            vlm_loss = nn.functional.cross_entropy(shift_logits, shift_targets)
        else:
            vlm_loss = torch.tensor(0.0, device=hidden.device, requires_grad=True)

        # --- Extract last-token hidden state for diffusion conditioning ---
        last_indices = batch["attention_mask"].sum(dim=-1) - 1
        features = hidden[torch.arange(hidden.shape[0], device=hidden.device), last_indices]
        global_cond = self.vlm_projector(features.float())

        # --- Diffusion head ---
        diff_out = self.diffusion_head.forward_train(
            gt_traj=batch["target_traj"].to(global_cond.device),
            command=batch["command"].to(global_cond.device),
            global_cond=global_cond,
        )

        return {
            "loss": vlm_loss + diff_out["loss"],
            "vlm_loss": vlm_loss.detach(),
            "diff_loss": diff_out["loss"].detach(),
            "pred_traj": diff_out["pred_traj"],
        }

    def get_collator(self) -> CoTDiffusionTrainCollator:
        return CoTDiffusionTrainCollator(self.processor)

    def get_model_config(self) -> Dict[str, Any]:
        return {
            "future_steps": self.future_steps,
            "vlm_proj_dim": self.vlm_proj_dim,
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

    # -------------------------------------------------------------------
    # Inference helpers
    # -------------------------------------------------------------------

    def _prepare_messages(self, image_path: str, prompt: str):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info([messages])
        inputs = self.processor(text=[text], images=images, videos=videos, padding=True, return_tensors="pt")
        device = next(self.parameters()).device
        return {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    def generate_perception_answer(self, image_path: str, perception_question: str, max_new_tokens: int = 64) -> str:
        prompt = f"Perception question: {perception_question}"
        inputs = self._prepare_messages(image_path, prompt)
        with torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        prompt_len = inputs["input_ids"].shape[1]
        answer_ids = generated[0, prompt_len:]
        return self.processor.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

    def generate_command_and_features(
        self, image_path: str, perception_question: str, perception_answer: str
    ) -> Dict[str, Any]:
        """Generate command word and extract VLM hidden state for diffusion head.

        Uses the prompt hidden state (before generation) as scene context.
        The prompt already contains image + perception Q/A + motion prompt,
        which encodes the scene understanding needed by the diffusion head.
        """
        context = DEFAULT_PERCEPTION_TEMPLATE.format(
            question=perception_question, answer=perception_answer
        )
        prompt = f"{context}\n\n{DEFAULT_MOTION_PROMPT}"
        inputs = self._prepare_messages(image_path, prompt)

        # Extract hidden state from prompt (scene context for diffusion)
        model_inputs = {
            k: v for k, v in inputs.items()
            if k in {"input_ids", "attention_mask", "mm_token_type_ids", "pixel_values", "image_grid_thw"}
        }
        with torch.inference_mode():
            outputs = self.model(**model_inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        hidden = outputs.hidden_states[-1]
        last_idx = inputs["attention_mask"].sum(dim=-1) - 1
        features = hidden[0, last_idx[0]]
        global_cond = self.vlm_projector(features.float().unsqueeze(0))

        # Generate command text
        with torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=5)
        prompt_len = inputs["input_ids"].shape[1]
        command_text = self.processor.tokenizer.decode(
            generated[0, prompt_len:], skip_special_tokens=True
        ).strip().lower()

        # Parse command
        command_idx = 2  # default straight
        for word, idx in COMMAND_TO_INDEX.items():
            if word in command_text:
                command_idx = idx
                break

        return {
            "command_text": command_text,
            "command_idx": command_idx,
            "global_cond": global_cond,
        }

    def generate_trajectory(
        self,
        image_path: str,
        perception_question: str,
        perception_answer: str,
    ) -> Dict[str, Any]:
        """Full inference pipeline: command generation + diffusion trajectory."""
        cmd_result = self.generate_command_and_features(
            image_path, perception_question, perception_answer
        )
        device = next(self.parameters()).device
        command = torch.tensor([cmd_result["command_idx"]], dtype=torch.long, device=device)
        pred_traj = self.diffusion_head.forward_inference(command, cmd_result["global_cond"])
        return {
            "trajectory": pred_traj[0].cpu().numpy(),
            "command_text": cmd_result["command_text"],
            "command_idx": cmd_result["command_idx"],
        }


# ---------------------------------------------------------------------------
# Eval helper
# ---------------------------------------------------------------------------

def choose_eval_perception_qa(sample: Dict[str, Any]) -> Dict[str, Any]:
    candidates = sample["qa"]["perception"]
    idx = deterministic_perception_index(sample["frame_token"], len(candidates))
    return candidates[idx]
