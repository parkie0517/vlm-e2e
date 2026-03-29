import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

from utils.trajectory_tokenizer import TrajectoryTokenCodec, deterministic_perception_index


DEFAULT_MODEL_PATH = str(
    Path(__file__).resolve().parents[1] / "checkpoints" / "Qwen3-VL-2B-Instruct"
)
DEFAULT_PERCEPTION_TEMPLATE = "Perception question: {question}\nPerception answer: {answer}"
DEFAULT_MOTION_PROMPT = (
    "You are an autonomous driving expert. Predict the future ego trajectory tokens "
    "from the front camera image and the perception context."
)


class MinimumCoTTrainCollator:
    def __init__(
        self,
        processor,
        codec: TrajectoryTokenCodec,
        perception_template: str = DEFAULT_PERCEPTION_TEMPLATE,
        motion_prompt: str = DEFAULT_MOTION_PROMPT,
    ):
        self.processor = processor
        self.codec = codec
        self.perception_template = perception_template
        self.motion_prompt = motion_prompt

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompt_messages = []
        full_texts = []
        prompt_texts = []
        selected_questions = []
        selected_answers = []

        for sample in samples:
            qa = random.choice(sample["qa"]["perception"])
            selected_questions.append(qa["Q"])
            selected_answers.append(qa["A"])
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
        trajectory_texts = [
            self.codec.tokens_to_text(
                self.codec.encode_trajectory(sample["ego_future_traj"], sample["ego_future_mask"])
            )
            for sample in samples
        ]
        full_texts = [f"{prompt_text}{traj_text}" for prompt_text, traj_text in zip(prompt_texts, trajectory_texts)]

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

        labels = full_inputs["input_ids"].clone()
        labels[full_inputs["attention_mask"] == 0] = -100
        prompt_lengths = prompt_inputs["attention_mask"].sum(dim=-1)
        for batch_idx, prompt_len in enumerate(prompt_lengths.tolist()):
            labels[batch_idx, :prompt_len] = -100

        batch = dict(full_inputs)
        batch["labels"] = labels
        batch["target_traj"] = torch.from_numpy(
            np.stack([sample["ego_future_traj"] for sample in samples], axis=0)
        ).float()
        batch["target_mask"] = torch.from_numpy(
            np.stack([sample["ego_future_mask"] for sample in samples], axis=0)
        ).float()
        batch["future_gt_boxes"] = [sample["future_gt_boxes"] for sample in samples]
        batch["frame_token"] = [sample["frame_token"] for sample in samples]
        batch["frame_idx"] = [sample["frame_idx"] for sample in samples]
        batch["selected_perception_q"] = selected_questions
        batch["selected_perception_a"] = selected_answers
        return batch


class Qwen3MinimumCoTTokenModel(torch.nn.Module):
    def __init__(
        self,
        token_codec: TrajectoryTokenCodec,
        model_name: str = DEFAULT_MODEL_PATH,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.codec = token_codec
        self.processor = AutoProcessor.from_pretrained(model_name)
        self._add_trajectory_tokens()

        preferred_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        base_model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            dtype=preferred_dtype if torch.cuda.is_available() else torch.float32,
        )
        self._resize_and_initialize_embeddings(base_model)
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
            trainable_token_indices=self.token_ids_to_train,
            ensure_weight_tying=True,
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(base_model, lora_config)
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            self.model.enable_input_require_grads()

        self._unfreeze_visual_adaptors()

    def _add_trajectory_tokens(self):
        added = self.processor.tokenizer.add_tokens(self.codec.added_tokens, special_tokens=False)
        self.codec_token_ids = {
            token: self.processor.tokenizer.convert_tokens_to_ids(token) for token in self.codec.added_tokens
        }
        self.dx_token_ids = [self.codec_token_ids[token] for token in self.codec.dx_tokens]
        self.dy_token_ids = [self.codec_token_ids[token] for token in self.codec.dy_tokens]
        self.token_ids_to_train = [self.codec_token_ids[token] for token in self.codec.added_tokens]
        self.sot_token_id = self.codec_token_ids[self.codec.sot_token]
        self.eot_token_id = self.codec_token_ids[self.codec.eot_token]

    def _resize_and_initialize_embeddings(self, base_model):
        tokenizer_len = len(self.processor.tokenizer)
        old_emb = base_model.get_input_embeddings().weight.data.clone()
        base_model.resize_token_embeddings(tokenizer_len)
        new_emb = base_model.get_input_embeddings().weight.data
        if tokenizer_len > old_emb.shape[0]:
            mean_embedding = old_emb.mean(dim=0, keepdim=True)
            new_emb[old_emb.shape[0] :] = mean_embedding
            output_emb = base_model.get_output_embeddings().weight.data
            output_emb[old_emb.shape[0] :] = mean_embedding

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
            if key
            in {
                "input_ids",
                "attention_mask",
                "mm_token_type_ids",
                "pixel_values",
                "image_grid_thw",
                "labels",
            }
        }
        outputs = self.model(**model_inputs, return_dict=True, use_cache=False)
        return {"loss": outputs.loss, "logits": outputs.logits}

    def get_model_config(self) -> Dict[str, Any]:
        return {
            "token_codec": self.codec.to_dict(),
        }

    def get_collator(self) -> MinimumCoTTrainCollator:
        return MinimumCoTTrainCollator(self.processor, self.codec)

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

    def _prepare_messages(self, image_path: str, prompt: str):
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        ]
        text = self.processor.apply_chat_template(messages[0], tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        inputs = self.processor(text=[text], images=images, videos=videos, padding=True, return_tensors="pt")
        device = next(self.parameters()).device
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        return inputs

    def _topk_for_last_position(self, inputs: Dict[str, Any], k: int = 10) -> List[Dict[str, Any]]:
        model_inputs = {k_: v for k_, v in inputs.items() if k_ in {"input_ids", "attention_mask", "mm_token_type_ids", "pixel_values", "image_grid_thw"}}
        with torch.inference_mode():
            outputs = self.model(**model_inputs, return_dict=True, use_cache=False)
        logits = outputs.logits[0, -1]
        top_vals, top_ids = torch.topk(logits, k=min(k, logits.shape[0]))
        result = []
        for score, token_id in zip(top_vals.tolist(), top_ids.tolist()):
            result.append(
                {
                    "token_id": int(token_id),
                    "token": self.processor.tokenizer.convert_ids_to_tokens(int(token_id)),
                    "score": float(score),
                }
            )
        return result

    def generate_perception_answer(self, image_path: str, perception_question: str, max_new_tokens: int = 64) -> str:
        prompt = f"Perception question: {perception_question}"
        inputs = self._prepare_messages(image_path, prompt)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                bad_words_ids=[[token_id] for token_id in self.token_ids_to_train],
            )
        prompt_len = inputs["input_ids"].shape[1]
        answer_ids = generated[0, prompt_len:]
        return self.processor.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

    def generate_trajectory(
        self,
        image_path: str,
        perception_question: str,
        perception_answer: str,
        max_new_tokens: int = 26,
        return_debug: bool = False,
    ) -> np.ndarray | Dict[str, Any]:
        context = DEFAULT_PERCEPTION_TEMPLATE.format(
            question=perception_question,
            answer=perception_answer,
        )
        prompt = f"{context}\n\n{DEFAULT_MOTION_PROMPT}"
        # Place SOT as the first token of the assistant response (matching training),
        # not inside the user message.
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
        text += self.codec.sot_token
        images, videos = process_vision_info([messages])
        inputs = self.processor(text=[text], images=images, videos=videos, padding=True, return_tensors="pt")
        device = next(self.parameters()).device
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        topk_before_generation = self._topk_for_last_position(inputs, k=10)
        prompt_len = inputs["input_ids"].shape[1]

        def prefix_allowed_tokens_fn(batch_id: int, input_ids: torch.Tensor) -> List[int]:
            generated_so_far = input_ids.shape[0] - prompt_len
            if generated_so_far >= self.codec.future_steps * 2:
                return [self.eot_token_id]
            if generated_so_far % 2 == 0:
                return self.dx_token_ids + [self.eot_token_id]
            return self.dy_token_ids

        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max(1, max_new_tokens - 1),
                eos_token_id=self.eot_token_id,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            )
        generated_ids = [self.sot_token_id] + generated[0, prompt_len:].tolist()
        decoded = self.codec.decode_token_ids(generated_ids, self.processor.tokenizer)
        if not return_debug:
            return decoded

        analysis = self.codec.analyze_token_ids(generated_ids, self.processor.tokenizer)
        return {
            "trajectory": decoded,
            "generated_ids": generated_ids,
            "generated_tokens": analysis["tokens"],
            "saw_sot": analysis["saw_sot"],
            "saw_eot": analysis["saw_eot"],
            "num_pairs": analysis["num_pairs"],
            "topk_before_generation": topk_before_generation,
        }


def choose_training_perception_qa(sample: Dict[str, Any]) -> Dict[str, Any]:
    return random.choice(sample["qa"]["perception"])


def choose_eval_perception_qa(sample: Dict[str, Any]) -> Dict[str, Any]:
    candidates = sample["qa"]["perception"]
    idx = deterministic_perception_index(sample["frame_token"], len(candidates))
    return candidates[idx]
