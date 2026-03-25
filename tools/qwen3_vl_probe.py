#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


DEFAULT_MODEL = "/mnt/hj_wd/vlm-e2e/checkpoints/Qwen3-VL-2B-Instruct"
DEFAULT_PROMPT = (
    "you are an autonomous driving expert, please tell me what you see in the driving scene."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a one-image Qwen3-VL inference probe.")
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL,
        help="Hugging Face model id or local path.",
    )
    parser.add_argument(
        "--image-path",
        default="random_scene.png",
        help="Path to the input image.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt text to send with the image.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image_path).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print(f"Model: {args.model_name}")
    print(f"Image: {image_path}")
    print(f"Prompt: {args.prompt}")
    print(f"Device: {device}")

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        device_map="auto" if device == "cuda" else None,
    )
    processor = AutoProcessor.from_pretrained(args.model_name)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    if device == "cuda":
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    print("\nResponse:")
    print(output_text[0].strip())

    if device == "cuda":
        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"\nPeak CUDA memory allocated: {peak_mem_gb:.2f} GB")


if __name__ == "__main__":
    main()
