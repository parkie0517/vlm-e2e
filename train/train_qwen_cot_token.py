#!/usr/bin/env python3
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dataset.drivelm_uniad_dataset import DriveLMUniADDataset
from eval.eval_qwen_cot_token import evaluate_dataset
from models.qwen3_token_cot import Qwen3MinimumCoTTokenModel
from utils.trajectory_tokenizer import TrajectoryTokenCodec, save_codec


def parse_args():
    parser = argparse.ArgumentParser(description="Train the minimum CoT tokenized trajectory model.")
    parser.add_argument("--train-json", default=str(REPO_ROOT / "challenge/data/v1_1_train_nus.json"))
    parser.add_argument("--val-json", default=str(REPO_ROOT / "challenge/data/v1_1_val_nus_q_only.json"))
    parser.add_argument("--train-pkl", default=str(REPO_ROOT / "UniAD/data/infos/nuscenes_infos_temporal_train.pkl"))
    parser.add_argument("--val-pkl", default=str(REPO_ROOT / "UniAD/data/infos/nuscenes_infos_temporal_val.pkl"))
    parser.add_argument("--model-name", default=str(REPO_ROOT / "checkpoints/Qwen3-VL-2B-Instruct"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs/qwen_cot_token"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--subset-size", type=int, default=0)
    parser.add_argument("--val-subset-size", type=int, default=0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", default="", help="Path to a checkpoint to resume from.")
    parser.add_argument("--auto-resume", action="store_true")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def maybe_subset(dataset, subset_size: int, seed: int):
    if subset_size <= 0 or subset_size >= len(dataset):
        return dataset
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    return Subset(dataset, indices[:subset_size])


def move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def save_checkpoint(path: Path, model, optimizer, epoch: int, metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_config": model.get_model_config(),
        "trainable_state_dict": model.get_trainable_state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics": metrics,
    }
    torch.save(payload, path)


def resolve_resume_path(args, output_dir: Path):
    if args.resume:
        return Path(args.resume)
    if args.auto_resume:
        candidate = output_dir / "last.pt"
        if candidate.exists():
            return candidate
    return None


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = DriveLMUniADDataset(args.train_json, args.train_pkl, camera_names=("CAM_FRONT",))
    val_dataset = DriveLMUniADDataset(args.val_json, args.val_pkl, camera_names=("CAM_FRONT",))
    train_dataset = maybe_subset(train_dataset, args.subset_size, args.seed)
    val_dataset = maybe_subset(val_dataset, args.val_subset_size, args.seed + 1)

    resume_path = resolve_resume_path(args, output_dir)
    checkpoint = None
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        codec = TrajectoryTokenCodec.from_dict(checkpoint["model_config"]["token_codec"])
    else:
        codec = TrajectoryTokenCodec.build_from_dataset(train_dataset)
        save_codec(output_dir / "token_codec.json", codec)

    model = Qwen3MinimumCoTTokenModel(token_codec=codec, model_name=args.model_name)
    collator = model.get_collator()
    device = torch.device(args.device)
    model = model.to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
    )
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and amp_dtype == torch.float16)

    best_ade = float("inf")
    start_epoch = 1
    if checkpoint is not None:
        model.load_trainable_state_dict(checkpoint["trainable_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_ade = float(checkpoint.get("metrics", {}).get("ade", float("inf")))
        print(json.dumps({"resume_from": str(resume_path), "start_epoch": start_epoch, "best_ade": best_ade}, indent=2))

    if start_epoch > args.epochs:
        print(json.dumps({"message": "Training already complete for requested epoch range.", "start_epoch": start_epoch, "requested_epochs": args.epochs}, indent=2))
        return

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = []
        train_pbar = tqdm(train_loader, desc=f"train {epoch}/{args.epochs}", ncols=80, leave=True)
        for step, batch in enumerate(train_pbar, start=1):
            batch = move_batch_to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                outputs = model(batch)
                loss = outputs["loss"] / args.grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % args.grad_accum_steps == 0:
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss.append(float(outputs["loss"].item()))
            train_pbar.set_postfix(loss=f"{outputs['loss'].item():.4f}")

        metrics = evaluate_dataset(model, val_dataset, device, limit=None)
        metrics["train_loss"] = float(np.mean(running_loss)) if running_loss else float("nan")
        print(json.dumps({"epoch": epoch, **metrics}, indent=2))
        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, metrics)
        if not np.isnan(metrics["ade"]) and metrics["ade"] < best_ade:
            best_ade = metrics["ade"]
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, metrics)


if __name__ == "__main__":
    main()
