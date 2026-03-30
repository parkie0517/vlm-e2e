#!/usr/bin/env python3
"""Train the CoT + Diffusion trajectory model."""
import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dataset.drivelm_uniad_dataset import DriveLMUniADDataset
from eval.eval_qwen_cot_diffusion import evaluate_dataset
from models.qwen3_cot_diffusion import Qwen3CoTDiffusionModel


def parse_args():
    parser = argparse.ArgumentParser(description="Train CoT + Diffusion trajectory model.")
    parser.add_argument("--train-json", default=str(REPO_ROOT / "challenge/data/v1_1_train_nus.json"))
    parser.add_argument("--val-json", default=str(REPO_ROOT / "challenge/data/v1_1_val_nus_q_only.json"))
    parser.add_argument("--train-pkl", default=str(REPO_ROOT / "UniAD/data/infos/nuscenes_infos_temporal_train.pkl"))
    parser.add_argument("--val-pkl", default=str(REPO_ROOT / "UniAD/data/infos/nuscenes_infos_temporal_val.pkl"))
    parser.add_argument("--model-name", default=str(REPO_ROOT / "checkpoints/Qwen3-VL-2B-Instruct"))
    parser.add_argument("--anchor-path", default=str(REPO_ROOT / "data/anchors/plan_anchors.npy"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs/cot_diffusion"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--diff-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--subset-size", type=int, default=0)
    parser.add_argument("--val-subset-size", type=int, default=0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", default="")
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
    return {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }


def save_checkpoint(path: Path, model, optimizer, epoch: int, metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_config": model.get_model_config(),
        "trainable_state_dict": model.get_trainable_state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics": metrics,
    }, path)


def resolve_resume_path(args, output_dir: Path):
    if args.resume:
        return Path(args.resume)
    if args.auto_resume:
        candidate = output_dir / "last.pt"
        if candidate.exists():
            return candidate
    return None


def build_param_groups(model, vlm_lr: float, diff_lr: float, weight_decay: float):
    """Separate LR groups: VLM LoRA params vs diffusion head + projector."""
    vlm_params = []
    diff_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "diffusion_head" in name or "vlm_projector" in name:
            diff_params.append(param)
        else:
            vlm_params.append(param)
    return [
        {"params": vlm_params, "lr": vlm_lr, "weight_decay": weight_decay},
        {"params": diff_params, "lr": diff_lr, "weight_decay": weight_decay},
    ]


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = DriveLMUniADDataset(args.train_json, args.train_pkl, camera_names=("CAM_FRONT",))
    val_dataset = DriveLMUniADDataset(args.val_json, args.val_pkl, camera_names=("CAM_FRONT",))
    train_dataset = maybe_subset(train_dataset, args.subset_size, args.seed)
    val_dataset = maybe_subset(val_dataset, args.val_subset_size, args.seed + 1)

    model = Qwen3CoTDiffusionModel(
        anchor_path=args.anchor_path,
        model_name=args.model_name,
    )
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
    param_groups = build_param_groups(model, args.lr, args.diff_lr, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups)

    total_steps = len(train_loader) * args.epochs // args.grad_accum_steps
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - args.warmup_steps), eta_min=args.lr * 0.1)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[args.warmup_steps])

    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and amp_dtype == torch.float16)

    best_ade = float("inf")
    start_epoch = 1
    resume_path = resolve_resume_path(args, output_dir)
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_trainable_state_dict(checkpoint["trainable_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_ade = float(checkpoint.get("metrics", {}).get("ade", float("inf")))
        print(json.dumps({"resume_from": str(resume_path), "start_epoch": start_epoch, "best_ade": best_ade}, indent=2))

    if start_epoch > args.epochs:
        print(json.dumps({"message": "Training already complete.", "start_epoch": start_epoch}, indent=2))
        return

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss, running_vlm, running_diff = [], [], []
        train_pbar = tqdm(train_loader, desc=f"train {epoch}/{args.epochs}", ncols=100, leave=True)

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
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.max_grad_norm,
                )
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss.append(float(outputs["loss"].item()))
            running_vlm.append(float(outputs["vlm_loss"].item()))
            running_diff.append(float(outputs["diff_loss"].item()))
            train_pbar.set_postfix(
                loss=f"{outputs['loss'].item():.4f}",
                vlm=f"{outputs['vlm_loss'].item():.4f}",
                diff=f"{outputs['diff_loss'].item():.4f}",
            )

        metrics = evaluate_dataset(model, val_dataset, device, limit=None)
        metrics["train_loss"] = float(np.mean(running_loss)) if running_loss else float("nan")
        metrics["train_vlm_loss"] = float(np.mean(running_vlm)) if running_vlm else float("nan")
        metrics["train_diff_loss"] = float(np.mean(running_diff)) if running_diff else float("nan")
        print(json.dumps({"epoch": epoch, **metrics}, indent=2))

        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, metrics)
        if not np.isnan(metrics["ade"]) and metrics["ade"] < best_ade:
            best_ade = metrics["ade"]
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, metrics)


if __name__ == "__main__":
    main()
