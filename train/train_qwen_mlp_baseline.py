#!/usr/bin/env python3
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dataset.drivelm_uniad_dataset import DriveLMUniADDataset
from models.qwen3_mlp_baseline import Qwen3MLPTrajectoryModel
from utils.trajectory_metrics import compute_collision_metrics, compute_regression_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Qwen3-VL + MLP trajectory baseline.")
    parser.add_argument("--train-json", default=str(REPO_ROOT / "challenge/data/v1_1_train_nus.json"))
    parser.add_argument("--val-json", default=str(REPO_ROOT / "challenge/data/v1_1_val_nus_q_only.json"))
    parser.add_argument("--train-pkl", default=str(REPO_ROOT / "UniAD/data/infos/nuscenes_infos_temporal_train.pkl"))
    parser.add_argument("--val-pkl", default=str(REPO_ROOT / "UniAD/data/infos/nuscenes_infos_temporal_val.pkl"))
    parser.add_argument("--model-name", default=str(REPO_ROOT / "checkpoints/Qwen3-VL-2B-Instruct"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs/qwen_mlp_baseline"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--subset-size", type=int, default=0)
    parser.add_argument("--val-subset-size", type=int, default=0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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


def evaluate(model, dataloader, device):
    model.eval()
    reg_metrics = []
    col_metrics = []
    losses = []
    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            outputs = model(batch)
            losses.append(float(outputs["loss"].item()))
            reg_metrics.append(
                compute_regression_metrics(outputs["pred_traj"], batch["target_traj"], batch["target_mask"])
            )
            col_metrics.append(
                compute_collision_metrics(
                    outputs["pred_traj"],
                    batch["target_traj"],
                    batch["target_mask"],
                    batch["future_gt_boxes"],
                )
            )

    def mean_metric(items, key):
        values = [item[key] for item in items if not np.isnan(item[key])]
        return float(np.mean(values)) if values else float("nan")

    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "l2": mean_metric(reg_metrics, "l2"),
        "ade": mean_metric(reg_metrics, "ade"),
        "fde": mean_metric(reg_metrics, "fde"),
        "obj_col": mean_metric(col_metrics, "obj_col"),
        "obj_box_col": mean_metric(col_metrics, "obj_box_col"),
    }


def save_checkpoint(path: Path, model, optimizer, epoch: int, metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "trainable_state_dict": model.get_trainable_state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics": metrics,
    }
    torch.save(payload, path)


def main():
    args = parse_args()
    set_seed(args.seed)

    train_dataset = DriveLMUniADDataset(
        drivelm_json_path=args.train_json,
        uniad_pkl_path=args.train_pkl,
        camera_names=("CAM_FRONT",),
    )
    val_dataset = DriveLMUniADDataset(
        drivelm_json_path=args.val_json,
        uniad_pkl_path=args.val_pkl,
        camera_names=("CAM_FRONT",),
    )
    train_dataset = maybe_subset(train_dataset, args.subset_size, args.seed)
    val_dataset = maybe_subset(val_dataset, args.val_subset_size, args.seed + 1)

    model = Qwen3MLPTrajectoryModel(model_name=args.model_name)
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
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_ade = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = []

        for step, batch in enumerate(train_loader, start=1):
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

        metrics = evaluate(model, val_loader, device)
        metrics["train_loss"] = float(np.mean(running_loss)) if running_loss else float("nan")
        print(json.dumps({"epoch": epoch, **metrics}, indent=2))

        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, metrics)
        if not np.isnan(metrics["ade"]) and metrics["ade"] < best_ade:
            best_ade = metrics["ade"]
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, metrics)


if __name__ == "__main__":
    main()
