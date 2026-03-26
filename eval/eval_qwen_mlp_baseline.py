#!/usr/bin/env python3
import argparse
import json
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
    parser = argparse.ArgumentParser(description="Evaluate the Qwen3-VL + MLP trajectory baseline.")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--json-path", default="")
    parser.add_argument("--pkl-path", default="")
    parser.add_argument("--model-name", default=str(REPO_ROOT / "checkpoints/Qwen3-VL-2B-Instruct"))
    parser.add_argument("--checkpoint", default=str(REPO_ROOT / "outputs/qwen_mlp_baseline/best.pt"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--subset-size", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def default_split_paths(split: str):
    if split == "train":
        return (
            REPO_ROOT / "challenge/data/v1_1_train_nus.json",
            REPO_ROOT / "UniAD/data/infos/nuscenes_infos_temporal_train.pkl",
        )
    return (
        REPO_ROOT / "challenge/data/v1_1_val_nus_q_only.json",
        REPO_ROOT / "UniAD/data/infos/nuscenes_infos_temporal_val.pkl",
    )


def move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def main():
    args = parse_args()
    default_json, default_pkl = default_split_paths(args.split)
    json_path = Path(args.json_path) if args.json_path else default_json
    pkl_path = Path(args.pkl_path) if args.pkl_path else default_pkl

    dataset = DriveLMUniADDataset(
        drivelm_json_path=json_path,
        uniad_pkl_path=pkl_path,
        camera_names=("CAM_FRONT",),
    )
    if args.subset_size > 0 and args.subset_size < len(dataset):
        dataset = Subset(dataset, list(range(args.subset_size)))

    model = Qwen3MLPTrajectoryModel(model_name=args.model_name)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_trainable_state_dict(checkpoint["trainable_state_dict"])
    collator = model.get_collator()
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
    )

    losses = []
    reg_metrics = []
    col_metrics = []

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

    metrics = {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "l2": mean_metric(reg_metrics, "l2"),
        "ade": mean_metric(reg_metrics, "ade"),
        "fde": mean_metric(reg_metrics, "fde"),
        "obj_col": mean_metric(col_metrics, "obj_col"),
        "obj_box_col": mean_metric(col_metrics, "obj_box_col"),
    }
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
