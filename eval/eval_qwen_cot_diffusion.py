#!/usr/bin/env python3
"""Evaluate the CoT + Diffusion trajectory model."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dataset.drivelm_uniad_dataset import DriveLMUniADDataset
from models.qwen3_cot_diffusion import Qwen3CoTDiffusionModel, choose_eval_perception_qa, COMMAND_WORDS
from utils.trajectory_metrics import compute_collision_metrics, compute_regression_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CoT + Diffusion trajectory model.")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--json-path", default="")
    parser.add_argument("--pkl-path", default="")
    parser.add_argument("--model-name", default=str(REPO_ROOT / "checkpoints/Qwen3-VL-2B-Instruct"))
    parser.add_argument("--anchor-path", default=str(REPO_ROOT / "data/anchors/plan_anchors.npy"))
    parser.add_argument("--checkpoint", default=str(REPO_ROOT / "outputs/cot_diffusion/best.pt"))
    parser.add_argument("--subset-size", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--debug-indices", default="")
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


def parse_debug_indices(raw: str):
    if not raw.strip():
        return []
    return [int(p.strip()) for p in raw.split(",") if p.strip()]


def evaluate_dataset(model, dataset, device, limit=None, debug_indices=None):
    model.eval()
    reg_metrics = []
    col_metrics = []
    debug_indices = set(debug_indices or [])
    debug_samples = []
    num_items = len(dataset) if limit is None else min(len(dataset), limit)
    iterator = tqdm(range(num_items), desc="eval cot+diff", ncols=80, leave=True)

    for idx in iterator:
        sample = dataset[idx]
        perception_qa = choose_eval_perception_qa(sample)

        # Pass 1: generate perception answer
        perception_answer = model.generate_perception_answer(
            image_path=sample["image_paths"]["CAM_FRONT"],
            perception_question=perception_qa["Q"],
        )

        # Pass 2: extract hidden state + diffusion trajectory (GT command)
        result = model.generate_trajectory(
            image_path=sample["image_paths"]["CAM_FRONT"],
            perception_question=perception_qa["Q"],
            perception_answer=perception_answer,
            command=sample["command"],
        )

        pred_traj = result["trajectory"]

        if idx in debug_indices:
            debug_samples.append({
                "dataset_index": idx,
                "frame_token": sample["frame_token"],
                "perception_question": perception_qa["Q"],
                "perception_answer": perception_answer,
                "gt_command": COMMAND_WORDS[sample["command"]],
                "pred_traj": pred_traj.tolist(),
                "target_traj": sample["ego_future_traj"].tolist(),
                "target_mask": sample["ego_future_mask"].tolist(),
            })

        pred_t = torch.from_numpy(pred_traj).unsqueeze(0).to(device).float()
        target_t = torch.from_numpy(sample["ego_future_traj"]).unsqueeze(0).to(device).float()
        mask_t = torch.from_numpy(sample["ego_future_mask"]).unsqueeze(0).to(device).float()
        reg_metrics.append(compute_regression_metrics(pred_t, target_t, mask_t))
        col_metrics.append(compute_collision_metrics(pred_t, target_t, mask_t, [sample["future_gt_boxes"]]))

    def mean_metric(items, key):
        values = [item[key] for item in items if not np.isnan(item[key])]
        return float(np.mean(values)) if values else float("nan")

    metrics = {
        "l2": mean_metric(reg_metrics, "l2"),
        "ade": mean_metric(reg_metrics, "ade"),
        "fde": mean_metric(reg_metrics, "fde"),
        "obj_col": mean_metric(col_metrics, "obj_col"),
        "obj_box_col": mean_metric(col_metrics, "obj_box_col"),
    }
    if debug_samples:
        metrics["debug_samples"] = debug_samples
    return metrics


def main():
    args = parse_args()
    default_json, default_pkl = default_split_paths(args.split)
    json_path = Path(args.json_path) if args.json_path else default_json
    pkl_path = Path(args.pkl_path) if args.pkl_path else default_pkl

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = Qwen3CoTDiffusionModel(
        anchor_path=args.anchor_path,
        model_name=args.model_name,
    )
    model.load_trainable_state_dict(checkpoint["trainable_state_dict"])
    device = torch.device(args.device)
    model = model.to(device)

    dataset = DriveLMUniADDataset(json_path, pkl_path, camera_names=("CAM_FRONT",))
    limit = args.subset_size if args.subset_size > 0 else None
    metrics = evaluate_dataset(model, dataset, device, limit=limit, debug_indices=parse_debug_indices(args.debug_indices))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
