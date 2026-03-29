#!/usr/bin/env python3
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
from models.qwen3_token_cot import Qwen3MinimumCoTTokenModel, choose_eval_perception_qa
from utils.trajectory_metrics import compute_collision_metrics, compute_regression_metrics
from utils.trajectory_tokenizer import TrajectoryTokenCodec


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the minimum CoT tokenized trajectory model.")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--json-path", default="")
    parser.add_argument("--pkl-path", default="")
    parser.add_argument("--model-name", default=str(REPO_ROOT / "checkpoints/Qwen3-VL-2B-Instruct"))
    parser.add_argument("--checkpoint", default=str(REPO_ROOT / "outputs/qwen_cot_token/best.pt"))
    parser.add_argument("--subset-size", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--debug-indices", default="", help="Comma-separated dataset indices to print full generation debug for.")
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
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def evaluate_dataset(model, dataset, device, limit=None, debug_indices=None):
    model.eval()
    reg_metrics = []
    col_metrics = []
    losses = []
    debug_indices = set(debug_indices or [])
    debug_samples = []
    decode_failures = 0
    saw_sot_count = 0
    saw_eot_count = 0
    num_items = len(dataset) if limit is None else min(len(dataset), limit)
    iterator = tqdm(range(num_items), desc="eval token cot", ncols=80, leave=True)
    for idx in iterator:
        sample = dataset[idx]
        perception_qa = choose_eval_perception_qa(sample)
        perception_answer = model.generate_perception_answer(
            image_path=sample["image_paths"]["CAM_FRONT"],
            perception_question=perception_qa["Q"],
        )
        generation = model.generate_trajectory(
            image_path=sample["image_paths"]["CAM_FRONT"],
            perception_question=perception_qa["Q"],
            perception_answer=perception_answer,
            return_debug=True,
        )
        pred_traj = generation["trajectory"]
        if generation["saw_sot"]:
            saw_sot_count += 1
        if generation["saw_eot"]:
            saw_eot_count += 1
        if generation["num_pairs"] == 0:
            decode_failures += 1
        if idx in debug_indices:
            debug_samples.append(
                {
                    "dataset_index": idx,
                    "frame_idx": sample["frame_idx"],
                    "frame_token": sample["frame_token"],
                    "perception_question": perception_qa["Q"],
                    "perception_answer": perception_answer,
                    "generated_ids": generation["generated_ids"],
                    "generated_tokens": generation["generated_tokens"],
                    "num_pairs": generation["num_pairs"],
                    "topk_before_generation": generation["topk_before_generation"],
                    "pred_traj": pred_traj.tolist(),
                    "target_traj": sample["ego_future_traj"].tolist(),
                    "target_mask": sample["ego_future_mask"].tolist(),
                }
            )
        pred_traj_t = torch.from_numpy(pred_traj).unsqueeze(0).to(device).float()
        target_traj = torch.from_numpy(sample["ego_future_traj"]).unsqueeze(0).to(device).float()
        target_mask = torch.from_numpy(sample["ego_future_mask"]).unsqueeze(0).to(device).float()
        reg_metrics.append(compute_regression_metrics(pred_traj_t, target_traj, target_mask))
        col_metrics.append(
            compute_collision_metrics(
                pred_traj_t,
                target_traj,
                target_mask,
                [sample["future_gt_boxes"]],
            )
        )
        valid = target_mask.sum().item()
        if valid > 0:
            diff = pred_traj_t - target_traj
            per_step = ((diff**2).sum(dim=-1) * target_mask).sum() / valid
            losses.append(float(per_step.item()))

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
        "decode_failures": int(decode_failures),
        "saw_sot_rate": float(saw_sot_count / num_items) if num_items > 0 else float("nan"),
        "saw_eot_rate": float(saw_eot_count / num_items) if num_items > 0 else float("nan"),
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
    codec = TrajectoryTokenCodec.from_dict(checkpoint["model_config"]["token_codec"])
    model = Qwen3MinimumCoTTokenModel(token_codec=codec, model_name=args.model_name)
    model.load_trainable_state_dict(checkpoint["trainable_state_dict"])
    device = torch.device(args.device)
    model = model.to(device)

    dataset = DriveLMUniADDataset(json_path, pkl_path, camera_names=("CAM_FRONT",))
    limit = args.subset_size if args.subset_size > 0 else None
    metrics = evaluate_dataset(model, dataset, device, limit=limit, debug_indices=parse_debug_indices(args.debug_indices))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
