#!/usr/bin/env python3
"""Generate K-means clustered trajectory anchors per driving command.

Output: anchors.npy of shape [3, num_modes, future_steps, 2]
  - 3 command classes: right=0, left=1, straight=2
  - num_modes trajectory modes per command (default 6)
  - future_steps waypoints (default 12)
  - 2D coordinates in ego frame (x=forward, y=left)

Also saves normalization statistics (traj_norm.npz) for the diffusion head.

Usage:
    python tools/kmeans_plan_anchors.py --output-dir data/anchors
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dataset.drivelm_uniad_dataset import DriveLMUniADDataset


COMMAND_NAMES = {0: "right", 1: "left", 2: "straight"}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate K-means trajectory anchors per command.")
    parser.add_argument("--train-json", default=str(REPO_ROOT / "challenge/data/v1_1_train_nus.json"))
    parser.add_argument("--train-pkl", default=str(REPO_ROOT / "UniAD/data/infos/nuscenes_infos_temporal_train.pkl"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "data/anchors"))
    parser.add_argument("--num-modes", type=int, default=6)
    parser.add_argument("--future-steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    dataset = DriveLMUniADDataset(args.train_json, args.train_pkl, camera_names=("CAM_FRONT",))
    print(f"Dataset size: {len(dataset)}")

    # Collect trajectories per command
    trajs_by_cmd = {0: [], 1: [], 2: []}
    all_trajs = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        traj = sample["ego_future_traj"]
        mask = sample["ego_future_mask"]
        cmd = sample["command"]
        if mask.sum() < args.future_steps:
            continue
        trajs_by_cmd[cmd].append(traj)
        all_trajs.append(traj)

    print(f"Samples per command: " + ", ".join(
        f"{COMMAND_NAMES[k]}={len(v)}" for k, v in trajs_by_cmd.items()
    ))

    # K-means clustering per command
    anchors = np.zeros((3, args.num_modes, args.future_steps, 2), dtype=np.float32)
    for cmd in range(3):
        cmd_trajs = np.array(trajs_by_cmd[cmd], dtype=np.float32)
        if len(cmd_trajs) == 0:
            print(f"WARNING: no trajectories for command {COMMAND_NAMES[cmd]}, using zeros")
            continue
        flat = cmd_trajs.reshape(len(cmd_trajs), -1)
        n_clusters = min(args.num_modes, len(cmd_trajs))
        kmeans = KMeans(n_clusters=n_clusters, random_state=args.seed, n_init=10)
        kmeans.fit(flat)
        centers = kmeans.cluster_centers_.reshape(n_clusters, args.future_steps, 2)
        anchors[cmd, :n_clusters] = centers
        print(f"  {COMMAND_NAMES[cmd]}: {len(cmd_trajs)} trajs → {n_clusters} clusters")

    # Save anchors
    anchor_path = output_dir / "plan_anchors.npy"
    np.save(anchor_path, anchors)
    print(f"Saved anchors to {anchor_path}, shape {anchors.shape}")

    # Compute and save normalization statistics
    all_trajs = np.array(all_trajs, dtype=np.float32)
    traj_mean = all_trajs.mean(axis=0)  # (12, 2)
    traj_std = all_trajs.std(axis=0).clip(min=1e-6)  # (12, 2)
    # Also compute global per-axis scale for simple normalization
    x_scale = float(np.percentile(np.abs(all_trajs[:, :, 0]), 99))
    y_scale = float(np.percentile(np.abs(all_trajs[:, :, 1]), 99))

    norm_path = output_dir / "traj_norm.npz"
    np.savez(
        norm_path,
        mean=traj_mean,
        std=traj_std,
        x_scale=x_scale,
        y_scale=y_scale,
    )
    print(f"Saved normalization stats to {norm_path}")
    print(f"  x_scale (forward, 99th pct): {x_scale:.2f} m")
    print(f"  y_scale (lateral, 99th pct): {y_scale:.2f} m")


if __name__ == "__main__":
    main()
