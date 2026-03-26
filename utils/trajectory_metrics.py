from typing import Dict, List

import numpy as np
import torch
from shapely.geometry import Point, Polygon


EGO_LENGTH = 4.084
EGO_WIDTH = 1.85


def compute_regression_metrics(
    pred_traj: torch.Tensor, target_traj: torch.Tensor, target_mask: torch.Tensor
) -> Dict[str, float]:
    pred = pred_traj.detach().cpu()
    target = target_traj.detach().cpu()
    mask = target_mask.detach().cpu().bool()

    diff = pred - target
    per_step_dist = torch.sqrt(torch.sum(diff ** 2, dim=-1))

    valid_dist = per_step_dist[mask]
    l2 = float(torch.sqrt(torch.mean(valid_dist ** 2)).item()) if valid_dist.numel() else float("nan")
    ade = float(valid_dist.mean().item()) if valid_dist.numel() else float("nan")

    fdes = []
    for i in range(mask.shape[0]):
        valid_indices = torch.nonzero(mask[i], as_tuple=False).flatten()
        if len(valid_indices) == 0:
            continue
        last_idx = int(valid_indices[-1].item())
        fdes.append(float(per_step_dist[i, last_idx].item()))
    fde = float(np.mean(fdes)) if fdes else float("nan")

    return {"l2": l2, "ade": ade, "fde": fde}


def compute_collision_metrics(
    pred_traj: torch.Tensor,
    target_traj: torch.Tensor,
    target_mask: torch.Tensor,
    future_gt_boxes: List[List[np.ndarray]],
) -> Dict[str, float]:
    pred = pred_traj.detach().cpu().numpy()
    target = target_traj.detach().cpu().numpy()
    mask = target_mask.detach().cpu().numpy().astype(bool)

    obj_hits = 0
    obj_box_hits = 0
    valid_steps = 0

    for batch_idx in range(pred.shape[0]):
        sample_boxes = future_gt_boxes[batch_idx]
        for step in range(pred.shape[1]):
            if not mask[batch_idx, step]:
                continue

            boxes = sample_boxes[step] if step < len(sample_boxes) else np.zeros((0, 5), dtype=np.float32)
            gt_ego_poly = ego_polygon(target[batch_idx, step, 0], target[batch_idx, step, 1])
            if any(gt_ego_poly.intersects(box_polygon(box)) for box in boxes):
                continue

            valid_steps += 1
            pred_point = Point(float(pred[batch_idx, step, 0]), float(pred[batch_idx, step, 1]))
            pred_ego_poly = ego_polygon(pred[batch_idx, step, 0], pred[batch_idx, step, 1])

            if any(box_polygon(box).covers(pred_point) for box in boxes):
                obj_hits += 1
            if any(pred_ego_poly.intersects(box_polygon(box)) for box in boxes):
                obj_box_hits += 1

    if valid_steps == 0:
        return {"obj_col": float("nan"), "obj_box_col": float("nan")}
    return {
        "obj_col": obj_hits / valid_steps,
        "obj_box_col": obj_box_hits / valid_steps,
    }


def ego_polygon(x: float, y: float) -> Polygon:
    half_l = EGO_LENGTH / 2.0
    half_w = EGO_WIDTH / 2.0
    corners = np.array(
        [
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w],
        ],
        dtype=np.float32,
    )
    corners[:, 0] += float(x)
    corners[:, 1] += float(y)
    return Polygon(corners)


def box_polygon(box: np.ndarray) -> Polygon:
    x, y, length, width, yaw = [float(v) for v in box]
    half_l = length / 2.0
    half_w = width / 2.0
    corners = np.array(
        [
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w],
        ],
        dtype=np.float32,
    )
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    corners = corners @ rot.T
    corners[:, 0] += x
    corners[:, 1] += y
    return Polygon(corners)
