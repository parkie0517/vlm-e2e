#!/usr/bin/env python3
import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.drivelm_uniad_dataset import DriveLMUniADDataset


CAMERA_LAYOUT = (
    ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"),
    ("CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT"),
)

TRAIN_JSON = "challenge/data/v1_1_train_nus.json"
VAL_JSON = "challenge/data/v1_1_val_nus_q_only.json"
TRAIN_PKL = "UniAD/data/infos/nuscenes_infos_temporal_train.pkl"
VAL_PKL = "UniAD/data/infos/nuscenes_infos_temporal_val.pkl"


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize aligned DriveLM/UniAD samples.")
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/aligned_samples")
    return parser.parse_args()


def build_dataset(split):
    if split == "train":
        return DriveLMUniADDataset(
            drivelm_json_path=TRAIN_JSON,
            uniad_pkl_path=TRAIN_PKL,
            camera_names=tuple(cam for row in CAMERA_LAYOUT for cam in row),
        )
    return DriveLMUniADDataset(
        drivelm_json_path=VAL_JSON,
        uniad_pkl_path=VAL_PKL,
        camera_names=tuple(cam for row in CAMERA_LAYOUT for cam in row),
    )


def load_image(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError("Failed to read image: {}".format(path))
    return image


def project_ego_traj_to_front(front_image, front_meta, ego_future_traj, ego_future_mask):
    image = front_image.copy()
    valid_points = ego_future_traj[ego_future_mask > 0]
    if len(valid_points) == 0:
        return image

    pts_ego = np.concatenate(
        [
            valid_points[:, 0:1].astype(np.float32),    # ego_x (forward)
            valid_points[:, 1:2].astype(np.float32),    # ego_y (left)
            np.zeros((len(valid_points), 1), dtype=np.float32),
        ],
        axis=1,
    ).T

    rotation = Quaternion(front_meta["sensor2ego_rotation"]).rotation_matrix
    translation = front_meta["sensor2ego_translation"].reshape(3, 1)
    pts_cam = rotation.T.dot(pts_ego - translation)

    depth_mask = pts_cam[2, :] > 1e-3
    if not np.any(depth_mask):
        return image

    pts_cam = pts_cam[:, depth_mask]
    projected = view_points(pts_cam, front_meta["cam_intrinsic"], normalize=True)[:2, :].T

    h, w = image.shape[:2]
    points = []
    for x, y in projected:
        xi = int(round(x))
        yi = int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            points.append((xi, yi))

    for i, point in enumerate(points):
        cv2.circle(image, point, 6, (0, 255, 255), -1)
        if i > 0:
            cv2.line(image, points[i - 1], point, (0, 255, 0), 3)
    return image


def resize_keep_label(image, label, tile_size):
    resized = cv2.resize(image, tile_size, interpolation=cv2.INTER_LINEAR)
    cv2.putText(
        resized,
        label,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return resized


def compose_left_panel(sample, tile_size=(480, 270)):
    front_meta = sample["camera_meta"]["CAM_FRONT"]
    images = {}
    for cam_name, path in sample["image_paths"].items():
        image = load_image(path)
        if cam_name == "CAM_FRONT":
            image = project_ego_traj_to_front(
                image,
                front_meta,
                sample["ego_future_traj"],
                sample["ego_future_mask"],
            )
        images[cam_name] = resize_keep_label(image, cam_name, tile_size)

    row_images = []
    for row in CAMERA_LAYOUT:
        row_images.append(np.concatenate([images[cam] for cam in row], axis=1))
    panel_bgr = np.concatenate(row_images, axis=0)
    return Image.fromarray(cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2RGB))


def save_visualization(sample, sample_index, split, output_dir):
    left_panel = compose_left_panel(sample)
    output_path = output_dir / "{}_{}.png".format(split, sample_index)
    left_panel.save(output_path)
    return output_path


def print_qna(split, sample):
    print("split: {}".format(split))
    print("frame_idx: {}".format(sample["frame_idx"]))
    for group_name in ["perception", "prediction", "planning", "behavior"]:
        qa_list = sample["qa"].get(group_name, [])
        if not qa_list:
            continue
        print("[{}]".format(group_name))
        qa = qa_list[0]
        q_text = qa.get("Q", "")
        a_text = qa.get("A", "")
        print("Q1: {}".format(q_text))
        print("A1: {}".format(a_text if a_text else "<empty>"))
        print("")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(args.split)
    sample_count = min(args.num_samples, len(dataset))
    indices = random.sample(range(len(dataset)), sample_count)

    print("Split: {}".format(args.split))
    print("Dataset size: {}".format(len(dataset)))
    print("Selected indices: {}".format(indices))

    for idx in indices:
        sample = dataset[idx]
        print_qna(args.split, sample)
        output_path = save_visualization(sample, idx, args.split, output_dir)
        print("Saved {}".format(output_path))


if __name__ == "__main__":
    main()
