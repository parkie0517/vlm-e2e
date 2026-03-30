import json
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
from nuscenes.prediction import convert_global_coords_to_local # 이 쓰레기 같은 helper
from pyquaternion import Quaternion
from torch.utils.data import Dataset


DEFAULT_CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)


class DriveLMUniADDataset(Dataset):
    def __init__(
        self,
        drivelm_json_path,
        uniad_pkl_path,
        camera_names=("CAM_FRONT",),
        history_steps=4,
        future_steps=12,
        require_match=True,
    ):
        self.repo_root = Path(__file__).resolve().parents[1]
        self.uniad_root = self.repo_root / "UniAD"
        self.drivelm_json_path = Path(drivelm_json_path)
        self.uniad_pkl_path = Path(uniad_pkl_path)
        self.camera_names = tuple(camera_names)
        self.history_steps = int(history_steps)
        self.future_steps = int(future_steps)
        self.require_match = require_match

        self._validate_camera_names(self.camera_names)
        self.drivelm_data = self._load_json(self.drivelm_json_path)
        self.uniad_infos = self._load_pkl(self.uniad_pkl_path)["infos"]
        self.uniad_index = {info["token"]: info for info in self.uniad_infos}

        self.samples = self._build_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        info = sample["uniad_info"]

        ego_past_traj, ego_past_mask = self._build_ego_history(info)
        ego_future_traj, ego_future_mask = self._build_ego_future(info)
        future_gt_boxes = self._build_future_gt_boxes(info)

        command = self._derive_command(ego_future_traj, ego_future_mask)

        return {
            "scene_token": sample["scene_token"],
            "frame_token": sample["frame_token"],
            "frame_idx": info["frame_idx"],
            "timestamp": info["timestamp"],
            "scene_description": sample["scene_description"],
            "image_paths": self._select_image_paths(info),
            "camera_meta": self._select_camera_meta(info),
            "qa": sample["qa"],
            "ego_past_traj": ego_past_traj,
            "ego_past_mask": ego_past_mask,
            "ego_future_traj": ego_future_traj,
            "ego_future_mask": ego_future_mask,
            "future_gt_boxes": future_gt_boxes,
            "can_bus": np.asarray(info["can_bus"], dtype=np.float32),
            "command": command,
        }

    def _validate_camera_names(self, camera_names):
        invalid = [cam for cam in camera_names if cam not in DEFAULT_CAMERAS]
        if invalid:
            raise ValueError("Invalid camera names: {}".format(", ".join(invalid)))

    def _load_json(self, path: Path):
        with path.open("r") as f:
            return json.load(f)

    def _load_pkl(self, path: Path):
        with path.open("rb") as f:
            return pickle.load(f)

    def _build_samples(self):
        samples = []
        for scene_token, scene in self.drivelm_data.items():
            scene_description = scene.get("scene_description", "")
            for frame_token, frame in scene["key_frames"].items():
                info = self.uniad_index.get(frame_token)
                if info is None:
                    if self.require_match:
                        continue
                    info = None

                samples.append(
                    {
                        "scene_token": scene_token,
                        "frame_token": frame_token,
                        "scene_description": scene_description,
                        "qa": frame["QA"],
                        "uniad_info": info,
                    }
                )
        return samples

    def _select_image_paths(self, info):
        image_paths = {}
        for cam_name in self.camera_names:
            cam_info = info["cams"].get(cam_name)
            if cam_info is None:
                raise KeyError("Camera {} not found for token {}".format(cam_name, info["token"]))
            image_paths[cam_name] = self._normalize_uniad_path(cam_info["data_path"])
        return image_paths

    def _select_camera_meta(self, info):
        camera_meta = {}
        for cam_name in self.camera_names:
            cam_info = info["cams"].get(cam_name)
            if cam_info is None:
                raise KeyError("Camera {} not found for token {}".format(cam_name, info["token"]))
            camera_meta[cam_name] = {
                "sensor2ego_translation": np.asarray(cam_info["sensor2ego_translation"], dtype=np.float32),
                "sensor2ego_rotation": np.asarray(cam_info["sensor2ego_rotation"], dtype=np.float32),
                "cam_intrinsic": np.asarray(cam_info["cam_intrinsic"], dtype=np.float32),
            }
        return camera_meta

    def _normalize_uniad_path(self, path_str):
        path_str = str(path_str)
        if path_str.startswith("./"):
            return str((self.uniad_root / path_str[2:]).resolve())
        if path_str.startswith("data/"):
            return str((self.uniad_root / path_str).resolve())
        path = Path(path_str)
        if path.is_absolute():
            return str(path)
        return str((self.uniad_root / path).resolve())

    def _build_ego_future(self, info):
        traj = np.zeros((self.future_steps, 2), dtype=np.float32)
        mask = np.zeros((self.future_steps,), dtype=np.float32)

        points_global = []
        cursor = info
        for _ in range(self.future_steps):
            next_token = cursor.get("next", "")
            if not next_token:
                break
            next_info = self.uniad_index.get(next_token)
            if next_info is None or next_info["scene_token"] != info["scene_token"]:
                break
            points_global.append(next_info["ego2global_translation"][:2])
            cursor = next_info

        if points_global:
            local = convert_global_coords_to_local(
                coordinates=np.asarray(points_global, dtype=np.float32),
                translation=info["ego2global_translation"],
                rotation=info["ego2global_rotation"],
            ).astype(np.float32)
            # convert_global_coords_to_local returns [lateral(right+), longitudinal(fwd+)].
            # Remap to nuscenes ego frame: [x=forward, y=left].
            ego = self._remap_local_to_ego(local)
            steps = min(len(ego), self.future_steps)
            traj[:steps] = ego[:steps]
            mask[:steps] = 1.0

        return traj, mask

    def _build_ego_history(self, info):
        traj = np.zeros((self.history_steps, 2), dtype=np.float32)
        mask = np.zeros((self.history_steps,), dtype=np.float32)

        points_global = []
        cursor = info
        for _ in range(self.history_steps):
            prev_token = cursor.get("prev", "")
            if not prev_token:
                break
            prev_info = self.uniad_index.get(prev_token)
            if prev_info is None or prev_info["scene_token"] != info["scene_token"]:
                break
            points_global.append(prev_info["ego2global_translation"][:2])
            cursor = prev_info

        if points_global:
            points_global.reverse()
            local = convert_global_coords_to_local(
                coordinates=np.asarray(points_global, dtype=np.float32),
                translation=info["ego2global_translation"],
                rotation=info["ego2global_rotation"],
            ).astype(np.float32)
            # convert_global_coords_to_local returns [lateral(right+), longitudinal(fwd+)].
            # Remap to nuscenes ego frame: [x=forward, y=left].
            ego = self._remap_local_to_ego(local)
            steps = min(len(ego), self.history_steps)
            traj[-steps:] = ego[-steps:]
            mask[-steps:] = 1.0

        return traj, mask

    def _build_future_gt_boxes(self, info):
        future_boxes = []
        cursor = info
        for _ in range(self.future_steps):
            next_token = cursor.get("next", "")
            if not next_token:
                future_boxes.append(np.zeros((0, 5), dtype=np.float32))
                cursor = {"next": ""}
                continue

            next_info = self.uniad_index.get(next_token)
            if next_info is None or next_info["scene_token"] != info["scene_token"]:
                future_boxes.append(np.zeros((0, 5), dtype=np.float32))
                cursor = {"next": ""}
                continue

            future_boxes.append(self._transform_boxes_to_current_ego(info, next_info))
            cursor = next_info

        return future_boxes

    def _transform_boxes_to_current_ego(self, current_info, future_info):
        gt_boxes = np.asarray(future_info["gt_boxes"], dtype=np.float32)
        if gt_boxes.size == 0:
            return np.zeros((0, 5), dtype=np.float32)

        centers_future = gt_boxes[:, :2]
        future_yaw = self._yaw_from_quaternion(future_info["ego2global_rotation"])
        current_yaw = self._yaw_from_quaternion(current_info["ego2global_rotation"])

        centers_global = self._rotate_points(centers_future, future_yaw)
        centers_global += np.asarray(future_info["ego2global_translation"][:2], dtype=np.float32)

        local = convert_global_coords_to_local(
            coordinates=centers_global,
            translation=current_info["ego2global_translation"],
            rotation=current_info["ego2global_rotation"],
        ).astype(np.float32)
        centers_current = self._remap_local_to_ego(local)

        box_yaw = gt_boxes[:, 6] + future_yaw - current_yaw
        box_yaw = (box_yaw + np.pi) % (2 * np.pi) - np.pi

        dims = gt_boxes[:, 3:5].astype(np.float32)
        return np.concatenate([centers_current, dims, box_yaw[:, None]], axis=1).astype(np.float32)

    def _derive_command(self, ego_future_traj, ego_future_mask):
        """Derive 3-class driving command from final valid waypoint lateral displacement.

        Ego frame: x=forward, y=left.
        Returns: 0=right, 1=left, 2=straight (matches DiffusionDrive convention).
        """
        valid = ego_future_mask.astype(bool)
        if not valid.any():
            return 2  # default straight
        last_valid_idx = int(np.where(valid)[0][-1])
        lateral_y = float(ego_future_traj[last_valid_idx, 1])
        if lateral_y >= 2.0:
            return 1  # left
        elif lateral_y <= -2.0:
            return 0  # right
        return 2  # straight

    def _remap_local_to_ego(self, local):
        return np.column_stack([local[:, 1], -local[:, 0]]).astype(np.float32)

    def _yaw_from_quaternion(self, quat):
        return float(Quaternion(quat).yaw_pitch_roll[0])

    def _rotate_points(self, points, yaw):
        c, s = np.cos(yaw), np.sin(yaw)
        rot = np.asarray([[c, -s], [s, c]], dtype=np.float32)
        return points @ rot.T

    def __repr__(self):
        return (
            "DriveLMUniADDataset(num_samples={}, cameras={}, history_steps={}, future_steps={})".format(
                len(self),
                list(self.camera_names),
                self.history_steps,
                self.future_steps,
            )
        )
