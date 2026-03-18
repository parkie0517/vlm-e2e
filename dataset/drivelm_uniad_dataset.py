import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from nuscenes.prediction import convert_global_coords_to_local
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

        return {
            "scene_token": sample["scene_token"],
            "frame_token": sample["frame_token"],
            "frame_idx": info["frame_idx"],
            "timestamp": info["timestamp"],
            "scene_description": sample["scene_description"],
            "image_paths": self._select_image_paths(info),
            "qa": sample["qa"],
            "ego_past_traj": ego_past_traj,
            "ego_past_mask": ego_past_mask,
            "ego_future_traj": ego_future_traj,
            "ego_future_mask": ego_future_mask,
            "can_bus": np.asarray(info["can_bus"], dtype=np.float32),
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
            steps = min(len(local), self.future_steps)
            traj[:steps] = local[:steps]
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
            steps = min(len(local), self.history_steps)
            traj[-steps:] = local[-steps:]
            mask[-steps:] = 1.0

        return traj, mask

    def __repr__(self):
        return (
            "DriveLMUniADDataset(num_samples={}, cameras={}, history_steps={}, future_steps={})".format(
                len(self),
                list(self.camera_names),
                self.history_steps,
                self.future_steps,
            )
        )
