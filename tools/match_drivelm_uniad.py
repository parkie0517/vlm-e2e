#!/usr/bin/env python3
import argparse
import json
import pickle
from pathlib import Path
from typing import Optional


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def load_pkl(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def normalize_path(path_str: str) -> str:
    parts = Path(path_str).parts
    if "samples" in parts:
        idx = parts.index("samples")
        return str(Path(*parts[idx:]))
    return path_str


def build_uniad_index(pkl_path: Path):
    data = load_pkl(pkl_path)
    infos = data["infos"]
    return {info["token"]: info for info in infos}


def iterate_drivelm_frames(data):
    for scene_token, scene in data.items():
        for frame_token, frame in scene["key_frames"].items():
            yield scene_token, frame_token, frame


def get_first_qa(frame):
    qa_groups = frame.get("QA", {})
    for group_name in ["perception", "prediction", "planning", "behavior"]:
        for qa in qa_groups.get(group_name, []):
            return group_name, qa.get("Q"), qa.get("A")
    return None, None, None


def summarize_split(split_name: str, json_path: Path, pkl_path: Optional[Path], num_examples: int):
    print(f"\n=== {split_name.upper()} ===")
    print(f"DriveLM JSON: {json_path}")
    if not json_path.exists():
        print("JSON file is missing.")
        return

    drivelm = load_json(json_path)
    total_scenes = len(drivelm)
    total_frames = sum(len(scene["key_frames"]) for scene in drivelm.values())
    print(f"DriveLM scenes: {total_scenes}")
    print(f"DriveLM key frames: {total_frames}")

    if pkl_path is None:
        print("UniAD PKL: not provided")
        return
    print(f"UniAD PKL: {pkl_path}")
    if not pkl_path.exists():
        print("UniAD PKL is missing. Skipping matching for this split.")
        return

    uniad_index = build_uniad_index(pkl_path)
    print(f"UniAD indexed samples: {len(uniad_index)}")

    matched = 0
    unmatched = []
    examples = []

    for scene_token, frame_token, frame in iterate_drivelm_frames(drivelm):
        info = uniad_index.get(frame_token)
        if info is None:
            unmatched.append((scene_token, frame_token))
            continue

        matched += 1
        if len(examples) < num_examples:
            drivelm_front = frame["image_paths"].get("CAM_FRONT")
            uniad_front = info.get("cams", {}).get("CAM_FRONT", {}).get("data_path")
            qa_group, question, answer = get_first_qa(frame)
            examples.append(
                {
                    "scene_token": scene_token,
                    "frame_token": frame_token,
                    "uniad_scene_token": info.get("scene_token"),
                    "frame_idx": info.get("frame_idx"),
                    "drivelm_front": drivelm_front,
                    "uniad_front": uniad_front,
                    "front_path_match": normalize_path(drivelm_front) == normalize_path(uniad_front),
                    "qa_group": qa_group,
                    "question": question,
                    "answer": answer,
                }
            )

    print(f"Matched frames: {matched}/{total_frames}")
    print(f"Unmatched frames: {len(unmatched)}")

    if unmatched:
        print("First unmatched tokens:")
        for scene_token, frame_token in unmatched[: min(5, len(unmatched))]:
            print(f"  scene={scene_token} frame={frame_token}")

    if examples:
        print("\nSample matched pairs:")
        for idx, example in enumerate(examples, start=1):
            print(f"[{idx}] scene_token       : {example['scene_token']}")
            print(f"    frame_token       : {example['frame_token']}")
            print(f"    uniad_scene_token : {example['uniad_scene_token']}")
            print(f"    frame_idx         : {example['frame_idx']}")
            print(f"    drivelm_front     : {example['drivelm_front']}")
            print(f"    uniad_front       : {example['uniad_front']}")
            print(f"    front_path_match  : {example['front_path_match']}")
            print(f"    qa_group          : {example['qa_group']}")
            print(f"    question          : {example['question']}")
            print(f"    answer            : {example['answer']}")


def main():
    parser = argparse.ArgumentParser(description="Match DriveLM frames with UniAD nuScenes infos by token.")
    parser.add_argument(
        "--drivelm-train-json",
        type=Path,
        default=Path("challenge/data/v1_1_train_nus.json"),
    )
    parser.add_argument(
        "--drivelm-val-json",
        type=Path,
        default=Path("challenge/data/v1_1_val_nus_q_only.json"),
    )
    parser.add_argument(
        "--uniad-train-pkl",
        type=Path,
        default=Path("UniAD/data/infos/nuscenes_infos_temporal_train.pkl"),
    )
    parser.add_argument(
        "--uniad-val-pkl",
        type=Path,
        default=Path("UniAD/data/infos/nuscenes_infos_temporal_val.pkl"),
    )
    parser.add_argument("--num-examples", type=int, default=3)
    args = parser.parse_args()

    summarize_split("train", args.drivelm_train_json, args.uniad_train_pkl, args.num_examples)
    summarize_split("val", args.drivelm_val_json, args.uniad_val_pkl, args.num_examples)


if __name__ == "__main__":
    main()
