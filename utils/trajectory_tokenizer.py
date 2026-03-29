import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


NUM_AXIS_BINS = 128


@dataclass
class TrajectoryTokenCodec:
    dx_min: float
    dx_max: float
    dy_min: float
    dy_max: float
    future_steps: int = 12
    num_axis_bins: int = NUM_AXIS_BINS
    dx_tokens: List[str] = None
    dy_tokens: List[str] = None
    sot_token: str = "<traj_sot>"
    eot_token: str = "<traj_eot>"

    def __post_init__(self):
        if self.dx_tokens is None:
            self.dx_tokens = [f"<traj_dx_{i:03d}>" for i in range(self.num_axis_bins)]
        if self.dy_tokens is None:
            self.dy_tokens = [f"<traj_dy_{i:03d}>" for i in range(self.num_axis_bins)]
        self.dx_edges = np.linspace(self.dx_min, self.dx_max, self.num_axis_bins + 1, dtype=np.float32)
        self.dy_edges = np.linspace(self.dy_min, self.dy_max, self.num_axis_bins + 1, dtype=np.float32)
        self.dx_centers = ((self.dx_edges[:-1] + self.dx_edges[1:]) / 2.0).astype(np.float32)
        self.dy_centers = ((self.dy_edges[:-1] + self.dy_edges[1:]) / 2.0).astype(np.float32)
        self.dx_to_index = {token: idx for idx, token in enumerate(self.dx_tokens)}
        self.dy_to_index = {token: idx for idx, token in enumerate(self.dy_tokens)}

    @property
    def added_tokens(self) -> List[str]:
        return self.dx_tokens + self.dy_tokens + [self.sot_token, self.eot_token]

    def to_dict(self) -> Dict:
        return {
            "dx_min": self.dx_min,
            "dx_max": self.dx_max,
            "dy_min": self.dy_min,
            "dy_max": self.dy_max,
            "future_steps": self.future_steps,
            "num_axis_bins": self.num_axis_bins,
            "dx_tokens": self.dx_tokens,
            "dy_tokens": self.dy_tokens,
            "sot_token": self.sot_token,
            "eot_token": self.eot_token,
        }

    @classmethod
    def from_dict(cls, payload: Dict) -> "TrajectoryTokenCodec":
        return cls(**payload)

    @classmethod
    def build_from_dataset(
        cls,
        dataset,
        lower_percentile: float = 1.0,
        upper_percentile: float = 99.0,
        future_steps: int = 12,
    ) -> "TrajectoryTokenCodec":
        dx_values = []
        dy_values = []
        for idx in range(len(dataset)):
            sample = dataset[idx]
            traj = np.asarray(sample["ego_future_traj"], dtype=np.float32)
            mask = np.asarray(sample["ego_future_mask"], dtype=bool)
            valid = traj[mask]
            if len(valid) == 0:
                continue
            deltas = np.diff(np.vstack([np.zeros((1, 2), dtype=np.float32), valid]), axis=0)
            dx_values.append(deltas[:, 0])
            dy_values.append(deltas[:, 1])

        dx_values = np.concatenate(dx_values, axis=0)
        dy_values = np.concatenate(dy_values, axis=0)
        dx_min, dx_max = np.percentile(dx_values, [lower_percentile, upper_percentile]).tolist()
        dy_min, dy_max = np.percentile(dy_values, [lower_percentile, upper_percentile]).tolist()
        return cls(
            dx_min=float(dx_min),
            dx_max=float(dx_max),
            dy_min=float(dy_min),
            dy_max=float(dy_max),
            future_steps=future_steps,
        )

    def encode_trajectory(self, traj: np.ndarray, mask: np.ndarray) -> List[str]:
        valid = np.asarray(traj, dtype=np.float32)[np.asarray(mask, dtype=bool)]
        if len(valid) == 0:
            return [self.sot_token, self.eot_token]

        deltas = np.diff(np.vstack([np.zeros((1, 2), dtype=np.float32), valid]), axis=0)
        tokens = [self.sot_token]
        for step in range(min(len(deltas), self.future_steps)):
            dx_idx = self._quantize(deltas[step, 0], self.dx_edges)
            dy_idx = self._quantize(deltas[step, 1], self.dy_edges)
            tokens.append(self.dx_tokens[dx_idx])
            tokens.append(self.dy_tokens[dy_idx])
        tokens.append(self.eot_token)
        return tokens

    def decode_token_ids(
        self,
        token_ids: Sequence[int],
        tokenizer,
    ) -> np.ndarray:
        tokens = tokenizer.convert_ids_to_tokens(list(token_ids))
        return self.decode_tokens(tokens)

    def analyze_token_ids(self, token_ids: Sequence[int], tokenizer) -> Dict:
        tokens = tokenizer.convert_ids_to_tokens(list(token_ids))
        return self.analyze_tokens(tokens)

    def decode_tokens(self, tokens: Sequence[str]) -> np.ndarray:
        analysis = self.analyze_tokens(tokens)
        parsed_pairs = analysis["parsed_pairs"]
        if len(parsed_pairs) == 0:
            offsets = np.zeros((self.future_steps, 2), dtype=np.float32)
        else:
            offsets = np.zeros((self.future_steps, 2), dtype=np.float32)
            parsed_pairs = np.asarray(parsed_pairs, dtype=np.float32)
            steps = min(len(parsed_pairs), self.future_steps)
            offsets[:steps] = parsed_pairs[:steps]
        return np.cumsum(offsets, axis=0)

    def analyze_tokens(self, tokens: Sequence[str]) -> Dict:
        parsed_pairs = []
        started = False
        saw_sot = False
        saw_eot = False
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token == self.sot_token:
                started = True
                saw_sot = True
                i += 1
                continue
            if not started:
                i += 1
                continue
            if token == self.eot_token:
                saw_eot = True
                break
            if i + 1 >= len(tokens):
                break
            dx_token = tokens[i]
            dy_token = tokens[i + 1]
            if dx_token not in self.dx_to_index or dy_token not in self.dy_to_index:
                i += 1
                continue
            parsed_pairs.append(
                [
                    float(self.dx_centers[self.dx_to_index[dx_token]]),
                    float(self.dy_centers[self.dy_to_index[dy_token]]),
                ]
            )
            i += 2

        return {
            "tokens": list(tokens),
            "saw_sot": saw_sot,
            "saw_eot": saw_eot,
            "num_pairs": len(parsed_pairs),
            "parsed_pairs": parsed_pairs,
        }

    def tokens_to_text(self, tokens: Sequence[str]) -> str:
        return "".join(tokens)

    def _quantize(self, value: float, edges: np.ndarray) -> int:
        clipped = float(np.clip(value, edges[0], edges[-1]))
        idx = int(np.digitize(clipped, edges[1:-1], right=False))
        return max(0, min(idx, len(edges) - 2))


def deterministic_perception_index(frame_token: str, num_questions: int) -> int:
    digest = hashlib.sha256(frame_token.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % num_questions


def save_codec(path: Path, codec: TrajectoryTokenCodec) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(codec.to_dict(), indent=2))


def load_codec(path: Path) -> TrajectoryTokenCodec:
    return TrajectoryTokenCodec.from_dict(json.loads(path.read_text()))
