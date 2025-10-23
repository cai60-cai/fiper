"""Utility helpers for SPL project."""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Maximum dimensions for padded feature blocks.
OBS_DIM = 512
STATE_DIM = 256
AGENT_POS_DIM = 32
ACTION_DIM = 32


@dataclass(frozen=True)
class NormalizationStats:
    """Mean and standard deviation for feature scaling."""

    mean: np.ndarray
    std: np.ndarray

    def to_json(self) -> Dict[str, List[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @staticmethod
    def from_json(data: Dict[str, Sequence[float]]) -> "NormalizationStats":
        mean = np.asarray(data["mean"], dtype=np.float32)
        std = np.asarray(data["std"], dtype=np.float32)
        return NormalizationStats(mean=mean, std=std)


@dataclass(frozen=True)
class EpisodeInfo:
    """Metadata describing a rollout episode."""

    task: str
    path: Path
    successful: bool
    num_steps: int


def _load_pickle(path: Path) -> Dict[str, object]:
    with path.open("rb") as f:
        return pickle.load(f)


def _ensure_float_array(value: Optional[object]) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.dtype == object:
        arr = np.array(arr.tolist(), dtype=np.float32)
    arr = arr.astype(np.float32)
    return arr


def _pad_or_truncate(arr: np.ndarray, dim: int) -> np.ndarray:
    flat = arr.reshape(-1)
    if flat.size >= dim:
        return flat[:dim]
    result = np.zeros(dim, dtype=np.float32)
    result[: flat.size] = flat
    return result


def extract_step_features(step: Dict[str, object]) -> np.ndarray:
    """Convert a rollout step dictionary into a fixed-length feature vector."""

    obs = _ensure_float_array(step.get("obs_embedding"))
    obs_vec = _pad_or_truncate(obs, OBS_DIM) if obs is not None else np.zeros(OBS_DIM, dtype=np.float32)

    state = _ensure_float_array(step.get("state_embedding"))
    state_vec = _pad_or_truncate(state, STATE_DIM) if state is not None else np.zeros(STATE_DIM, dtype=np.float32)

    agent = _ensure_float_array(step.get("agent_pos"))
    agent_vec = _pad_or_truncate(agent, AGENT_POS_DIM) if agent is not None else np.zeros(AGENT_POS_DIM, dtype=np.float32)

    action = _ensure_float_array(step.get("action"))
    if action is not None:
        if action.ndim > 1:
            action = action.reshape(action.shape[0], -1)[0]
        action_vec = _pad_or_truncate(action, ACTION_DIM)
    else:
        action_vec = np.zeros(ACTION_DIM, dtype=np.float32)

    action_pred = _ensure_float_array(step.get("action_pred"))
    if action_pred is not None:
        if action_pred.ndim == 1:
            mean_vec = action_pred
            std_vec = np.zeros_like(mean_vec)
        else:
            reshaped = action_pred.reshape(-1, action_pred.shape[-1])
            mean_vec = reshaped.mean(axis=0)
            std_vec = reshaped.std(axis=0)
        action_pred_mean = _pad_or_truncate(mean_vec, ACTION_DIM)
        action_pred_std = _pad_or_truncate(std_vec, ACTION_DIM)
    else:
        action_pred_mean = np.zeros(ACTION_DIM, dtype=np.float32)
        action_pred_std = np.zeros(ACTION_DIM, dtype=np.float32)

    feature = np.concatenate(
        [obs_vec, state_vec, agent_vec, action_vec, action_pred_mean, action_pred_std],
        axis=0,
    ).astype(np.float32)
    return feature


def sliding_windows(array: np.ndarray, window_size: int, stride: int) -> List[np.ndarray]:
    if array.shape[0] < window_size:
        pad_length = window_size - array.shape[0]
        padding = np.zeros((pad_length, array.shape[1]), dtype=array.dtype)
        working = np.concatenate([array, padding], axis=0)
    else:
        working = array
    windows: List[np.ndarray] = []
    max_start = working.shape[0] - window_size
    if max_start < 0:
        windows.append(working[-window_size:])
        return windows
    for start in range(0, max_start + 1, stride):
        windows.append(working[start : start + window_size])
    if not windows:
        windows.append(working[-window_size:])
    return windows


def compute_normalization_stats(samples: Iterable[np.ndarray]) -> NormalizationStats:
    total_steps = 0
    sum_vec: Optional[np.ndarray] = None
    sum_sq_vec: Optional[np.ndarray] = None
    for seq in samples:
        seq = seq.astype(np.float32)
        if sum_vec is None:
            sum_vec = np.zeros(seq.shape[-1], dtype=np.float32)
            sum_sq_vec = np.zeros(seq.shape[-1], dtype=np.float32)
        sum_vec += seq.sum(axis=0)
        sum_sq_vec += np.square(seq).sum(axis=0)
        total_steps += seq.shape[0]
    assert sum_vec is not None and sum_sq_vec is not None, "No samples provided for normalization"
    mean = sum_vec / float(total_steps)
    var = sum_sq_vec / float(total_steps) - np.square(mean)
    std = np.sqrt(np.maximum(var, 1e-6)).astype(np.float32)
    return NormalizationStats(mean=mean.astype(np.float32), std=std)


def apply_normalization(sample: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    return (sample - stats.mean) / stats.std


def discover_tasks(root: Path) -> List[str]:
    """Return sorted task directory names available under ``root``."""

    tasks: List[str] = []
    for task_dir in sorted(root.iterdir()):
        if task_dir.is_dir() and (task_dir / "rollouts").exists():
            tasks.append(task_dir.name)
    return tasks


def list_rollout_files(
    root: Path,
    split: str,
    tasks: Optional[Sequence[str]] = None,
) -> List[EpisodeInfo]:
    """Enumerate rollout files for a given split.

    Args:
        root: Root directory containing task subdirectories.
        split: One of {"train", "eval"}.
    """

    if split not in {"train", "eval"}:
        raise ValueError(f"Unsupported split: {split}")

    rollouts: List[EpisodeInfo] = []
    task_filter = set(tasks) if tasks else None
    for task_dir in sorted(root.iterdir()):
        if not task_dir.is_dir():
            continue
        if task_filter and task_dir.name not in task_filter:
            continue
        rollout_dir = task_dir / "rollouts"
        if not rollout_dir.exists():
            continue
        if split == "train":
            candidate_dirs = [rollout_dir / "calibration", rollout_dir / "calibration_unused"]
        else:
            candidate_dirs = [rollout_dir / "test"]
        for directory in candidate_dirs:
            if not directory.exists():
                continue
            for path in sorted(directory.glob("*.pkl")):
                payload = _load_pickle(path)
                metadata = payload.get("metadata", {})
                successful = bool(metadata.get("successful", False))
                if split == "train" and not successful:
                    continue
                rollout = payload.get("rollout")
                num_steps = len(rollout) if isinstance(rollout, Sequence) else int(metadata.get("num_steps", 0))
                rollouts.append(EpisodeInfo(task=task_dir.name, path=path, successful=successful, num_steps=num_steps))
    return rollouts


def save_json(data: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def episode_windows(path: Path, window_size: int, stride: int) -> Tuple[np.ndarray, List[np.ndarray]]:
    payload = _load_pickle(path)
    rollout: Sequence[Dict[str, object]] = payload["rollout"]
    features = np.stack([extract_step_features(step) for step in rollout], axis=0)
    windows = sliding_windows(features, window_size=window_size, stride=stride)
    return features, windows
