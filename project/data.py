"""Dataset utilities for SPL training and evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import (
    EpisodeInfo,
    NormalizationStats,
    apply_normalization,
    compute_normalization_stats,
    episode_windows,
    list_rollout_files,
)


@dataclass
class EpisodeWindowBundle:
    info: EpisodeInfo
    windows: List[np.ndarray]
    step_count: int


class SuccessWindowDataset(Dataset):
    """Dataset of sliding windows built from successful calibration rollouts."""

    def __init__(
        self,
        root: Path,
        window_size: int,
        stride: int,
        stats: NormalizationStats | None = None,
        tasks: Optional[Sequence[str]] = None,
    ) -> None:
        self.root = root
        self.window_size = window_size
        self.stride = stride
        self._raw_samples: List[np.ndarray] = []
        self._episodes: List[EpisodeInfo] = []
        self._episode_windows_raw: List[tuple[EpisodeInfo, List[np.ndarray]]] = []
        for info in list_rollout_files(root, split="train", tasks=tasks):
            features, windows = episode_windows(info.path, window_size=window_size, stride=stride)
            if not windows:
                continue
            self._raw_samples.extend(windows)
            self._episodes.append(info)
            self._episode_windows_raw.append((info, windows))
        if not self._raw_samples:
            raise ValueError("No successful calibration rollouts found.")
        if stats is None:
            self.stats = compute_normalization_stats(self._raw_samples)
        else:
            self.stats = stats
        self.samples: List[np.ndarray] = []
        self._episode_windows: List[tuple[EpisodeInfo, List[np.ndarray]]] = []
        for info, windows in self._episode_windows_raw:
            normalized = [apply_normalization(sample, self.stats).astype(np.float32) for sample in windows]
            self.samples.extend(normalized)
            self._episode_windows.append((info, normalized))
        self.feature_dim = self.samples[0].shape[-1]
        self.episodes = [info for info, _ in self._episode_windows]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        array = self.samples[idx]
        return torch.from_numpy(array)

    def iter_original_windows(self) -> Iterable[np.ndarray]:
        for sample in self._raw_samples:
            yield sample

    def iter_episode_windows(self) -> Iterable[tuple[EpisodeInfo, List[np.ndarray]]]:
        for info, windows in self._episode_windows:
            yield info, windows


def load_evaluation_episodes(
    root: Path,
    stats: NormalizationStats,
    window_size: int,
    stride: int,
    tasks: Optional[Sequence[str]] = None,
) -> List[EpisodeWindowBundle]:
    bundles: List[EpisodeWindowBundle] = []
    for info in list_rollout_files(root, split="eval", tasks=tasks):
        features, windows = episode_windows(info.path, window_size=window_size, stride=stride)
        normalized = [apply_normalization(window, stats).astype(np.float32) for window in windows]
        bundles.append(EpisodeWindowBundle(info=info, windows=normalized, step_count=features.shape[0]))
    return bundles
