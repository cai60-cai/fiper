"""Evaluation utilities for SPLNet."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

from .data import EpisodeWindowBundle, load_evaluation_episodes
from .model import SPLNet
from .utils import NormalizationStats, load_json, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SPLNet failure detection performance.")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Root directory with task rollouts.")
    parser.add_argument("--result-dir", type=Path, default=Path("result"), help="Directory containing trained artefacts.")
    parser.add_argument("--window-size", type=int, default=32, help="Sliding window length used during training.")
    parser.add_argument("--stride", type=int, default=4, help="Stride used during training.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for evaluation windows.")
    parser.add_argument("--output", type=Path, default=Path("result/eval_metrics.json"), help="Where to store evaluation metrics.")
    return parser.parse_args()


def load_model(result_dir: Path) -> Tuple[SPLNet, NormalizationStats, float, Dict[str, object]]:
    config = load_json(result_dir / "train_config.json")
    stats = NormalizationStats.from_json(load_json(result_dir / "normalization_stats.json"))
    threshold_info = load_json(result_dir / "threshold.json")
    threshold = float(threshold_info["threshold"])
    model = SPLNet(
        input_dim=config["feature_dim"],
        hidden_dim=config.get("hidden_dim", 256),
        latent_dim=config.get("latent_dim", 128),
        num_layers=config.get("layers", 4),
        num_heads=config.get("heads", 8),
        dropout=config.get("dropout", 0.1),
    )
    state_path = result_dir / "splnet.pt"
    model.load_state_dict(torch.load(state_path, map_location="cpu"))
    return model, stats, threshold, config


def score_episode(
    model: SPLNet,
    bundle: EpisodeWindowBundle,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    scores: List[np.ndarray] = []
    windows = bundle.windows
    if not windows:
        return np.zeros(1, dtype=np.float32)
    for start in range(0, len(windows), batch_size):
        chunk = windows[start : start + batch_size]
        batch = torch.from_numpy(np.stack(chunk, axis=0)).to(device)
        with torch.no_grad():
            values = model.reconstruction_score(batch)
        scores.append(values.cpu().numpy())
    return np.concatenate(scores, axis=0)


def compute_metrics(
    episodes: List[EpisodeWindowBundle],
    episode_scores: List[np.ndarray],
    threshold: float,
) -> Dict[str, float]:
    assert len(episodes) == len(episode_scores)
    tp = fp = tn = fn = 0
    total_steps = 0
    correct_steps = 0
    for bundle, scores in zip(episodes, episode_scores):
        label = 0 if bundle.info.successful else 1
        episode_score = float(np.max(scores))
        prediction = 1 if episode_score > threshold else 0
        if prediction == 1 and label == 1:
            tp += 1
        elif prediction == 1 and label == 0:
            fp += 1
        elif prediction == 0 and label == 0:
            tn += 1
        else:
            fn += 1
        total_steps += bundle.step_count
        if prediction == label:
            correct_steps += bundle.step_count
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    twa = correct_steps / total_steps if total_steps else 0.0
    return {
        "accuracy": accuracy,
        "TPR": tpr,
        "TNR": tnr,
        "TWA": twa,
        "episodes": total,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, stats, threshold, config = load_model(args.result_dir)
    model.to(device)
    model.eval()

    episodes = load_evaluation_episodes(args.data_root, stats=stats, window_size=args.window_size, stride=args.stride)
    episode_scores: List[np.ndarray] = []
    for bundle in tqdm(episodes, desc="episodes"):
        scores = score_episode(model, bundle, device, args.batch_size)
        episode_scores.append(scores)

    metrics = compute_metrics(episodes, episode_scores, threshold)
    print(json.dumps(metrics, indent=2))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(metrics, args.output)

    detail_path = args.output.with_name("eval_details.json")
    details = []
    for bundle, scores in zip(episodes, episode_scores):
        details.append(
            {
                "task": bundle.info.task,
                "episode": bundle.info.path.stem,
                "successful": bundle.info.successful,
                "max_score": float(np.max(scores)),
                "mean_score": float(np.mean(scores)),
                "num_windows": len(scores),
            }
        )
    save_json({"threshold": threshold, "details": details}, detail_path)


if __name__ == "__main__":
    main()
