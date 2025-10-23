"""Evaluation utilities for SPLNet."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

from .data import EpisodeWindowBundle, load_evaluation_episodes
from .model import SPLNet
from .utils import NormalizationStats, discover_tasks, load_json, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SPLNet failure detection performance.")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Root directory with task rollouts.")
    parser.add_argument("--result-dir", type=Path, default=Path("result"), help="Directory containing trained artefacts.")
    parser.add_argument("--tasks", type=str, nargs="*", default=None, help="Specific tasks to evaluate; defaults to tasks with saved models.")
    parser.add_argument("--window-size", type=int, default=32, help="Sliding window length used during training.")
    parser.add_argument("--stride", type=int, default=4, help="Stride used during training.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for evaluation windows.")
    parser.add_argument("--output", type=Path, default=Path("result/eval_metrics.json"), help="Where to store evaluation metrics.")
    return parser.parse_args()


def load_model(
    result_dir: Path,
) -> Tuple[
    SPLNet,
    NormalizationStats,
    float,
    Dict[str, object],
    Dict[str, Dict[str, float]],
    np.ndarray,
    np.ndarray,
    str,
]:
    config = load_json(result_dir / "train_config.json")
    stats = NormalizationStats.from_json(load_json(result_dir / "normalization_stats.json"))
    threshold_info = load_json(result_dir / "threshold.json")
    threshold = float(threshold_info["threshold"])
    aggregation = str(threshold_info.get("aggregation", "max"))
    calibration_path = result_dir / "calibration_components.json"
    if not calibration_path.exists():
        raise FileNotFoundError(f"Missing calibration profile at {calibration_path}.")
    calibration_profile = load_json(calibration_path)["components"]
    latent_mean_path = result_dir / "latent_mean.npy"
    latent_precision_path = result_dir / "latent_precision.npy"
    if not latent_mean_path.exists() or not latent_precision_path.exists():
        raise FileNotFoundError("Latent statistics files are missing for evaluation.")
    latent_mean = np.load(latent_mean_path)
    latent_precision = np.load(latent_precision_path)
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
    return (
        model,
        stats,
        threshold,
        config,
        calibration_profile,
        latent_mean,
        latent_precision,
        aggregation,
    )


def mahalanobis_distance_torch(
    vectors: torch.Tensor, mean: torch.Tensor, precision: torch.Tensor
) -> torch.Tensor:
    diff = vectors - mean.unsqueeze(0)
    return torch.sum(torch.matmul(diff, precision) * diff, dim=-1)


def combine_components_torch(
    values: Dict[str, torch.Tensor], component_stats: Dict[str, Dict[str, float]]
) -> torch.Tensor:
    reference = next(iter(values.values()))
    combined = torch.zeros_like(reference)
    for name, stats in component_stats.items():
        if name not in values:
            continue
        weight = float(stats.get("weight", 0.0))
        if weight == 0.0:
            continue
        mean = reference.new_tensor(float(stats.get("mean", 0.0)))
        std = reference.new_tensor(float(stats.get("std", 1.0)))
        std = torch.clamp(std, min=1e-6)
        z_score = (values[name] - mean) / std
        combined = combined + weight * z_score
    return combined


def aggregate_scores(values: np.ndarray, aggregation: str) -> float:
    """Aggregate window scores according to the configured strategy."""

    if values.size == 0:
        return 0.0
    agg = aggregation.lower()
    if agg == "mean":
        return float(np.mean(values))
    if agg == "median":
        return float(np.median(values))
    if agg.startswith("quantile"):
        try:
            _, frac = agg.split(":", 1)
            quantile = float(frac)
        except ValueError:
            quantile = 0.95
        quantile = float(np.clip(quantile, 0.0, 1.0))
        return float(np.quantile(values, quantile))
    return float(np.max(values))


def score_episode(
    model: SPLNet,
    bundle: EpisodeWindowBundle,
    device: torch.device,
    batch_size: int,
    component_stats: Dict[str, Dict[str, float]],
    latent_mean: np.ndarray,
    latent_precision: np.ndarray,
) -> np.ndarray:
    scores: List[np.ndarray] = []
    windows = bundle.windows
    if not windows:
        return np.zeros(1, dtype=np.float32)
    mean_tensor = torch.from_numpy(latent_mean.astype(np.float32)).to(device)
    precision_tensor = torch.from_numpy(latent_precision.astype(np.float32)).to(device)
    for start in range(0, len(windows), batch_size):
        chunk = windows[start : start + batch_size]
        batch = torch.from_numpy(np.stack(chunk, axis=0)).to(device)
        with torch.no_grad():
            components = model.score_components(batch)
            mahal = mahalanobis_distance_torch(components["latent_summary"], mean_tensor, precision_tensor)
            component_values: Dict[str, torch.Tensor] = {
                "reconstruction": components["reconstruction"],
                "dynamics": components["dynamics"],
                "latent_energy": components["latent_energy"],
                "mahalanobis": mahal,
            }
            combined = combine_components_torch(component_values, component_stats)
        scores.append(combined.cpu().numpy())
    return np.concatenate(scores, axis=0)


def compute_metrics(
    episodes: List[EpisodeWindowBundle],
    episode_scores: List[np.ndarray],
    threshold: float,
    aggregation: str,
) -> Dict[str, float]:
    assert len(episodes) == len(episode_scores)
    tp = fp = tn = fn = 0
    total_steps = 0
    correct_steps = 0
    for bundle, scores in zip(episodes, episode_scores):
        label = 0 if bundle.info.successful else 1
        episode_score = aggregate_scores(scores, aggregation)
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
        "total_steps": total_steps,
        "correct_steps": correct_steps,
        "aggregation": aggregation,
    }


def aggregate_metrics(metrics_list: Sequence[Dict[str, float]]) -> Dict[str, float]:
    totals = {"TP": 0.0, "TN": 0.0, "FP": 0.0, "FN": 0.0, "total_steps": 0.0, "correct_steps": 0.0}
    for metrics in metrics_list:
        totals["TP"] += float(metrics.get("TP", 0.0))
        totals["TN"] += float(metrics.get("TN", 0.0))
        totals["FP"] += float(metrics.get("FP", 0.0))
        totals["FN"] += float(metrics.get("FN", 0.0))
        totals["total_steps"] += float(metrics.get("total_steps", 0.0))
        totals["correct_steps"] += float(metrics.get("correct_steps", 0.0))

    total_episodes = totals["TP"] + totals["TN"] + totals["FP"] + totals["FN"]
    accuracy = (totals["TP"] + totals["TN"]) / total_episodes if total_episodes else 0.0
    tpr = totals["TP"] / (totals["TP"] + totals["FN"]) if (totals["TP"] + totals["FN"]) else 0.0
    tnr = totals["TN"] / (totals["TN"] + totals["FP"]) if (totals["TN"] + totals["FP"]) else 0.0
    twa = totals["correct_steps"] / totals["total_steps"] if totals["total_steps"] else 0.0

    return {
        "accuracy": accuracy,
        "TPR": tpr,
        "TNR": tnr,
        "TWA": twa,
        "episodes": int(total_episodes),
        "TP": int(totals["TP"]),
        "TN": int(totals["TN"]),
        "FP": int(totals["FP"]),
        "FN": int(totals["FN"]),
        "total_steps": int(totals["total_steps"]),
        "correct_steps": int(totals["correct_steps"]),
        "evaluated_tasks": len(metrics_list),
    }


def evaluate_task(
    task: str,
    args: argparse.Namespace,
    device: torch.device,
) -> Optional[Tuple[Dict[str, float], Dict[str, object]]]:
    task_result_dir = args.result_dir / task
    if not task_result_dir.exists():
        raise FileNotFoundError(f"Missing artefacts for task '{task}' at {task_result_dir}.")

    (
        model,
        stats,
        threshold,
        config,
        component_stats,
        latent_mean,
        latent_precision,
        aggregation,
    ) = load_model(task_result_dir)
    model.to(device)
    model.eval()

    window_size = int(config.get("window_size", args.window_size))
    stride = int(config.get("stride", args.stride))

    episodes = load_evaluation_episodes(
        args.data_root,
        stats=stats,
        window_size=window_size,
        stride=stride,
        tasks=[task],
    )
    if not episodes:
        print(f"[WARN] No evaluation episodes for task '{task}'. Skipping.")
        return None

    episode_scores: List[np.ndarray] = []
    for bundle in tqdm(episodes, desc=f"{task}:episodes"):
        scores = score_episode(
            model,
            bundle,
            device,
            args.batch_size,
            component_stats,
            latent_mean,
            latent_precision,
        )
        episode_scores.append(scores)

    metrics = compute_metrics(episodes, episode_scores, threshold, aggregation)
    num_windows = int(sum(len(scores) for scores in episode_scores))
    metrics.update(
        {
            "task": task,
            "threshold": float(threshold),
            "window_size": window_size,
            "stride": stride,
            "num_windows": num_windows,
        }
    )

    details = []
    for bundle, scores in zip(episodes, episode_scores):
        aggregate_value = aggregate_scores(scores, aggregation)
        details.append(
            {
                "task": bundle.info.task,
                "episode": bundle.info.path.stem,
                "successful": bundle.info.successful,
                "aggregate_score": aggregate_value,
                "aggregation": aggregation,
                "max_score": float(np.max(scores)),
                "mean_score": float(np.mean(scores)),
                "num_windows": len(scores),
            }
        )

    save_json(metrics, task_result_dir / "eval_metrics.json")
    save_json(
        {
            "threshold": float(threshold),
            "window_size": window_size,
            "stride": stride,
            "details": details,
        },
        task_result_dir / "eval_details.json",
    )

    print(json.dumps(metrics, indent=2))

    return metrics, {
        "threshold": float(threshold),
        "window_size": window_size,
        "stride": stride,
        "details": details,
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.tasks:
        requested_tasks: Sequence[str] = args.tasks
    else:
        trained_tasks = sorted([p.name for p in args.result_dir.iterdir() if p.is_dir()]) if args.result_dir.exists() else []
        requested_tasks = trained_tasks if trained_tasks else discover_tasks(args.data_root)

    if not requested_tasks:
        raise ValueError("No tasks available for evaluation.")

    metrics_map: Dict[str, Dict[str, float]] = {}
    details_map: Dict[str, Dict[str, object]] = {}
    collected_metrics: List[Dict[str, float]] = []

    for task in requested_tasks:
        try:
            result = evaluate_task(task, args, device)
        except FileNotFoundError as exc:
            print(f"[WARN] {exc}")
            continue
        if result is None:
            continue
        metrics, task_details = result
        metrics_map[task] = metrics
        details_map[task] = task_details
        collected_metrics.append(metrics)

    if not collected_metrics:
        raise RuntimeError("Evaluation produced no results; ensure models and test rollouts exist.")

    overall = aggregate_metrics(collected_metrics)
    print("[OVERALL]")
    print(json.dumps(overall, indent=2))

    payload = {"tasks": metrics_map, "overall": overall}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(payload, args.output)

    detail_path = args.output.with_name("eval_details.json")
    save_json({"tasks": details_map, "overall": overall}, detail_path)


if __name__ == "__main__":
    main()
