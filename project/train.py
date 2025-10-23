"""Training script for the SPLNet stability model."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Configure matplotlib to work in headless environments.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data import SuccessWindowDataset
from .model import SPLNet
from .utils import EpisodeInfo, NormalizationStats, discover_tasks, save_json


COMPONENT_WEIGHTS = {
    "reconstruction": 0.45,
    "dynamics": 0.15,
    "latent_energy": 0.15,
    "mahalanobis": 0.25,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SPLNet on successful calibration rollouts.")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Root directory containing task rollouts.")
    parser.add_argument("--tasks", type=str, nargs="*", default=None, help="Specific task names to train; defaults to all tasks found under data root.")
    parser.add_argument("--window-size", type=int, default=32, help="Sliding window length in steps.")
    parser.add_argument("--stride", type=int, default=4, help="Stride for sliding windows.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Transformer hidden dimension.")
    parser.add_argument("--latent-dim", type=int, default=128, help="Latent summary dimension.")
    parser.add_argument("--layers", type=int, default=4, help="Number of transformer encoder layers.")
    parser.add_argument("--heads", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Transformer dropout.")
    parser.add_argument("--threshold-quantile", type=float, default=0.97, help="Quantile for anomaly threshold.")
    parser.add_argument("--result-dir", type=Path, default=Path("result"), help="Directory to store trained artefacts.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loader workers.")
    return parser.parse_args()


def build_dataset(args: argparse.Namespace, task: str) -> SuccessWindowDataset:
    dataset = SuccessWindowDataset(
        root=args.data_root,
        window_size=args.window_size,
        stride=args.stride,
        tasks=[task],
    )
    return dataset


def train_epoch(
    model: SPLNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    task_name: str,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc=f"{task_name}:train", leave=False):
        batch = batch.to(device)
        outputs = model(batch)
        recon_loss = F.mse_loss(outputs.reconstruction, batch)
        if batch.size(1) > 1:
            smooth_loss = F.mse_loss(outputs.encoded_sequence[:, 1:, :], outputs.encoded_sequence[:, :-1, :])
        else:
            smooth_loss = torch.tensor(0.0, device=device)
        latent_reg = torch.mean(torch.square(outputs.latent_summary))
        loss = recon_loss + 0.1 * smooth_loss + 1e-3 * latent_reg
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.detach().item()
    return total_loss / max(len(loader), 1)


def _gather_episode_windows(
    dataset: SuccessWindowDataset,
) -> tuple[List[np.ndarray], List[tuple[int, int]], List[EpisodeInfo]]:
    """Flatten per-episode windows while recording index slices."""

    all_windows: List[np.ndarray] = []
    window_slices: List[tuple[int, int]] = []
    episode_infos: List[EpisodeInfo] = []

    for info, windows in dataset.iter_episode_windows():
        if not windows:
            continue
        start = len(all_windows)
        all_windows.extend(windows)
        end = len(all_windows)
        if end <= start:
            continue
        window_slices.append((start, end))
        episode_infos.append(info)

    return all_windows, window_slices, episode_infos


def _score_windows(
    model: SPLNet,
    windows: List[np.ndarray],
    device: torch.device,
    task_name: str,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    """Run ``model.score_components`` over a list of windows."""

    model.eval()
    buffers: Dict[str, List[np.ndarray]] = {
        "reconstruction": [],
        "dynamics": [],
        "latent_energy": [],
        "latent_summary": [],
    }

    total = len(windows)
    if total == 0:
        raise ValueError("No calibration windows available for scoring.")

    ranges = range(0, total, batch_size)
    for start in tqdm(ranges, desc=f"{task_name}:calibrate", leave=False):
        chunk = windows[start : start + batch_size]
        batch_array = np.stack(chunk, axis=0).astype(np.float32)
        batch = torch.from_numpy(batch_array).to(device)
        with torch.no_grad():
            components = model.score_components(batch)
        buffers["reconstruction"].append(components["reconstruction"].cpu().numpy())
        buffers["dynamics"].append(components["dynamics"].cpu().numpy())
        buffers["latent_energy"].append(components["latent_energy"].cpu().numpy())
        buffers["latent_summary"].append(components["latent_summary"].cpu().numpy())

    return {key: np.concatenate(value, axis=0) for key, value in buffers.items()}


def compute_latent_statistics(latent_vectors: np.ndarray, ridge_coef: float = 1e-3) -> Dict[str, np.ndarray]:
    mean = latent_vectors.mean(axis=0)
    centered = latent_vectors - mean
    denom = max(latent_vectors.shape[0] - 1, 1)
    cov = (centered.T @ centered) / float(denom)
    cov = cov.astype(np.float32)
    trace = float(np.trace(cov))
    if not np.isfinite(trace) or trace <= 0.0:
        trace = float(cov.shape[0])
    ridge = ridge_coef * trace / float(cov.shape[0])
    cov += ridge * np.eye(cov.shape[0], dtype=np.float32)
    precision = np.linalg.inv(cov)
    return {
        "mean": mean.astype(np.float32),
        "precision": precision.astype(np.float32),
        "ridge": float(ridge),
    }


def compute_mahalanobis(latent_vectors: np.ndarray, mean: np.ndarray, precision: np.ndarray) -> np.ndarray:
    diff = latent_vectors - mean
    scores = np.einsum("bi,ij,bj->b", diff, precision, diff)
    return scores.astype(np.float32)


def combine_component_scores(components: Dict[str, np.ndarray]) -> tuple[np.ndarray, Dict[str, Dict[str, float]]]:
    stats: Dict[str, Dict[str, float]] = {}
    combined = None
    for name, values in components.items():
        mean = float(values.mean())
        std = float(values.std() + 1e-6)
        weight = float(COMPONENT_WEIGHTS.get(name, 0.0))
        stats[name] = {"mean": mean, "std": std, "weight": weight}
        normalized = (values - mean) / std
        if combined is None:
            combined = weight * normalized
        else:
            combined += weight * normalized
    assert combined is not None
    return combined.astype(np.float32), stats


def prepare_calibration_profile(
    model: SPLNet,
    dataset: SuccessWindowDataset,
    device: torch.device,
    task_name: str,
    batch_size: int,
) -> Dict[str, object]:
    """Compute calibration statistics using per-episode aggregation."""

    windows, slices, infos = _gather_episode_windows(dataset)
    if not windows:
        raise ValueError("No windows gathered for calibration.")

    raw_components = _score_windows(model, windows, device, task_name, batch_size)
    latent_vectors = raw_components.pop("latent_summary")
    latent_stats = compute_latent_statistics(latent_vectors)
    mahal = compute_mahalanobis(latent_vectors, latent_stats["mean"], latent_stats["precision"])
    raw_components["mahalanobis"] = mahal
    combined_scores, component_stats = combine_component_scores(raw_components)

    episode_maxima: List[float] = []
    episode_means: List[float] = []
    for start, end in slices:
        episode_slice = combined_scores[start:end]
        episode_maxima.append(float(np.max(episode_slice)))
        episode_means.append(float(np.mean(episode_slice)))

    profile: Dict[str, object] = {
        "combined_scores": combined_scores,
        "component_stats": component_stats,
        "latent_stats": latent_stats,
        "episode_maxima": np.asarray(episode_maxima, dtype=np.float32),
        "episode_means": np.asarray(episode_means, dtype=np.float32),
        "window_slices": slices,
        "episode_infos": infos,
    }
    return profile


def plot_losses(losses: List[float], path: Path) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(losses) + 1), losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPLNet Training Loss")
    plt.grid(True, linestyle="--", alpha=0.4)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def train_single_task(task: str, args: argparse.Namespace, device: torch.device) -> Dict[str, object]:
    print(f"[INFO] Training task '{task}'")
    dataset = build_dataset(args, task)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = SPLNet(
        input_dim=dataset.feature_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_layers=args.layers,
        num_heads=args.heads,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    losses: List[float] = []
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, loader, optimizer, device, task)
        losses.append(loss)
        print(f"[TRAIN] task={task} epoch={epoch:03d} loss={loss:.6f}")

    task_result_dir = args.result_dir / task
    task_result_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), task_result_dir / "splnet.pt")

    stats: NormalizationStats = dataset.stats
    save_json({"mean": stats.mean.tolist(), "std": stats.std.tolist()}, task_result_dir / "normalization_stats.json")

    plot_losses(losses, task_result_dir / "train_loss.png")
    save_json({"losses": losses}, task_result_dir / "train_loss.json")

    profile = prepare_calibration_profile(model, dataset, device, task, args.batch_size)
    combined_scores = profile["combined_scores"]
    episode_maxima: np.ndarray = profile["episode_maxima"]
    episode_means: np.ndarray = profile["episode_means"]
    component_stats = profile["component_stats"]
    latent_stats = profile["latent_stats"]
    if episode_maxima.size == 0:
        raise ValueError("Calibration produced no episode-level scores.")
    threshold = float(np.quantile(episode_maxima, args.threshold_quantile))
    save_json(
        {
            "threshold": threshold,
            "quantile": args.threshold_quantile,
            "scores_mean": float(combined_scores.mean()),
            "scores_std": float(combined_scores.std()),
            "episode_max_mean": float(episode_maxima.mean()),
            "episode_max_std": float(episode_maxima.std()),
            "num_scores": int(combined_scores.shape[0]),
            "num_episode_scores": int(episode_maxima.shape[0]),
            "component_weights": COMPONENT_WEIGHTS,
            "aggregation": "max",
            "calibration_target": "episode_max",
        },
        task_result_dir / "threshold.json",
    )
    save_json(
        {
            "components": component_stats,
            "num_windows": int(combined_scores.shape[0]),
            "latent_dim": int(latent_stats["mean"].shape[0]),
            "ridge": float(latent_stats["ridge"]),
            "calibration_target": "episode_max",
            "num_episode_scores": int(episode_maxima.shape[0]),
            "episode_max_mean": float(episode_maxima.mean()),
            "episode_max_std": float(episode_maxima.std()),
        },
        task_result_dir / "calibration_components.json",
    )
    calibration_details = []
    episode_infos: List[EpisodeInfo] = profile["episode_infos"]
    window_slices: List[tuple[int, int]] = profile["window_slices"]
    for idx, info in enumerate(episode_infos):
        start, end = window_slices[idx]
        calibration_details.append(
            {
                "task": info.task,
                "episode": info.path.stem,
                "successful": True,
                "num_windows": int(end - start),
                "max_score": float(episode_maxima[idx]),
                "mean_score": float(episode_means[idx]),
                "aggregation": "max",
            }
        )
    save_json({"episodes": calibration_details, "aggregation": "max"}, task_result_dir / "calibration_details.json")
    np.save(task_result_dir / "latent_mean.npy", latent_stats["mean"])
    np.save(task_result_dir / "latent_precision.npy", latent_stats["precision"])

    config: Dict[str, object] = {**vars(args)}
    config.update(
        {
            "task": task,
            "device": str(device),
            "feature_dim": dataset.feature_dim,
            "num_windows": len(dataset),
            "num_calibration_episodes": len(dataset.episodes),
        }
    )
    save_json(config, task_result_dir / "train_config.json")

    final_loss = losses[-1] if losses else float("nan")
    summary = {
        "task": task,
        "result_dir": str(task_result_dir),
        "num_windows": len(dataset),
        "num_calibration_episodes": len(dataset.episodes),
        "final_loss": final_loss,
        "best_loss": float(min(losses)) if losses else float("nan"),
        "threshold": threshold,
        "threshold_quantile": args.threshold_quantile,
        "feature_dim": dataset.feature_dim,
        "calibration_scores_mean": float(combined_scores.mean()),
        "calibration_scores_std": float(combined_scores.std()),
        "episode_max_mean": float(episode_maxima.mean()),
        "episode_max_std": float(episode_maxima.std()),
        "num_episode_scores": int(episode_maxima.shape[0]),
        "component_stats": component_stats,
    }
    print(
        (
            "[SUMMARY] task={task} windows={windows} episodes={episodes} final_loss={final_loss:.6f} "
            "threshold={threshold:.6f} episode_max_mean={episode_mean:.6f}"
        ).format(
            task=task,
            windows=len(dataset),
            episodes=len(dataset.episodes),
            final_loss=final_loss,
            threshold=threshold,
            episode_mean=float(episode_maxima.mean()),
        )
    )
    return summary


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    requested_tasks: Sequence[str]
    if args.tasks:
        requested_tasks = args.tasks
    else:
        requested_tasks = discover_tasks(args.data_root)
    if not requested_tasks:
        raise ValueError("No tasks found for training.")

    args.result_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict[str, object]] = []
    for task in requested_tasks:
        try:
            summary = train_single_task(task, args, device)
            summaries.append(summary)
        except ValueError as exc:
            print(f"[WARN] Skipping task '{task}': {exc}")

    if not summaries:
        raise RuntimeError("Training failed for all requested tasks.")

    save_json({"tasks": summaries}, args.result_dir / "train_summary.json")


if __name__ == "__main__":
    main()
