"""Training script for the SPLNet stability model."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Configure matplotlib to work in headless environments.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import SuccessWindowDataset
from model import SPLNet
from utils import NormalizationStats, discover_tasks, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SPLNet on successful calibration rollouts.")
    parser.add_argument("--data-root", type=Path, default=Path("../data"), help="Root directory containing task rollouts.")
    parser.add_argument("--tasks", type=str, nargs="*", default=None, help="Specific task names to train; defaults to all tasks found under data root.")
    parser.add_argument("--window-size", type=int, default=32, help="Sliding window length in steps.")
    parser.add_argument("--stride", type=int, default=4, help="Stride for sliding windows.")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size.")
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


def calibrate_task(
    model: SPLNet,
    dataset: SuccessWindowDataset,
    device: torch.device,
    task_name: str,
    batch_size: int,
) -> Dict[str, object]:
    """Compute reconstruction-based calibration scores for a trained model."""

    model.eval()
    all_scores: List[np.ndarray] = []
    episode_maxima: List[float] = []
    episode_means: List[float] = []
    details: List[Dict[str, object]] = []

    for info, windows in dataset.iter_episode_windows():
        if not windows:
            continue
        scores_chunks: List[np.ndarray] = []
        for start in range(0, len(windows), batch_size):
            chunk = windows[start : start + batch_size]
            batch = torch.from_numpy(np.stack(chunk, axis=0)).to(device)
            with torch.no_grad():
                scores = model.reconstruction_score(batch)
            scores_chunks.append(scores.cpu().numpy())
        episode_scores = np.concatenate(scores_chunks, axis=0)
        all_scores.append(episode_scores)
        max_score = float(np.max(episode_scores))
        mean_score = float(np.mean(episode_scores))
        episode_maxima.append(max_score)
        episode_means.append(mean_score)
        details.append(
            {
                "task": info.task,
                "episode": info.path.stem,
                "successful": True,
                "num_windows": int(episode_scores.shape[0]),
                "max_score": max_score,
                "mean_score": mean_score,
                "aggregation": "max",
            }
        )

    if not all_scores:
        raise ValueError("Calibration produced no windows for threshold estimation.")

    combined_scores = np.concatenate(all_scores, axis=0)
    maxima_array = np.asarray(episode_maxima, dtype=np.float32)
    means_array = np.asarray(episode_means, dtype=np.float32)

    return {
        "combined_scores": combined_scores,
        "episode_maxima": maxima_array,
        "episode_means": means_array,
        "details": details,
    }


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

    calibration = calibrate_task(model, dataset, device, task, args.batch_size)
    combined_scores: np.ndarray = calibration["combined_scores"]
    episode_maxima: np.ndarray = calibration["episode_maxima"]
    episode_means: np.ndarray = calibration["episode_means"]
    details: List[Dict[str, object]] = calibration["details"]
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
            "aggregation": "max",
            "calibration_target": "episode_max",
        },
        task_result_dir / "threshold.json",
    )
    save_json(
        {
            "num_windows": int(combined_scores.shape[0]),
            "num_episode_scores": int(episode_maxima.shape[0]),
            "episode_max_mean": float(episode_maxima.mean()),
            "episode_max_std": float(episode_maxima.std()),
            "episode_mean_mean": float(episode_means.mean()),
            "episode_mean_std": float(episode_means.std()),
        },
        task_result_dir / "calibration_summary.json",
    )
    save_json({"episodes": details, "aggregation": "max"}, task_result_dir / "calibration_details.json")

    config: Dict[str, object] = {**vars(args)}
    for key, value in config.items():
        if isinstance(value, Path):
            config[key] = str(value)
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
