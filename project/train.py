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
from .utils import NormalizationStats, discover_tasks, save_json


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


def evaluate_training_error(
    model: SPLNet,
    loader: DataLoader,
    device: torch.device,
    task_name: str,
) -> np.ndarray:
    model.eval()
    scores: List[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{task_name}:calibrate", leave=False):
            batch = batch.to(device)
            score = model.reconstruction_score(batch)
            scores.append(score.cpu().numpy())
    return np.concatenate(scores, axis=0)


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

    calib_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    calibration_scores = evaluate_training_error(model, calib_loader, device, task)
    threshold = float(np.quantile(calibration_scores, args.threshold_quantile))
    save_json(
        {
            "threshold": threshold,
            "quantile": args.threshold_quantile,
            "scores_mean": float(calibration_scores.mean()),
            "scores_std": float(calibration_scores.std()),
            "num_scores": int(calibration_scores.shape[0]),
        },
        task_result_dir / "threshold.json",
    )

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
        "calibration_scores_mean": float(calibration_scores.mean()),
        "calibration_scores_std": float(calibration_scores.std()),
    }
    print(
        "[SUMMARY] task={task} windows={windows} episodes={episodes} final_loss={final_loss:.6f} threshold={threshold:.6f}".format(
            task=task,
            windows=len(dataset),
            episodes=len(dataset.episodes),
            final_loss=final_loss,
            threshold=threshold,
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
