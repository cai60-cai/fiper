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


def collect_calibration_components(
    model: SPLNet,
    loader: DataLoader,
    device: torch.device,
    task_name: str,
) -> Dict[str, np.ndarray]:
    model.eval()
    comp_buffers: Dict[str, List[np.ndarray]] = {
        "reconstruction": [],
        "dynamics": [],
        "latent_energy": [],
        "latent_summary": [],
    }
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{task_name}:calibrate", leave=False):
            batch = batch.to(device)
            components = model.score_components(batch)
            comp_buffers["reconstruction"].append(components["reconstruction"].cpu().numpy())
            comp_buffers["dynamics"].append(components["dynamics"].cpu().numpy())
            comp_buffers["latent_energy"].append(components["latent_energy"].cpu().numpy())
            comp_buffers["latent_summary"].append(components["latent_summary"].cpu().numpy())

    aggregated = {key: np.concatenate(value, axis=0) for key, value in comp_buffers.items()}
    return aggregated


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
    loader: DataLoader,
    device: torch.device,
    task_name: str,
) -> tuple[np.ndarray, Dict[str, Dict[str, float]], Dict[str, np.ndarray]]:
    raw_components = collect_calibration_components(model, loader, device, task_name)
    latent_vectors = raw_components.pop("latent_summary")
    latent_stats = compute_latent_statistics(latent_vectors)
    mahal = compute_mahalanobis(latent_vectors, latent_stats["mean"], latent_stats["precision"])
    raw_components["mahalanobis"] = mahal
    combined_scores, component_stats = combine_component_scores(raw_components)
    return combined_scores, component_stats, latent_stats


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
    calibration_scores, component_stats, latent_stats = prepare_calibration_profile(model, calib_loader, device, task)
    threshold = float(np.quantile(calibration_scores, args.threshold_quantile))
    save_json(
        {
            "threshold": threshold,
            "quantile": args.threshold_quantile,
            "scores_mean": float(calibration_scores.mean()),
            "scores_std": float(calibration_scores.std()),
            "num_scores": int(calibration_scores.shape[0]),
            "component_weights": COMPONENT_WEIGHTS,
        },
        task_result_dir / "threshold.json",
    )
    save_json(
        {
            "components": component_stats,
            "num_windows": int(calibration_scores.shape[0]),
            "latent_dim": int(latent_stats["mean"].shape[0]),
            "ridge": float(latent_stats["ridge"]),
        },
        task_result_dir / "calibration_components.json",
    )
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
        "calibration_scores_mean": float(calibration_scores.mean()),
        "calibration_scores_std": float(calibration_scores.std()),
        "component_stats": component_stats,
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
