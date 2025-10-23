"""Training script for the SPLNet stability model."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

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
from .utils import NormalizationStats, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SPLNet on successful calibration rollouts.")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Root directory containing task rollouts.")
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


def build_dataloader(args: argparse.Namespace) -> SuccessWindowDataset:
    dataset = SuccessWindowDataset(
        root=args.data_root,
        window_size=args.window_size,
        stride=args.stride,
    )
    return dataset


def train_epoch(model: SPLNet, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
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


def evaluate_training_error(model: SPLNet, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    scores: List[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="calibrate", leave=False):
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


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = build_dataloader(args)
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
        loss = train_epoch(model, loader, optimizer, device)
        losses.append(loss)
        print(f"Epoch {epoch:03d} | loss={loss:.6f}")

    args.result_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.result_dir / "splnet.pt")

    stats: NormalizationStats = dataset.stats
    save_json({"mean": stats.mean.tolist(), "std": stats.std.tolist()}, args.result_dir / "normalization_stats.json")

    plot_losses(losses, args.result_dir / "train_loss.png")
    save_json({"losses": losses}, args.result_dir / "train_loss.json")

    calib_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    calibration_scores = evaluate_training_error(model, calib_loader, device)
    threshold = float(np.quantile(calibration_scores, args.threshold_quantile))
    save_json(
        {
            "threshold": threshold,
            "quantile": args.threshold_quantile,
            "scores_mean": float(calibration_scores.mean()),
            "scores_std": float(calibration_scores.std()),
        },
        args.result_dir / "threshold.json",
    )

    config = vars(args)
    config["device"] = str(device)
    config["feature_dim"] = dataset.feature_dim
    save_json(config, args.result_dir / "train_config.json")


if __name__ == "__main__":
    main()
