"""Model definitions for SPL stability lattice."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class SinusoidalPosition(nn.Module):
    """Standard sine/cosine positional encoding for transformer inputs."""

    def __init__(self, dim: int, max_len: int = 2048) -> None:
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.size(1)
        return x + self.pe[:seq_len].unsqueeze(0)


@dataclass
class SPLForwardOutput:
    reconstruction: Tensor
    latent_summary: Tensor
    encoded_sequence: Tensor


class SPLNet(nn.Module):
    """Symplectic-inspired stability autoencoder for robot rollouts."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.positional = SinusoidalPosition(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.reconstruction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, input_dim),
        )
        self.latent_head = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, inputs: Tensor) -> SPLForwardOutput:
        x = self.input_proj(inputs)
        x = self.positional(x)
        encoded = self.encoder(x)
        reconstruction = self.reconstruction_head(encoded)
        latent_tokens = self.latent_head(encoded)
        latent_summary = latent_tokens.mean(dim=1)
        return SPLForwardOutput(reconstruction=reconstruction, latent_summary=latent_summary, encoded_sequence=encoded)

    def reconstruction_score(self, inputs: Tensor) -> Tensor:
        with torch.no_grad():
            outputs = self.forward(inputs)
            recon_error = F.mse_loss(outputs.reconstruction, inputs, reduction="none").mean(dim=(1, 2))
            if outputs.encoded_sequence.size(1) > 1:
                dynamics = outputs.encoded_sequence[:, 1:, :] - outputs.encoded_sequence[:, :-1, :]
                dynamics_penalty = torch.mean(torch.square(dynamics), dim=(1, 2))
            else:
                dynamics_penalty = torch.zeros(outputs.reconstruction.size(0), device=inputs.device, dtype=inputs.dtype)
            latent_norm = torch.mean(torch.square(outputs.latent_summary), dim=1)
            return recon_error + 0.1 * dynamics_penalty + 0.01 * latent_norm
