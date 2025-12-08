from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import torch


@dataclass
class Normalizer:
    """
    Centralized mean/std normalizer for observations, controls, and parameters.

    All stats live on CPU; call .to(device) to clone on a specific device.
    """

    obs_mean: torch.Tensor  # (obs_dim,)
    obs_std: torch.Tensor   # (obs_dim,)
    ctrl_mean: torch.Tensor  # (4,)
    ctrl_std: torch.Tensor   # (4,)
    theta_mean: torch.Tensor  # (d_theta,)
    theta_std: torch.Tensor   # (d_theta,)
    eps: float = 1e-8

    def to(self, device: torch.device) -> "Normalizer":
        """
        Return a copy with tensors moved to `device`.
        """
        return Normalizer(
            obs_mean=self.obs_mean.to(device),
            obs_std=self.obs_std.to(device),
            ctrl_mean=self.ctrl_mean.to(device),
            ctrl_std=self.ctrl_std.to(device),
            theta_mean=self.theta_mean.to(device),
            theta_std=self.theta_std.to(device),
            eps=self.eps,
        )

    def normalize_x(self, x_phys: torch.Tensor, obs_dim: int) -> torch.Tensor:
        """
        Normalize concatenated [obs || controls] tensor in physical units.

        Args:
            x_phys: (N, T, D_in) tensor in physical units.
            obs_dim: number of observation channels (front slice of D_in).
        """
        obs = x_phys[..., :obs_dim]
        ctrls = x_phys[..., obs_dim:]

        obs_n = (obs - self.obs_mean) / (self.obs_std + self.eps)
        ctrl_n = (ctrls - self.ctrl_mean) / (self.ctrl_std + self.eps)

        return torch.cat([obs_n, ctrl_n], dim=-1)

    def normalize_theta(self, theta_phys: torch.Tensor) -> torch.Tensor:
        return (theta_phys - self.theta_mean) / (self.theta_std + self.eps)

    def unnormalize_theta(self, theta_norm: torch.Tensor) -> torch.Tensor:
        return theta_norm * (self.theta_std + self.eps) + self.theta_mean

    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "obs_mean": self.obs_mean.cpu().tolist(),
            "obs_std": self.obs_std.cpu().tolist(),
            "ctrl_mean": self.ctrl_mean.cpu().tolist(),
            "ctrl_std": self.ctrl_std.cpu().tolist(),
            "theta_mean": self.theta_mean.cpu().tolist(),
            "theta_std": self.theta_std.cpu().tolist(),
            "eps": self.eps,
        }

    @staticmethod
    def from_json(data: Dict[str, Any], device: torch.device | None = None) -> "Normalizer":
        dev = device if device is not None else "cpu"
        return Normalizer(
            obs_mean=torch.tensor(data["obs_mean"], dtype=torch.float32, device=dev),
            obs_std=torch.tensor(data["obs_std"], dtype=torch.float32, device=dev),
            ctrl_mean=torch.tensor(data["ctrl_mean"], dtype=torch.float32, device=dev),
            ctrl_std=torch.tensor(data["ctrl_std"], dtype=torch.float32, device=dev),
            theta_mean=torch.tensor(data["theta_mean"], dtype=torch.float32, device=dev),
            theta_std=torch.tensor(data["theta_std"], dtype=torch.float32, device=dev),
            eps=float(data.get("eps", 1e-8)),
        )


def fit_normalizer(theta_phys: torch.Tensor, x_phys: torch.Tensor, obs_dim: int, eps: float = 1e-8) -> Normalizer:
    """
    Fit mean/std from simulated training data in physical units.

    Args:
        theta_phys: (N, d) tensor of parameters.
        x_phys:     (N, T, D_in) tensor of concatenated [obs || controls].
        obs_dim:    number of observation channels (front slice of D_in).
    """
    if theta_phys.ndim != 2:
        raise ValueError(f"theta must be (N, d), got {theta_phys.shape}")
    if x_phys.ndim != 3:
        raise ValueError(f"x must be (N, T, D), got {x_phys.shape}")

    obs = x_phys[..., :obs_dim]  # (N, T, obs_dim)
    ctrls = x_phys[..., obs_dim:]  # (N, T, 4)

    obs_mean = obs.mean(dim=(0, 1))
    obs_std = obs.std(dim=(0, 1), unbiased=False).clamp_min(eps)

    ctrl_mean = ctrls.mean(dim=(0, 1))
    ctrl_std = ctrls.std(dim=(0, 1), unbiased=False).clamp_min(eps)

    theta_mean = theta_phys.mean(dim=0)
    theta_std = theta_phys.std(dim=0, unbiased=False).clamp_min(eps)

    return Normalizer(
        obs_mean=obs_mean,
        obs_std=obs_std,
        ctrl_mean=ctrl_mean,
        ctrl_std=ctrl_std,
        theta_mean=theta_mean,
        theta_std=theta_std,
        eps=eps,
    )


def save_normalizer(normalizer: Normalizer, path: Path) -> None:
    path = Path(path)
    with path.open("w") as f:
        json.dump(normalizer.to_jsonable(), f, indent=2)


def load_normalizer(path: Path, device: torch.device | None = None) -> Normalizer:
    path = Path(path)
    with path.open("r") as f:
        data = json.load(f)
    return Normalizer.from_json(data, device=device)
