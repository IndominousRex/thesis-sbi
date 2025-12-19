"""
NPSE (Neural Posterior Score Estimation) method implementation.

Uses score-based diffusion models to learn the posterior.
"""

import time
from pathlib import Path
from typing import Any, Dict
import pickle

import torch
from sbi.inference import NPSE

from .base import BaseMethod


class NPSEMethod(BaseMethod):
    """
    Neural Posterior Score Estimation using diffusion models.

    This method learns the score function (gradient of log-density) of the
    posterior using denoising score matching, then samples via diffusion.

    Supports different SDE types:
    - 've': Variance Exploding (SMLD) - recommended
    - 'vp': Variance Preserving (DDPM)
    - 'subvp': sub-Variance Preserving
    """

    name = "npse"
    model_filename = "score_estimator.pt"

    def __init__(
        self,
        cfg,
        prior,
        device: torch.device,
        sde_type: str = "ve",
    ):
        super().__init__(cfg, prior, device)
        self.sde_type = sde_type
        self.score_estimator = None
        self._training_summary = {}

    def build(self, input_dim: int, seq_len: int) -> None:
        """Build NPSE inference object."""
        # NPSE handles its own network architecture
        self.inference = NPSE(
            prior=self.prior,
            sde_type=self.sde_type,
            device=str(self.device),
        )

        # Store dimensions for later
        self._input_dim = input_dim
        self._seq_len = seq_len

    def train(
        self,
        theta_train: torch.Tensor,
        x_train: torch.Tensor,
    ) -> Dict[str, Any]:
        """Train NPSE score estimator on the provided data."""
        if self.inference is None:
            raise RuntimeError("Method not built. Call build() first.")

        # Append simulations
        self.inference.append_simulations(theta_train, x_train)

        # Train
        train_start = time.time()
        self.model = self.inference.train(
            learning_rate=self.cfg.learning_rate,
            training_batch_size=self.cfg.training_batch_size,
            validation_fraction=self.cfg.validation_fraction,
            stop_after_epochs=self.cfg.stop_after_epochs,
            show_train_summary=True,
        )
        train_time = time.time() - train_start

        self.score_estimator = self.model

        # Store summary
        summary = self.inference.summary
        self._training_summary = {
            "train_loss": summary.get("training_loss", []),
            "val_loss": summary.get("validation_loss", []),
            "train_time_s": train_time,
            "sde_type": self.sde_type,
        }

        return self._training_summary

    def build_posterior(self) -> Any:
        """Build posterior from trained score estimator."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        self.posterior = self.inference.build_posterior(self.model)
        return self.posterior

    def save(self, exp_dir: Path) -> Path:
        """Save score estimator weights and pickled posterior."""
        if self.model is None:
            raise RuntimeError("No model to save.")

        # Save model weights
        model_path = exp_dir / self.model_filename
        torch.save(self.model.state_dict(), model_path)

        # Save pickled posterior
        if self.posterior is not None:
            with open(exp_dir / "posterior.pkl", "wb") as f:
                pickle.dump(self.posterior, f)

        return model_path

    def load(self, exp_dir: Path) -> None:
        """Load model from disk."""
        # Try pickled posterior first
        posterior_path = exp_dir / "posterior.pkl"
        if posterior_path.exists():
            try:
                with open(posterior_path, "rb") as f:
                    self.posterior = pickle.load(f)
                if hasattr(self.posterior, "to"):
                    self.posterior.to(self.device)
                print(f"[NPSE] Loaded pickled posterior from {posterior_path}")
                return
            except Exception as e:
                print(f"[NPSE] Failed to load pickled posterior: {e}")

        # Fallback: load weights (requires rebuild)
        model_path = exp_dir / self.model_filename
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # For NPSE, we need to rebuild the inference and load weights
        # This is a limitation - better to use pickled posterior
        raise NotImplementedError(
            "NPSE weight-only loading not fully supported. "
            "Please ensure posterior.pkl is saved during training."
        )

    @property
    def supports_lc2st(self) -> bool:
        """NPSE does not support LC2ST-NF (no flow transform)."""
        return False

    def sample_posterior(
        self, x_obs: torch.Tensor, num_samples: int, **kwargs
    ) -> torch.Tensor:
        """
        Sample from NPSE posterior.

        Note: NPSE sampling is slower than NPE due to diffusion process.
        Consider using fewer samples for diagnostics.
        """
        return super().sample_posterior(x_obs, num_samples, **kwargs)
