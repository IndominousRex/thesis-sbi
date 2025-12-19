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
from models.models import build_embedding


class EmbeddingWrapper(torch.nn.Module):
    """Wrapper to apply embedding and cache embedded data for NPSE."""

    def __init__(self, embedding_net: torch.nn.Module):
        super().__init__()
        self.embedding_net = embedding_net
        self._output_dim = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, features) or (batch, seq_len * features)
        embedded = self.embedding_net(x)
        self._output_dim = embedded.shape[-1]
        return embedded

    @property
    def output_dim(self) -> int:
        return self._output_dim


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
        self.embedding_net = None
        self._training_summary = {}

    def build(self, input_dim: int, seq_len: int) -> None:
        """Build NPSE inference object with embedding network."""
        # Store dimensions for later
        self._input_dim = input_dim
        self._seq_len = seq_len

        # Build embedding network (same as NPE)
        base_embedding = build_embedding(self.cfg, input_dim, seq_len, self.device)
        self.embedding_net = EmbeddingWrapper(base_embedding)

        # NPSE inference object - pass embedding_net directly
        # The sbi library will use it to embed x before the score network
        self.inference = NPSE(
            prior=self.prior,
            sde_type=self.sde_type,
            embedding_net=self.embedding_net,
            device=str(self.device),
        )

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
        """Save model weights only (NPSE inference/posterior can't be pickled)."""
        if self.model is None:
            raise RuntimeError("No model to save.")

        # Save model weights only - inference object has unpicklable lambdas
        model_path = exp_dir / self.model_filename
        torch.save(self.model.state_dict(), model_path)
        print(f"[NPSE] Saved model weights to {model_path}")

        # Save architecture info for rebuilding
        arch_info = {
            "input_dim": self._input_dim,
            "seq_len": self._seq_len,
            "sde_type": self.sde_type,
        }
        arch_path = exp_dir / "npse_arch.pkl"
        with open(arch_path, "wb") as f:
            pickle.dump(arch_info, f)

        return model_path

    def load(self, exp_dir: Path) -> None:
        """Load model weights and rebuild inference/posterior."""
        model_path = exp_dir / self.model_filename
        arch_path = exp_dir / "npse_arch.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load architecture info
        if arch_path.exists():
            with open(arch_path, "rb") as f:
                arch_info = pickle.load(f)
            self._input_dim = arch_info["input_dim"]
            self._seq_len = arch_info["seq_len"]
            self.sde_type = arch_info["sde_type"]
        else:
            raise FileNotFoundError(
                f"Architecture info not found: {arch_path}. "
                "Cannot rebuild NPSE model without dimensions."
            )

        # Rebuild inference object
        self.build(self._input_dim, self._seq_len)

        # Load weights into the neural net
        # We need to append dummy data to initialize the neural net first
        dummy_theta = self.prior.sample((2,))
        dummy_x = torch.randn(2, self._seq_len, self._input_dim)
        self.inference.append_simulations(dummy_theta, dummy_x)

        # Now load the saved weights
        self.model = self.inference._neural_net
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        print(f"[NPSE] Loaded model weights from {model_path}")

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
