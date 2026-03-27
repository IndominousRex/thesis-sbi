"""
NPE (Neural Posterior Estimation) method implementation.

Uses normalizing flows (MAF) to directly learn the posterior density.
"""

import time
from pathlib import Path
from typing import Any, Dict
import pickle

import torch
from sbi.neural_nets import posterior_nn
from sbi import inference as sbi_inference

from .base import BaseMethod
from models.models import build_embedding


class NPEMethod(BaseMethod):
    """
    Neural Posterior Estimation using Masked Autoregressive Flows.

    This method learns the posterior density directly using normalizing flows,
    conditioned on summary statistics from an embedding network.
    """

    name = "npe"
    model_filename = "density_estimator.pt"

    def __init__(self, cfg, prior, device: torch.device):
        super().__init__(cfg, prior, device)
        self.embedding_net = None
        self.density_estimator = None
        self._training_summary = {}

    def build(self, input_dim: int, seq_len: int) -> None:
        """Build NPE inference with embedding network and MAF flow."""
        # Build embedding network
        self.embedding_net = build_embedding(self.cfg, input_dim, seq_len, self.device)

        # Build MAF density estimator
        density_estimator_builder = posterior_nn(
            model="maf",
            embedding_net=self.embedding_net,
            hidden_features=self.cfg.maf_hidden_features,
            num_transforms=self.cfg.maf_num_transforms,
            z_score_x="none",  # Manual normalization
            z_score_theta="none",
        )

        # NPE inference object
        self.inference = sbi_inference.NPE(
            prior=self.prior,
            density_estimator=density_estimator_builder,
            device=self.device,
        )

    def train(
        self,
        theta_train: torch.Tensor,
        x_train: torch.Tensor,
    ) -> Dict[str, Any]:
        """Train NPE on the provided data."""
        if self.inference is None:
            raise RuntimeError("Method not built. Call build() first.")

        # Append simulations
        self.inference.append_simulations(theta_train, x_train)

        # Train
        train_start = time.time()
        original_get_dataloaders = self._install_fixed_epoch_full_data_loaders()
        try:
            self.model = self.inference.train(
                learning_rate=self.cfg.learning_rate,
                training_batch_size=self.cfg.training_batch_size,
                validation_fraction=self.cfg.validation_fraction,
                stop_after_epochs=self.cfg.num_epochs + 1,
                max_num_epochs=max(0, self.cfg.num_epochs - 1),
                clip_max_norm=self.cfg.clip_max_norm,
                show_train_summary=True,
                use_combined_loss=False,
            )
        finally:
            if original_get_dataloaders is not None:
                self.inference.get_dataloaders = original_get_dataloaders
        train_time = time.time() - train_start

        self.model.to(self.device).eval()

        # Store summary
        summary = self.inference.summary
        self._training_summary = {
            "train_loss": summary.get("training_loss", []),
            "train_time_s": train_time,
            "val_loss": [],
            "best_val_loss": None,
            "epochs_trained": self.cfg.num_epochs,
            "fixed_epoch_schedule": True,
        }

        return self._training_summary

    def build_posterior(self) -> Any:
        """Build posterior from trained density estimator."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        self.posterior = self.inference.build_posterior(self.model)
        return self.posterior

    def save(self, exp_dir: Path) -> Path:
        """Save density estimator weights and pickled posterior."""
        if self.model is None:
            raise RuntimeError("No model to save.")

        # Save model weights
        model_path = exp_dir / self.model_filename
        torch.save(self.model.state_dict(), model_path)

        # Save pickled posterior for faster loading
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
                print(f"[NPE] Loaded pickled posterior from {posterior_path}")
                return
            except Exception as e:
                print(f"[NPE] Failed to load pickled posterior: {e}")

        # Fallback: load weights
        model_path = exp_dir / self.model_filename
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if self.model is None:
            raise RuntimeError("Must call build() before load() when loading weights")

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device).eval()
        self.posterior = self.inference.build_posterior(self.model)
        print(f"[NPE] Loaded model weights from {model_path}")

    @property
    def supports_lc2st(self) -> bool:
        """NPE with MAF supports LC2ST-NF diagnostics."""
        return True

    def get_flow_transform(self):
        """Get flow transform for LC2ST-NF diagnostics."""
        if self.model is None or not hasattr(self.model, "net"):
            return None, None, None

        flow_transform = self.model.net._transform
        flow_embed = self.model.net._embedding_net

        d_theta = len(self.cfg.active_parameters)
        flow_base_dist = torch.distributions.MultivariateNormal(
            torch.zeros(d_theta, device=self.device),
            torch.eye(d_theta, device=self.device),
        )

        return flow_transform, flow_base_dist, flow_embed
