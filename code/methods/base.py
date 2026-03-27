"""
Base class for SBI methods.

This module defines the abstract interface that all methods (NPE, NPSE, FNPE)
must implement to ensure fair and consistent experiments.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any, Dict, Optional, Tuple
import torch
from torch.utils import data


@dataclass
class TrainedMethod:
    """Container for a trained method's artifacts."""

    method_name: str
    model: Any  # The trained model (density_estimator, score_estimator, params, etc.)
    posterior: Any  # Callable posterior object for sampling
    training_summary: Dict[str, Any]  # Training losses, time, etc.
    model_path: Optional[Path] = None  # Path to saved model weights


class BaseMethod(ABC):
    """
    Abstract base class for SBI inference methods.

    All methods must implement:
    - build(): Set up the inference object
    - train(): Train the model on data
    - build_posterior(): Build a samplable posterior from trained model
    - save(): Save model to disk
    - load(): Load model from disk
    """

    name: str = "base"

    def __init__(self, cfg, prior, device: torch.device):
        """
        Initialize the method.

        Args:
            cfg: ExperimentConfig object
            prior: Prior distribution (normalized)
            device: Torch device
        """
        self.cfg = cfg
        self.prior = prior
        self.device = device
        self.inference = None
        self.model = None
        self.posterior = None

    @abstractmethod
    def build(self, input_dim: int, seq_len: int) -> None:
        """
        Build the inference object and any required neural networks.

        Args:
            input_dim: Observation input dimension (D_in)
            seq_len: Sequence length (T)
        """
        pass

    @abstractmethod
    def train(
        self,
        theta_train: torch.Tensor,
        x_train: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Train the method on the provided data.

        Args:
            theta_train: Training parameters, shape (N, d_theta)
            x_train: Training observations, shape (N, T, D_in)

        Returns:
            Training summary dict with losses and timing info
        """
        pass

    @abstractmethod
    def build_posterior(self) -> Any:
        """
        Build a posterior object that can be sampled from.

        Returns:
            Posterior object with .sample(shape, x=x_obs) method
        """
        pass

    @abstractmethod
    def save(self, exp_dir: Path) -> Path:
        """
        Save model weights and any required artifacts.

        Args:
            exp_dir: Experiment directory

        Returns:
            Path to saved model file
        """
        pass

    @abstractmethod
    def load(self, exp_dir: Path) -> None:
        """
        Load model weights from disk.

        Args:
            exp_dir: Experiment directory containing saved model
        """
        pass

    def get_trained_method(self, exp_dir: Path) -> TrainedMethod:
        """
        Get a TrainedMethod container with all artifacts.

        Args:
            exp_dir: Experiment directory

        Returns:
            TrainedMethod dataclass
        """
        return TrainedMethod(
            method_name=self.name,
            model=self.model,
            posterior=self.posterior,
            training_summary=getattr(self, "_training_summary", {}),
            model_path=exp_dir / self.model_filename,
        )

    @property
    @abstractmethod
    def model_filename(self) -> str:
        """Return the filename used for saving the model."""
        pass

    @property
    def supports_lc2st(self) -> bool:
        """Whether this method supports LC2ST-NF diagnostics (requires flow)."""
        return False

    def sample_posterior(
        self, x_obs: torch.Tensor, num_samples: int, **kwargs
    ) -> torch.Tensor:
        """
        Sample from the posterior given observations.

        Args:
            x_obs: Observation tensor
            num_samples: Number of samples to draw
            **kwargs: Additional method-specific arguments

        Returns:
            Posterior samples, shape (num_samples, d_theta)
        """
        if self.posterior is None:
            raise RuntimeError("Posterior not built. Call build_posterior() first.")

        with torch.no_grad():
            samples = self.posterior.sample((num_samples,), x=x_obs, **kwargs)

        return samples

    def _install_fixed_epoch_full_data_loaders(self):
        """
        Patch sbi trainers to monitor on the full dataset instead of creating a
        held-out validation split. This keeps all simulations in the training set
        while still satisfying the trainer's expected train/val loader interface.
        """
        if self.inference is None or not hasattr(self.inference, "get_simulations"):
            return None

        original_get_dataloaders = self.inference.get_dataloaders

        def _get_dataloaders(
            inference_self,
            starting_round: int = 0,
            training_batch_size: int = 200,
            validation_fraction: float = 0.1,
            resume_training: bool = False,
            dataloader_kwargs: Optional[dict] = None,
        ):
            theta, x, prior_masks = inference_self.get_simulations(starting_round)
            dataset = data.TensorDataset(theta, x, prior_masks)
            num_examples = int(theta.size(0))
            batch_size = max(1, min(int(training_batch_size), num_examples))
            base_kwargs = {
                "batch_size": batch_size,
                "drop_last": True,
            }
            if dataloader_kwargs is not None:
                base_kwargs = dict(base_kwargs, **dataloader_kwargs)

            train_loader = data.DataLoader(dataset, shuffle=True, **base_kwargs)
            val_loader = data.DataLoader(dataset, shuffle=False, **base_kwargs)

            full_indices = torch.arange(num_examples)
            inference_self.train_indices = full_indices
            inference_self.val_indices = full_indices
            return train_loader, val_loader

        self.inference.get_dataloaders = MethodType(
            _get_dataloaders, self.inference
        )
        return original_get_dataloaders
