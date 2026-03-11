"""
Methods package for SBI experiments.

Provides unified interfaces for different inference methods:
- NPE: Neural Posterior Estimation (normalizing flows)
- NPSE: Neural Posterior Score Estimation (diffusion/score-based)
- FNPE: Flow-based NPE using MarkovSBI (JAX-based)
"""

from .base import BaseMethod, TrainedMethod
from .npe_method import NPEMethod
from .npse_method import NPSEMethod
from .fnpe_method import FNPEMethod
from .simformer_method import SimformerMethod


AVAILABLE_METHODS = {
    "npe": NPEMethod,
    "npse": NPSEMethod,
    "fnpe": FNPEMethod,
    "simformer": SimformerMethod,
}


def get_method_class(method_name: str) -> type:
    """
    Get method class by name.

    Args:
        method_name: One of 'npe', 'npse', 'fnpe'

    Returns:
        Method class
    """
    method_name = method_name.lower()
    if method_name not in AVAILABLE_METHODS:
        raise ValueError(
            f"Unknown method '{method_name}'. "
            f"Available: {list(AVAILABLE_METHODS.keys())}"
        )
    return AVAILABLE_METHODS[method_name]


def build_method(method_name: str, cfg, prior, device, **method_kwargs) -> BaseMethod:
    """
    Factory function to build a method instance.

    Args:
        method_name: One of 'npe', 'npse', 'fnpe'
        cfg: ExperimentConfig
        prior: Prior distribution
        device: Torch device
        **method_kwargs: Additional method-specific arguments

    Returns:
        Configured method instance
    """
    method_cls = get_method_class(method_name)
    return method_cls(cfg, prior, device, **method_kwargs)


__all__ = [
    "BaseMethod",
    "TrainedMethod",
    "NPEMethod",
    "NPSEMethod",
    "FNPEMethod",
    "SimformerMethod",
    "AVAILABLE_METHODS",
    "get_method_class",
    "build_method",
]
