import os
from typing import Tuple

import torch
import jax
import numpy as np


def setup_environment(seed: int = 42) -> None:
    """
    Set environment variables and global random seeds for reproducibility.
    Also configures JAX and PyTorch behaviour (e.g. matmul precision).
    """
    # JAX & HPC-related env vars
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")  # Force JAX to use CPU
    os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("SLURM_CPUS_PER_TASK", "4"))

    # PyTorch numeric behaviour
    torch.set_float32_matmul_precision("high")

    # Seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(preference: str = "cuda") -> torch.device:
    """
    Decide whether to use CPU or CUDA.

    preference:
        - "auto": use CUDA if available, else CPU
        - "cuda": force CUDA, error if not available
        - "cpu" : force CPU
    """
    if preference == "cpu":
        device = torch.device("cpu")
    elif preference == "cuda":
        assert torch.cuda.is_available(), "CUDA requested but not available."
        device = torch.device("cuda")
    else:  # auto
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        f"Torch device: {device} | CUDA available: {torch.cuda.is_available()} "
        f"CUDA version: {torch.version.cuda}"
    )
    print("JAX backend:", jax.default_backend())

    if device.type == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory
        try:
            free, total2 = torch.cuda.mem_get_info()
        except AttributeError:
            free = total - torch.cuda.memory_reserved(0)
        print(
            f"Total VRAM: {total/1024**2:.2f} MB | Free (driver): {free/1024**2:.2f} MB"
        )
    else:
        print("No GPU available; running on CPU.")

    return device
