"""
Experiment configuration for SBI methods.

Provides a unified configuration dataclass that works with all methods
(NPE, NPSE, FNPE) and supports dataset caching for fair comparisons.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import hashlib
import json


# Parameter ordering expected by the JAX vehicle model (friction, drag, mass)
PARAMETER_ORDER: Tuple[str, ...] = ("mu", "cd", "m")


@dataclass
class ExperimentConfig:
    """
    Unified configuration for all SBI experiments.

    This config supports:
    - Method selection (npe, npse, fnpe)
    - Dataset caching for reproducible comparisons
    - Separate seeds for simulation and training
    - All encoder/training hyperparameters
    """

    # --- General ---
    exp_name: str = "baseline"
    method: str = "npe"  # npe | npse | fnpe
    device: str = "cuda"  # "auto", "cpu", or "cuda"

    # --- Seeds (separate for reproducibility) ---
    random_seed: int = 42  # Legacy: will be used as sim_seed if sim_seed not set
    sim_seed: Optional[int] = None  # Seed for simulation/data generation
    train_seed: Optional[int] = None  # Seed for training randomness

    # --- Simulation settings ---
    dt: float = 0.01
    T_seg: int = 3000  # Sequence length (T)
    state_dim: int = 10
    obs_dim: int = 9

    # --- Prior (3 params: mu, C_d, m) ---
    prior_low_mu: float = 0.50
    prior_high_mu: float = 1.50
    prior_low_cd: float = 0.05
    prior_high_cd: float = 0.60
    prior_low_m: float = 1700.0
    prior_high_m: float = 2200.0

    # --- Which parameters to infer + defaults for fixed ones ---
    active_parameters: Tuple[str, ...] = PARAMETER_ORDER
    fixed_mu: float = 1.0
    fixed_cd: float = 0.27
    fixed_m: float = 1720.0

    # --- Dataset ---
    num_simulations: int = 2000
    batch_sim: int = 512
    jit_warmup: bool = True

    # --- Dataset caching (for fair method comparisons) ---
    dataset_id: Optional[str] = None  # Unique ID for cached dataset
    dataset_cache_dir: str = "datasets"
    cache_dataset: bool = True  # Whether to cache generated dataset
    reuse_dataset: bool = False  # Whether to load cached dataset if available

    # --- Encoder / embedding ---
    encoder_type: str = "bigru"  # bigru | causalcnn | transformer
    encoder_hidden: int = 32

    # ---- Transformer ----
    transformer_layers: int = 3
    transformer_heads: int = 4
    transformer_head_dim: int = 16
    transformer_feature_dim: int = 64  # Will be computed in post_init

    # ---- CausalCNN ----
    causalcnn_num_layers: int = 4
    causalcnn_kernel_size: int = 3
    causalcnn_pool_kernel: int = 64

    # ---- Common embedding ----
    embedding_output_dim: int = 64

    # --- NPE-specific: MAF flow ---
    maf_hidden_features: int = 128
    maf_num_transforms: int = 8

    # --- NPSE-specific: SDE type ---
    sde_type: str = "ve"  # ve | vp | subvp

    # --- FNPE-specific ---
    fnpe_hidden_dim: int = 128
    fnpe_num_hidden: int = 5
    fnpe_model_type: str = "gru"  # gru | linear
    fnpe_window_size: int = 2  # Markov window size (CRITICAL: keep small, e.g. 2-10)
    fnpe_steps_per_epoch: int = 10000  # Steps per epoch (like Lotka-Volterra example)
    fnpe_num_diffusion_steps: int = 500
    # Score composition method:
    # - "fnpe" (DEFAULT): Fast, uses (1-N)*prior + sum(scores) or normalized mean
    # - "gauss_corrected": Paper GAUSS method - VERY SLOW (estimates covariances per obs)
    # - "uncorrected": Uses marginal prior score
    fnpe_score_fn_type: str = "fnpe"
    # Max observation length at inference
    # With normalize_score_by_windows=True, can use longer sequences (100-500)
    # Without normalization, keep small (11) to avoid (1-N)*prior_score dominating
    fnpe_max_obs_len: int = 100
    # Use mean instead of sum over windows for numerical stability
    fnpe_normalize_score: bool = True
    # Proposal type for training data generation:
    # - "pred" (DEFAULT, CORRECT): Sample states from pilot simulation pool
    # - "naive": Sample from initial state distribution only
    # - "trajectory" (OLD, INCORRECT): Divide trajectories into pairs
    fnpe_proposal_type: str = "pred"

    # --- Training ---
    learning_rate: float = 5e-4  # Lower LR for complex data (MarkovSBI large uses 5e-4)
    training_batch_size: int = 1024  # Larger batch for stability
    validation_fraction: float = 0.15  # 15% for stable validation metrics
    stop_after_epochs: int = 50  # High patience, let LR schedule do its work
    clip_max_norm: float = (
        20.0  # Higher clip for complex data (MarkovSBI large uses 20)
    )
    num_epochs: int = 200  # More epochs for complex data (MarkovSBI large uses 100)

    # --- SBC / diagnostics ---
    num_sbc_samples: int = 200
    num_posterior_samples_sbc: int = 1000
    num_calibration_items: int = 5
    num_lc2st_samples: Optional[int] = None  # Computed in post_init
    num_swd_projections: int = 1000

    # --- Diagnostic options ---
    run_sbc: bool = True
    run_lc2st: bool = True  # Only for NPE
    run_swd: bool = True
    run_one_step_rmse: bool = True
    run_posterior_plots: bool = True

    # --- Real data eval ---
    # Default path to real measurement data for evaluation
    real_data_csv: Optional[str] = "../data/measurements/Jeversen_2022_10_12_110132.csv"

    # --- Logging / saving ---
    results_root: str = "experiments"
    no_plots: bool = False

    # --- Execution mode ---
    do_train: bool = True
    do_eval: bool = True
    checkpoint: Optional[str] = None  # Path to load checkpoint from

    def __post_init__(self) -> None:
        """
        Normalize and validate configuration.
        """
        # --- Normalize active_parameters ---
        cleaned = []
        for name in self.active_parameters:
            n = str(name).strip().lower()
            if not n:
                continue
            if n not in PARAMETER_ORDER:
                raise ValueError(
                    f"Unknown parameter '{name}'. Valid options: {PARAMETER_ORDER}"
                )
            if n not in cleaned:
                cleaned.append(n)
        self.active_parameters = tuple(cleaned) if cleaned else PARAMETER_ORDER

        # --- Fix derived defaults ---
        if self.num_lc2st_samples is None:
            self.num_lc2st_samples = max(100, int(0.1 * self.num_simulations))

        if self.transformer_feature_dim == 64:
            self.transformer_feature_dim = (
                self.transformer_heads * self.transformer_head_dim
            )

        # --- Handle seed inheritance ---
        if self.sim_seed is None:
            self.sim_seed = self.random_seed
        if self.train_seed is None:
            self.train_seed = self.random_seed

        # --- Validate method ---
        valid_methods = {"npe", "npse", "fnpe"}
        if self.method.lower() not in valid_methods:
            raise ValueError(
                f"Unknown method '{self.method}'. Valid options: {valid_methods}"
            )
        self.method = self.method.lower()

        # --- Validate sde_type ---
        valid_sde = {"ve", "vp", "subvp"}
        if self.sde_type.lower() not in valid_sde:
            raise ValueError(
                f"Unknown sde_type '{self.sde_type}'. Valid options: {valid_sde}"
            )
        self.sde_type = self.sde_type.lower()

        # --- Generate dataset_id if caching and not provided ---
        if (self.cache_dataset or self.reuse_dataset) and self.dataset_id is None:
            self.dataset_id = self._generate_dataset_id()

    def _generate_dataset_id(self) -> str:
        """Generate a unique dataset ID based on simulation parameters."""
        data_params = {
            "sim_seed": self.sim_seed,
            "num_simulations": self.num_simulations,
            "T_seg": self.T_seg,
            "dt": self.dt,
            "active_parameters": self.active_parameters,
            "prior_bounds": self.param_bounds(),
            "fixed_values": self.fixed_param_values(),
        }
        param_str = json.dumps(data_params, sort_keys=True)
        hash_val = hashlib.md5(param_str.encode()).hexdigest()[:12]
        return f"dataset_{hash_val}"

    def param_bounds(self) -> Dict[str, tuple]:
        """Bounds for each physical parameter keyed by name."""
        return {
            "mu": (self.prior_low_mu, self.prior_high_mu),
            "cd": (self.prior_low_cd, self.prior_high_cd),
            "m": (self.prior_low_m, self.prior_high_m),
        }

    def fixed_param_values(self) -> Dict[str, float]:
        """Defaults to plug in for parameters that are not inferred."""
        return {"mu": self.fixed_mu, "cd": self.fixed_cd, "m": self.fixed_m}

    def active_param_dim(self) -> int:
        """Number of parameters being inferred."""
        return len(self.active_parameters)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def get_dataset_cache_path(self) -> Path:
        """Get path for cached dataset."""
        cache_dir = Path(self.dataset_cache_dir)
        return cache_dir / f"{self.dataset_id}.pt"

    def get_experiment_name(self) -> str:
        """Generate a descriptive experiment name."""
        parts = [self.method, self.exp_name]
        if len(self.active_parameters) < 3:
            parts.append("_".join(self.active_parameters))
        return "_".join(parts)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        """Create config from dictionary."""
        if "active_parameters" in d and isinstance(d["active_parameters"], list):
            d["active_parameters"] = tuple(d["active_parameters"])
        return cls(**d)

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        """Load config from JSON file."""
        with open(path, "r") as f:
            d = json.load(f)
        return cls.from_dict(d)

    def save(self, path: str) -> None:
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
