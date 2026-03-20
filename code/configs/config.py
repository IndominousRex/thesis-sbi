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
    steer_scale: float = 0.5  # Scale factor applied to steering during data creation
    init_speed_center_ms: float = 11.0  # Target initial v_body_x (m/s)
    init_speed_range_ms: float = 10.0  # Sample +/- range around center (m/s)
    brake_block_fraction: float = 0.5  # Relative number of brake blocks vs default
    accel_scale: float = 1.2  # Scale factor for accel blocks during data creation
    emergency_brake_fraction: float = (
        0.4  # Fraction of brake blocks using aggressive types
    )
    ramp_s: float = 0.1  # Ramp duration between blocks (s); 0.1 = snappier transitions

    # --- Sim-to-real noise injection ---
    # Calibrated from 7 real measurement CSVs (see simulation/noise.py).
    # Scales are global multipliers: 0.0 = off, 1.0 = calibrated level.
    obs_noise_scale: float = 1.0  # Observation (sensor) noise scale
    process_noise_scale: float = 1.0  # Process (dynamics drift) noise scale

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
    test_dataset_id: Optional[str] = None  # Unique ID for cached held-out test dataset
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
    fnpe_num_simulations: int = 100000  # Simulation budget for FNPE runs
    fnpe_max_epochs: int = 200  # Hard cap for FNPE epochs
    fnpe_budget_epoch_multiplier: float = 8.0  # Scale budget-derived epochs; 20k+ sims hit the 200-epoch cap
    fnpe_stop_after_epochs: int = 20  # Match sbi NPSE default patience
    fnpe_ema_loss_decay: float = 0.1  # Match sbi NPSE default EMA decay
    fnpe_convergence_std_threshold: float = 2.0  # Match sbi NPSE convergence threshold
    # Score composition method:
    # - "gauss_corrected" (DEFAULT): Paper GAUSS method - accurate but slow at inference
    # - "fnpe": Fast, uses (1-N)*prior + sum(scores)
    # - "uncorrected": Uses marginal prior score
    fnpe_score_fn_type: str = "gauss_corrected"
    # Proposal type for training data generation:
    # - "pred" (DEFAULT, CORRECT): Sample states from pilot simulation pool
    # - "naive": Sample from initial state distribution only
    # - "trajectory" (OLD, INCORRECT): Divide trajectories into pairs
    fnpe_proposal_type: str = "pred"
    # SDE T_min: minimum diffusion time. Higher = more stable but less precise.
    # Default 0.05 (was 0.01, which caused score explosion near t→0).
    fnpe_t_min: float = 0.05
    # Debugging: skip internal FNPE normalization of thetas and observations.
    # When True, raw (physical-unit) data is fed directly to the score network.
    # Use to diagnose whether normalization is causing posterior issues.
    fnpe_skip_normalize: bool = False
    # Clip diffusion samples to prior bounds during reverse sampling.
    # Prevents physically impossible values (e.g., negative mu).
    fnpe_clip_samples: bool = True
    # Proposal hyperparameters (only used when proposal_type="pred")
    fnpe_pilot_fraction: float = 0.02  # Fraction of num_simulations for pilot sims (2%)
    fnpe_pilot_length: int = 1500  # Length of each pilot trajectory
    fnpe_proposal_noise: float = (
        0.03  # Noise scale: noise_scale = proposal_noise * std(pool)
    )
    # Precision scale for GaussCorrectedScoreFn (None = auto-estimate).
    fnpe_gauss_precision_scale: Optional[float] = None

    # --- Simformer-specific ---
    simformer_token_dim: int = 40  # Token dimension for value embedding
    simformer_condition_token_dim: int = 10  # Dimension for condition mask embedding
    simformer_time_embedding_dim: int = 128  # Time embedding dimension
    simformer_num_heads: int = 4  # Transformer attention heads
    simformer_num_layers: int = 6  # Transformer layers
    simformer_attn_size: int = 10  # Attention size per head
    simformer_widening_factor: int = 3  # MLP widening factor in transformer
    simformer_sigma_min: float = 0.01  # VESDE minimum noise
    simformer_sigma_max: float = 15.0  # VESDE maximum noise
    simformer_t_min: float = 0.02  # Minimum diffusion time
    simformer_t_max: float = 1.0  # Maximum diffusion time (not too large)
    simformer_num_diffusion_steps: int = 500  # Reverse SDE sampling steps
    simformer_learning_rate: float = 1e-3  # Training learning rate
    simformer_num_train_steps: int = 50000  # Number of training steps
    simformer_batch_size: int = 1024  # Training batch size
    simformer_embedding_batch_size: int = 128  # GPU batch size for pre-embedding long sequences

    # --- Training ---
    learning_rate: float = 5e-4  # Lower LR for complex data (MarkovSBI large uses 5e-4)
    training_batch_size: int = 512  # Larger batch for stability
    validation_fraction: float = 0.15  # 15% for stable validation metrics
    stop_after_epochs: int = 30  # High patience, let LR schedule do its work
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
    run_swd: bool = False  # Deprecated: prior-vs-DAP SWD removed from eval pipeline
    run_one_step_rmse: bool = True
    run_posterior_plots: bool = True
    unify_eval_budgets: bool = True  # Use identical evaluation budgets across methods
    run_simulated_test_eval: bool = True  # Run held-out synthetic test evaluation
    run_simulated_ppc: bool = True  # Run simulated PPC on held-out synthetic cases
    num_test_simulations: int = 100  # Size of held-out synthetic test set
    num_simulated_ppc_examples: int = 2  # Number of held-out PPC plots to save
    simulated_test_ppc_samples: int = 200  # PPC samples per held-out test case
    benchmark_eval_seed: int = 314159  # Fixed seed for shared held-out test set
    test_region_theta_tail_frac: float = 0.25  # Parameter-tail width for holdout region
    test_region_require_joint_holdout: bool = True  # Require theta + driving holdout
    test_region_speed_margin_frac: float = 0.25  # Margin relative to init speed range
    test_region_min_abs_steer_deg: float = 6.0  # Steering threshold for holdout region
    test_region_min_brake: float = 180.0  # Brake threshold for holdout region
    test_region_min_driving_flags: int = 2  # Number of driving-condition flags required
    benchmark_max_attempt_factor: int = 150  # Max rejection-sampling multiplier

    # --- Real data eval ---
    # Disabled by default. Set explicitly to enable real-data evaluation.
    real_data_csv: Optional[str] = None
    # Directory containing all real measurement CSVs for multi-trajectory PPC
    real_data_dir: str = "../data/measurements"
    # Number of posterior predictive samples per trajectory in PPC plots
    K_ppc: int = 300
    # Window selection: prefer windows with low brake pressure
    prefer_low_brake: bool = False

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
        valid_methods = {"npe", "npse", "fnpe", "simformer"}
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
        if (
            self.run_simulated_test_eval
            and (self.cache_dataset or self.reuse_dataset)
            and self.test_dataset_id is None
        ):
            self.test_dataset_id = self._generate_test_dataset_id()

    def _generate_dataset_id(self) -> str:
        """Generate a unique dataset ID based on simulation parameters."""
        data_params = {
            "holdout_region_version": "prior_quantile_v2",
            "sim_seed": self.sim_seed,
            "num_simulations": self.num_simulations,
            "T_seg": self.T_seg,
            "dt": self.dt,
            "steer_scale": self.steer_scale,
            "init_speed_center_ms": self.init_speed_center_ms,
            "init_speed_range_ms": self.init_speed_range_ms,
            "brake_block_fraction": self.brake_block_fraction,
            "accel_scale": self.accel_scale,
            "emergency_brake_fraction": self.emergency_brake_fraction,
            "ramp_s": self.ramp_s,
            "obs_noise_scale": self.obs_noise_scale,
            "process_noise_scale": self.process_noise_scale,
            "active_parameters": self.active_parameters,
            "prior_bounds": self.param_bounds(),
            "fixed_values": self.fixed_param_values(),
            "test_region_theta_tail_frac": self.test_region_theta_tail_frac,
            "test_region_require_joint_holdout": self.test_region_require_joint_holdout,
            "test_region_speed_margin_frac": self.test_region_speed_margin_frac,
            "test_region_min_abs_steer_deg": self.test_region_min_abs_steer_deg,
            "test_region_min_brake": self.test_region_min_brake,
            "test_region_min_driving_flags": self.test_region_min_driving_flags,
        }
        param_str = json.dumps(data_params, sort_keys=True)
        hash_val = hashlib.md5(param_str.encode()).hexdigest()[:12]
        return f"dataset_{hash_val}"

    def _generate_test_dataset_id(self) -> str:
        """Generate a unique ID for the shared held-out synthetic test set."""
        test_params = {
            "holdout_region_version": "prior_quantile_v2",
            "benchmark_eval_seed": self.benchmark_eval_seed,
            "num_test_simulations": self.num_test_simulations,
            "T_seg": self.T_seg,
            "dt": self.dt,
            "steer_scale": self.steer_scale,
            "init_speed_center_ms": self.init_speed_center_ms,
            "init_speed_range_ms": self.init_speed_range_ms,
            "brake_block_fraction": self.brake_block_fraction,
            "accel_scale": self.accel_scale,
            "emergency_brake_fraction": self.emergency_brake_fraction,
            "ramp_s": self.ramp_s,
            "obs_noise_scale": self.obs_noise_scale,
            "process_noise_scale": self.process_noise_scale,
            "active_parameters": self.active_parameters,
            "prior_bounds": self.param_bounds(),
            "fixed_values": self.fixed_param_values(),
            "test_region_theta_tail_frac": self.test_region_theta_tail_frac,
            "test_region_require_joint_holdout": self.test_region_require_joint_holdout,
            "test_region_speed_margin_frac": self.test_region_speed_margin_frac,
            "test_region_min_abs_steer_deg": self.test_region_min_abs_steer_deg,
            "test_region_min_brake": self.test_region_min_brake,
            "test_region_min_driving_flags": self.test_region_min_driving_flags,
        }
        param_str = json.dumps(test_params, sort_keys=True)
        hash_val = hashlib.md5(param_str.encode()).hexdigest()[:12]
        return f"dataset_test_{hash_val}"

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

    def get_test_dataset_cache_path(self) -> Path:
        """Get path for cached held-out test dataset."""
        cache_dir = Path(self.dataset_cache_dir)
        return cache_dir / f"{self.test_dataset_id}.pt"

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
