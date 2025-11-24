from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple


# Parameter ordering expected by the JAX vehicle model (friction, drag, mass)
PARAMETER_ORDER: Tuple[str, ...] = ("mu", "cd", "m")


@dataclass
class ExperimentConfig:
    # --- General ---
    exp_name: str = "baseline_bigru_maf"
    random_seed: int = 42
    device: str = "cuda"  # "auto", "cpu", or "cuda"

    # --- Simulation settings ---
    dt: float = 0.01
    T_seg: int = 3000
    decimate: int = 2
    state_dim: int = 10
    obs_dim: int = 9

    # --- Prior (3 params: mu, C_d, m) ---
    prior_low_mu: float = 0.50
    prior_high_mu: float = 1.50
    prior_low_cd: float = 0.05
    prior_high_cd: float = 0.60
    prior_low_m: float = 1200.0
    prior_high_m: float = 2200.0

    # --- Which parameters to infer + defaults for fixed ones ---
    # active_parameters controls theta dimensionality (1, 2, or 3 elements)
    active_parameters: Tuple[str, ...] = PARAMETER_ORDER
    fixed_mu: float = 1.0
    fixed_cd: float = 0.30
    fixed_m: float = 1700.0

    # --- Dataset ---
    num_simulations: int = 2000
    batch_sim: int = 512
    jit_warmup: bool = True

    # --- Encoder / density estimator ---
    encoder_type: str = "bigru"  # bigru | causalcnn | transformer
    encoder_hidden: int = 32

    # ---- Transformer ----
    transformer_layers: int = 2
    transformer_heads: int = 4
    transformer_head_dim: int = 16
    transformer_feature_dim: int = transformer_heads * transformer_head_dim

    # ---- CausalCNN ----
    causalcnn_num_layers: int = 4
    causalcnn_kernel_size: int = 3
    causalcnn_pool_kernel: int = 64

    # ---- common ----
    embedding_output_dim: int = 64  # used by CNN / Transformer

    maf_hidden_features: int = 128
    maf_num_transforms: int = 8

    # --- Training ---
    learning_rate: float = 1e-3
    training_batch_size: int = 512
    validation_fraction: float = 0.10
    stop_after_epochs: int = 20
    clip_max_norm: float = 5.0

    # --- SBC / diagnostics ---
    num_sbc_samples: int = 200
    num_posterior_samples_sbc: int = 1000
    num_calibration_items: int = 5
    num_lc2st_samples: int = 1000
    num_swd_projections: int = 1000

    # --- Real data eval ---
    real_data_csv: Optional[str] = None

    # --- Logging / saving ---
    results_root: str = "experiments"

    def __post_init__(self) -> None:
        """
        Normalize and validate the active_parameters list.
        """
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

        # Fallback to full set if user passed an empty list/tuple
        self.active_parameters = tuple(cleaned) if cleaned else PARAMETER_ORDER

    def param_bounds(self) -> Dict[str, tuple]:
        """
        Bounds for each physical parameter keyed by name.
        """
        return {
            "mu": (self.prior_low_mu, self.prior_high_mu),
            "cd": (self.prior_low_cd, self.prior_high_cd),
            "m": (self.prior_low_m, self.prior_high_m),
        }

    def fixed_param_values(self) -> Dict[str, float]:
        """
        Defaults to plug in for parameters that are not inferred.
        """
        return {"mu": self.fixed_mu, "cd": self.fixed_cd, "m": self.fixed_m}

    def active_param_dim(self) -> int:
        return len(self.active_parameters)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
