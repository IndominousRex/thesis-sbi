"""Unified experiment configuration for SBI runs."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import hashlib
import json


PARAMETER_ORDER: Tuple[str, ...] = ("mu", "cd", "m")
VALID_METHODS = {"npe", "npse", "fnpe", "simformer"}
VALID_ENCODERS = {"bigru", "causalcnn", "transformer"}
VALID_SDE_TYPES = {"ve", "vp", "subvp"}
VALID_FNPE_MODEL_TYPES = {"gru", "linear"}
VALID_FNPE_OPTIMIZERS = {"adam", "adamw"}
VALID_FNPE_SCHEDULERS = {"constant", "cosine"}
VALID_FNPE_SCORE_FNS = {"fnpe", "uncorrected", "gauss_corrected"}
VALID_FNPE_PROPOSALS = {"pred", "naive", "trajectory"}


@dataclass
class ExperimentConfig:
    config_version: str = "experiment_config_v3"
    dataset_recipe_version: str = "prior_quantile_v2"
    test_dataset_format_version: str = "true_state0_v1"

    exp_name: str = "baseline"
    method: str = "npe"
    device: str = "cuda"

    random_seed: int = 42
    sim_seed: Optional[int] = None
    train_seed: Optional[int] = None

    dt: float = 0.01
    T_seg: int = 3000
    state_dim: int = 10
    obs_dim: int = 9
    steer_scale: float = 0.5
    init_speed_center_ms: float = 11.0
    init_speed_range_ms: float = 10.0
    brake_block_fraction: float = 0.5
    accel_scale: float = 1.2
    emergency_brake_fraction: float = 0.4
    ramp_s: float = 0.1
    obs_noise_scale: float = 1.0
    process_noise_scale: float = 1.0

    prior_low_mu: float = 0.50
    prior_high_mu: float = 1.50
    prior_low_cd: float = 0.05
    prior_high_cd: float = 0.60
    prior_low_m: float = 1700.0
    prior_high_m: float = 2200.0

    active_parameters: Tuple[str, ...] = PARAMETER_ORDER
    fixed_mu: float = 1.0
    fixed_cd: float = 0.27
    fixed_m: float = 1720.0

    num_simulations: int = 2000
    requested_budget_steps: Optional[int] = None
    derived_num_simulations: Optional[int] = None
    batch_sim: int = 512
    jit_warmup: bool = True

    dataset_id: Optional[str] = None
    test_dataset_id: Optional[str] = None
    dataset_cache_dir: str = "datasets"
    cache_dataset: bool = False
    reuse_dataset: bool = False

    encoder_type: str = "bigru"
    encoder_hidden: int = 32
    transformer_layers: int = 3
    transformer_heads: int = 4
    transformer_head_dim: int = 16
    transformer_feature_dim: Optional[int] = None
    causalcnn_num_layers: int = 4
    causalcnn_kernel_size: int = 3
    causalcnn_pool_kernel: int = 64
    embedding_output_dim: int = 64

    maf_hidden_features: int = 128
    maf_num_transforms: int = 8
    sde_type: str = "ve"

    fnpe_hidden_dim: int = 128
    fnpe_num_hidden: int = 5
    fnpe_model_type: str = "gru"
    fnpe_window_size: int = 2
    fnpe_num_diffusion_steps: int = 500
    fnpe_num_simulations: int = 100000
    fnpe_num_outer_epochs: int = 100
    fnpe_num_inner_epochs: int = 50
    fnpe_batch_size: int = 1000
    fnpe_learning_rate: float = 5e-4
    fnpe_clip_max_norm: float = 20.0
    fnpe_optimizer: str = "adamw"
    fnpe_scheduler: str = "cosine"
    fnpe_score_fn_type: str = "gauss_corrected"
    fnpe_proposal_type: str = "pred"
    fnpe_t_min: float = 0.05
    fnpe_skip_normalize: bool = False
    fnpe_clip_samples: bool = True
    fnpe_pilot_fraction: float = 0.02
    fnpe_pilot_length: int = 1500
    fnpe_proposal_noise: float = 0.03
    fnpe_gauss_precision_scale: Optional[float] = None
    fnpe_sampling_batch_size: int = 64
    fnpe_gauss_hyper_num_steps: int = 100
    fnpe_gauss_hyper_num_samples: int = 128

    simformer_num_timepoints: int = 64
    simformer_token_dim: int = 40
    simformer_condition_token_dim: int = 10
    simformer_condition_token_init_scale: float = 0.01
    simformer_condition_token_init_mean: float = 0.0
    simformer_condition_mode: str = "concat"
    simformer_time_embedding_dim: int = 128
    simformer_num_heads: int = 4
    simformer_num_layers: int = 8
    simformer_attn_size: int = 10
    simformer_widening_factor: int = 4
    simformer_num_hidden_layers: int = 1
    simformer_skip_connection_attn: bool = True
    simformer_skip_connection_mlp: bool = True
    simformer_layer_norm: bool = True
    simformer_use_metadata: bool = True
    simformer_condition_mask_name: str = "structured_random"
    simformer_condition_mask_p_joint: float = 0.2
    simformer_condition_mask_p_posterior: float = 0.2
    simformer_condition_mask_p_likelihood: float = 0.2
    simformer_condition_mask_p_rnd1: float = 0.2
    simformer_condition_mask_p_rnd2: float = 0.2
    simformer_condition_mask_rnd1_prob: float = 0.3
    simformer_condition_mask_rnd2_prob: float = 0.7
    simformer_edge_mask_name: str = "none"
    simformer_rebalance_loss: bool = False
    simformer_sigma_min: float = 0.01
    simformer_sigma_max: float = 15.0
    simformer_t_min: float = 0.02
    simformer_t_max: float = 1.0
    simformer_num_diffusion_steps: int = 500
    simformer_sampling_batch_size: int = 128
    simformer_learning_rate: float = 1e-3
    simformer_min_learning_rate: float = 1e-6
    simformer_clip_max_norm: float = 10.0
    simformer_batch_size: int = 32
    simformer_train_steps_scaling: int = 3
    simformer_min_train_steps: int = 5000
    simformer_max_train_steps: int = 100000
    simformer_validation_fraction: float = 0.0
    simformer_val_repeat: int = 5
    simformer_val_every: int = 50
    simformer_stop_early_count: int = 5
    simformer_val_error_ratio: float = 1.1

    learning_rate: float = 5e-4
    training_batch_size: int = 512
    validation_fraction: float = 0.15
    clip_max_norm: float = 20.0
    num_epochs: int = 300

    num_sbc_samples: int = 400
    num_posterior_samples_sbc: int = 2000
    num_calibration_items: int = 5
    num_swd_projections: int = 2000

    run_sbc: bool = False
    run_one_step_rmse: bool = True
    run_posterior_plots: bool = True
    unify_eval_budgets: bool = True
    run_simulated_test_eval: bool = True
    run_simulated_ppc: bool = True
    num_test_simulations: int = 500
    num_simulated_ppc_examples: int = 50
    num_simulated_ppc_plot_examples: int = 2
    simulated_test_ppc_samples: int = 400
    benchmark_posterior_plot_examples: int = 3
    benchmark_posterior_plot_samples: int = 10000
    benchmark_pairplot_examples: int = 3
    benchmark_pairplot_posterior_samples: int = 1000
    benchmark_c2st_examples: int = 3
    benchmark_c2st_posterior_samples: int = 1000
    benchmark_one_step_cases: int = 50
    benchmark_one_step_posterior_samples: int = 100
    benchmark_w2_cases: int = 100
    benchmark_w2_posterior_samples: int = 400
    benchmark_eval_seed: int = 314159
    test_region_theta_tail_frac: float = 0.25
    test_region_require_joint_holdout: bool = True
    test_region_speed_margin_frac: float = 0.25
    test_region_min_abs_steer_deg: float = 6.0
    test_region_min_brake: float = 180.0
    test_region_min_driving_flags: int = 2
    benchmark_max_attempt_factor: int = 150

    real_data_csv: Optional[str] = None
    real_data_dir: str = "../data/measurements"
    K_ppc: int = 300
    prefer_low_brake: bool = False

    results_root: str = "experiments"
    output_dir: Optional[str] = None
    no_plots: bool = False

    do_train: bool = True
    do_eval: bool = True
    checkpoint: Optional[str] = None

    def __post_init__(self) -> None:
        self.method = self._normalize_choice("method", self.method, VALID_METHODS)
        self.encoder_type = self._normalize_choice(
            "encoder_type", self.encoder_type, VALID_ENCODERS
        )
        self.sde_type = self._normalize_choice(
            "sde_type", self.sde_type, VALID_SDE_TYPES
        )
        self.fnpe_model_type = self._normalize_choice(
            "fnpe_model_type", self.fnpe_model_type, VALID_FNPE_MODEL_TYPES
        )
        self.fnpe_optimizer = self._normalize_choice(
            "fnpe_optimizer", self.fnpe_optimizer, VALID_FNPE_OPTIMIZERS
        )
        self.fnpe_scheduler = self._normalize_choice(
            "fnpe_scheduler", self.fnpe_scheduler, VALID_FNPE_SCHEDULERS
        )
        self.fnpe_score_fn_type = self._normalize_choice(
            "fnpe_score_fn_type", self.fnpe_score_fn_type, VALID_FNPE_SCORE_FNS
        )
        self.fnpe_proposal_type = self._normalize_choice(
            "fnpe_proposal_type", self.fnpe_proposal_type, VALID_FNPE_PROPOSALS
        )
        self.active_parameters = self._normalize_active_parameters(
            self.active_parameters
        )
        self.dataset_cache_dir = str(self.dataset_cache_dir).strip()
        self.real_data_dir = str(self.real_data_dir).strip()
        if not self.dataset_cache_dir:
            raise ValueError("dataset_cache_dir must not be blank.")
        if not self.real_data_dir:
            raise ValueError("real_data_dir must not be blank.")
        if self.real_data_csv is not None:
            self.real_data_csv = str(self.real_data_csv).strip()
            if not self.real_data_csv:
                raise ValueError("real_data_csv must not be blank when provided.")
        if self.sim_seed is None:
            self.sim_seed = int(self.random_seed)
        if self.train_seed is None:
            self.train_seed = int(self.random_seed) + 1
        if self.requested_budget_steps is None and self.derived_num_simulations is None:
            self.derived_num_simulations = int(self.num_simulations)
        if self.transformer_feature_dim is None:
            self.transformer_feature_dim = int(
                self.transformer_heads * self.transformer_head_dim
            )
        self._validate_base_fields()
        self._validate_relationships()
        self._validate_method_specific_fields()
        if (
            self.requested_budget_steps is None
            or self.derived_num_simulations is not None
        ):
            self.ensure_dataset_ids()

    @staticmethod
    def _normalize_choice(name: str, value: str, valid: set[str]) -> str:
        normalized = str(value).strip().lower()
        if normalized not in valid:
            raise ValueError(
                f"Unknown {name} '{value}'. Valid options: {sorted(valid)}"
            )
        return normalized

    @staticmethod
    def _normalize_active_parameters(
        value: Tuple[str, ...] | list[str] | str,
    ) -> Tuple[str, ...]:
        raw_names = (
            [part.strip() for part in value.split(",")]
            if isinstance(value, str)
            else list(value)
        )
        cleaned = []
        for name in raw_names:
            normalized = str(name).strip().lower()
            if not normalized:
                continue
            if normalized not in PARAMETER_ORDER:
                raise ValueError(
                    f"Unknown parameter '{name}'. Valid options: {PARAMETER_ORDER}"
                )
            if normalized not in cleaned:
                cleaned.append(normalized)
        if not cleaned:
            raise ValueError("active_parameters must not be empty.")
        return tuple(cleaned)

    def _require_positive(self, name: str, value: float | int) -> None:
        if value <= 0:
            raise ValueError(f"{name} must be > 0, got {value}.")

    def _require_nonnegative(self, name: str, value: float | int) -> None:
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}.")

    def _require_fraction(
        self,
        name: str,
        value: float,
        *,
        lower: float = 0.0,
        upper: float = 1.0,
        inclusive_lower: bool = True,
        inclusive_upper: bool = True,
    ) -> None:
        lower_ok = value >= lower if inclusive_lower else value > lower
        upper_ok = value <= upper if inclusive_upper else value < upper
        if not (lower_ok and upper_ok):
            left = "[" if inclusive_lower else "("
            right = "]" if inclusive_upper else ")"
            raise ValueError(
                f"{name} must be in {left}{lower}, {upper}{right}, got {value}."
            )

    def _validate_base_fields(self) -> None:
        for name, value in {
            "dt": self.dt,
            "T_seg": self.T_seg,
            "state_dim": self.state_dim,
            "obs_dim": self.obs_dim,
            "num_simulations": self.num_simulations,
            "batch_sim": self.batch_sim,
            "encoder_hidden": self.encoder_hidden,
            "embedding_output_dim": self.embedding_output_dim,
            "maf_hidden_features": self.maf_hidden_features,
            "maf_num_transforms": self.maf_num_transforms,
            "training_batch_size": self.training_batch_size,
            "learning_rate": self.learning_rate,
            "clip_max_norm": self.clip_max_norm,
            "num_epochs": self.num_epochs,
            "num_sbc_samples": self.num_sbc_samples,
            "num_posterior_samples_sbc": self.num_posterior_samples_sbc,
            "num_calibration_items": self.num_calibration_items,
            "num_swd_projections": self.num_swd_projections,
            "benchmark_posterior_plot_examples": self.benchmark_posterior_plot_examples,
            "benchmark_posterior_plot_samples": self.benchmark_posterior_plot_samples,
            "benchmark_pairplot_examples": self.benchmark_pairplot_examples,
            "benchmark_pairplot_posterior_samples": self.benchmark_pairplot_posterior_samples,
            "benchmark_c2st_examples": self.benchmark_c2st_examples,
            "benchmark_c2st_posterior_samples": self.benchmark_c2st_posterior_samples,
            "benchmark_one_step_cases": self.benchmark_one_step_cases,
            "benchmark_one_step_posterior_samples": self.benchmark_one_step_posterior_samples,
            "benchmark_w2_cases": self.benchmark_w2_cases,
            "benchmark_w2_posterior_samples": self.benchmark_w2_posterior_samples,
            "test_region_min_driving_flags": self.test_region_min_driving_flags,
            "benchmark_max_attempt_factor": self.benchmark_max_attempt_factor,
            "K_ppc": self.K_ppc,
        }.items():
            self._require_positive(name, value)
        for name, value in {
            "brake_block_fraction": self.brake_block_fraction,
            "ramp_s": self.ramp_s,
            "obs_noise_scale": self.obs_noise_scale,
            "process_noise_scale": self.process_noise_scale,
            "num_test_simulations": self.num_test_simulations,
            "num_simulated_ppc_examples": self.num_simulated_ppc_examples,
            "num_simulated_ppc_plot_examples": self.num_simulated_ppc_plot_examples,
        }.items():
            if value is not None:
                self._require_nonnegative(name, value)
        for name, value in {
            "requested_budget_steps": self.requested_budget_steps,
            "derived_num_simulations": self.derived_num_simulations,
        }.items():
            if value is not None:
                self._require_positive(name, value)
        self._require_positive("steer_scale", self.steer_scale)
        self._require_positive("init_speed_center_ms", self.init_speed_center_ms)
        self._require_positive("init_speed_range_ms", self.init_speed_range_ms)
        self._require_positive("accel_scale", self.accel_scale)
        self._require_positive(
            "simulated_test_ppc_samples", self.simulated_test_ppc_samples
        )
        self._require_positive(
            "test_region_min_abs_steer_deg", self.test_region_min_abs_steer_deg
        )
        self._require_positive("test_region_min_brake", self.test_region_min_brake)
        self._require_fraction(
            "emergency_brake_fraction", self.emergency_brake_fraction
        )
        self._require_fraction(
            "validation_fraction",
            self.validation_fraction,
            upper=1.0,
            inclusive_upper=False,
        )
        self._require_fraction(
            "simformer_validation_fraction",
            self.simformer_validation_fraction,
            upper=1.0,
            inclusive_upper=False,
        )
        self._require_fraction(
            "fnpe_pilot_fraction",
            self.fnpe_pilot_fraction,
            lower=0.0,
            inclusive_lower=False,
        )
        self._require_fraction(
            "test_region_theta_tail_frac",
            self.test_region_theta_tail_frac,
            lower=0.0,
            inclusive_lower=False,
        )
        self._require_fraction(
            "test_region_speed_margin_frac",
            self.test_region_speed_margin_frac,
            lower=0.0,
            inclusive_lower=False,
        )
        for name, (low, high) in self.param_bounds().items():
            if low >= high:
                raise ValueError(
                    f"Prior bounds for {name} must satisfy low < high, got ({low}, {high})."
                )
        for name, value in self.fixed_param_values().items():
            low, high = self.param_bounds()[name]
            if not (low <= value <= high):
                raise ValueError(
                    f"fixed_{name}={value} must lie within prior bounds [{low}, {high}]."
                )

    def _validate_relationships(self) -> None:
        if self.run_simulated_ppc and not self.run_simulated_test_eval:
            raise ValueError(
                "run_simulated_ppc=True requires run_simulated_test_eval=True."
            )
        if self.num_simulated_ppc_plot_examples > self.num_simulated_ppc_examples:
            raise ValueError(
                "num_simulated_ppc_plot_examples must be <= num_simulated_ppc_examples."
            )
        if self.run_simulated_test_eval and self.num_test_simulations <= 0:
            raise ValueError(
                "num_test_simulations must be > 0 when run_simulated_test_eval=True."
            )
        if self.run_simulated_test_eval:
            for name, value in {
                "benchmark_posterior_plot_examples": self.benchmark_posterior_plot_examples,
                "benchmark_pairplot_examples": self.benchmark_pairplot_examples,
                "benchmark_c2st_examples": self.benchmark_c2st_examples,
            }.items():
                if value > self.num_test_simulations:
                    raise ValueError(
                        f"{name}={value} must be <= num_test_simulations={self.num_test_simulations}."
                    )
        for name, value in {
            "transformer_layers": self.transformer_layers,
            "transformer_heads": self.transformer_heads,
            "transformer_head_dim": self.transformer_head_dim,
            "transformer_feature_dim": int(self.transformer_feature_dim),
            "causalcnn_num_layers": self.causalcnn_num_layers,
            "causalcnn_kernel_size": self.causalcnn_kernel_size,
            "causalcnn_pool_kernel": self.causalcnn_pool_kernel,
        }.items():
            self._require_positive(name, value)
        if (
            self.encoder_type == "transformer"
            and self.transformer_feature_dim % self.transformer_heads != 0
        ):
            raise ValueError(
                "transformer_feature_dim must be divisible by transformer_heads."
            )

    def _validate_method_specific_fields(self) -> None:
        if self.method == "fnpe":
            for name, value in {
                "fnpe_hidden_dim": self.fnpe_hidden_dim,
                "fnpe_num_hidden": self.fnpe_num_hidden,
                "fnpe_window_size": self.fnpe_window_size,
                "fnpe_num_diffusion_steps": self.fnpe_num_diffusion_steps,
                "fnpe_num_simulations": self.fnpe_num_simulations,
                "fnpe_num_outer_epochs": self.fnpe_num_outer_epochs,
                "fnpe_num_inner_epochs": self.fnpe_num_inner_epochs,
                "fnpe_batch_size": self.fnpe_batch_size,
                "fnpe_learning_rate": self.fnpe_learning_rate,
                "fnpe_clip_max_norm": self.fnpe_clip_max_norm,
                "fnpe_pilot_length": self.fnpe_pilot_length,
                "fnpe_sampling_batch_size": self.fnpe_sampling_batch_size,
                "fnpe_gauss_hyper_num_steps": self.fnpe_gauss_hyper_num_steps,
                "fnpe_gauss_hyper_num_samples": self.fnpe_gauss_hyper_num_samples,
            }.items():
                self._require_positive(name, value)
            self._require_positive("fnpe_t_min", self.fnpe_t_min)
            self._require_nonnegative("fnpe_proposal_noise", self.fnpe_proposal_noise)
            if self.fnpe_gauss_precision_scale is not None:
                self._require_positive(
                    "fnpe_gauss_precision_scale", self.fnpe_gauss_precision_scale
                )
        if self.method == "simformer":
            for name, value in {
                "simformer_num_timepoints": self.simformer_num_timepoints,
                "simformer_token_dim": self.simformer_token_dim,
                "simformer_condition_token_dim": self.simformer_condition_token_dim,
                "simformer_time_embedding_dim": self.simformer_time_embedding_dim,
                "simformer_num_heads": self.simformer_num_heads,
                "simformer_num_layers": self.simformer_num_layers,
                "simformer_attn_size": self.simformer_attn_size,
                "simformer_widening_factor": self.simformer_widening_factor,
                "simformer_num_hidden_layers": self.simformer_num_hidden_layers,
                "simformer_num_diffusion_steps": self.simformer_num_diffusion_steps,
                "simformer_sampling_batch_size": self.simformer_sampling_batch_size,
                "simformer_learning_rate": self.simformer_learning_rate,
                "simformer_min_learning_rate": self.simformer_min_learning_rate,
                "simformer_clip_max_norm": self.simformer_clip_max_norm,
                "simformer_batch_size": self.simformer_batch_size,
                "simformer_train_steps_scaling": self.simformer_train_steps_scaling,
                "simformer_min_train_steps": self.simformer_min_train_steps,
                "simformer_max_train_steps": self.simformer_max_train_steps,
                "simformer_val_repeat": self.simformer_val_repeat,
                "simformer_val_every": self.simformer_val_every,
                "simformer_stop_early_count": self.simformer_stop_early_count,
            }.items():
                self._require_positive(name, value)
            self._require_positive("simformer_sigma_min", self.simformer_sigma_min)
            self._require_positive("simformer_sigma_max", self.simformer_sigma_max)
            self._require_positive("simformer_t_min", self.simformer_t_min)
            self._require_positive(
                "simformer_val_error_ratio", self.simformer_val_error_ratio
            )
            if self.simformer_t_min >= self.simformer_t_max:
                raise ValueError("simformer_t_min must be < simformer_t_max.")
            if self.simformer_sigma_min >= self.simformer_sigma_max:
                raise ValueError("simformer_sigma_min must be < simformer_sigma_max.")
            for name, value in {
                "simformer_condition_mask_p_joint": self.simformer_condition_mask_p_joint,
                "simformer_condition_mask_p_posterior": self.simformer_condition_mask_p_posterior,
                "simformer_condition_mask_p_likelihood": self.simformer_condition_mask_p_likelihood,
                "simformer_condition_mask_p_rnd1": self.simformer_condition_mask_p_rnd1,
                "simformer_condition_mask_p_rnd2": self.simformer_condition_mask_p_rnd2,
                "simformer_condition_mask_rnd1_prob": self.simformer_condition_mask_rnd1_prob,
                "simformer_condition_mask_rnd2_prob": self.simformer_condition_mask_rnd2_prob,
            }.items():
                self._require_fraction(name, value)
            mask_sum = (
                self.simformer_condition_mask_p_joint
                + self.simformer_condition_mask_p_posterior
                + self.simformer_condition_mask_p_likelihood
                + self.simformer_condition_mask_p_rnd1
                + self.simformer_condition_mask_p_rnd2
            )
            if abs(mask_sum - 1.0) > 1e-6:
                raise ValueError(
                    f"Simformer condition-mask probabilities must sum to 1.0, got {mask_sum}."
                )

    def assert_budget_resolution_ready(self, entrypoint: str) -> None:
        if (
            self.requested_budget_steps is not None
            and self.derived_num_simulations is None
        ):
            raise ValueError(
                f"{entrypoint} requires derived_num_simulations to be resolved before data generation for budget-driven runs."
            )

    def ensure_dataset_ids(self) -> None:
        if (self.cache_dataset or self.reuse_dataset) and self.dataset_id is None:
            self.dataset_id = self._generate_dataset_id()
        if (
            self.run_simulated_test_eval
            and (self.cache_dataset or self.reuse_dataset)
            and self.test_dataset_id is None
        ):
            self.test_dataset_id = self._generate_test_dataset_id()

    def _generate_dataset_id(self) -> str:
        payload = {
            "config_version": self.config_version,
            "dataset_recipe_version": self.dataset_recipe_version,
            "sim_seed": self.sim_seed,
            "num_simulations": self.num_simulations,
            "obs_dim": self.obs_dim,
            "state_dim": self.state_dim,
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
        hash_val = hashlib.md5(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()[:12]
        return f"dataset_{hash_val}"

    def _generate_test_dataset_id(self) -> str:
        payload = {
            "config_version": self.config_version,
            "dataset_recipe_version": self.dataset_recipe_version,
            "test_dataset_format_version": self.test_dataset_format_version,
            "benchmark_eval_seed": self.benchmark_eval_seed,
            "num_test_simulations": self.num_test_simulations,
            "obs_dim": self.obs_dim,
            "state_dim": self.state_dim,
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
        hash_val = hashlib.md5(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()[:12]
        return f"dataset_test_{hash_val}"

    def param_bounds(self) -> Dict[str, tuple]:
        return {
            "mu": (self.prior_low_mu, self.prior_high_mu),
            "cd": (self.prior_low_cd, self.prior_high_cd),
            "m": (self.prior_low_m, self.prior_high_m),
        }

    def fixed_param_values(self) -> Dict[str, float]:
        return {"mu": self.fixed_mu, "cd": self.fixed_cd, "m": self.fixed_m}

    def active_param_dim(self) -> int:
        return len(self.active_parameters)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def get_dataset_cache_path(self) -> Path:
        return Path(self.dataset_cache_dir) / f"{self.dataset_id}.pt"

    def get_test_dataset_cache_path(self) -> Path:
        return Path(self.dataset_cache_dir) / f"{self.test_dataset_id}.pt"

    def get_experiment_name(self) -> str:
        params_slug = "_".join(self.active_parameters)
        budget_part = (
            f"b{int(self.requested_budget_steps)}"
            if self.requested_budget_steps is not None
            else f"n{int(self.num_simulations)}"
        )
        exp_prefix = self.exp_name
        if not (
            exp_prefix == self.method
            or exp_prefix.startswith(f"{self.method}_")
            or exp_prefix.endswith(f"_{self.method}")
        ):
            exp_prefix = f"{self.method}_{exp_prefix}"
        return f"{exp_prefix}_p{params_slug}_t{int(self.T_seg)}_s{int(self.sim_seed)}_tr{int(self.train_seed)}_{budget_part}"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        if "active_parameters" in d and isinstance(d["active_parameters"], list):
            d["active_parameters"] = tuple(d["active_parameters"])
        # Filter out unknown keys that may exist in saved configs from older versions
        import dataclasses

        valid_fields = {f.name for f in dataclasses.fields(cls)}
        d = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**d)

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
