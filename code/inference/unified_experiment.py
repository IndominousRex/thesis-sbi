"""
Unified experiment runner for all SBI methods.

This module provides a method-agnostic experiment pipeline that:
- Uses identical data generation and normalization for all methods
- Provides consistent evaluation metrics
- Supports dataset caching for fair comparisons
- Saves results in a uniform format
"""

import json
import os
import pickle
import time
import hashlib
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

import torch
import numpy as np
from scipy.stats import binomtest, kstest
from sbi import utils as sbi_utils
from sbi.diagnostics import run_sbc, check_sbc
from tqdm.auto import tqdm

from configs.config import ExperimentConfig
from utils.env_utils import setup_environment, get_device
from utils.metrics import (
    per_parameter_posterior_metrics,
    sliced_wasserstein_prior_vs_dap,
    one_step_rmse_observation,
    real_data_trajectory_metrics,
    wasserstein2_posterior_vs_true,
)
from utils.normalization import (
    fit_normalizer,
    save_normalizer,
    load_normalizer,
    Normalizer,
)
from utils.real_data import (
    initial_state_from_obs,
    simulate_y_batch_for_thetas,
    build_real_window_from_csv,
    build_simulated_window_for_eval,
    posterior_predictive_from_real,
    OBS_LABELS,
    prep_x_obs_from_df,
)
from utils.plots import (
    plot_prior_posterior_1d,
    plot_prior_posterior_grid,
    plot_sbc_rank_hist,
    plot_expected_coverage_curve,
    plot_training_curves,
    plot_ppc_trajectories,
    plot_obs_1d_hist_custom,
    plot_pairplot,
    compute_c2st,
    plot_c2st_comparison,
    plot_diffusion_traces,
)
from simulation.simulation import (
    init_simulation_from_config,
    make_simulator,
    generate_dataset,
)
from models.models import build_prior
from methods import build_method, AVAILABLE_METHODS


def make_experiment_dir(cfg: ExperimentConfig) -> Path:
    """Create experiment directory with method prefix and timestamp."""
    if cfg.output_dir:
        exp_dir = Path(cfg.output_dir)
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path(cfg.results_root)
    root.mkdir(parents=True, exist_ok=True)

    # Include method in directory name
    exp_name = cfg.get_experiment_name()
    exp_dir = root / f"{exp_name}_{ts}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def tensor_to_python(x):
    """Convert PyTorch tensors to Python scalars/lists recursively."""
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return x.item()
        else:
            return x.cpu().numpy().tolist()
    return x


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=tensor_to_python)


def _update_run_status(
    status_path: Path,
    *,
    state: str,
    stage: str,
    exp_dir: Path,
    completed_stages: Optional[List[str]] = None,
    error: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "state": state,
        "stage": stage,
        "exp_dir": str(exp_dir),
        "updated_at": datetime.now().isoformat(),
    }
    if completed_stages is not None:
        payload["completed_stages"] = list(completed_stages)
    if error is not None:
        payload["error"] = error
    if extra:
        payload.update(extra)
    _write_json(status_path, payload)


def _hash_payload(*items: Any) -> str:
    digest = hashlib.sha256()
    for item in items:
        if item is None:
            digest.update(b"<none>")
            continue
        if isinstance(item, torch.Tensor):
            arr = item.detach().cpu().contiguous().numpy()
        elif isinstance(item, np.ndarray):
            arr = np.ascontiguousarray(item)
        else:
            digest.update(
                json.dumps(item, sort_keys=True, default=tensor_to_python).encode(
                    "utf-8"
                )
            )
            continue
        digest.update(str(arr.shape).encode("utf-8"))
        digest.update(str(arr.dtype).encode("utf-8"))
        digest.update(arr.tobytes())
    return digest.hexdigest()


def _dataset_artifact_metadata(
    *,
    role: str,
    bundle: Dict[str, Any],
    cache_path: Path,
    cache_hit: bool,
    cache_enabled: bool,
    config_hash: Optional[str],
) -> Dict[str, Any]:
    theta = bundle.get("theta")
    x = bundle.get("x")
    state0 = bundle.get("state0")
    acceptance = bundle.get("acceptance", {})
    artifact_hash = bundle.get("artifact_hash")
    if artifact_hash is None:
        artifact_hash = _hash_payload(theta, x, state0)

    metadata = {
        "role": role,
        "cache_path": str(cache_path),
        "cache_enabled": bool(cache_enabled),
        "cache_hit": bool(cache_hit),
        "config_hash": config_hash,
        "artifact_hash": str(artifact_hash),
        "num_examples": (
            int(theta.shape[0]) if isinstance(theta, torch.Tensor) else None
        ),
        "theta_shape": list(theta.shape) if isinstance(theta, torch.Tensor) else None,
        "x_shape": list(x.shape) if isinstance(x, torch.Tensor) else None,
        "state0_shape": (
            list(state0.shape) if isinstance(state0, torch.Tensor) else None
        ),
        "acceptance": acceptance,
    }
    cached_meta = bundle.get("cache_metadata")
    if isinstance(cached_meta, dict):
        metadata["cache_metadata"] = cached_meta
    return metadata


def _posterior_case_summary(
    *,
    theta_true_np: np.ndarray,
    theta_samples_np: np.ndarray,
    param_names: List[str],
) -> Dict[str, Any]:
    theta_true_np = np.asarray(theta_true_np, dtype=np.float64).reshape(-1)
    theta_samples_np = np.asarray(theta_samples_np, dtype=np.float64)
    if theta_samples_np.ndim != 2:
        theta_samples_np = theta_samples_np.reshape(-1, theta_samples_np.shape[-1])

    posterior_mean = theta_samples_np.mean(axis=0)
    posterior_std = theta_samples_np.std(
        axis=0, ddof=1 if theta_samples_np.shape[0] > 1 else 0
    )
    q05, q50, q95 = np.quantile(theta_samples_np, [0.05, 0.5, 0.95], axis=0)
    mean_abs_error = np.abs(posterior_mean - theta_true_np)
    sq_dists = np.sum((theta_samples_np - theta_true_np[None, :]) ** 2, axis=1)
    w2_pointmass = float(np.sqrt(np.mean(sq_dists)))
    l2_posterior_mean = float(np.linalg.norm(posterior_mean - theta_true_np))
    entropy_diag_gaussian = float(
        0.5 * np.sum(np.log(2.0 * np.pi * np.e * np.maximum(posterior_std**2, 1e-12)))
    )

    return {
        "theta_true": {
            name: float(value) for name, value in zip(param_names, theta_true_np)
        },
        "posterior_mean": {
            name: float(value) for name, value in zip(param_names, posterior_mean)
        },
        "posterior_std": {
            name: float(value) for name, value in zip(param_names, posterior_std)
        },
        "posterior_quantiles": {
            "q05": {name: float(value) for name, value in zip(param_names, q05)},
            "q50": {name: float(value) for name, value in zip(param_names, q50)},
            "q95": {name: float(value) for name, value in zip(param_names, q95)},
        },
        "mean_abs_error_per_parameter": {
            name: float(value) for name, value in zip(param_names, mean_abs_error)
        },
        "posterior_mean_l2_error": l2_posterior_mean,
        "posterior_w2_pointmass": w2_pointmass,
        "posterior_spread_l2": float(np.linalg.norm(posterior_std)),
        "posterior_entropy_diag_gaussian": entropy_diag_gaussian,
    }


def _figure_metadata_entry(
    *,
    figure_type: str,
    file_path: Path,
    example_idx: int,
    theta_true_np: np.ndarray,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    entry = {
        "figure_type": figure_type,
        "file_path": str(file_path),
        "example_idx": int(example_idx),
        "theta_true": np.asarray(theta_true_np, dtype=np.float32).reshape(-1).tolist(),
    }
    if extra:
        entry.update(extra)
    return entry


def _sample_theta_from_prior_cpu(prior_cpu, seed: int) -> torch.Tensor:
    """Sample one physical-theta draw deterministically on CPU."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return prior_cpu.sample((1,))


# =============================================================================
# Dataset Management (with caching for fair comparisons)
# =============================================================================


def _acquire_cache_lock(lock_path: Path, timeout_s: float = 1800.0) -> None:
    """Acquire a simple filesystem lock using exclusive file creation."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(str(os.getpid()))
            return
        except FileExistsError:
            if time.time() - start > timeout_s:
                raise TimeoutError(f"Timed out waiting for cache lock {lock_path}")
            time.sleep(0.5)


def _release_cache_lock(lock_path: Path) -> None:
    """Release a filesystem lock created by _acquire_cache_lock."""
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def _get_test_region_metadata(cfg: ExperimentConfig) -> Dict[str, Any]:
    """Describe the held-out synthetic test region."""
    bounds = cfg.param_bounds()
    theta_thresholds: Dict[str, Dict[str, float | str]] = {}
    frac = float(cfg.test_region_theta_tail_frac)
    for name in cfg.active_parameters:
        low, high = bounds[name]
        mean = 0.5 * (low + high)
        std = (high - low) / 6.0
        normal = torch.distributions.Normal(
            torch.tensor(mean, dtype=torch.float32),
            torch.tensor(std, dtype=torch.float32),
        )
        if name == "mu":
            quantile = frac
            threshold = float(normal.icdf(torch.tensor(quantile)).item())
            theta_thresholds[name] = {
                "direction": "lower",
                "quantile": float(quantile),
                "threshold": float(np.clip(threshold, low, high)),
            }
        else:
            quantile = 1.0 - frac
            threshold = float(normal.icdf(torch.tensor(quantile)).item())
            theta_thresholds[name] = {
                "direction": "upper",
                "quantile": float(quantile),
                "threshold": float(np.clip(threshold, low, high)),
            }

    speed_threshold = float(
        cfg.init_speed_center_ms
        + cfg.test_region_speed_margin_frac * cfg.init_speed_range_ms
    )
    return {
        "theta_tail_fraction": frac,
        "theta_thresholds": theta_thresholds,
        "require_joint_holdout": bool(cfg.test_region_require_joint_holdout),
        "driving_thresholds": {
            "initial_speed_ms_min": speed_threshold,
            "max_abs_steer_rad_min": float(
                np.deg2rad(cfg.test_region_min_abs_steer_deg) * cfg.steer_scale
            ),
            "max_brake_min": float(cfg.test_region_min_brake),
            "min_flags_required": int(cfg.test_region_min_driving_flags),
        },
        "channels": {
            "obs_v_body_x_idx": 1,
            "ctrl_steer_idx": int(cfg.obs_dim + 0),
            "ctrl_brake_idx": int(cfg.obs_dim + 2),
        },
    }


def _sample_theta_holdout_batch(
    cfg: ExperimentConfig,
    prior,
    region_meta: Dict[str, Any],
    num_samples: int,
) -> Tuple[torch.Tensor, int]:
    """Sample theta values from the prior conditioned on the parameter holdout region."""
    if num_samples <= 0:
        return torch.empty(0, cfg.active_param_dim(), dtype=torch.float32), 0

    theta_items: List[torch.Tensor] = []
    collected = 0
    attempted = 0
    proposal_batch = max(int(cfg.batch_sim), int(num_samples), 256)
    max_theta_attempts = max(int(num_samples) * 5000, proposal_batch)

    with torch.inference_mode():
        while collected < num_samples and attempted < max_theta_attempts:
            theta_prop = prior.sample((proposal_batch,)).to("cpu")
            attempted += int(theta_prop.shape[0])
            mask_theta = _theta_holdout_mask(cfg, theta_prop, region_meta)
            keep_idx = mask_theta.nonzero(as_tuple=False).squeeze(-1)
            if keep_idx.numel() == 0:
                continue
            take = min(int(keep_idx.numel()), num_samples - collected)
            theta_items.append(theta_prop[keep_idx[:take]])
            collected += take

    if collected < num_samples:
        raise RuntimeError(
            f"Failed to sample enough theta values inside the hold-out parameter region "
            f"({collected}/{num_samples} accepted after {attempted} theta draws)."
        )

    return torch.cat(theta_items, dim=0), attempted


def _theta_holdout_mask(
    cfg: ExperimentConfig,
    theta_phys: torch.Tensor,
    region_meta: Dict[str, Any],
) -> torch.Tensor:
    """Return boolean mask for the parameter-side holdout predicate."""
    if theta_phys.ndim != 2:
        raise ValueError(f"theta_phys must be (N,d), got {tuple(theta_phys.shape)}")
    mask = torch.ones(theta_phys.shape[0], dtype=torch.bool)
    theta_thresholds = region_meta["theta_thresholds"]
    for dim_idx, name in enumerate(cfg.active_parameters):
        spec = theta_thresholds.get(name)
        if spec is None:
            continue
        threshold = float(spec["threshold"])
        if spec["direction"] == "lower":
            mask &= theta_phys[:, dim_idx] <= threshold
        else:
            mask &= theta_phys[:, dim_idx] >= threshold
    return mask


def _driving_holdout_mask(
    cfg: ExperimentConfig,
    x_phys: torch.Tensor,
    region_meta: Dict[str, Any],
) -> torch.Tensor:
    """Return boolean mask for the driving-condition holdout predicate."""
    if x_phys.ndim != 3:
        raise ValueError(f"x_phys must be (N,T,D), got {tuple(x_phys.shape)}")

    thresholds = region_meta["driving_thresholds"]
    channels = region_meta["channels"]

    initial_speed = x_phys[:, 0, channels["obs_v_body_x_idx"]]
    max_abs_steer = x_phys[:, :, channels["ctrl_steer_idx"]].abs().amax(dim=1)
    max_brake = x_phys[:, :, channels["ctrl_brake_idx"]].amax(dim=1)

    flags = torch.stack(
        [
            initial_speed >= float(thresholds["initial_speed_ms_min"]),
            max_abs_steer >= float(thresholds["max_abs_steer_rad_min"]),
            max_brake >= float(thresholds["max_brake_min"]),
        ],
        dim=1,
    )
    return flags.sum(dim=1) >= int(thresholds["min_flags_required"])


def _joint_holdout_mask(
    cfg: ExperimentConfig,
    theta_phys: torch.Tensor,
    x_phys: torch.Tensor,
    region_meta: Dict[str, Any],
) -> torch.Tensor:
    """Return the joint held-out-region mask."""
    theta_mask = _theta_holdout_mask(cfg, theta_phys, region_meta)
    driving_mask = _driving_holdout_mask(cfg, x_phys, region_meta)
    if cfg.test_region_require_joint_holdout:
        return theta_mask & driving_mask
    return theta_mask | driving_mask


def _generate_train_dataset_with_holdout(
    cfg: ExperimentConfig,
    prior,
    simulator,
    region_meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate a training dataset excluding the held-out joint region."""
    if cfg.num_simulations <= 0:
        raise ValueError("cfg.num_simulations must be positive")

    N = int(cfg.num_simulations)
    batch = int(cfg.batch_sim)
    theta_items: List[torch.Tensor] = []
    x_items: List[torch.Tensor] = []
    generated = 0
    attempted = 0
    max_attempts = max(N * int(cfg.benchmark_max_attempt_factor), N)

    if cfg.jit_warmup:
        theta_w = prior.sample((min(8, N),)).to("cpu")
        _ = simulator(theta_w)

    pbar = tqdm(
        total=N,
        unit="sims",
        desc="Generating train split",
        dynamic_ncols=True,
    )
    with torch.inference_mode():
        while generated < N and attempted < max_attempts:
            b = min(batch, max(N - generated, batch))
            theta_b = prior.sample((b,)).to("cpu")
            x_b, _ = simulator(theta_b)
            x_b = x_b.cpu()
            mask_holdout = _joint_holdout_mask(cfg, theta_b, x_b, region_meta)
            keep_idx = (~mask_holdout).nonzero(as_tuple=False).squeeze(-1)
            if keep_idx.numel() > 0:
                take = min(int(keep_idx.numel()), N - generated)
                keep_idx = keep_idx[:take]
                theta_items.append(theta_b[keep_idx])
                x_items.append(x_b[keep_idx])
                generated += take
                pbar.update(take)
                pbar.set_postfix_str(f"{generated}/{N}")
            attempted += b
    pbar.close()

    if generated < N:
        raise RuntimeError(
            f"Failed to generate enough training samples outside the held-out region "
            f"({generated}/{N} accepted after {attempted} attempts)."
        )

    return {
        "theta": torch.cat(theta_items, dim=0),
        "x": torch.cat(x_items, dim=0),
        "region_metadata": region_meta,
        "acceptance": {
            "accepted": generated,
            "attempted": attempted,
            "acceptance_rate": float(generated / max(attempted, 1)),
        },
    }


def _generate_heldout_test_dataset(
    cfg: ExperimentConfig,
    prior,
    simulator,
    region_meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate a held-out test dataset inside the challenge region."""
    if cfg.num_test_simulations <= 0:
        return {
            "theta": torch.empty(0, cfg.active_param_dim(), dtype=torch.float32),
            "x": torch.empty(0, cfg.T_seg, cfg.obs_dim + 4, dtype=torch.float32),
            "state0": torch.empty(0, cfg.state_dim, dtype=torch.float32),
            "region_metadata": region_meta,
            "acceptance": {"accepted": 0, "attempted": 0, "acceptance_rate": 0.0},
        }

    N = int(cfg.num_test_simulations)
    batch = int(cfg.batch_sim)
    theta_items: List[torch.Tensor] = []
    x_items: List[torch.Tensor] = []
    state0_items: List[torch.Tensor] = []
    generated = 0
    attempted = 0
    theta_attempted = 0
    max_attempts = max(N * int(cfg.benchmark_max_attempt_factor), N)

    with torch.inference_mode():
        while generated < N and attempted < max_attempts:
            b = min(batch, max(N - generated, batch))
            theta_b, theta_draws = _sample_theta_holdout_batch(
                cfg, prior, region_meta, b
            )
            theta_attempted += theta_draws
            x_b, _ = simulator(theta_b)
            x_b = x_b.cpu()
            state0_b = getattr(simulator, "_last_state0_batch", None)
            mask_holdout = _joint_holdout_mask(cfg, theta_b, x_b, region_meta)
            keep_idx = mask_holdout.nonzero(as_tuple=False).squeeze(-1)
            if keep_idx.numel() > 0:
                take = min(int(keep_idx.numel()), N - generated)
                keep_idx = keep_idx[:take]
                theta_items.append(theta_b[keep_idx])
                x_items.append(x_b[keep_idx])
                if state0_b is not None:
                    state0_items.append(state0_b[keep_idx].cpu())
                generated += take
            attempted += b

    if generated < N:
        raise RuntimeError(
            f"Failed to generate enough held-out test samples inside the hold-out region "
            f"({generated}/{N} accepted after {attempted} attempts)."
        )

    return {
        "theta": torch.cat(theta_items, dim=0),
        "x": torch.cat(x_items, dim=0),
        "state0": (
            torch.cat(state0_items, dim=0)
            if state0_items
            else torch.empty(0, cfg.state_dim, dtype=torch.float32)
        ),
        "region_metadata": region_meta,
        "acceptance": {
            "accepted": generated,
            "attempted": attempted,
            "theta_draws": theta_attempted,
            "acceptance_rate": float(generated / max(attempted, 1)),
        },
    }


def _conditioning_x_on_device(x_cond: Any, device: torch.device) -> Any:
    """Move conditioning observations to the active device when they are tensors."""
    if isinstance(x_cond, torch.Tensor):
        return x_cond.to(device)
    return x_cond


def get_or_generate_training_dataset(
    cfg: ExperimentConfig,
    prior,
    simulator,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any], Dict[str, Any]]:
    """
    Get the shared training dataset from cache or generate a new one.

    Returns:
        theta_train: Training parameters (physical space)
        x_train: Training observations (physical space)
        metadata: Generation metadata including held-out-region thresholds
    """
    cache_path = cfg.get_dataset_cache_path()
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    region_meta = _get_test_region_metadata(cfg)

    if cfg.reuse_dataset and cache_path.exists():
        print(f"[DATA] Loading cached dataset from {cache_path}")
        cached = torch.load(cache_path, weights_only=False)
        meta = _dataset_artifact_metadata(
            role="training",
            bundle=cached,
            cache_path=cache_path,
            cache_hit=True,
            cache_enabled=bool(cfg.cache_dataset or cfg.reuse_dataset),
            config_hash=cfg.dataset_id,
        )
        return (
            cached["theta"],
            cached["x"],
            cached.get("region_metadata", region_meta),
            meta,
        )

    if cfg.cache_dataset or cfg.reuse_dataset:
        _acquire_cache_lock(lock_path)
        try:
            if cfg.reuse_dataset and cache_path.exists():
                print(f"[DATA] Loading cached dataset from {cache_path}")
                cached = torch.load(cache_path, weights_only=False)
                return (
                    cached["theta"],
                    cached["x"],
                    cached.get("region_metadata", region_meta),
                    _dataset_artifact_metadata(
                        role="training",
                        bundle=cached,
                        cache_path=cache_path,
                        cache_hit=True,
                        cache_enabled=bool(cfg.cache_dataset or cfg.reuse_dataset),
                        config_hash=cfg.dataset_id,
                    ),
                )
            print(f"[DATA] Generating {cfg.num_simulations} train simulations...")
            bundle = _generate_train_dataset_with_holdout(
                cfg, prior, simulator, region_meta
            )
            artifact_hash = _hash_payload(bundle["theta"], bundle["x"])
            bundle["artifact_hash"] = artifact_hash
            bundle["cache_metadata"] = {
                "created_at": datetime.now().isoformat(),
                "role": "training",
                "config_hash": cfg.dataset_id,
            }
            if cfg.cache_dataset:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                print(f"[DATA] Caching training dataset to {cache_path}")
                torch.save(
                    {
                        "theta": bundle["theta"],
                        "x": bundle["x"],
                        "region_metadata": bundle["region_metadata"],
                        "acceptance": bundle["acceptance"],
                        "config_hash": cfg.dataset_id,
                        "artifact_hash": artifact_hash,
                        "cache_metadata": bundle["cache_metadata"],
                    },
                    cache_path,
                )
            return (
                bundle["theta"],
                bundle["x"],
                bundle["region_metadata"],
                _dataset_artifact_metadata(
                    role="training",
                    bundle=bundle,
                    cache_path=cache_path,
                    cache_hit=False,
                    cache_enabled=bool(cfg.cache_dataset or cfg.reuse_dataset),
                    config_hash=cfg.dataset_id,
                ),
            )
        finally:
            _release_cache_lock(lock_path)

    print(f"[DATA] Generating {cfg.num_simulations} train simulations...")
    bundle = _generate_train_dataset_with_holdout(cfg, prior, simulator, region_meta)
    bundle["artifact_hash"] = _hash_payload(bundle["theta"], bundle["x"])
    return (
        bundle["theta"],
        bundle["x"],
        bundle["region_metadata"],
        _dataset_artifact_metadata(
            role="training",
            bundle=bundle,
            cache_path=cache_path,
            cache_hit=False,
            cache_enabled=bool(cfg.cache_dataset or cfg.reuse_dataset),
            config_hash=cfg.dataset_id,
        ),
    )


def get_or_generate_test_dataset(
    cfg: ExperimentConfig,
    prior,
    simulator,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any], torch.Tensor, Dict[str, Any]]:
    """
    Get the shared held-out synthetic test dataset from cache or generate it.

    Returns:
        theta_test: Held-out test parameters (physical space)
        x_test: Held-out test observations (physical space)
        metadata: Held-out region metadata and thresholds
        state0_test: True initial hidden simulator states for the held-out cases
    """
    if not cfg.run_simulated_test_eval:
        region_meta = _get_test_region_metadata(cfg)
        return (
            torch.empty(0, cfg.active_param_dim(), dtype=torch.float32),
            torch.empty(0, cfg.T_seg, cfg.obs_dim + 4, dtype=torch.float32),
            region_meta,
            torch.empty(0, cfg.state_dim, dtype=torch.float32),
            {
                "role": "heldout_test",
                "cache_enabled": bool(cfg.cache_dataset or cfg.reuse_dataset),
                "cache_hit": False,
                "cache_path": str(cfg.get_test_dataset_cache_path()),
                "config_hash": cfg.test_dataset_id,
                "acceptance": {"accepted": 0, "attempted": 0, "acceptance_rate": 0.0},
            },
        )

    cache_path = cfg.get_test_dataset_cache_path()
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    region_meta = _get_test_region_metadata(cfg)

    if cfg.reuse_dataset and cache_path.exists():
        print(f"[DATA] Loading cached held-out test dataset from {cache_path}")
        cached = torch.load(cache_path, weights_only=False)
        return (
            cached["theta"],
            cached["x"],
            cached.get("region_metadata", region_meta),
            cached.get("state0", torch.empty(0, cfg.state_dim, dtype=torch.float32)),
            _dataset_artifact_metadata(
                role="heldout_test",
                bundle=cached,
                cache_path=cache_path,
                cache_hit=True,
                cache_enabled=bool(cfg.cache_dataset or cfg.reuse_dataset),
                config_hash=cfg.test_dataset_id,
            ),
        )

    if cfg.cache_dataset or cfg.reuse_dataset:
        _acquire_cache_lock(lock_path)
        try:
            if cfg.reuse_dataset and cache_path.exists():
                print(f"[DATA] Loading cached held-out test dataset from {cache_path}")
                cached = torch.load(cache_path, weights_only=False)
                return (
                    cached["theta"],
                    cached["x"],
                    cached.get("region_metadata", region_meta),
                    cached.get(
                        "state0", torch.empty(0, cfg.state_dim, dtype=torch.float32)
                    ),
                    _dataset_artifact_metadata(
                        role="heldout_test",
                        bundle=cached,
                        cache_path=cache_path,
                        cache_hit=True,
                        cache_enabled=bool(cfg.cache_dataset or cfg.reuse_dataset),
                        config_hash=cfg.test_dataset_id,
                    ),
                )
            print(
                f"[DATA] Generating {cfg.num_test_simulations} held-out test simulations..."
            )
            original_seed = cfg.random_seed
            original_sim_seed = cfg.sim_seed
            original_batch_idx = getattr(simulator, "_batch_idx", 0)
            try:
                cfg.random_seed = cfg.benchmark_eval_seed
                cfg.sim_seed = cfg.benchmark_eval_seed
                setup_environment(cfg.benchmark_eval_seed)
                simulator._batch_idx = 0
                bundle = _generate_heldout_test_dataset(
                    cfg, prior, simulator, region_meta
                )
            finally:
                simulator._batch_idx = original_batch_idx
                cfg.random_seed = original_seed
                cfg.sim_seed = original_sim_seed
                setup_environment(cfg.sim_seed)
            artifact_hash = _hash_payload(
                bundle["theta"], bundle["x"], bundle["state0"]
            )
            bundle["artifact_hash"] = artifact_hash
            bundle["cache_metadata"] = {
                "created_at": datetime.now().isoformat(),
                "role": "heldout_test",
                "config_hash": cfg.test_dataset_id,
            }
            if cfg.cache_dataset:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                print(f"[DATA] Caching held-out test dataset to {cache_path}")
                torch.save(
                    {
                        "theta": bundle["theta"],
                        "x": bundle["x"],
                        "state0": bundle["state0"],
                        "region_metadata": bundle["region_metadata"],
                        "acceptance": bundle["acceptance"],
                        "config_hash": cfg.test_dataset_id,
                        "artifact_hash": artifact_hash,
                        "cache_metadata": bundle["cache_metadata"],
                    },
                    cache_path,
                )
            return (
                bundle["theta"],
                bundle["x"],
                bundle["region_metadata"],
                bundle["state0"],
                _dataset_artifact_metadata(
                    role="heldout_test",
                    bundle=bundle,
                    cache_path=cache_path,
                    cache_hit=False,
                    cache_enabled=bool(cfg.cache_dataset or cfg.reuse_dataset),
                    config_hash=cfg.test_dataset_id,
                ),
            )
        finally:
            _release_cache_lock(lock_path)

    original_seed = cfg.random_seed
    original_sim_seed = cfg.sim_seed
    original_batch_idx = getattr(simulator, "_batch_idx", 0)
    try:
        cfg.random_seed = cfg.benchmark_eval_seed
        cfg.sim_seed = cfg.benchmark_eval_seed
        setup_environment(cfg.benchmark_eval_seed)
        simulator._batch_idx = 0
        bundle = _generate_heldout_test_dataset(cfg, prior, simulator, region_meta)
    finally:
        simulator._batch_idx = original_batch_idx
        cfg.random_seed = original_seed
        cfg.sim_seed = original_sim_seed
        setup_environment(cfg.sim_seed)

    bundle["artifact_hash"] = _hash_payload(
        bundle["theta"], bundle["x"], bundle["state0"]
    )
    return (
        bundle["theta"],
        bundle["x"],
        bundle["region_metadata"],
        bundle["state0"],
        _dataset_artifact_metadata(
            role="heldout_test",
            bundle=bundle,
            cache_path=cache_path,
            cache_hit=False,
            cache_enabled=bool(cfg.cache_dataset or cfg.reuse_dataset),
            config_hash=cfg.test_dataset_id,
        ),
    )


def get_or_generate_dataset(
    cfg: ExperimentConfig,
    prior,
    simulator,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
    """
    Backwards-compatible wrapper around the shared training dataset cache.
    """
    theta_train, x_train, region_metadata, _ = get_or_generate_training_dataset(
        cfg, prior, simulator, device
    )
    return theta_train, x_train, region_metadata


# =============================================================================
# Diagnostics (shared across all methods)
# =============================================================================


def build_shared_diagnostic_examples(
    cfg: ExperimentConfig,
    *,
    prior,
    simulator,
    normalizer: Normalizer,
    device: torch.device,
    num_examples: int,
    seed_offset: int = 2000,
    method=None,
) -> List[Dict[str, Any]]:
    """
    Build a shared set of synthetic examples (theta_true, x_phys, x_cond) so all
    diagnostics refer to identical conditioning data for the same ex_idx.

    This avoids accidental apples/oranges comparisons when different diagnostics
    sample different synthetic examples under the same label "ex{idx}".

    Returns:
        List of dicts with keys:
          - ex_idx: int
          - theta_true: np.ndarray (physical units, shape (d_theta,))
          - x_phys: np.ndarray (physical units, shape (T, D_in))
          - x_cond: torch.Tensor (normalized, shape (1, T, D_in or obs_dim))
    """
    examples: List[Dict[str, Any]] = []

    # FNPE: generate via task's JAX prior + simulator for consistency
    if cfg.method == "fnpe" and method is not None and hasattr(method, "task"):
        import jax

        task = method.task
        jax_prior = task.get_prior()
        simulator_fn = task.get_simulator()

        key = jax.random.PRNGKey(int(cfg.random_seed) + int(seed_offset))

        for ex_idx in range(int(num_examples)):
            key, key_theta, key_sim = jax.random.split(key, 3)
            theta_true_jax = jax_prior.sample(key_theta, (1,))[0]
            theta_true_np = np.array(theta_true_jax).astype(np.float32)

            x_phys_jax = simulator_fn(key_sim, theta_true_jax, int(cfg.T_seg))
            x_phys_np = np.array(x_phys_jax).astype(np.float32)

            x_phys_t = torch.tensor(x_phys_np, dtype=torch.float32, device=device)
            if x_phys_t.ndim == 2:
                x_phys_t = x_phys_t.unsqueeze(0)  # (1, T, D)

            # Condition in normalized space (unified interface). Support both obs-only
            # and [obs||ctrl] layouts depending on how the simulator is configured.
            D = int(x_phys_t.shape[-1])
            if D == int(cfg.obs_dim):
                x_cond = (x_phys_t - normalizer.obs_mean) / (
                    normalizer.obs_std + normalizer.eps
                )
            else:
                x_cond = normalizer.normalize_x(x_phys_t, int(cfg.obs_dim))

            examples.append(
                {
                    "ex_idx": ex_idx,
                    "theta_true": theta_true_np,
                    "x_phys": x_phys_np,
                    "x_cond": x_cond,
                }
            )

        return examples

    # Standard path for NPE/NPSE (torch prior + simulator)
    prior_cpu = build_prior(cfg, torch.device("cpu"))
    old_batch_idx = getattr(simulator, "_batch_idx", 0)
    simulator._batch_idx = int(seed_offset)
    try:
        for ex_idx in range(int(num_examples)):
            theta_true = _sample_theta_from_prior_cpu(
                prior_cpu, int(cfg.random_seed) + int(seed_offset) + ex_idx
            ).to(device)
            theta_true_np = theta_true.detach().cpu().numpy()[0].astype(np.float32)

            sim_out = simulator(theta_true)
            x_sim = sim_out[0] if isinstance(sim_out, tuple) else sim_out
            x_phys_t = x_sim.to(device)
            x_cond = normalizer.normalize_x(x_phys_t, int(cfg.obs_dim)).to(device)

            examples.append(
                {
                    "ex_idx": ex_idx,
                    "theta_true": theta_true_np,
                    "x_phys": x_phys_t.detach().cpu().numpy()[0].astype(np.float32),
                    "x_cond": x_cond,
                }
            )
    finally:
        simulator._batch_idx = old_batch_idx

    return examples


def build_shared_eval_dataset(
    cfg: ExperimentConfig,
    *,
    prior,
    simulator,
    normalizer: Normalizer,
    device: torch.device,
    num_cases: int,
    seed_offset: int = 3000,
    method=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build a deterministic synthetic holdout set shared across methods.

    Returns:
        theta_eval_phys: (N, d_theta) in physical units on CPU
        x_eval_norm:     (N, T, D_in) normalized conditioning inputs on CPU
    """
    theta_items: List[torch.Tensor] = []
    x_items: List[torch.Tensor] = []

    if cfg.method == "fnpe" and method is not None and hasattr(method, "task"):
        import jax

        task = method.task
        jax_prior = task.get_prior()
        simulator_fn = task.get_simulator()
        key = jax.random.PRNGKey(int(cfg.random_seed) + int(seed_offset))

        for _ in range(int(num_cases)):
            key, key_theta, key_sim = jax.random.split(key, 3)
            theta_true_jax = jax_prior.sample(key_theta, (1,))[0]
            theta_true_np = np.array(theta_true_jax).astype(np.float32)
            x_phys_jax = simulator_fn(key_sim, theta_true_jax, int(cfg.T_seg))
            x_phys_np = np.array(x_phys_jax).astype(np.float32)

            x_phys_t = torch.tensor(x_phys_np, dtype=torch.float32, device=device)
            if x_phys_t.ndim == 2:
                x_phys_t = x_phys_t.unsqueeze(0)

            D = int(x_phys_t.shape[-1])
            if D == int(cfg.obs_dim):
                x_norm = (x_phys_t - normalizer.obs_mean) / (
                    normalizer.obs_std + normalizer.eps
                )
            else:
                x_norm = normalizer.normalize_x(x_phys_t, int(cfg.obs_dim))

            theta_items.append(torch.tensor(theta_true_np, dtype=torch.float32))
            x_items.append(x_norm.squeeze(0).detach().cpu())

        return torch.stack(theta_items, dim=0), torch.stack(x_items, dim=0)

    prior_cpu = build_prior(cfg, torch.device("cpu"))
    old_batch_idx = getattr(simulator, "_batch_idx", 0)
    simulator._batch_idx = int(seed_offset)
    try:
        for idx in range(int(num_cases)):
            theta_true = _sample_theta_from_prior_cpu(
                prior_cpu, int(cfg.random_seed) + int(seed_offset) + idx
            ).to(device)
            sim_out = simulator(theta_true)
            x_phys_t = sim_out[0] if isinstance(sim_out, tuple) else sim_out
            x_norm = normalizer.normalize_x(x_phys_t.to(device), int(cfg.obs_dim))

            theta_items.append(theta_true.squeeze(0).detach().cpu())
            x_items.append(x_norm.squeeze(0).detach().cpu())
    finally:
        simulator._batch_idx = old_batch_idx

    return torch.stack(theta_items, dim=0), torch.stack(x_items, dim=0)


def build_examples_from_dataset(
    theta_phys: torch.Tensor,
    x_phys: torch.Tensor,
    x_norm: torch.Tensor,
    *,
    num_examples: int,
) -> List[Dict[str, Any]]:
    """Build diagnostic example objects from a cached held-out dataset."""
    examples: List[Dict[str, Any]] = []
    total = min(int(num_examples), int(theta_phys.shape[0]), int(x_norm.shape[0]))
    for ex_idx in range(total):
        examples.append(
            {
                "ex_idx": ex_idx,
                "theta_true": theta_phys[ex_idx]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32),
                "x_phys": x_phys[ex_idx].detach().cpu().numpy().astype(np.float32),
                "x_cond": x_norm[ex_idx : ex_idx + 1],
            }
        )
    return examples


def _controls_dict_from_x_case(
    x_case_phys: torch.Tensor,
    obs_dim: int,
) -> Dict[str, Any]:
    """Extract simulator-style control dict from a single [obs||ctrl] trajectory."""
    if x_case_phys.ndim != 2:
        raise ValueError(f"x_case_phys must be (T,D), got {tuple(x_case_phys.shape)}")
    ctrl = (
        x_case_phys[:, obs_dim : obs_dim + 4].detach().cpu().numpy().astype(np.float32)
    )
    import jax.numpy as jnp

    return {
        "steer_ang": jnp.asarray(ctrl[:, 0]),
        "engine_torque": jnp.asarray(ctrl[:, 1]),
        "break_torque": jnp.asarray(ctrl[:, 2]),
        "gear_transmission": jnp.asarray(ctrl[:, 3]),
    }


def sample_posterior_on_dataset(
    cfg: ExperimentConfig,
    theta_test_phys: torch.Tensor,
    x_test_norm: torch.Tensor,
    posterior,
    normalizer: Normalizer,
    device: torch.device,
    *,
    num_cases: int,
    num_posterior_samples: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
    """Sample posteriors for a dataset of conditioning observations."""
    N = min(int(num_cases), int(theta_test_phys.shape[0]), int(x_test_norm.shape[0]))
    theta_sub_phys = theta_test_phys[:N]
    x_sub = x_test_norm[:N]
    all_samples: List[torch.Tensor] = []
    sample_times_s: List[float] = []
    successful_theta: List[torch.Tensor] = []

    for i in range(N):
        x_i = x_sub[i : i + 1]
        try:
            t0 = time.time()
            samples_i = posterior.sample((num_posterior_samples,), x=x_i.to(device))
            sample_times_s.append(time.time() - t0)
            if isinstance(samples_i, np.ndarray):
                samples_i = torch.from_numpy(samples_i).float()
            samples_i = normalizer.unnormalize_theta(samples_i.to(device)).cpu()
            if samples_i.ndim == 3:
                samples_i = samples_i.reshape(-1, samples_i.shape[-1])
            all_samples.append(samples_i)
            successful_theta.append(theta_sub_phys[i].detach().cpu())
        except Exception as e:
            print(f"[POST] Warning: Failed to sample case {i}: {e}")

    if not all_samples:
        raise RuntimeError("No successful posterior samples on held-out dataset")

    return (
        torch.stack(successful_theta, dim=0),
        torch.stack(all_samples, dim=0),
        sample_times_s,
    )


def compute_heldout_posterior_stats(
    cfg: ExperimentConfig,
    theta_true_phys: torch.Tensor,
    theta_samples_phys: torch.Tensor,
    *,
    bootstrap_samples: int = 1000,
) -> Dict[str, Any]:
    """Compute formal held-out posterior statistics."""
    theta_true_np = theta_true_phys.detach().cpu().numpy().astype(np.float64)
    theta_samples_np = theta_samples_phys.detach().cpu().numpy().astype(np.float64)
    N, K, d = theta_samples_np.shape

    sq_dist = np.sum((theta_samples_np - theta_true_np[:, None, :]) ** 2, axis=2)
    w2_per_case = np.sqrt(np.mean(sq_dist, axis=1))
    posterior_mean = theta_samples_np.mean(axis=1)
    posterior_std = theta_samples_np.std(axis=1, ddof=1 if K > 1 else 0)
    q05 = np.quantile(theta_samples_np, 0.05, axis=1)
    q50 = np.quantile(theta_samples_np, 0.50, axis=1)
    q95 = np.quantile(theta_samples_np, 0.95, axis=1)
    l2_per_case = np.sqrt(np.sum((posterior_mean - theta_true_np) ** 2, axis=1))
    mean_abs_error = np.abs(posterior_mean - theta_true_np)
    spread_l2 = np.linalg.norm(posterior_std, axis=1)
    entropy_diag_gaussian = 0.5 * np.sum(
        np.log(2.0 * np.pi * np.e * np.maximum(posterior_std**2, 1e-12)),
        axis=1,
    )

    rng = np.random.default_rng(int(cfg.random_seed))
    boot_idx = rng.integers(0, N, size=(bootstrap_samples, N))
    w2_boot = w2_per_case[boot_idx].mean(axis=1)
    l2_boot = l2_per_case[boot_idx].mean(axis=1)

    rank_uniformity: Dict[str, Any] = {}
    for dim_idx, name in enumerate(cfg.active_parameters):
        ranks = np.sum(
            theta_samples_np[:, :, dim_idx] < theta_true_np[:, None, dim_idx], axis=1
        )
        rank_scaled = (ranks + 1.0) / (K + 1.0)
        ks_result = kstest(rank_scaled, "uniform")
        rank_uniformity[name] = {
            "ks_statistic": float(ks_result.statistic),
            "ks_pvalue": float(ks_result.pvalue),
            "mean_rank": float(np.mean(ranks)),
        }

    coverage_tests: Dict[str, Any] = {}
    for level in (0.5, 0.9):
        alpha = 1.0 - level
        lower = np.quantile(theta_samples_np, alpha / 2.0, axis=1)
        upper = np.quantile(theta_samples_np, 1.0 - alpha / 2.0, axis=1)
        level_key = f"coverage_{int(level * 100)}"
        coverage_tests[level_key] = {}
        overall_hits = []
        for dim_idx, name in enumerate(cfg.active_parameters):
            hits = (theta_true_np[:, dim_idx] >= lower[:, dim_idx]) & (
                theta_true_np[:, dim_idx] <= upper[:, dim_idx]
            )
            overall_hits.append(hits)
            test = binomtest(int(hits.sum()), int(hits.size), p=level)
            coverage_tests[level_key][name] = {
                "empirical": float(np.mean(hits)),
                "count": int(hits.sum()),
                "n": int(hits.size),
                "pvalue": float(test.pvalue),
            }
        overall_flat = np.concatenate(overall_hits)
        overall_test = binomtest(
            int(overall_flat.sum()), int(overall_flat.size), p=level
        )
        coverage_tests[level_key]["overall"] = {
            "empirical": float(np.mean(overall_flat)),
            "count": int(overall_flat.sum()),
            "n": int(overall_flat.size),
            "pvalue": float(overall_test.pvalue),
            "abs_error_to_nominal": float(abs(np.mean(overall_flat) - level)),
        }

    return {
        "rank_uniformity": rank_uniformity,
        "coverage_tests": coverage_tests,
        "coverage_distance_summary": {
            "coverage_50_abs_error": float(
                abs(coverage_tests["coverage_50"]["overall"]["empirical"] - 0.50)
            ),
            "coverage_90_abs_error": float(
                abs(coverage_tests["coverage_90"]["overall"]["empirical"] - 0.90)
            ),
        },
        "bootstrap_ci": {
            "w2_mean_95": [
                float(np.quantile(w2_boot, 0.025)),
                float(np.quantile(w2_boot, 0.975)),
            ],
            "l2_error_mean_95": [
                float(np.quantile(l2_boot, 0.025)),
                float(np.quantile(l2_boot, 0.975)),
            ],
        },
        "per_case": {
            "w2": [float(v) for v in w2_per_case],
            "l2_error": [float(v) for v in l2_per_case],
            "posterior_mean": posterior_mean.tolist(),
            "posterior_std": posterior_std.tolist(),
            "posterior_quantiles": {
                "q05": q05.tolist(),
                "q50": q50.tolist(),
                "q95": q95.tolist(),
            },
            "posterior_mean_abs_error": mean_abs_error.tolist(),
            "posterior_spread_l2": [float(v) for v in spread_l2],
            "posterior_entropy_diag_gaussian": [
                float(v) for v in entropy_diag_gaussian
            ],
        },
        "num_cases": int(N),
        "num_posterior_samples": int(K),
    }


def run_simulated_ppc_diagnostic(
    cfg: ExperimentConfig,
    fig_dir: Path,
    posterior,
    x_test_phys: torch.Tensor,
    theta_test_phys: Optional[torch.Tensor],
    state0_test_phys: Optional[torch.Tensor],
    normalizer: Normalizer,
    device: torch.device,
    *,
    num_examples: int,
    num_plot_examples: int,
    num_posterior_samples: int,
) -> Dict[str, Any]:
    """Run simulated PPC on held-out synthetic test trajectories."""
    total = min(int(num_examples), int(x_test_phys.shape[0]))
    if total <= 0:
        return {}

    per_example: List[Dict[str, Any]] = []
    figure_metadata: List[Dict[str, Any]] = []
    ppc_times: List[float] = []
    for ex_idx in range(total):
        x_case = x_test_phys[ex_idx : ex_idx + 1].to(device)
        controls = _controls_dict_from_x_case(x_test_phys[ex_idx], cfg.obs_dim)
        state0_case = None
        if state0_test_phys is not None and ex_idx < int(state0_test_phys.shape[0]):
            state0_case = state0_test_phys[ex_idx].detach().cpu().numpy()
        y_real, y_ppc, ppc_meta = posterior_predictive_from_real(
            posterior,
            x_case,
            controls,
            cfg,
            normalizer=normalizer,
            device=device,
            K_ppc=num_posterior_samples,
            state0=state0_case,
            return_metadata=True,
        )
        ppc_times.append(
            float(ppc_meta["posterior_sampling_time_s"])
            + float(ppc_meta["ppc_simulation_time_s"])
        )
        theta_true_np = None
        if theta_test_phys is not None and ex_idx < int(theta_test_phys.shape[0]):
            theta_true_np = (
                theta_test_phys[ex_idx].detach().cpu().numpy().astype(np.float32)
            )
        posterior_summary = (
            _posterior_case_summary(
                theta_true_np=theta_true_np,
                theta_samples_np=np.asarray(
                    ppc_meta["posterior_samples_phys"], dtype=np.float32
                ),
                param_names=list(cfg.active_parameters),
            )
            if theta_true_np is not None
            else None
        )
        metrics = real_data_trajectory_metrics(y_real, y_ppc, normalize_w2=True)
        plot_path = None
        if not cfg.no_plots and ex_idx < int(num_plot_examples):
            ppc_path = fig_dir / f"ppc_timeseries_simulated_test_ex{ex_idx}.png"
            plot_ppc_trajectories(
                y_real=y_real,
                y_ppc=y_ppc,
                obs_labels=OBS_LABELS,
                dt=cfg.dt,
                out_path=ppc_path,
                max_trajs=20,
                plot_all_trajs=True,
                max_dims=cfg.obs_dim,
                title=f"PPC on Held-out Simulated Test Case {ex_idx}",
            )
            plot_path = ppc_path
            if theta_true_np is not None:
                figure_metadata.append(
                    _figure_metadata_entry(
                        figure_type="heldout_simulated_ppc",
                        file_path=ppc_path,
                        example_idx=ex_idx,
                        theta_true_np=theta_true_np,
                        extra={
                            "ppc_rmse": float(metrics["rmse_overall"]),
                            "ppc_w2": float(metrics["w2"]),
                            "difficulty_score": float(metrics["rmse_overall"]),
                        },
                    )
                )
        if plot_path is not None and theta_true_np is not None:
            figure_metadata[-1]["ppc_rmse"] = float(metrics["rmse_overall"])
            figure_metadata[-1]["ppc_w2"] = float(metrics["w2"])
        per_example.append(
            {
                "example_idx": ex_idx,
                "theta_true": (
                    theta_true_np.tolist() if theta_true_np is not None else None
                ),
                "metrics": metrics,
                "posterior_summary": posterior_summary,
                "timing_s": {
                    "posterior_sampling_time_s": float(
                        ppc_meta["posterior_sampling_time_s"]
                    ),
                    "ppc_simulation_time_s": float(ppc_meta["ppc_simulation_time_s"]),
                    "total_case_time_s": float(
                        ppc_meta["posterior_sampling_time_s"]
                        + ppc_meta["ppc_simulation_time_s"]
                    ),
                },
                "figure_path": str(plot_path) if plot_path is not None else None,
            }
        )

    rmse_vals = [float(item["metrics"]["rmse_overall"]) for item in per_example]
    w2_vals = [float(item["metrics"]["w2"]) for item in per_example]
    aggregate = {
        "num_examples": total,
        "num_plot_examples": int(min(total, num_plot_examples)),
        "num_posterior_samples": int(num_posterior_samples),
        "rmse_mean": float(np.mean(rmse_vals)),
        "rmse_std": float(np.std(rmse_vals)),
        "w2_mean": float(np.mean(w2_vals)),
        "w2_std": float(np.std(w2_vals)),
        "ppc_time_mean_s": float(np.mean(ppc_times)) if ppc_times else None,
        "ppc_time_total_s": float(np.sum(ppc_times)) if ppc_times else None,
    }
    return {
        "per_example": per_example,
        "per_case": per_example,
        "aggregate": aggregate,
        "figure_metadata": figure_metadata,
    }


def build_budget_metadata(
    cfg: ExperimentConfig, training_summary: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Compute benchmark budget metadata for reporting and aggregation."""
    cfg.assert_budget_resolution_ready("build_budget_metadata")
    training_summary = training_summary or {}
    if cfg.method == "fnpe":
        num_pilots = int(round(cfg.fnpe_num_simulations * cfg.fnpe_pilot_fraction))
        effective_budget_steps = int(
            cfg.fnpe_num_simulations * cfg.fnpe_window_size
            + num_pilots * cfg.fnpe_pilot_length
        )
    else:
        num_pilots = 0
        effective_budget_steps = int(cfg.num_simulations * cfg.T_seg)

    requested_budget_steps = (
        int(cfg.requested_budget_steps)
        if cfg.requested_budget_steps is not None
        else effective_budget_steps
    )

    metadata = {
        "method": cfg.method,
        "num_simulations": int(cfg.num_simulations),
        "derived_num_simulations": int(cfg.derived_num_simulations),
        "training_num_simulations": int(
            cfg.fnpe_num_simulations if cfg.method == "fnpe" else cfg.num_simulations
        ),
        "num_test_simulations": int(cfg.num_test_simulations),
        "T_seg": int(cfg.T_seg),
        "active_parameters": list(cfg.active_parameters),
        "requested_budget_steps": requested_budget_steps,
        "effective_budget_steps": effective_budget_steps,
        "total_simulation_budget_steps": effective_budget_steps,
        "budget_match_ratio": (
            float(effective_budget_steps / requested_budget_steps)
            if requested_budget_steps > 0
            else None
        ),
        "fnpe_num_pilot_simulations": int(num_pilots),
        "benchmark_eval_seed": int(cfg.benchmark_eval_seed),
    }
    if training_summary:
        metadata["training_batch_size"] = training_summary.get("training_batch_size")
        metadata["num_train_steps"] = training_summary.get("num_train_steps")
        metadata["optimizer_examples_seen"] = training_summary.get(
            "optimizer_examples_seen"
        )
    if cfg.method == "simformer":
        metadata.update(
            {
                "optimizer_budget_steps": training_summary.get("num_train_steps"),
                "simformer_num_timepoints": getattr(
                    cfg, "simformer_num_timepoints", None
                ),
                "simformer_best_validation_step": training_summary.get(
                    "best_validation_step"
                ),
                "simformer_best_validation_loss": training_summary.get(
                    "best_validation_loss"
                ),
            }
        )
    elif cfg.method == "fnpe":
        metadata.update(
            {
                "optimizer_budget_steps": training_summary.get(
                    "total_optimizer_updates"
                ),
                "fnpe_num_outer_epochs": training_summary.get("num_outer_epochs"),
                "fnpe_num_inner_epochs": training_summary.get("num_inner_epochs"),
                "fnpe_best_validation_loss": training_summary.get(
                    "best_validation_loss"
                ),
            }
        )
    return metadata


def run_parameter_posterior_plots(
    cfg: ExperimentConfig,
    fig_dir: Path,
    prior,
    posterior,
    simulator,
    normalizer: Normalizer,
    device: torch.device,
    *,
    num_examples: int = 3,
    num_prior_samples: int = 5000,  # 5k is plenty for a histogram; was 20k
    num_posterior_samples: int = 5000,
    method=None,  # Pass method for FNPE to use its own data generation
    examples: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Generate prior vs posterior marginal plots."""
    if cfg.no_plots:
        return []

    param_names = list(cfg.active_parameters)
    figure_metadata: List[Dict[str, Any]] = []

    # For FNPE, use JAX-based prior sampling and simulator but unified posterior interface
    if cfg.method == "fnpe" and method is not None and hasattr(method, "task"):
        import jax

        task = method.task
        jax_prior = task.get_prior()
        key = jax.random.PRNGKey(cfg.random_seed + 1000)

        # Sample from JAX prior for reference (physical units)
        key, key_prior = jax.random.split(key)
        prior_pool_jax = jax_prior.sample(key_prior, (num_prior_samples,))
        prior_pool_np = np.array(prior_pool_jax)

        if examples is None:
            raise ValueError(
                "[DIAG] examples must be provided for consistent diagnostics. "
                "Use build_shared_diagnostic_examples() in run_experiment()."
            )

        for ex in examples[:num_examples]:
            ex_idx = int(ex["ex_idx"])
            print(f"[DIAG] Posterior plots (FNPE): example {ex_idx+1}/{num_examples}")

            theta_true_np = np.asarray(ex["theta_true"], dtype=np.float32).reshape(-1)
            x_norm = ex["x_cond"]

            print(
                f"[DIAG] Sampling {num_posterior_samples} posterior samples...",
                flush=True,
            )
            theta_post_norm = posterior.sample((num_posterior_samples,), x=x_norm)

            theta_post_phys = normalizer.unnormalize_theta(theta_post_norm)
            theta_post_np = theta_post_phys.detach().cpu().numpy()
            print(
                f"[DIAG] theta_post shape: {theta_post_np.shape}, "
                f"NaN count: {np.sum(np.isnan(theta_post_np))}",
                flush=True,
            )
            if theta_post_np.ndim == 3:
                theta_post_np = theta_post_np.reshape(-1, theta_post_np.shape[-1])

            # One grid figure per example (all params) — much faster than per-param saves
            grid_path = fig_dir / f"prior_posterior_grid_ex{ex_idx}.png"
            plot_prior_posterior_grid(
                prior_np=prior_pool_np,
                post_np=theta_post_np,
                theta_true_np=theta_true_np,
                param_names=param_names,
                out_path=grid_path,
                example_id=f"ex{ex_idx}",
            )
            posterior_summary = _posterior_case_summary(
                theta_true_np=theta_true_np,
                theta_samples_np=theta_post_np,
                param_names=param_names,
            )
            figure_metadata.append(
                _figure_metadata_entry(
                    figure_type="prior_posterior_grid",
                    file_path=grid_path,
                    example_idx=ex_idx,
                    theta_true_np=theta_true_np,
                    extra={
                        "difficulty_score_l2": posterior_summary[
                            "posterior_mean_l2_error"
                        ],
                        "difficulty_score_w2": posterior_summary[
                            "posterior_w2_pointmass"
                        ],
                        "posterior_summary": posterior_summary,
                    },
                )
            )
        return figure_metadata

    # Standard path for NPE/NPSE
    with torch.no_grad():
        prior_pool = prior.sample((num_prior_samples,)).to(device)
    prior_pool_np = prior_pool.detach().cpu().numpy()

    if examples is None:
        raise ValueError(
            "[DIAG] examples must be provided for consistent diagnostics. "
            "Use build_shared_diagnostic_examples() in run_experiment()."
        )

    for ex in examples[:num_examples]:
        ex_idx = int(ex["ex_idx"])
        print(f"[DIAG] Posterior plots: example {ex_idx+1}/{num_examples}")

        theta_true_np = np.asarray(ex["theta_true"], dtype=np.float32).reshape(-1)
        x_cond = _conditioning_x_on_device(ex["x_cond"], device)

        with torch.no_grad():
            theta_post_norm = posterior.sample((num_posterior_samples,), x=x_cond)

        if isinstance(theta_post_norm, np.ndarray):
            theta_post_np = theta_post_norm
        else:
            theta_post_np = (
                normalizer.unnormalize_theta(theta_post_norm).detach().cpu().numpy()
            )

        if theta_post_np.ndim == 3:
            theta_post_np = theta_post_np.reshape(-1, theta_post_np.shape[-1])

        # One grid figure per example (all params) — much faster than per-param saves
        grid_path = fig_dir / f"prior_posterior_grid_ex{ex_idx}.png"
        plot_prior_posterior_grid(
            prior_np=prior_pool_np,
            post_np=theta_post_np,
            theta_true_np=theta_true_np,
            param_names=param_names,
            out_path=grid_path,
            example_id=f"ex{ex_idx}",
        )
        posterior_summary = _posterior_case_summary(
            theta_true_np=theta_true_np,
            theta_samples_np=theta_post_np,
            param_names=param_names,
        )
        figure_metadata.append(
            _figure_metadata_entry(
                figure_type="prior_posterior_grid",
                file_path=grid_path,
                example_idx=ex_idx,
                theta_true_np=theta_true_np,
                extra={
                    "difficulty_score_l2": posterior_summary["posterior_mean_l2_error"],
                    "difficulty_score_w2": posterior_summary["posterior_w2_pointmass"],
                    "posterior_summary": posterior_summary,
                },
            )
        )

    return figure_metadata


def run_sbc_diagnostic(
    cfg: ExperimentConfig,
    fig_dir: Path,
    prior_norm,
    posterior,
    simulator,
    simulator_for_sbi,
    normalizer: Normalizer,
    device: torch.device,
) -> Dict[str, Any]:
    """Run Simulation-Based Calibration diagnostic."""
    num_sbc = cfg.num_sbc_samples
    num_post = cfg.num_posterior_samples_sbc

    if not getattr(cfg, "unify_eval_budgets", False):
        if cfg.method == "fnpe":
            num_sbc = min(num_sbc, 10)
            num_post = min(num_post, 50)
        elif cfg.method == "npse":
            num_sbc = min(num_sbc, 100)
            num_post = min(num_post, 500)
        elif cfg.method == "simformer":
            # Simformer uses JAX SDE integration (slow on CPU fallback); keep small
            num_sbc = min(num_sbc, 50)
            num_post = min(num_post, 500)

    print(f"[SBC] Running with {num_sbc} samples, {num_post} posterior samples each...")

    sbc_device = device
    if hasattr(posterior, "to"):
        posterior.to(sbc_device)

    cuda_devices = []
    if sbc_device.type == "cuda":
        cuda_devices = [
            (
                sbc_device.index
                if sbc_device.index is not None
                else torch.cuda.current_device()
            )
        ]

    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(int(cfg.random_seed) + 9000)
        if sbc_device.type == "cuda":
            torch.cuda.manual_seed_all(int(cfg.random_seed) + 9000)
        if cfg.method == "fnpe":
            theta_sbc_phys = prior_norm.sample((num_sbc,)).to(device)
            theta_sbc_eval = normalizer.normalize_theta(theta_sbc_phys).to(sbc_device)
        else:
            theta_sbc_eval = prior_norm.sample((num_sbc,)).to(sbc_device)
            theta_sbc_phys = normalizer.unnormalize_theta(theta_sbc_eval)

    old_batch_idx = getattr(simulator, "_batch_idx", 0)
    simulator._batch_idx = 9000
    try:
        x_sbc_phys = simulator_for_sbi(theta_sbc_phys)
    finally:
        simulator._batch_idx = old_batch_idx
    x_sbc_norm = normalizer.normalize_x(x_sbc_phys, cfg.obs_dim)
    x_sbc_eval = x_sbc_norm.to(sbc_device)

    ranks, dap_samples_norm = run_sbc(
        thetas=theta_sbc_eval,
        xs=x_sbc_eval,
        posterior=posterior,
        num_posterior_samples=num_post,
        num_workers=1,  # Use 1 worker to avoid multiprocessing issues with JAX
        use_batched_sampling=False,  # FNPE doesn't support batched sampling
    )

    check_stats = check_sbc(ranks, theta_sbc_eval, dap_samples_norm, num_post)
    print("SBC check statistics:", check_stats)

    if hasattr(posterior, "to") and sbc_device != device:
        posterior.to(device)

    if not cfg.no_plots:
        sbc_fig_path = fig_dir / "sbc_rank_hist.png"
        plot_sbc_rank_hist(ranks, num_post, sbc_fig_path)

    return {
        "check_stats": {k: tensor_to_python(v) for k, v in check_stats.items()},
        "dap_samples_norm": dap_samples_norm,
    }


def run_pairplot_diagnostic(
    cfg: ExperimentConfig,
    fig_dir: Path,
    prior,
    posterior,
    simulator,
    normalizer: Normalizer,
    device: torch.device,
    *,
    num_posterior_samples: int = 1000,
    num_examples: int = 3,
    method=None,
    examples: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Run pairplot visualization diagnostic.

    Generates pairplot showing 2D marginal distributions (like markovsbi notebooks).

    Args:
        cfg: Experiment configuration
        fig_dir: Directory to save figures
        prior: Prior distribution
        posterior: Trained posterior
        simulator: Simulator function
        normalizer: Data normalizer
        device: Torch device
        num_posterior_samples: Number of posterior samples
        num_examples: Number of observation examples to generate
        method: The method object (for FNPE-specific handling)
    """
    if cfg.no_plots:
        return []

    print("[PAIRPLOT] Generating pairplot visualizations...")

    param_names = list(cfg.active_parameters)
    figure_metadata: List[Dict[str, Any]] = []

    # Outside strict comparison mode, keep slower methods lighter.
    if not getattr(cfg, "unify_eval_budgets", False) and cfg.method in ["npse", "fnpe"]:
        num_posterior_samples = min(num_posterior_samples, 500)
        num_examples = min(num_examples, 2)

    # For FNPE, use JAX-based data generation
    if cfg.method == "fnpe" and method is not None and hasattr(method, "task"):
        if examples is None:
            raise ValueError(
                "[PAIRPLOT] examples must be provided for consistent diagnostics. "
                "Use build_shared_diagnostic_examples() in run_experiment()."
            )

        for ex in examples[:num_examples]:
            ex_idx = int(ex["ex_idx"])
            print(f"[PAIRPLOT] Example {ex_idx+1}/{num_examples} (FNPE)")

            theta_true_np = np.asarray(ex["theta_true"], dtype=np.float32).reshape(-1)
            x_norm = ex["x_cond"]

            # Sample from posterior
            theta_post_norm = posterior.sample((num_posterior_samples,), x=x_norm)
            theta_post_phys = normalizer.unnormalize_theta(theta_post_norm)
            theta_post_np = theta_post_phys.detach().cpu().numpy()

            if theta_post_np.ndim == 3:
                theta_post_np = theta_post_np.reshape(-1, theta_post_np.shape[-1])

            # Pairplot
            pairplot_path = fig_dir / f"pairplot_ex{ex_idx}.png"
            plot_pairplot(
                samples=theta_post_np,
                out_path=pairplot_path,
                theta_true=theta_true_np,
                param_names=param_names,
                title="Posterior Pairplot",
                example_id=f"ex{ex_idx}",
            )
            posterior_summary = _posterior_case_summary(
                theta_true_np=theta_true_np,
                theta_samples_np=theta_post_np,
                param_names=param_names,
            )
            figure_metadata.append(
                _figure_metadata_entry(
                    figure_type="pairplot",
                    file_path=pairplot_path,
                    example_idx=ex_idx,
                    theta_true_np=theta_true_np,
                    extra={
                        "difficulty_score_l2": posterior_summary[
                            "posterior_mean_l2_error"
                        ],
                        "difficulty_score_w2": posterior_summary[
                            "posterior_w2_pointmass"
                        ],
                        "posterior_summary": posterior_summary,
                    },
                )
            )
        return figure_metadata

    # Standard path for NPE/NPSE
    if examples is None:
        raise ValueError(
            "[PAIRPLOT] examples must be provided for consistent diagnostics. "
            "Use build_shared_diagnostic_examples() in run_experiment()."
        )

    for ex in examples[:num_examples]:
        ex_idx = int(ex["ex_idx"])
        print(f"[PAIRPLOT] Example {ex_idx+1}/{num_examples}")

        theta_true_np = np.asarray(ex["theta_true"], dtype=np.float32).reshape(-1)
        x_cond = _conditioning_x_on_device(ex["x_cond"], device)

        with torch.no_grad():
            theta_post_norm = posterior.sample((num_posterior_samples,), x=x_cond)

        # Handle different return types
        if isinstance(theta_post_norm, np.ndarray):
            theta_post_np = theta_post_norm
        else:
            theta_post_np = (
                normalizer.unnormalize_theta(theta_post_norm).detach().cpu().numpy()
            )

        if theta_post_np.ndim == 3:
            theta_post_np = theta_post_np.reshape(-1, theta_post_np.shape[-1])

        # Pairplot
        pairplot_path = fig_dir / f"pairplot_ex{ex_idx}.png"
        plot_pairplot(
            samples=theta_post_np,
            out_path=pairplot_path,
            theta_true=theta_true_np,
            param_names=param_names,
            title="Posterior Pairplot",
            example_id=f"ex{ex_idx}",
        )
        posterior_summary = _posterior_case_summary(
            theta_true_np=theta_true_np,
            theta_samples_np=theta_post_np,
            param_names=param_names,
        )
        figure_metadata.append(
            _figure_metadata_entry(
                figure_type="pairplot",
                file_path=pairplot_path,
                example_idx=ex_idx,
                theta_true_np=theta_true_np,
                extra={
                    "difficulty_score_l2": posterior_summary["posterior_mean_l2_error"],
                    "difficulty_score_w2": posterior_summary["posterior_w2_pointmass"],
                    "posterior_summary": posterior_summary,
                },
            )
        )

    return figure_metadata


def run_c2st_diagnostic(
    cfg: ExperimentConfig,
    fig_dir: Path,
    prior,
    posterior,
    simulator,
    normalizer: Normalizer,
    device: torch.device,
    *,
    num_posterior_samples: int = 1000,
    num_examples: int = 3,
    method=None,
    examples: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Run C2ST (Classifier Two-Sample Test) diagnostic.

    Computes C2ST comparing posterior samples to prior samples.
    C2ST ≈ 0.5 means samples are indistinguishable (posterior = prior, bad).
    C2ST > 0.5 means posterior is different from prior (expected behavior).

    Args:
        cfg: Experiment configuration
        fig_dir: Directory to save figures
        prior: Prior distribution
        posterior: Trained posterior
        simulator: Simulator function
        normalizer: Data normalizer
        device: Torch device
        num_posterior_samples: Number of posterior samples
        num_examples: Number of observation examples to generate
        method: The method object (for FNPE-specific handling)

    Returns:
        Dict with C2ST values for each example
    """
    print("[C2ST] Computing C2ST diagnostics...")

    param_names = list(cfg.active_parameters)
    c2st_results = {}

    # Outside strict comparison mode, keep slower methods lighter.
    if not getattr(cfg, "unify_eval_budgets", False) and cfg.method in ["npse", "fnpe"]:
        num_posterior_samples = min(num_posterior_samples, 500)
        num_examples = min(num_examples, 2)

    # Sample from prior for reference
    with torch.no_grad():
        prior_samples = prior.sample((num_posterior_samples,)).to(device)
    prior_samples_np = prior_samples.detach().cpu().numpy()

    # For FNPE, use shared examples if provided (preferred for consistency)
    if cfg.method == "fnpe" and method is not None and hasattr(method, "task"):
        if examples is None:
            raise ValueError(
                "[C2ST] examples must be provided for consistent diagnostics. "
                "Use build_shared_diagnostic_examples() in run_experiment()."
            )

        for ex in examples[:num_examples]:
            ex_idx = int(ex["ex_idx"])
            print(f"[C2ST] Example {ex_idx+1}/{num_examples} (FNPE)")

            x_norm = ex["x_cond"]

            # Sample from posterior
            theta_post_norm = posterior.sample((num_posterior_samples,), x=x_norm)
            theta_post_phys = normalizer.unnormalize_theta(theta_post_norm)
            theta_post_np = theta_post_phys.detach().cpu().numpy()

            if theta_post_np.ndim == 3:
                theta_post_np = theta_post_np.reshape(-1, theta_post_np.shape[-1])

            # C2ST: compare posterior to prior
            c2st_val = compute_c2st(
                theta_post_np, prior_samples_np[: len(theta_post_np)]
            )
            c2st_results[f"c2st_prior_ex{ex_idx}"] = c2st_val
            print(f"[C2ST] Example {ex_idx+1}: C2ST(posterior, prior) = {c2st_val:.4f}")

        # Plot C2ST comparison
        if c2st_results and not cfg.no_plots:
            c2st_path = fig_dir / "c2st_comparison.png"
            plot_c2st_comparison(
                c2st_results, c2st_path, title="C2ST: Posterior vs Prior"
            )

        return c2st_results

    # Standard path for NPE/NPSE
    if examples is None:
        raise ValueError(
            "[C2ST] examples must be provided for consistent diagnostics. "
            "Use build_shared_diagnostic_examples() in run_experiment()."
        )

    for ex in examples[:num_examples]:
        ex_idx = int(ex["ex_idx"])
        print(f"[C2ST] Example {ex_idx+1}/{num_examples}")

        x_cond = _conditioning_x_on_device(ex["x_cond"], device)

        with torch.no_grad():
            theta_post_norm = posterior.sample((num_posterior_samples,), x=x_cond)

        # Handle different return types
        if isinstance(theta_post_norm, np.ndarray):
            theta_post_np = theta_post_norm
        else:
            theta_post_np = (
                normalizer.unnormalize_theta(theta_post_norm).detach().cpu().numpy()
            )

        if theta_post_np.ndim == 3:
            theta_post_np = theta_post_np.reshape(-1, theta_post_np.shape[-1])

        # C2ST: compare posterior to prior
        c2st_val = compute_c2st(theta_post_np, prior_samples_np[: len(theta_post_np)])
        c2st_results[f"c2st_prior_ex{ex_idx}"] = c2st_val
        print(f"[C2ST] Example {ex_idx+1}: C2ST(posterior, prior) = {c2st_val:.4f}")

    # Plot C2ST comparison
    if c2st_results and not cfg.no_plots:
        c2st_path = fig_dir / "c2st_comparison.png"
        plot_c2st_comparison(c2st_results, c2st_path, title="C2ST: Posterior vs Prior")

    return c2st_results


def run_diffusion_traces_diagnostic(
    cfg: ExperimentConfig,
    fig_dir: Path,
    posterior,
    prior,
    simulator,
    normalizer: Normalizer,
    device: torch.device,
    *,
    num_traces: int = 50,
    num_examples: int = 2,
    method=None,
    examples: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Run diffusion trace visualization for FNPE.

    This diagnostic shows how samples evolve during the reverse diffusion
    process, helping diagnose convergence and mixing behavior.

    Only applicable to FNPE (NPSE uses sbi's internal sampler which doesn't
    expose traces).

    Args:
        cfg: Experiment configuration
        fig_dir: Directory to save figures
        posterior: Trained posterior (must have sample_with_traces method)
        prior: Prior distribution
        simulator: Simulator function
        normalizer: Data normalizer
        device: Torch device
        num_traces: Number of diffusion traces to plot
        num_examples: Number of observation examples
        method: The method object (for FNPE-specific handling)
    """
    if cfg.no_plots:
        return []

    # Only for FNPE with trace support
    if cfg.method != "fnpe":
        print("[TRACES] Diffusion traces only available for FNPE, skipping...")
        return []

    if not hasattr(posterior, "sample_with_traces"):
        print("[TRACES] Posterior doesn't support sample_with_traces, skipping...")
        return []

    print("[TRACES] Generating diffusion trace plots...")

    param_names = list(cfg.active_parameters)
    figure_metadata: List[Dict[str, Any]] = []

    # Use shared examples if provided (preferred for consistency)
    if method is not None and hasattr(method, "task"):
        if examples is None:
            raise ValueError(
                "[TRACES] examples must be provided for consistent diagnostics. "
                "Use build_shared_diagnostic_examples() in run_experiment()."
            )

        for ex in examples[:num_examples]:
            ex_idx = int(ex["ex_idx"])
            print(f"[TRACES] Example {ex_idx+1}/{num_examples}")

            theta_true_np = np.asarray(ex["theta_true"], dtype=np.float32).reshape(-1)
            x_norm = ex["x_cond"]

            # Get diffusion traces
            try:
                # x_norm is in normalized units (unified interface), so set
                # return_physical=False to unnormalize conditioning internally.
                traces = posterior.sample_with_traces(
                    (num_traces,), x=x_norm, return_physical=False
                )
                # traces shape: (num_traces, num_steps, d_theta)

                trace_path = fig_dir / f"diffusion_traces_ex{ex_idx}.png"
                plot_diffusion_traces(
                    traces=traces,
                    out_path=trace_path,
                    theta_true=theta_true_np,
                    param_names=param_names,
                    title="Diffusion Sampling Traces",
                    example_id=f"ex{ex_idx}",
                    max_traces=num_traces,
                )
                figure_metadata.append(
                    _figure_metadata_entry(
                        figure_type="diffusion_traces",
                        file_path=trace_path,
                        example_idx=ex_idx,
                        theta_true_np=theta_true_np,
                        extra={"num_traces": int(num_traces)},
                    )
                )
            except Exception as e:
                print(f"[TRACES] Failed to get traces for example {ex_idx}: {e}")
        return figure_metadata

    return figure_metadata


def run_swd_diagnostic(
    cfg: ExperimentConfig,
    prior_phys,
    dap_samples_norm: torch.Tensor,
    normalizer: Normalizer,
) -> float:
    """Compute Sliced Wasserstein Distance between prior and DAP."""
    print("[SWD] Computing Sliced Wasserstein Distance...")

    dap_tensor = normalizer.unnormalize_theta(
        dap_samples_norm.view(-1, dap_samples_norm.shape[-1])
    )
    swd_val = sliced_wasserstein_prior_vs_dap(
        prior_phys,
        dap_tensor,
        num_projections=cfg.num_swd_projections,
        seed=cfg.random_seed,
    )
    print(f"SWD (prior vs DAP): {swd_val:.4f}")
    return swd_val


def run_w2_posterior_diagnostic(
    cfg: ExperimentConfig,
    fig_dir: Path,
    theta_test_phys: torch.Tensor,
    x_test_norm: torch.Tensor,
    posterior,
    normalizer: Normalizer,
    device: torch.device,
    *,
    num_cases: int = 100,
    num_posterior_samples: int = 500,
    theta_samples_phys: Optional[torch.Tensor] = None,
    sample_times_s: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Compute Wasserstein-style metrics comparing posterior to ground truth theta.

    This is particularly useful for simulation-only experiments where we have
    true theta values for each observation.
    """
    # Outside strict comparison mode, keep slower methods lighter.
    if not getattr(cfg, "unify_eval_budgets", False) and cfg.method in [
        "npse",
        "fnpe",
        "simformer",
    ]:
        num_cases = min(num_cases, 50)
        num_posterior_samples = min(num_posterior_samples, 200)

    print(f"[W2-POST] Computing posterior vs true theta metrics ({num_cases} cases)...")

    if theta_samples_phys is None:
        try:
            theta_true_sub, theta_samples, sample_times = sample_posterior_on_dataset(
                cfg,
                theta_test_phys,
                x_test_norm,
                posterior,
                normalizer,
                device,
                num_cases=num_cases,
                num_posterior_samples=num_posterior_samples,
            )
            sample_times_s = sample_times
        except Exception as e:
            print(f"[W2-POST] No successful samples, skipping metric: {e}")
            return {}
    else:
        theta_samples = theta_samples_phys[
            : min(num_cases, theta_samples_phys.shape[0])
        ]
        theta_true_sub = theta_test_phys[: theta_samples.shape[0]].cpu()

    # Compute metrics
    w2_results = wasserstein2_posterior_vs_true(
        theta_true_sub,
        theta_samples,
        num_projections=cfg.num_swd_projections,
        seed=cfg.random_seed,
    )
    w2_results["per_parameter_physical"] = per_parameter_posterior_metrics(
        theta_true_sub,
        theta_samples,
        param_names=list(cfg.active_parameters),
    )
    theta_true_norm = normalizer.normalize_theta(theta_true_sub.to(device)).cpu()
    theta_samples_norm = (
        normalizer.normalize_theta(
            theta_samples.to(device).reshape(-1, theta_samples.shape[-1])
        )
        .reshape(theta_samples.shape[0], theta_samples.shape[1], theta_samples.shape[2])
        .cpu()
    )
    w2_results["per_parameter_normalized"] = per_parameter_posterior_metrics(
        theta_true_norm,
        theta_samples_norm,
        param_names=list(cfg.active_parameters),
    )
    if sample_times_s:
        w2_results["sampling_time_mean_s"] = float(np.mean(sample_times_s))
        w2_results["sampling_time_std_s"] = float(np.std(sample_times_s))
        w2_results["sampling_time_total_s"] = float(np.sum(sample_times_s))

    if (
        not cfg.no_plots
        and "coverage_curve_alpha" in w2_results
        and "coverage_curve_empirical" in w2_results
    ):
        cov_path = fig_dir / "expected_coverage_simulated.png"
        plot_expected_coverage_curve(
            w2_results["coverage_curve_alpha"],
            w2_results["coverage_curve_empirical"],
            cov_path,
            title="Expected Coverage (Posterior vs Ground Truth)",
        )

    print(f"[W2-POST] Results:")
    print(f"  - W2 to ground truth (mean): {w2_results['w2_mean']:.4f}")
    print(f"  - L2 error of posterior mean: {w2_results['l2_error_mean']:.4f}")
    print(f"  - Coverage 90%: {w2_results['coverage_90']:.2%}")
    print(f"  - Coverage 50%: {w2_results['coverage_50']:.2%}")
    print(f"  - Coverage curve MAE: {w2_results['coverage_curve_mae']:.4f}")
    print(f"  - SWD posterior vs true: {w2_results['swd_posterior_vs_true']:.4f}")
    if "sampling_time_mean_s" in w2_results:
        print(f"  - Sampling time / case: {w2_results['sampling_time_mean_s']:.4f}s")

    return w2_results


def run_one_step_rmse_diagnostic(
    cfg: ExperimentConfig,
    prior,
    posterior,
    simulator,
    normalizer: Normalizer,
    device: torch.device,
    *,
    num_cases: int = 20,
    num_posterior_samples: int = 100,
) -> Dict[str, Any]:
    """1-step-ahead RMSE diagnostic."""
    # Outside strict comparison mode, keep slower methods lighter.
    if not getattr(cfg, "unify_eval_budgets", False) and cfg.method in [
        "npse",
        "fnpe",
        "simformer",
    ]:
        num_posterior_samples = min(num_posterior_samples, 50)

    print(f"[1-STEP] Running RMSE diagnostic ({num_cases} cases)...")

    obs_dim = cfg.obs_dim
    state_dim = cfg.state_dim

    all_true_next = []
    all_pred_next = []
    prior_cpu = build_prior(cfg, torch.device("cpu"))
    old_batch_idx = getattr(simulator, "_batch_idx", 0)
    simulator._batch_idx = 8000
    try:
        for i in range(num_cases):
            theta_true = _sample_theta_from_prior_cpu(
                prior_cpu, int(cfg.random_seed) + 8000 + i
            ).to(device)

            sim_out = simulator(theta_true)
            if not isinstance(sim_out, tuple):
                print(f"[1-STEP] Skipping - simulator doesn't return controls")
                return {"rmse_overall": None, "rmse_per_dim": None}

            x_sim, controls_sim = sim_out
            x_np = x_sim[0].detach().cpu().numpy()
            y_seq = x_np[:, :obs_dim]

            if y_seq.shape[0] < 2:
                continue

            y_next_true = y_seq[1].astype(np.float32)
            # Use full trajectory for conditioning (posterior expects full T_seg shape)
            x_cond = normalizer.normalize_x(x_sim, cfg.obs_dim).to(device)

            with torch.no_grad():
                theta_post_norm = posterior.sample((num_posterior_samples,), x=x_cond)

            if isinstance(theta_post_norm, np.ndarray):
                theta_post_np = theta_post_norm
            else:
                theta_post_np = (
                    normalizer.unnormalize_theta(theta_post_norm).detach().cpu().numpy()
                )

            if theta_post_np.ndim == 3:
                theta_post_np = theta_post_np.reshape(-1, theta_post_np.shape[-1])

            controls_2 = {k: v[:2] for k, v in controls_sim.items()}
            state0 = initial_state_from_obs(y_seq[0])

            y_pred_batch = simulate_y_batch_for_thetas(
                theta_batch_np=theta_post_np.astype(np.float32),
                controls=controls_2,
                state_dim=state_dim,
                cfg=cfg,
                state0=state0,
            )

            y_pred_next = y_pred_batch[:, 1, :]
            all_true_next.append(y_next_true)
            all_pred_next.append(y_pred_next)
    finally:
        simulator._batch_idx = old_batch_idx

    if not all_true_next:
        return {"rmse_overall": None, "rmse_per_dim": None}

    y_true_next_np = np.stack(all_true_next, axis=0)
    K_common = min(arr.shape[0] for arr in all_pred_next)
    y_pred_next_np = np.stack([arr[:K_common] for arr in all_pred_next], axis=0)

    y_true_next_t = torch.from_numpy(y_true_next_np.astype(np.float32))
    y_pred_next_t = torch.from_numpy(y_pred_next_np.astype(np.float32))

    rmse_overall, rmse_per_dim = one_step_rmse_observation(y_true_next_t, y_pred_next_t)

    print(f"[1-STEP] Overall RMSE: {rmse_overall:.4f}")

    return {
        "rmse_overall": rmse_overall,
        "rmse_per_dim": (
            rmse_per_dim.tolist()
            if hasattr(rmse_per_dim, "tolist")
            else list(rmse_per_dim)
        ),
    }


def run_real_data_evaluation(
    cfg: ExperimentConfig,
    exp_dir: Path,
    fig_dir: Path,
    posterior,
    prior_phys,
    simulator,
    normalizer: Normalizer,
    device: torch.device,
    T_event: int,
    K_ppc: int = 300,
) -> Dict[str, Any]:
    """
    Run real data evaluation on a trained posterior.

    Args:
        cfg: Experiment configuration
        exp_dir: Experiment directory for saving results
        fig_dir: Figure directory
        posterior: Trained posterior
        prior_phys: Prior in physical units
        simulator: Simulator function
        normalizer: Data normalizer
        device: Torch device
        T_event: Expected time series length
        K_ppc: Number of posterior predictive samples

    Returns:
        Dict with real data metrics
    """
    import pandas as pd

    csv_path = Path(cfg.real_data_csv)
    if not csv_path.exists():
        print(f"[REAL] CSV not found: {csv_path}")
        return {}

    print(f"[REAL] Loading real data from {csv_path}")
    df_real = pd.read_csv(csv_path)

    # Build real window
    try:
        x_obs_full, controls_real, start_idx = build_real_window_from_csv(
            df_real,
            cfg,
            device,
            start_idx=None,
            prefer_low_brake=cfg.prefer_low_brake,
            brake_thresh=5.0,
            max_viol_frac=0.01,
            rate_body_z_in_deg_s=True,
            tire_rates_in_rpm=False,
            vel_body_in_kmh=False,
        )
    except Exception as e:
        print(f"[REAL] Failed to build real window: {e}")
        return {}

    # Check time length
    if x_obs_full.shape[1] != T_event:
        print(
            f"[REAL] Window length {x_obs_full.shape[1]} != model T_event={T_event}, skipping"
        )
        return {}

    print(f"[REAL] Real data window: shape={x_obs_full.shape}, start_idx={start_idx}")

    # Posterior predictive on real segment
    y_real, y_ppc = posterior_predictive_from_real(
        posterior,
        x_obs_full,
        controls_real,
        cfg,
        normalizer=normalizer,
        device=device,
        K_ppc=K_ppc,
    )
    print(f"[REAL] PPC shapes: y_real={y_real.shape}, y_ppc={y_ppc.shape}")

    # Plot PPC time-series on real segment
    if not cfg.no_plots:
        ppc_path = fig_dir / "ppc_timeseries_real.png"
        plot_ppc_trajectories(
            y_real=y_real,
            y_ppc=y_ppc,
            obs_labels=OBS_LABELS,
            dt=cfg.dt,
            out_path=ppc_path,
            max_trajs=20,
            plot_all_trajs=True,
            max_dims=cfg.obs_dim,
            title="Posterior Predictive Check on Real Drive Segment",
        )

    # PPC on simulated holdout
    try:
        x_sim_full, controls_sim = build_simulated_window_for_eval(
            cfg, prior_phys, simulator, device
        )
        y_sim, y_ppc_sim = posterior_predictive_from_real(
            posterior,
            x_sim_full,
            controls_sim,
            cfg,
            normalizer=normalizer,
            device=device,
            K_ppc=K_ppc,
        )
        if not cfg.no_plots:
            ppc_sim_path = fig_dir / "ppc_timeseries_simulated.png"
            plot_ppc_trajectories(
                y_real=y_sim,
                y_ppc=y_ppc_sim,
                obs_labels=OBS_LABELS,
                dt=cfg.dt,
                out_path=ppc_sim_path,
                max_trajs=20,
                plot_all_trajs=True,
                max_dims=cfg.obs_dim,
                title="Posterior Predictive Check on Simulated Holdout",
            )
    except Exception as e:
        print(f"[REAL] Simulated PPC failed: {e}")

    # Train-like observations histogram
    if not cfg.no_plots:
        try:
            N_hist = min(2000, cfg.num_simulations)
            with torch.no_grad():
                theta_hist = prior_phys.sample((N_hist,)).to(device)
                sim_out = simulator(theta_hist)
                x_hist = sim_out[0] if isinstance(sim_out, tuple) else sim_out

            train_obs = (
                x_hist[:, :, : cfg.obs_dim]
                .detach()
                .cpu()
                .numpy()
                .reshape(-1, cfg.obs_dim)
            )

            x_obs_all = prep_x_obs_from_df(
                df_real,
                start_idx=0,
                T=len(df_real),
                rate_body_z_in_deg_s=True,
                tire_rates_in_rpm=False,
                vel_body_in_kmh=False,
            )
            real_obs_all = x_obs_all.numpy()

            custom_ranges = {
                "yaw_rate [rad/s]": (-2, 2),
                "v_body_x [m/s]": (5, 25),
                "v_body_y [m/s]": (-2, 2),
                "a_body_x [m/s²]": (-6, 6),
                "a_body_y [m/s²]": (-6, 6),
                "tire_FL [rad/s]": (0, 90),
                "tire_FR [rad/s]": (0, 90),
                "tire_RL [rad/s]": (0, 90),
                "tire_RR [rad/s]": (0, 90),
            }

            hist_path = fig_dir / "obs_hist_train_vs_real.png"
            plot_obs_1d_hist_custom(
                train_obs=train_obs,
                real_obs=real_obs_all,
                obs_labels=OBS_LABELS,
                custom_ranges=custom_ranges,
                out_path=hist_path,
                bins=80,
            )
        except Exception as e:
            print(f"[REAL] Histogram generation failed: {e}")

    # Compute metrics
    metrics = real_data_trajectory_metrics(y_real, y_ppc, normalize_w2=True)

    print("\n=== Real-data trajectory metrics ===")
    print(f"Overall RMSE (mixed units): {metrics['rmse_overall']:.4f}")
    print("Per-dimension RMSEs:")
    for label, val in zip(OBS_LABELS, metrics["rmse_per_dim"]):
        print(f"  {label:<20}: {val:.4f}")
    print(f"W2 (normalized): {metrics['w2']:.4f}")
    print(f"Mean per-sample W2: {metrics['w2_mean_per_sample']:.4f}")

    # Save metrics
    real_metrics_path = exp_dir / "real_metrics.json"
    with real_metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[REAL] Saved real-data metrics to {real_metrics_path}")

    return metrics


def run_multi_trajectory_ppc(
    cfg: ExperimentConfig,
    exp_dir: Path,
    fig_dir: Path,
    posterior,
    normalizer: Normalizer,
    device: torch.device,
    T_event: int,
    data_dir: str = "../data/measurements",
    K_ppc: int = 300,
    pso_trajectory_params: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """
    Run posterior predictive checks on ALL measurement CSV trajectories.

    For each CSV in *data_dir*, builds a real-data window, samples from the
    posterior, forward-simulates, plots PPC time-series, and computes metrics.
    Aggregated results are saved to ``exp_dir / multi_traj_ppc_metrics.json``.

    Args:
        cfg:          Experiment configuration (T_seg, obs_dim, etc.)
        exp_dir:      Experiment output directory
        fig_dir:      Directory for figures
        posterior:    Trained posterior
        normalizer:   Data normalizer (for conditioning and theta mapping)
        device:       Torch device
        T_event:      Expected time-series length (must match training)
        data_dir:     Path to directory containing measurement CSVs
        K_ppc:        Number of posterior predictive samples per trajectory
        pso_trajectory_params:
            Optional list of per-trajectory PSO results (each dict should
            contain at least ``"csv"`` and ``"params"``).  Used only for
            logging the PSO-optimized mu alongside the posterior estimate.

    Returns:
        Dict with per-trajectory and aggregate PPC metrics.
    """
    import pandas as pd
    from pathlib import Path as _Path

    data_path = _Path(data_dir)
    csv_files = sorted(data_path.glob("*.csv"))
    if not csv_files:
        print(f"[MULTI-PPC] No CSV files found in {data_path}")
        return {}

    print(f"\n{'='*60}")
    print(f"Multi-Trajectory Posterior Predictive Check")
    print(f"{'='*60}")
    print(f"  Data dir:    {data_path}")
    print(f"  Num CSVs:    {len(csv_files)}")
    print(f"  K_ppc:       {K_ppc}")
    print(f"  T_event:     {T_event}")
    print(f"{'='*60}\n")

    # Create sub-directory for multi-trajectory PPC figures
    ppc_fig_dir = fig_dir / "multi_traj_ppc"
    ppc_fig_dir.mkdir(parents=True, exist_ok=True)

    # Build lookup for PSO per-trajectory params (if provided)
    pso_mu_lookup: Dict[str, float] = {}
    if pso_trajectory_params:
        for tp in pso_trajectory_params:
            csv_name = _Path(tp.get("csv", "")).name
            mu_val = tp.get("params", {}).get("mu")
            if csv_name and mu_val is not None:
                pso_mu_lookup[csv_name] = float(mu_val)

    per_traj_results: List[Dict[str, Any]] = []

    for i, csv_path in enumerate(csv_files):
        traj_name = csv_path.stem
        print(f"\n--- Trajectory {i+1}/{len(csv_files)}: {traj_name} ---")

        try:
            df_real = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  [SKIP] Failed to read CSV: {e}")
            continue

        # Build observation window + controls
        try:
            x_obs_full, controls_real, start_idx = build_real_window_from_csv(
                df_real,
                cfg,
                device,
                start_idx=None,
                prefer_low_brake=cfg.prefer_low_brake,
                brake_thresh=5.0,
                max_viol_frac=0.01,
                rate_body_z_in_deg_s=True,
                tire_rates_in_rpm=False,
                vel_body_in_kmh=False,
            )
        except Exception as e:
            print(f"  [SKIP] Failed to build real window: {e}")
            continue

        # Check time length matches training
        if x_obs_full.shape[1] != T_event:
            print(f"  [SKIP] Window length {x_obs_full.shape[1]} != T_event={T_event}")
            continue

        print(f"  Real window: shape={x_obs_full.shape}, start_idx={start_idx}")

        # Posterior predictive
        try:
            y_real, y_ppc = posterior_predictive_from_real(
                posterior,
                x_obs_full,
                controls_real,
                cfg,
                normalizer=normalizer,
                device=device,
                K_ppc=K_ppc,
            )
        except Exception as e:
            print(f"  [SKIP] PPC simulation failed: {e}")
            continue

        print(f"  PPC shapes: y_real={y_real.shape}, y_ppc={y_ppc.shape}")

        # Plot PPC for this trajectory
        if not cfg.no_plots:
            ppc_path = ppc_fig_dir / f"ppc_{traj_name}.png"
            title = f"PPC: {traj_name}"
            if csv_path.name in pso_mu_lookup:
                title += f" (PSO mu={pso_mu_lookup[csv_path.name]:.4f})"
            plot_ppc_trajectories(
                y_real=y_real,
                y_ppc=y_ppc,
                obs_labels=OBS_LABELS,
                dt=cfg.dt,
                out_path=ppc_path,
                max_trajs=20,
                plot_all_trajs=True,
                max_dims=cfg.obs_dim,
                title=title,
            )

        # Compute metrics
        traj_metrics = real_data_trajectory_metrics(y_real, y_ppc, normalize_w2=True)

        result_entry = {
            "csv": csv_path.name,
            "trajectory": traj_name,
            "start_idx": int(start_idx),
            "metrics": traj_metrics,
        }
        if csv_path.name in pso_mu_lookup:
            result_entry["pso_mu"] = pso_mu_lookup[csv_path.name]

        per_traj_results.append(result_entry)

        print(f"  RMSE: {traj_metrics['rmse_overall']:.4f}")
        print(f"  W2:   {traj_metrics['w2']:.4f}")

    # Aggregate metrics across trajectories
    if per_traj_results:
        rmses = [r["metrics"]["rmse_overall"] for r in per_traj_results]
        w2s = [r["metrics"]["w2"] for r in per_traj_results]
        aggregate = {
            "num_trajectories": len(per_traj_results),
            "rmse_mean": float(np.mean(rmses)),
            "rmse_std": float(np.std(rmses)),
            "rmse_min": float(np.min(rmses)),
            "rmse_max": float(np.max(rmses)),
            "w2_mean": float(np.mean(w2s)),
            "w2_std": float(np.std(w2s)),
            "w2_min": float(np.min(w2s)),
            "w2_max": float(np.max(w2s)),
        }
    else:
        aggregate = {"num_trajectories": 0}

    combined = {
        "aggregate": aggregate,
        "per_trajectory": per_traj_results,
    }

    # Save
    out_path = exp_dir / "multi_traj_ppc_metrics.json"
    with out_path.open("w") as f:
        json.dump(combined, f, indent=2, default=tensor_to_python)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Multi-Trajectory PPC Summary")
    print(f"{'='*60}")
    if per_traj_results:
        print(f"  Trajectories evaluated: {len(per_traj_results)}/{len(csv_files)}")
        print(
            f"  RMSE  mean={aggregate['rmse_mean']:.4f}  std={aggregate['rmse_std']:.4f}"
        )
        print(f"  W2    mean={aggregate['w2_mean']:.4f}  std={aggregate['w2_std']:.4f}")
        print(f"\n  Per-trajectory breakdown:")
        for r in per_traj_results:
            mu_str = f"  (PSO mu={r['pso_mu']:.4f})" if "pso_mu" in r else ""
            print(
                f"    {r['trajectory']:<50}  RMSE={r['metrics']['rmse_overall']:.4f}  "
                f"W2={r['metrics']['w2']:.4f}{mu_str}"
            )
    else:
        print("  No trajectories evaluated successfully.")
    print(f"\n  Results saved to {out_path}")
    print(f"{'='*60}\n")

    return combined


# =============================================================================
# Main Experiment Runner
# =============================================================================


def run_experiment(cfg: ExperimentConfig) -> Dict[str, Any]:
    """
    Run a complete SBI experiment with the specified method.

    This function:
    1. Sets up environment and device
    2. Builds task (simulator, prior)
    3. Generates or loads dataset
    4. Trains the specified method
    5. Runs diagnostics (SBC, SWD, 1-step RMSE)
    6. Saves all results

    Args:
        cfg: Experiment configuration

    Returns:
        Dict with experiment results and paths
    """
    print(f"\n{'='*60}")
    print(f"SBI Experiment: {cfg.method.upper()}")
    print(f"{'='*60}")
    print(f"Parameters: {cfg.active_parameters}")
    sim_budget = (
        cfg.fnpe_num_simulations if cfg.method == "fnpe" else cfg.num_simulations
    )
    print(f"Simulations: {sim_budget}")
    print(f"Sequence length: {cfg.T_seg}")
    print(f"{'='*60}\n")

    # --- Setup ---
    setup_environment(cfg.sim_seed)
    device = get_device(cfg.device)
    init_simulation_from_config(cfg)

    # --- Build prior and simulator ---
    prior_phys = build_prior(cfg, device)
    simulator = make_simulator(cfg, device)

    def simulator_for_sbi(theta: torch.Tensor):
        x, _ = simulator(theta)
        return x

    # Probe dimensions
    probe_theta = prior_phys.sample((1,))
    probe_x = simulator_for_sbi(probe_theta)
    _, T_event, D_in = probe_x.shape
    simulator._batch_idx = 0
    print(
        f"[SETUP] input_dim={D_in}, T_event={T_event}, d_theta={cfg.active_param_dim()}"
    )

    checkpoint_dir: Path | None = None
    if cfg.checkpoint:
        checkpoint_path = Path(cfg.checkpoint)
        checkpoint_dir = (
            checkpoint_path if checkpoint_path.is_dir() else checkpoint_path.parent
        )

    # --- Create experiment directory ---
    exp_dir = make_experiment_dir(cfg)
    fig_dir = exp_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    status_path = exp_dir / "run_status.json"
    print(f"[SETUP] Experiment dir: {exp_dir}")
    print(f"[SETUP] Directory exists: {exp_dir.exists()}")
    cfg.save(str(exp_dir / "config.json"))
    print(f"[SETUP] Saved config to {exp_dir / 'config.json'}")

    completed_stages: List[str] = []
    timing_breakdown: Dict[str, Any] = {}
    figure_metadata: Dict[str, Any] = {}
    stage_failures: Dict[str, Any] = {}

    def mark_status(
        stage: str, *, state: str = "running", extra: Optional[Dict[str, Any]] = None
    ) -> None:
        if stage not in completed_stages and state in {
            "completed",
            "complete",
            "complete_with_failures",
        }:
            completed_stages.append(stage)
        _update_run_status(
            status_path,
            state=state,
            stage=stage,
            exp_dir=exp_dir,
            completed_stages=completed_stages,
            extra=extra,
        )

    def record_stage_failure(
        stage: str, exc: Exception, *, fatal: bool = False
    ) -> None:
        stage_failures[stage] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        _update_run_status(
            status_path,
            state="failed" if fatal else "running",
            stage=stage,
            exp_dir=exp_dir,
            completed_stages=completed_stages,
            error=stage_failures[stage],
            extra={
                "stage_failures": stage_failures,
                "timing_breakdown": timing_breakdown,
            },
        )

    mark_status(
        "started",
        extra={
            "method": cfg.method,
            "seed": int(cfg.random_seed),
            "sim_seed": int(cfg.sim_seed),
            "train_seed": int(cfg.train_seed),
            "requested_budget_steps": cfg.requested_budget_steps,
        },
    )

    train_region_metadata: Dict[str, Any] | None = None
    test_region_metadata: Dict[str, Any] | None = None
    train_dataset_artifact: Dict[str, Any] | None = None
    test_dataset_artifact: Dict[str, Any] | None = None
    theta_test_phys = None
    x_test_phys = None
    x_test_norm = None
    x_test_state0 = None

    if cfg.run_simulated_test_eval or cfg.run_simulated_ppc:
        try:
            t0 = time.time()
            (
                theta_test_phys,
                x_test_phys,
                test_region_metadata,
                x_test_state0,
                test_dataset_artifact,
            ) = get_or_generate_test_dataset(cfg, prior_phys, simulator, device)
            heldout_dataset_time_s = float(time.time() - t0)
            timing_breakdown["heldout_dataset_prepare_time_s"] = heldout_dataset_time_s
            if test_dataset_artifact and test_dataset_artifact.get("cache_hit"):
                timing_breakdown["heldout_dataset_load_time_s"] = heldout_dataset_time_s
            else:
                timing_breakdown["heldout_dataset_generation_time_s"] = (
                    heldout_dataset_time_s
                )
            mark_status(
                "heldout_dataset_ready",
                state="completed",
                extra={"heldout_dataset_artifact": test_dataset_artifact},
            )
        except Exception as e:
            record_stage_failure("heldout_dataset", e, fatal=True)
            raise

    # --- Get or generate training dataset (skip training data for FNPE) ---
    if cfg.method == "fnpe":
        print(
            "[DATA] Skipping training dataset generation for FNPE (it generates its own data)",
            flush=True,
        )
        theta_train_phys = None
        x_train_phys = None
        normalizer = None
        theta_train = None
        x_train = None
    elif cfg.checkpoint and not cfg.do_train:
        print(
            "[DATA] Eval-only mode: skipping training dataset generation and loading "
            "saved normalization stats.",
            flush=True,
        )
        theta_train_phys = None
        x_train_phys = None
        theta_train = None
        x_train = None

        norm_path = (checkpoint_dir or exp_dir) / "stats_normalization.json"
        if not norm_path.exists():
            raise FileNotFoundError(
                f"Expected saved normalization stats for eval-only run: {norm_path}"
            )
        normalizer_cpu = load_normalizer(norm_path)
        normalizer = normalizer_cpu.to(device)
        if x_test_phys is not None:
            x_test_norm = normalizer_cpu.normalize_x(x_test_phys, cfg.obs_dim)
    else:
        try:
            t0 = time.time()
            (
                theta_train_phys,
                x_train_phys,
                train_region_metadata,
                train_dataset_artifact,
            ) = get_or_generate_training_dataset(cfg, prior_phys, simulator, device)
            training_dataset_time_s = float(time.time() - t0)
            timing_breakdown["training_dataset_prepare_time_s"] = (
                training_dataset_time_s
            )
            if train_dataset_artifact and train_dataset_artifact.get("cache_hit"):
                timing_breakdown["training_dataset_load_time_s"] = (
                    training_dataset_time_s
                )
            else:
                timing_breakdown["training_dataset_generation_time_s"] = (
                    training_dataset_time_s
                )
            mark_status(
                "training_dataset_ready",
                state="completed",
                extra={"training_dataset_artifact": train_dataset_artifact},
            )
        except Exception as e:
            record_stage_failure("training_dataset", e, fatal=True)
            raise

        # --- Fit normalization ---
        norm_t0 = time.time()
        normalizer_cpu = fit_normalizer(theta_train_phys, x_train_phys, cfg.obs_dim)
        norm_path = exp_dir / "stats_normalization.json"
        print(
            f"[NORM] Saving to {norm_path}, parent exists: {norm_path.parent.exists()}"
        )
        save_normalizer(normalizer_cpu, norm_path)
        print(f"[NORM] Saved stats to {norm_path}")

        normalizer = normalizer_cpu.to(device)
        timing_breakdown["normalization_fit_time_s"] = float(time.time() - norm_t0)

        # --- Normalize data ---
        theta_train = normalizer_cpu.normalize_theta(theta_train_phys)
        x_train = normalizer_cpu.normalize_x(x_train_phys, cfg.obs_dim)
        if x_test_phys is not None:
            x_test_norm = normalizer_cpu.normalize_x(x_test_phys, cfg.obs_dim)
        mark_status("normalization_ready", state="completed")

    # --- Build normalized prior (skip for FNPE - uses its own task-based prior) ---
    bounds = cfg.param_bounds()
    low_list = [bounds[name][0] for name in cfg.active_parameters]
    high_list = [bounds[name][1] for name in cfg.active_parameters]

    if cfg.method == "fnpe":
        # FNPE uses physical prior directly via its task
        prior_norm = prior_phys
    else:
        theta_low_norm = normalizer.normalize_theta(
            torch.tensor(low_list, dtype=torch.float32, device=device)
        )
        theta_high_norm = normalizer.normalize_theta(
            torch.tensor(high_list, dtype=torch.float32, device=device)
        )
        prior_norm = sbi_utils.BoxUniform(low=theta_low_norm, high=theta_high_norm)

    # --- Build method ---
    print(f"\n[METHOD] Building {cfg.method.upper()}...", flush=True)

    method_kwargs = {}
    if cfg.method == "npse":
        method_kwargs["sde_type"] = cfg.sde_type
    elif cfg.method == "simformer":
        method_kwargs.update(
            {
                "num_timepoints": cfg.simformer_num_timepoints,
                "token_dim": cfg.simformer_token_dim,
                "condition_token_dim": cfg.simformer_condition_token_dim,
                "condition_token_init_scale": cfg.simformer_condition_token_init_scale,
                "condition_token_init_mean": cfg.simformer_condition_token_init_mean,
                "condition_mode": cfg.simformer_condition_mode,
                "time_embedding_dim": cfg.simformer_time_embedding_dim,
                "num_heads": cfg.simformer_num_heads,
                "num_layers": cfg.simformer_num_layers,
                "attn_size": cfg.simformer_attn_size,
                "widening_factor": cfg.simformer_widening_factor,
                "num_hidden_layers": cfg.simformer_num_hidden_layers,
                "skip_connection_attn": cfg.simformer_skip_connection_attn,
                "skip_connection_mlp": cfg.simformer_skip_connection_mlp,
                "layer_norm": cfg.simformer_layer_norm,
                "sigma_min": cfg.simformer_sigma_min,
                "sigma_max": cfg.simformer_sigma_max,
                "T_min": cfg.simformer_t_min,
                "T_max": cfg.simformer_t_max,
                "num_diffusion_steps": cfg.simformer_num_diffusion_steps,
                "learning_rate": cfg.simformer_learning_rate,
                "min_learning_rate": cfg.simformer_min_learning_rate,
                "clip_max_norm": cfg.simformer_clip_max_norm,
                "batch_size": cfg.simformer_batch_size,
                "train_steps_scaling": cfg.simformer_train_steps_scaling,
                "min_train_steps": cfg.simformer_min_train_steps,
                "max_train_steps": cfg.simformer_max_train_steps,
                "validation_fraction": cfg.simformer_validation_fraction,
                "val_repeat": cfg.simformer_val_repeat,
                "val_every": cfg.simformer_val_every,
                "stop_early_count": cfg.simformer_stop_early_count,
                "val_error_ratio": cfg.simformer_val_error_ratio,
                "condition_mask_name": cfg.simformer_condition_mask_name,
                "condition_mask_kwargs": {
                    "p_joint": cfg.simformer_condition_mask_p_joint,
                    "p_posterior": cfg.simformer_condition_mask_p_posterior,
                    "p_likelihood": cfg.simformer_condition_mask_p_likelihood,
                    "p_rnd1": cfg.simformer_condition_mask_p_rnd1,
                    "p_rnd2": cfg.simformer_condition_mask_p_rnd2,
                    "rnd1_prob": cfg.simformer_condition_mask_rnd1_prob,
                    "rnd2_prob": cfg.simformer_condition_mask_rnd2_prob,
                },
                "use_metadata": cfg.simformer_use_metadata,
                "rebalance_loss": cfg.simformer_rebalance_loss,
            }
        )
        print(
            f"[Simformer] layers={cfg.simformer_num_layers}, heads={cfg.simformer_num_heads}, "
            f"timepoints={cfg.simformer_num_timepoints}, "
            f"steps=clip(n*{cfg.simformer_train_steps_scaling}, "
            f"{cfg.simformer_min_train_steps}, {cfg.simformer_max_train_steps})",
            flush=True,
        )
    elif cfg.method == "fnpe":
        method_kwargs.update(
            {
                "hidden_dim": cfg.fnpe_hidden_dim,
                "num_hidden": cfg.fnpe_num_hidden,
                "model_type": cfg.fnpe_model_type,
                "window_size": cfg.fnpe_window_size,  # CRITICAL: small Markov window
                "num_outer_epochs": cfg.fnpe_num_outer_epochs,
                "num_inner_epochs": cfg.fnpe_num_inner_epochs,
                "batch_size": cfg.fnpe_batch_size,
                "learning_rate": cfg.fnpe_learning_rate,
                "clip_max_norm": cfg.fnpe_clip_max_norm,
                "optimizer_name": cfg.fnpe_optimizer,
                "scheduler_name": cfg.fnpe_scheduler,
                "num_diffusion_steps": cfg.fnpe_num_diffusion_steps,
                "score_fn_type": cfg.fnpe_score_fn_type,
                "proposal_type": cfg.fnpe_proposal_type,  # "pred" (correct), "naive", or "trajectory" (old)
                "pilot_fraction": cfg.fnpe_pilot_fraction,  # Fraction of sims for pilots (default 2%)
                "pilot_length": cfg.fnpe_pilot_length,  # Length of pilot trajectories (default 500)
                "proposal_noise": cfg.fnpe_proposal_noise,  # Noise scale (default 0.03 * std)
                "gauss_posterior_precission_scale": getattr(
                    cfg, "fnpe_gauss_precision_scale", None
                ),
            }
        )
        proposal_desc = {
            "pred": "proposal from pilot sims (CORRECT)",
            "naive": "initial state distribution only",
            "trajectory": "trajectory pairs (OLD/INCORRECT)",
        }.get(cfg.fnpe_proposal_type, cfg.fnpe_proposal_type)
        print(
            f"[FNPE] window_size={cfg.fnpe_window_size}",
            flush=True,
        )
        print(
            f"[FNPE] schedule: outer_epochs={cfg.fnpe_num_outer_epochs}, "
            f"inner_epochs={cfg.fnpe_num_inner_epochs}, batch_size={cfg.fnpe_batch_size}",
            flush=True,
        )
        print(
            f"[FNPE] proposal_type='{cfg.fnpe_proposal_type}' - {proposal_desc}",
            flush=True,
        )
        if getattr(cfg, "fnpe_skip_normalize", False):
            print(
                "[FNPE] *** NORMALIZATION DISABLED (fnpe_skip_normalize=True) ***",
                flush=True,
            )
        if getattr(cfg, "fnpe_clip_samples", False):
            print("[FNPE] Diffusion sample clipping ENABLED", flush=True)

    build_t0 = time.time()
    try:
        method = build_method(cfg.method, cfg, prior_norm, device, **method_kwargs)
        method.build(input_dim=D_in, seq_len=T_event)
        timing_breakdown["method_build_time_s"] = float(time.time() - build_t0)
        mark_status("method_built", state="completed")
    except Exception as e:
        record_stage_failure("method_build", e, fatal=True)
        raise

    # --- Train ---
    training_summary = {}
    if cfg.do_train:
        print(f"\n[TRAIN] Training {cfg.method.upper()}...", flush=True)
        setup_environment(cfg.train_seed)  # Use train seed
        train_t0 = time.time()
        try:
            if cfg.method == "fnpe":
                # FNPE generates its own data
                training_summary = method.train(
                    num_simulations=cfg.fnpe_num_simulations, T_obs=T_event
                )
                # After training, create normalizer from FNPE's task stats for unified interface
                norm_stats = method.task.get_normalization_stats()
                from utils.normalization import Normalizer

                obs_dim = cfg.obs_dim
                skip_norm = getattr(cfg, "fnpe_skip_normalize", False)
                if skip_norm or norm_stats.get("obs_mean") is None:
                    # Identity normalizer: mean=0, std=1 → no-op transform
                    print("[NORM] Using identity normalizer (normalization disabled)")
                    normalizer = Normalizer(
                        obs_mean=torch.zeros(obs_dim, dtype=torch.float32),
                        obs_std=torch.ones(obs_dim, dtype=torch.float32),
                        ctrl_mean=torch.zeros(4, dtype=torch.float32),
                        ctrl_std=torch.ones(4, dtype=torch.float32),
                        theta_mean=torch.zeros(
                            cfg.active_param_dim(), dtype=torch.float32
                        ),
                        theta_std=torch.ones(
                            cfg.active_param_dim(), dtype=torch.float32
                        ),
                    ).to(device)
                else:
                    ctrl_mean = norm_stats.get("ctrl_mean")
                    ctrl_std = norm_stats.get("ctrl_std")
                    if ctrl_mean is None or ctrl_std is None:
                        ctrl_mean = np.zeros(4, dtype=np.float32)
                        ctrl_std = np.ones(4, dtype=np.float32)
                    normalizer = Normalizer(
                        obs_mean=torch.tensor(
                            np.array(norm_stats["obs_mean"]), dtype=torch.float32
                        ),
                        obs_std=torch.tensor(
                            np.array(norm_stats["obs_std"]), dtype=torch.float32
                        ),
                        ctrl_mean=torch.tensor(
                            np.array(ctrl_mean), dtype=torch.float32
                        ),
                        ctrl_std=torch.tensor(np.array(ctrl_std), dtype=torch.float32),
                        theta_mean=torch.tensor(
                            np.array(norm_stats["theta_mean"]), dtype=torch.float32
                        ),
                        theta_std=torch.tensor(
                            np.array(norm_stats["theta_std"]), dtype=torch.float32
                        ),
                    ).to(device)
                # Save normalizer for consistency
                save_normalizer(normalizer, exp_dir / "stats_normalization.json")
                print(f"[NORM] Created normalizer from FNPE task stats")
                if x_test_phys is not None:
                    x_test_norm = normalizer.normalize_x(x_test_phys, cfg.obs_dim).cpu()
            else:
                training_summary = method.train(theta_train, x_train)

            # Save model
            model_path = method.save(exp_dir)
            print(f"[TRAIN] Saved model to {model_path}", flush=True)
            timing_breakdown["model_training_time_s"] = float(time.time() - train_t0)
            mark_status(
                "trained",
                state="completed",
                extra={"model_path": str(model_path)},
            )
        except Exception as e:
            record_stage_failure("training", e, fatal=True)
            raise

    elif cfg.checkpoint:
        print(f"\n[LOAD] Loading checkpoint from {cfg.checkpoint}", flush=True)
        load_t0 = time.time()
        try:
            method.load(checkpoint_dir or Path(cfg.checkpoint))
            training_summary = getattr(method, "_training_summary", {}) or {}
            # For FNPE checkpoint, also load/create normalizer
            if cfg.method == "fnpe":
                norm_path = checkpoint_dir / "stats_normalization.json"
                if norm_path.exists():
                    normalizer = load_normalizer(norm_path).to(device)
                    print(f"[NORM] Loaded normalizer from {norm_path}")
                else:
                    # Create from task stats
                    norm_stats = method.task.get_normalization_stats()
                    ctrl_mean = norm_stats.get("ctrl_mean")
                    ctrl_std = norm_stats.get("ctrl_std")
                    if ctrl_mean is None or ctrl_std is None:
                        ctrl_mean = np.zeros(4, dtype=np.float32)
                        ctrl_std = np.ones(4, dtype=np.float32)
                    normalizer = Normalizer(
                        obs_mean=torch.tensor(
                            np.array(norm_stats["obs_mean"]), dtype=torch.float32
                        ),
                        obs_std=torch.tensor(
                            np.array(norm_stats["obs_std"]), dtype=torch.float32
                        ),
                        ctrl_mean=torch.tensor(
                            np.array(ctrl_mean), dtype=torch.float32
                        ),
                        ctrl_std=torch.tensor(np.array(ctrl_std), dtype=torch.float32),
                        theta_mean=torch.tensor(
                            np.array(norm_stats["theta_mean"]), dtype=torch.float32
                        ),
                        theta_std=torch.tensor(
                            np.array(norm_stats["theta_std"]), dtype=torch.float32
                        ),
                    ).to(device)
                    print(f"[NORM] Created normalizer from FNPE task stats")
            if x_test_phys is not None and normalizer is not None:
                x_test_norm = normalizer.normalize_x(x_test_phys, cfg.obs_dim).cpu()
            timing_breakdown["checkpoint_load_time_s"] = float(time.time() - load_t0)
            mark_status(
                "trained",
                state="completed",
                extra={
                    "loaded_from_checkpoint": True,
                    "checkpoint": str(cfg.checkpoint),
                },
            )
        except Exception as e:
            record_stage_failure("checkpoint_load", e, fatal=True)
            raise

    # --- Build posterior ---
    # Methods that return samples in normalized theta-space need the
    # normalizer to build a consistent posterior interface.
    posterior_build_t0 = time.time()
    try:
        if cfg.method in {"fnpe", "simformer"}:
            posterior = method.build_posterior(normalizer=normalizer)
        else:
            posterior = method.build_posterior()

        # Save pickled posterior (skip for NPSE/FNPE - they have unpicklable JAX/torch lambdas)
        if cfg.method == "npe":
            with open(exp_dir / "posterior.pkl", "wb") as f:
                pickle.dump(posterior, f)
        timing_breakdown["posterior_build_time_s"] = float(
            time.time() - posterior_build_t0
        )
        mark_status("posterior_ready", state="completed")
    except Exception as e:
        record_stage_failure("posterior_build", e, fatal=True)
        raise

    # --- Training curve plot ---
    if training_summary and not cfg.no_plots:
        train_curve_path = fig_dir / "training_loss.png"
        plot_training_curves(training_summary, train_curve_path)

    # --- Run diagnostics ---
    metrics: Dict[str, Any] = {
        "method": cfg.method,
        "training_summary": training_summary,
        "budget_metadata": build_budget_metadata(cfg, training_summary),
        "timing_breakdown": timing_breakdown,
        "dataset_artifacts": {
            "training": train_dataset_artifact,
            "heldout_test": test_dataset_artifact,
        },
        "dataset_acceptance": {
            "training": (
                train_dataset_artifact.get("acceptance")
                if isinstance(train_dataset_artifact, dict)
                else None
            ),
            "heldout_test": (
                test_dataset_artifact.get("acceptance")
                if isinstance(test_dataset_artifact, dict)
                else None
            ),
        },
    }
    if train_region_metadata is not None:
        metrics["train_region_metadata"] = train_region_metadata
    if test_region_metadata is not None:
        metrics["test_region_metadata"] = test_region_metadata

    if cfg.do_eval:
        if cfg.no_plots:
            posterior_plot_examples = 0
            pairplot_examples = 0
            if device.type == "cpu":
                pairplot_posterior_samples = 0
                c2st_examples = 0
                c2st_posterior_samples = 0
                if cfg.method == "fnpe":
                    one_step_cases = min(2, int(cfg.num_test_simulations))
                    one_step_posterior_samples = 5
                    w2_cases = min(3, int(cfg.num_test_simulations))
                    w2_posterior_samples = 5
                else:
                    one_step_cases = 5
                    one_step_posterior_samples = 10
                    w2_cases = min(10, int(cfg.num_test_simulations))
                    w2_posterior_samples = 20
            else:
                pairplot_posterior_samples = 100
                c2st_examples = 1
                c2st_posterior_samples = 100
                one_step_cases = 10
                one_step_posterior_samples = 20
                w2_cases = min(20, int(cfg.num_test_simulations))
                w2_posterior_samples = 50
        elif getattr(cfg, "unify_eval_budgets", False):
            posterior_plot_examples = int(cfg.benchmark_posterior_plot_examples)
            posterior_plot_samples = int(cfg.benchmark_posterior_plot_samples)
            pairplot_examples = int(cfg.benchmark_pairplot_examples)
            pairplot_posterior_samples = int(cfg.benchmark_pairplot_posterior_samples)
            c2st_examples = int(cfg.benchmark_c2st_examples)
            c2st_posterior_samples = int(cfg.benchmark_c2st_posterior_samples)
            one_step_cases = min(
                int(cfg.benchmark_one_step_cases), int(cfg.num_test_simulations)
            )
            one_step_posterior_samples = int(cfg.benchmark_one_step_posterior_samples)
            w2_cases = min(int(cfg.benchmark_w2_cases), int(cfg.num_test_simulations))
            w2_posterior_samples = int(cfg.benchmark_w2_posterior_samples)
        else:
            posterior_plot_examples = 3 if cfg.method == "npe" else 2
            posterior_plot_samples = 5000
            pairplot_examples = 3 if cfg.method == "npe" else 2
            pairplot_posterior_samples = 1000 if cfg.method == "npe" else 500
            c2st_examples = 3 if cfg.method == "npe" else 2
            c2st_posterior_samples = 1000 if cfg.method == "npe" else 500
            one_step_cases = 20
            one_step_posterior_samples = 100
            w2_cases = 100
            w2_posterior_samples = 500
        if not cfg.no_plots and not getattr(cfg, "unify_eval_budgets", False):
            posterior_plot_samples = 5000
        elif cfg.no_plots:
            posterior_plot_samples = 0

        shared_examples: Optional[List[Dict[str, Any]]] = None
        shared_eval_theta_phys: Optional[torch.Tensor] = theta_test_phys
        shared_eval_x_norm: Optional[torch.Tensor] = x_test_norm
        shared_eval_x_phys: Optional[torch.Tensor] = x_test_phys

        if (
            shared_eval_theta_phys is not None
            and shared_eval_x_norm is not None
            and shared_eval_x_phys is not None
            and shared_eval_theta_phys.numel() > 0
        ):
            shared_examples = build_examples_from_dataset(
                shared_eval_theta_phys,
                shared_eval_x_phys,
                shared_eval_x_norm,
                num_examples=max(
                    posterior_plot_examples,
                    pairplot_examples,
                    c2st_examples,
                    cfg.num_simulated_ppc_plot_examples,
                    2,
                ),
            )
        else:
            print(
                "[DIAG] Held-out dataset unavailable, falling back to random diagnostic examples."
            )
            try:
                shared_examples = build_shared_diagnostic_examples(
                    cfg,
                    prior=prior_phys,
                    simulator=simulator,
                    normalizer=normalizer,
                    device=device,
                    num_examples=3,
                    seed_offset=1000,
                    method=method,
                )
            except Exception as e:
                print(f"[DIAG] Failed to build shared diagnostic examples: {e}")
                shared_examples = None

        # Posterior plots
        if cfg.run_posterior_plots and shared_examples is not None:
            print("\n[DIAG] Generating posterior plots...")
            try:
                figure_metadata["posterior_plots"] = run_parameter_posterior_plots(
                    cfg,
                    fig_dir,
                    prior_phys,
                    posterior,
                    simulator,
                    normalizer,
                    device,
                    num_examples=posterior_plot_examples,
                    num_posterior_samples=posterior_plot_samples,
                    method=method,  # Pass method for FNPE
                    examples=shared_examples[:posterior_plot_examples],
                )
            except Exception as e:
                print(f"[DIAG] Posterior plots failed: {e}")
                record_stage_failure("posterior_plots", e)
        elif cfg.run_posterior_plots:
            print("[DIAG] Skipping posterior plots - no shared examples available")

        # SBC
        if cfg.run_sbc:
            print("\n[DIAG] Running SBC...")
            sbc_results = run_sbc_diagnostic(
                cfg,
                fig_dir,
                prior_norm,
                posterior,
                simulator,
                simulator_for_sbi,
                normalizer,
                device,
            )
            metrics["sbc_check_stats"] = sbc_results["check_stats"]

        # 1-step RMSE
        if cfg.run_one_step_rmse:
            print("\n[DIAG] Running 1-step RMSE...")
            try:
                one_step_results = run_one_step_rmse_diagnostic(
                    cfg,
                    prior_phys,
                    posterior,
                    simulator,
                    normalizer,
                    device,
                    num_cases=one_step_cases,
                    num_posterior_samples=one_step_posterior_samples,
                )
                metrics["one_step_rmse"] = one_step_results

                # Save separately
                with (exp_dir / "one_step_rmse.json").open("w") as f:
                    json.dump(one_step_results, f, indent=2)
            except Exception as e:
                print(f"[DIAG] 1-step RMSE failed: {e}")

        # W2 Posterior vs True (simulation-only diagnostic)
        print("\n[DIAG] Running W2 posterior vs true theta...")
        try:
            if shared_eval_theta_phys is None or shared_eval_x_norm is None:
                raise RuntimeError("shared evaluation dataset unavailable")
            theta_eval_subset, theta_eval_samples, sample_times_s = (
                sample_posterior_on_dataset(
                    cfg,
                    shared_eval_theta_phys,
                    shared_eval_x_norm,
                    posterior,
                    normalizer,
                    device,
                    num_cases=w2_cases,
                    num_posterior_samples=w2_posterior_samples,
                )
            )
            w2_post_results = run_w2_posterior_diagnostic(
                cfg,
                fig_dir=fig_dir,
                theta_test_phys=theta_eval_subset,
                x_test_norm=shared_eval_x_norm[: theta_eval_subset.shape[0]],
                posterior=posterior,
                normalizer=normalizer,
                device=device,
                num_cases=theta_eval_subset.shape[0],
                num_posterior_samples=w2_posterior_samples,
                theta_samples_phys=theta_eval_samples,
                sample_times_s=sample_times_s,
            )
            if w2_post_results:
                metrics["w2_posterior_vs_true"] = w2_post_results
                metrics["heldout_test_posterior_vs_true"] = w2_post_results
                metrics["heldout_test_per_parameter_physical"] = w2_post_results.get(
                    "per_parameter_physical", {}
                )
                metrics["heldout_test_per_parameter_normalized"] = w2_post_results.get(
                    "per_parameter_normalized", {}
                )
                # Save separately
                with (exp_dir / "w2_posterior_metrics.json").open("w") as f:
                    json.dump(w2_post_results, f, indent=2)

                heldout_stats = compute_heldout_posterior_stats(
                    cfg,
                    theta_eval_subset,
                    theta_eval_samples,
                )
                metrics["heldout_test_stats"] = heldout_stats
                with (exp_dir / "heldout_test_stats.json").open("w") as f:
                    json.dump(heldout_stats, f, indent=2)
                timing_breakdown["heldout_posterior_sampling_time_total_s"] = float(
                    w2_post_results.get("sampling_time_total_s", 0.0)
                )
                timing_breakdown["heldout_posterior_sampling_time_mean_s"] = float(
                    w2_post_results.get("sampling_time_mean_s", 0.0)
                )
                mark_status(
                    "posterior_eval_complete",
                    state="completed",
                    extra={"heldout_posterior_cases": int(theta_eval_subset.shape[0])},
                )
        except Exception as e:
            print(f"[DIAG] W2 posterior diagnostic failed: {e}")
            record_stage_failure("heldout_posterior_eval", e)

        if (
            cfg.run_simulated_ppc
            and shared_eval_x_phys is not None
            and shared_eval_x_phys.shape[0] > 0
        ):
            print("\n[DIAG] Running held-out simulated PPC...")
            try:
                simulated_ppc = run_simulated_ppc_diagnostic(
                    cfg,
                    fig_dir,
                    posterior,
                    shared_eval_x_phys,
                    shared_eval_theta_phys,
                    x_test_state0,
                    normalizer,
                    device,
                    num_examples=cfg.num_simulated_ppc_examples,
                    num_plot_examples=cfg.num_simulated_ppc_plot_examples,
                    num_posterior_samples=cfg.simulated_test_ppc_samples,
                )
                if simulated_ppc:
                    metrics["heldout_test_ppc"] = simulated_ppc
                    figure_metadata["heldout_simulated_ppc"] = simulated_ppc.get(
                        "figure_metadata", []
                    )
                    timing_breakdown["heldout_ppc_time_total_s"] = float(
                        simulated_ppc.get("aggregate", {}).get("ppc_time_total_s", 0.0)
                    )
                    timing_breakdown["heldout_ppc_time_mean_s"] = float(
                        simulated_ppc.get("aggregate", {}).get("ppc_time_mean_s", 0.0)
                    )
                    with (exp_dir / "heldout_test_ppc_metrics.json").open("w") as f:
                        json.dump(simulated_ppc, f, indent=2)
                    mark_status(
                        "ppc_complete",
                        state="completed",
                        extra={
                            "heldout_ppc_examples": int(
                                simulated_ppc.get("aggregate", {}).get(
                                    "num_examples", 0
                                )
                            )
                        },
                    )
            except Exception as e:
                print(f"[DIAG] Held-out simulated PPC failed: {e}")
                record_stage_failure("heldout_simulated_ppc", e)

        # Pairplot visualization (markovsbi-style)
        if shared_examples is not None and pairplot_examples > 0:
            print("\n[DIAG] Running pairplot diagnostics...")
            try:
                figure_metadata["pairplots"] = run_pairplot_diagnostic(
                    cfg,
                    fig_dir,
                    prior_phys,
                    posterior,
                    simulator,
                    normalizer,
                    device,
                    num_posterior_samples=pairplot_posterior_samples,
                    num_examples=pairplot_examples,
                    method=method,
                    examples=shared_examples[:pairplot_examples],
                )
            except Exception as e:
                print(f"[DIAG] Pairplot failed: {e}")
                record_stage_failure("pairplots", e)
        elif pairplot_examples > 0:
            print("[DIAG] Skipping pairplot - no shared examples available")

        # C2ST diagnostic (markovsbi-style)
        if shared_examples is not None and c2st_examples > 0:
            print("\n[DIAG] Running C2ST diagnostics...")
            try:
                c2st_results = run_c2st_diagnostic(
                    cfg,
                    fig_dir,
                    prior_phys,
                    posterior,
                    simulator,
                    normalizer,
                    device,
                    num_posterior_samples=c2st_posterior_samples,
                    num_examples=c2st_examples,
                    method=method,
                    examples=shared_examples[:c2st_examples],
                )
                if c2st_results:
                    metrics["c2st"] = c2st_results
            except Exception as e:
                print(f"[DIAG] C2ST failed: {e}")
        elif c2st_examples > 0:
            print("[DIAG] Skipping C2ST - no shared examples available")

        # Diffusion traces (FNPE only)
        if cfg.method == "fnpe" and shared_examples is not None:
            print("\n[DIAG] Running diffusion trace diagnostics...")
            try:
                figure_metadata["diffusion_traces"] = run_diffusion_traces_diagnostic(
                    cfg,
                    fig_dir,
                    posterior,
                    prior_phys,
                    simulator,
                    normalizer,
                    device,
                    num_traces=50,
                    num_examples=2,
                    method=method,
                    examples=shared_examples[:2],
                )
            except Exception as e:
                print(f"[DIAG] Diffusion traces failed: {e}")
                record_stage_failure("diffusion_traces", e)
        elif cfg.method == "fnpe":
            print("[DIAG] Skipping diffusion traces - no shared examples available")

        mark_status(
            "evaluation_complete",
            state="completed",
            extra={"stage_failures": stage_failures},
        )

    # --- Real data eval (inline) ---
    if cfg.real_data_csv and cfg.do_eval:
        print("\n[EVAL] Running real-data evaluation (inline)...")
        real_metrics = run_real_data_evaluation(
            cfg=cfg,
            exp_dir=exp_dir,
            fig_dir=fig_dir,
            posterior=posterior,
            prior_phys=prior_phys,
            simulator=simulator,
            normalizer=normalizer,
            device=device,
            T_event=T_event,
            K_ppc=cfg.K_ppc,
        )

        # Multi-trajectory PPC over all CSVs in the measurements directory
        try:
            multi_ppc_metrics = run_multi_trajectory_ppc(
                cfg=cfg,
                exp_dir=exp_dir,
                fig_dir=fig_dir,
                posterior=posterior,
                normalizer=normalizer,
                device=device,
                T_event=T_event,
                data_dir=cfg.real_data_dir,
                K_ppc=cfg.K_ppc,
            )
            if multi_ppc_metrics:
                metrics["multi_traj_ppc"] = multi_ppc_metrics.get("aggregate", {})
        except Exception as _e:
            print(f"[EVAL] Multi-trajectory PPC failed: {_e}")
            record_stage_failure("multi_traj_ppc", _e)
        if real_metrics:
            metrics["real_metrics"] = real_metrics

    # --- Save config and metrics ---
    cfg.save(str(exp_dir / "config.json"))
    metrics["timing_breakdown"] = timing_breakdown
    metrics["dataset_artifacts"] = {
        "training": train_dataset_artifact,
        "heldout_test": test_dataset_artifact,
    }
    metrics["dataset_acceptance"] = {
        "training": (
            train_dataset_artifact.get("acceptance")
            if isinstance(train_dataset_artifact, dict)
            else None
        ),
        "heldout_test": (
            test_dataset_artifact.get("acceptance")
            if isinstance(test_dataset_artifact, dict)
            else None
        ),
    }
    metrics["figure_metadata"] = figure_metadata
    metrics["stage_failures"] = stage_failures
    metrics["run_status_summary"] = {
        "completed_stages": completed_stages,
        "final_state": "complete_with_failures" if stage_failures else "complete",
    }

    metrics_path = exp_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2, default=tensor_to_python)

    mark_status(
        "complete",
        state="complete_with_failures" if stage_failures else "complete",
        extra={
            "metrics_path": str(metrics_path),
            "stage_failures": stage_failures,
            "available_artifacts": sorted(path.name for path in exp_dir.iterdir()),
        },
    )

    cfg.assert_budget_resolution_ready("run_experiment")
    cfg.ensure_dataset_ids()
    print(f"\n{'='*60}")
    print(f"Experiment completed: {exp_dir}")
    print(f"{'='*60}\n")

    return {
        "exp_dir": exp_dir,
        "method": method,
        "posterior": posterior,
        "normalizer": normalizer,
        "metrics": metrics,
    }
