"""
Unified experiment runner for all SBI methods.

This module provides a method-agnostic experiment pipeline that:
- Uses identical data generation and normalization for all methods
- Provides consistent evaluation metrics
- Supports dataset caching for fair comparisons
- Saves results in a uniform format
"""

import json
import pickle
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

import torch
import numpy as np
from sbi import utils as sbi_utils
from sbi.diagnostics import run_sbc, check_sbc

from configs.config import ExperimentConfig
from utils.env_utils import setup_environment, get_device
from utils.metrics import (
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


def _sample_theta_from_prior_cpu(prior_cpu, seed: int) -> torch.Tensor:
    """Sample one physical-theta draw deterministically on CPU."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return prior_cpu.sample((1,))


# =============================================================================
# Dataset Management (with caching for fair comparisons)
# =============================================================================


def get_or_generate_dataset(
    cfg: ExperimentConfig,
    prior,
    simulator,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
    """
    Get dataset from cache or generate new one.

    Returns:
        theta_train: Training parameters (physical space)
        x_train: Training observations (physical space)
        controls: Control inputs (if available)
    """
    cache_path = cfg.get_dataset_cache_path()

    # Try loading from cache
    if cfg.reuse_dataset and cache_path.exists():
        print(f"[DATA] Loading cached dataset from {cache_path}")
        cached = torch.load(cache_path)
        return cached["theta"], cached["x"], cached.get("controls")

    # Generate new dataset
    print(f"[DATA] Generating {cfg.num_simulations} simulations...")
    theta_train, x_train, controls = generate_dataset(
        cfg, prior, simulator, show_pbar=True
    )

    # Cache if requested
    if cfg.cache_dataset:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[DATA] Caching dataset to {cache_path}")
        torch.save(
            {
                "theta": theta_train,
                "x": x_train,
                "controls": controls,
                "config_hash": cfg.dataset_id,
            },
            cache_path,
        )

    return theta_train, x_train, controls


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
) -> None:
    """Generate prior vs posterior marginal plots."""
    if cfg.no_plots:
        return

    param_names = list(cfg.active_parameters)

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
        return

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
        x_cond = ex["x_cond"]

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
    # Reduce samples for slower methods
    num_sbc = cfg.num_sbc_samples
    num_post = cfg.num_posterior_samples_sbc

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

    sbc_device = torch.device("cpu") if cfg.method == "simformer" else device
    if hasattr(posterior, "to"):
        posterior.to(sbc_device)

    cuda_devices = []
    if sbc_device.type == "cuda":
        cuda_devices = [
            sbc_device.index
            if sbc_device.index is not None
            else torch.cuda.current_device()
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
) -> None:
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
        return

    print("[PAIRPLOT] Generating pairplot visualizations...")

    param_names = list(cfg.active_parameters)

    # Reduce samples for slower methods
    if cfg.method in ["npse", "fnpe"]:
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
        return

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
        x_cond = ex["x_cond"]

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

    # Reduce samples for slower methods
    if cfg.method in ["npse", "fnpe"]:
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

        x_cond = ex["x_cond"]

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
) -> None:
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
        return

    # Only for FNPE with trace support
    if cfg.method != "fnpe":
        print("[TRACES] Diffusion traces only available for FNPE, skipping...")
        return

    if not hasattr(posterior, "sample_with_traces"):
        print("[TRACES] Posterior doesn't support sample_with_traces, skipping...")
        return

    print("[TRACES] Generating diffusion trace plots...")

    param_names = list(cfg.active_parameters)

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
            except Exception as e:
                print(f"[TRACES] Failed to get traces for example {ex_idx}: {e}")


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
) -> Dict[str, Any]:
    """
    Compute Wasserstein-style metrics comparing posterior to ground truth theta.

    This is particularly useful for simulation-only experiments where we have
    true theta values for each observation.
    """
    # Reduce samples for slower methods
    if cfg.method in ["npse", "fnpe", "simformer"]:
        num_cases = min(num_cases, 50)
        num_posterior_samples = min(num_posterior_samples, 200)

    print(f"[W2-POST] Computing posterior vs true theta metrics ({num_cases} cases)...")

    # Select subset
    N = min(num_cases, theta_test_phys.shape[0])
    theta_sub_phys = theta_test_phys[:N]  # (N, d) in physical units
    x_sub = x_test_norm[:N]  # (N, T, D) normalized

    # Sample from posterior for each observation
    all_samples = []
    sample_times_s = []
    for i in range(N):
        x_i = x_sub[i : i + 1]  # (1, T, D) or (T, D)
        try:
            t0 = time.time()
            samples_i = posterior.sample((num_posterior_samples,), x=x_i.to(device))
            sample_times_s.append(time.time() - t0)
            if isinstance(samples_i, np.ndarray):
                samples_i = torch.from_numpy(samples_i).float()
            samples_i = normalizer.unnormalize_theta(samples_i.to(device))
            all_samples.append(samples_i.cpu())
        except Exception as e:
            print(f"[W2-POST] Warning: Failed to sample for case {i}: {e}")
            continue

    if len(all_samples) == 0:
        print("[W2-POST] No successful samples, skipping metric.")
        return {}

    # Stack samples: (N_success, K, d)
    theta_samples = torch.stack(all_samples, dim=0)
    N_success = theta_samples.shape[0]
    theta_true_sub = theta_sub_phys[:N_success].cpu()

    # Compute metrics
    w2_results = wasserstein2_posterior_vs_true(
        theta_true_sub,
        theta_samples,
        num_projections=cfg.num_swd_projections,
        seed=cfg.random_seed,
    )
    if sample_times_s:
        w2_results["sampling_time_mean_s"] = float(np.mean(sample_times_s))
        w2_results["sampling_time_std_s"] = float(np.std(sample_times_s))

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
    # Reduce samples for slower methods
    if cfg.method in ["npse", "fnpe", "simformer"]:
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
    print(
        f"[SETUP] input_dim={D_in}, T_event={T_event}, d_theta={cfg.active_param_dim()}"
    )

    # --- Create experiment directory ---
    exp_dir = make_experiment_dir(cfg)
    fig_dir = exp_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"[SETUP] Experiment dir: {exp_dir}")
    print(f"[SETUP] Directory exists: {exp_dir.exists()}")

    # --- Get or generate dataset (skip for FNPE - it generates its own) ---
    if cfg.method == "fnpe":
        print(
            "[DATA] Skipping dataset generation for FNPE (it generates its own data)",
            flush=True,
        )
        # For FNPE, we still need normalization stats, but we'll get them from the task
        # Create dummy tensors just to set up the normalizer structure
        theta_train_phys = None
        x_train_phys = None
        normalizer = None
        theta_train = None
        x_train = None
    else:
        theta_train_phys, x_train_phys, _ = get_or_generate_dataset(
            cfg, prior_phys, simulator, device
        )

        # --- Fit normalization ---
        normalizer_cpu = fit_normalizer(theta_train_phys, x_train_phys, cfg.obs_dim)
        norm_path = exp_dir / "stats_normalization.json"
        print(
            f"[NORM] Saving to {norm_path}, parent exists: {norm_path.parent.exists()}"
        )
        save_normalizer(normalizer_cpu, norm_path)
        print(f"[NORM] Saved stats to {norm_path}")

        normalizer = normalizer_cpu.to(device)

        # --- Normalize data ---
        theta_train = normalizer_cpu.normalize_theta(theta_train_phys)
        x_train = normalizer_cpu.normalize_x(x_train_phys, cfg.obs_dim)

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
                "token_dim": cfg.simformer_token_dim,
                "condition_token_dim": cfg.simformer_condition_token_dim,
                "time_embedding_dim": cfg.simformer_time_embedding_dim,
                "num_heads": cfg.simformer_num_heads,
                "num_layers": cfg.simformer_num_layers,
                "attn_size": cfg.simformer_attn_size,
                "widening_factor": cfg.simformer_widening_factor,
                "sigma_min": cfg.simformer_sigma_min,
                "sigma_max": cfg.simformer_sigma_max,
                "T_min": cfg.simformer_t_min,
                "T_max": cfg.simformer_t_max,
                "num_diffusion_steps": cfg.simformer_num_diffusion_steps,
                "learning_rate": cfg.simformer_learning_rate,
                "num_train_steps": cfg.simformer_num_train_steps,
                "batch_size": cfg.simformer_batch_size,
            }
        )
        print(
            f"[Simformer] layers={cfg.simformer_num_layers}, heads={cfg.simformer_num_heads}, "
            f"train_steps={cfg.simformer_num_train_steps}",
            flush=True,
        )
    elif cfg.method == "fnpe":
        method_kwargs.update(
            {
                "hidden_dim": cfg.fnpe_hidden_dim,
                "num_hidden": cfg.fnpe_num_hidden,
                "model_type": cfg.fnpe_model_type,
                "window_size": cfg.fnpe_window_size,  # CRITICAL: small Markov window
                "num_epochs": cfg.num_epochs,
                "steps_per_epoch": cfg.fnpe_steps_per_epoch,
                "batch_size": cfg.training_batch_size,
                "num_diffusion_steps": cfg.fnpe_num_diffusion_steps,
                "score_fn_type": cfg.fnpe_score_fn_type,
                "stop_after_epochs": cfg.stop_after_epochs,
                "validation_fraction": cfg.validation_fraction,
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

    method = build_method(cfg.method, cfg, prior_norm, device, **method_kwargs)
    method.build(input_dim=D_in, seq_len=T_event)

    # --- Train ---
    training_summary = {}
    if cfg.do_train:
        print(f"\n[TRAIN] Training {cfg.method.upper()}...", flush=True)
        setup_environment(cfg.train_seed)  # Use train seed

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
                    theta_mean=torch.zeros(cfg.active_param_dim(), dtype=torch.float32),
                    theta_std=torch.ones(cfg.active_param_dim(), dtype=torch.float32),
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
                    ctrl_mean=torch.tensor(np.array(ctrl_mean), dtype=torch.float32),
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
        else:
            training_summary = method.train(theta_train, x_train)

        # Save model
        model_path = method.save(exp_dir)
        print(f"[TRAIN] Saved model to {model_path}", flush=True)

    elif cfg.checkpoint:
        print(f"\n[LOAD] Loading checkpoint from {cfg.checkpoint}", flush=True)
        method.load(Path(cfg.checkpoint))
        # For FNPE checkpoint, also load/create normalizer
        if cfg.method == "fnpe":
            norm_path = Path(cfg.checkpoint).parent / "stats_normalization.json"
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
                    ctrl_mean=torch.tensor(np.array(ctrl_mean), dtype=torch.float32),
                    ctrl_std=torch.tensor(np.array(ctrl_std), dtype=torch.float32),
                    theta_mean=torch.tensor(
                        np.array(norm_stats["theta_mean"]), dtype=torch.float32
                    ),
                    theta_std=torch.tensor(
                        np.array(norm_stats["theta_std"]), dtype=torch.float32
                    ),
                ).to(device)
                print(f"[NORM] Created normalizer from FNPE task stats")

    # --- Build posterior ---
    # Methods that return samples in normalized theta-space need the
    # normalizer to build a consistent posterior interface.
    if cfg.method in {"fnpe", "simformer"}:
        posterior = method.build_posterior(normalizer=normalizer)
    else:
        posterior = method.build_posterior()

    # Save pickled posterior (skip for NPSE/FNPE - they have unpicklable JAX/torch lambdas)
    if cfg.method == "npe":
        with open(exp_dir / "posterior.pkl", "wb") as f:
            pickle.dump(posterior, f)

    # --- Training curve plot ---
    if training_summary and not cfg.no_plots:
        train_curve_path = fig_dir / "training_loss.png"
        plot_training_curves(training_summary, train_curve_path)

    # --- Run diagnostics ---
    metrics: Dict[str, Any] = {
        "method": cfg.method,
        "training_summary": training_summary,
    }

    if cfg.do_eval:
        # ---------------------------------------------------------------------
        # Build shared synthetic examples so ex_idx refers to the same
        # (theta_true, x_cond) across all diagnostics.
        # ---------------------------------------------------------------------
        shared_examples: Optional[List[Dict[str, Any]]] = None
        shared_eval_theta_phys: Optional[torch.Tensor] = None
        shared_eval_x_norm: Optional[torch.Tensor] = None
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

        try:
            shared_eval_theta_phys, shared_eval_x_norm = build_shared_eval_dataset(
                cfg,
                prior=prior_phys,
                simulator=simulator,
                normalizer=normalizer,
                device=device,
                num_cases=100,
                seed_offset=3000,
                method=method,
            )
        except Exception as e:
            print(f"[DIAG] Failed to build shared evaluation dataset: {e}")
            shared_eval_theta_phys = None
            shared_eval_x_norm = None

        # Posterior plots
        if cfg.run_posterior_plots and shared_examples is not None:
            print("\n[DIAG] Generating posterior plots...")
            num_ex = 3 if cfg.method == "npe" else 2
            run_parameter_posterior_plots(
                cfg,
                fig_dir,
                prior_phys,
                posterior,
                simulator,
                normalizer,
                device,
                num_examples=num_ex,
                num_posterior_samples=5000,
                method=method,  # Pass method for FNPE
                examples=shared_examples[:num_ex],
            )
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
            w2_post_results = run_w2_posterior_diagnostic(
                cfg,
                fig_dir=fig_dir,
                theta_test_phys=shared_eval_theta_phys,
                x_test_norm=shared_eval_x_norm,
                posterior=posterior,
                normalizer=normalizer,
                device=device,
                num_cases=100,
                num_posterior_samples=500,
            )
            if w2_post_results:
                metrics["w2_posterior_vs_true"] = w2_post_results
                # Save separately
                with (exp_dir / "w2_posterior_metrics.json").open("w") as f:
                    json.dump(w2_post_results, f, indent=2)
        except Exception as e:
            print(f"[DIAG] W2 posterior diagnostic failed: {e}")

        # Pairplot visualization (markovsbi-style)
        if shared_examples is not None:
            print("\n[DIAG] Running pairplot diagnostics...")
            try:
                num_ex = 3 if cfg.method == "npe" else 2
                run_pairplot_diagnostic(
                    cfg,
                    fig_dir,
                    prior_phys,
                    posterior,
                    simulator,
                    normalizer,
                    device,
                    num_posterior_samples=1000 if cfg.method == "npe" else 500,
                    num_examples=num_ex,
                    method=method,
                    examples=shared_examples[:num_ex],
                )
            except Exception as e:
                print(f"[DIAG] Pairplot failed: {e}")
        else:
            print("[DIAG] Skipping pairplot - no shared examples available")

        # C2ST diagnostic (markovsbi-style)
        if shared_examples is not None:
            print("\n[DIAG] Running C2ST diagnostics...")
            try:
                num_ex = 3 if cfg.method == "npe" else 2
                c2st_results = run_c2st_diagnostic(
                    cfg,
                    fig_dir,
                    prior_phys,
                    posterior,
                    simulator,
                    normalizer,
                    device,
                    num_posterior_samples=1000 if cfg.method == "npe" else 500,
                    num_examples=num_ex,
                    method=method,
                    examples=shared_examples[:num_ex],
                )
                if c2st_results:
                    metrics["c2st"] = c2st_results
            except Exception as e:
                print(f"[DIAG] C2ST failed: {e}")
        else:
            print("[DIAG] Skipping C2ST - no shared examples available")

        # Diffusion traces (FNPE only)
        if cfg.method == "fnpe" and shared_examples is not None:
            print("\n[DIAG] Running diffusion trace diagnostics...")
            try:
                run_diffusion_traces_diagnostic(
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
        elif cfg.method == "fnpe":
            print("[DIAG] Skipping diffusion traces - no shared examples available")

    # --- Real data eval (inline) ---
    # Simformer/comparison runs are simulation-only; skip real-data inference.
    if cfg.real_data_csv and cfg.do_eval and cfg.method != "simformer":
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
        if real_metrics:
            metrics["real_metrics"] = real_metrics
    elif cfg.real_data_csv and cfg.do_eval and cfg.method == "simformer":
        print(
            "[EVAL] Skipping real-data evaluation for Simformer (simulation-only mode)."
        )

    # --- Save config and metrics ---
    cfg.save(str(exp_dir / "config.json"))

    with (exp_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2, default=tensor_to_python)

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
