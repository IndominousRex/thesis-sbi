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
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

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
    plot_sbc_rank_hist,
    plot_training_curves,
    plot_ppc_trajectories,
    plot_obs_1d_hist_custom,
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
    num_prior_samples: int = 20000,
    num_posterior_samples: int = 5000,
    method=None,  # Pass method for FNPE to use its own data generation
) -> None:
    """Generate prior vs posterior marginal plots."""
    if cfg.no_plots:
        return

    param_names = list(cfg.active_parameters)

    # For FNPE, use JAX-based prior sampling
    if cfg.method == "fnpe" and method is not None and hasattr(method, "task"):
        import jax
        import jax.numpy as jnp

        task = method.task
        jax_prior = task.get_prior()
        key = jax.random.PRNGKey(cfg.random_seed + 1000)

        # Sample from JAX prior for reference
        key, key_prior = jax.random.split(key)
        prior_pool_jax = jax_prior.sample(key_prior, (num_prior_samples,))
        prior_pool_np = np.array(prior_pool_jax)

        for ex_idx in range(num_examples):
            print(f"[DIAG] Posterior plots (FNPE): example {ex_idx+1}/{num_examples}")

            # Sample true theta and simulate
            key, key_theta, key_sim = jax.random.split(key, 3)
            theta_true_jax = jax_prior.sample(key_theta, (1,))[0]
            theta_true_np = np.array(theta_true_jax)

            # Generate observation using FNPE's task simulator
            simulator_fn = task.get_simulator()
            x_phys = simulator_fn(key_sim, theta_true_jax, cfg.T_seg)  # (T, obs_dim)

            # Sample from posterior - pass physical observation (FNPE normalizes internally)
            theta_post = posterior.sample((num_posterior_samples,), x=x_phys)
            theta_post_np = np.array(theta_post)

            if theta_post_np.ndim == 3:
                theta_post_np = theta_post_np.reshape(-1, theta_post_np.shape[-1])

            for j, pname in enumerate(param_names):
                out_path = fig_dir / f"prior_posterior_{pname}_ex{ex_idx}.png"
                plot_prior_posterior_1d(
                    prior_1d=prior_pool_np[:, j],
                    post_1d=theta_post_np[:, j],
                    theta_ref=theta_true_np[j],
                    param_name=pname,
                    out_path=out_path,
                    bins=80,
                )
        return

    # Standard path for NPE/NPSE
    with torch.no_grad():
        prior_pool = prior.sample((num_prior_samples,)).to(device)
    prior_pool_np = prior_pool.detach().cpu().numpy()

    for ex_idx in range(num_examples):
        print(f"[DIAG] Posterior plots: example {ex_idx+1}/{num_examples}")

        with torch.no_grad():
            theta_true = prior.sample((1,)).to(device)
        theta_true_np = theta_true.detach().cpu().numpy()[0]

        sim_out = simulator(theta_true)
        x_sim = sim_out[0] if isinstance(sim_out, tuple) else sim_out
        x_cond = normalizer.normalize_x(x_sim, cfg.obs_dim).to(device)

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

        for j, pname in enumerate(param_names):
            out_path = fig_dir / f"prior_posterior_{pname}_ex{ex_idx}.png"
            plot_prior_posterior_1d(
                prior_1d=prior_pool_np[:, j],
                post_1d=theta_post_np[:, j],
                theta_ref=theta_true_np[j],
                param_name=pname,
                out_path=out_path,
                bins=80,
            )


def run_sbc_diagnostic(
    cfg: ExperimentConfig,
    fig_dir: Path,
    prior_norm,
    posterior,
    simulator_for_sbi,
    normalizer: Normalizer,
    device: torch.device,
) -> Dict[str, Any]:
    """Run Simulation-Based Calibration diagnostic."""
    # Reduce samples for slower methods
    num_sbc = cfg.num_sbc_samples
    num_post = cfg.num_posterior_samples_sbc

    if cfg.method in ["npse", "fnpe"]:
        num_sbc = min(num_sbc, 100)
        num_post = min(num_post, 500)

    print(f"[SBC] Running with {num_sbc} samples, {num_post} posterior samples each...")

    theta_sbc_norm = prior_norm.sample((num_sbc,)).to(device)
    theta_sbc_phys = normalizer.unnormalize_theta(theta_sbc_norm)
    x_sbc_phys = simulator_for_sbi(theta_sbc_phys)
    x_sbc_norm = normalizer.normalize_x(x_sbc_phys, cfg.obs_dim)

    ranks, dap_samples_norm = run_sbc(
        thetas=theta_sbc_norm,
        xs=x_sbc_norm,
        posterior=posterior,
        num_posterior_samples=num_post,
        num_workers=4,
        use_sample_batched=False,
    )

    check_stats = check_sbc(ranks, theta_sbc_norm, dap_samples_norm, num_post)
    print("SBC check statistics:", check_stats)

    if not cfg.no_plots:
        sbc_fig_path = fig_dir / "sbc_rank_hist.png"
        plot_sbc_rank_hist(ranks, num_post, sbc_fig_path)

    return {
        "check_stats": {k: tensor_to_python(v) for k, v in check_stats.items()},
        "dap_samples_norm": dap_samples_norm,
    }


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
    if cfg.method in ["npse", "fnpe"]:
        num_posterior_samples = min(num_posterior_samples, 50)

    print(f"[1-STEP] Running RMSE diagnostic ({num_cases} cases)...")

    obs_dim = cfg.obs_dim
    state_dim = cfg.state_dim

    all_true_next = []
    all_pred_next = []

    for i in range(num_cases):
        with torch.no_grad():
            theta_true = prior.sample((1,)).to(device)

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
        x_cond = normalizer.normalize_x(x_sim[:, :1, :], cfg.obs_dim).to(device)

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

    # FNPE uses incompatible data format
    if cfg.method == "fnpe":
        print(
            "[REAL] FNPE uses incompatible data format - skipping real data evaluation"
        )
        return {}

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
            prefer_low_brake=True,
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
    print(f"Simulations: {cfg.num_simulations}")
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

    # --- Get or generate dataset ---
    theta_train_phys, x_train_phys, _ = get_or_generate_dataset(
        cfg, prior_phys, simulator, device
    )

    # --- Fit normalization ---
    normalizer_cpu = fit_normalizer(theta_train_phys, x_train_phys, cfg.obs_dim)
    norm_path = exp_dir / "stats_normalization.json"
    save_normalizer(normalizer_cpu, norm_path)
    print(f"[NORM] Saved stats to {norm_path}")

    normalizer = normalizer_cpu.to(device)

    # --- Normalize data ---
    theta_train = normalizer_cpu.normalize_theta(theta_train_phys)
    x_train = normalizer_cpu.normalize_x(x_train_phys, cfg.obs_dim)

    # --- Build normalized prior ---
    bounds = cfg.param_bounds()
    low_list = [bounds[name][0] for name in cfg.active_parameters]
    high_list = [bounds[name][1] for name in cfg.active_parameters]
    theta_low_norm = normalizer.normalize_theta(
        torch.tensor(low_list, dtype=torch.float32, device=device)
    )
    theta_high_norm = normalizer.normalize_theta(
        torch.tensor(high_list, dtype=torch.float32, device=device)
    )
    prior_norm = sbi_utils.BoxUniform(low=theta_low_norm, high=theta_high_norm)

    # --- Build method ---
    print(f"\n[METHOD] Building {cfg.method.upper()}...")

    method_kwargs = {}
    if cfg.method == "npse":
        method_kwargs["sde_type"] = cfg.sde_type
    elif cfg.method == "fnpe":
        method_kwargs.update(
            {
                "hidden_dim": cfg.fnpe_hidden_dim,
                "num_hidden": cfg.fnpe_num_hidden,
                "model_type": cfg.fnpe_model_type,
                "num_epochs": cfg.num_epochs,
                "steps_per_epoch": cfg.fnpe_steps_per_epoch,
                "batch_size": cfg.training_batch_size,
                "num_diffusion_steps": cfg.fnpe_num_diffusion_steps,
                "score_fn_type": cfg.fnpe_score_fn_type,
            }
        )

    method = build_method(cfg.method, cfg, prior_norm, device, **method_kwargs)
    method.build(input_dim=D_in, seq_len=T_event)

    # --- Train ---
    training_summary = {}
    if cfg.do_train:
        print(f"\n[TRAIN] Training {cfg.method.upper()}...")
        setup_environment(cfg.train_seed)  # Use train seed

        if cfg.method == "fnpe":
            # FNPE generates its own data
            training_summary = method.train(
                num_simulations=cfg.num_simulations, T_obs=T_event
            )
        else:
            training_summary = method.train(theta_train, x_train)

        # Save model
        model_path = method.save(exp_dir)
        print(f"[TRAIN] Saved model to {model_path}")

    elif cfg.checkpoint:
        print(f"\n[LOAD] Loading checkpoint from {cfg.checkpoint}")
        method.load(Path(cfg.checkpoint))

    # --- Build posterior ---
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
        # Posterior plots
        if cfg.run_posterior_plots:
            print("\n[DIAG] Generating posterior plots...")
            run_parameter_posterior_plots(
                cfg,
                fig_dir,
                prior_phys,
                posterior,
                simulator,
                normalizer,
                device,
                num_posterior_samples=5000 if cfg.method == "npe" else 2000,
                method=method,  # Pass method for FNPE
            )

        # SBC (skip for FNPE - incompatible data format)
        if cfg.run_sbc and cfg.method != "fnpe":
            print("\n[DIAG] Running SBC...")
            sbc_results = run_sbc_diagnostic(
                cfg,
                fig_dir,
                prior_norm,
                posterior,
                simulator_for_sbi,
                normalizer,
                device,
            )
            metrics["sbc_check_stats"] = sbc_results["check_stats"]

            # SWD
            if cfg.run_swd:
                swd_val = run_swd_diagnostic(
                    cfg, prior_phys, sbc_results["dap_samples_norm"], normalizer
                )
                metrics["swd_prior_vs_dap"] = float(swd_val)
        elif cfg.run_sbc and cfg.method == "fnpe":
            print("\n[DIAG] Skipping SBC for FNPE (incompatible data format)")

        # 1-step RMSE (skip for FNPE - incompatible data format)
        if cfg.run_one_step_rmse and cfg.method != "fnpe":
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
            K_ppc=300,
        )
        if real_metrics:
            metrics["real_metrics"] = real_metrics

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
