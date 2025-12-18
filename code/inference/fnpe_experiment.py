"""
FNPE Experiment Runner for Vehicle Dynamics.

This module provides an end-to-end experiment pipeline using the MarkovSBI
framework (Flow-based Neural Posterior Estimation via diffusion).
"""

import json
import pickle
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import optax


def log(msg: str):
    """Print with timestamp and flush immediately."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# MarkovSBI imports
from markovsbi.tasks import VehicleDynamicsTask
from markovsbi.utils.sde_utils import init_sde
from markovsbi.models.simple_scoremlp import build_score_mlp, precondition_functions
from markovsbi.models.train_utils import build_batch_sampler, build_loss_fn
from markovsbi.sampling.score_fn import FNPEScoreFn, UncorrectedScoreFn
from markovsbi.sampling.sample import Diffuser
from markovsbi.sampling.kernels import EulerMaruyama

# Your existing imports
from configs.config import ExperimentConfig
from utils.env_utils import setup_environment


def make_fnpe_experiment_dir(cfg: ExperimentConfig, prefix: str = "fnpe") -> Path:
    """Create experiment directory."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path(cfg.results_root)
    root.mkdir(parents=True, exist_ok=True)
    exp_dir = root / f"{prefix}_{cfg.exp_name}_{ts}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def train_score_network(
    data: Dict[str, jnp.ndarray],
    sde,
    weight_fn,
    *,
    window_size: int,
    hidden_dim: int = 128,
    num_hidden: int = 5,
    model_type: str = "gru",
    num_epochs: int = 20,
    steps_per_epoch: int = 10000,
    batch_size: int = 256,
    learning_rate: float = 5e-4,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[Any, Any, list]:
    """
    Train a score network using denoising score matching.

    Returns:
        params: Trained network parameters
        score_net: Score network function
        losses: List of epoch losses
    """
    if verbose:
        log("[TRAIN] Starting score network setup...")

    key = jax.random.PRNGKey(seed)
    key, key_init = jax.random.split(key)

    d = data["thetas"].shape[1]

    # Preconditioning
    if verbose:
        log("[TRAIN] Building preconditioning functions...")
    c_in, c_noise, c_out = precondition_functions(sde)

    # Build score network
    if verbose:
        log(
            f"[TRAIN] Building score network (hidden={hidden_dim}, layers={num_hidden}, type={model_type})..."
        )
    init_fn, score_net = build_score_mlp(
        window_size=window_size,
        num_hidden=num_hidden,
        hidden_dim=hidden_dim,
        x_o_processing=model_type,
        c_in=c_in,
        c_noise=c_noise,
        c_out=c_out,
    )

    # Build batch sampler and loss
    if verbose:
        log("[TRAIN] Building batch sampler and loss function...")
    batch_sampler = build_batch_sampler(data)
    loss_fn = build_loss_fn("dsm", score_net, sde, weight_fn, control_variate=True)

    # Initialize
    if verbose:
        log("[TRAIN] Initializing network parameters...")
    theta_batch, x_batch = batch_sampler(key_init, batch_size)
    params = init_fn(key_init, jnp.ones((batch_size,)), theta_batch, x_batch)

    if verbose:
        n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
        log(f"[TRAIN] Score network initialized: {n_params:,} parameters")

    # Optimizer
    total_steps = num_epochs * steps_per_epoch
    if verbose:
        log(f"[TRAIN] Setting up optimizer (total_steps={total_steps})...")
    schedule = optax.cosine_onecycle_schedule(total_steps, learning_rate)
    optimizer = optax.chain(
        optax.adaptive_grad_clip(10.0),
        optax.adamw(schedule),
    )
    opt_state = optimizer.init(params)

    # JIT update
    @jax.jit
    def update(params, rng, opt_state, theta_batch, x_batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, rng, theta_batch, x_batch)
        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        return loss, params, opt_state

    # JIT warmup - first call compiles the function
    if verbose:
        log("[TRAIN] JIT compiling update function (this may take a while)...")

    key, key_batch, key_loss = jax.random.split(key, 3)
    theta_batch, x_batch = batch_sampler(key_batch, batch_size)
    t0 = time.time()
    loss, params, opt_state = update(params, key_loss, opt_state, theta_batch, x_batch)
    # Block until computation is done
    loss_val = float(loss)
    if verbose:
        log(
            f"[TRAIN] JIT compilation done in {time.time() - t0:.1f}s, first loss = {loss_val:.6f}"
        )

    # Training loop
    losses = []
    train_start = time.time()
    for epoch in range(num_epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        for step in range(steps_per_epoch):
            key, key_batch, key_loss = jax.random.split(key, 3)
            theta_batch, x_batch = batch_sampler(key_batch, batch_size)
            loss, params, opt_state = update(
                params, key_loss, opt_state, theta_batch, x_batch
            )
            epoch_loss += float(loss) / steps_per_epoch

            # Progress update every 1000 steps
            if verbose and (step + 1) % 1000 == 0:
                log(
                    f"[TRAIN] Epoch {epoch+1}/{num_epochs}, Step {step+1}/{steps_per_epoch}"
                )

        losses.append(epoch_loss)
        epoch_time = time.time() - epoch_start
        if verbose:
            log(
                f"[TRAIN] Epoch {epoch+1}/{num_epochs}: Loss = {epoch_loss:.6f} ({epoch_time:.1f}s)"
            )

    if verbose:
        total_time = time.time() - train_start
        log(f"[TRAIN] Training complete in {total_time:.1f}s ({total_time/60:.1f} min)")

    return params, score_net, losses


def run_fnpe_experiment(
    cfg: ExperimentConfig,
    *,
    num_simulations: int = 100000,
    T_obs: int = 100,
    hidden_dim: int = 128,
    num_hidden: int = 5,
    model_type: str = "gru",
    num_epochs: int = 20,
    steps_per_epoch: int = 10000,
    batch_size: int = 256,
    learning_rate: float = 5e-4,
    num_diffusion_steps: int = 500,
    num_posterior_samples: int = 1000,
    score_fn_type: str = "fnpe",  # "fnpe" or "uncorrected"
) -> Dict[str, Any]:
    """
    Run full FNPE experiment.

    Args:
        cfg: ExperimentConfig
        num_simulations: Training data size
        T_obs: Observation trajectory length
        hidden_dim: Score network hidden dimension
        num_hidden: Number of hidden layers
        model_type: Score network observation processing ("gru" or "linear")
        num_epochs: Training epochs
        steps_per_epoch: Steps per epoch
        batch_size: Training batch size
        learning_rate: Peak learning rate
        num_diffusion_steps: Reverse diffusion steps
        num_posterior_samples: Samples for diagnostics
        score_fn_type: "fnpe" or "uncorrected"

    Returns:
        Dict with experiment results
    """
    experiment_start = time.time()
    setup_environment(cfg.random_seed)

    exp_dir = make_fnpe_experiment_dir(cfg)
    fig_dir = exp_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 60)
    log("FNPE EXPERIMENT STARTING")
    log("=" * 60)
    log(f"Experiment dir: {exp_dir}")
    log(f"Active parameters: {cfg.active_parameters}")
    log(f"Config: num_sim={num_simulations}, T_obs={T_obs}, epochs={num_epochs}")
    log(f"        steps_per_epoch={steps_per_epoch}, batch_size={batch_size}")
    log(f"        model_type={model_type}, score_fn_type={score_fn_type}")

    # 1. Create task
    log("[STEP 1/6] Creating VehicleDynamicsTask...")
    t0 = time.time()
    task = VehicleDynamicsTask(cfg=cfg, normalize=True, seed=cfg.random_seed)
    prior = task.get_prior()
    log(f"[STEP 1/6] Task created in {time.time() - t0:.1f}s")

    # 2. Generate data
    log(f"[STEP 2/6] Generating {num_simulations} training trajectories (T={T_obs})...")
    t0 = time.time()
    key = jax.random.PRNGKey(cfg.random_seed)
    key, key_data = jax.random.split(key)
    data = task.get_data(key_data, num_simulations, T_obs)
    log(f"[STEP 2/6] Data generation done in {time.time() - t0:.1f}s")
    log(f"           Shapes: thetas={data['thetas'].shape}, xs={data['xs'].shape}")

    # 3. Initialize SDE
    log("[STEP 3/6] Initializing SDE...")
    t0 = time.time()
    sde, weight_fn = init_sde(data)
    log(f"[STEP 3/6] SDE initialized in {time.time() - t0:.1f}s")

    # 4. Train score network
    log(
        f"[STEP 4/6] Training score network ({num_epochs} epochs, model={model_type})..."
    )
    t0 = time.time()
    params, score_net, losses = train_score_network(
        data,
        sde,
        weight_fn,
        window_size=T_obs,
        hidden_dim=hidden_dim,
        num_hidden=num_hidden,
        model_type=model_type,
        num_epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=cfg.random_seed,
    )
    log(f"[STEP 4/6] Training done in {time.time() - t0:.1f}s")

    # 5. Setup sampler
    log("[STEP 5/6] Setting up sampler...")
    t0 = time.time()
    prior_norm = task.get_normalized_prior()

    if score_fn_type.lower() == "fnpe":
        score_fn = FNPEScoreFn(score_net, params, sde, prior_norm)
    else:
        score_fn = UncorrectedScoreFn(score_net, params, sde, prior_norm)

    kernel = EulerMaruyama(score_fn)
    time_grid = jnp.linspace(sde.T_min, sde.T_max, num_diffusion_steps)
    sampler = Diffuser(kernel, time_grid, task.input_shape)
    log(f"[STEP 5/6] Sampler setup done in {time.time() - t0:.1f}s")

    # 6. Diagnostic: sample from a test case
    log(
        f"[STEP 6/6] Running diagnostics ({num_posterior_samples} posterior samples)..."
    )
    t0 = time.time()
    key, key_test = jax.random.split(key)
    theta_true_phys = prior.sample(key_test)
    log(f"           True theta: {theta_true_phys}")

    key, key_sim = jax.random.split(key)
    simulator = task.get_simulator()
    log("           Simulating observed trajectory...")
    x_o_raw = simulator(key_sim, theta_true_phys, T_obs)
    x_o = task.normalize_x(x_o_raw)
    log(f"           Observed trajectory shape: {x_o.shape}")

    log(
        f"           Sampling {num_posterior_samples} posterior samples (JIT compiling)..."
    )
    key, key_sample = jax.random.split(key)

    # Do sampling with progress tracking
    sample_start = time.time()
    samples_norm = jax.vmap(sampler.sample, in_axes=(0, None))(
        jax.random.split(key_sample, num_posterior_samples), x_o
    )
    # Block until done
    samples_norm = jax.block_until_ready(samples_norm)
    log(f"           Sampling done in {time.time() - sample_start:.1f}s")

    samples_phys = jax.vmap(task.unnormalize_theta)(samples_norm)
    log(f"[STEP 6/6] Diagnostics done in {time.time() - t0:.1f}s")

    # Compute simple metrics
    samples_np = np.array(samples_phys)
    theta_true_np = np.array(theta_true_phys)

    posterior_mean = samples_np.mean(axis=0)
    posterior_std = samples_np.std(axis=0)
    abs_error = np.abs(posterior_mean - theta_true_np)

    # 7. Save everything
    # Parameters
    with open(exp_dir / "params.pkl", "wb") as f:
        pickle.dump(params, f)

    # Normalization stats
    norm_stats = task.get_normalization_stats()
    norm_stats_json = {k: v.tolist() for k, v in norm_stats.items()}
    with open(exp_dir / "normalization_stats.json", "w") as f:
        json.dump(norm_stats_json, f, indent=2)

    # Config
    config_dict = cfg.to_dict()
    config_dict.update(
        {
            "fnpe_num_simulations": num_simulations,
            "fnpe_T_obs": T_obs,
            "fnpe_hidden_dim": hidden_dim,
            "fnpe_num_hidden": num_hidden,
            "fnpe_model_type": model_type,
            "fnpe_num_epochs": num_epochs,
            "fnpe_steps_per_epoch": steps_per_epoch,
            "fnpe_batch_size": batch_size,
            "fnpe_learning_rate": learning_rate,
            "fnpe_num_diffusion_steps": num_diffusion_steps,
            "fnpe_score_fn_type": score_fn_type,
        }
    )
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # Metrics
    metrics = {
        "train_losses": losses,
        "final_loss": losses[-1] if losses else None,
        "diagnostic_theta_true": theta_true_np.tolist(),
        "diagnostic_posterior_mean": posterior_mean.tolist(),
        "diagnostic_posterior_std": posterior_std.tolist(),
        "diagnostic_abs_error": abs_error.tolist(),
    }
    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Generate plots
    log("[PLOTS] Generating plots...")
    try:
        import matplotlib.pyplot as plt

        # 1. Training loss curve
        plt.figure(figsize=(8, 4))
        plt.plot(losses, "b-", linewidth=1.5)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("FNPE Training Loss (DSM)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_dir / "training_loss.png", dpi=150)
        plt.close()
        log(f"  Saved: {fig_dir / 'training_loss.png'}")

        # 2. Corner plot of posterior samples
        try:
            import corner

            param_names = list(cfg.active_parameters)
            fig = corner.corner(
                samples_np,
                labels=param_names,
                truths=theta_true_np,
                quantiles=[0.16, 0.5, 0.84],
                show_titles=True,
                title_kwargs={"fontsize": 10},
            )
            fig.savefig(fig_dir / "corner_plot.png", dpi=150)
            plt.close(fig)
            log(f"  Saved: {fig_dir / 'corner_plot.png'}")
        except ImportError:
            log("  [WARN] corner package not installed, skipping corner plot")

        # 3. 1D marginal posteriors
        d = len(cfg.active_parameters)
        fig, axes = plt.subplots(1, d, figsize=(4 * d, 3))
        if d == 1:
            axes = [axes]
        for i, (ax, name) in enumerate(zip(axes, cfg.active_parameters)):
            ax.hist(
                samples_np[:, i], bins=50, density=True, alpha=0.7, color="steelblue"
            )
            ax.axvline(
                theta_true_np[i], color="red", linestyle="--", linewidth=2, label="True"
            )
            ax.axvline(
                posterior_mean[i],
                color="orange",
                linestyle="-",
                linewidth=2,
                label="Mean",
            )
            ax.set_xlabel(name)
            ax.set_ylabel("Density")
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.suptitle("FNPE Posterior Marginals")
        plt.tight_layout()
        plt.savefig(fig_dir / "marginals.png", dpi=150)
        plt.close()
        log(f"  Saved: {fig_dir / 'marginals.png'}")

        # 4. Posterior predictive check (PPC) - simulate from posterior samples
        log("[PLOTS] Running posterior predictive check...")
        n_ppc = min(100, num_posterior_samples)
        ppc_thetas = samples_phys[:n_ppc]

        # Get initial states for PPC
        key, key_ppc = jax.random.split(key)
        rng_ppc = np.random.default_rng(int(key_ppc[0]) % (2**31))
        from simulation.utils import sample_initial_states

        x0_ppc = sample_initial_states(rng_ppc, n_ppc)

        # Run simulations
        ppc_trajs = []
        for i in range(n_ppc):
            key, key_sim_i = jax.random.split(key)
            y_i = simulator(key_sim_i, ppc_thetas[i], T_obs)
            ppc_trajs.append(y_i)
        ppc_trajs = jnp.stack(ppc_trajs)  # (n_ppc, T, obs_dim)

        # Plot PPC - compare to observed trajectory
        obs_labels = ["ax", "ay", "dpsi", "vx", "vy", "beta", "psi", "X", "Y"]
        n_obs = min(9, ppc_trajs.shape[-1])

        fig, axes = plt.subplots(3, 3, figsize=(12, 10))
        axes = axes.flatten()
        t_axis = np.arange(T_obs) * cfg.dt

        for i in range(n_obs):
            ax = axes[i]
            # Plot PPC samples (thin lines)
            for j in range(min(30, n_ppc)):
                ax.plot(
                    t_axis,
                    np.array(ppc_trajs[j, :, i]),
                    alpha=0.2,
                    color="steelblue",
                    linewidth=0.5,
                )
            # Plot observed trajectory
            ax.plot(
                t_axis,
                np.array(x_o_raw[:, i]),
                color="red",
                linewidth=1.5,
                label="Observed",
            )
            # Plot PPC median
            ppc_median = np.median(np.array(ppc_trajs[:, :, i]), axis=0)
            ax.plot(
                t_axis,
                ppc_median,
                color="orange",
                linewidth=1.5,
                linestyle="--",
                label="PPC median",
            )
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(obs_labels[i] if i < len(obs_labels) else f"obs_{i}")
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(loc="upper right", fontsize=8)

        plt.suptitle("Posterior Predictive Check")
        plt.tight_layout()
        plt.savefig(fig_dir / "ppc_trajectories.png", dpi=150)
        plt.close()
        log(f"  Saved: {fig_dir / 'ppc_trajectories.png'}")

        # 5. Error summary bar plot
        fig, ax = plt.subplots(figsize=(6, 4))
        x_pos = np.arange(len(cfg.active_parameters))
        ax.bar(x_pos, abs_error, color="steelblue", alpha=0.7)
        ax.errorbar(
            x_pos,
            posterior_mean - theta_true_np,
            yerr=posterior_std,
            fmt="none",
            color="black",
            capsize=5,
        )
        ax.axhline(0, color="gray", linestyle="--")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(list(cfg.active_parameters))
        ax.set_ylabel("Absolute Error")
        ax.set_title("Posterior Mean Error (with ±1σ)")
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(fig_dir / "error_summary.png", dpi=150)
        plt.close()
        log(f"  Saved: {fig_dir / 'error_summary.png'}")

    except Exception as e:
        log(f"[ERROR] Error generating plots: {e}")
        import traceback

        traceback.print_exc()

    # Final summary
    total_experiment_time = time.time() - experiment_start
    log("=" * 60)
    log("EXPERIMENT COMPLETED")
    log("=" * 60)
    log(
        f"Total time: {total_experiment_time:.1f}s ({total_experiment_time/60:.1f} min)"
    )
    log(f"Results saved to: {exp_dir}")
    log("")
    log("=== Diagnostic Summary ===")
    for i, name in enumerate(cfg.active_parameters):
        log(
            f"  {name}: true={theta_true_np[i]:.4f}, "
            f"mean={posterior_mean[i]:.4f}, "
            f"std={posterior_std[i]:.4f}, "
            f"error={abs_error[i]:.4f}"
        )

    return {
        "exp_dir": exp_dir,
        "params": params,
        "score_net": score_net,
        "sampler": sampler,
        "task": task,
        "sde": sde,
        "metrics": metrics,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run FNPE experiment")
    parser.add_argument("--exp-name", type=str, default="fnpe_vehicle")
    parser.add_argument("--num-sim", type=int, default=50000)
    parser.add_argument("--T-obs", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps-per-epoch", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-hidden", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num-diff-steps", type=int, default=500)
    parser.add_argument("--num-post-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-type",
        type=str,
        default="gru",
        choices=["gru", "linear"],
        help="Score network observation processing type",
    )
    parser.add_argument(
        "--score-fn-type", type=str, default="fnpe", choices=["fnpe", "uncorrected"]
    )
    parser.add_argument(
        "--params",
        type=str,
        default="mu,cd,m",
        help="Comma-separated parameters to infer",
    )
    args = parser.parse_args()

    active_params = tuple(p.strip() for p in args.params.split(","))

    cfg = ExperimentConfig(
        exp_name=args.exp_name,
        random_seed=args.seed,
        active_parameters=active_params,
        T_seg=args.T_obs,
    )

    run_fnpe_experiment(
        cfg,
        num_simulations=args.num_sim,
        T_obs=args.T_obs,
        num_epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_hidden=args.num_hidden,
        model_type=args.model_type,
        learning_rate=args.lr,
        num_diffusion_steps=args.num_diff_steps,
        num_posterior_samples=args.num_post_samples,
        score_fn_type=args.score_fn_type,
    )
