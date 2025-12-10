import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import pickle

import torch
import numpy as np
from sbi import utils as sbi_utils
from sbi.diagnostics import run_sbc, check_sbc
from sbi.diagnostics.lc2st import LC2ST_NF

from configs.config import ExperimentConfig
from utils.env_utils import setup_environment, get_device
from utils.metrics import sliced_wasserstein_prior_vs_dap, one_step_rmse_observation
from utils.normalization import (
    fit_normalizer,
    save_normalizer,
    Normalizer,
)
from utils.real_data import initial_state_from_obs, simulate_y_batch_for_thetas
from utils.plots import *
from simulation.simulation import (
    init_simulation_from_config,
    make_simulator,
    generate_dataset,
)
from models.models import build_prior, build_density_estimator


def make_experiment_dir(cfg: ExperimentConfig) -> Path:
    """
    Create a directory like experiments/<exp_name>_<timestamp>/.
    """
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path(cfg.results_root)
    root.mkdir(parents=True, exist_ok=True)
    exp_dir = root / f"{cfg.exp_name}_{ts}"
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


def run_parameter_posterior_plots(
    cfg,
    fig_dir: Path,
    prior,
    posterior,
    simulator,
    normalizer: Normalizer,
    device: torch.device,
    *,
    num_examples: int = 3,
    num_prior_samples: int = 20000,
    num_posterior_samples: int = 20000,
) -> None:
    """
    For a few fixed theta samples:
      - simulate x from the simulator,
      - condition the trained posterior on that x,
      - plot 1D prior vs posterior marginals with a vertical line at theta_true.

    Saves one PNG per (example, parameter) into exp_dir / 'figures' / 'theta_plots'.
    """

    # Determine parameter names (if you store them in cfg)
    if hasattr(cfg, "active_parameters") and cfg.active_parameters is not None:
        param_names = list(cfg.active_parameters)
    else:
        # fallback: generic names
        # infer dimensionality from a single prior sample
        theta_probe = prior.sample((1,)).to(device)
        d_theta = theta_probe.shape[-1]
        param_names = [f"theta[{i}]" for i in range(d_theta)]

    d_params = len(param_names)

    # Sample a big prior pool ONCE and reuse for all parameters
    with torch.no_grad():
        prior_pool = prior.sample((num_prior_samples,)).to(device)  # (N_prior, d)
    prior_pool_np = prior_pool.detach().cpu().numpy()  # (N_prior, d_params)

    for ex_idx in range(num_examples):
        print(
            f"[diagnostics] Parameter posterior plots: example {ex_idx+1}/{num_examples}"
        )

        # 1) Fix a theta from the prior
        with torch.no_grad():
            theta_true = prior.sample((1,)).to(device)  # (1, d_params)
        theta_true_np = theta_true.detach().cpu().numpy()[0]  # (d_params,)

        # 2) Simulate x from the simulator using this theta
        sim_out = simulator(theta_true)
        if isinstance(sim_out, tuple):
            x_sim, _ = sim_out  # (1, T, D_in)
        else:
            x_sim = sim_out

        # Normalize context before passing to posterior
        x_cond = normalizer.normalize_x(x_sim, cfg.obs_dim).to(device)

        # 3) Posterior samples conditioned on this x
        with torch.no_grad():
            theta_post_norm = posterior.sample((num_posterior_samples,), x=x_cond)
        theta_post_np = (
            normalizer.unnormalize_theta(theta_post_norm).detach().cpu().numpy()
        )

        # Handle possible (K, C, d) shape (multiple chains)
        if theta_post_np.ndim == 3:
            K, C, d = theta_post_np.shape
            theta_post_np = theta_post_np.reshape(K * C, d)
        elif theta_post_np.ndim != 2:
            raise ValueError(f"Unexpected posterior sample shape {theta_post_np.shape}")

        # 4) For each parameter dimension, make a 1D plot
        for j, pname in enumerate(param_names):
            prior_1d = prior_pool_np[:, j]
            post_1d = theta_post_np[:, j]
            theta_ref = theta_true_np[j]

            out_path = fig_dir / f"prior_posterior_{pname}_ex{ex_idx}.png"
            plot_prior_posterior_1d(
                prior_1d=prior_1d,
                post_1d=post_1d,
                theta_ref=theta_ref,
                param_name=pname,
                out_path=out_path,
                bins=80,
            )


def one_step_prediction_rmse_diagnostic(
    cfg: ExperimentConfig,
    prior,
    posterior,
    simulator,
    normalizer: Normalizer,
    device: torch.device,
    *,
    num_cases: int = 20,
    num_posterior_samples: int = 200,
):
    """
    1-step-ahead RMSE diagnostic in observation space, using synthetic data.

    For each synthetic case:
      - Sample theta_true ~ prior.
      - Simulate a trajectory x_sim = simulator(theta_true).
      - Condition posterior on the first time step x_0.
      - Sample theta from p(theta | x_0).
      - Using JAX vehicle model (via simulate_y_batch_for_thetas), generate
        a 2-step rollout and take y_pred at t=1 for each theta sample.
      - Compare mean predicted y_1 to true y_1 via RMSE.

    Returns:
        A dict with:
            "rmse_overall": float
            "rmse_per_dim": np.ndarray of shape (obs_dim,)
    """
    obs_dim = cfg.obs_dim
    state_dim = cfg.state_dim

    all_true_next = []
    all_pred_next = []

    for i in range(num_cases):
        print(f"[1-step RMSE] Case {i+1}/{num_cases}")

        # 1) Sample theta_true
        with torch.no_grad():
            theta_true = prior.sample((1,)).to(device)  # (1, d_theta)

        # 2) Simulate trajectory with full simulator (PyTorch side)
        with torch.no_grad():
            sim_out = simulator(theta_true)
        if isinstance(sim_out, tuple):
            x_sim, controls_sim = sim_out  # x_sim: (1, T, D_in)
        else:
            x_sim = sim_out
            raise RuntimeError(
                "Simulator must return (x, controls) for 1-step diagnostic."
            )

        x_np = x_sim[0].detach().cpu().numpy()  # (T, D_in)
        y_seq = x_np[:, :obs_dim]  # (T, obs_dim)

        if y_seq.shape[0] < 2:
            raise ValueError("Trajectory too short for 1-step RMSE (needs T >= 2).")

        # True next-step observation (t = 1)
        y_next_true = y_seq[1].astype(np.float32)  # (obs_dim,)

        # 3) Conditioning data: here we condition on the *first* time step only
        x_cond = normalizer.normalize_x(x_sim[:, :1, :], cfg.obs_dim).to(
            device
        )  # (1, 1, D_in)

        # 4) Sample theta from posterior conditioned on x_0
        with torch.no_grad():
            theta_post_norm = posterior.sample(
                (num_posterior_samples,),
                x=x_cond,
            )  # (K, d_theta) or (K, C, d_theta)

        theta_post_np = (
            normalizer.unnormalize_theta(theta_post_norm).detach().cpu().numpy()
        )
        if theta_post_np.ndim == 3:
            K, C, d = theta_post_np.shape
            theta_post_np = theta_post_np.reshape(K * C, d)
        elif theta_post_np.ndim != 2:
            raise ValueError(
                f"Unexpected posterior samples shape {theta_post_np.shape}"
            )
        K_eff = theta_post_np.shape[0]

        # 5) Build 2-step controls (t=0 and t=1) for JAX side
        #    controls_sim should be a dict of jnp arrays with shape (T,)
        controls_2 = {k: v[:2] for k, v in controls_sim.items()}

        # 6) Initial state from first observation
        state0 = initial_state_from_obs(y_seq[0])  # (state_dim,)

        # 7) Use JAX helper to simulate y for each posterior theta sample
        y_pred_batch = simulate_y_batch_for_thetas(
            theta_batch_np=theta_post_np.astype(np.float32),  # (K_eff, d_active)
            controls=controls_2,
            state_dim=state_dim,
            cfg=cfg,
            state0=state0,
        )  # (K_eff, T_local=2, obs_dim)

        # y_pred at t=1 for each posterior sample
        y_pred_next = y_pred_batch[:, 1, :]  # (K_eff, obs_dim)

        all_true_next.append(y_next_true)  # (obs_dim,)
        all_pred_next.append(y_pred_next)  # (K_eff, obs_dim)

    # Stack across cases
    y_true_next_np = np.stack(all_true_next, axis=0)  # (N, obs_dim)
    # For simplicity, truncate all_pred_next to same K across cases (use min K)
    K_common = min(arr.shape[0] for arr in all_pred_next)
    y_pred_next_np = np.stack([arr[:K_common] for arr in all_pred_next], axis=0)
    # shapes: y_true_next_np: (N, D), y_pred_next_np: (N, K_common, D)

    y_true_next_t = torch.from_numpy(y_true_next_np.astype(np.float32))
    y_pred_next_t = torch.from_numpy(y_pred_next_np.astype(np.float32))

    rmse_overall, rmse_per_dim = one_step_rmse_observation(y_true_next_t, y_pred_next_t)

    print("\n[1-step RMSE] overall:", rmse_overall)
    print("[1-step RMSE] per dim:", rmse_per_dim)

    return {
        "rmse_overall": rmse_overall,
        "rmse_per_dim": rmse_per_dim,
    }


def run_experiment(cfg: ExperimentConfig) -> None:
    """
    End-to-end run:
      - env + device
      - prior, simulator, dataset
      - NPE training
      - SBC + LC2ST + SWD
      - save model + metrics + config
    """
    # --- Setup env & device ---
    setup_environment(cfg.random_seed)
    device = get_device(cfg.device)

    # --- Simulation config ---
    init_simulation_from_config(cfg)

    # --- Prior (physical space) & simulator ---
    prior_phys = build_prior(cfg, device)
    simulator = make_simulator(cfg, device)

    # --- Wrapper so SBI gets only x, not (x, ctrls) ---
    def simulator_for_sbi(theta: torch.Tensor):
        x, _ = simulator(theta)  # discard controls
        return x

    # Probe once for input dim
    probe_theta = prior_phys.sample((1,))  # respects active parameter dimensionality
    probe_x = simulator_for_sbi(probe_theta)  # (1, T_event, D_in)
    _, T_event, D_in = probe_x.shape
    print(
        f"Encoder input_dim = {D_in} | T_event = {T_event} | "
        f"active parameters = {cfg.active_parameters}"
    )

    # --- Build density estimator & inference object ---
    # --- Prepare experiment directories ---
    exp_dir = make_experiment_dir(cfg)
    fig_dir = exp_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # --- Generate dataset (physical units) ---
    theta_train_phys, x_train_phys, _ = generate_dataset(
        cfg, prior_phys, simulator, show_pbar=True
    )

    # --- Fit & persist normalization stats ---
    normalizer_cpu = fit_normalizer(theta_train_phys, x_train_phys, cfg.obs_dim)
    norm_path = exp_dir / "stats_normalization.json"
    save_normalizer(normalizer_cpu, norm_path)
    print(f"[normalization] Saved stats to {norm_path}")

    # Clone stats to training device for convenience
    normalizer = normalizer_cpu.to(device)

    # --- Normalize training data (stay on CPU to avoid GPU blowup) ---
    theta_train = normalizer_cpu.normalize_theta(theta_train_phys)
    x_train = normalizer_cpu.normalize_x(x_train_phys, cfg.obs_dim)

    # --- Build normalized prior for training (affine transform of physical bounds) ---
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

    # --- Build density estimator & inference object (normalized space) ---
    encoder, density_estimator, inference = build_density_estimator(
        cfg,
        input_dim=D_in,
        prior=prior_norm,
        device=device,
    )

    # Use normalized data for training
    inference.append_simulations(theta_train, x_train)

    # --- Train NPE ---
    density_estimator = inference.train(
        learning_rate=cfg.learning_rate,
        training_batch_size=cfg.training_batch_size,
        validation_fraction=cfg.validation_fraction,
        stop_after_epochs=cfg.stop_after_epochs,
        clip_max_norm=cfg.clip_max_norm,
        show_train_summary=True,
        use_combined_loss=False,
    )

    density_estimator.to(device).eval()

    posterior = inference.build_posterior(density_estimator)

    # Save pickled posterior for faster loading later
    with open(exp_dir / "posterior.pkl", "wb") as f:
        pickle.dump(posterior, f)

    # Create prior vs posterior plots for a few synthetic thetas
    run_parameter_posterior_plots(
        cfg=cfg,
        fig_dir=fig_dir,
        prior=prior_phys,
        posterior=posterior,
        simulator=simulator,
        normalizer=normalizer,
        device=device,
        num_examples=3,
        num_prior_samples=20000,
        num_posterior_samples=20000,
    )

    # --- SBC ---
    num_sbc = cfg.num_sbc_samples
    num_post = cfg.num_posterior_samples_sbc

    theta_sbc_norm = prior_norm.sample((num_sbc,)).to(device)  # normalized theta
    theta_sbc_phys = normalizer.unnormalize_theta(theta_sbc_norm)
    x_sbc_phys = simulator_for_sbi(theta_sbc_phys)  # (num_sbc, T_event, D_in)
    x_sbc_norm = normalizer.normalize_x(x_sbc_phys, cfg.obs_dim)

    ranks, dap_samples_norm = run_sbc(
        thetas=theta_sbc_norm,
        xs=x_sbc_norm,
        posterior=posterior,
        num_posterior_samples=num_post,
        num_workers=8,
        use_sample_batched=False,
    )

    check_stats = check_sbc(ranks, theta_sbc_norm, dap_samples_norm, num_post)
    print("SBC check statistics:", check_stats)

    sbc_fig_path = fig_dir / "sbc_rank_hist.png"
    plot_sbc_rank_hist(ranks, num_post, sbc_fig_path)

    # --- SWD(prior, DAP) ---
    dap_tensor = normalizer.unnormalize_theta(
        dap_samples_norm.view(-1, dap_samples_norm.shape[-1])
    )
    swd_val = sliced_wasserstein_prior_vs_dap(
        prior_phys,
        dap_tensor,
        num_projections=cfg.num_swd_projections,
        seed=cfg.random_seed,
    )
    print(f"Sliced Wasserstein distance (prior vs DAP): {swd_val:.4f}")

    # ================================
    # LC2ST-NF (flow-space diagnostics)
    # ================================
    NUM_LC2ST = cfg.num_lc2st_samples
    CONF_ALPHA = 0.05

    # Build MCMC posterior for diagnostics
    # posterior_lc2st = inference.build_posterior(
    #     density_estimator,
    #     sample_with="mcmc",
    #     mcmc_method="slice_np_vectorized",
    #     mcmc_parameters={"num_chains": 1, "thin": 1, "warmup_steps": 20},
    # )

    # 1) Calibration data from prior and simulator
    theta_cal_norm = prior_norm.sample((NUM_LC2ST,)).to(device)  # (N, d_norm)
    theta_cal = normalizer.unnormalize_theta(theta_cal_norm)  # physical for sim
    x_cal_phys = simulator_for_sbi(theta_cal)  # (N, T_event, D_in)
    x_cal = normalizer.normalize_x(x_cal_phys, cfg.obs_dim)
    x_cal_flat = x_cal.reshape(NUM_LC2ST, -1).cpu()  # (N, D_flat)

    # 2) One posterior sample for each calibration x (shape: (N, d))
    #    Sample one context at a time to avoid batch-size mismatches in MCMC.
    # with torch.no_grad():
    #     post_samples_norm = (
    #         # posterior_lc2st.sample_batched(
    #             (1,),  # one sample per x
    #             x=x_cal,
    #             num_chains=1,
    #             init_strategy="proposal",
    #             show_progress_bars=True,
    #         )[0]
    #         .squeeze(0)
    #         .cpu()
    #     )with torch.no_grad():
    # Direct sampling from flow, no MCMC
    post_samples_norm = (
        posterior.sample_batched(
            sample_shape=(1,),
            x=x_cal,
            max_sampling_batch_size=64,
        )[0]
        .squeeze(0)
        .cpu()
    )

    # 3) Flow-space transform helpers
    assert hasattr(density_estimator, "net") and hasattr(
        density_estimator.net, "_transform"
    ), "Posterior does not expose flow transform needed for LC2ST-NF."
    flow_transform = density_estimator.net._transform
    flow_base_dist = torch.distributions.MultivariateNormal(
        torch.zeros(theta_cal_norm.shape[1], device=device),
        torch.eye(theta_cal_norm.shape[1], device=device),
    )
    flow_embed = density_estimator.net._embedding_net

    def flow_inverse_transform(theta, x_flattened):
        x_batch = x_flattened.to(device).view(-1, T_event, D_in)
        ctx = flow_embed(x_batch)
        z, _ = flow_transform(theta.to(device), context=ctx)
        return z

    # 4) Construct LC2ST-NF object.
    lc2st = LC2ST_NF(
        thetas=theta_cal_norm.cpu(),  # (N, d_norm)
        xs=x_cal_flat,  # (N, D_flat)
        posterior_samples=post_samples_norm,  # (N, d)
        flow_inverse_transform=flow_inverse_transform,
        flow_base_dist=flow_base_dist,
        classifier="mlp",
        num_ensemble=1,
    )

    print("Training LC2ST-NF classifiers under H0 ...")
    _ = lc2st.train_under_null_hypothesis()
    print("Training LC2ST-NF on observed data ...")
    _ = lc2st.train_on_observed_data()

    # 5) Compute diagnostics for one representative calibration point.
    x_o = x_cal_flat[0]  # (D_flat,)

    T_data = lc2st.get_statistic_on_observed_data(x_o=x_o)
    T_null = lc2st.get_statistics_under_null_hypothesis(x_o=x_o)
    lc2st_pval = lc2st.p_value(x_o)
    lc2st_reject = lc2st.reject_test(x_o, alpha=CONF_ALPHA)

    probs_data, _ = lc2st.get_scores(
        x_o=x_o, return_probs=True, trained_clfs=lc2st.trained_clfs
    )
    probs_null, _ = lc2st.get_statistics_under_null_hypothesis(
        x_o=x_o, return_probs=True
    )

    print(
        f"LC2ST-NF p-value (first calibration point): {lc2st_pval} | "
        f"reject (alpha={CONF_ALPHA}): {lc2st_reject}"
    )

    lc2st_hist_path = fig_dir / "lc2st_nf_statistic.png"
    plot_lc2st_histogram(
        T_null,
        float(T_data),
        CONF_ALPHA,
        lc2st_hist_path,
        title="LC2ST-NF statistic (first calibration point)",
        p_value=float(lc2st_pval),
    )

    lc2st_pp_path = fig_dir / "lc2st_nf_pp_plot.png"
    plot_lc2st_pp_plot(
        probs_data,
        probs_null,
        CONF_ALPHA,
        lc2st_pp_path,
        title="LC2ST-NF PP-plot (first calibration point)",
    )
    # ==========================================
    # 1-step-ahead RMSE diagnostic (synthetic)
    # ==========================================
    try:
        one_step_results = one_step_prediction_rmse_diagnostic(
            cfg=cfg,
            prior=prior_phys,
            posterior=posterior,
            simulator=simulator,
            normalizer=normalizer,
            device=device,
            num_cases=20,
            num_posterior_samples=200,
        )

        one_step_path = exp_dir / "one_step_rmse.json"
        with one_step_path.open("w") as f:
            json.dump(
                {
                    "rmse_overall": one_step_results["rmse_overall"],
                    "rmse_per_dim": (
                        one_step_results["rmse_per_dim"].tolist()
                        if hasattr(one_step_results["rmse_per_dim"], "tolist")
                        else list(one_step_results["rmse_per_dim"])
                    ),
                },
                f,
                indent=2,
            )
    except Exception as e:
        print(f"[1-step RMSE] Diagnostic failed with error: {e}")

    # --- Save everything ---
    # 1) Save config
    with (exp_dir / "config.json").open("w") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    # 2) Save training summary & metrics
    summary = inference.summary

    train_curve_path = fig_dir / "training_loss.png"
    plot_training_curves(summary, train_curve_path)

    metrics: Dict[str, Any] = {
        "train_loss": summary.get("training_loss", []),
        "val_loss": summary.get("validation_loss", []),
        "sbc_check_stats": {k: tensor_to_python(v) for k, v in check_stats.items()},
        "swd_prior_vs_dap": float(swd_val),
        "lc2st_nf_p_value": float(lc2st_pval),
        "lc2st_nf_reject_alpha_0.05": bool(lc2st_reject),
        "lc2st_nf_T_data": float(T_data),
        "normalization_stats_path": str(norm_path),
    }
    with (exp_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    # 3) Save model weights
    torch.save(density_estimator.state_dict(), exp_dir / "density_estimator.pt")

    # ---------------------------------
    # After training: run real-data eval
    # ---------------------------------
    REAL_DATA_CSV = cfg.real_data_csv
    if REAL_DATA_CSV is not None:
        print("\n=== Running real-data evaluation ===")

        import subprocess, sys

        eval_script = Path(__file__).resolve().parents[2] / "code" / "eval_real_data.py"

        cmd = [
            sys.executable,
            str(eval_script),
            "--exp-dir",
            str(exp_dir),
            "--csv",
            REAL_DATA_CSV,
            "--device",
            cfg.device,
        ]

        print("Executing:", " ".join(cmd))
        subprocess.run(cmd, check=True)

        print("Real-data evaluation completed.")

    print(f"Experiment completed. Results saved in: {exp_dir}")
