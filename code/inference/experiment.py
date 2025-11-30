import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import pickle

import torch
import numpy as np
from sbi.diagnostics import run_sbc, check_sbc
from sbi.diagnostics.lc2st import LC2ST_NF

from configs.config import ExperimentConfig
from utils.env_utils import setup_environment, get_device
from utils.metrics import sliced_wasserstein_prior_vs_dap
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

    # --- Prior & simulator ---
    prior = build_prior(cfg, device)
    simulator = make_simulator(cfg, device)

    # Probe once for input dim
    probe_theta = prior.sample((1,))  # respects active parameter dimensionality
    probe_x = simulator(probe_theta)  # (1, T_event, D_in)
    _, T_event, D_in = probe_x.shape
    print(
        f"Encoder input_dim = {D_in} | T_event = {T_event} | "
        f"active parameters = {cfg.active_parameters}"
    )

    # --- Build density estimator & inference object ---
    encoder, density_estimator, inference = build_density_estimator(
        cfg,
        input_dim=D_in,
        prior=prior,
        device=device,
    )

    # --- Prepare experiment directories ---
    exp_dir = make_experiment_dir(cfg)
    fig_dir = exp_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # --- Generate dataset ---
    theta_train, x_train = generate_dataset(cfg, prior, simulator, show_pbar=True)
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

    posterior = inference.build_posterior(
        density_estimator,
        sample_with="direct",
    )

    # Save pickled posterior for faster loading later
    with open(exp_dir / "posterior.pkl", "wb") as f:
        pickle.dump(posterior, f)

    # --- SBC ---
    num_sbc = cfg.num_sbc_samples
    num_post = cfg.num_posterior_samples_sbc

    theta_sbc = prior.sample((num_sbc,))  # (num_sbc, d)
    x_sbc = simulator(theta_sbc)  # (num_sbc, T_event, D_in)

    ranks, dap_samples = run_sbc(
        thetas=theta_sbc,
        xs=x_sbc,
        posterior=posterior,
        num_posterior_samples=num_post,
        num_workers=8,
        use_sample_batched=True,
    )

    check_stats = check_sbc(ranks, theta_sbc, dap_samples, num_post)
    print("SBC check statistics:", check_stats)

    sbc_fig_path = fig_dir / "sbc_rank_hist.png"
    plot_sbc_rank_hist(ranks, num_post, sbc_fig_path)

    # --- SWD(prior, DAP) ---
    dap_tensor = dap_samples.view(-1, dap_samples.shape[-1])
    swd_val = sliced_wasserstein_prior_vs_dap(
        prior, dap_tensor, num_projections=cfg.num_swd_projections, seed=cfg.random_seed
    )
    print(f"Sliced Wasserstein distance (prior vs DAP): {swd_val:.4f}")

    # ================================
    # LC2ST-NF (flow-space diagnostics)
    # ================================
    NUM_LC2ST = cfg.num_lc2st_samples
    CONF_ALPHA = 0.05

    # 1) Calibration data from prior and simulator
    theta_cal = prior.sample((NUM_LC2ST,)).to(device)  # (N, d)
    x_cal = simulator(theta_cal)  # (N, T_event, D_in)
    x_cal_flat = x_cal.reshape(NUM_LC2ST, -1).cpu()  # (N, D_flat)

    # 2) One posterior sample for each calibration x (shape: (N, d))
    with torch.no_grad():
        post_samples = posterior.sample_batched(
            (1,), x=x_cal, max_sampling_batch_size=10
        )[0].cpu()

    # 3) Flow-space transform helpers
    assert hasattr(density_estimator, "net") and hasattr(
        density_estimator.net, "_transform"
    ), "Posterior does not expose flow transform needed for LC2ST-NF."
    flow_transform = density_estimator.net._transform
    flow_base_dist = torch.distributions.MultivariateNormal(
        torch.zeros(theta_cal.shape[1], device=device),
        torch.eye(theta_cal.shape[1], device=device),
    )
    flow_embed = density_estimator.net._embedding_net

    def flow_inverse_transform(theta, x_flattened):
        x_batch = x_flattened.to(device).view(-1, T_event, D_in)
        ctx = flow_embed(x_batch)
        z, _ = flow_transform(theta.to(device), context=ctx)
        return z

    # 4) Construct LC2ST-NF object.
    lc2st = LC2ST_NF(
        thetas=theta_cal.cpu(),  # (N, d)
        xs=x_cal_flat,  # (N, D_flat)
        posterior_samples=post_samples,  # (N, d)
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
