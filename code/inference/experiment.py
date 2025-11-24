import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import pickle

import torch
import numpy as np
from sbi.diagnostics import run_sbc, check_sbc
from sbi.diagnostics.lc2st import LC2ST

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
    # LC2ST – using direct sampling
    # ================================
    NUM_LC2ST = cfg.num_lc2st_samples

    # 1) Calibration data from prior and simulator
    theta_cal = prior.sample((NUM_LC2ST,)).to(device)  # (N, d)
    x_cal = simulator(theta_cal)  # (N, T_event, D_in)
    x_cal_flat = x_cal.reshape(NUM_LC2ST, -1).cpu()  # (N, D)

    # 2) One posterior sample for each calibration x (shape: (N, d))
    #    We do this in a loop to avoid huge batched sampling.
    post_samples_list = []
    with torch.no_grad():
        for i in range(NUM_LC2ST):
            # x_cal: (N, T_event, D_in)
            xi = x_cal[i : i + 1].to(device)  # (1, T, D)
            # For NPE + direct posterior, this is very cheap:
            samp_i = posterior.sample((1,), x=xi)  # (1, d)
            post_samples_list.append(samp_i[0].cpu())  # (d,)

    post_samples_cal = torch.stack(post_samples_list, dim=0)  # (N, d)

    # 3) Construct LC2ST object.
    lc2st = LC2ST(
        thetas=theta_cal.cpu(),  # (N, d)
        xs=x_cal_flat,  # (N, D)
        posterior_samples=post_samples_cal,  # (N, d)
        classifier="mlp",
        num_ensemble=1,
    )

    print("Training LC2ST classifiers under H0 ...")
    _ = lc2st.train_under_null_hypothesis()
    print("Training LC2ST on observed data ...")
    _ = lc2st.train_on_observed_data()

    # 4) Compute a p-value at one representative calibration point.
    theta_o = post_samples_cal[0:1, :]  # (1, d)
    x_o = x_cal_flat[0]  # (D,)

    lc2st_pval = lc2st.p_value(theta_o, x_o)
    print("LC2ST p-value (first calibration point):", lc2st_pval)

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
        "lc2st_p_values": float(lc2st_pval),
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
