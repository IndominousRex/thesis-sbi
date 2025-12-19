import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import torch
import pandas as pd

from sbi import utils as sbi_utils
from sbi.inference import NPSE
from configs.config import ExperimentConfig
from utils.env_utils import setup_environment, get_device
from simulation.simulation import init_simulation_from_config, make_simulator
from models.models import build_prior, build_density_estimator
from utils.real_data import (
    build_real_window_from_csv,
    build_simulated_window_for_eval,
    posterior_predictive_from_real,
    OBS_LABELS,
    prep_x_obs_from_df,
)
from utils.normalization import load_normalizer
from utils.metrics import real_data_trajectory_metrics
from utils.plots import plot_ppc_trajectories, plot_obs_1d_hist_custom


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a trained NPE experiment on real driving data."
    )
    p.add_argument(
        "--exp-dir",
        type=str,
        required=True,
        help="Path to experiment folder (contains config.json and density_estimator.pt).",
    )
    p.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to real measurement CSV (e.g. Jeversen_2022_10_12_110132.csv).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device preference for evaluation.",
    )
    p.add_argument(
        "--start-idx",
        type=int,
        default=None,
        help=(
            "Optional explicit start index in the CSV. If omitted, "
            "a low-brake window is chosen automatically."
        ),
    )
    p.add_argument(
        "--K-ppc",
        type=int,
        default=300,
        help="Number of posterior predictive trajectories.",
    )
    return p.parse_args()


def load_config(exp_dir: Path) -> ExperimentConfig:
    cfg_path = exp_dir / "config.json"
    with cfg_path.open("r") as f:
        cfg_dict = json.load(f)
    # Config may not have 'device' if you used earlier scripts; set default
    cfg_dict.setdefault("device", "auto")
    # Backward compatibility: drop decimation if present
    cfg_dict.pop("decimate", None)
    return ExperimentConfig(**cfg_dict)


def main():
    args = parse_args()

    exp_dir = Path(args.exp_dir).resolve()
    assert exp_dir.exists(), f"Experiment directory not found: {exp_dir}"

    # --- 1) Load config and set up environment/device ---
    cfg = load_config(exp_dir)
    setup_environment(cfg.random_seed)
    # override device from CLI if given
    cfg.device = args.device if args.device is not None else cfg.device
    device = get_device(cfg.device)

    # --- 2) Init simulation / priors / simulator ---
    init_simulation_from_config(cfg)
    prior_phys = build_prior(cfg, device)
    simulator = make_simulator(cfg, device)

    norm_path = exp_dir / "stats_normalization.json"
    assert norm_path.exists(), f"Missing normalization stats at {norm_path}"
    normalizer = load_normalizer(norm_path, device=device)

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

    # Probe to recover T_event and D_in (not strictly needed, but nice sanity check)
    probe_theta = prior_phys.sample((1,)).to(device)
    probe_out = simulator(probe_theta)
    if isinstance(probe_out, tuple):
        probe_x, _ = probe_out
    else:
        probe_x = probe_out
    _, T_event, D_in = probe_x.shape
    print(f"[eval] Model expects T_event={T_event}, D_in={D_in}")

    # --- 3) Detect method type and rebuild inference + load trained weights ---
    # Check if this is an NPSE experiment
    cfg_path = exp_dir / "config.json"
    with cfg_path.open("r") as f:
        cfg_raw = json.load(f)
    method = cfg_raw.get("method", "NPE")  # Default to NPE for backward compatibility
    sde_type = cfg_raw.get("sde_type", "ve")  # Default SDE type for NPSE

    print(f"[eval] Detected method: {method}")

    posterior = None
    posterior_path = exp_dir / "posterior.pkl"

    if posterior_path.exists():
        try:
            with posterior_path.open("rb") as f:
                posterior = pickle.load(f)

            if hasattr(posterior, "to"):
                posterior.to(device)

            print(f"[eval] Loaded pickled posterior from {posterior_path}")

        except Exception as exc:
            print(f"[eval] Failed to load pickled posterior ({exc}); rebuilding.")
            posterior = None

    # Fallback: rebuild inference, load state_dict, then build posterior
    if posterior is None:
        if method == "NPSE":
            # NPSE experiment: load score_estimator.pt
            state_dict_path = exp_dir / "score_estimator.pt"
            assert state_dict_path.exists(), f"Missing {state_dict_path}"

            # Build NPSE inference object
            inference = NPSE(
                prior=prior_norm,
                sde_type=sde_type,
                device=str(device),
            )

            # Load trained score estimator
            score_estimator = inference.append_simulations(
                torch.zeros(1, len(cfg.active_parameters)),  # dummy
                torch.zeros(1, T_event, D_in),  # dummy
            )._neural_net
            score_estimator.load_state_dict(
                torch.load(state_dict_path, map_location=device)
            )
            score_estimator.to(device).eval()

            posterior = inference.build_posterior(score_estimator)
            print(
                f"[eval] Built NPSE posterior from score_estimator.pt (sde_type={sde_type})"
            )
        else:
            # NPE experiment: load density_estimator.pt
            _, density_estimator, inference = build_density_estimator(
                cfg, input_dim=D_in, prior=prior_norm, device=device
            )

            state_dict_path = exp_dir / "density_estimator.pt"
            assert state_dict_path.exists(), f"Missing {state_dict_path}"

            # Let sbi create a fresh neural posterior with the right architecture
            density_estimator_net = inference._neural_net
            density_estimator_net.load_state_dict(
                torch.load(state_dict_path, map_location=device)
            )
            density_estimator_net.to(device).eval()

            posterior = inference.build_posterior(density_estimator_net)
            print("[eval] Built NPE posterior from density_estimator.pt")

    # --- 4) Load real CSV and build window matching training format ---
    df_real = pd.read_csv(args.csv)
    x_obs_full, controls_real, start_idx = build_real_window_from_csv(
        df_real,
        cfg,
        device,
        start_idx=args.start_idx,
        prefer_low_brake=True,
        brake_thresh=5.0,
        max_viol_frac=0.01,
        rate_body_z_in_deg_s=True,
        tire_rates_in_rpm=False,
        vel_body_in_kmh=False,
    )

    # Check that time length matches T_event
    assert (
        x_obs_full.shape[1] == T_event
    ), f"Window length {x_obs_full.shape[1]} does not match model T_event={T_event}"

    # --- 5) Posterior predictive checks on real segment ---
    y_real, y_ppc = posterior_predictive_from_real(
        posterior,
        x_obs_full,
        controls_real,
        cfg,
        normalizer=normalizer,
        device=device,
        K_ppc=args.K_ppc,
    )

    print("PPC shapes: y_real =", y_real.shape, "| y_ppc =", y_ppc.shape)

    fig_dir = exp_dir / "figures"

    # 1) PPC time-series plot on real segment
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

    # 1b) PPC on a fresh simulated window (held-out controls/theta)
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
        K_ppc=args.K_ppc,
    )
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

    # 2) Build "train-like" observations from fresh simulations (for hist diagnostics)
    N_hist = min(2000, getattr(cfg, "num_simulations", 2000))
    with torch.no_grad():
        theta_hist = prior_phys.sample((N_hist,)).to(device)
        sim_out = simulator(theta_hist)
        if isinstance(sim_out, tuple):
            x_hist, _ = sim_out
        else:
            x_hist = sim_out

    # First obs_dim dims are observations
    train_obs = (
        x_hist[:, :, : cfg.obs_dim].detach().cpu().numpy().reshape(-1, cfg.obs_dim)
    )

    # Real obs over entire CSV
    x_obs_all = prep_x_obs_from_df(
        df_real,
        start_idx=0,
        T=len(df_real),
        rate_body_z_in_deg_s=True,
        tire_rates_in_rpm=False,
        vel_body_in_kmh=False,
    )
    real_obs_all = x_obs_all.numpy()

    # Custom zoom ranges
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

    # --- 6) Compute real-data metrics ---
    metrics = real_data_trajectory_metrics(y_real, y_ppc, normalize_w2=True)

    print("\n=== Real-data trajectory metrics ===")
    print(f"Overall RMSE (mixed units): {metrics['rmse_overall']:.4f}")
    print("Per-dimension RMSEs:")
    for label, val in zip(OBS_LABELS, metrics["rmse_per_dim"]):
        print(f"  {label:<20}: {val:.4f}")
    print(f"W2 (normalized): {metrics['w2']:.4f}")
    print(f"Mean per-sample W2: {metrics['w2_mean_per_sample']:.4f}")

    # --- 7) Save metrics to experiment folder ---
    real_metrics_path = exp_dir / "real_metrics.json"
    with real_metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved real-data metrics to {real_metrics_path}")


if __name__ == "__main__":
    main()
