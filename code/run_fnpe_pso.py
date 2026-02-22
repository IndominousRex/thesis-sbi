#!/usr/bin/env python
"""
Run an FNPE experiment using PSO-optimized vehicle model parameters.

This script:
  1. Loads the PSO optimization results JSON.
  2. Monkey-patches VehicleModel.default_params and radius_tire with the
     optimized global parameters *before* any JIT compilation happens.
  3. Builds an ExperimentConfig for FNPE with prior ranges centred on the
     PSO-optimized mu / air_resistance / mass values.
  4. Delegates to the standard run_experiment() pipeline.

Usage (local):
    python run_fnpe_pso.py --pso-json notebooks/experiments/pso_optimization_results.json

Usage (SLURM):
    srun python run_fnpe_pso.py \\
        --pso-json notebooks/experiments/pso_optimization_results.json \\
        --exp-name fnpe_pso_optimized \\
        --device cuda \\
        --fnpe-num-sim 100000 \\
        --T-seg 3000
"""

import argparse
import json
import sys
from pathlib import Path


def load_pso_results(json_path: str) -> dict:
    """Load PSO optimization results and return the dict."""
    path = Path(json_path)
    if not path.exists():
        sys.exit(f"ERROR: PSO results not found at {path}")
    with open(path) as f:
        results = json.load(f)
    print(f"[PSO] Loaded results from {path}")
    print(f"[PSO] Method: {results.get('method', '?')}")
    print(f"[PSO] Final error: {results.get('final_error', '?')}")
    return results


def patch_vehicle_model(pso_results: dict):
    """
    Monkey-patch VehicleModel.default_params and radius_tire with
    PSO-optimized global parameters.

    This MUST be called before any JAX JIT compilation of the vehicle model
    (i.e., before importing simulation.simulation or markovsbi).
    """
    import simulation.VehicleModel as VM

    global_params = pso_results["global_params"]

    # Map PSO global param names -> default_params keys
    # Supports both old format (c_1x etc. per-trajectory) and new global format
    param_mapping = {
        "mass": "mass",
        "Inertia_z": "Inertia_z",
        "Inertia_tire": "Inertia_tire",
        "Inertia_engine": "Inertia_engine",
        "air_resistance": "air_resistance",
        "c_1y": "c_1y",
        "c_2y": "c_2y",
        "C_y": "C_y",
        "E_y": "E_y",
        "C_roll1": "C_roll1",
        "C_roll2": "C_roll2",
        # These are global in the new PSO format (run_pso_global.py)
        "c_1x": "c_1x",
        "c_2x": "c_2x",
        "C_x": "C_x",
        "E_x": "E_x",
    }

    print("\n[PSO] Patching VehicleModel.default_params (global params):")
    for pso_key, vm_key in param_mapping.items():
        if pso_key in global_params:
            old_val = VM.default_params.get(vm_key, "N/A")
            new_val = float(global_params[pso_key])
            VM.default_params[vm_key] = new_val
            print(f"  {vm_key:20s}: {old_val} -> {new_val}")

    # Patch radius_tire (module-level constant)
    if "radius_tire" in global_params:
        old_rt = VM.radius_tire
        VM.radius_tire = float(global_params["radius_tire"])
        print(f"  {'radius_tire':20s}: {old_rt} -> {VM.radius_tire}")

    # --- Per-trajectory params: average across trajectories ---
    # (c_1x, c_2x, C_x, E_x varied per trajectory in PSO; use mean as default)
    traj_params = pso_results.get("trajectory_params", [])
    per_traj_keys = ["c_1x", "c_2x", "C_x", "E_x"]
    if traj_params:
        print("\n[PSO] Patching per-trajectory params (averaged across trajectories):")
        for key in per_traj_keys:
            vals = [
                tp["params"][key] for tp in traj_params if key in tp.get("params", {})
            ]
            if vals:
                avg_val = sum(vals) / len(vals)
                old_val = VM.default_params.get(key, "N/A")
                VM.default_params[key] = avg_val
                print(
                    f"  {key:20s}: {old_val} -> {avg_val:.6f}  (mean of {len(vals)} trajs)"
                )

    # Also patch the imported copies in simulation.simulation and markovsbi.
    # default_params is a mutable dict (in-place mutation propagates), but
    # radius_tire is a float (immutable), so we must patch each module's binding.
    for mod_name in ["simulation.simulation", "markovsbi.tasks.vehicle_dynamics"]:
        try:
            mod = __import__(mod_name, fromlist=["default_params", "radius_tire"])
            mod.radius_tire = VM.radius_tire
            print(f"[PSO] Patched radius_tire in {mod_name}")
        except Exception:
            pass

    print()


def compute_prior_bounds(pso_results: dict) -> dict:
    """
    Compute reasonable prior bounds for mu, cd (air_resistance), and m (mass)
    based on PSO-optimized values.

    For mu: uses the range of per-trajectory values with some padding.
    For cd and m: centres on the PSO value with ±30% margin.
    """
    global_params = pso_results["global_params"]
    traj_params = pso_results.get("trajectory_params", [])

    # --- mu: use range of per-trajectory friction coefficients ---
    mu_values = [
        tp["params"]["mu"] for tp in traj_params if "mu" in tp.get("params", {})
    ]
    if mu_values:
        mu_center = sum(mu_values) / len(mu_values)
        mu_spread = max(mu_values) - min(mu_values)
        # Pad by 50% of spread, minimum ±0.15
        mu_pad = max(0.15, mu_spread * 0.5)
        mu_low = max(0.1, mu_center - mu_pad)
        mu_high = min(2.0, mu_center + mu_pad)
    else:
        mu_low, mu_high = 0.5, 1.5

    # --- air_resistance (cd) ---
    cd_val = float(global_params.get("air_resistance", 0.27))
    cd_low = max(0.05, cd_val * 0.7)
    cd_high = cd_val * 1.3

    # --- mass (m) ---
    m_val = float(global_params.get("mass", 1720))
    m_low = m_val * 0.9
    m_high = m_val * 1.1

    bounds = {
        "mu": (round(mu_low, 3), round(mu_high, 3)),
        "cd": (round(cd_low, 4), round(cd_high, 4)),
        "m": (round(m_low, 1), round(m_high, 1)),
    }
    print("[PSO] Derived prior bounds:")
    for k, (lo, hi) in bounds.items():
        print(f"  {k}: [{lo}, {hi}]")
    print()
    return bounds


def parse_args():
    p = argparse.ArgumentParser(
        description="Run FNPE experiment with PSO-optimized vehicle parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--pso-json",
        type=str,
        required=True,
        help="Path to pso_optimization_results.json",
    )
    p.add_argument("--exp-name", type=str, default="fnpe_pso_optimized")
    p.add_argument(
        "--device", type=str, default="cuda", choices=["auto", "cpu", "cuda"]
    )
    p.add_argument("--seed", type=int, default=42)

    # --- Data ---
    p.add_argument(
        "--num-sim",
        type=int,
        default=None,
        help="Num sims (uses fnpe-num-sim for FNPE)",
    )
    p.add_argument("--T-seg", type=int, default=3000)
    p.add_argument("--params", type=str, default="mu,cd,m", help="Parameters to infer")

    # --- Prior override (optional, otherwise auto-computed from PSO) ---
    p.add_argument("--prior-mu-low", type=float, default=None)
    p.add_argument("--prior-mu-high", type=float, default=None)
    p.add_argument("--prior-cd-low", type=float, default=None)
    p.add_argument("--prior-cd-high", type=float, default=None)
    p.add_argument("--prior-m-low", type=float, default=None)
    p.add_argument("--prior-m-high", type=float, default=None)

    # --- FNPE-specific ---
    p.add_argument("--fnpe-num-sim", type=int, default=100000)
    p.add_argument("--fnpe-hidden-dim", type=int, default=128)
    p.add_argument("--fnpe-num-hidden", type=int, default=5)
    p.add_argument(
        "--fnpe-model-type", type=str, default="gru", choices=["gru", "linear"]
    )
    p.add_argument("--fnpe-window-size", type=int, default=2)
    p.add_argument(
        "--fnpe-t-min",
        type=float,
        default=0.05,
        help="SDE T_min (default 0.05, was 0.01)",
    )
    p.add_argument("--fnpe-steps-per-epoch", type=int, default=10000)
    p.add_argument("--fnpe-diffusion-steps", type=int, default=500)
    p.add_argument("--fnpe-score-fn", type=str, default="gauss_corrected")
    p.add_argument("--fnpe-proposal-type", type=str, default="pred")
    p.add_argument("--fnpe-pilot-fraction", type=float, default=0.02)
    p.add_argument("--fnpe-pilot-length", type=int, default=1500)
    p.add_argument("--fnpe-proposal-noise", type=float, default=0.03)
    p.add_argument(
        "--fnpe-skip-normalize",
        action="store_true",
        help="Disable internal FNPE normalization (debug mode)",
    )
    p.add_argument(
        "--fnpe-clip-samples",
        action="store_true",
        default=True,
        help="Clip diffusion samples to prior bounds (default: True)",
    )
    p.add_argument(
        "--fnpe-no-clip-samples",
        action="store_true",
        help="Disable diffusion sample clipping",
    )
    p.add_argument(
        "--fnpe-gauss-precision-scale",
        type=float,
        default=None,
        help="Fixed precision scale for GaussCorrectedScoreFn (None = auto-estimate)",
    )

    # --- Training ---
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-epochs", type=int, default=200)
    p.add_argument("--stop-after-epochs", type=int, default=30)

    # --- Encoder ---
    p.add_argument("--encoder-type", type=str, default="bigru")
    p.add_argument("--encoder-hidden", type=int, default=32)

    # --- Diagnostics ---
    p.add_argument("--no-sbc", action="store_true")
    p.add_argument("--no-swd", action="store_true")
    p.add_argument("--no-one-step", action="store_true")
    p.add_argument("--no-plots", action="store_true")

    # --- Real data ---
    p.add_argument(
        "--real-data-csv",
        type=str,
        default="../data/measurements/Jeversen_2022_10_12_110132.csv",
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default="../data/measurements",
        help="Directory containing all measurement CSVs for multi-trajectory PPC",
    )
    p.add_argument(
        "--K-ppc",
        type=int,
        default=300,
        help="Number of posterior predictive samples per trajectory",
    )
    p.add_argument(
        "--skip-multi-ppc",
        action="store_true",
        help="Skip multi-trajectory PPC evaluation",
    )

    # --- Output ---
    p.add_argument("--results-root", type=str, default="experiments")

    # --- Execution mode ---
    p.add_argument("--train", action="store_true")
    p.add_argument("--eval", action="store_true")
    p.add_argument("--checkpoint", type=str, default=None)

    return p.parse_args()


def main():
    args = parse_args()

    # ---------------------------------------------------------------
    # 1) Load PSO results
    # ---------------------------------------------------------------
    pso_results = load_pso_results(args.pso_json)

    # ---------------------------------------------------------------
    # 2) Patch vehicle model with optimized params (BEFORE any JIT)
    # ---------------------------------------------------------------
    patch_vehicle_model(pso_results)

    # ---------------------------------------------------------------
    # 3) Compute prior bounds from PSO results (unless overridden)
    # ---------------------------------------------------------------
    bounds = compute_prior_bounds(pso_results)

    mu_low = args.prior_mu_low if args.prior_mu_low is not None else bounds["mu"][0]
    mu_high = args.prior_mu_high if args.prior_mu_high is not None else bounds["mu"][1]
    cd_low = args.prior_cd_low if args.prior_cd_low is not None else bounds["cd"][0]
    cd_high = args.prior_cd_high if args.prior_cd_high is not None else bounds["cd"][1]
    m_low = args.prior_m_low if args.prior_m_low is not None else bounds["m"][0]
    m_high = args.prior_m_high if args.prior_m_high is not None else bounds["m"][1]

    # Fixed param values = PSO-optimized centres
    global_params = pso_results["global_params"]
    traj_params = pso_results.get("trajectory_params", [])
    mu_values = [
        tp["params"]["mu"] for tp in traj_params if "mu" in tp.get("params", {})
    ]
    fixed_mu = sum(mu_values) / len(mu_values) if mu_values else 0.8
    fixed_cd = float(global_params.get("air_resistance", 0.27))
    fixed_m = float(global_params.get("mass", 1720))

    # Parse active parameters
    active_params = tuple(
        p.strip().lower() for p in args.params.split(",") if p.strip()
    )

    # ---------------------------------------------------------------
    # 4) Build config and run
    # ---------------------------------------------------------------
    from configs.config import ExperimentConfig
    from inference.unified_experiment import run_experiment

    do_train = True
    do_eval = True
    if args.train and not args.eval:
        do_eval = False
    elif args.eval and not args.train:
        do_train = False

    cfg = ExperimentConfig(
        # Core
        exp_name=args.exp_name,
        method="fnpe",
        device=args.device,
        # Seeds
        random_seed=args.seed,
        # Data
        num_simulations=args.num_sim or 2000,
        dt=0.01,
        T_seg=args.T_seg,
        active_parameters=active_params,
        # Prior bounds (from PSO)
        prior_low_mu=mu_low,
        prior_high_mu=mu_high,
        prior_low_cd=cd_low,
        prior_high_cd=cd_high,
        prior_low_m=m_low,
        prior_high_m=m_high,
        # Fixed defaults (PSO-optimized centres)
        fixed_mu=round(fixed_mu, 4),
        fixed_cd=round(fixed_cd, 4),
        fixed_m=round(fixed_m, 1),
        # Encoder
        encoder_type=args.encoder_type,
        encoder_hidden=args.encoder_hidden,
        # Training
        learning_rate=args.lr,
        training_batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        stop_after_epochs=args.stop_after_epochs,
        # FNPE
        fnpe_hidden_dim=args.fnpe_hidden_dim,
        fnpe_num_hidden=args.fnpe_num_hidden,
        fnpe_model_type=args.fnpe_model_type,
        fnpe_window_size=args.fnpe_window_size,
        fnpe_t_min=args.fnpe_t_min,
        fnpe_steps_per_epoch=args.fnpe_steps_per_epoch,
        fnpe_num_diffusion_steps=args.fnpe_diffusion_steps,
        fnpe_score_fn_type=args.fnpe_score_fn,
        fnpe_proposal_type=args.fnpe_proposal_type,
        fnpe_pilot_fraction=args.fnpe_pilot_fraction,
        fnpe_pilot_length=args.fnpe_pilot_length,
        fnpe_proposal_noise=args.fnpe_proposal_noise,
        fnpe_num_simulations=args.fnpe_num_sim,
        fnpe_skip_normalize=args.fnpe_skip_normalize,
        fnpe_clip_samples=args.fnpe_clip_samples and not args.fnpe_no_clip_samples,
        fnpe_gauss_precision_scale=args.fnpe_gauss_precision_scale,
        # Diagnostics
        run_sbc=not args.no_sbc,
        run_swd=not args.no_swd,
        run_one_step_rmse=not args.no_one_step,
        no_plots=args.no_plots,
        # Real data
        real_data_csv=args.real_data_csv,
        # Output
        results_root=args.results_root,
        # Execution mode
        do_train=do_train,
        do_eval=do_eval,
        checkpoint=args.checkpoint,
    )

    # Print summary
    print("=" * 70)
    print("FNPE Experiment with PSO-Optimized Parameters")
    print("=" * 70)
    print(f"  Active params:  {cfg.active_parameters}")
    print(f"  Prior mu:       [{cfg.prior_low_mu}, {cfg.prior_high_mu}]")
    print(f"  Prior cd:       [{cfg.prior_low_cd}, {cfg.prior_high_cd}]")
    print(f"  Prior m:        [{cfg.prior_low_m}, {cfg.prior_high_m}]")
    print(f"  Fixed mu:       {cfg.fixed_mu}")
    print(f"  Fixed cd:       {cfg.fixed_cd}")
    print(f"  Fixed m:        {cfg.fixed_m}")
    print(f"  FNPE sims:      {cfg.fnpe_num_simulations}")
    print(f"  T_seg:          {cfg.T_seg}")
    print(f"  Device:         {cfg.device}")
    print("=" * 70)

    results = run_experiment(cfg)

    # ---------------------------------------------------------------
    # 5) Multi-trajectory PPC evaluation on all measurement CSVs
    # ---------------------------------------------------------------
    if not args.skip_multi_ppc and results.get("posterior") is not None:
        from inference.unified_experiment import run_multi_trajectory_ppc
        from utils.env_utils import get_device
        import torch

        exp_dir = results["exp_dir"]
        fig_dir = exp_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        # T_event = cfg.T_seg (training and real window share same length)
        T_event = cfg.T_seg
        device = get_device(cfg.device)

        run_multi_trajectory_ppc(
            cfg=cfg,
            exp_dir=exp_dir,
            fig_dir=fig_dir,
            posterior=results["posterior"],
            normalizer=results["normalizer"],
            device=device,
            T_event=T_event,
            data_dir=args.data_dir,
            K_ppc=args.K_ppc,
            pso_trajectory_params=pso_results.get("trajectory_params"),
        )
    elif args.skip_multi_ppc:
        print("[PSO] Skipping multi-trajectory PPC (--skip-multi-ppc)")
    else:
        print("[PSO] Skipping multi-trajectory PPC (no posterior available)")


if __name__ == "__main__":
    main()
