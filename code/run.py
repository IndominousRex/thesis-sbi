#!/usr/bin/env python
"""
Unified entry point for SBI experiments.

Supports all methods (NPE, NPSE, FNPE) with identical data generation,
normalization, and evaluation for fair comparisons.

Usage:
    python run.py --method npe --exp-name baseline --num-sim 2000
    python run.py --method npse --exp-name baseline --sde-type ve
    python run.py --method fnpe --exp-name baseline --fnpe-model-type gru
"""

import argparse
from configs.config import ExperimentConfig, PARAMETER_ORDER
from inference.unified_experiment import run_experiment


def active_param_type(value: str):
    """Parse comma-separated parameter names (mu, cd, m)."""
    allowed = set(PARAMETER_ORDER)
    parts = [p.strip().lower() for p in value.split(",") if p.strip()]

    if not parts:
        return PARAMETER_ORDER

    cleaned = []
    for p in parts:
        if p not in allowed:
            raise argparse.ArgumentTypeError(
                f"Unknown parameter '{p}'. Choose from {PARAMETER_ORDER}."
            )
        if p not in cleaned:
            cleaned.append(p)
    return tuple(cleaned)


def parse_args():
    p = argparse.ArgumentParser(
        description="Run SBI experiment with unified pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ==========================================================================
    # Core arguments
    # ==========================================================================
    core = p.add_argument_group("Core settings")
    core.add_argument(
        "--method",
        type=str,
        choices=["npe", "npse", "fnpe"],
        default="npe",
        help="Inference method to use",
    )
    core.add_argument("--exp-name", type=str, default="baseline")
    core.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )

    # ==========================================================================
    # Execution mode
    # ==========================================================================
    mode = p.add_argument_group("Execution mode")
    mode.add_argument("--train", action="store_true", help="Run training only")
    mode.add_argument("--eval", action="store_true", help="Run evaluation only")
    mode.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint for evaluation without training",
    )

    # ==========================================================================
    # Seeds
    # ==========================================================================
    seeds = p.add_argument_group("Random seeds")
    seeds.add_argument("--seed", type=int, default=42, help="Base random seed")
    seeds.add_argument(
        "--sim-seed",
        type=int,
        default=None,
        help="Seed for data simulation (defaults to --seed)",
    )
    seeds.add_argument(
        "--train-seed",
        type=int,
        default=None,
        help="Seed for training (defaults to --seed)",
    )

    # ==========================================================================
    # Simulation / Data
    # ==========================================================================
    data = p.add_argument_group("Data settings")
    data.add_argument("--num-sim", type=int, default=2000)
    data.add_argument("--dt", type=float, default=0.01)
    data.add_argument("--T-seg", type=int, default=3000, help="Sequence length")
    data.add_argument(
        "--params",
        type=active_param_type,
        default=PARAMETER_ORDER,
        help="Comma-separated parameters to infer (mu, cd, m)",
    )

    # Fixed parameter values
    data.add_argument("--fixed-mu", type=float, default=1.0)
    data.add_argument("--fixed-cd", type=float, default=0.3)
    data.add_argument("--fixed-m", type=float, default=1700.0)

    # ==========================================================================
    # Dataset caching
    # ==========================================================================
    cache = p.add_argument_group("Dataset caching (for fair comparisons)")
    cache.add_argument(
        "--dataset-id",
        type=str,
        default=None,
        help="Unique ID for cached dataset",
    )
    cache.add_argument(
        "--reuse-dataset",
        action="store_true",
        help="Load cached dataset if available",
    )
    cache.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable dataset caching",
    )

    # ==========================================================================
    # Encoder settings
    # ==========================================================================
    encoder = p.add_argument_group("Encoder settings")
    encoder.add_argument(
        "--encoder-type",
        type=str,
        choices=["bigru", "causalcnn", "transformer"],
        default="bigru",
    )
    encoder.add_argument("--encoder-hidden", type=int, default=32)

    # ==========================================================================
    # Training settings
    # ==========================================================================
    training = p.add_argument_group("Training settings")
    training.add_argument("--lr", type=float, default=1e-3)
    training.add_argument("--batch-size", type=int, default=512)
    training.add_argument(
        "--stop-after-epochs",
        type=int,
        default=30,
        help="Patience for early stopping (default 30 for noisy val loss)",
    )
    training.add_argument("--num-epochs", type=int, default=20)

    # ==========================================================================
    # NPE-specific
    # ==========================================================================
    npe = p.add_argument_group("NPE-specific")
    npe.add_argument("--maf-hidden", type=int, default=128)
    npe.add_argument("--maf-transforms", type=int, default=8)

    # ==========================================================================
    # NPSE-specific
    # ==========================================================================
    npse = p.add_argument_group("NPSE-specific")
    npse.add_argument(
        "--sde-type",
        type=str,
        choices=["ve", "vp", "subvp"],
        default="ve",
        help="SDE type: ve (SMLD), vp (DDPM), subvp",
    )

    # ==========================================================================
    # FNPE-specific
    # ==========================================================================
    fnpe = p.add_argument_group("FNPE-specific")
    fnpe.add_argument("--fnpe-hidden-dim", type=int, default=128)
    fnpe.add_argument("--fnpe-num-hidden", type=int, default=5)
    fnpe.add_argument(
        "--fnpe-model-type",
        type=str,
        choices=["gru", "linear"],
        default="gru",
    )
    fnpe.add_argument(
        "--fnpe-window-size",
        type=int,
        default=2,
        help="Markov window size (keep small, e.g. 2-10)",
    )
    fnpe.add_argument(
        "--fnpe-max-obs-len",
        type=int,
        default=100,
        help="Max observation length for inference. With --fnpe-normalize-score, can use 100-500.",
    )
    fnpe.add_argument(
        "--fnpe-normalize-score",
        action="store_true",
        default=True,
        help="Use mean instead of sum over windows (default: True, more stable)",
    )
    fnpe.add_argument(
        "--fnpe-no-normalize-score",
        action="store_true",
        help="Use original FNPE formula (sum over windows, may cause NaN for large N)",
    )
    fnpe.add_argument("--fnpe-steps-per-epoch", type=int, default=10000)
    fnpe.add_argument("--fnpe-diffusion-steps", type=int, default=500)
    fnpe.add_argument(
        "--fnpe-score-fn",
        type=str,
        choices=["fnpe", "uncorrected", "gauss_corrected"],
        default="gauss_corrected",
        help="Score composition: 'gauss_corrected' (paper default), 'fnpe', or 'uncorrected'",
    )
    fnpe.add_argument(
        "--fnpe-proposal-type",
        type=str,
        choices=["pred", "naive", "trajectory"],
        default="pred",
        help="Proposal type for FNPE training: 'pred' (correct, default), 'naive', or 'trajectory' (old)",
    )

    # ==========================================================================
    # Diagnostics
    # ==========================================================================
    diag = p.add_argument_group("Diagnostics")
    diag.add_argument("--num-sbc-samples", type=int, default=200)
    diag.add_argument("--num-lc2st-samples", type=int, default=None)
    diag.add_argument("--no-sbc", action="store_true")
    diag.add_argument("--no-swd", action="store_true")
    diag.add_argument("--no-one-step", action="store_true")
    diag.add_argument("--no-plots", action="store_true")

    # ==========================================================================
    # Real data evaluation
    # ==========================================================================
    real = p.add_argument_group("Real data evaluation")
    real.add_argument(
        "--real-data-csv",
        type=str,
        default="../data/measurements/Jeversen_2022_10_12_110132.csv",
        help="Path to real-data CSV for evaluation (set to empty string to disable)",
    )

    # ==========================================================================
    # Output
    # ==========================================================================
    output = p.add_argument_group("Output")
    output.add_argument("--results-root", type=str, default="experiments")

    return p.parse_args()


def main():
    args = parse_args()

    # Determine execution mode
    do_train = True
    do_eval = True
    if args.train and not args.eval:
        do_eval = False
    elif args.eval and not args.train:
        do_train = False

    # Build config
    cfg = ExperimentConfig(
        # Core
        exp_name=args.exp_name,
        method=args.method,
        device=args.device,
        # Seeds
        random_seed=args.seed,
        sim_seed=args.sim_seed,
        train_seed=args.train_seed,
        # Data
        num_simulations=args.num_sim,
        dt=args.dt,
        T_seg=args.T_seg,
        active_parameters=args.params,
        fixed_mu=args.fixed_mu,
        fixed_cd=args.fixed_cd,
        fixed_m=args.fixed_m,
        # Dataset caching
        dataset_id=args.dataset_id,
        reuse_dataset=args.reuse_dataset,
        cache_dataset=not args.no_cache,
        # Encoder
        encoder_type=args.encoder_type,
        encoder_hidden=args.encoder_hidden,
        # Training
        learning_rate=args.lr,
        training_batch_size=args.batch_size,
        stop_after_epochs=args.stop_after_epochs,
        num_epochs=args.num_epochs,
        # NPE
        maf_hidden_features=args.maf_hidden,
        maf_num_transforms=args.maf_transforms,
        # NPSE
        sde_type=args.sde_type,
        # FNPE
        fnpe_hidden_dim=args.fnpe_hidden_dim,
        fnpe_num_hidden=args.fnpe_num_hidden,
        fnpe_model_type=args.fnpe_model_type,
        fnpe_window_size=args.fnpe_window_size,
        fnpe_max_obs_len=args.fnpe_max_obs_len,
        fnpe_normalize_score=not args.fnpe_no_normalize_score,
        fnpe_steps_per_epoch=args.fnpe_steps_per_epoch,
        fnpe_num_diffusion_steps=args.fnpe_diffusion_steps,
        fnpe_score_fn_type=args.fnpe_score_fn,
        fnpe_proposal_type=args.fnpe_proposal_type,  # "pred" (correct), "naive", "trajectory" (old)
        # Diagnostics
        num_sbc_samples=args.num_sbc_samples,
        num_lc2st_samples=args.num_lc2st_samples,
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

    run_experiment(cfg)


if __name__ == "__main__":
    main()
