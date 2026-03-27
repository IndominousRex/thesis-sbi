#!/usr/bin/env python
"""
Run comparison experiments for all SBI methods (NPE, NPSE, FNPE, Simformer).

NPE, NPSE, and Simformer reuse the same cached simulation dataset for a fair
comparison. FNPE keeps its own task-specific simulation pipeline, so including
it is useful for a broader benchmark but not a strict shared-dataset study.

Usage:
    python scripts/run_simulation_comparison.py --exp-name sim_compare --num-simulations 2000
"""

import argparse
import json
import sys
import traceback
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

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
    parser = argparse.ArgumentParser(
        description="Compare all SBI methods on simulation data"
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default=f"simformer_compare_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        help="Base experiment name",
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=2000,
        help="Number of simulations (default: 2000)",
    )
    parser.add_argument(
        "--T-seg",
        type=int,
        default=1000,
        help="Sequence length (default: 1000)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "auto"],
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["npe", "npse", "simformer"],
        choices=["npe", "npse", "fnpe", "simformer"],
        help="Methods to run (default: npe npse simformer)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: fewer simulations and training steps",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Minimal orchestration smoke test: tiny shared dataset, tiny models, "
            "no evaluation diagnostics"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--params",
        type=active_param_type,
        default=PARAMETER_ORDER,
        help="Comma-separated parameters to infer (mu, cd, m)",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Do not write the comparison summary JSON.",
    )
    parser.add_argument(
        "--independent-datasets",
        action="store_true",
        help="Disable dataset cache/reuse so each method run generates its own data.",
    )
    parser.add_argument(
        "--requested-budget-steps",
        type=int,
        default=None,
        help="Requested total simulator-step budget for budget-driven benchmark runs.",
    )
    parser.add_argument(
        "--run-sbc",
        action="store_true",
        help="Enable SBC diagnostics. Disabled by default for benchmark runs.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the experiment directory containing the saved model for eval-only runs.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and evaluate from --checkpoint.",
    )
    return parser.parse_args()


def _fnpe_effective_budget_steps(
    num_simulations: int,
    window_size: int,
    pilot_fraction: float,
    pilot_length: int,
) -> int:
    num_pilots = int(round(num_simulations * pilot_fraction))
    return int(num_simulations * window_size + num_pilots * pilot_length)


def _infer_fnpe_num_simulations_for_budget(
    requested_budget_steps: int,
    window_size: int,
    pilot_fraction: float,
    pilot_length: int,
) -> int:
    per_sim_step_cost = window_size + pilot_fraction * pilot_length
    approx = max(1, int(round(requested_budget_steps / max(per_sim_step_cost, 1e-9))))
    best = approx
    best_gap = abs(
        _fnpe_effective_budget_steps(approx, window_size, pilot_fraction, pilot_length)
        - requested_budget_steps
    )
    for delta in range(-256, 257):
        candidate = approx + delta
        if candidate <= 0:
            continue
        gap = abs(
            _fnpe_effective_budget_steps(
                candidate, window_size, pilot_fraction, pilot_length
            )
            - requested_budget_steps
        )
        if gap < best_gap:
            best = candidate
            best_gap = gap
    return int(best)


def create_config(
    method: str,
    exp_name: str,
    num_simulations: int,
    T_seg: int,
    device: str,
    seed: int,
    active_parameters=PARAMETER_ORDER,
    quick: bool = False,
    smoke: bool = False,
    requested_budget_steps: int | None = None,
    run_sbc: bool = False,
    checkpoint: str | None = None,
    do_train: bool = True,
    dataset_id: str = None,  # type: ignore
    reuse_dataset: bool = False,
    independent_datasets: bool = False,
) -> ExperimentConfig:
    """Create experiment config for a method."""

    cfg_exp_name = exp_name if exp_name.endswith(f"_{method}") else f"{exp_name}_{method}"

    # Base config
    cfg_kwargs = {
        "method": method,
        "exp_name": cfg_exp_name,
        "num_simulations": num_simulations,
        "requested_budget_steps": requested_budget_steps,
        "derived_num_simulations": num_simulations,
        "T_seg": T_seg,
        "active_parameters": tuple(active_parameters),
        "device": device,
        "random_seed": seed,
        "sim_seed": seed,
        "train_seed": seed + 1,
        # Dataset caching
        "cache_dataset": not independent_datasets,
        "dataset_id": None if independent_datasets else dataset_id,
        "reuse_dataset": False if independent_datasets else reuse_dataset,
        # Diagnostics
        "run_sbc": run_sbc,
        "run_swd": False,
        "run_one_step_rmse": True,
        "run_posterior_plots": True,
        "run_lc2st": method == "npe",  # Only for NPE
        "unify_eval_budgets": True,
        "num_sbc_samples": 400,
        "num_posterior_samples_sbc": 2000,
        "num_test_simulations": 500,
        "run_simulated_test_eval": True,
        "run_simulated_ppc": True,
        "num_simulated_ppc_examples": 50,
        "num_simulated_ppc_plot_examples": 2,
        "simulated_test_ppc_samples": 400,
        # Simulated-data-only runs
        "real_data_csv": None,
        "checkpoint": checkpoint,
        "do_train": do_train,
    }

    if smoke:
        cfg_kwargs.update(
            {
                "num_simulations": min(32, num_simulations),
                "T_seg": min(128, T_seg),
                "batch_sim": min(32, num_simulations),
                "training_batch_size": 32,
                "num_epochs": 2,
                "stop_after_epochs": 1,
                "encoder_hidden": 8,
                "embedding_output_dim": 16,
                "maf_hidden_features": 32,
                "maf_num_transforms": 2,
                "run_sbc": False,
                "run_one_step_rmse": False,
                "run_posterior_plots": False,
                "run_lc2st": False,
                "run_simulated_test_eval": False,
                "run_simulated_ppc": False,
                "num_test_simulations": 0,
                "no_plots": True,
                "do_eval": False,
            }
        )

    # Quick mode settings
    if quick and not smoke:
        cfg_kwargs.update(
            {
                "num_simulations": min(128, num_simulations),
                "batch_sim": min(128, num_simulations),
                "training_batch_size": 64,
                "num_sbc_samples": 5,
                "num_posterior_samples_sbc": 50,
                "num_test_simulations": 20,
                "num_simulated_ppc_examples": 5,
                "num_simulated_ppc_plot_examples": 1,
                "simulated_test_ppc_samples": 20,
                "num_epochs": 20,
                "stop_after_epochs": 5,
                "run_posterior_plots": False,
                "run_lc2st": False,
                "no_plots": True,
            }
        )
        if method == "simformer":
            cfg_kwargs.update(
                {
                    "simformer_num_timepoints": 8,
                    "simformer_token_dim": 16,
                    "simformer_condition_token_dim": 8,
                    "simformer_time_embedding_dim": 32,
                    "simformer_num_layers": 2,
                    "simformer_num_heads": 2,
                    "simformer_attn_size": 8,
                    "simformer_batch_size": 16,
                    "simformer_train_steps_scaling": 1,
                    "simformer_min_train_steps": 250,
                    "simformer_max_train_steps": 250,
                    "simformer_val_repeat": 2,
                    "simformer_val_every": 10,
                    "simformer_stop_early_count": 3,
                    "simformer_num_diffusion_steps": 50,
                }
            )
        elif method == "fnpe":
            cfg_kwargs["fnpe_num_simulations"] = min(
                2000, max(256, num_simulations * 4)
            )
            cfg_kwargs.update(
                {
                    "fnpe_num_outer_epochs": 8,
                    "fnpe_num_inner_epochs": 4,
                    "fnpe_batch_size": 256,
                    "fnpe_validation_size": 0,
                }
            )

        if device == "cpu":
            cfg_kwargs.update(
                {
                    "num_sbc_samples": 3,
                    "num_posterior_samples_sbc": 20,
                    "num_test_simulations": 10,
                    "simulated_test_ppc_samples": 10,
                    "num_simulated_ppc_plot_examples": 1,
                }
            )
            if method == "simformer":
                cfg_kwargs.update(
                    {
                        "simformer_num_timepoints": 6,
                        "simformer_num_diffusion_steps": 20,
                        "simformer_batch_size": 8,
                        "simformer_min_train_steps": 50,
                        "simformer_max_train_steps": 50,
                        "simformer_val_repeat": 1,
                        "simformer_val_every": 5,
                    }
                )
            elif method == "fnpe":
                cfg_kwargs.update(
                    {
                        "run_sbc": False,
                        "fnpe_num_simulations": 64,
                        "fnpe_num_outer_epochs": 4,
                        "fnpe_num_inner_epochs": 2,
                        "fnpe_batch_size": 64,
                        "fnpe_validation_size": 0,
                        "fnpe_pilot_fraction": 0.05,
                        "fnpe_pilot_length": 256,
                        "num_test_simulations": 3,
                        "num_simulated_ppc_examples": 2,
                        "num_simulated_ppc_plot_examples": 1,
                        "simulated_test_ppc_samples": 5,
                    }
                )

    # Method-specific overrides
    if method == "simformer":
        if smoke:
            cfg_kwargs.update(
                {
                    "simformer_num_timepoints": 4,
                    "simformer_token_dim": 16,
                    "simformer_condition_token_dim": 8,
                    "simformer_time_embedding_dim": 32,
                    "simformer_num_layers": 2,
                    "simformer_num_heads": 2,
                    "simformer_attn_size": 8,
                    "simformer_batch_size": 8,
                    "simformer_train_steps_scaling": 1,
                    "simformer_min_train_steps": 25,
                    "simformer_max_train_steps": 25,
                    "simformer_val_repeat": 1,
                    "simformer_val_every": 5,
                    "simformer_stop_early_count": 2,
                    "simformer_num_diffusion_steps": 20,
                }
            )
        # Use smaller model for faster training if quick mode
        elif quick:
            pass
    elif method == "fnpe":
        # FNPE generates its own data
        if smoke:
            cfg_kwargs["fnpe_num_simulations"] = 128
            cfg_kwargs.update(
                {
                    "fnpe_num_outer_epochs": 2,
                    "fnpe_num_inner_epochs": 2,
                    "fnpe_batch_size": 64,
                    "fnpe_validation_size": 0,
                }
            )
        else:
            # For comparison runs, align FNPE to the same nominal simulation budget
            # as the shared-data methods. FNPE still differs in training windows and
            # optimization dynamics, but it should not silently get 10x more data.
            cfg_kwargs["fnpe_num_simulations"] = (
                num_simulations
                if not quick
                else cfg_kwargs.get("fnpe_num_simulations", 2000)
            )

    cfg = ExperimentConfig(**cfg_kwargs)
    if (
        method == "fnpe"
        and requested_budget_steps is not None
        and not quick
        and not smoke
    ):
        cfg.fnpe_num_simulations = _infer_fnpe_num_simulations_for_budget(
            requested_budget_steps=requested_budget_steps,
            window_size=cfg.fnpe_window_size,
            pilot_fraction=cfg.fnpe_pilot_fraction,
            pilot_length=cfg.fnpe_pilot_length,
        )

    return cfg


def _create_method_configs(args) -> tuple[dict[str, ExperimentConfig], str | None]:
    """Create all per-method configs and return shared dataset ID if applicable."""
    if args.eval_only and not args.checkpoint:
        raise ValueError("--eval-only requires --checkpoint.")
    if args.checkpoint and len(args.methods) != 1:
        raise ValueError("--checkpoint can only be used with a single method.")

    configs_used: dict[str, ExperimentConfig] = {}
    shared_methods = [m for m in args.methods if m != "fnpe"]
    shared_dataset_id = None

    if shared_methods and not args.independent_datasets:
        seed_cfg = create_config(
            method=shared_methods[0],
            exp_name=args.exp_name,
            num_simulations=args.num_simulations,
            T_seg=args.T_seg,
            device=args.device,
            seed=args.seed,
            active_parameters=args.params,
            quick=args.quick,
            smoke=args.smoke,
            requested_budget_steps=args.requested_budget_steps,
            run_sbc=args.run_sbc,
            checkpoint=args.checkpoint,
            do_train=not args.eval_only,
            dataset_id=None,
            reuse_dataset=True,
            independent_datasets=args.independent_datasets,
        )
        shared_dataset_id = seed_cfg.dataset_id

    for method in args.methods:
        cfg = create_config(
            method=method,
            exp_name=args.exp_name,
            num_simulations=args.num_simulations,
            T_seg=args.T_seg,
            device=args.device,
            seed=args.seed,
            active_parameters=args.params,
            quick=args.quick,
            smoke=args.smoke,
            requested_budget_steps=args.requested_budget_steps,
            run_sbc=args.run_sbc,
            checkpoint=args.checkpoint,
            do_train=not args.eval_only,
            dataset_id=shared_dataset_id,
            reuse_dataset=method != "fnpe" and shared_dataset_id is not None,
            independent_datasets=args.independent_datasets,
        )
        configs_used[method] = cfg

    return configs_used, shared_dataset_id


def _run_serial(
    args,
    configs_used: dict[str, ExperimentConfig],
) -> dict[str, dict]:
    """Fallback serial execution path."""
    results: dict[str, dict] = {}
    for i, method in enumerate(args.methods):
        print(f"\n{'='*70}")
        print(f"Running {method.upper()} ({i+1}/{len(args.methods)})")
        print("=" * 70)
        cfg = configs_used[method]
        try:
            exp_results = run_experiment(cfg)
            results[method] = {
                "method": method,
                "exp_dir": str(exp_results.get("exp_dir")),
                "metrics": exp_results.get("metrics", {}),
            }
        except Exception as e:
            print(f"[ERROR] {method} failed: {e}")
            traceback.print_exc()
            results[method] = {"method": method, "error": str(e)}
    return results


def _print_and_save_summary(args, results, configs_used):
    """Print comparison summary and save it to disk."""
    print("=" * 70)
    print(f"SBI Method Comparison: {args.exp_name}")
    print(f"Methods: {args.methods}")
    print(f"Simulations: {args.num_simulations}, T_seg: {args.T_seg}")
    if args.requested_budget_steps is not None:
        print(f"Requested budget steps: {args.requested_budget_steps}")
    print(f"Device: {args.device}, Quick: {args.quick}, Smoke: {args.smoke}")
    if "fnpe" in args.methods:
        print(
            "[WARN] FNPE uses its own simulator/training-data pipeline; "
            "only NPE/NPSE/Simformer are strict shared-dataset comparisons."
        )
    print("=" * 70)

    # Print summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    summary_metrics = {}
    for method in args.methods:
        res = results.get(method, {"error": "missing result"})
        if "error" in res:
            print(f"\n{method.upper()}: FAILED - {res['error']}")
            continue

        print(f"\n{method.upper()}:")
        summary = {"method": method}
        cfg_used = configs_used.get(method)
        if cfg_used is not None:
            summary["configured_num_simulations"] = cfg_used.num_simulations
            summary["training_num_simulations"] = (
                cfg_used.fnpe_num_simulations
                if method == "fnpe"
                else cfg_used.num_simulations
            )
            summary["T_seg"] = cfg_used.T_seg
            if method == "fnpe":
                summary["fnpe_num_outer_epochs"] = cfg_used.fnpe_num_outer_epochs
                summary["fnpe_num_inner_epochs"] = cfg_used.fnpe_num_inner_epochs

        metrics = res.get("metrics", {})
        if "budget_metadata" in metrics:
            summary["budget_metadata"] = metrics["budget_metadata"]
            budget = metrics["budget_metadata"]
            requested_budget = budget.get("requested_budget_steps")
            effective_budget = budget.get("effective_budget_steps")
            if isinstance(requested_budget, int):
                print(f"  Requested budget steps: {requested_budget}")
                summary["requested_budget_steps"] = requested_budget
            if isinstance(effective_budget, int):
                print(f"  Effective budget steps: {effective_budget}")
                summary["effective_budget_steps"] = effective_budget

        # Training summary
        train = metrics.get("training_summary", {})
        if train:
            if "final_loss" in train:
                print(f"  Final loss: {train['final_loss']:.4f}")
                summary["final_loss"] = train["final_loss"]
            if "train_time_s" in train:
                print(f"  Training time: {train['train_time_s']:.1f}s")
                summary["train_time_s"] = train["train_time_s"]
            if "num_train_steps" in train and train["num_train_steps"] is not None:
                print(f"  Optimizer steps: {train['num_train_steps']}")
                summary["num_train_steps"] = train["num_train_steps"]

        # SBC
        if "sbc_check_stats" in metrics:
            sbc = metrics["sbc_check_stats"]
            if "c2st_accuracy" in sbc:
                print(f"  SBC C2ST accuracy: {sbc['c2st_accuracy']:.3f}")
                summary["sbc_c2st"] = sbc["c2st_accuracy"]

        # W2 posterior
        if "w2_posterior_vs_true" in metrics:
            w2 = metrics["w2_posterior_vs_true"]
            posterior_vs_true = {
                "w2_mean": w2.get("w2_mean"),
                "w2_std": w2.get("w2_std"),
                "l2_error_mean": w2.get("l2_error_mean"),
                "l2_error_std": w2.get("l2_error_std"),
                "swd_posterior_vs_true": w2.get("swd_posterior_vs_true"),
                "coverage_90": w2.get("coverage_90"),
                "coverage_50": w2.get("coverage_50"),
                "coverage_curve_mae": w2.get("coverage_curve_mae"),
                "w1_per_dim": w2.get("w1_per_dim"),
                "sampling_time_mean_s": w2.get("sampling_time_mean_s"),
                "sampling_time_std_s": w2.get("sampling_time_std_s"),
            }
            print(f"  W2 to ground truth (mean): {w2.get('w2_mean', 'N/A')}")
            print(f"  W2 to ground truth (std): {w2.get('w2_std', 'N/A')}")
            print(f"  L2 error (mean): {w2.get('l2_error_mean', 'N/A')}")
            print(f"  L2 error (std): {w2.get('l2_error_std', 'N/A')}")
            if isinstance(w2.get("swd_posterior_vs_true"), (int, float)):
                print(f"  SWD posterior vs true: {w2['swd_posterior_vs_true']:.4f}")
            print(
                f"  Coverage 90%: {w2.get('coverage_90', 'N/A'):.2%}"
                if isinstance(w2.get("coverage_90"), (int, float))
                else f"  Coverage 90%: N/A"
            )
            print(
                f"  Coverage 50%: {w2.get('coverage_50', 'N/A'):.2%}"
                if isinstance(w2.get("coverage_50"), (int, float))
                else f"  Coverage 50%: N/A"
            )
            if isinstance(w2.get("coverage_curve_mae"), (int, float)):
                print(f"  Coverage curve MAE: {w2['coverage_curve_mae']:.4f}")
            if isinstance(w2.get("sampling_time_mean_s"), (int, float)):
                print(f"  Sampling time / case: {w2['sampling_time_mean_s']:.4f}s")
            summary["posterior_vs_true"] = posterior_vs_true
            summary["w2_mean"] = w2.get("w2_mean")
            summary["w2_std"] = w2.get("w2_std")
            summary["l2_error"] = w2.get("l2_error_mean")
            summary["l2_error_std"] = w2.get("l2_error_std")
            summary["swd_posterior_vs_true"] = w2.get("swd_posterior_vs_true")
            summary["coverage_90"] = w2.get("coverage_90")
            summary["coverage_50"] = w2.get("coverage_50")
            summary["coverage_curve_mae"] = w2.get("coverage_curve_mae")
            summary["w1_per_dim"] = w2.get("w1_per_dim")
            summary["sampling_time_mean_s"] = w2.get("sampling_time_mean_s")
            summary["sampling_time_std_s"] = w2.get("sampling_time_std_s")

        if "heldout_test_stats" in metrics:
            heldout_stats = metrics["heldout_test_stats"]
            summary["heldout_test_stats"] = heldout_stats
            rank_uniformity = heldout_stats.get("rank_uniformity", {})
            min_p = min(
                (
                    float(v.get("ks_pvalue"))
                    for v in rank_uniformity.values()
                    if isinstance(v, dict) and v.get("ks_pvalue") is not None
                ),
                default=None,
            )
            if min_p is not None:
                print(f"  Held-out min KS p-value: {min_p:.4g}")
                summary["heldout_min_ks_pvalue"] = min_p

        if "heldout_test_ppc" in metrics:
            heldout_ppc = metrics["heldout_test_ppc"]
            aggregate = heldout_ppc.get("aggregate", {})
            summary["heldout_test_ppc"] = aggregate
            if isinstance(aggregate.get("rmse_mean"), (int, float)):
                print(f"  Held-out PPC RMSE mean: {aggregate['rmse_mean']:.4f}")
                summary["heldout_ppc_rmse_mean"] = aggregate["rmse_mean"]
            if isinstance(aggregate.get("w2_mean"), (int, float)):
                print(f"  Held-out PPC W2 mean: {aggregate['w2_mean']:.4f}")
                summary["heldout_ppc_w2_mean"] = aggregate["w2_mean"]

        # C2ST
        if "c2st" in metrics:
            c2st = metrics["c2st"]
            vals = [float(v) for v in c2st.values() if isinstance(v, (int, float))]
            if vals:
                c2st_mean = float(sum(vals) / len(vals))
                print(f"  C2ST mean: {c2st_mean:.3f}")
                summary["c2st_mean"] = c2st_mean

        # 1-step RMSE
        if "one_step_rmse" in metrics:
            rmse = metrics["one_step_rmse"]
            if "rmse_overall" in rmse and rmse["rmse_overall"] is not None:
                print(f"  1-step RMSE: {rmse['rmse_overall']:.4f}")
                summary["one_step_rmse"] = rmse["rmse_overall"]

        summary_metrics[method] = summary

    # Save summary to file
    summary_path = Path("experiments") / f"{args.exp_name}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary_metrics, f, indent=2)
    print(f"\n[INFO] Summary saved to {summary_path}")
    return summary_metrics


def run_comparison(args):
    """Run comparison experiments."""
    if args.quick and args.smoke:
        raise ValueError("Use either --quick or --smoke, not both.")

    configs_used, shared_dataset_id = _create_method_configs(args)

    print("=" * 70)
    print(f"SBI Method Comparison: {args.exp_name}")
    print(f"Methods: {args.methods}")
    print(f"Simulations: {args.num_simulations}, T_seg: {args.T_seg}")
    if args.requested_budget_steps is not None:
        print(f"Requested budget steps: {args.requested_budget_steps}")
    print(f"Device: {args.device}, Quick: {args.quick}, Smoke: {args.smoke}")
    if "fnpe" in args.methods:
        print(
            "[WARN] FNPE uses its own simulator/training-data pipeline; "
            "only NPE/NPSE/Simformer are strict shared-dataset comparisons."
        )
    if shared_dataset_id is not None:
        print(f"[INFO] Shared dataset ID: {shared_dataset_id}")
    results = _run_serial(args, configs_used)

    if not args.no_summary:
        _print_and_save_summary(args, results, configs_used)
    return results


if __name__ == "__main__":
    args = parse_args()
    run_comparison(args)
