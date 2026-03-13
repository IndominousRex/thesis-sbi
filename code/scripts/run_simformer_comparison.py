#!/usr/bin/env python
"""
Run comparison experiments for all SBI methods (NPE, NPSE, FNPE, Simformer).

NPE, NPSE, and Simformer reuse the same cached simulation dataset for a fair
comparison. FNPE keeps its own task-specific simulation pipeline, so including
it is useful for a broader benchmark but not a strict shared-dataset study.

Usage:
    python scripts/run_simformer_comparison.py --exp-name sim_compare --num-simulations 2000
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import ExperimentConfig
from inference.unified_experiment import run_experiment


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
    return parser.parse_args()


def create_config(
    method: str,
    exp_name: str,
    num_simulations: int,
    T_seg: int,
    device: str,
    seed: int,
    quick: bool = False,
    smoke: bool = False,
    dataset_id: str = None,  # type: ignore
    reuse_dataset: bool = False,
) -> ExperimentConfig:
    """Create experiment config for a method."""

    # Base config
    cfg_kwargs = {
        "method": method,
        "exp_name": f"{exp_name}_{method}",
        "num_simulations": num_simulations,
        "T_seg": T_seg,
        "device": device,
        "random_seed": seed,
        "sim_seed": seed,
        "train_seed": seed + 1,
        # Dataset caching
        "cache_dataset": True,
        "dataset_id": dataset_id,
        "reuse_dataset": reuse_dataset,
        # Diagnostics
        "run_sbc": True,
        "run_swd": False,
        "run_one_step_rmse": True,
        "run_posterior_plots": True,
        "run_lc2st": method == "npe",  # Only for NPE
        # Simulated-data-only runs
        "real_data_csv": None,
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
                "no_plots": True,
                "do_eval": False,
            }
        )

    # Quick mode settings
    if quick and not smoke:
        cfg_kwargs.update(
            {
                "num_simulations": min(500, num_simulations),
                "num_sbc_samples": 50,
                "num_epochs": 50,
                "stop_after_epochs": 10,
            }
        )
        if method == "simformer":
            cfg_kwargs["simformer_num_train_steps"] = 5000
        elif method == "fnpe":
            cfg_kwargs["fnpe_num_simulations"] = min(10000, num_simulations * 5)
            cfg_kwargs["fnpe_max_epochs"] = 500

    # Method-specific overrides
    if method == "simformer":
        if smoke:
            cfg_kwargs.update(
                {
                    "simformer_token_dim": 16,
                    "simformer_condition_token_dim": 8,
                    "simformer_time_embedding_dim": 32,
                    "simformer_num_layers": 2,
                    "simformer_num_heads": 2,
                    "simformer_attn_size": 8,
                    "simformer_num_train_steps": 25,
                    "simformer_batch_size": 32,
                    "simformer_num_diffusion_steps": 20,
                }
            )
        # Use smaller model for faster training if quick mode
        elif quick:
            cfg_kwargs.update(
                {
                    "simformer_num_layers": 4,
                    "simformer_num_heads": 4,
                    "simformer_num_train_steps": 10000,
                }
            )
    elif method == "fnpe":
        # FNPE generates its own data
        if smoke:
            cfg_kwargs["fnpe_num_simulations"] = 128
            cfg_kwargs["fnpe_max_epochs"] = 5
        else:
            cfg_kwargs["fnpe_num_simulations"] = (
                num_simulations * 10 if not quick else 10000
            )

    return ExperimentConfig(**cfg_kwargs)


def run_comparison(args):
    """Run comparison experiments."""
    if args.quick and args.smoke:
        raise ValueError("Use either --quick or --smoke, not both.")

    results = {}
    dataset_id = None

    print("=" * 70)
    print(f"SBI Method Comparison: {args.exp_name}")
    print(f"Methods: {args.methods}")
    print(f"Simulations: {args.num_simulations}, T_seg: {args.T_seg}")
    print(f"Device: {args.device}, Quick: {args.quick}, Smoke: {args.smoke}")
    if "fnpe" in args.methods:
        print(
            "[WARN] FNPE uses its own simulator/training-data pipeline; "
            "only NPE/NPSE/Simformer are strict shared-dataset comparisons."
        )
    print("=" * 70)

    # Run each method
    for i, method in enumerate(args.methods):
        print(f"\n{'='*70}")
        print(f"Running {method.upper()} ({i+1}/{len(args.methods)})")
        print("=" * 70)

        # First method generates dataset, others reuse it
        reuse = i > 0 and dataset_id is not None

        cfg = create_config(
            method=method,
            exp_name=args.exp_name,
            num_simulations=args.num_simulations,
            T_seg=args.T_seg,
            device=args.device,
            seed=args.seed,
            quick=args.quick,
            smoke=args.smoke,
            dataset_id=dataset_id,
            reuse_dataset=reuse,
        )

        # All non-FNPE methods should share the same cached dataset for an
        # apples-to-apples comparison. Capture the deterministic dataset ID
        # from the first config so later runs can explicitly set reuse_dataset.
        if dataset_id is None:
            dataset_id = cfg.dataset_id
            print(f"[INFO] Shared dataset ID: {dataset_id}")

        try:
            # Run experiment
            exp_results = run_experiment(cfg)
            results[method] = exp_results

        except Exception as e:
            print(f"[ERROR] {method} failed: {e}")
            import traceback

            traceback.print_exc()
            results[method] = {"error": str(e)}

    # Print summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    summary_metrics = {}
    for method, res in results.items():
        if "error" in res:
            print(f"\n{method.upper()}: FAILED - {res['error']}")
            continue

        print(f"\n{method.upper()}:")
        summary = {"method": method}

        metrics = res.get("metrics", {})

        # Training summary
        train = metrics.get("training_summary", {})
        if train:
            if "final_loss" in train:
                print(f"  Final loss: {train['final_loss']:.4f}")
                summary["final_loss"] = train["final_loss"]
            if "train_time_s" in train:
                print(f"  Training time: {train['train_time_s']:.1f}s")
                summary["train_time_s"] = train["train_time_s"]

        # SBC
        if "sbc_check_stats" in metrics:
            sbc = metrics["sbc_check_stats"]
            if "c2st_accuracy" in sbc:
                print(f"  SBC C2ST accuracy: {sbc['c2st_accuracy']:.3f}")
                summary["sbc_c2st"] = sbc["c2st_accuracy"]

        # W2 posterior
        if "w2_posterior_vs_true" in metrics:
            w2 = metrics["w2_posterior_vs_true"]
            print(f"  W2 to ground truth (mean): {w2.get('w2_mean', 'N/A')}")
            print(f"  L2 error (mean): {w2.get('l2_error_mean', 'N/A')}")
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
                summary["coverage_curve_mae"] = w2["coverage_curve_mae"]
            summary["w2_mean"] = w2.get("w2_mean")
            summary["l2_error"] = w2.get("l2_error_mean")
            summary["coverage_90"] = w2.get("coverage_90")
            summary["coverage_50"] = w2.get("coverage_50")

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

    return results


if __name__ == "__main__":
    args = parse_args()
    run_comparison(args)
