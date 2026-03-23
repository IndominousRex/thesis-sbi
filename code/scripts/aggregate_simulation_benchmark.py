#!/usr/bin/env python
"""Aggregate simulation benchmark runs across seeds."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wilcoxon


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate simulation benchmark results.")
    p.add_argument("--experiments-root", default="experiments")
    p.add_argument("--exp-prefix", required=True, help="Prefix shared by benchmark exp_name values.")
    p.add_argument("--output-json", default=None)
    p.add_argument("--output-csv", default=None)
    return p.parse_args()


def _flatten_active_parameters(active_parameters: list[str] | tuple[str, ...]) -> str:
    return ",".join(active_parameters)


def _holm_correction(pairs: list[dict[str, Any]]) -> None:
    sorted_pairs = sorted(
        enumerate(pairs),
        key=lambda item: item[1]["pvalue"] if item[1]["pvalue"] is not None else 1.0,
    )
    m = len(sorted_pairs)
    corrected = [None] * m
    running_max = 0.0
    for rank, (orig_idx, pair) in enumerate(sorted_pairs, start=1):
        pval = pair["pvalue"]
        if pval is None:
            corrected[orig_idx] = None
            continue
        adjusted = min(1.0, pval * (m - rank + 1))
        running_max = max(running_max, adjusted)
        corrected[orig_idx] = running_max
    for idx, corr in enumerate(corrected):
        pairs[idx]["pvalue_holm"] = corr


def _metric_summary(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    ci_low, ci_high = np.quantile(arr, [0.025, 0.975]) if len(arr) > 1 else (arr[0], arr[0])
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "ci95": [float(ci_low), float(ci_high)],
    }


def _slugify(value: str) -> str:
    return value.replace(",", "_").replace(" ", "_")


def _plot_metric_vs_budget(
    aggregate_rows: list[dict[str, Any]],
    params: str,
    metric_key: str,
    metric_label: str,
    output_path: Path,
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    param_rows = [row for row in aggregate_rows if row["params"] == params]
    if not param_rows:
        return False

    tsegs = sorted({int(row["T_seg"]) for row in param_rows})
    if not tsegs:
        return False

    ncols = min(3, len(tsegs))
    nrows = int(np.ceil(len(tsegs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    methods = sorted({row["method"] for row in param_rows})
    for ax_idx, tseg in enumerate(tsegs):
        ax = axes_flat[ax_idx]
        tseg_rows = [row for row in param_rows if int(row["T_seg"]) == tseg]
        for method in methods:
            method_rows = [row for row in tseg_rows if row["method"] == method]
            xs: list[float] = []
            ys: list[float] = []
            for row in sorted(
                method_rows,
                key=lambda r: (
                    float(r.get("requested_budget_steps", r.get("total_budget_steps", 0)) or 0)
                ),
            ):
                summary = row.get(metric_key)
                if not isinstance(summary, dict) or summary.get("mean") is None:
                    continue
                xs.append(float(row["requested_budget_steps"]))
                ys.append(float(summary["mean"]))
            if xs and ys:
                ax.plot(xs, ys, marker="o", label=method.upper())
        ax.set_title(f"T_seg={tseg}")
        ax.set_xlabel("Requested Budget Steps")
        ax.set_ylabel(metric_label)
        ax.grid(True, alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend()

    for ax in axes_flat[len(tsegs) :]:
        ax.axis("off")

    fig.suptitle(f"{metric_label} vs Budget | params={params}")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> None:
    args = parse_args()
    experiments_root = Path(args.experiments_root)
    output_json = Path(args.output_json) if args.output_json else experiments_root / f"{args.exp_prefix}_aggregate.json"
    output_csv = Path(args.output_csv) if args.output_csv else experiments_root / f"{args.exp_prefix}_aggregate.csv"

    rows: list[dict[str, Any]] = []
    for config_path in experiments_root.glob("*/config.json"):
        metrics_path = config_path.with_name("metrics.json")
        if not metrics_path.exists():
            continue
        with config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        if not str(cfg.get("exp_name", "")).startswith(args.exp_prefix):
            continue
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)

        heldout = metrics.get("heldout_test_posterior_vs_true") or metrics.get("w2_posterior_vs_true") or {}
        heldout_ppc = metrics.get("heldout_test_ppc", {}).get("aggregate", {})
        budget = metrics.get("budget_metadata", {})
        training = metrics.get("training_summary", {})
        requested_budget_steps = budget.get("requested_budget_steps")
        effective_budget_steps = budget.get("effective_budget_steps")
        if requested_budget_steps is None:
            requested_budget_steps = budget.get("total_simulation_budget_steps")
        if effective_budget_steps is None:
            effective_budget_steps = budget.get("total_simulation_budget_steps")
        rows.append(
            {
                "exp_dir": str(config_path.parent),
                "method": cfg.get("method"),
                "seed": int(cfg.get("random_seed", cfg.get("sim_seed", 0))),
                "params": _flatten_active_parameters(cfg.get("active_parameters", [])),
                "T_seg": int(cfg.get("T_seg")),
                "num_simulations": int(cfg.get("num_simulations")),
                "requested_budget_steps": requested_budget_steps,
                "effective_budget_steps": effective_budget_steps,
                "total_budget_steps": effective_budget_steps,
                "w2_mean": heldout.get("w2_mean"),
                "l2_error_mean": heldout.get("l2_error_mean"),
                "coverage_90": heldout.get("coverage_90"),
                "coverage_50": heldout.get("coverage_50"),
                "coverage_curve_mae": heldout.get("coverage_curve_mae"),
                "heldout_ppc_rmse_mean": heldout_ppc.get("rmse_mean"),
                "heldout_ppc_w2_mean": heldout_ppc.get("w2_mean"),
                "train_time_s": training.get("train_time_s"),
                "epochs_trained": training.get("epochs_trained"),
                "num_train_steps": training.get("num_train_steps"),
                "sampling_time_mean_s": heldout.get("sampling_time_mean_s"),
            }
        )

    if not rows:
        raise SystemExit(f"No experiments found for prefix '{args.exp_prefix}' in {experiments_root}")

    grouped: dict[tuple[str, str, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        requested_budget_steps = row.get("requested_budget_steps")
        if requested_budget_steps is None:
            requested_budget_steps = row.get("total_budget_steps")
        key = (row["method"], row["params"], int(requested_budget_steps), row["T_seg"])
        grouped.setdefault(key, []).append(row)

    aggregate_rows: list[dict[str, Any]] = []
    for (method, params, requested_budget_steps, tseg), group_rows in sorted(grouped.items()):
        record: dict[str, Any] = {
            "method": method,
            "params": params,
            "T_seg": tseg,
            "num_runs": len(group_rows),
            "requested_budget_steps": requested_budget_steps,
            "effective_budget_steps": group_rows[0]["effective_budget_steps"],
            "total_budget_steps": group_rows[0]["total_budget_steps"],
        }
        for field in [
            "w2_mean",
            "l2_error_mean",
            "coverage_90",
            "coverage_50",
            "coverage_curve_mae",
            "heldout_ppc_rmse_mean",
            "heldout_ppc_w2_mean",
            "train_time_s",
            "sampling_time_mean_s",
        ]:
            values = [float(r[field]) for r in group_rows if r.get(field) is not None]
            if values:
                record[field] = _metric_summary(values)
        aggregate_rows.append(record)

    pairwise_tests: list[dict[str, Any]] = []
    group_keys = sorted(
        {
            (
                row["params"],
                int(row.get("requested_budget_steps", row.get("total_budget_steps", 0)) or 0),
                row["T_seg"],
            )
            for row in rows
        }
    )
    for params, requested_budget_steps, tseg in group_keys:
        candidate_rows = [
            row
            for row in rows
            if row["params"] == params
            and row["T_seg"] == tseg
            and int(row.get("requested_budget_steps", row.get("total_budget_steps", 0)) or 0)
            == requested_budget_steps
        ]
        methods = sorted({row["method"] for row in candidate_rows})
        for method_a, method_b in combinations(methods, 2):
            rows_a = {row["seed"]: row for row in candidate_rows if row["method"] == method_a}
            rows_b = {row["seed"]: row for row in candidate_rows if row["method"] == method_b}
            shared_seeds = sorted(set(rows_a) & set(rows_b))
            for metric_key in ["w2_mean", "heldout_ppc_rmse_mean"]:
                x = [rows_a[s][metric_key] for s in shared_seeds if rows_a[s].get(metric_key) is not None and rows_b[s].get(metric_key) is not None]
                y = [rows_b[s][metric_key] for s in shared_seeds if rows_a[s].get(metric_key) is not None and rows_b[s].get(metric_key) is not None]
                pvalue = None
                stat = None
                if len(x) >= 2 and len(y) >= 2:
                    try:
                        result = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
                        pvalue = float(result.pvalue)
                        stat = float(result.statistic)
                    except ValueError:
                        pvalue = None
                        stat = None
                pairwise_tests.append(
                    {
                        "params": params,
                        "requested_budget_steps": requested_budget_steps,
                        "T_seg": tseg,
                        "metric": metric_key,
                        "method_a": method_a,
                        "method_b": method_b,
                        "num_shared_seeds": len(shared_seeds),
                        "statistic": stat,
                        "pvalue": pvalue,
                    }
                )

    for metric_key in {"w2_mean", "heldout_ppc_rmse_mean"}:
        for params, requested_budget_steps, tseg in group_keys:
            subset = [
                pair
                for pair in pairwise_tests
                if pair["metric"] == metric_key
                and pair["params"] == params
                and pair["requested_budget_steps"] == requested_budget_steps
                and pair["T_seg"] == tseg
            ]
            _holm_correction(subset)

    output = {
        "exp_prefix": args.exp_prefix,
        "num_runs": len(rows),
        "aggregate_rows": aggregate_rows,
        "pairwise_tests": pairwise_tests,
        "raw_rows": rows,
    }

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "params",
        "T_seg",
        "num_runs",
        "total_budget_steps",
        "requested_budget_steps",
        "effective_budget_steps",
        "w2_mean_mean",
        "l2_error_mean_mean",
        "coverage_90_mean",
        "coverage_50_mean",
        "coverage_curve_mae_mean",
        "heldout_ppc_rmse_mean_mean",
        "heldout_ppc_w2_mean_mean",
        "train_time_s_mean",
        "sampling_time_mean_s_mean",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in aggregate_rows:
            writer.writerow(
                {
                    "method": row["method"],
                    "params": row["params"],
                    "T_seg": row["T_seg"],
                    "num_runs": row["num_runs"],
                    "total_budget_steps": row["total_budget_steps"],
                    "requested_budget_steps": row["requested_budget_steps"],
                    "effective_budget_steps": row["effective_budget_steps"],
                    "w2_mean_mean": row.get("w2_mean", {}).get("mean"),
                    "l2_error_mean_mean": row.get("l2_error_mean", {}).get("mean"),
                    "coverage_90_mean": row.get("coverage_90", {}).get("mean"),
                    "coverage_50_mean": row.get("coverage_50", {}).get("mean"),
                    "coverage_curve_mae_mean": row.get("coverage_curve_mae", {}).get("mean"),
                    "heldout_ppc_rmse_mean_mean": row.get("heldout_ppc_rmse_mean", {}).get("mean"),
                    "heldout_ppc_w2_mean_mean": row.get("heldout_ppc_w2_mean", {}).get("mean"),
                    "train_time_s_mean": row.get("train_time_s", {}).get("mean"),
                    "sampling_time_mean_s_mean": row.get("sampling_time_mean_s", {}).get("mean"),
                }
            )

    plot_specs = [
        ("w2_mean", "Held-out W2", "w2"),
        ("l2_error_mean", "Held-out Posterior Mean L2", "l2"),
        ("heldout_ppc_rmse_mean", "Held-out PPC RMSE", "ppc_rmse"),
        ("coverage_90", "Coverage 90%", "coverage90"),
        ("coverage_50", "Coverage 50%", "coverage50"),
        ("train_time_s", "Training Time (s)", "train_time"),
        ("sampling_time_mean_s", "Sampling Time / Case (s)", "sampling_time"),
    ]
    generated_plots: list[str] = []
    for params in sorted({row["params"] for row in aggregate_rows}):
        for metric_key, metric_label, slug in plot_specs:
            output_path = output_json.parent / f"{output_json.stem}_{slug}_{_slugify(params)}.png"
            if _plot_metric_vs_budget(aggregate_rows, params, metric_key, metric_label, output_path):
                generated_plots.append(str(output_path))
    output["generated_plots"] = generated_plots

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {output_json}")
    print(f"Wrote {output_csv}")
    for plot_path in generated_plots:
        print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
