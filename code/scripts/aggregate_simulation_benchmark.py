#!/usr/bin/env python
"""Aggregate simulation benchmark runs across seeds."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path
from statistics import mean, median
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
        rows.append(
            {
                "exp_dir": str(config_path.parent),
                "method": cfg.get("method"),
                "seed": int(cfg.get("random_seed", cfg.get("sim_seed", 0))),
                "params": _flatten_active_parameters(cfg.get("active_parameters", [])),
                "T_seg": int(cfg.get("T_seg")),
                "num_simulations": int(cfg.get("num_simulations")),
                "total_budget_steps": budget.get("total_simulation_budget_steps"),
                "w2_mean": heldout.get("w2_mean"),
                "l2_error_mean": heldout.get("l2_error_mean"),
                "coverage_90": heldout.get("coverage_90"),
                "coverage_50": heldout.get("coverage_50"),
                "coverage_curve_mae": heldout.get("coverage_curve_mae"),
                "heldout_ppc_rmse_mean": heldout_ppc.get("rmse_mean"),
                "heldout_ppc_w2_mean": heldout_ppc.get("w2_mean"),
            }
        )

    if not rows:
        raise SystemExit(f"No experiments found for prefix '{args.exp_prefix}' in {experiments_root}")

    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["method"], row["params"], row["T_seg"])
        grouped.setdefault(key, []).append(row)

    aggregate_rows: list[dict[str, Any]] = []
    for (method, params, tseg), group_rows in sorted(grouped.items()):
        record: dict[str, Any] = {
            "method": method,
            "params": params,
            "T_seg": tseg,
            "num_runs": len(group_rows),
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
        ]:
            values = [float(r[field]) for r in group_rows if r.get(field) is not None]
            if values:
                record[field] = _metric_summary(values)
        aggregate_rows.append(record)

    pairwise_tests: list[dict[str, Any]] = []
    group_keys = sorted({(row["params"], row["T_seg"]) for row in rows})
    for params, tseg in group_keys:
        candidate_rows = [row for row in rows if row["params"] == params and row["T_seg"] == tseg]
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
        for params, tseg in group_keys:
            subset = [
                pair
                for pair in pairwise_tests
                if pair["metric"] == metric_key and pair["params"] == params and pair["T_seg"] == tseg
            ]
            _holm_correction(subset)

    output = {
        "exp_prefix": args.exp_prefix,
        "num_runs": len(rows),
        "aggregate_rows": aggregate_rows,
        "pairwise_tests": pairwise_tests,
        "raw_rows": rows,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "params",
        "T_seg",
        "num_runs",
        "total_budget_steps",
        "w2_mean_mean",
        "l2_error_mean_mean",
        "coverage_90_mean",
        "coverage_50_mean",
        "coverage_curve_mae_mean",
        "heldout_ppc_rmse_mean_mean",
        "heldout_ppc_w2_mean_mean",
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
                    "w2_mean_mean": row.get("w2_mean", {}).get("mean"),
                    "l2_error_mean_mean": row.get("l2_error_mean", {}).get("mean"),
                    "coverage_90_mean": row.get("coverage_90", {}).get("mean"),
                    "coverage_50_mean": row.get("coverage_50", {}).get("mean"),
                    "coverage_curve_mae_mean": row.get("coverage_curve_mae", {}).get("mean"),
                    "heldout_ppc_rmse_mean_mean": row.get("heldout_ppc_rmse_mean", {}).get("mean"),
                    "heldout_ppc_w2_mean_mean": row.get("heldout_ppc_w2_mean", {}).get("mean"),
                }
            )

    print(f"Wrote {output_json}")
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    main()
