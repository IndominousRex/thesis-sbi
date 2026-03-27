#!/usr/bin/env python
"""Aggregate simulation benchmark runs across seeds and budgets."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wilcoxon


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate simulation benchmark results.")
    p.add_argument("--experiments-root", default="experiments")
    p.add_argument(
        "--exp-prefix", required=True, help="Prefix shared by benchmark exp_name values."
    )
    p.add_argument("--output-json", default=None)
    p.add_argument("--output-csv", default=None)
    return p.parse_args()


def _flatten_active_parameters(active_parameters: list[str] | tuple[str, ...]) -> str:
    return ",".join(active_parameters)


def _slugify(value: str) -> str:
    return value.replace(",", "_").replace(" ", "_")


def _parse_run_timestamp(exp_dir: Path) -> datetime:
    match = re.search(r"(\d{8}-\d{6})$", exp_dir.name)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%d-%H%M%S")
    return datetime.fromtimestamp(exp_dir.stat().st_mtime)


def _dedupe_key(cfg: dict[str, Any], metrics: dict[str, Any]) -> tuple[Any, ...]:
    budget = metrics.get("budget_metadata", {})
    requested_budget_steps = budget.get("requested_budget_steps")
    if requested_budget_steps is None:
        requested_budget_steps = budget.get("total_simulation_budget_steps")
    return (
        cfg.get("method"),
        _flatten_active_parameters(cfg.get("active_parameters", [])),
        int(cfg.get("T_seg")),
        int(cfg.get("random_seed", cfg.get("sim_seed", 0))),
        requested_budget_steps,
    )


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
    ci_low, ci_high = (
        np.quantile(arr, [0.025, 0.975]) if len(arr) > 1 else (arr[0], arr[0])
    )
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "ci95": [float(ci_low), float(ci_high)],
    }


def _flatten_per_parameter_metrics(
    block: dict[str, Any], suffix: str
) -> dict[str, float | None]:
    flattened: dict[str, float | None] = {}
    for param_name, metrics in block.items():
        if not isinstance(metrics, dict):
            continue
        for metric_name, metric_value in metrics.items():
            key = f"{param_name}_{metric_name}_{suffix}"
            flattened[key] = (
                float(metric_value) if isinstance(metric_value, (int, float)) else None
            )
    return flattened


def _compute_axis_limits(
    xs: list[float],
    ys: list[float],
    *,
    x_pad_frac: float = 0.03,
    y_pad_frac: float = 0.08,
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    if not xs or not ys:
        return None, None

    x_min = float(min(xs))
    x_max = float(max(xs))
    y_min = float(min(ys))
    y_max = float(max(ys))

    x_span = max(x_max - x_min, 1.0)
    y_span = max(y_max - y_min, 1e-9)

    x_limits = (x_min - x_span * x_pad_frac, x_max + x_span * x_pad_frac)
    y_limits = (y_min - y_span * y_pad_frac, y_max + y_span * y_pad_frac)
    return x_limits, y_limits


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
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False
    )
    axes_flat = axes.flatten()

    methods = sorted({row["method"] for row in param_rows})
    all_xs: list[float] = []
    all_ys: list[float] = []
    for row in param_rows:
        summary = row.get(metric_key)
        if not isinstance(summary, dict) or summary.get("mean") is None:
            continue
        all_xs.append(float(row["requested_budget_steps"]))
        all_ys.append(float(summary["mean"]))
    x_limits, y_limits = _compute_axis_limits(all_xs, all_ys)

    for ax_idx, tseg in enumerate(tsegs):
        ax = axes_flat[ax_idx]
        tseg_rows = [row for row in param_rows if int(row["T_seg"]) == tseg]
        for method in methods:
            method_rows = [row for row in tseg_rows if row["method"] == method]
            xs: list[float] = []
            ys: list[float] = []
            for row in sorted(
                method_rows,
                key=lambda r: float(
                    r.get("requested_budget_steps", r.get("total_budget_steps", 0)) or 0
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
        if x_limits is not None:
            ax.set_xlim(*x_limits)
        if y_limits is not None:
            ax.set_ylim(*y_limits)
        ax.grid(True, alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend()

    for ax in axes_flat[len(tsegs) :]:
        ax.axis("off")

    fig.suptitle(f"{metric_label} vs Budget | params={params}")
    if metric_key == "train_time_s":
        fig.text(
            0.5,
            0.02,
            "Training time reflects the configured training schedule for each run; older experiments may include best-validation stopping.",
            ha="center",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    else:
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_metric_vs_tseg(
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

    budgets = sorted(
        {
            int(row.get("requested_budget_steps", row.get("total_budget_steps", 0)) or 0)
            for row in param_rows
        }
    )
    if not budgets:
        return False

    ncols = min(3, len(budgets))
    nrows = int(np.ceil(len(budgets) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False
    )
    axes_flat = axes.flatten()

    methods = sorted({row["method"] for row in param_rows})
    all_xs: list[float] = []
    all_ys: list[float] = []
    for row in param_rows:
        summary = row.get(metric_key)
        if not isinstance(summary, dict) or summary.get("mean") is None:
            continue
        all_xs.append(float(row["T_seg"]))
        all_ys.append(float(summary["mean"]))
    x_limits, y_limits = _compute_axis_limits(all_xs, all_ys)

    for ax_idx, budget in enumerate(budgets):
        ax = axes_flat[ax_idx]
        budget_rows = [
            row
            for row in param_rows
            if int(row.get("requested_budget_steps", row.get("total_budget_steps", 0)) or 0)
            == budget
        ]
        for method in methods:
            method_rows = [row for row in budget_rows if row["method"] == method]
            xs: list[float] = []
            ys: list[float] = []
            for row in sorted(method_rows, key=lambda r: int(r["T_seg"])):
                summary = row.get(metric_key)
                if not isinstance(summary, dict) or summary.get("mean") is None:
                    continue
                xs.append(float(row["T_seg"]))
                ys.append(float(summary["mean"]))
            if xs and ys:
                ax.plot(xs, ys, marker="o", label=method.upper())
        ax.set_title(f"Budget={budget:.0e}")
        ax.set_xlabel("T_seg")
        ax.set_ylabel(metric_label)
        ax.set_xticks(sorted({int(row['T_seg']) for row in budget_rows}))
        if x_limits is not None:
            ax.set_xlim(*x_limits)
        if y_limits is not None:
            ax.set_ylim(*y_limits)
        ax.grid(True, alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend()

    for ax in axes_flat[len(budgets) :]:
        ax.axis("off")

    fig.suptitle(f"{metric_label} vs T_seg | params={params}")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def _collect_plot_gallery(
    completed_runs: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return {"copied": [], "contact_sheets": []}

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    patterns = {
        "posterior": "prior_posterior_grid_ex*.png",
        "ppc": "ppc_timeseries_simulated_test_ex*.png",
    }

    copied: list[dict[str, Any]] = []
    grouped_images: dict[tuple[str, str, str], list[dict[str, Any]]] = {}

    for record in completed_runs:
        cfg = record["cfg"]
        metrics = record["metrics"]
        exp_dir = record["exp_dir"]
        figures_dir = exp_dir / "figures"
        if not figures_dir.exists():
            continue

        params = _flatten_active_parameters(cfg.get("active_parameters", []))
        params_slug = _slugify(params)
        budget = (
            metrics.get("budget_metadata", {}).get("requested_budget_steps")
            or metrics.get("budget_metadata", {}).get("total_simulation_budget_steps")
            or 0
        )
        method = str(cfg.get("method"))
        tseg = int(cfg.get("T_seg"))
        seed = int(cfg.get("random_seed", cfg.get("sim_seed", 0)))

        for plot_kind, pattern in patterns.items():
            for src in sorted(figures_dir.glob(pattern)):
                example_key = src.stem
                dst_dir = output_dir / plot_kind / params_slug
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst_name = (
                    f"{method}_b{int(budget)}_t{tseg}_s{seed}_{src.name}"
                )
                dst = dst_dir / dst_name
                shutil.copy2(src, dst)
                entry = {
                    "plot_kind": plot_kind,
                    "params": params,
                    "params_slug": params_slug,
                    "method": method,
                    "requested_budget_steps": int(budget),
                    "T_seg": tseg,
                    "seed": seed,
                    "source": str(src),
                    "copied_to": str(dst),
                    "example_key": example_key,
                }
                copied.append(entry)
                grouped_images.setdefault(
                    (plot_kind, params_slug, example_key), []
                ).append(entry)

    contact_sheets: list[str] = []
    for (plot_kind, params_slug, example_key), entries in grouped_images.items():
        entries = sorted(
            entries,
            key=lambda item: (
                item["requested_budget_steps"],
                item["T_seg"],
                item["method"],
                item["seed"],
            ),
        )
        ncols = min(4, len(entries))
        nrows = int(np.ceil(len(entries) / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4.6 * ncols, 3.8 * nrows),
            squeeze=False,
        )
        axes_flat = axes.flatten()
        for ax, entry in zip(axes_flat, entries):
            image = plt.imread(entry["copied_to"])
            ax.imshow(image)
            ax.set_title(
                f"{entry['method'].upper()} | b={entry['requested_budget_steps']:.0e}\n"
                f"T={entry['T_seg']} | s={entry['seed']}",
                fontsize=9,
            )
            ax.axis("off")
        for ax in axes_flat[len(entries) :]:
            ax.axis("off")
        fig.suptitle(
            f"{plot_kind.upper()} collection | params={params_slug} | {example_key}",
            fontsize=12,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        sheet_path = (
            output_dir
            / f"{plot_kind}_{params_slug}_{example_key}_contact_sheet.png"
        )
        fig.savefig(sheet_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        contact_sheets.append(str(sheet_path))

    return {"copied": copied, "contact_sheets": contact_sheets}


def _load_completed_runs(experiments_root: Path, exp_prefix: str) -> list[dict[str, Any]]:
    deduped_runs: dict[tuple[Any, ...], dict[str, Any]] = {}

    for config_path in experiments_root.glob("*/config.json"):
        metrics_path = config_path.with_name("metrics.json")
        if not metrics_path.exists():
            continue

        with config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        if not str(cfg.get("exp_name", "")).startswith(exp_prefix):
            continue

        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)

        exp_dir = config_path.parent
        record = {
            "cfg": cfg,
            "metrics": metrics,
            "exp_dir": exp_dir,
            "timestamp": _parse_run_timestamp(exp_dir),
        }
        key = _dedupe_key(cfg, metrics)
        existing = deduped_runs.get(key)
        if (
            existing is None
            or record["timestamp"] > existing["timestamp"]
            or (
                record["timestamp"] == existing["timestamp"]
                and str(exp_dir) > str(existing["exp_dir"])
            )
        ):
            deduped_runs[key] = record

    return list(deduped_runs.values())


def _build_raw_rows(completed_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in completed_runs:
        cfg = record["cfg"]
        metrics = record["metrics"]
        config_path = record["exp_dir"] / "config.json"

        heldout = (
            metrics.get("heldout_test_posterior_vs_true")
            or metrics.get("w2_posterior_vs_true")
            or {}
        )
        heldout_ppc = metrics.get("heldout_test_ppc", {}).get("aggregate", {})
        budget = metrics.get("budget_metadata", {})
        training = metrics.get("training_summary", {})
        requested_budget_steps = budget.get("requested_budget_steps")
        effective_budget_steps = budget.get("effective_budget_steps")
        if requested_budget_steps is None:
            requested_budget_steps = budget.get("total_simulation_budget_steps")
        if effective_budget_steps is None:
            effective_budget_steps = budget.get("total_simulation_budget_steps")

        row = {
            "exp_dir": str(config_path.parent),
            "method": cfg.get("method"),
            "seed": int(cfg.get("random_seed", cfg.get("sim_seed", 0))),
            "params": _flatten_active_parameters(cfg.get("active_parameters", [])),
            "T_seg": int(cfg.get("T_seg")),
            "num_simulations": int(
                cfg.get("fnpe_num_simulations")
                if cfg.get("method") == "fnpe"
                else cfg.get("num_simulations")
            ),
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
            "optimizer_examples_seen": training.get("optimizer_examples_seen"),
            "training_batch_size": training.get("training_batch_size"),
            "num_outer_epochs": training.get("num_outer_epochs"),
            "num_inner_epochs": training.get("num_inner_epochs"),
            "sampling_time_mean_s": heldout.get("sampling_time_mean_s"),
        }
        row.update(
            _flatten_per_parameter_metrics(
                metrics.get("heldout_test_per_parameter_physical", {}), "phys"
            )
        )
        row.update(
            _flatten_per_parameter_metrics(
                metrics.get("heldout_test_per_parameter_normalized", {}), "norm"
            )
        )
        rows.append(row)
    return rows


def _build_aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        requested_budget_steps = row.get("requested_budget_steps")
        if requested_budget_steps is None:
            requested_budget_steps = row.get("total_budget_steps")
        key = (row["method"], row["params"], int(requested_budget_steps), row["T_seg"])
        grouped.setdefault(key, []).append(row)

    aggregate_rows: list[dict[str, Any]] = []
    ignore_fields = {
        "exp_dir",
        "method",
        "seed",
        "params",
        "T_seg",
        "num_simulations",
        "requested_budget_steps",
        "effective_budget_steps",
        "total_budget_steps",
    }
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
        candidate_fields = sorted(
            {field for row in group_rows for field in row.keys() if field not in ignore_fields}
        )
        for field in candidate_fields:
            values = [float(r[field]) for r in group_rows if r.get(field) is not None]
            if values:
                record[field] = _metric_summary(values)
        aggregate_rows.append(record)
    return aggregate_rows


def _build_pairwise_tests(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
            and int(
                row.get("requested_budget_steps", row.get("total_budget_steps", 0)) or 0
            )
            == requested_budget_steps
        ]
        methods = sorted({row["method"] for row in candidate_rows})
        for method_a, method_b in combinations(methods, 2):
            rows_a = {row["seed"]: row for row in candidate_rows if row["method"] == method_a}
            rows_b = {row["seed"]: row for row in candidate_rows if row["method"] == method_b}
            shared_seeds = sorted(set(rows_a) & set(rows_b))
            for metric_key in ["w2_mean", "heldout_ppc_rmse_mean"]:
                x = [
                    rows_a[s][metric_key]
                    for s in shared_seeds
                    if rows_a[s].get(metric_key) is not None and rows_b[s].get(metric_key) is not None
                ]
                y = [
                    rows_b[s][metric_key]
                    for s in shared_seeds
                    if rows_a[s].get(metric_key) is not None and rows_b[s].get(metric_key) is not None
                ]
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
    return pairwise_tests


def _write_csv(output_csv: Path, aggregate_rows: list[dict[str, Any]]) -> None:
    base_fields = [
        "method",
        "params",
        "T_seg",
        "num_runs",
        "total_budget_steps",
        "requested_budget_steps",
        "effective_budget_steps",
    ]
    metric_fields = sorted(
        {
            key
            for row in aggregate_rows
            for key, value in row.items()
            if isinstance(value, dict) and value.get("mean") is not None
        }
    )
    fieldnames = base_fields + [f"{field}_mean" for field in metric_fields]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in aggregate_rows:
            out_row = {field: row.get(field) for field in base_fields}
            for field in metric_fields:
                out_row[f"{field}_mean"] = row.get(field, {}).get("mean")
            writer.writerow(out_row)


def main() -> None:
    args = parse_args()
    experiments_root = Path(args.experiments_root)
    output_json = (
        Path(args.output_json)
        if args.output_json
        else experiments_root / f"{args.exp_prefix}_aggregate.json"
    )
    output_csv = (
        Path(args.output_csv)
        if args.output_csv
        else experiments_root / f"{args.exp_prefix}_aggregate.csv"
    )

    completed_runs = _load_completed_runs(experiments_root, args.exp_prefix)
    if not completed_runs:
        raise SystemExit(
            f"No completed experiments found for prefix '{args.exp_prefix}' in {experiments_root}"
        )

    rows = _build_raw_rows(completed_runs)
    aggregate_rows = _build_aggregate_rows(rows)
    pairwise_tests = _build_pairwise_tests(rows)

    output = {
        "exp_prefix": args.exp_prefix,
        "num_runs": len(rows),
        "aggregate_rows": aggregate_rows,
        "pairwise_tests": pairwise_tests,
        "raw_rows": rows,
    }

    _write_csv(output_csv, aggregate_rows)

    plot_specs = [
        ("w2_mean", "Held-out W2", "w2"),
        ("l2_error_mean", "Held-out Posterior Mean L2", "l2"),
        ("heldout_ppc_rmse_mean", "Held-out PPC RMSE", "ppc_rmse"),
        ("coverage_90", "Coverage 90%", "coverage90"),
        ("coverage_50", "Coverage 50%", "coverage50"),
        ("train_time_s", "Training Time (s)", "train_time"),
        ("sampling_time_mean_s", "Sampling Time / Case (s)", "sampling_time"),
    ]
    per_param_metric_keys = sorted(
        {
            key
            for row in aggregate_rows
            for key in row.keys()
            if key.endswith("_rmse_phys")
            or key.endswith("_rmse_norm")
            or key.endswith("_mae_phys")
            or key.endswith("_mae_norm")
        }
    )
    for metric_key in per_param_metric_keys:
        label = metric_key.replace("_", " ")
        plot_specs.append((metric_key, label, metric_key))

    generated_plots: list[str] = []
    for params in sorted({row["params"] for row in aggregate_rows}):
        for metric_key, metric_label, slug in plot_specs:
            output_path = output_json.parent / f"{output_json.stem}_{slug}_{_slugify(params)}.png"
            if _plot_metric_vs_budget(aggregate_rows, params, metric_key, metric_label, output_path):
                generated_plots.append(str(output_path))
        tseg_w2_path = (
            output_json.parent
            / f"{output_json.stem}_w2_vs_tseg_{_slugify(params)}.png"
        )
        if _plot_metric_vs_tseg(
            aggregate_rows,
            params,
            "w2_mean",
            "Held-out W2",
            tseg_w2_path,
        ):
            generated_plots.append(str(tseg_w2_path))
    output["generated_plots"] = generated_plots

    plot_collection_dir = output_json.parent / f"{args.exp_prefix}_plot_collection"
    plot_collection = _collect_plot_gallery(completed_runs, plot_collection_dir)
    output["plot_collection"] = {
        "root": str(plot_collection_dir),
        **plot_collection,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {output_json}")
    print(f"Wrote {output_csv}")
    print(f"Deduplicated completed runs: {len(completed_runs)}")
    for plot_path in generated_plots:
        print(f"Wrote {plot_path}")
    print(f"Wrote plot collection to {plot_collection_dir}")
    for sheet_path in plot_collection.get("contact_sheets", []):
        print(f"Wrote {sheet_path}")


if __name__ == "__main__":
    main()
