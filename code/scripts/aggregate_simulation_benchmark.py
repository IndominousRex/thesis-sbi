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


def _compute_seed_offsets(
    seeds: list[int],
    x_values: list[float],
    *,
    offset_frac: float = 0.02,
) -> dict[int, float]:
    if not seeds:
        return {}
    if len(seeds) == 1:
        return {seeds[0]: 0.0}

    x_min = float(min(x_values))
    x_max = float(max(x_values))
    x_span = max(x_max - x_min, 1.0)
    max_offset = x_span * offset_frac
    offsets = np.linspace(-max_offset, max_offset, len(seeds))
    return {seed: float(offset) for seed, offset in zip(seeds, offsets)}


def _method_color_map(methods: list[str], plt: Any) -> dict[str, str]:
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not color_cycle:
        color_cycle = ["C0", "C1", "C2", "C3", "C4", "C5"]
    return {method: color_cycle[idx % len(color_cycle)] for idx, method in enumerate(methods)}


def _apply_reference_lines(ax: Any, plot_spec: dict[str, Any]) -> None:
    for value in plot_spec.get("reference_lines", []):
        ax.axhline(
            float(value),
            color="0.45",
            linestyle="--",
            linewidth=1.0,
            alpha=0.9,
            zorder=0,
        )


def _plot_footer_lines(metric_key: str) -> list[str]:
    lines = [
        "Each point is one seed; faint same-color lines connect the same seed. Small horizontal offsets separate seeds at shared x-values."
    ]
    if metric_key == "train_time_s":
        lines.append(
            "Training time reflects the configured training schedule for each run; older experiments may include best-validation stopping."
        )
    return lines


def _plot_seed_traces(
    ax: Any,
    rows: list[dict[str, Any]],
    methods: list[str],
    method_colors: dict[str, str],
    metric_key: str,
    *,
    x_key: str,
    x_offsets: dict[int, float],
    sort_key: Any,
) -> None:
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method and row.get(metric_key) is not None]
        if not method_rows:
            continue

        rows_by_seed: dict[int, list[dict[str, Any]]] = {}
        for row in method_rows:
            rows_by_seed.setdefault(int(row["seed"]), []).append(row)

        label_used = False
        for seed in sorted(rows_by_seed):
            seed_rows = sorted(rows_by_seed[seed], key=sort_key)
            xs = [float(row[x_key]) + x_offsets.get(seed, 0.0) for row in seed_rows]
            ys = [float(row[metric_key]) for row in seed_rows]
            if len(xs) >= 2:
                ax.plot(
                    xs,
                    ys,
                    color=method_colors[method],
                    alpha=0.35,
                    linewidth=1.2,
                    zorder=1,
                )
            ax.scatter(
                xs,
                ys,
                color=method_colors[method],
                alpha=0.9,
                s=45,
                label=method.upper() if not label_used else None,
                zorder=2,
            )
            label_used = True


def _base_plot_specs() -> list[dict[str, Any]]:
    return [
        {
            "metric_key": "w2_mean",
            "metric_label": "Held-out W2",
            "slug": "w2",
            "figure_families": ["budget", "tseg"],
            "category": "posterior_quality",
        },
        {
            "metric_key": "l2_error_mean",
            "metric_label": "Held-out Posterior Mean L2",
            "slug": "l2",
            "figure_families": ["budget"],
            "category": "posterior_quality",
        },
        {
            "metric_key": "heldout_ppc_rmse_mean",
            "metric_label": "Held-out PPC RMSE",
            "slug": "ppc_rmse",
            "figure_families": ["budget", "tseg"],
            "category": "predictive_quality",
        },
        {
            "metric_key": "heldout_ppc_w2_mean",
            "metric_label": "Held-out PPC W2",
            "slug": "ppc_w2",
            "figure_families": ["budget"],
            "category": "predictive_quality",
        },
        {
            "metric_key": "coverage_50",
            "metric_label": "Coverage 50%",
            "slug": "coverage50",
            "figure_families": ["budget"],
            "category": "calibration",
            "reference_lines": [0.5],
        },
        {
            "metric_key": "coverage_90",
            "metric_label": "Coverage 90%",
            "slug": "coverage90",
            "figure_families": ["budget"],
            "category": "calibration",
            "reference_lines": [0.9],
        },
        {
            "metric_key": "coverage_curve_mae",
            "metric_label": "Coverage Curve MAE (lower is better)",
            "slug": "coverage_curve_mae",
            "figure_families": ["budget", "tseg"],
            "category": "calibration",
        },
        {
            "metric_key": "train_time_s",
            "metric_label": "Training Time (s)",
            "slug": "train_time",
            "figure_families": ["budget"],
            "category": "runtime",
        },
        {
            "metric_key": "sampling_time_mean_s",
            "metric_label": "Sampling Time / Case (s)",
            "slug": "sampling_time",
            "figure_families": ["budget"],
            "category": "runtime",
        },
    ]


def _per_parameter_plot_specs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    suffix_specs = [
        ("rmse_phys", "RMSE (physical units)", None),
        ("rmse_norm", "RMSE (normalized)", None),
        ("mae_mean_phys", "MAE (physical units)", None),
        ("mae_mean_norm", "MAE (normalized)", None),
        ("bias_mean_phys", "Bias Mean (physical units)", [0.0]),
        ("bias_mean_norm", "Bias Mean (normalized)", [0.0]),
        ("w1_mean_phys", "W1 Mean (physical units)", None),
        ("w1_mean_norm", "W1 Mean (normalized)", None),
    ]
    available_keys = {key for row in rows for key in row.keys()}
    plot_specs: list[dict[str, Any]] = []
    for suffix, suffix_label, reference_lines in suffix_specs:
        matching_keys = sorted(
            key for key in available_keys if key.endswith(f"_{suffix}")
        )
        for key in matching_keys:
            param_name = key[: -(len(suffix) + 1)]
            spec = {
                "metric_key": key,
                "metric_label": f"{param_name} {suffix_label}",
                "slug": key,
                "figure_families": ["budget"],
                "category": "per_parameter",
            }
            if reference_lines is not None:
                spec["reference_lines"] = reference_lines
            plot_specs.append(spec)
    return plot_specs


def _pareto_plot_specs() -> list[dict[str, Any]]:
    return [
        {
            "slug": "pareto_train_time_vs_w2",
            "x_metric_key": "train_time_s",
            "x_label": "Training Time (s)",
            "y_metric_key": "w2_mean",
            "y_label": "Held-out W2",
            "log_x": True,
        },
        {
            "slug": "pareto_sampling_time_vs_ppc_rmse",
            "x_metric_key": "sampling_time_mean_s",
            "x_label": "Sampling Time / Case (s)",
            "y_metric_key": "heldout_ppc_rmse_mean",
            "y_label": "Held-out PPC RMSE",
            "log_x": True,
        },
    ]


def _build_plot_registry(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    base_specs = _base_plot_specs()
    per_param_specs = _per_parameter_plot_specs(rows)
    all_specs = base_specs + per_param_specs
    return {
        "all": all_specs,
        "budget": [spec for spec in all_specs if "budget" in spec["figure_families"]],
        "tseg": [spec for spec in all_specs if "tseg" in spec["figure_families"]],
        "pareto": _pareto_plot_specs(),
    }


def _plot_metric_vs_budget(
    raw_rows: list[dict[str, Any]],
    params: str,
    plot_spec: dict[str, Any],
    output_path: Path,
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    param_rows = [row for row in raw_rows if row["params"] == params]
    if not param_rows:
        return False

    metric_key = plot_spec["metric_key"]
    metric_label = plot_spec["metric_label"]
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
    method_colors = _method_color_map(methods, plt)
    seeds = sorted({int(row["seed"]) for row in param_rows})
    budget_values = [
        float(row["requested_budget_steps"])
        for row in param_rows
        if row.get(metric_key) is not None and row.get("requested_budget_steps") is not None
    ]
    seed_offsets = _compute_seed_offsets(seeds, budget_values)
    all_xs: list[float] = []
    all_ys: list[float] = []
    for row in param_rows:
        if row.get(metric_key) is None or row.get("requested_budget_steps") is None:
            continue
        all_xs.append(float(row["requested_budget_steps"]) + seed_offsets.get(int(row["seed"]), 0.0))
        all_ys.append(float(row[metric_key]))
    x_limits, y_limits = _compute_axis_limits(all_xs, all_ys)

    for ax_idx, tseg in enumerate(tsegs):
        ax = axes_flat[ax_idx]
        tseg_rows = [row for row in param_rows if int(row["T_seg"]) == tseg]
        _plot_seed_traces(
            ax,
            tseg_rows,
            methods,
            method_colors,
            metric_key,
            x_key="requested_budget_steps",
            x_offsets=seed_offsets,
            sort_key=lambda r: float(
                r.get("requested_budget_steps", r.get("total_budget_steps", 0)) or 0
            ),
        )
        ax.set_title(f"T_seg={tseg}")
        ax.set_xlabel("Requested Budget Steps")
        ax.set_ylabel(metric_label)
        _apply_reference_lines(ax, plot_spec)
        ax.set_xticks(
            sorted(
                {
                    int(row.get("requested_budget_steps", row.get("total_budget_steps", 0)) or 0)
                    for row in tseg_rows
                }
            )
        )
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
    footer_lines = _plot_footer_lines(metric_key)
    fig.text(
        0.5,
        0.02,
        "\n".join(footer_lines),
        ha="center",
        fontsize=9,
    )
    if len(footer_lines) > 1:
        fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    else:
        fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_metric_vs_tseg(
    raw_rows: list[dict[str, Any]],
    params: str,
    plot_spec: dict[str, Any],
    output_path: Path,
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    param_rows = [row for row in raw_rows if row["params"] == params]
    if not param_rows:
        return False

    metric_key = plot_spec["metric_key"]
    metric_label = plot_spec["metric_label"]
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
    method_colors = _method_color_map(methods, plt)
    seeds = sorted({int(row["seed"]) for row in param_rows})
    tseg_values = [float(row["T_seg"]) for row in param_rows if row.get(metric_key) is not None]
    seed_offsets = _compute_seed_offsets(seeds, tseg_values)
    all_xs: list[float] = []
    all_ys: list[float] = []
    for row in param_rows:
        if row.get(metric_key) is None:
            continue
        all_xs.append(float(row["T_seg"]) + seed_offsets.get(int(row["seed"]), 0.0))
        all_ys.append(float(row[metric_key]))
    x_limits, y_limits = _compute_axis_limits(all_xs, all_ys)

    for ax_idx, budget in enumerate(budgets):
        ax = axes_flat[ax_idx]
        budget_rows = [
            row
            for row in param_rows
            if int(row.get("requested_budget_steps", row.get("total_budget_steps", 0)) or 0)
            == budget
        ]
        _plot_seed_traces(
            ax,
            budget_rows,
            methods,
            method_colors,
            metric_key,
            x_key="T_seg",
            x_offsets=seed_offsets,
            sort_key=lambda r: int(r["T_seg"]),
        )
        ax.set_title(f"Budget={budget:.0e}")
        ax.set_xlabel("T_seg")
        ax.set_ylabel(metric_label)
        _apply_reference_lines(ax, plot_spec)
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
    fig.text(
        0.5,
        0.02,
        "\n".join(_plot_footer_lines(metric_key)),
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def _pareto_front_mask(points: list[tuple[float, float]]) -> list[bool]:
    mask = [True] * len(points)
    for idx, (x_i, y_i) in enumerate(points):
        for jdx, (x_j, y_j) in enumerate(points):
            if idx == jdx:
                continue
            dominates = (x_j <= x_i and y_j <= y_i) and (x_j < x_i or y_j < y_i)
            if dominates:
                mask[idx] = False
                break
    return mask


def _plot_pareto_scatter(
    raw_rows: list[dict[str, Any]],
    params: str,
    plot_spec: dict[str, Any],
    output_path: Path,
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception:
        return False

    x_metric_key = plot_spec["x_metric_key"]
    y_metric_key = plot_spec["y_metric_key"]
    param_rows = [
        row
        for row in raw_rows
        if row["params"] == params
        and row.get(x_metric_key) is not None
        and row.get(y_metric_key) is not None
    ]
    if not param_rows:
        return False

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    methods = sorted({row["method"] for row in param_rows})
    method_colors = _method_color_map(methods, plt)
    tsegs = sorted({int(row["T_seg"]) for row in param_rows})
    marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    tseg_markers = {
        tseg: marker_cycle[idx % len(marker_cycle)] for idx, tseg in enumerate(tsegs)
    }

    points = [
        (float(row[x_metric_key]), float(row[y_metric_key])) for row in param_rows
    ]
    pareto_mask = _pareto_front_mask(points)

    for row, is_front in zip(param_rows, pareto_mask):
        x_val = float(row[x_metric_key])
        y_val = float(row[y_metric_key])
        ax.scatter(
            x_val,
            y_val,
            color=method_colors[row["method"]],
            marker=tseg_markers[int(row["T_seg"])],
            s=85 if is_front else 60,
            alpha=0.95 if is_front else 0.65,
            edgecolors="black" if is_front else "none",
            linewidths=0.5 if is_front else 0.0,
            zorder=3 if is_front else 2,
        )
        if is_front:
            label = (
                f"{row['method'].upper()} "
                f"b={int(row['requested_budget_steps']):.0e} "
                f"T={int(row['T_seg'])} s={int(row['seed'])}"
            )
            ax.annotate(
                label,
                (x_val, y_val),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8,
            )

    if plot_spec.get("log_x"):
        ax.set_xscale("log")
    ax.set_xlabel(plot_spec["x_label"])
    ax.set_ylabel(plot_spec["y_label"])
    ax.set_title(f"{plot_spec['y_label']} vs {plot_spec['x_label']} | params={params}")
    ax.grid(True, alpha=0.3)

    method_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=method_colors[method],
            markeredgecolor="none",
            markersize=8,
            label=method.upper(),
        )
        for method in methods
    ]
    tseg_handles = [
        Line2D(
            [0],
            [0],
            marker=tseg_markers[tseg],
            color="black",
            linestyle="None",
            markersize=8,
            label=f"T_seg={tseg}",
        )
        for tseg in tsegs
    ]
    first_legend = ax.legend(handles=method_handles, title="Method", loc="upper right")
    ax.add_artist(first_legend)
    ax.legend(handles=tseg_handles, title="T_seg", loc="lower left")

    fig.text(
        0.5,
        0.02,
        "Each point is one seed-run. Labels are shown only for non-dominated Pareto-front points.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.98))
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
    grouped_images: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}

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
                    (plot_kind, params_slug, example_key, seed), []
                ).append(entry)

    contact_sheets: list[str] = []
    sheet_metadata: list[dict[str, Any]] = []
    for (plot_kind, params_slug, example_key, seed), entries in grouped_images.items():
        entries = sorted(entries, key=lambda item: (item["requested_budget_steps"], item["T_seg"], item["method"]))
        methods = sorted({str(item["method"]) for item in entries})
        combos = sorted(
            {
                (int(item["requested_budget_steps"]), int(item["T_seg"]))
                for item in entries
            }
        )
        if not methods or not combos:
            continue

        entry_lookup = {
            (str(item["method"]), int(item["requested_budget_steps"]), int(item["T_seg"])): item
            for item in entries
        }
        nrows = len(methods)
        ncols = len(combos)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(3.6 * ncols, 3.2 * nrows),
            squeeze=False,
        )
        for row_idx, method in enumerate(methods):
            for col_idx, (budget, tseg) in enumerate(combos):
                ax = axes[row_idx][col_idx]
                entry = entry_lookup.get((method, budget, tseg))
                if entry is None:
                    ax.axis("off")
                    ax.text(0.5, 0.5, "No plot", ha="center", va="center", fontsize=9, color="0.4")
                else:
                    image = plt.imread(entry["copied_to"])
                    ax.imshow(image)
                    ax.axis("off")
                if row_idx == 0:
                    ax.set_title(f"b={budget:.0e}\nT={tseg}", fontsize=9)
                if col_idx == 0:
                    ax.set_ylabel(method.upper(), fontsize=10)
        fig.suptitle(
            f"{plot_kind.upper()} aligned collection | params={params_slug} | {example_key} | seed={seed}",
            fontsize=12,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        sheet_path = (
            output_dir
            / f"{plot_kind}_{params_slug}_{example_key}_seed{seed}_aligned_sheet.png"
        )
        fig.savefig(sheet_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        contact_sheets.append(str(sheet_path))
        sheet_metadata.append(
            {
                "plot_kind": plot_kind,
                "params_slug": params_slug,
                "example_key": example_key,
                "seed": seed,
                "sheet_path": str(sheet_path),
                "methods": methods,
                "columns": [
                    {
                        "requested_budget_steps": int(budget),
                        "T_seg": int(tseg),
                    }
                    for budget, tseg in combos
                ],
            }
        )

    return {
        "copied": copied,
        "contact_sheets": contact_sheets,
        "aligned_sheets": sheet_metadata,
    }


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
        "notes": [
            "Pairwise Wilcoxon tests are descriptive only. With very small shared seed counts they should not be interpreted as strong inferential evidence."
        ],
    }

    _write_csv(output_csv, aggregate_rows)

    plot_registry = _build_plot_registry(rows)
    output["plot_registry"] = plot_registry

    generated_plots: list[str] = []
    for params in sorted({row["params"] for row in rows}):
        params_slug = _slugify(params)
        for plot_spec in plot_registry["budget"]:
            output_path = output_json.parent / f"{output_json.stem}_{plot_spec['slug']}_{params_slug}.png"
            if _plot_metric_vs_budget(rows, params, plot_spec, output_path):
                generated_plots.append(str(output_path))
        for plot_spec in plot_registry["tseg"]:
            tseg_path = (
                output_json.parent
                / f"{output_json.stem}_{plot_spec['slug']}_vs_tseg_{params_slug}.png"
            )
            if _plot_metric_vs_tseg(rows, params, plot_spec, tseg_path):
                generated_plots.append(str(tseg_path))
        for plot_spec in plot_registry["pareto"]:
            pareto_path = (
                output_json.parent
                / f"{output_json.stem}_{plot_spec['slug']}_{params_slug}.png"
            )
            if _plot_pareto_scatter(rows, params, plot_spec, pareto_path):
                generated_plots.append(str(pareto_path))
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
