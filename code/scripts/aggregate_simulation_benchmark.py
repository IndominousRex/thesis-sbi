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
from scipy.stats import rankdata, wilcoxon


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate simulation benchmark results.")
    p.add_argument("--experiments-root", default="experiments")
    p.add_argument(
        "--exp-prefix",
        required=True,
        help="Prefix shared by benchmark exp_name values.",
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
        "quantile_interval_95_empirical": [float(ci_low), float(ci_high)],
    }


def _require_consistent_value(group_rows: list[dict[str, Any]], field: str) -> Any:
    values = {row.get(field) for row in group_rows}
    if len(values) > 1:
        raise ValueError(
            f"Inconsistent {field} within aggregate group: {sorted(values, key=str)}"
        )
    return next(iter(values)) if values else None


def _budget_to_label(budget: int | float) -> str:
    return f"{float(budget):.0e}"


def _format_budget_axis(ax: Any) -> None:
    from matplotlib.ticker import FuncFormatter

    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: _budget_to_label(x) if x != 0 else "0")
    )


def _metric_direction(plot_spec: dict[str, Any]) -> str:
    return str(plot_spec.get("direction", "lower"))


def _metric_target(plot_spec: dict[str, Any]) -> float:
    reference_lines = plot_spec.get("reference_lines", [])
    if reference_lines:
        return float(reference_lines[0])
    return float(plot_spec.get("target", 0.0))


def _comparison_value(value: float, plot_spec: dict[str, Any]) -> float:
    direction = _metric_direction(plot_spec)
    value = float(value)
    if direction == "lower":
        return value
    if direction == "higher":
        return -value
    if direction == "target":
        return abs(value - _metric_target(plot_spec))
    if direction == "signed":
        return abs(value)
    raise ValueError(f"Unsupported metric direction: {direction}")


def _comparison_delta(
    method_value: float, baseline_value: float, plot_spec: dict[str, Any]
) -> float:
    direction = _metric_direction(plot_spec)
    if direction in {"lower", "higher"}:
        return float(method_value) - float(baseline_value)
    return _comparison_value(method_value, plot_spec) - _comparison_value(
        baseline_value, plot_spec
    )


def _comparison_delta_footer(plot_spec: dict[str, Any], baseline_method: str) -> str:
    direction = _metric_direction(plot_spec)
    baseline_label = baseline_method.upper()
    metric_label = plot_spec["metric_label"]
    if direction == "higher":
        return f"Delta = method - {baseline_label}. Positive values are better for this higher-is-better metric."
    if direction == "target":
        target = _metric_target(plot_spec)
        return (
            f"Delta = |{metric_label} - {target:g}| difference vs {baseline_label}. "
            "Negative values are better (closer to target)."
        )
    if direction == "signed":
        return (
            f"Delta = |{metric_label}| difference vs {baseline_label}. "
            "Negative values are better (closer to zero)."
        )
    return f"Delta = method - {baseline_label}. Negative values are better for this lower-is-better metric."


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
    if not x_values:
        return {seed: 0.0 for seed in seeds}
    if len(seeds) == 1:
        return {seeds[0]: 0.0}

    x_min = float(min(x_values))
    x_max = float(max(x_values))
    x_span = max(x_max - x_min, 1.0)
    max_offset = x_span * offset_frac
    offsets = np.linspace(-max_offset, max_offset, len(seeds))
    return {seed: float(offset) for seed, offset in zip(seeds, offsets)}


# Colorblind-friendly, visually distinct palette for thesis plots.
# Prioritizes discriminability in print and on screen.
_THESIS_PALETTE = [
    "#0077BB",  # strong blue  (NPE)
    "#EE7733",  # orange       (NPSE)
    "#009988",  # teal         (FNPE)
    "#CC3311",  # red          (Simformer)
    "#AA3377",  # magenta
    "#33BBEE",  # cyan
    "#BBBBBB",  # grey
    "#EE3377",  # pink
]

# Thesis layout constants (KOMA-Script scrbook, A4, DIV=13, BCOR=5mm, 12pt)
# Text width ≈ 150 mm ≈ 5.91 in.  All plots should fit within this width.
_THESIS_TEXTWIDTH_IN = 5.91
_THESIS_DPI = 300


def _method_color_map(methods: list[str], plt: Any) -> dict[str, str]:
    return {
        method: _THESIS_PALETTE[idx % len(_THESIS_PALETTE)]
        for idx, method in enumerate(methods)
    }


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
    return ["Solid lines: mean across seeds. Shaded bands: min\u2013max range."]


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
        method_rows = [
            row
            for row in rows
            if row["method"] == method and row.get(metric_key) is not None
        ]
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
    # Primary metrics get individual budget plots; secondary metrics appear
    # only in dashboards, heatmaps, and multi-metric summary figures.
    return [
        {
            "metric_key": "w2_mean",
            "metric_label": "Wasserstein-2 Distance",
            "slug": "w2",
            "figure_families": ["budget"],
            "category": "posterior_quality",
            "direction": "lower",
        },
        {
            "metric_key": "l2_error_mean",
            "metric_label": "Posterior Mean L2 Error",
            "slug": "l2",
            "figure_families": [],
            "category": "posterior_quality",
            "direction": "lower",
        },
        {
            "metric_key": "swd_posterior_vs_true",
            "metric_label": "Sliced Wasserstein Distance",
            "slug": "swd",
            "figure_families": [],
            "category": "posterior_quality",
            "direction": "lower",
        },
        {
            "metric_key": "heldout_ppc_rmse_mean",
            "metric_label": "PPC RMSE (held-out)",
            "slug": "ppc_rmse",
            "figure_families": ["budget"],
            "category": "predictive_quality",
            "direction": "lower",
        },
        {
            "metric_key": "heldout_ppc_w2_mean",
            "metric_label": "PPC W2 (held-out)",
            "slug": "ppc_w2",
            "figure_families": [],
            "category": "predictive_quality",
            "direction": "lower",
        },
        {
            "metric_key": "one_step_rmse",
            "metric_label": "One-Step Prediction RMSE",
            "slug": "one_step_rmse",
            "figure_families": [],
            "category": "predictive_quality",
            "direction": "lower",
        },
        {
            "metric_key": "coverage_50",
            "metric_label": "Coverage 50%",
            "slug": "coverage50",
            "figure_families": [],
            "category": "calibration",
            "reference_lines": [0.5],
            "direction": "target",
        },
        {
            "metric_key": "coverage_90",
            "metric_label": "Coverage 90%",
            "slug": "coverage90",
            "figure_families": [],
            "category": "calibration",
            "reference_lines": [0.9],
            "direction": "target",
        },
        {
            "metric_key": "coverage_curve_mae",
            "metric_label": "Coverage Curve MAE",
            "slug": "coverage_curve_mae",
            "figure_families": ["budget"],
            "category": "calibration",
            "direction": "lower",
        },
        {
            "metric_key": "coverage_50_abs_error",
            "metric_label": "Coverage 50% Abs Error",
            "slug": "coverage50_abs_error",
            "figure_families": [],
            "category": "calibration",
            "direction": "lower",
        },
        {
            "metric_key": "coverage_90_abs_error",
            "metric_label": "Coverage 90% Abs Error",
            "slug": "coverage90_abs_error",
            "figure_families": [],
            "category": "calibration",
            "direction": "lower",
        },
        {
            "metric_key": "c2st_mean",
            "metric_label": "C2ST",
            "slug": "c2st",
            "figure_families": ["budget"],
            "category": "calibration",
            "reference_lines": [0.5],
            "direction": "target",
        },
        {
            "metric_key": "train_time_s",
            "metric_label": "Training Time (s)",
            "slug": "train_time",
            "figure_families": [],
            "category": "runtime",
            "direction": "lower",
            "log_y": True,
        },
        {
            "metric_key": "sampling_time_mean_s",
            "metric_label": "Sampling Time / Case (s)",
            "slug": "sampling_time",
            "figure_families": [],
            "category": "runtime",
            "direction": "lower",
            "log_y": True,
        },
        {
            "metric_key": "real_ppc_rmse_mean",
            "metric_label": "Real-Data PPC RMSE",
            "slug": "real_ppc_rmse",
            "figure_families": ["budget"],
            "category": "predictive_quality",
            "direction": "lower",
        },
        {
            "metric_key": "real_ppc_w2_mean",
            "metric_label": "Real-Data PPC W2",
            "slug": "real_ppc_w2",
            "figure_families": [],
            "category": "predictive_quality",
            "direction": "lower",
        },
    ]


def _per_parameter_plot_specs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    suffix_specs = [
        ("rmse_phys", "RMSE (physical units)", None),
        ("rmse_norm", "RMSE (normalized)", None),
        ("bias_mean_phys", "Bias Mean (physical units)", [0.0]),
        ("bias_mean_norm", "Bias Mean (normalized)", [0.0]),
        ("coverage_90_phys", "Coverage 90% (physical)", [0.9]),
        ("coverage_90_norm", "Coverage 90% (normalized)", [0.9]),
    ]
    available_keys = {key for row in rows for key in row.keys()}
    plot_specs: list[dict[str, Any]] = []
    for suffix, suffix_label, reference_lines in suffix_specs:
        matching_keys = sorted(
            key for key in available_keys if key.endswith(f"_{suffix}")
        )
        for key in matching_keys:
            param_name = key[: -(len(suffix) + 1)]
            if "coverage_90" in suffix:
                direction = "target"
            elif "bias_mean" in suffix:
                direction = "signed"
            else:
                direction = "lower"
            spec = {
                "metric_key": key,
                "metric_label": f"{param_name} {suffix_label}",
                "slug": key,
                "figure_families": ["budget"],
                "category": "per_parameter",
                "direction": direction,
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
    base_by_key = {spec["metric_key"]: spec for spec in base_specs}
    return {
        "all": all_specs,
        # Primary metrics: individual budget plots
        "budget": [spec for spec in all_specs if "budget" in spec["figure_families"]],
        # Pareto efficiency plots
        "pareto": _pareto_plot_specs(),
        # Curated rank heatmaps (key metrics only)
        "heatmap": [
            base_by_key[key]
            for key in [
                "w2_mean",
                "heldout_ppc_rmse_mean",
                "coverage_curve_mae",
                "c2st_mean",
                "train_time_s",
            ]
            if key in base_by_key
        ],
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
        nrows,
        ncols,
        figsize=(_THESIS_TEXTWIDTH_IN, 3.2 * nrows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    methods = sorted({row["method"] for row in param_rows})
    method_colors = _method_color_map(methods, plt)

    # Collect global y-range for shared axes
    global_ys: list[float] = []
    for row in param_rows:
        val = row.get(metric_key)
        if val is not None:
            global_ys.append(float(val))

    if not global_ys:
        plt.close(fig)
        return False

    plotted_any = False
    for ax_idx, tseg in enumerate(tsegs):
        ax = axes_flat[ax_idx]
        tseg_rows = [
            row
            for row in param_rows
            if int(row["T_seg"]) == tseg and row.get(metric_key) is not None
        ]

        for method in methods:
            m_rows = [r for r in tseg_rows if r["method"] == method]
            if not m_rows:
                continue
            # Group by budget
            by_budget: dict[int, list[float]] = {}
            for r in m_rows:
                b = int(
                    r.get("requested_budget_steps", r.get("total_budget_steps", 0)) or 0
                )
                by_budget.setdefault(b, []).append(float(r[metric_key]))
            budgets_sorted = sorted(by_budget.keys())
            means = [float(np.mean(by_budget[b])) for b in budgets_sorted]
            lo = [float(np.min(by_budget[b])) for b in budgets_sorted]
            hi = [float(np.max(by_budget[b])) for b in budgets_sorted]
            ax.plot(
                budgets_sorted,
                means,
                marker="o",
                color=method_colors[method],
                linewidth=1.6,
                markersize=4,
                label=method.upper(),
                zorder=3,
            )
            ax.fill_between(
                budgets_sorted,
                lo,
                hi,
                color=method_colors[method],
                alpha=0.15,
                zorder=1,
            )
            plotted_any = True

        ax.set_title(f"$T_{{seg}}={tseg}$", fontsize=9, fontweight="semibold")
        ax.set_xlabel("Simulation Budget", fontsize=8)
        ax.set_ylabel(metric_label, fontsize=8)
        _apply_reference_lines(ax, plot_spec)
        ax.set_xticks(
            sorted(
                {
                    int(
                        row.get(
                            "requested_budget_steps", row.get("total_budget_steps", 0)
                        )
                        or 0
                    )
                    for row in tseg_rows
                }
            )
        )
        _format_budget_axis(ax)
        if plot_spec.get("log_y") and global_ys and min(global_ys) > 0:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.2, linewidth=0.5)
        ax.tick_params(labelsize=7)
        if ax_idx == 0 and ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7, framealpha=0.7)

    # Shared y-limits
    if global_ys:
        y_lo, y_hi = min(global_ys), max(global_ys)
        y_pad = (y_hi - y_lo) * 0.08 if y_hi > y_lo else 0.1
        for ax in axes_flat[: len(tsegs)]:
            if not (plot_spec.get("log_y") and y_lo > 0):
                ax.set_ylim(y_lo - y_pad, y_hi + y_pad)

    for ax in axes_flat[len(tsegs) :]:
        ax.set_visible(False)

    if not plotted_any:
        plt.close(fig)
        return False

    fig.suptitle(
        f"{metric_label} vs Budget | params={params}",
        fontsize=10,
        fontweight="bold",
    )
    footer_lines = _plot_footer_lines(metric_key)
    fig.text(
        0.5,
        0.01,
        "\n".join(footer_lines),
        ha="center",
        fontsize=7,
        color="0.4",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=_THESIS_DPI, bbox_inches="tight")
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


def _plot_pareto_aggregate_scatter(
    raw_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    params: str,
    plot_spec: dict[str, Any],
    output_path: Path,
) -> bool:
    """Pareto efficiency scatter: aggregate means with error bars, Pareto frontier
    highlighted. No text annotations — colour encodes method, marker encodes T_seg."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception:
        return False

    x_metric_key = plot_spec["x_metric_key"]
    y_metric_key = plot_spec["y_metric_key"]

    summary_rows = [
        row
        for row in aggregate_rows
        if row["params"] == params
        and isinstance(row.get(x_metric_key), dict)
        and isinstance(row.get(y_metric_key), dict)
        and row[x_metric_key].get("mean") is not None
        and row[y_metric_key].get("mean") is not None
    ]
    if not summary_rows:
        return False

    methods = sorted({row["method"] for row in summary_rows})
    method_colors = _method_color_map(methods, plt)
    tsegs = sorted({int(row["T_seg"]) for row in summary_rows})
    marker_cycle = ["o", "s", "^", "D", "P"]
    tseg_markers = {
        tseg: marker_cycle[idx % len(marker_cycle)] for idx, tseg in enumerate(tsegs)
    }

    fig, ax = plt.subplots(figsize=(_THESIS_TEXTWIDTH_IN, _THESIS_TEXTWIDTH_IN * 0.72))

    # --- Compute Pareto mask ---
    summary_points = [
        (float(row[x_metric_key]["mean"]), float(row[y_metric_key]["mean"]))
        for row in summary_rows
    ]
    pareto_mask = _pareto_front_mask(summary_points)

    # --- Draw Pareto frontier step-line (sorted by x) ---
    front_pts = sorted(
        [pt for pt, on_front in zip(summary_points, pareto_mask) if on_front],
        key=lambda p: p[0],
    )
    if len(front_pts) >= 2:
        fx = [p[0] for p in front_pts]
        fy = [p[1] for p in front_pts]
        ax.step(
            fx,
            fy,
            where="post",
            color="0.55",
            linewidth=1.2,
            linestyle="--",
            zorder=1,
            label="_nolegend_",
        )

    # --- Plot aggregate means with error bars ---
    for row, is_front in zip(summary_rows, pareto_mask):
        x_val = float(row[x_metric_key]["mean"])
        y_val = float(row[y_metric_key]["mean"])
        x_err = float(row[x_metric_key].get("std") or 0.0)
        y_err = float(row[y_metric_key].get("std") or 0.0)
        color = method_colors[row["method"]]
        marker = tseg_markers[int(row["T_seg"])]

        ax.errorbar(
            x_val,
            y_val,
            xerr=x_err if x_err > 0 else None,
            yerr=y_err if y_err > 0 else None,
            fmt="none",
            ecolor=color,
            elinewidth=0.8,
            capsize=2.5,
            alpha=0.55,
            zorder=2,
        )
        ax.scatter(
            x_val,
            y_val,
            color=color,
            marker=marker,
            s=90 if is_front else 55,
            edgecolors="black" if is_front else "none",
            linewidths=0.8,
            alpha=1.0,
            zorder=3,
        )

    if plot_spec.get("log_x"):
        ax.set_xscale("log")

    ax.set_xlabel(plot_spec["x_label"], fontsize=9)
    ax.set_ylabel(plot_spec["y_label"], fontsize=9)
    ax.set_title(
        f"{plot_spec['y_label']} vs {plot_spec['x_label']}",
        fontsize=10,
        pad=6,
    )
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.tick_params(labelsize=8)

    # --- Single legend outside (below), method colours + T_seg markers ---
    method_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=method_colors[m],
            markeredgecolor="none",
            markersize=7,
            label=m.upper(),
        )
        for m in methods
    ]
    tseg_handles = [
        Line2D(
            [0],
            [0],
            marker=tseg_markers[t],
            color="0.3",
            linestyle="None",
            markersize=7,
            label=f"$T_{{\\mathrm{{seg}}}}$={t}",
        )
        for t in tsegs
    ]
    # Separator handle
    sep = Line2D([0], [0], color="none", label=" ")
    # Pareto front indicator
    pareto_handle = Line2D(
        [0],
        [0],
        color="0.55",
        linestyle="--",
        linewidth=1.2,
        label="Pareto front",
    )

    all_handles = method_handles + [sep] + tseg_handles + [sep, pareto_handle]
    fig.legend(
        handles=all_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=len(methods) + len(tsegs) + 2,
        fontsize=8,
        frameon=True,
        borderpad=0.5,
    )

    fig.tight_layout(rect=(0, 0.10, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=_THESIS_DPI, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_metric_rank_heatmap(
    aggregate_rows: list[dict[str, Any]],
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

    metric_key = plot_spec["metric_key"]
    param_rows = [
        row
        for row in aggregate_rows
        if row["params"] == params
        and isinstance(row.get(metric_key), dict)
        and row[metric_key].get("mean") is not None
    ]
    if not param_rows:
        return False

    methods = sorted({str(row["method"]) for row in param_rows})
    combos = sorted(
        {
            (
                int(
                    row.get("requested_budget_steps", row.get("total_budget_steps", 0))
                    or 0
                ),
                int(row["T_seg"]),
            )
            for row in param_rows
        }
    )
    if not methods or not combos:
        return False

    heat = np.full((len(methods), len(combos)), np.nan, dtype=np.float64)
    method_to_idx = {method: idx for idx, method in enumerate(methods)}
    combo_to_idx = {combo: idx for idx, combo in enumerate(combos)}
    for combo in combos:
        combo_rows = [
            row
            for row in param_rows
            if (
                int(
                    row.get("requested_budget_steps", row.get("total_budget_steps", 0))
                    or 0
                ),
                int(row["T_seg"]),
            )
            == combo
        ]
        values = np.asarray(
            [
                _comparison_value(float(row[metric_key]["mean"]), plot_spec)
                for row in combo_rows
            ],
            dtype=np.float64,
        )
        ranks = rankdata(values, method="average")
        for row, rank in zip(combo_rows, ranks):
            heat[method_to_idx[str(row["method"])], combo_to_idx[combo]] = float(rank)

    fig_width = max(8.0, 1.2 * len(combos))
    fig, ax = plt.subplots(figsize=(fig_width, 3.8 + 0.45 * len(methods)))
    im = ax.imshow(
        heat, aspect="auto", cmap="viridis_r", vmin=1, vmax=max(len(methods), 1)
    )
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([method.upper() for method in methods])
    ax.set_xticks(range(len(combos)))
    ax.set_xticklabels(
        [f"b={_budget_to_label(budget)}\nT={tseg}" for budget, tseg in combos],
        rotation=45,
        ha="right",
    )
    ax.set_title(f"Method rank heatmap | {plot_spec['metric_label']} | params={params}")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Rank (1 = best)")

    for row_idx in range(len(methods)):
        for col_idx in range(len(combos)):
            val = heat[row_idx, col_idx]
            if not np.isnan(val):
                ax.text(
                    col_idx,
                    row_idx,
                    f"{val:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


# =====================================================================
# NEW ANALYTICAL PLOT FUNCTIONS
# =====================================================================


def _plot_method_overview_dashboard(
    aggregate_rows: list[dict[str, Any]],
    params: str,
    output_path: Path,
) -> bool:
    """Multi-panel dashboard: 6 key metrics vs simulation budget in one thesis figure."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    # Key metrics for the overview — covers posterior quality, predictive
    # quality, calibration, and computational cost.
    dashboard_metrics = [
        ("w2_mean", "Wasserstein-2 Distance", False, None),
        ("heldout_ppc_rmse_mean", "PPC RMSE (held-out)", False, None),
        ("coverage_curve_mae", "Coverage Curve MAE", False, None),
        ("c2st_mean", "C2ST", False, [0.5]),
        ("train_time_s", "Training Time (s)", True, None),
        ("one_step_rmse", "One-Step Prediction RMSE", False, None),
    ]

    param_rows = [row for row in aggregate_rows if row["params"] == params]
    if not param_rows:
        return False

    methods = sorted({row["method"] for row in param_rows})
    method_colors = _method_color_map(methods, plt)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes_flat = axes.flatten()
    plotted_any = False

    for idx, (metric_key, label, log_y, ref_lines) in enumerate(dashboard_metrics):
        ax = axes_flat[idx]
        has_data = False
        for method in methods:
            method_rows = sorted(
                [
                    r
                    for r in param_rows
                    if r["method"] == method
                    and isinstance(r.get(metric_key), dict)
                    and r[metric_key].get("mean") is not None
                ],
                key=lambda r: int(r.get("requested_budget_steps") or 0),
            )
            if not method_rows:
                continue
            has_data = True
            budgets = [int(r["requested_budget_steps"]) for r in method_rows]
            means = [float(r[metric_key]["mean"]) for r in method_rows]
            stds = [float(r[metric_key]["std"]) for r in method_rows]
            ax.plot(
                budgets,
                means,
                marker="o",
                label=method.upper(),
                color=method_colors[method],
                linewidth=1.8,
                markersize=5,
                zorder=3,
            )
            ax.fill_between(
                budgets,
                [m - s for m, s in zip(means, stds)],
                [m + s for m, s in zip(means, stds)],
                color=method_colors[method],
                alpha=0.12,
                zorder=1,
            )
        if ref_lines:
            for val in ref_lines:
                ax.axhline(val, color="0.45", linestyle="--", linewidth=1.0, alpha=0.7)
        if log_y and has_data:
            ax.set_yscale("log")
        ax.set_xlabel("Simulation Budget", fontsize=9)
        ax.set_ylabel(label, fontsize=9)
        ax.set_title(label, fontsize=10, fontweight="semibold")
        ax.grid(True, alpha=0.2, linewidth=0.5)
        _format_budget_axis(ax)
        ax.tick_params(labelsize=8)
        if has_data:
            plotted_any = True
        if idx == 0 and has_data:
            ax.legend(fontsize=8, framealpha=0.7)

    if not plotted_any:
        plt.close(fig)
        return False

    fig.suptitle(
        f"Method Comparison Overview | params={params}",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    fig.text(
        0.5,
        0.01,
        "Solid lines: aggregate mean across seeds. Shaded bands: \u00b11 std.",
        ha="center",
        fontsize=8,
        color="0.4",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_radar_chart(
    aggregate_rows: list[dict[str, Any]],
    params: str,
    output_path: Path,
) -> bool:
    """Radar / spider chart comparing methods across normalized metrics at each budget."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    radar_metrics = [
        ("w2_mean", "W2", "lower"),
        ("heldout_ppc_rmse_mean", "PPC RMSE", "lower"),
        ("coverage_curve_mae", "Cov. MAE", "lower"),
        ("c2st_mean", "C2ST", "target_0.5"),
        ("train_time_s", "Train Time", "lower"),
        ("one_step_rmse", "1-Step RMSE", "lower"),
    ]

    param_rows = [row for row in aggregate_rows if row["params"] == params]
    if not param_rows:
        return False
    methods = sorted({row["method"] for row in param_rows})
    budgets = sorted({int(row["requested_budget_steps"]) for row in param_rows})
    method_colors = _method_color_map(methods, plt)

    if not budgets:
        return False

    ncols = min(3, len(budgets))
    nrows = int(np.ceil(len(budgets) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.5 * ncols, 5.0 * nrows),
        subplot_kw={"projection": "polar"},
    )
    if isinstance(axes, np.ndarray):
        axes = axes.reshape(nrows, ncols) if axes.ndim != 2 else axes
    else:
        axes = np.array([[axes]])
    axes_flat = axes.flatten()

    num_metrics = len(radar_metrics)
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]
    plotted_any = False

    for bidx, budget in enumerate(budgets):
        ax = axes_flat[bidx]
        budget_rows = [
            r for r in param_rows if int(r["requested_budget_steps"]) == budget
        ]

        raw_values: dict[str, list[float | None]] = {}
        for method in methods:
            vals: list[float | None] = []
            method_row = next((r for r in budget_rows if r["method"] == method), None)
            for metric_key, _, direction in radar_metrics:
                if method_row is None or not isinstance(
                    method_row.get(metric_key), dict
                ):
                    vals.append(None)
                    continue
                v = method_row[metric_key].get("mean")
                if v is None:
                    vals.append(None)
                    continue
                if direction == "target_0.5":
                    v = abs(float(v) - 0.5)
                vals.append(float(v))
            raw_values[method] = vals

        all_vals_per_metric: list[list[float]] = []
        for midx in range(num_metrics):
            metric_vals = [
                raw_values[m][midx] for m in methods if raw_values[m][midx] is not None
            ]
            all_vals_per_metric.append(metric_vals)

        if all(len(v) == 0 for v in all_vals_per_metric):
            ax.set_visible(False)
            continue

        normalized: dict[str, list[float]] = {}
        for method in methods:
            norm_vals: list[float] = []
            for midx in range(num_metrics):
                v = raw_values[method][midx]
                metric_vals = all_vals_per_metric[midx]
                if v is None or len(metric_vals) < 2:
                    norm_vals.append(0.5)
                    continue
                vmin, vmax = min(metric_vals), max(metric_vals)
                span = vmax - vmin
                if span < 1e-12:
                    norm_vals.append(0.5)
                else:
                    norm_vals.append(1.0 - (float(v) - vmin) / span)
            normalized[method] = norm_vals

        for method in methods:
            values = normalized[method] + normalized[method][:1]
            ax.fill(angles, values, alpha=0.1, color=method_colors[method])
            ax.plot(
                angles,
                values,
                linewidth=1.6,
                label=method.upper(),
                color=method_colors[method],
            )
            plotted_any = True

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([label for _, label, _ in radar_metrics], fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["", "", "", "Best"], fontsize=7, color="0.5")
        ax.set_title(
            f"Budget = {_budget_to_label(budget)}",
            fontsize=10,
            fontweight="semibold",
            pad=15,
        )
        if bidx == 0:
            ax.legend(
                loc="upper right",
                bbox_to_anchor=(1.3, 1.15),
                fontsize=8,
                framealpha=0.7,
            )

    for ax in axes_flat[len(budgets) :]:
        ax.set_visible(False)

    if not plotted_any:
        plt.close(fig)
        return False

    fig.suptitle(
        f"Method Profile Comparison | params={params}",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        0.01,
        "All metrics normalized: 1 = best (outermost), 0 = worst (center). Lower-is-better metrics are inverted.",
        ha="center",
        fontsize=8,
        color="0.4",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_calibration_panel(
    aggregate_rows: list[dict[str, Any]],
    params: str,
    output_path: Path,
) -> bool:
    """Combined calibration diagnostic: coverage 50/90 bars, coverage MAE, C2ST."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    param_rows = [row for row in aggregate_rows if row["params"] == params]
    if not param_rows:
        return False

    methods = sorted({row["method"] for row in param_rows})
    budgets = sorted({int(row["requested_budget_steps"]) for row in param_rows})
    method_colors = _method_color_map(methods, plt)

    fig, axes = plt.subplots(1, 3, figsize=(_THESIS_TEXTWIDTH_IN, 3.5))

    # Panel 1: coverage 90% — one bar per (method, budget), averaged over T_seg values
    ax = axes.flat[0]
    _cov90_metric = "coverage_90"
    _cov90_target = 0.9
    n_methods = len(methods)
    bar_width = 0.8 / max(n_methods, 1)
    has_data = False
    for midx, method in enumerate(methods):
        means: list[float] = []
        stds: list[float] = []
        valid_budgets: list[int] = []
        for budget in budgets:
            brows = [
                r
                for r in param_rows
                if r["method"] == method
                and int(r.get("requested_budget_steps") or 0) == budget
                and isinstance(r.get(_cov90_metric), dict)
                and r[_cov90_metric].get("mean") is not None
            ]
            if not brows:
                continue
            means.append(
                float(np.mean([float(r[_cov90_metric]["mean"]) for r in brows]))
            )
            stds.append(
                float(np.mean([float(r[_cov90_metric].get("std", 0.0)) for r in brows]))
            )
            valid_budgets.append(budget)
        if not valid_budgets:
            continue
        has_data = True
        x_pos = np.array([budgets.index(b) for b in valid_budgets])
        ax.bar(
            x_pos + midx * bar_width,
            means,
            bar_width,
            yerr=stds,
            label=method.upper(),
            color=method_colors[method],
            alpha=0.85,
            capsize=2,
            error_kw={"linewidth": 0.8},
        )
    ax.axhline(
        _cov90_target,
        color="red",
        linestyle="--",
        linewidth=1.0,
        alpha=0.7,
        label=f"Ideal = {_cov90_target}",
    )
    if has_data:
        ax.set_xticks(np.arange(len(budgets)) + bar_width * (n_methods - 1) / 2)
        ax.set_xticklabels(
            [_budget_to_label(b) for b in budgets], fontsize=8, rotation=30
        )
    ax.set_ylabel("Coverage 90%", fontsize=9)
    ax.set_title("Coverage 90%", fontsize=10, fontweight="semibold")
    ax.grid(True, axis="y", alpha=0.2, linewidth=0.5)
    ax.tick_params(labelsize=8)

    # Panel 2: coverage curve MAE
    ax = axes.flat[1]
    for method in methods:
        method_rows = sorted(
            [
                r
                for r in param_rows
                if r["method"] == method
                and isinstance(r.get("coverage_curve_mae"), dict)
                and r["coverage_curve_mae"].get("mean") is not None
            ],
            key=lambda r: int(r.get("requested_budget_steps") or 0),
        )
        if not method_rows:
            continue
        budgets_m = [int(r["requested_budget_steps"]) for r in method_rows]
        means = [float(r["coverage_curve_mae"]["mean"]) for r in method_rows]
        stds = [float(r["coverage_curve_mae"]["std"]) for r in method_rows]
        ax.plot(
            budgets_m,
            means,
            marker="o",
            color=method_colors[method],
            linewidth=1.6,
            markersize=5,
            label=method.upper(),
        )
        ax.fill_between(
            budgets_m,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            color=method_colors[method],
            alpha=0.12,
        )
    ax.set_xlabel("Simulation Budget", fontsize=9)
    ax.set_ylabel("Coverage Curve MAE", fontsize=9)
    ax.set_title("Coverage Curve MAE", fontsize=10, fontweight="semibold")
    ax.grid(True, alpha=0.2, linewidth=0.5)
    _format_budget_axis(ax)
    ax.tick_params(labelsize=8)

    # Panel 3: C2ST
    ax = axes.flat[2]
    for method in methods:
        method_rows = sorted(
            [
                r
                for r in param_rows
                if r["method"] == method
                and isinstance(r.get("c2st_mean"), dict)
                and r["c2st_mean"].get("mean") is not None
            ],
            key=lambda r: int(r.get("requested_budget_steps") or 0),
        )
        if not method_rows:
            continue
        budgets_m = [int(r["requested_budget_steps"]) for r in method_rows]
        means = [float(r["c2st_mean"]["mean"]) for r in method_rows]
        stds = [float(r["c2st_mean"]["std"]) for r in method_rows]
        ax.plot(
            budgets_m,
            means,
            marker="o",
            color=method_colors[method],
            linewidth=1.6,
            markersize=5,
            label=method.upper(),
        )
        ax.fill_between(
            budgets_m,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            color=method_colors[method],
            alpha=0.12,
        )
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Simulation Budget", fontsize=9)
    ax.set_ylabel("C2ST Mean", fontsize=9)
    ax.set_title("C2ST", fontsize=10, fontweight="semibold")
    ax.grid(True, alpha=0.2, linewidth=0.5)
    _format_budget_axis(ax)
    ax.tick_params(labelsize=8)

    # Collect all unique handles/labels from all panels for a single outside legend
    _handles, _labels = [], []
    for _ax in axes.flat:
        for _h, _l in zip(*_ax.get_legend_handles_labels()):
            if _l not in _labels:
                _handles.append(_h)
                _labels.append(_l)
    fig.legend(
        _handles,
        _labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=len(_labels),
        fontsize=7,
        framealpha=0.8,
        borderaxespad=0.0,
    )
    fig.suptitle(
        f"Calibration & Posterior Quality Diagnostics | params={params}",
        fontsize=10,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.04,
        "Coverage bars show mean \u00b11 std across seeds. Dashed lines indicate ideal targets.",
        ha="center",
        fontsize=7,
        color="0.4",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=_THESIS_DPI, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_per_parameter_bars(
    aggregate_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    params: str,
    output_path: Path,
) -> bool:
    """Grouped bar chart: per-parameter RMSE, bias, and coverage across methods."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    param_rows = [row for row in aggregate_rows if row["params"] == params]
    if not param_rows:
        return False

    methods = sorted({row["method"] for row in param_rows})
    method_colors = _method_color_map(methods, plt)

    # Discover parameter names from raw rows (using normalized RMSE as the discovery key)
    raw_param_rows = [row for row in raw_rows if row["params"] == params]
    param_names = sorted(
        {
            key.rsplit("_rmse_norm", 1)[0]
            for row in raw_param_rows
            for key in row.keys()
            if key.endswith("_rmse_norm") and row[key] is not None
        }
    )
    if not param_names:
        return False

    # Use the largest budget for the bar comparison
    budgets = sorted({int(r["requested_budget_steps"]) for r in param_rows})
    if not budgets:
        return False
    max_budget = budgets[-1]
    best_rows = [
        r for r in param_rows if int(r["requested_budget_steps"]) == max_budget
    ]

    metrics_to_plot = [
        ("rmse_norm", "RMSE (normalised)", []),
        ("bias_mean_norm", "Bias (normalised)", [0.0]),
        ("coverage_90_phys", "Coverage 90%", [0.9]),
    ]

    # Joint W2 per method (not per-parameter — only joint value exists in data)
    w2_by_method: dict[str, float] = {}
    w2_std_by_method: dict[str, float] = {}
    for _method in methods:
        _m_rows = [r for r in best_rows if r["method"] == _method]
        _vals = [
            float(r["w2_mean"]["mean"])
            for r in _m_rows
            if isinstance(r.get("w2_mean"), dict)
            and r["w2_mean"].get("mean") is not None
        ]
        _stds = [
            float(r["w2_mean"]["std"])
            for r in _m_rows
            if isinstance(r.get("w2_mean"), dict)
            and r["w2_mean"].get("std") is not None
        ]
        w2_by_method[_method] = float(np.mean(_vals)) if _vals else float("nan")
        w2_std_by_method[_method] = float(np.mean(_stds)) if _stds else 0.0

    fig, axes = plt.subplots(1, 4, figsize=(_THESIS_TEXTWIDTH_IN * 1.35, 3.8))
    n_params = len(param_names)
    n_methods = len(methods)
    bar_width = 0.75 / max(n_methods, 1)
    plotted_any = False

    for panel_idx, (suffix, panel_label, ref_vals) in enumerate(metrics_to_plot):
        ax = axes[panel_idx]
        for midx, method in enumerate(methods):
            method_row = next((r for r in best_rows if r["method"] == method), None)
            if method_row is None:
                continue
            vals: list[float] = []
            errs: list[float] = []
            for pname in param_names:
                key = f"{pname}_{suffix}"
                entry = method_row.get(key)
                if isinstance(entry, dict) and entry.get("mean") is not None:
                    vals.append(float(entry["mean"]))
                    errs.append(float(entry.get("std", 0.0)))
                else:
                    vals.append(0.0)
                    errs.append(0.0)
            x_pos = np.arange(n_params)
            ax.bar(
                x_pos + midx * bar_width,
                vals,
                bar_width,
                yerr=errs,
                label=method.upper() if panel_idx == 0 else None,
                color=method_colors[method],
                alpha=0.85,
                capsize=2,
                error_kw={"linewidth": 0.8},
            )
            plotted_any = True
        ax.set_xticks(np.arange(n_params) + bar_width * (n_methods - 1) / 2)
        ax.set_xticklabels(param_names, fontsize=9, fontweight="semibold")
        for val in ref_vals:
            ax.axhline(val, color="red", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_ylabel(panel_label, fontsize=9)
        ax.set_title(panel_label, fontsize=10, fontweight="semibold")
        ax.grid(True, axis="y", alpha=0.2, linewidth=0.5)
        ax.tick_params(labelsize=8)

    # Panel 4: joint W2 per method (no per-param breakdown available)
    ax = axes[3]
    for midx, method in enumerate(methods):
        w2v = w2_by_method.get(method, float("nan"))
        w2s = w2_std_by_method.get(method, 0.0)
        if np.isnan(w2v):
            continue
        ax.bar(
            midx * bar_width,
            w2v,
            bar_width,
            yerr=w2s,
            color=method_colors[method],
            alpha=0.85,
            capsize=2,
            error_kw={"linewidth": 0.8},
        )
        plotted_any = True
    ax.set_xticks(np.arange(n_methods) * bar_width)
    ax.set_xticklabels(
        [m.upper() for m in methods], fontsize=7, rotation=30, ha="right"
    )
    ax.set_ylabel("W2 (joint, physical)", fontsize=9)
    ax.set_title("W2\n(joint, physical)", fontsize=10, fontweight="semibold")
    ax.grid(True, axis="y", alpha=0.2, linewidth=0.5)
    ax.tick_params(labelsize=8)

    if not plotted_any:
        plt.close(fig)
        return False

    # Legend outside all subplots, below the figure
    _handles, _labels = axes[0].get_legend_handles_labels()
    fig.legend(
        _handles,
        _labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=len(_labels),
        fontsize=7,
        framealpha=0.8,
        borderaxespad=0.0,
    )
    fig.suptitle(
        f"Per-Parameter Comparison at Budget={_budget_to_label(max_budget)} | params={params}",
        fontsize=10,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.05,
        f"RMSE & Bias normalised by prior range (scale-free). W2 is joint physical-space Wasserstein-2. Budget={_budget_to_label(max_budget)}. Error bars: \u00b11 std.",
        ha="center",
        fontsize=7,
        color="0.4",
    )
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=_THESIS_DPI, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_pairwise_win_matrix(
    pairwise_tests: list[dict[str, Any]],
    params: str,
    output_path: Path,
) -> bool:
    """Win/loss matrix heatmap aggregated across all metrics and budget combinations."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    relevant_tests = [t for t in pairwise_tests if t["params"] == params]
    if not relevant_tests:
        return False

    methods = sorted(
        {t["method_a"] for t in relevant_tests}
        | {t["method_b"] for t in relevant_tests}
    )
    if len(methods) < 2:
        return False

    n = len(methods)
    method_to_idx = {m: i for i, m in enumerate(methods)}
    win_matrix = np.zeros((n, n), dtype=float)

    for test in relevant_tests:
        ia = method_to_idx[test["method_a"]]
        ib = method_to_idx[test["method_b"]]
        win_matrix[ia, ib] += test.get("wins_method_a", 0)
        win_matrix[ib, ia] += test.get("wins_method_b", 0)

    # Compute fractions
    frac_matrix = np.full((n, n), np.nan, dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            total = win_matrix[i, j] + win_matrix[j, i]
            frac_matrix[i, j] = win_matrix[i, j] / total if total > 0 else 0.5

    fig, ax = plt.subplots(figsize=(6 + 0.8 * n, 5 + 0.6 * n))
    display = np.where(np.eye(n, dtype=bool), np.nan, frac_matrix)
    im = ax.imshow(display, cmap="RdYlGn", vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_xticklabels([m.upper() for m in methods], fontsize=10)
    ax.set_yticks(range(n))
    ax.set_yticklabels([m.upper() for m in methods], fontsize=10)
    ax.set_xlabel("Opponent", fontsize=10)
    ax.set_ylabel("Method", fontsize=10)

    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(
                    j, i, "\u2014", ha="center", va="center", fontsize=10, color="0.5"
                )
            else:
                val = frac_matrix[i, j]
                total = int(win_matrix[i, j] + win_matrix[j, i])
                text = f"{val:.0%}\n({int(win_matrix[i, j])}/{total})"
                color = "white" if val > 0.7 or val < 0.3 else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Win Rate (row vs column)", fontsize=9)
    ax.set_title(
        f"Pairwise Win Rate Matrix | params={params}",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "Win rates aggregated across all metrics and budget/T_seg combinations. Read rows: how often row-method beats column-method.",
        ha="center",
        fontsize=8,
        color="0.4",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_convergence_profile(
    aggregate_rows: list[dict[str, Any]],
    params: str,
    output_path: Path,
) -> bool:
    """Normalized convergence curves showing sample efficiency per method."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    convergence_metrics = [
        ("w2_mean", "Wasserstein-2", "lower"),
        ("heldout_ppc_rmse_mean", "PPC RMSE", "lower"),
        ("coverage_curve_mae", "Coverage MAE", "lower"),
    ]

    param_rows = [row for row in aggregate_rows if row["params"] == params]
    if not param_rows:
        return False
    methods = sorted({row["method"] for row in param_rows})
    method_colors = _method_color_map(methods, plt)

    fig, axes = plt.subplots(
        1, len(convergence_metrics), figsize=(6 * len(convergence_metrics), 5)
    )
    if len(convergence_metrics) == 1:
        axes = [axes]
    plotted_any = False

    for midx, (metric_key, label, _direction) in enumerate(convergence_metrics):
        ax = axes[midx]
        for method in methods:
            method_rows = sorted(
                [
                    r
                    for r in param_rows
                    if r["method"] == method
                    and isinstance(r.get(metric_key), dict)
                    and r[metric_key].get("mean") is not None
                ],
                key=lambda r: int(r.get("requested_budget_steps") or 0),
            )
            if len(method_rows) < 2:
                continue
            budgets = [int(r["requested_budget_steps"]) for r in method_rows]
            means = [float(r[metric_key]["mean"]) for r in method_rows]
            baseline_val = means[0]
            if abs(baseline_val) < 1e-12:
                continue
            normalized = [v / baseline_val for v in means]
            ax.plot(
                budgets,
                normalized,
                marker="o",
                color=method_colors[method],
                linewidth=1.8,
                markersize=5,
                label=method.upper(),
            )
            plotted_any = True

        ax.axhline(1.0, color="0.5", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.set_xlabel("Simulation Budget", fontsize=9)
        ax.set_ylabel(f"Relative {label} (1.0 = first budget)", fontsize=9)
        ax.set_title(f"{label} Convergence", fontsize=10, fontweight="semibold")
        ax.grid(True, alpha=0.2, linewidth=0.5)
        _format_budget_axis(ax)
        ax.tick_params(labelsize=8)
        if midx == 0:
            ax.legend(fontsize=8, framealpha=0.7)

    if not plotted_any:
        plt.close(fig)
        return False

    fig.suptitle(
        f"Sample Efficiency / Convergence Profile | params={params}",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "Values normalized to smallest budget (= 1.0). Lower means more improvement. Shows sample efficiency per method.",
        ha="center",
        fontsize=8,
        color="0.4",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_training_time_scaling(
    aggregate_rows: list[dict[str, Any]],
    params: str,
    output_path: Path,
) -> bool:
    """Training time vs budget with O(N) and O(log N) reference curves."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    param_rows = [row for row in aggregate_rows if row["params"] == params]
    if not param_rows:
        return False
    methods = sorted({row["method"] for row in param_rows})
    method_colors = _method_color_map(methods, plt)

    fig, ax = plt.subplots(figsize=(_THESIS_TEXTWIDTH_IN, 4.0))
    all_budgets: list[float] = []
    all_times: list[float] = []
    plotted_any = False

    for method in methods:
        method_rows = sorted(
            [
                r
                for r in param_rows
                if r["method"] == method
                and isinstance(r.get("train_time_s"), dict)
                and r["train_time_s"].get("mean") is not None
            ],
            key=lambda r: int(r.get("requested_budget_steps") or 0),
        )
        if not method_rows:
            continue
        budgets = [int(r["requested_budget_steps"]) for r in method_rows]
        means = [float(r["train_time_s"]["mean"]) for r in method_rows]
        stds = [float(r["train_time_s"]["std"]) for r in method_rows]
        ax.plot(
            budgets,
            means,
            marker="o",
            color=method_colors[method],
            linewidth=2.0,
            markersize=6,
            label=method.upper(),
            zorder=3,
        )
        ax.fill_between(
            budgets,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            color=method_colors[method],
            alpha=0.1,
            zorder=1,
        )
        all_budgets.extend(budgets)
        all_times.extend(means)
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return False

    # Reference scaling curves
    budget_range = np.linspace(min(all_budgets), max(all_budgets), 100)
    med_time = float(np.median(all_times))
    med_budget = float(np.median(all_budgets))
    if med_budget > 0:
        on_ref = med_time * (budget_range / med_budget)
        ax.plot(
            budget_range,
            on_ref,
            "--",
            color="0.55",
            linewidth=1.2,
            alpha=0.6,
            label="$\\mathcal{O}(N)$ ref.",
            zorder=0,
        )
    if med_budget > 1:
        olog_ref = med_time * np.log(budget_range) / np.log(med_budget)
        ax.plot(
            budget_range,
            olog_ref,
            ":",
            color="0.55",
            linewidth=1.2,
            alpha=0.6,
            label="$\\mathcal{O}(\\log N)$ ref.",
            zorder=0,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Simulation Budget $N$", fontsize=9)
    ax.set_ylabel("Training Time (s)", fontsize=9)
    ax.set_title(
        f"Training Time Scaling | params={params}",
        fontsize=10,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.legend(fontsize=7, framealpha=0.7)
    ax.tick_params(labelsize=7)
    fig.text(
        0.5,
        0.01,
        "Dashed & dotted lines show $\\mathcal{O}(N)$ and $\\mathcal{O}(\\log N)$ reference scaling from median values.",
        ha="center",
        fontsize=7,
        color="0.4",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=_THESIS_DPI, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_training_time_per_tseg(
    aggregate_rows: list[dict[str, Any]],
    params: str,
    output_path: Path,
) -> bool:
    """Training time vs budget with one clean line per (method, T_seg) combination.

    Each line is coloured by method and distinguished by marker/dash style per
    T_seg, avoiding the zigzag artefact of the aggregated training-time plot.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    param_rows = [row for row in aggregate_rows if row["params"] == params]
    if not param_rows:
        return False
    methods = sorted({row["method"] for row in param_rows})
    tsegs = sorted({int(r["T_seg"]) for r in param_rows})
    method_colors = _method_color_map(methods, plt)

    tseg_linestyles = ["-", "--", ":"]
    tseg_markers = ["o", "s", "^"]

    fig, ax = plt.subplots(figsize=(_THESIS_TEXTWIDTH_IN, 4.0))
    plotted_any = False

    for method in methods:
        for tidx, tseg in enumerate(tsegs):
            tseg_rows = sorted(
                [
                    r
                    for r in param_rows
                    if r["method"] == method
                    and int(r.get("T_seg", 0)) == tseg
                    and isinstance(r.get("train_time_s"), dict)
                    and r["train_time_s"].get("mean") is not None
                ],
                key=lambda r: int(r.get("requested_budget_steps") or 0),
            )
            if not tseg_rows:
                continue
            budgets = [int(r["requested_budget_steps"]) for r in tseg_rows]
            means = [float(r["train_time_s"]["mean"]) for r in tseg_rows]
            stds = [float(r["train_time_s"]["std"]) for r in tseg_rows]
            ls = tseg_linestyles[tidx % len(tseg_linestyles)]
            mk = tseg_markers[tidx % len(tseg_markers)]
            label = (
                f"{method.upper()} $T_{{seg}}$={tseg}"
                if tidx == 0
                else f"$T_{{seg}}$={tseg}"
            )
            ax.plot(
                budgets,
                means,
                linestyle=ls,
                marker=mk,
                color=method_colors[method],
                linewidth=1.6,
                markersize=5,
                label=f"{method.upper()} / $T_{{seg}}$={tseg}",
                zorder=3,
            )
            ax.fill_between(
                budgets,
                [max(1e-1, m - s) for m, s in zip(means, stds)],
                [m + s for m, s in zip(means, stds)],
                color=method_colors[method],
                alpha=0.08,
                zorder=1,
            )
            plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return False

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Simulation Budget $N$", fontsize=9)
    ax.set_ylabel("Training Time (s)", fontsize=9)
    ax.set_title(
        f"Training Time per Segment Length | params={params}",
        fontsize=10,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.legend(
        fontsize=6,
        framealpha=0.7,
        ncol=len(tsegs),
        loc="upper left",
        title="Method / $T_{seg}$",
        title_fontsize=7,
    )
    ax.tick_params(labelsize=7)
    fig.text(
        0.5,
        0.01,
        "Each line = one method × segment length. Shaded band: ±1 std across seeds.",
        ha="center",
        fontsize=7,
        color="0.4",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=_THESIS_DPI, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_metric_value_heatmap(
    aggregate_rows: list[dict[str, Any]],
    params: str,
    output_path: Path,
    tseg_filter: int | None = None,
) -> bool:
    """Summary heatmap of metric values, color-coded by column-normalized performance.

    *tseg_filter*: if given, restrict rows to that T_seg; otherwise average over all
    T_seg values at the largest budget.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    param_rows = [row for row in aggregate_rows if row["params"] == params]
    if tseg_filter is not None:
        param_rows = [r for r in param_rows if int(r.get("T_seg", 0)) == tseg_filter]
    if not param_rows:
        return False

    methods = sorted({row["method"] for row in param_rows})
    budgets = sorted({int(row["requested_budget_steps"]) for row in param_rows})
    n_methods = len(methods)

    if not budgets or n_methods == 0:
        return False

    # Use the largest budget for the summary
    max_budget = budgets[-1]
    best_rows = [
        r for r in param_rows if int(r["requested_budget_steps"]) == max_budget
    ]

    # Avg normalised RMSE: mean of {param}_rmse_norm across active parameters.
    # Each param's RMSE is already divided by its prior range, so all params
    # contribute equally regardless of physical scale.
    active_params = [p.strip() for p in params.split(",")]
    norm_rmse_keys = [f"{p}_rmse_norm" for p in active_params]
    avg_norm_rmse_by_method: dict[str, float] = {}
    for _method in methods:
        _m_rows = [r for r in best_rows if r["method"] == _method]
        _vals = [
            float(r[k]["mean"])
            for r in _m_rows
            for k in norm_rmse_keys
            if isinstance(r.get(k), dict) and r[k].get("mean") is not None
        ]
        avg_norm_rmse_by_method[_method] = (
            float(np.mean(_vals)) if _vals else float("nan")
        )

    # Joint W2: Wasserstein-2 distance between inferred and ground-truth posterior.
    # Raw physical-space value — scale dominated by the largest-scale parameter.
    w2_by_method: dict[str, float] = {}
    for _method in methods:
        _m_rows = [r for r in best_rows if r["method"] == _method]
        _vals = [
            float(r["w2_mean"]["mean"])
            for r in _m_rows
            if isinstance(r.get("w2_mean"), dict)
            and r["w2_mean"].get("mean") is not None
        ]
        w2_by_method[_method] = float(np.mean(_vals)) if _vals else float("nan")

    _have_norm_rmse = any(not np.isnan(v) for v in avg_norm_rmse_by_method.values())
    _have_w2 = any(not np.isnan(v) for v in w2_by_method.values())
    summary_metrics: list[tuple[str, str, str]] = []
    if _have_norm_rmse:
        summary_metrics.append(("_avg_norm_rmse", "RMSE\n(norm)", "lower"))
    if _have_w2:
        summary_metrics.append(("_w2_joint", "W2\n(joint)", "lower"))
    if not summary_metrics:
        summary_metrics.append(("w2_mean", "W2", "lower"))
    summary_metrics += [
        ("heldout_ppc_rmse_mean", "PPC RMSE", "lower"),
        ("c2st_mean", "C2ST", "target_0.5"),
        ("one_step_rmse", "1-Step", "lower"),
        ("train_time_s", "Train (s)", "lower"),
    ]
    n_metrics = len(summary_metrics)

    heat = np.full((n_methods, n_metrics), np.nan, dtype=np.float64)
    raw_values = np.full((n_methods, n_metrics), np.nan, dtype=np.float64)

    for midx, method in enumerate(methods):
        # Average over all T_seg rows at max_budget (fixes arbitrary T_seg selection)
        method_rows_at_budget = [r for r in best_rows if r["method"] == method]
        if not method_rows_at_budget:
            continue
        for cidx, (metric_key, _, direction) in enumerate(summary_metrics):
            if metric_key == "_avg_norm_rmse":
                v = avg_norm_rmse_by_method.get(method, float("nan"))
                if not np.isnan(v):
                    raw_values[midx, cidx] = v
                    heat[midx, cidx] = v
                continue
            if metric_key == "_w2_joint":
                v = w2_by_method.get(method, float("nan"))
                if not np.isnan(v):
                    raw_values[midx, cidx] = v
                    heat[midx, cidx] = v
                continue
            vals = [
                float(r[metric_key]["mean"])
                for r in method_rows_at_budget
                if isinstance(r.get(metric_key), dict)
                and r[metric_key].get("mean") is not None
            ]
            if not vals:
                continue
            v = float(np.mean(vals))
            raw_values[midx, cidx] = v
            if direction == "target_0.5":
                heat[midx, cidx] = abs(v - 0.5)
            else:
                heat[midx, cidx] = v

    # Column-normalize: 0 = best, 1 = worst
    norm_heat = np.full_like(heat, np.nan)
    for cidx in range(n_metrics):
        col = heat[:, cidx]
        valid = ~np.isnan(col)
        if valid.sum() < 2:
            norm_heat[valid, cidx] = 0.5
            continue
        vmin, vmax = np.nanmin(col), np.nanmax(col)
        span = vmax - vmin
        if span < 1e-12:
            norm_heat[valid, cidx] = 0.5
        else:
            norm_heat[:, cidx] = (col - vmin) / span

    fig, ax = plt.subplots(figsize=(_THESIS_TEXTWIDTH_IN, 2.5 + 0.5 * n_methods))
    im = ax.imshow(norm_heat, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n_metrics))
    ax.set_xticklabels(
        [label for _, label, _ in summary_metrics],
        fontsize=9,
        fontweight="semibold",
        rotation=15,
        ha="right",
    )
    ax.set_yticks(range(n_methods))
    ax.set_yticklabels([m.upper() for m in methods], fontsize=10, fontweight="semibold")

    for midx in range(n_methods):
        for cidx in range(n_metrics):
            raw = raw_values[midx, cidx]
            if np.isnan(raw):
                ax.text(
                    cidx,
                    midx,
                    "\u2014",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="0.5",
                )
            else:
                nv = norm_heat[midx, cidx]
                color = "white" if (nv > 0.7 or nv < 0.2) else "black"
                if abs(raw) >= 100:
                    text = f"{raw:.0f}"
                elif abs(raw) >= 1:
                    text = f"{raw:.2f}"
                else:
                    text = f"{raw:.4f}"
                ax.text(
                    cidx, midx, text, ha="center", va="center", fontsize=9, color=color
                )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Normalized (0 = best, 1 = worst)", fontsize=9)
    _tseg_label = (
        f" | $T_{{seg}}$={tseg_filter}"
        if tseg_filter is not None
        else " | T_seg: averaged"
    )
    ax.set_title(
        f"Metric Summary — Budget={_budget_to_label(max_budget)} | params={params}{_tseg_label}",
        fontsize=11,
        fontweight="bold",
    )
    n_param_fields = len(params.split(","))
    joint_note = (
        " Metrics reflect the joint posterior over all parameters simultaneously."
        if n_param_fields > 1
        else ""
    )
    fig.text(
        0.5,
        0.01,
        "RMSE(norm) = avg per-param RMSE÷prior range. W2(joint) = Wasserstein-2 vs ground truth (physical space). Green = best."
        + joint_note,
        ha="center",
        fontsize=8,
        color="0.4",
        wrap=True,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=_THESIS_DPI, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_metric_vs_tseg(
    aggregate_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    params: str,
    metric_key: str,
    metric_label: str,
    output_path: Path,
    *,
    log_y: bool = False,
    reference_lines: list[float] | None = None,
) -> bool:
    """Aggregate metric vs T_seg, one subplot per budget, with mean and min-max bands.

    All subplots share y-axis limits for direct visual comparison.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    param_agg = [row for row in aggregate_rows if row["params"] == params]
    if not param_agg:
        return False

    budgets = sorted({int(r["requested_budget_steps"]) for r in param_agg})
    if not budgets:
        return False

    methods = sorted({r["method"] for r in param_agg})
    method_colors = _method_color_map(methods, plt)

    ncols = min(3, len(budgets))
    nrows = int(np.ceil(len(budgets) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(_THESIS_TEXTWIDTH_IN, 3.2 * nrows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    # Compute global y-limits from raw seed values
    raw_param = [
        row
        for row in raw_rows
        if row["params"] == params and row.get(metric_key) is not None
    ]
    global_ys: list[float] = [float(r[metric_key]) for r in raw_param]

    plotted_any = False
    for bidx, budget in enumerate(budgets):
        ax = axes_flat[bidx]
        # Use raw rows to compute mean / min / max per method per T_seg
        budget_raw = [
            r for r in raw_param if int(r.get("requested_budget_steps") or 0) == budget
        ]

        for method in methods:
            m_rows = [r for r in budget_raw if r["method"] == method]
            if not m_rows:
                continue
            by_tseg: dict[int, list[float]] = {}
            for r in m_rows:
                by_tseg.setdefault(int(r["T_seg"]), []).append(float(r[metric_key]))
            tsegs_sorted = sorted(by_tseg.keys())
            means = [float(np.mean(by_tseg[t])) for t in tsegs_sorted]
            lo = [float(np.min(by_tseg[t])) for t in tsegs_sorted]
            hi = [float(np.max(by_tseg[t])) for t in tsegs_sorted]
            ax.plot(
                tsegs_sorted,
                means,
                marker="o",
                color=method_colors[method],
                linewidth=1.6,
                markersize=4,
                label=method.upper(),
                zorder=3,
            )
            ax.fill_between(
                tsegs_sorted,
                lo,
                hi,
                color=method_colors[method],
                alpha=0.15,
                zorder=1,
            )
            plotted_any = True

        if reference_lines:
            for val in reference_lines:
                ax.axhline(val, color="0.45", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_title(
            f"Budget = {_budget_to_label(budget)}",
            fontsize=9,
            fontweight="semibold",
        )
        ax.set_xlabel("Segment Length ($T_{seg}$)", fontsize=8)
        ax.set_ylabel(metric_label, fontsize=8)
        ax.grid(True, alpha=0.2, linewidth=0.5)
        ax.tick_params(labelsize=7)
        if bidx == 0:
            ax.legend(fontsize=7, framealpha=0.7)

    # Shared y-limits
    if global_ys:
        y_lo, y_hi = min(global_ys), max(global_ys)
        y_pad = (y_hi - y_lo) * 0.08 if y_hi > y_lo else 0.1
        for ax in axes_flat[: len(budgets)]:
            if log_y and y_lo > 0:
                ax.set_yscale("log")
            else:
                ax.set_ylim(y_lo - y_pad, y_hi + y_pad)

    for ax in axes_flat[len(budgets) :]:
        ax.set_visible(False)

    if not plotted_any:
        plt.close(fig)
        return False

    fig.suptitle(
        f"{metric_label} vs $T_{{seg}}$ | params={params}",
        fontsize=10,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "Solid lines: mean across seeds. Shaded bands: min\u2013max range.",
        ha="center",
        fontsize=7,
        color="0.4",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=_THESIS_DPI, bbox_inches="tight")
    plt.close(fig)
    return True


def _build_rank_summaries(
    aggregate_rows: list[dict[str, Any]],
    plot_specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for params in sorted({row["params"] for row in aggregate_rows}):
        for plot_spec in plot_specs:
            metric_key = plot_spec["metric_key"]
            param_rows = [
                row
                for row in aggregate_rows
                if row["params"] == params
                and isinstance(row.get(metric_key), dict)
                and row[metric_key].get("mean") is not None
            ]
            if not param_rows:
                continue

            methods = sorted({str(row["method"]) for row in param_rows})
            combos = sorted(
                {
                    (
                        int(
                            row.get(
                                "requested_budget_steps",
                                row.get("total_budget_steps", 0),
                            )
                            or 0
                        ),
                        int(row["T_seg"]),
                    )
                    for row in param_rows
                }
            )
            if not methods or not combos:
                continue

            method_rank_values: dict[str, list[float]] = {
                method: [] for method in methods
            }
            first_place_counts: dict[str, int] = {method: 0 for method in methods}
            top2_counts: dict[str, int] = {method: 0 for method in methods}

            for combo in combos:
                combo_rows = [
                    row
                    for row in param_rows
                    if (
                        int(
                            row.get(
                                "requested_budget_steps",
                                row.get("total_budget_steps", 0),
                            )
                            or 0
                        ),
                        int(row["T_seg"]),
                    )
                    == combo
                ]
                if not combo_rows:
                    continue
                scores = np.asarray(
                    [
                        _comparison_value(float(row[metric_key]["mean"]), plot_spec)
                        for row in combo_rows
                    ],
                    dtype=np.float64,
                )
                ranks = rankdata(scores, method="average")
                for row, rank in zip(combo_rows, ranks):
                    method = str(row["method"])
                    method_rank_values[method].append(float(rank))
                    if np.isclose(rank, 1.0):
                        first_place_counts[method] += 1
                    if rank <= 2.0 + 1e-9:
                        top2_counts[method] += 1

            summaries.append(
                {
                    "params": params,
                    "metric": metric_key,
                    "metric_label": plot_spec["metric_label"],
                    "direction": _metric_direction(plot_spec),
                    "num_combos": len(combos),
                    "combo_labels": [
                        {
                            "requested_budget_steps": int(budget),
                            "T_seg": int(tseg),
                        }
                        for budget, tseg in combos
                    ],
                    "method_summaries": [
                        {
                            "method": method,
                            "mean_rank": (
                                float(np.mean(method_rank_values[method]))
                                if method_rank_values[method]
                                else None
                            ),
                            "first_place_finishes": int(first_place_counts[method]),
                            "top2_finishes": int(top2_counts[method]),
                            "num_ranked_combos": int(len(method_rank_values[method])),
                        }
                        for method in methods
                    ],
                }
            )
    return summaries


def _collect_plot_gallery(
    completed_runs: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    """Auto-select best config per method: highest budget, largest T_seg, first seed.

    Copies one posterior and one PPC plot per method into a flat gallery directory.
    No contact sheets are generated.
    """
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    patterns = {
        "posterior": "prior_posterior_grid_ex*.png",
        "ppc": "ppc_timeseries_simulated_test_ex*.png",
    }

    # Index all available figures
    entries: list[dict[str, Any]] = []
    for record in completed_runs:
        cfg = record["cfg"]
        metrics = record["metrics"]
        exp_dir = record["exp_dir"]
        figures_dir = exp_dir / "figures"
        if not figures_dir.exists():
            continue

        params = _flatten_active_parameters(cfg.get("active_parameters", []))
        params_slug = _slugify(params)
        budget = int(
            metrics.get("budget_metadata", {}).get("requested_budget_steps")
            or metrics.get("budget_metadata", {}).get("total_simulation_budget_steps")
            or 0
        )
        method = str(cfg.get("method"))
        tseg = int(cfg.get("T_seg"))
        seed = int(cfg.get("random_seed", cfg.get("sim_seed", 0)))

        for plot_kind, pattern in patterns.items():
            for src in sorted(figures_dir.glob(pattern)):
                entries.append(
                    {
                        "plot_kind": plot_kind,
                        "params": params,
                        "params_slug": params_slug,
                        "method": method,
                        "requested_budget_steps": budget,
                        "T_seg": tseg,
                        "seed": seed,
                        "source": str(src),
                        "example_key": src.stem,
                    }
                )

    # For each (method, params, plot_kind, example_key), pick:
    #   highest budget -> largest T_seg -> smallest seed
    best: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for e in entries:
        key = (e["method"], e["params_slug"], e["plot_kind"], e["example_key"])
        prev = best.get(key)
        if prev is None or (
            e["requested_budget_steps"],
            e["T_seg"],
            -e["seed"],
        ) > (
            prev["requested_budget_steps"],
            prev["T_seg"],
            -prev["seed"],
        ):
            best[key] = e

    copied: list[dict[str, Any]] = []
    for (_method, _params_slug, _plot_kind, _example_key), entry in sorted(
        best.items()
    ):
        src = Path(entry["source"])
        if not src.exists():
            continue
        dst_dir = output_dir / _plot_kind / _params_slug
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_name = (
            f"{_method}_b{entry['requested_budget_steps']}"
            f"_t{entry['T_seg']}_s{entry['seed']}_{src.name}"
        )
        dst = dst_dir / dst_name
        shutil.copy2(src, dst)
        copied.append(
            {
                **entry,
                "copied_to": str(dst),
            }
        )

    return {"copied": copied}


def _load_completed_runs(
    experiments_root: Path, exp_prefix: str
) -> list[dict[str, Any]]:
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
        timing = metrics.get("timing_breakdown", {})
        one_step = metrics.get("one_step_rmse", {})
        multi_ppc = metrics.get("multi_traj_ppc", {})
        run_status = metrics.get("run_status_summary", {})
        requested_budget_steps = budget.get("requested_budget_steps")
        effective_budget_steps = budget.get("effective_budget_steps")
        if requested_budget_steps is None:
            requested_budget_steps = budget.get("total_simulation_budget_steps")
        if effective_budget_steps is None:
            effective_budget_steps = budget.get("total_simulation_budget_steps")

        # C2ST mean across examples
        c2st_raw = metrics.get("c2st", {})
        c2st_vals = [float(v) for v in c2st_raw.values() if isinstance(v, (int, float))]
        c2st_mean = float(sum(c2st_vals) / len(c2st_vals)) if c2st_vals else None

        # Prefer direct coverage abs errors from w2 results, fall back to manual
        cov50 = heldout.get("coverage_50")
        cov90 = heldout.get("coverage_90")
        coverage_50_abs_error = heldout.get("coverage_50_abs_error")
        if coverage_50_abs_error is None and cov50 is not None:
            coverage_50_abs_error = abs(float(cov50) - 0.5)
        coverage_90_abs_error = heldout.get("coverage_90_abs_error")
        if coverage_90_abs_error is None and cov90 is not None:
            coverage_90_abs_error = abs(float(cov90) - 0.9)

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
            # Posterior quality
            "w2_mean": heldout.get("w2_mean"),
            "l2_error_mean": heldout.get("l2_error_mean"),
            "swd_posterior_vs_true": heldout.get("swd_posterior_vs_true"),
            # Calibration
            "coverage_90": cov90,
            "coverage_50": cov50,
            "coverage_curve_mae": heldout.get("coverage_curve_mae"),
            "coverage_50_abs_error": coverage_50_abs_error,
            "coverage_90_abs_error": coverage_90_abs_error,
            "c2st_mean": c2st_mean,
            # Predictive quality
            "heldout_ppc_rmse_mean": heldout_ppc.get("rmse_mean"),
            "heldout_ppc_w2_mean": heldout_ppc.get("w2_mean"),
            "heldout_ppc_rmse_std": heldout_ppc.get("rmse_std"),
            "heldout_ppc_w2_std": heldout_ppc.get("w2_std"),
            "one_step_rmse": one_step.get("rmse_overall"),
            # Real-data PPC
            "real_ppc_rmse_mean": multi_ppc.get("rmse_mean"),
            "real_ppc_w2_mean": multi_ppc.get("w2_mean"),
            # Training
            "train_time_s": training.get("train_time_s"),
            "epochs_trained": training.get("epochs_trained"),
            "num_train_steps": training.get("num_train_steps"),
            "optimizer_examples_seen": training.get("optimizer_examples_seen"),
            "training_batch_size": training.get("training_batch_size"),
            "num_outer_epochs": training.get("num_outer_epochs"),
            "num_inner_epochs": training.get("num_inner_epochs"),
            # Runtime
            "sampling_time_mean_s": heldout.get("sampling_time_mean_s"),
            "dataset_generation_time_s": timing.get("dataset_generation_time_s"),
            # Status
            "run_status": run_status.get("final_state"),
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
        "run_status",
    }
    for (method, params, requested_budget_steps, tseg), group_rows in sorted(
        grouped.items()
    ):
        record: dict[str, Any] = {
            "method": method,
            "params": params,
            "T_seg": tseg,
            "num_runs": len(group_rows),
            "requested_budget_steps": requested_budget_steps,
            "effective_budget_steps": _require_consistent_value(
                group_rows, "effective_budget_steps"
            ),
            "total_budget_steps": _require_consistent_value(
                group_rows, "total_budget_steps"
            ),
        }
        candidate_fields = sorted(
            {
                field
                for row in group_rows
                for field in row.keys()
                if field not in ignore_fields
            }
        )
        for field in candidate_fields:
            values = [float(r[field]) for r in group_rows if r.get(field) is not None]
            if values:
                record[field] = _metric_summary(values)
        aggregate_rows.append(record)
    return aggregate_rows


def _build_pairwise_tests(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairwise_tests: list[dict[str, Any]] = []
    pairwise_plot_specs = {
        spec["metric_key"]: spec
        for spec in [
            *_base_plot_specs(),
            {
                "metric_key": "coverage_50_abs_error",
                "metric_label": "Coverage 50% Abs Error",
                "direction": "lower",
            },
            {
                "metric_key": "coverage_90_abs_error",
                "metric_label": "Coverage 90% Abs Error",
                "direction": "lower",
            },
        ]
    }
    pairwise_metrics = [
        "w2_mean",
        "l2_error_mean",
        "swd_posterior_vs_true",
        "heldout_ppc_rmse_mean",
        "one_step_rmse",
        "coverage_curve_mae",
        "coverage_50_abs_error",
        "coverage_90_abs_error",
        "c2st_mean",
    ]
    group_keys = sorted(
        {
            (
                row["params"],
                int(
                    row.get("requested_budget_steps", row.get("total_budget_steps", 0))
                    or 0
                ),
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
            rows_a = {
                row["seed"]: row for row in candidate_rows if row["method"] == method_a
            }
            rows_b = {
                row["seed"]: row for row in candidate_rows if row["method"] == method_b
            }
            shared_seeds = sorted(set(rows_a) & set(rows_b))
            for metric_key in pairwise_metrics:
                plot_spec = pairwise_plot_specs[metric_key]
                shared_metric_seeds = [
                    s
                    for s in shared_seeds
                    if rows_a[s].get(metric_key) is not None
                    and rows_b[s].get(metric_key) is not None
                ]
                x = [float(rows_a[s][metric_key]) for s in shared_metric_seeds]
                y = [float(rows_b[s][metric_key]) for s in shared_metric_seeds]
                comparison_x = [_comparison_value(value, plot_spec) for value in x]
                comparison_y = [_comparison_value(value, plot_spec) for value in y]
                deltas = [
                    _comparison_delta(a_val, b_val, plot_spec)
                    for a_val, b_val in zip(x, y)
                ]
                wins_method_a = int(
                    sum(
                        comparison_a < comparison_b
                        and not np.isclose(comparison_a, comparison_b)
                        for comparison_a, comparison_b in zip(
                            comparison_x, comparison_y
                        )
                    )
                )
                wins_method_b = int(
                    sum(
                        comparison_a > comparison_b
                        and not np.isclose(comparison_a, comparison_b)
                        for comparison_a, comparison_b in zip(
                            comparison_x, comparison_y
                        )
                    )
                )
                ties = int(
                    sum(
                        np.isclose(comparison_a, comparison_b)
                        for comparison_a, comparison_b in zip(
                            comparison_x, comparison_y
                        )
                    )
                )
                pvalue = None
                stat = None
                if (
                    len(x) >= 2
                    and len(y) >= 2
                    and not all(
                        np.isclose(comparison_a, comparison_b)
                        for comparison_a, comparison_b in zip(
                            comparison_x, comparison_y
                        )
                    )
                ):
                    try:
                        result = wilcoxon(
                            comparison_x,
                            comparison_y,
                            zero_method="wilcox",
                            alternative="two-sided",
                        )
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
                        "num_shared_seeds": len(shared_metric_seeds),
                        "metric_direction": _metric_direction(plot_spec),
                        "delta_definition": _comparison_delta_footer(
                            plot_spec, method_b
                        ),
                        "delta_mean": float(np.mean(deltas)) if deltas else None,
                        "delta_median": float(np.median(deltas)) if deltas else None,
                        "wins_method_a": wins_method_a,
                        "wins_method_b": wins_method_b,
                        "ties": ties,
                        "paired_deltas": [
                            {
                                "seed": int(seed),
                                "method_a_value": float(a_val),
                                "method_b_value": float(b_val),
                                "method_a_comparison_value": float(comparison_a),
                                "method_b_comparison_value": float(comparison_b),
                                "delta": float(delta),
                            }
                            for seed, a_val, b_val, comparison_a, comparison_b, delta in zip(
                                shared_metric_seeds,
                                x,
                                y,
                                comparison_x,
                                comparison_y,
                                deltas,
                            )
                        ],
                        "statistic": stat,
                        "pvalue": pvalue,
                    }
                )

    for metric_key in set(pairwise_metrics):
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
    summary_suffixes = ["n", "mean", "std", "min", "max"]
    fieldnames = base_fields + [
        f"{field}_{suffix}" for field in metric_fields for suffix in summary_suffixes
    ]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in aggregate_rows:
            out_row = {field: row.get(field) for field in base_fields}
            for field in metric_fields:
                summary = row.get(field, {})
                for suffix in summary_suffixes:
                    out_row[f"{field}_{suffix}"] = summary.get(suffix)
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
    plot_registry = _build_plot_registry(rows)
    pairwise_tests = _build_pairwise_tests(rows)
    rank_summaries = _build_rank_summaries(aggregate_rows, plot_registry["heatmap"])
    notes = [
        "Pairwise Wilcoxon tests are descriptive only. With very small shared seed counts they should not be interpreted as strong inferential evidence."
    ]

    _write_csv(output_csv, aggregate_rows)

    plots_dir = output_json.parent / f"{args.exp_prefix}_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    generated_plots: list[str] = []
    for params in sorted({row["params"] for row in rows}):
        params_slug = _slugify(params)

        # --- 1. Individual budget plots (primary metrics only) ---
        for plot_spec in plot_registry["budget"]:
            output_path = plots_dir / f"{plot_spec['slug']}_{params_slug}.png"
            if _plot_metric_vs_budget(rows, params, plot_spec, output_path):
                generated_plots.append(str(output_path))

        # --- 2. Calibration diagnostic panel ---
        calibration_path = plots_dir / f"calibration_panel_{params_slug}.png"
        if _plot_calibration_panel(aggregate_rows, params, calibration_path):
            generated_plots.append(str(calibration_path))

        # --- 3. Per-parameter comparison bars ---
        per_param_path = plots_dir / f"per_parameter_bars_{params_slug}.png"
        if _plot_per_parameter_bars(aggregate_rows, rows, params, per_param_path):
            generated_plots.append(str(per_param_path))

        # --- 4. Training time scaling ---
        scaling_path = plots_dir / f"training_time_scaling_{params_slug}.png"
        if _plot_training_time_scaling(aggregate_rows, params, scaling_path):
            generated_plots.append(str(scaling_path))
        tseg_scaling_path = plots_dir / f"training_time_per_tseg_{params_slug}.png"
        if _plot_training_time_per_tseg(aggregate_rows, params, tseg_scaling_path):
            generated_plots.append(str(tseg_scaling_path))

        # --- 5. Metric value heatmap (averaged over T_seg) + one per T_seg ---
        value_heatmap_path = plots_dir / f"metric_value_heatmap_{params_slug}.png"
        if _plot_metric_value_heatmap(aggregate_rows, params, value_heatmap_path):
            generated_plots.append(str(value_heatmap_path))
        _tsegs_available = sorted(
            {int(r["T_seg"]) for r in aggregate_rows if r["params"] == params}
        )
        for _tseg in _tsegs_available:
            _tseg_path = (
                plots_dir / f"metric_value_heatmap_{params_slug}_tseg{_tseg}.png"
            )
            if _plot_metric_value_heatmap(
                aggregate_rows, params, _tseg_path, tseg_filter=_tseg
            ):
                generated_plots.append(str(_tseg_path))

        # --- 6. W2 vs T_seg ---
        w2_tseg_path = plots_dir / f"w2_vs_tseg_{params_slug}.png"
        if _plot_metric_vs_tseg(
            aggregate_rows,
            rows,
            params,
            "w2_mean",
            "Wasserstein-2 Distance",
            w2_tseg_path,
        ):
            generated_plots.append(str(w2_tseg_path))

        # --- 7. Sampling time vs T_seg ---
        sampling_tseg_path = plots_dir / f"sampling_time_vs_tseg_{params_slug}.png"
        if _plot_metric_vs_tseg(
            aggregate_rows,
            rows,
            params,
            "sampling_time_mean_s",
            "Sampling Time / Case (s)",
            sampling_tseg_path,
            log_y=True,
        ):
            generated_plots.append(str(sampling_tseg_path))

        # --- 8. PPC RMSE vs T_seg ---
        ppc_tseg_path = plots_dir / f"ppc_rmse_vs_tseg_{params_slug}.png"
        if _plot_metric_vs_tseg(
            aggregate_rows,
            rows,
            params,
            "heldout_ppc_rmse_mean",
            "PPC RMSE (held-out)",
            ppc_tseg_path,
        ):
            generated_plots.append(str(ppc_tseg_path))

        # --- 9. Pareto efficiency scatter (accuracy vs cost) ---
        for pareto_spec in plot_registry["pareto"]:
            pareto_path = plots_dir / f"{pareto_spec['slug']}_{params_slug}.png"
            if _plot_pareto_aggregate_scatter(
                rows, aggregate_rows, params, pareto_spec, pareto_path
            ):
                generated_plots.append(str(pareto_path))

    plot_collection_dir = plots_dir / "plot_collection"
    plot_collection = _collect_plot_gallery(completed_runs, plot_collection_dir)
    plot_collection_payload = {
        "root": str(plot_collection_dir),
        **plot_collection,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    metrics_json = output_json.parent / f"{output_json.stem}_metrics.json"
    pairwise_json = output_json.parent / f"{output_json.stem}_pairwise.json"
    plot_manifest_json = output_json.parent / f"{output_json.stem}_plot_manifest.json"

    metrics_payload = {
        "exp_prefix": args.exp_prefix,
        "num_runs": len(rows),
        "raw_rows": rows,
        "aggregate_rows": aggregate_rows,
        "rank_summaries": rank_summaries,
    }
    pairwise_payload = {
        "exp_prefix": args.exp_prefix,
        "notes": notes,
        "pairwise_tests": pairwise_tests,
    }
    plot_manifest_payload = {
        "exp_prefix": args.exp_prefix,
        "plot_registry": plot_registry,
        "generated_plots": generated_plots,
        "plot_collection": plot_collection_payload,
    }
    index_payload = {
        "exp_prefix": args.exp_prefix,
        "num_runs": len(rows),
        "notes": notes,
        "files": {
            "metrics": str(metrics_json),
            "pairwise": str(pairwise_json),
            "plot_manifest": str(plot_manifest_json),
            "csv": str(output_csv),
        },
    }

    for path, payload in [
        (metrics_json, metrics_payload),
        (pairwise_json, pairwise_payload),
        (plot_manifest_json, plot_manifest_payload),
        (output_json, index_payload),
    ]:
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    print(f"Wrote {output_json}")
    print(f"Wrote {metrics_json}")
    print(f"Wrote {pairwise_json}")
    print(f"Wrote {plot_manifest_json}")
    print(f"Wrote {output_csv}")
    print(f"Deduplicated completed runs: {len(completed_runs)}")
    print(f"Plots saved to {plots_dir}")
    for plot_path in generated_plots:
        print(f"Wrote {plot_path}")
    print(
        f"Wrote plot collection ({len(plot_collection.get('copied', []))} files) to {plot_collection_dir}"
    )


if __name__ == "__main__":
    main()
