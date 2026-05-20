#!/usr/bin/env python
"""Generate presentation-ready figures for the thesis defense.

Reads the same experiment data as aggregate_simulation_benchmark.py but
produces simplified, high-visibility plots with large labels/ticks and
minimal visual clutter — designed to be imported directly into PowerPoint.

Figures generated (all land in <output_dir>/):

  defense_w2_{params}.png            — W₂ vs budget  (T=1000 and T=3000 panels)
  defense_calibration_{params}.png   — Cov@90 · Coverage MAE · C2ST vs budget
  defense_metric_heatmap_{params}.png — value heatmap at max budget
  defense_pareto_{params}.png        — training-time vs W₂ Pareto scatter
  defense_w2_scaling.png             — grouped bars: W₂ by parameter count
  defense_real_ppc_rmse_{params}.png — horizontal bars: real-data PPC RMSE

Usage (from the code/ directory):
    python scripts/plot_defense_figures.py \\
        --experiments-root experiments \\
        --exp-prefix bench_budget_v3_fixed

    sbatch scripts/run_defense_figures.sh bench_budget_v3_fixed
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Style constants — must match the thesis defense slide palette (tab10)
# ──────────────────────────────────────────────────────────────────────────────

_COLORS: dict[str, str] = {
    "npe":       "#ff7f0e",   # orange  (tab10 C1)
    "npse":      "#2ca02c",   # green   (tab10 C2)
    "fnpe":      "#1f77b4",   # blue    (tab10 C0) — displayed as FNPSE
    "simformer": "#d62728",   # red     (tab10 C3)
}

# Canonical display names
_DISPLAY: dict[str, str] = {
    "npe":       "NPE",
    "npse":      "NPSE",
    "fnpe":      "FNPSE",
    "simformer": "Simformer",
}

# Preferred left-to-right ordering in legends / bars
_METHOD_ORDER = ["npe", "npse", "fnpe", "simformer"]

# Canonical parameter-config ordering (fewest → most params)
_PARAM_ORDER = ["mu", "mu,cd", "mu,m", "mu,cd,m"]
_PARAM_LABEL: dict[str, str] = {
    "mu":      "μ\n(1 param)",
    "mu,cd":   "(μ, c_d)\n(2 params)",
    "mu,m":    "(μ, m)\n(2 params)",
    "mu,cd,m": "(μ, c_d, m)\n(3 params)",
}

# Figure geometry — wide 16:9 proportions for slides
_W_WIDE   = 11.0   # wide multi-panel (inches)
_W_NARROW = 7.5    # single-panel or heatmap
_H_ROW    = 5.2    # height per panel row
_DPI      = 200

# Typography — all larger than thesis defaults
_FS_TITLE  = 17
_FS_LABEL  = 14
_FS_TICK   = 12
_FS_LEGEND = 12
_FS_CELL   = 11    # heatmap cell annotations

_LW        = 2.8   # line width
_MS        = 9     # marker size
_BAND_A    = 0.18  # shaded-band alpha


# ──────────────────────────────────────────────────────────────────────────────
# Small helpers
# ──────────────────────────────────────────────────────────────────────────────

def _color(method: str) -> str:
    return _COLORS.get(method.lower(), "#888888")


def _label(method: str) -> str:
    return _DISPLAY.get(method.lower(), method.upper())


def _budget_label(b: int) -> str:
    """Format simulation budget as compact string, e.g. '20M'."""
    if b >= 1_000_000:
        return f"{b // 1_000_000}M"
    if b >= 1_000:
        return f"{b // 1_000}k"
    return str(b)


def _slugify(params: str) -> str:
    return params.replace(",", "_").replace(" ", "_")


def _sorted_methods(available: list[str]) -> list[str]:
    """Return methods in canonical order; unknowns appended alphabetically."""
    seen = set(available)
    return [m for m in _METHOD_ORDER if m in seen] + sorted(
        m for m in seen if m not in _METHOD_ORDER
    )


def _get_mean(row: dict[str, Any], key: str) -> float | None:
    entry = row.get(key)
    if entry is None:
        return None
    if isinstance(entry, dict):
        return entry.get("mean")
    return float(entry)


def _get_std(row: dict[str, Any], key: str) -> float:
    entry = row.get(key)
    if isinstance(entry, dict):
        v = entry.get("std")
        return float(v) if v is not None else 0.0
    return 0.0


def _format_budget_ticks(ax: Any, budgets: list[int]) -> None:
    ax.set_xticks(budgets)
    ax.set_xticklabels([_budget_label(b) for b in budgets], fontsize=_FS_TICK)


# ──────────────────────────────────────────────────────────────────────────────
# Data loading  (mirrors aggregate_simulation_benchmark.py)
# ──────────────────────────────────────────────────────────────────────────────

def _parse_run_timestamp(exp_dir: Path) -> datetime:
    match = re.search(r"(\d{8}-\d{6})$", exp_dir.name)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%d-%H%M%S")
    return datetime.fromtimestamp(exp_dir.stat().st_mtime)


def _flatten_params(active_parameters: list[str]) -> str:
    return ",".join(active_parameters)


def _dedupe_key(cfg: dict[str, Any], metrics: dict[str, Any]) -> tuple[Any, ...]:
    bm = metrics.get("budget_metadata", {})
    budget = bm.get("requested_budget_steps") or bm.get("total_simulation_budget_steps")
    return (
        cfg.get("method"),
        _flatten_params(cfg.get("active_parameters", [])),
        int(cfg.get("T_seg")),
        int(cfg.get("random_seed", cfg.get("sim_seed", 0))),
        budget,
    )


def _load_completed_runs(
    experiments_root: Path, exp_prefix: str
) -> list[dict[str, Any]]:
    """Scan experiments_root for matching runs; deduplicate by (method, params,
    T_seg, seed, budget) keeping the most recent timestamp."""
    deduped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for config_path in experiments_root.glob("*/config.json"):
        metrics_path = config_path.with_name("metrics.json")
        if not metrics_path.exists():
            continue
        with config_path.open(encoding="utf-8") as f:
            cfg = json.load(f)
        if not str(cfg.get("exp_name", "")).startswith(exp_prefix):
            continue
        with metrics_path.open(encoding="utf-8") as f:
            metrics = json.load(f)
        record: dict[str, Any] = {
            "cfg":       cfg,
            "metrics":   metrics,
            "exp_dir":   config_path.parent,
            "timestamp": _parse_run_timestamp(config_path.parent),
        }
        key = _dedupe_key(cfg, metrics)
        existing = deduped.get(key)
        if existing is None or record["timestamp"] > existing["timestamp"]:
            deduped[key] = record
    return list(deduped.values())


def _build_raw_rows(completed_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten each completed run into a single-row dict of scalar metrics."""
    rows: list[dict[str, Any]] = []
    for record in completed_runs:
        cfg     = record["cfg"]
        metrics = record["metrics"]

        heldout = (
            metrics.get("heldout_test_posterior_vs_true")
            or metrics.get("w2_posterior_vs_true")
            or {}
        )
        heldout_ppc = metrics.get("heldout_test_ppc", {}).get("aggregate", {})
        bm          = metrics.get("budget_metadata", {})
        training    = metrics.get("training_summary", {})
        multi_ppc   = metrics.get("multi_traj_ppc", {})

        requested_budget = (
            bm.get("requested_budget_steps")
            or bm.get("total_simulation_budget_steps")
        )

        # C2ST: average over per-example values
        c2st_raw  = metrics.get("c2st", {})
        c2st_vals = [float(v) for v in c2st_raw.values() if isinstance(v, (int, float))]
        c2st_mean = float(np.mean(c2st_vals)) if c2st_vals else None

        # Per-parameter metrics (normalised and physical)
        per_param: dict[str, Any] = {}
        pp_block = (
            heldout.get("per_parameter_metrics")
            or metrics.get("per_parameter_metrics")
            or {}
        )
        for pname, pvals in pp_block.items():
            if not isinstance(pvals, dict):
                continue
            per_param[f"{pname}_rmse_norm"]         = pvals.get("rmse_norm")
            per_param[f"{pname}_rmse_phys"]         = pvals.get("rmse_phys")
            per_param[f"{pname}_bias_mean_norm"]    = pvals.get("bias_mean_norm")
            per_param[f"{pname}_bias_mean_phys"]    = pvals.get("bias_mean_phys")
            per_param[f"{pname}_coverage_90_phys"]  = pvals.get("coverage_90")

        row: dict[str, Any] = {
            "method":                 cfg.get("method"),
            "seed":                   int(cfg.get("random_seed", cfg.get("sim_seed", 0))),
            "params":                 _flatten_params(cfg.get("active_parameters", [])),
            "T_seg":                  int(cfg.get("T_seg")),
            "requested_budget_steps": requested_budget,
            # Posterior quality
            "w2_mean":                heldout.get("w2_mean"),
            # Calibration
            "coverage_90":            heldout.get("coverage_90"),
            "coverage_curve_mae":     heldout.get("coverage_curve_mae"),
            "c2st_mean":              c2st_mean,
            # Predictive quality (simulative)
            "heldout_ppc_rmse_mean":  heldout_ppc.get("rmse_mean"),
            # Real-data PPC
            "real_ppc_rmse_mean":     multi_ppc.get("rmse_mean"),
            # Runtime
            "train_time_s":           training.get("train_time_s"),
            **per_param,
        }
        rows.append(row)
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation  (mean ± std across seeds)
# ──────────────────────────────────────────────────────────────────────────────

def _aggregate_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group by (method, params, T_seg, budget); compute mean/std per metric."""
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        key = (
            row.get("method"),
            row.get("params"),
            row.get("T_seg"),
            row.get("requested_budget_steps"),
        )
        groups[key].append(row)

    agg_rows: list[dict[str, Any]] = []
    for (method, params, tseg, budget), group in sorted(groups.items()):
        agg: dict[str, Any] = {
            "method":                 method,
            "params":                 params,
            "T_seg":                  tseg,
            "requested_budget_steps": budget,
            "n_seeds":                len(group),
        }

        # Discover all scalar metric keys from the group
        skip = {"method", "params", "T_seg", "requested_budget_steps", "seed",
                "exp_dir", "n_seeds"}
        all_keys: set[str] = set()
        for row in group:
            all_keys.update(row.keys())
        metric_keys = sorted(all_keys - skip)

        for key in metric_keys:
            vals = [float(r[key]) for r in group
                    if r.get(key) is not None and isinstance(r.get(key), (int, float))]
            if not vals:
                agg[key] = {"mean": None, "std": 0.0, "n": 0}
            else:
                agg[key] = {
                    "mean": float(np.mean(vals)),
                    "std":  float(np.std(vals)) if len(vals) > 1 else 0.0,
                    "n":    len(vals),
                }

        agg_rows.append(agg)
    return agg_rows


# ──────────────────────────────────────────────────────────────────────────────
# PLOT 1 — W₂ distance vs simulation budget
# ──────────────────────────────────────────────────────────────────────────────

def plot_defense_w2(
    agg_rows: list[dict[str, Any]],
    params: str,
    output_path: Path,
    show_tsegs: tuple[int, ...] = (1000, 2000, 3000),
) -> None:
    """W₂ vs budget — one column per T_seg, all 4 methods, mean ± std band."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    param_rows = [r for r in agg_rows if r["params"] == params]
    methods    = _sorted_methods([r["method"] for r in param_rows])
    tsegs      = [t for t in show_tsegs
                  if any(int(r["T_seg"]) == t for r in param_rows)]
    if not tsegs or not methods:
        print(f"  [skip] no data for W₂ plot: params={params}")
        return

    n   = len(tsegs)
    fig, axes = plt.subplots(
        1, n,
        figsize=(_W_NARROW + (n - 1) * 4.5, _H_ROW),
        sharey=False,
        squeeze=False,
    )

    for ax, tseg in zip(axes[0], tsegs):
        t_rows  = [r for r in param_rows if int(r["T_seg"]) == tseg]
        budgets = sorted({
            int(r["requested_budget_steps"])
            for r in t_rows
            if r.get("requested_budget_steps")
        })

        for method in methods:
            by_budget = {
                int(r["requested_budget_steps"]): r
                for r in t_rows
                if r["method"] == method and r.get("requested_budget_steps")
            }
            xs  = [b for b in budgets if b in by_budget]
            ys  = [_get_mean(by_budget[b], "w2_mean") for b in xs]
            es  = [_get_std( by_budget[b], "w2_mean") for b in xs]
            if not any(y is not None for y in ys):
                continue
            ys_c = [y if y is not None else float("nan") for y in ys]
            ax.plot(xs, ys_c,
                    marker="o", color=_color(method),
                    linewidth=_LW, markersize=_MS,
                    label=_label(method), zorder=3)
            ax.fill_between(
                xs,
                [y - e for y, e in zip(ys_c, es)],
                [y + e for y, e in zip(ys_c, es)],
                color=_color(method), alpha=_BAND_A, zorder=1,
            )

        ax.set_title(f"$T_{{\\mathrm{{seg}}}}$ = {tseg}",
                     fontsize=_FS_TITLE, fontweight="bold", pad=10)
        ax.set_xlabel("Simulation Budget", fontsize=_FS_LABEL, labelpad=6)
        ax.set_ylabel("$W_2$ distance  (lower = better)",
                      fontsize=_FS_LABEL, labelpad=6)
        ax.tick_params(labelsize=_FS_TICK)
        _format_budget_ticks(ax, budgets)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.25, linewidth=0.8)

    # Single shared legend below the figure
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="lower center", ncol=len(methods),
               fontsize=_FS_LEGEND, frameon=True,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    _save(fig, output_path)


# ──────────────────────────────────────────────────────────────────────────────
# PLOT 2 — Calibration panel  (Cov@90 · Coverage MAE · C2ST)
# ──────────────────────────────────────────────────────────────────────────────

def plot_defense_calibration(
    agg_rows: list[dict[str, Any]],
    params: str,
    output_path: Path,
    tseg_filter: int = 1000,
) -> None:
    """Three calibration metrics vs budget.  T_seg filtered to tseg_filter."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    param_rows = [r for r in agg_rows
                  if r["params"] == params and int(r["T_seg"]) == tseg_filter]
    if not param_rows:
        # Fallback: aggregate over all T_seg
        param_rows = [r for r in agg_rows if r["params"] == params]
    methods = _sorted_methods([r["method"] for r in param_rows])
    budgets = sorted({
        int(r["requested_budget_steps"])
        for r in param_rows
        if r.get("requested_budget_steps")
    })
    if not budgets or not methods:
        print(f"  [skip] no data for calibration plot: params={params}")
        return

    panels = [
        ("coverage_90",      "Coverage @ 90%",      0.9,  (0.0, 1.05)),
        ("coverage_curve_mae", "Coverage Curve MAE", None, None),
        ("c2st_mean",        "C2ST  (vs prior)",    0.5,  (0.45, 1.05)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(_W_WIDE, _H_ROW))

    for ax, (mkey, ylabel, ref, ylim) in zip(axes, panels):
        for method in methods:
            by_b = {
                int(r["requested_budget_steps"]): r
                for r in param_rows
                if r["method"] == method and r.get("requested_budget_steps")
            }
            xs = [b for b in budgets if b in by_b]
            ys = [_get_mean(by_b[b], mkey) for b in xs]
            es = [_get_std( by_b[b], mkey) for b in xs]
            valid = [(x, y, e) for x, y, e in zip(xs, ys, es) if y is not None]
            if not valid:
                continue
            xv, yv, ev = zip(*valid)
            ax.plot(xv, yv,
                    marker="o", color=_color(method),
                    linewidth=_LW, markersize=_MS,
                    label=_label(method), zorder=3)
            ax.fill_between(
                xv,
                [y - e for y, e in zip(yv, ev)],
                [y + e for y, e in zip(yv, ev)],
                color=_color(method), alpha=_BAND_A, zorder=1,
            )
        if ref is not None:
            ax.axhline(ref, color="0.30", linestyle="--", linewidth=2.0,
                       alpha=0.80, zorder=0,
                       label=f"Ideal = {ref}")
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xlabel("Simulation Budget", fontsize=_FS_LABEL, labelpad=6)
        ax.set_ylabel(ylabel, fontsize=_FS_LABEL, labelpad=6)
        ax.tick_params(labelsize=_FS_TICK)
        _format_budget_ticks(ax, budgets)
        ax.grid(True, alpha=0.25, linewidth=0.8)

    # Deduplicated shared legend
    seen_labels: list[str] = []
    handles_all: list[Any] = []
    for ax in axes:
        for h, lbl in zip(*ax.get_legend_handles_labels()):
            if lbl not in seen_labels:
                handles_all.append(h)
                seen_labels.append(lbl)
    fig.legend(handles_all, seen_labels,
               loc="lower center", ncol=len(seen_labels),
               fontsize=_FS_LEGEND, frameon=True,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(
        f"Calibration  ·  params = {params}  ·  $T_{{\\mathrm{{seg}}}}$ = {tseg_filter}",
        fontsize=_FS_TITLE, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.95))
    _save(fig, output_path)


# ──────────────────────────────────────────────────────────────────────────────
# PLOT 3 — Metric value heatmap
# ──────────────────────────────────────────────────────────────────────────────

def plot_defense_metric_heatmap(
    agg_rows: list[dict[str, Any]],
    params: str,
    output_path: Path,
    budget_filter: int | None = None,
    tseg_filter: int = 1000,
) -> None:
    """Per-method metric summary at fixed (budget, T_seg).
    Cells are colour-coded per column: green = best, red = worst."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    param_rows = [r for r in agg_rows
                  if r["params"] == params and int(r["T_seg"]) == tseg_filter]
    if not param_rows:
        print(f"  [skip] no data for heatmap: params={params} T_seg={tseg_filter}")
        return

    budgets = sorted({
        int(r["requested_budget_steps"])
        for r in param_rows
        if r.get("requested_budget_steps")
    })
    if not budgets:
        return
    use_budget = budget_filter if budget_filter is not None else budgets[-1]
    best_rows  = [r for r in param_rows
                  if int(r.get("requested_budget_steps") or 0) == use_budget]
    if not best_rows:
        print(f"  [skip] no rows at budget={use_budget} for heatmap: params={params}")
        return

    methods = _sorted_methods([r["method"] for r in best_rows])

    # avg-normalised RMSE across active parameters (scale-free accuracy)
    active_params = [p.strip() for p in params.split(",")]
    avg_norm_rmse: dict[str, float | None] = {}
    for method in methods:
        row = next((r for r in best_rows if r["method"] == method), None)
        if row is None:
            avg_norm_rmse[method] = None
            continue
        vals = [_get_mean(row, f"{p}_rmse_norm") for p in active_params]
        vals = [v for v in vals if v is not None]
        avg_norm_rmse[method] = float(np.mean(vals)) if vals else None

    # Only include the avg-norm-RMSE column if at least one method has data for it.
    # When per_parameter_metrics is absent from metrics.json the column is entirely
    # NaN, which renders as a useless blank strip in the heatmap.
    has_norm_rmse = any(v is not None for v in avg_norm_rmse.values())

    metrics_cfg = [
        *([("_avg_norm_rmse", "RMSE\n(norm.)", "lower")] if has_norm_rmse else []),
        ("w2_mean",               "$W_2$",            "lower"),
        ("heldout_ppc_rmse_mean", "PPC\nRMSE",        "lower"),
        ("coverage_90",           "Cov\n@90",         "target_0.9"),
        ("coverage_curve_mae",    "Cov\nMAE",         "lower"),
        ("c2st_mean",             "C2ST",             "higher"),
        ("train_time_s",          "Train\ntime (s)",  "lower"),
    ]

    n_m, n_c = len(methods), len(metrics_cfg)
    raw  = np.full((n_m, n_c), np.nan)
    norm = np.full((n_m, n_c), np.nan)

    for midx, method in enumerate(methods):
        row = next((r for r in best_rows if r["method"] == method), None)
        for cidx, (mkey, _, _) in enumerate(metrics_cfg):
            if mkey == "_avg_norm_rmse":
                v = avg_norm_rmse.get(method)
            else:
                v = _get_mean(row, mkey) if row else None
            if v is not None:
                raw[midx, cidx] = v

    # Column-normalise: 0 → best (green end), 1 → worst (red end)
    for cidx, (_, _, direction) in enumerate(metrics_cfg):
        col   = raw[:, cidx]
        valid = ~np.isnan(col)
        if valid.sum() < 2:
            norm[valid, cidx] = 0.5
            continue
        vmin, vmax = np.nanmin(col), np.nanmax(col)
        span = vmax - vmin
        if span < 1e-12:
            norm[valid, cidx] = 0.5
        elif direction == "higher":
            norm[:, cidx] = (vmax - col) / span
        elif direction == "target_0.9":
            dev = np.abs(col - 0.9)
            norm[:, cidx] = dev / max(np.nanmax(dev), 1e-9)
        elif direction == "target_0.5":
            dev = np.abs(col - 0.5)
            norm[:, cidx] = dev / max(np.nanmax(dev), 1e-9)
        else:  # lower is better
            norm[:, cidx] = (col - vmin) / span

    cell_h   = 1.1
    fig_h    = max(3.5, n_m * cell_h + 2.0)
    fig, ax  = plt.subplots(figsize=(_W_NARROW, fig_h))

    im = ax.imshow(norm, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(n_c))
    ax.set_xticklabels(
        [lbl for _, lbl, _ in metrics_cfg],
        fontsize=_FS_TICK + 1, fontweight="semibold",
    )
    ax.set_yticks(range(n_m))
    ax.set_yticklabels(
        [_label(m) for m in methods],
        fontsize=_FS_TICK + 2, fontweight="semibold",
    )
    ax.tick_params(length=0)

    for midx in range(n_m):
        for cidx in range(n_c):
            v = raw[midx, cidx]
            if np.isnan(v):
                ax.text(cidx, midx, "—", ha="center", va="center",
                        fontsize=_FS_CELL, color="0.5")
            else:
                nv    = norm[midx, cidx]
                color = "white" if (nv > 0.72 or nv < 0.22) else "black"
                txt   = (
                    f"{v:.4f}" if abs(v) < 1
                    else f"{v:.2f}" if abs(v) < 100
                    else f"{v:.0f}"
                )
                ax.text(cidx, midx, txt, ha="center", va="center",
                        fontsize=_FS_CELL, color=color, fontweight="semibold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.55, pad=0.03)
    cbar.set_label("0 = best  ·  1 = worst", fontsize=_FS_TICK - 1)
    cbar.ax.tick_params(labelsize=_FS_TICK - 2)

    ax.set_title(
        f"params = {params}  ·  Budget = {_budget_label(use_budget)}"
        f"  ·  $T_{{\\mathrm{{seg}}}}$ = {tseg_filter}",
        fontsize=_FS_TITLE - 1, fontweight="bold", pad=12,
    )
    fig.tight_layout()
    _save(fig, output_path)


# ──────────────────────────────────────────────────────────────────────────────
# PLOT 4 — Pareto scatter  (training time vs W₂)
# ──────────────────────────────────────────────────────────────────────────────

def _pareto_mask(points: list[tuple[float, float]]) -> list[bool]:
    mask = [True] * len(points)
    for i, (xi, yi) in enumerate(points):
        for j, (xj, yj) in enumerate(points):
            if i == j:
                continue
            if (xj <= xi and yj <= yi) and (xj < xi or yj < yi):
                mask[i] = False
                break
    return mask


def plot_defense_pareto(
    agg_rows: list[dict[str, Any]],
    params: str,
    output_path: Path,
) -> None:
    """Training time vs W₂ scatter — Pareto frontier highlighted."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    param_rows = [r for r in agg_rows if r["params"] == params]
    methods    = _sorted_methods([r["method"] for r in param_rows])
    tsegs      = sorted({int(r["T_seg"]) for r in param_rows})
    markers    = ["o", "s", "^", "D"]
    tseg_mk    = {t: markers[i % len(markers)] for i, t in enumerate(tsegs)}
    tseg_sz    = {t: 85 + 30 * i for i, t in enumerate(tsegs)}

    # Collect (x=train_time, y=w2, method, T_seg) for all valid rows
    pts: list[tuple[float, float, str, int]] = []
    for r in param_rows:
        x = _get_mean(r, "train_time_s")
        y = _get_mean(r, "w2_mean")
        if x is not None and y is not None and x > 0 and y >= 0:
            pts.append((x, y, r["method"], int(r["T_seg"])))

    if not pts:
        print(f"  [skip] no data for Pareto plot: params={params}")
        return

    fig, ax = plt.subplots(figsize=(_W_NARROW, _H_ROW))

    # Pareto front
    xy       = [(x, y) for x, y, _, _ in pts]
    on_front = _pareto_mask(xy)
    front    = sorted(
        [(x, y) for (x, y, _, _), f in zip(pts, on_front) if f],
        key=lambda p: p[0],
    )
    if len(front) >= 2:
        fx, fy = zip(*front)
        # Step-line: extend the last front point rightward
        ax.step(list(fx) + [max(x for x, _, _, _ in pts) * 1.05],
                list(fy) + [fy[-1]],
                where="post",
                color="0.50", linewidth=2.2, linestyle="--",
                alpha=0.85, zorder=1, label="Pareto front")

    # Scatter: each point is one (method, T_seg, budget) aggregate
    for (x, y, method, tseg), on in zip(pts, on_front):
        ax.scatter(
            x, y,
            color=_color(method),
            marker=tseg_mk[tseg],
            s=tseg_sz[tseg] + (50 if on else 0),
            edgecolors="black" if on else "none",
            linewidths=1.8 if on else 0,
            alpha=1.0, zorder=3,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Training Time (s)", fontsize=_FS_LABEL, labelpad=6)
    ax.set_ylabel("$W_2$ distance", fontsize=_FS_LABEL, labelpad=6)
    ax.tick_params(labelsize=_FS_TICK)
    ax.grid(True, alpha=0.25, linewidth=0.8)

    method_handles = [
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor=_color(m), markeredgecolor="none",
               markersize=11, label=_label(m))
        for m in methods
    ]
    tseg_handles = [
        Line2D([0], [0], marker=tseg_mk[t], linestyle="none",
               color="0.25", markersize=9,
               label=f"$T_{{\\mathrm{{seg}}}}$ = {t}")
        for t in tsegs
    ]
    pareto_handle = Line2D(
        [0], [0], color="0.50", linestyle="--", linewidth=2.2,
        label="Pareto front"
    )
    ax.legend(
        handles=method_handles + tseg_handles + [pareto_handle],
        fontsize=_FS_LEGEND, frameon=True,
        loc="upper right", ncol=2,
    )
    ax.set_title(
        f"Accuracy vs Training Time  ·  params = {params}",
        fontsize=_FS_TITLE, fontweight="bold", pad=10,
    )
    fig.tight_layout()
    _save(fig, output_path)


# ──────────────────────────────────────────────────────────────────────────────
# PLOT 5 — W₂ scaling with parameter count  (NPSE degradation)
# ──────────────────────────────────────────────────────────────────────────────

def plot_defense_w2_scaling(
    agg_rows: list[dict[str, Any]],
    output_path: Path,
    budget_filter: int | None = None,
    tseg_filter: int = 1000,
) -> None:
    """Grouped bar chart: normalised RMSE per method × parameter configuration.

    Uses average-normalised RMSE (each parameter's RMSE divided by its prior
    range) so that all 4 parameter configs are on the same y-axis.  Falls back
    to W₂ if per-parameter metrics are unavailable.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_params  = sorted({r["params"] for r in agg_rows})
    ord_params  = [p for p in _PARAM_ORDER if p in all_params] + \
                  [p for p in all_params if p not in _PARAM_ORDER]

    if budget_filter is None:
        budgets = sorted({
            int(r["requested_budget_steps"])
            for r in agg_rows
            if r.get("requested_budget_steps")
        })
        budget_filter = budgets[-1] if budgets else 0

    methods  = _sorted_methods([r["method"] for r in agg_rows])
    n_params = len(ord_params)
    n_meth   = len(methods)
    bar_w    = 0.70 / n_meth
    x_pos    = np.arange(n_params)

    def _norm_metric(row: dict[str, Any], params: str) -> float | None:
        """Return avg-normalised RMSE if available, else W₂."""
        active = [p.strip() for p in params.split(",")]
        vals   = [_get_mean(row, f"{p}_rmse_norm") for p in active]
        vals   = [v for v in vals if v is not None]
        if vals:
            return float(np.mean(vals))
        return _get_mean(row, "w2_mean")

    fig, ax = plt.subplots(figsize=(_W_WIDE, _H_ROW))

    for midx, method in enumerate(methods):
        ys, es = [], []
        for params in ord_params:
            cands = [
                r for r in agg_rows
                if r["params"] == params
                and r["method"] == method
                and int(r["T_seg"]) == tseg_filter
                and int(r.get("requested_budget_steps") or 0) == budget_filter
            ]
            if cands:
                vals = [_norm_metric(c, params) for c in cands]
                vals = [v for v in vals if v is not None]
            else:
                vals = []
            ys.append(float(np.mean(vals)) if vals else float("nan"))
            es.append(float(np.std(vals))  if len(vals) > 1 else 0.0)

        offsets = x_pos + (midx - n_meth / 2 + 0.5) * bar_w
        ax.bar(
            offsets, ys, bar_w,
            yerr=es,
            label=_label(method),
            color=_color(method),
            capsize=5,
            error_kw={"linewidth": 2.0},
            alpha=0.86, zorder=3,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [_PARAM_LABEL.get(p, p) for p in ord_params],
        fontsize=_FS_TICK, fontweight="semibold",
    )
    ax.set_ylabel(
        "Avg. normalised RMSE  (lower = better)",
        fontsize=_FS_LABEL, labelpad=6,
    )
    ax.tick_params(axis="y", labelsize=_FS_TICK)
    ax.legend(fontsize=_FS_LEGEND, frameon=True, loc="upper left")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)
    ax.set_title(
        f"Inference quality by parameter count"
        f"  ·  Budget = {_budget_label(budget_filter)}"
        f"  ·  $T_{{\\mathrm{{seg}}}}$ = {tseg_filter}",
        fontsize=_FS_TITLE, fontweight="bold", pad=10,
    )
    fig.tight_layout()
    _save(fig, output_path)


# ──────────────────────────────────────────────────────────────────────────────
# PLOT 6 — Real-data PPC RMSE bar chart  (the rankings invert)
# ──────────────────────────────────────────────────────────────────────────────

def plot_defense_real_ppc_rmse(
    agg_rows: list[dict[str, Any]],
    params: str,
    output_path: Path,
    budget_filter: int | None = None,
    tseg_filter: int = 1000,
) -> None:
    """Horizontal bar chart: mean real-data PPC RMSE per method.

    Methods are sorted ascending (best at top).  Error bars = std across seeds.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    param_rows = [r for r in agg_rows
                  if r["params"] == params and int(r["T_seg"]) == tseg_filter]
    if not param_rows:
        print(f"  [skip] no data for real PPC RMSE: params={params} T_seg={tseg_filter}")
        return

    if budget_filter is None:
        budgets = sorted({
            int(r["requested_budget_steps"])
            for r in param_rows
            if r.get("requested_budget_steps")
        })
        budget_filter = budgets[-1] if budgets else 0

    best_rows = [r for r in param_rows
                 if int(r.get("requested_budget_steps") or 0) == budget_filter]
    if not best_rows:
        print(f"  [skip] no rows at budget={budget_filter} for real PPC: params={params}")
        return

    methods = _sorted_methods([r["method"] for r in best_rows])
    data: list[tuple[str, float, float]] = []
    for method in methods:
        row = next((r for r in best_rows if r["method"] == method), None)
        v   = _get_mean(row, "real_ppc_rmse_mean") if row else None
        e   = _get_std( row, "real_ppc_rmse_mean") if row else 0.0
        if v is not None:
            data.append((method, v, e))

    if not data:
        print(f"  [warn] real_ppc_rmse_mean not available for params={params} — "
              "ensure real-data evaluation was run with this exp_prefix.")
        return

    # Sort ascending so the lowest RMSE method is at the top
    data.sort(key=lambda t: t[1])
    meth_sorted, ys, es = zip(*data)

    fig_h = max(3.5, 1.1 * len(meth_sorted) + 1.8)
    fig, ax = plt.subplots(figsize=(_W_NARROW * 0.85, fig_h))

    y_pos = np.arange(len(meth_sorted))
    bars  = ax.barh(
        y_pos, ys,
        xerr=es,
        color=[_color(m) for m in meth_sorted],
        capsize=6,
        error_kw={"linewidth": 2.0, "ecolor": "0.3"},
        height=0.52,
        alpha=0.88, zorder=3,
    )
    # Value annotations
    x_pad = max(es) * 0.15 if max(es) > 0 else 0.05 * max(ys)
    for bar, v in zip(bars, ys):
        ax.text(
            v + x_pad,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.1f}",
            va="center", ha="left",
            fontsize=_FS_TICK + 1, fontweight="semibold",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels([_label(m) for m in meth_sorted],
                       fontsize=_FS_TICK + 2, fontweight="semibold")
    ax.set_xlabel(
        "Mean real-data PPC RMSE  (lower = better)",
        fontsize=_FS_LABEL, labelpad=6,
    )
    ax.tick_params(axis="x", labelsize=_FS_TICK)
    ax.grid(True, axis="x", alpha=0.25, linewidth=0.8)
    ax.set_title(
        f"Real-data rankings  ·  params = {params}"
        f"  ·  Budget = {_budget_label(budget_filter)}",
        fontsize=_FS_TITLE, fontweight="bold", pad=10,
    )
    # Note about ranking inversion
    ax.text(
        0.98, 0.02,
        "Simulation rank → best: NPE   Real-data rank → best: FNPSE",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=_FS_TICK - 1, color="0.40", style="italic",
    )
    fig.tight_layout()
    _save(fig, output_path)


# ──────────────────────────────────────────────────────────────────────────────
# Shared save helper
# ──────────────────────────────────────────────────────────────────────────────

def _save(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate defense presentation figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--experiments-root", default="experiments",
                   help="Root directory containing experiment subdirectories.")
    p.add_argument("--exp-prefix", required=True,
                   help="Prefix shared by all benchmark experiment names.")
    p.add_argument("--output-dir", default=None,
                   help="Output dir. Defaults to <experiments-root>/<exp-prefix>/plots_defense/")
    p.add_argument("--primary-params", default="mu",
                   help="Param config for calibration and real-PPC-RMSE plots.")
    p.add_argument("--budget", type=int, default=None,
                   help="Budget (steps) to use for heatmap/bar plots. "
                        "Defaults to largest available.")
    p.add_argument("--tseg", type=int, default=1000,
                   help="T_seg for single-panel plots.")
    p.add_argument("--w2-tsegs", type=int, nargs="+", default=[1000, 2000, 3000],
                   help="T_seg values to show as columns in the W₂ plot.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    experiments_root = Path(args.experiments_root)
    output_dir = (
        Path(args.output_dir) if args.output_dir
        else experiments_root / args.exp_prefix / "plots_defense"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    sep = "=" * 62
    print(sep)
    print("  Defense figure generator")
    print(f"  experiments_root : {experiments_root}")
    print(f"  exp_prefix       : {args.exp_prefix}")
    print(f"  output_dir       : {output_dir}")
    print(sep)

    print("Loading completed runs …")
    runs = _load_completed_runs(experiments_root, args.exp_prefix)
    print(f"  {len(runs)} deduplicated runs found")

    raw_rows = _build_raw_rows(runs)
    print(f"  {len(raw_rows)} raw rows built")

    agg_rows = _aggregate_rows(raw_rows)
    print(f"  {len(agg_rows)} aggregate rows built")

    all_params = sorted({r["params"] for r in agg_rows})
    print(f"  parameter configs: {all_params}")
    print()

    print("Generating per-parameter figures …")
    for params in all_params:
        s = _slugify(params)
        plot_defense_w2(
            agg_rows, params,
            output_dir / f"defense_w2_{s}.png",
            show_tsegs=tuple(args.w2_tsegs),
        )
        plot_defense_calibration(
            agg_rows, params,
            output_dir / f"defense_calibration_{s}.png",
            tseg_filter=args.tseg,
        )
        plot_defense_metric_heatmap(
            agg_rows, params,
            output_dir / f"defense_metric_heatmap_{s}.png",
            budget_filter=args.budget,
            tseg_filter=args.tseg,
        )
        plot_defense_pareto(
            agg_rows, params,
            output_dir / f"defense_pareto_{s}.png",
        )

    print("\nGenerating cross-parameter figures …")
    plot_defense_w2_scaling(
        agg_rows,
        output_dir / "defense_w2_scaling.png",
        budget_filter=args.budget,
        tseg_filter=args.tseg,
    )

    print("\nGenerating real-data figures …")
    plot_defense_real_ppc_rmse(
        agg_rows, args.primary_params,
        output_dir / f"defense_real_ppc_rmse_{_slugify(args.primary_params)}.png",
        budget_filter=args.budget,
        tseg_filter=args.tseg,
    )

    print()
    print(sep)
    print(f"  All figures written to:  {output_dir}")
    print(sep)


if __name__ == "__main__":
    main()
