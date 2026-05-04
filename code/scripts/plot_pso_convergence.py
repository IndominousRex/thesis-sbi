"""
Plot PSO convergence history from pso_global_log_optimization_results.json.
Saves a publication-quality figure to text (Latex)/Figures/pso/pso_convergence.pdf

Run from the repository root:
    python code/scripts/plot_pso_convergence.py
"""

import json
import pathlib
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
JSON_PATH = REPO_ROOT / "code" / "notebooks" / "experiments" / "pso_global_log_optimization_results.json"
OUT_DIR = REPO_ROOT / "text (Latex)" / "Figures" / "pso"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "pso_convergence.pdf"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with open(JSON_PATH, "r") as f:
    results = json.load(f)

history = np.array(results["convergence_history"])  # length = num_iterations
iterations = np.arange(1, len(history) + 1)

# Key milestones from the text
milestones = {
    1:    history[0],
    750:  history[749],
    1500: history[1499],
    3000: history[2999],
}
topology_switch = int(len(history) * results.get("topology_switch_frac", 0.70))

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

ANNOT_KW = dict(fontsize=9, color="#111111", fontweight="bold")
ARROW_KW = dict(arrowstyle="->", color="#555555", lw=0.9)
SCATTER_KW = dict(s=50, color="#f4a582", zorder=5, edgecolors="#c0502a", linewidths=0.6)

fig, (ax_main, ax_right) = plt.subplots(
    1, 2,
    figsize=(10, 3.8),
    gridspec_kw={"width_ratios": [2, 1]},
)

# ---- Main plot: full convergence on log-y --------------------------------
ax_main.semilogy(iterations, history, color="#2166ac", linewidth=0.9, alpha=0.85)

# Topology switch line
ax_main.axvline(topology_switch, color="#d6604d", linestyle="--", linewidth=1.0,
                label=f"Topology switch (iter {topology_switch})")

# Set y-limits with headroom so iter-1 annotation stays inside
ax_main.set_ylim(history[-1] * 0.997, history[0] * 1.025)

# Milestone scatter + annotations
annot_positions = {
    1:    (80,   history[0]    * 1.016),   # slightly right & above
    750:  (810,  history[749]  * 1.016),
    1500: (1560, history[1499] * 1.016),
    3000: (2770, history[2999] * 1.016),   # left of the topology switch line
}
for it, val in milestones.items():
    ax_main.scatter(it, val, **SCATTER_KW)
    xt, yt = annot_positions[it]
    ax_main.annotate(
        f"{val:.3f}",
        xy=(it, val),
        xytext=(xt, yt),
        arrowprops=ARROW_KW,
        **ANNOT_KW,
    )

ax_main.set_xlabel("PSO iteration")
ax_main.set_ylabel("NRMSE (log scale)")
ax_main.set_title("PSO convergence history")
ax_main.set_xlim(0, len(history))
ax_main.legend(loc="upper right", framealpha=0.8)
ax_main.yaxis.set_minor_formatter(ticker.NullFormatter())
ax_main.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.45, color="gray")

# ---- Right panel: early iterations 0–300 (linear-y) ---------------------
n_early = 300
early_iter = iterations[:n_early]
early_hist = history[:n_early]

ax_right.plot(early_iter, early_hist, color="#2166ac", linewidth=1.0)
ax_right.set_xlabel("Iteration (first 300)")
ax_right.set_ylabel("NRMSE")
ax_right.set_title("Early-phase detail")
ax_right.set_xlim(0, n_early)
ax_right.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.45, color="gray")

# y-limits with padding so bottom point is not flush with the axis
y_range = early_hist[0] - early_hist[-1]
ax_right.set_ylim(early_hist[-1] - y_range * 0.08, early_hist[0] + y_range * 0.08)

# Annotate iter 1 (top-left area, no arrow needed as it's at the start)
ax_right.scatter([1], [early_hist[0]], **SCATTER_KW)
ax_right.annotate(
    f"iter 1\n{early_hist[0]:.3f}",
    xy=(1, early_hist[0]),
    xytext=(25, early_hist[0] - y_range * 0.22),
    arrowprops=ARROW_KW,
    **ANNOT_KW,
)

# Annotate iter 300 - place well inside the plot above the point
ax_right.scatter([n_early], [early_hist[-1]], **SCATTER_KW)
ax_right.annotate(
    f"iter {n_early}\n{early_hist[-1]:.3f}",
    xy=(n_early, early_hist[-1]),
    xytext=(n_early - 155, early_hist[-1] + y_range * 0.48),
    arrowprops=ARROW_KW,
    **ANNOT_KW,
)

# ---------------------------------------------------------------------------
fig.tight_layout(pad=1.5)
fig.savefig(OUT_PATH, bbox_inches="tight")
print(f"Saved: {OUT_PATH}")
