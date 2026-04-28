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
# Plot
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

fig, (ax_main, ax_inset_host) = plt.subplots(
    1, 2,
    figsize=(10, 3.5),
    gridspec_kw={"width_ratios": [2, 1]},
)

# ---- Main plot: full convergence on log-y --------------------------------
ax_main.semilogy(iterations, history, color="#2166ac", linewidth=0.9, alpha=0.85)

# Topology switch
ax_main.axvline(topology_switch, color="#d6604d", linestyle="--", linewidth=1.0,
                label=f"Topology switch (iter {topology_switch})")

# Milestone annotations
for it, val in milestones.items():
    ax_main.scatter(it, val, s=40, color="#f4a582", zorder=5)
    offset_x = -120 if it == 3000 else 30
    offset_y = 1.04
    ax_main.annotate(
        f"  {val:.3f}",
        xy=(it, val),
        xytext=(it + offset_x, val * offset_y),
        fontsize=8,
        color="#555555",
    )

ax_main.set_xlabel("PSO iteration")
ax_main.set_ylabel("NRMSE (log scale)")
ax_main.set_title("PSO convergence history")
ax_main.set_xlim(0, len(history))
ax_main.legend(loc="upper right", framealpha=0.7)
ax_main.yaxis.set_minor_formatter(ticker.NullFormatter())

# ---- Inset: early iterations 0–300 (linear-y) ---------------------------
n_early = 300
ax_inset_host.plot(iterations[:n_early], history[:n_early], color="#2166ac", linewidth=1.0)
ax_inset_host.set_xlabel("Iteration (first 300)")
ax_inset_host.set_ylabel("NRMSE")
ax_inset_host.set_title("Early-phase detail")
ax_inset_host.set_xlim(0, n_early)

# Annotate iteration-1 start and the approx value at iter 300
ax_inset_host.scatter([1, n_early], [history[0], history[n_early - 1]], s=40,
                      color="#f4a582", zorder=5)
ax_inset_host.annotate(f"iter 1\n{history[0]:.3f}", xy=(1, history[0]),
                       xytext=(20, history[0] * 0.97), fontsize=8, color="#555555")
ax_inset_host.annotate(f"iter {n_early}\n{history[n_early-1]:.3f}",
                       xy=(n_early, history[n_early - 1]),
                       xytext=(n_early - 120, history[n_early - 1] * 0.96),
                       fontsize=8, color="#555555")

# ---------------------------------------------------------------------------
fig.tight_layout(pad=1.5)
fig.savefig(OUT_PATH, bbox_inches="tight")
print(f"Saved: {OUT_PATH}")
