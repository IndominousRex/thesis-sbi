"""
Generate a cover image for the thesis title page.

Shows the prior-to-posterior flow for the tire-road friction coefficient µ:
five panels arranged horizontally, each showing a density that interpolates
between a flat uniform prior and the sharp NPE posterior, evoking the diffusion
reversal process that underpins the score-based SBI methods in the thesis.

Prior:     Uniform(0.50, 1.50)
Posterior: N(µ* = 0.845, σ = 0.017)  -- consistent with NPE W2 ≈ 0.017

Output: text (Latex)/Figures/cover_image.pdf  (and .png at 300 DPI)

Run from repository root:
    python code/scripts/plot_cover_image.py
"""

import pathlib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
from scipy.stats import norm

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "text (Latex)" / "Figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PDF = OUT_DIR / "cover_image.pdf"
OUT_PNG = OUT_DIR / "cover_image.png"

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
MU_LOW, MU_HIGH = 0.50, 1.50          # prior support
MU_TRUE = 0.845                        # ground-truth used in thesis test case
POST_STD = 0.017                       # NPE posterior std (from W2 ≈ 0.017)
N_PTS = 2000                           # density evaluation points
MU = np.linspace(MU_LOW - 0.05, MU_HIGH + 0.05, N_PTS)

# ---------------------------------------------------------------------------
# Interpolation stages
# Mix between Uniform(MU_LOW, MU_HIGH) and N(MU_TRUE, POST_STD).
# alpha = fraction of Gaussian in the mixture; std shrinks linearly.
# ---------------------------------------------------------------------------
STAGES = [
    # (alpha, std, label)
    (0.00, 0.30, "prior\n$p(\\mu)$"),
    (0.25, 0.22, ""),
    (0.55, 0.12, ""),
    (0.82, 0.055, ""),
    (1.00, POST_STD, "posterior\n$p(\\mu\\mid\\mathbf{y})$"),
]
N_STAGES = len(STAGES)

def stage_density(alpha, std, mu_grid):
    """Mixture density for one stage."""
    uniform_val = 1.0 / (MU_HIGH - MU_LOW)
    prior = np.where((mu_grid >= MU_LOW) & (mu_grid <= MU_HIGH), uniform_val, 0.0)
    gaussian = norm.pdf(mu_grid, MU_TRUE, std)
    mix = (1 - alpha) * prior + alpha * gaussian
    return mix / (np.trapz(mix, mu_grid) + 1e-15)   # renormalise to 1

# ---------------------------------------------------------------------------
# Colour scheme
# ---------------------------------------------------------------------------
BG_COLOR   = "#0d1117"    # near-black background
AX_BG      = "#0d1117"
SPINE_COLOR = "#2e3b4e"
TEXT_COLOR  = "#e0e6f0"
TRUE_COLOR  = "#ff6b6b"   # soft red for the true-value line

# Gradient: prior = cool blue, posterior = warm amber/gold
CMAP_START = np.array(mcolors.to_rgb("#4a90d9"))   # blue
CMAP_END   = np.array(mcolors.to_rgb("#f5a623"))   # amber

def stage_color(i, n):
    t = i / (n - 1)
    return tuple((1 - t) * CMAP_START + t * CMAP_END)

# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------
fig_width  = 14.0   # inches — wide panoramic strip
fig_height = 2.8    # inches — slightly taller
fig = plt.figure(figsize=(fig_width, fig_height), facecolor=BG_COLOR)

# Axes: n_stages panels side by side, plus a thin arrow row below
# Use a GridSpec: top row = panels, bottom row = arrow
from matplotlib.gridspec import GridSpec
gs = GridSpec(
    2, N_STAGES,
    figure=fig,
    hspace=0.04,
    height_ratios=[8, 1],
    left=0.02, right=0.98,
    top=0.88, bottom=0.08,
    wspace=0.06,
)

axes = []
for col in range(N_STAGES):
    ax = fig.add_subplot(gs[0, col])
    axes.append(ax)

arrow_ax = fig.add_subplot(gs[1, :])

# ---------------------------------------------------------------------------
# Draw each stage
# ---------------------------------------------------------------------------
for i, (alpha, std, label) in enumerate(STAGES):
    ax = axes[i]
    ax.set_facecolor(AX_BG)

    density = stage_density(alpha, std, MU)
    color = stage_color(i, N_STAGES)

    d_max = density.max()

    # Clean fill: vertical gradient from opaque at base to transparent at peak
    # achieved by stacking many thin fill_between slices
    N_LAYERS = 80
    for k in range(N_LAYERS):
        lo = k / N_LAYERS
        hi = (k + 1) / N_LAYERS
        mask_lo = density >= lo * d_max
        mask_hi = density >= hi * d_max
        # fill the slice of the density between lo*d_max and hi*d_max
        y_lo = np.where(mask_lo, lo * d_max, density)
        y_hi = np.where(mask_hi, hi * d_max, density)
        layer_alpha = 0.08 + 0.55 * (1.0 - k / N_LAYERS)
        ax.fill_between(MU, y_lo, y_hi,
                        color=color, alpha=layer_alpha, linewidth=0)

    # Solid outline curve
    ax.plot(MU, density, color=color, linewidth=2.0, alpha=0.98)

    # True value line
    ax.axvline(MU_TRUE, color=TRUE_COLOR, linewidth=1.4, linestyle="--", alpha=0.85, zorder=5)

    # Axis formatting
    ax.set_xlim(MU[0], MU[-1])
    ax.set_ylim(0, d_max * 1.10)   # tight vertical ceiling — no dead space
    ax.set_xticks([0.5, MU_TRUE, 1.5])
    ax.set_xticklabels(["0.5", "", "1.5"], color=TEXT_COLOR, fontsize=7.0)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE_COLOR)
        spine.set_linewidth(0.6)
    ax.tick_params(colors=TEXT_COLOR, length=2.5, width=0.5)

    # µ* tick mark at bottom of true-value line (last panel only)
    if i == N_STAGES - 1:
        ax.set_xticks([0.5, MU_TRUE, 1.5])
        ax.set_xticklabels(["0.5", "$\\mu^*$", "1.5"], color=TEXT_COLOR, fontsize=7.0)
        ax.get_xticklabels()[1].set_color(TRUE_COLOR)

    # Panel label (only first and last)
    if label:
        lines = label.split("\n")
        title_text = "\n".join(lines)
        y_pos = d_max * 1.02
        x_pos = 0.5 * (MU[0] + MU[-1])
        ax.text(
            x_pos, y_pos, title_text,
            color=TEXT_COLOR, fontsize=8.5, fontweight="bold",
            ha="center", va="bottom", linespacing=1.2,
        )

# ---------------------------------------------------------------------------
# Arrow row: "simulation-based inference →"
# ---------------------------------------------------------------------------
arrow_ax.set_facecolor(BG_COLOR)
arrow_ax.set_xlim(0, 1)
arrow_ax.set_ylim(0, 1)
arrow_ax.axis("off")

# Arrow line
arrow_ax.annotate(
    "",
    xy=(0.95, 0.5), xytext=(0.05, 0.5),
    xycoords="axes fraction",
    arrowprops=dict(
        arrowstyle="-|>",
        color=SPINE_COLOR,
        lw=1.5,
        mutation_scale=12,
    ),
)

arrow_ax.text(
    0.5, 0.5,
    "simulation-based inference",
    color=SPINE_COLOR,
    fontsize=7.5,
    ha="center", va="center",
    style="italic",
    transform=arrow_ax.transAxes,
)

# ---------------------------------------------------------------------------
# Save (no suptitle — keep clean for title page)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
plt.close(fig)

print(f"Saved PDF : {OUT_PDF}")
print(f"Saved PNG : {OUT_PNG}")
