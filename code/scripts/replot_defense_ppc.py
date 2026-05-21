"""
Regenerate defense-quality PPC time-series figures from saved .npz arrays.

The .npz files (ppc_real_data.npz) are produced by unified_experiment.py
AFTER the code change that added np.savez_compressed for y_real / y_ppc.

Usage (after re-running the real-eval SLURM jobs):
    cd /bigwork/nhkbarit/thesis-code/code
    python scripts/replot_defense_ppc.py \
        --experiments-root ../experiments \
        --output-dir ../presentations/defense/figures

Or from Windows / local:
    python code/scripts/replot_defense_ppc.py \
        --experiments-root experiments \
        --output-dir presentations/defense/figures
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Observation channel labels with units ─────────────────────────────────────
OBS_LABELS_WITH_UNITS = [
    "Yaw rate [rad/s]",
    r"$v_x$  [m/s]",
    r"$v_y$  [m/s]",
    r"$a_x$  [m/s²]",
    r"$a_y$  [m/s²]",
    r"$\omega_\mathrm{FL}$  [rad/s]",
    r"$\omega_\mathrm{FR}$  [rad/s]",
    r"$\omega_\mathrm{RL}$  [rad/s]",
    r"$\omega_\mathrm{RR}$  [rad/s]",
]

# ── Defense color palette (tab10 positions, matching all other defense figures) ─
METHOD_COLORS = {
    "npe":      "#ff7f0e",   # orange
    "npse":     "#2ca02c",   # green
    "fnpe":     "#1f77b4",   # blue
    "simformer":"#d62728",   # red
}

METHOD_DISPLAY = {
    "npe":       "NPE",
    "npse":      "NPSE",
    "fnpe":      "FNPSE",
    "simformer": "Simformer",
}

REAL_COLOR = "#1b9e77"   # teal — same as in standard PPC plots

# ── Default experiment directory names (μ-only, T=1000) ───────────────────────
# These are looked up inside <experiments_root> automatically.
# The script also scans for ANY directory whose name contains the method prefix
# and "real_eval" and "pmu_t1000", so it stays robust to re-runs.
METHOD_DIR_PATTERNS = {
    "npe":       "npe_real_eval_npe_mu_pmu_t1000",
    "npse":      "npse_real_eval_npse_mu_pmu_t1000",
    "fnpe":      "fnpe_real_eval_fnpe_mu_pmu_t1000",
    "simformer": "simformer_real_eval_simformer_mu_pmu_t1000",
}

# ── Global plot style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.facecolor":    "white",
    "figure.facecolor":  "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


def find_npz(experiments_root: Path, method: str) -> Path | None:
    """Return the newest ppc_real_data.npz matching the method pattern, or None."""
    pattern = METHOD_DIR_PATTERNS[method]
    candidates = sorted(
        experiments_root.glob(f"{pattern}*/ppc_real_data.npz"),
        key=lambda p: p.parent.name,   # sort by dirname → timestamp suffix last
    )
    if not candidates:
        return None
    return candidates[-1]   # newest run


def plot_ppc_defense(
    y_real: np.ndarray,
    y_ppc: np.ndarray,
    dt: float,
    method: str,
    out_path: Path,
    max_trajs: int = 60,
    ncols: int = 3,
    dpi: int = 300,
) -> None:
    """
    Defense-quality PPC figure for one method.

    Args:
        y_real : (T, D)
        y_ppc  : (K, T, D)
        dt     : timestep in seconds
        method : one of npe / npse / fnpe / simformer
        out_path: destination PNG
        max_trajs: max PPC sample trajectories to draw
        ncols  : subplot columns
        dpi    : output resolution
    """
    assert y_real.ndim == 2, f"y_real must be (T,D), got {y_real.shape}"
    assert y_ppc.ndim  == 3, f"y_ppc must be (K,T,D), got {y_ppc.shape}"

    K, T, D = y_ppc.shape
    assert y_real.shape == (T, D), (
        f"y_real shape {y_real.shape} doesn't match y_ppc (T,D)=({T},{D})"
    )

    t  = np.arange(T) * dt
    n_dims = min(D, len(OBS_LABELS_WITH_UNITS))

    color    = METHOD_COLORS.get(method, "#333333")
    disp     = METHOD_DISPLAY.get(method, method.upper())

    # subsample PPC trajectories
    idxs = np.random.choice(K, size=min(K, max_trajs), replace=False)

    # PPC median
    y_med = np.median(y_ppc, axis=0)   # (T, D)

    ncols = max(1, ncols)
    nrows = int(np.ceil(n_dims / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.2 * ncols, 3.0 * nrows),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.array(axes).reshape(-1)

    fig.suptitle(
        f"{disp} — Posterior Predictive Check on Real Data   "
        f"(K = {K} samples)",
        fontsize=14, fontweight="bold",
    )

    for d in range(n_dims):
        ax = axes[d]
        row = d // ncols

        # Translucent sample trajectories in method color
        for k in idxs:
            ax.plot(t, y_ppc[k, :, d], alpha=0.05, lw=0.6, color=color)

        # Shaded IQR band
        q25 = np.percentile(y_ppc[:, :, d], 25, axis=0)
        q75 = np.percentile(y_ppc[:, :, d], 75, axis=0)
        ax.fill_between(t, q25, q75, alpha=0.18, color=color)

        # Median PPC
        ax.plot(t, y_med[:, d], color=color, lw=2.2,
                label=f"{disp} median")

        # Real measurement
        ax.plot(t, y_real[:, d], color=REAL_COLOR, lw=1.6, alpha=0.95,
                label="Real")

        # Y-axis limits: focus on real range with a small margin
        y_lo = float(np.min(y_real[:, d]))
        y_hi = float(np.max(y_real[:, d]))
        span = max(y_hi - y_lo, 1e-6)
        ref  = max(1.0, abs(y_lo), abs(y_hi))
        margin = max(span * 0.12, 0.05 * ref)
        ax.set_ylim(y_lo - margin, y_hi + margin)

        ax.set_ylabel(OBS_LABELS_WITH_UNITS[d], fontsize=12)
        ax.tick_params(labelsize=10)

        if d == 0:
            ax.legend(loc="upper right", fontsize=10, frameon=False)

        if row == nrows - 1:
            ax.set_xlabel("Time [s]", fontsize=12)

    for ax in axes[n_dims:]:
        ax.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"[replot] Saved {out_path}")


# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Regenerate defense PPC figures")
    p.add_argument(
        "--experiments-root", type=Path,
        default=Path("experiments"),
        help="Root directory containing real-eval experiment folders",
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=Path("presentations/defense/figures"),
        help="Where to write ppc_real_{method}.png files",
    )
    p.add_argument(
        "--methods", nargs="+",
        default=["npe", "npse", "fnpe", "simformer"],
        choices=["npe", "npse", "fnpe", "simformer"],
        help="Which methods to regenerate",
    )
    p.add_argument(
        "--dpi", type=int, default=300,
        help="Output figure resolution",
    )
    p.add_argument(
        "--max-trajs", type=int, default=60,
        help="Max PPC sample trajectories to draw per plot",
    )
    args = p.parse_args()

    exp_root = args.experiments_root
    out_dir  = args.output_dir
    missing  = []

    for method in args.methods:
        npz_path = find_npz(exp_root, method)
        if npz_path is None:
            print(f"[WARN] No ppc_real_data.npz found for {method} in {exp_root}")
            print(f"       Pattern searched: {METHOD_DIR_PATTERNS[method]}*/ppc_real_data.npz")
            missing.append(method)
            continue

        print(f"[{method.upper()}] Loading {npz_path}")
        data   = np.load(npz_path)
        y_real = data["y_real"]   # (T, D)
        y_ppc  = data["y_ppc"]    # (K, T, D)
        dt     = float(data["dt"])

        out_path = out_dir / f"ppc_real_{method}.png"
        plot_ppc_defense(
            y_real=y_real,
            y_ppc=y_ppc,
            dt=dt,
            method=method,
            out_path=out_path,
            max_trajs=args.max_trajs,
            dpi=args.dpi,
        )

    if missing:
        print()
        print("=" * 60)
        print("MISSING NPZ FILES — re-run the SLURM eval jobs first:")
        print()
        print("  cd /bigwork/nhkbarit/thesis-code/code")
        print("  sbatch --array=0-11 scripts/run_real_data_eval.sh")
        print()
        print("After the jobs complete, run this script again to")
        print("collect the updated figures.")
        print("=" * 60)
        sys.exit(1)

    print()
    print("All defense PPC figures saved to:", out_dir)


if __name__ == "__main__":
    main()
