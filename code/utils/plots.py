# sbi_vehicle/plots.py

from pathlib import Path
from typing import Sequence, Dict, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

from sbi.analysis.plot import sbc_rank_plot, pp_plot_lc2st


def _ensure_dir(path: Path):
    """Create parent directory for a path, if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# 1) Training curves (loss vs epoch)
# ---------------------------------------------------------------------


def plot_training_curves(
    summary: Dict[str, Sequence[float]],
    out_path: Path,
    title: str = "Training & Validation Loss",
):
    """
    Plot training and validation loss curves from sbi inference.summary.

    Args:
        summary: dict from inference.summary containing at least
                 'training_loss' and 'validation_loss'.
        out_path: where to save the PNG (e.g. exp_dir/'figures/train_loss.png').
        title: plot title.
    """
    train_loss = np.asarray(summary.get("training_loss", []), dtype=float)
    val_loss = np.asarray(summary.get("validation_loss", []), dtype=float)

    if train_loss.size == 0:
        print("[plot_training_curves] No training_loss found in summary; skipping.")
        return

    epochs = np.arange(1, len(train_loss) + 1)

    _ensure_dir(out_path)
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_loss, label="Training loss")
    if val_loss.size == train_loss.size:
        plt.plot(epochs, val_loss, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[plots] Saved training curves to {out_path}")


# ---------------------------------------------------------------------
# 2) SBC rank histogram
# ---------------------------------------------------------------------


def plot_sbc_rank_hist(
    ranks: torch.Tensor,
    num_posterior_samples: int,
    out_path: Path,
    title: str = "SBC rank histogram",
):
    """
    Plot SBC rank histogram using sbi's sbc_rank_plot and save to file.

    Args:
        ranks: torch.Tensor of ranks from run_sbc (shape (num_sbc_samples, d)).
        num_posterior_samples: number of posterior samples used per SBC datum.
        out_path: where to save the PNG.
    """
    _ensure_dir(out_path)
    plt.figure(figsize=(7, 5))

    # Calculate num_bins to avoid edge case with few samples
    num_sbc_samples = ranks.shape[0]
    num_bins = max(10, num_sbc_samples // 10)  # At least 10 bins

    sbc_rank_plot(ranks, num_posterior_samples, plot_type="hist", num_bins=num_bins)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[plots] Saved SBC rank histogram to {out_path}")


# ---------------------------------------------------------------------
# 2b) LC2ST-NF diagnostics
# ---------------------------------------------------------------------


def plot_lc2st_histogram(
    T_null: np.ndarray,
    T_data: float,
    alpha: float,
    out_path: Path,
    *,
    title: str = "LC2ST-NF statistic",
    p_value: float = None,
):
    """
    Plot histogram of LC2ST-NF statistics under H0 with observed statistic and CI.
    """
    T_null = np.asarray(T_null, dtype=float).ravel()
    q_low, q_high = np.quantile(T_null, [0.0, 1.0 - alpha])

    _ensure_dir(out_path)
    plt.figure(figsize=(6, 4))
    plt.hist(T_null, bins=50, density=True, alpha=0.6, label="Null")
    plt.axvline(T_data, color="red", label="Observed")
    plt.axvline(q_low, color="black", linestyle="--", label=f"{int((1-alpha)*100)}% CI")
    plt.axvline(q_high, color="black", linestyle="--")
    subtitle = "" if p_value is None else f" (p={p_value:.3f})"
    plt.title(title + subtitle)
    plt.xlabel("Test statistic")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.legend()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[plots] Saved LC2ST-NF histogram to {out_path}")


def plot_lc2st_pp_plot(
    probs_data: np.ndarray,
    probs_null: np.ndarray,
    alpha: float,
    out_path: Path,
    *,
    title: str = "LC2ST-NF PP-plot",
):
    """
    Wrapper around sbi's pp_plot_lc2st for convenience.
    """
    _ensure_dir(out_path)
    plt.figure(figsize=(5, 4))
    pp_plot_lc2st(
        probs=[np.asarray(probs_data)],
        probs_null=np.asarray(probs_null),
        conf_alpha=alpha,
        labels=["Classifier probs (observed)"],
        colors=["red"],
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[plots] Saved LC2ST-NF PP-plot to {out_path}")


# ---------------------------------------------------------------------
# 3) Posterior predictive trajectories (time series)
# ---------------------------------------------------------------------


def plot_ppc_trajectories(
    y_real: np.ndarray,
    y_ppc: np.ndarray,
    obs_labels: Sequence[str],
    dt: float,
    out_path: Path,
    max_trajs: int = 60,
    max_dims: Optional[int] = None,
    ncols: int = 3,
    title: str = "Posterior Predictive Check (time series)",
):
    """
    Plot PPC with:
      - transparent individual sample trajectories
      - median of PPC samples
      - real trajectory (half linewidth)
      - legend showing #samples

    Args:
        y_real: (T, D)
        y_ppc:  (K, T, D)
        obs_labels: labels for each dim
        dt: timestep
        out_path: output file
    """
    assert y_real.ndim == 2, f"Expected (T,D), got {y_real.shape}"
    assert y_ppc.ndim == 3, f"Expected (K,T,D), got {y_ppc.shape}"

    K, T, D = y_ppc.shape
    assert y_real.shape == (T, D)

    # Time axis
    t = np.arange(T) * dt

    # Limit dims and trajectories
    if max_dims is None:
        max_dims = D
    max_dims = min(max_dims, D)

    num_trajs = min(K, max_trajs)
    idxs = (
        np.random.choice(K, size=num_trajs, replace=False)
        if K > num_trajs
        else np.arange(K)
    )

    # Compute median trajectory across PPC samples
    y_median = np.median(y_ppc, axis=0)  # (T, D)

    # Figure
    ncols = max(1, ncols)
    nrows = int(np.ceil(max_dims / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.5 * ncols, 2.5 * nrows),
        sharex=True,
    )
    axes = np.array(axes).reshape(-1)

    # Enhanced title: show total number of samples
    fig.suptitle(f"{title}   (PPC samples: {K})", fontsize=12, y=0.98)

    for d in range(max_dims):
        ax = axes[d]
        row_idx = d // ncols

        # Plot transparent PPC sample trajectories
        for k in idxs:
            ax.plot(
                t,
                y_ppc[k, :, d],
                alpha=0.06,
                lw=0.7,
                color="C0",
            )

        # Median trajectory
        ax.plot(
            t,
            y_median[:, d],
            color="C1",
            lw=2.2,
            label="PPC median",
        )

        # Real trajectory — half thickness from earlier lw=1.8 → lw=0.9
        ax.plot(
            t,
            y_real[:, d],
            color="black",
            lw=0.9,
            label="Real",
        )

        label = obs_labels[d] if d < len(obs_labels) else f"obs_{d}"
        ax.set_ylabel(label)

        if d == 0:
            # Add single legend to first subplot only
            ax.legend(loc="upper right")

        if row_idx == nrows - 1:
            ax.set_xlabel("time [s]")

    for ax in axes[max_dims:]:
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _ensure_dir(out_path)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[plots] Saved PPC trajectories to {out_path}")


# ---------------------------------------------------------------------
# 4) States + controls comparison (sim vs real)
# ---------------------------------------------------------------------


def plot_states_and_controls_comparison(
    states_sim: np.ndarray,
    ctrls_sim: Dict[str, np.ndarray],
    dt: float,
    out_path: Path,
    *,
    states_real: Optional[np.ndarray] = None,
    ctrls_real: Optional[Dict[str, np.ndarray]] = None,
    title: str = "State and Control Comparison (Sim vs Real)",
    ncols: int = 3,
):
    """
    Compact grid plot for all state variables and controls (Sim vs Real).

    Uses physical state names instead of x0-x9.

    Args:
        states_sim: (T, 10) simulated states.
        ctrls_sim:  dict of simulated controls at model rate.
        dt:         time step.
        out_path:   PNG path to save figure.
        states_real: (T, 10) real states (if available).
        ctrls_real:  dict of real controls at model rate (if available).
    """
    T = states_sim.shape[0]
    t = np.arange(T) * dt

    state_labels = [
        "geo_pos_x [m]",
        "geo_pos_y [m]",
        "yaw [rad]",
        "yaw_rate [rad/s]",
        "veh_vel_x [m/s]",
        "veh_vel_y [m/s]",
        "tire_rate_FL [rad/s]",
        "tire_rate_FR [rad/s]",
        "tire_rate_RL [rad/s]",
        "tire_rate_RR [rad/s]",
    ]

    n_states = states_sim.shape[1]
    n_ctrls = 4
    total_plots = n_states + n_ctrls

    ncols = max(1, ncols)
    nrows = int(np.ceil(total_plots / ncols))

    _ensure_dir(out_path)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 2.4 * nrows),
        sharex=False,
    )
    axes = np.array(axes).reshape(-1)
    fig.suptitle(title, fontsize=14, y=1.02)

    # States
    for i in range(n_states):
        ax = axes[i]
        ax.plot(t, states_sim[:, i], label="Sim", lw=1.5)
        if states_real is not None:
            ax.plot(t, states_real[:, i], label="Real", lw=1.0, alpha=0.8)
        lbl = state_labels[i] if i < len(state_labels) else f"state_{i}"
        ax.set_title(lbl, fontsize=9)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="best", fontsize=8)

    # Controls
    ctrl_keys = ["engine_torque", "break_torque", "steer_ang", "gear_transmission"]
    ctrl_labels = ["Engine Torque [Nm]", "Brake [arb]", "Steer Angle [deg]", "Gear"]

    for j, (k, label) in enumerate(zip(ctrl_keys, ctrl_labels)):
        idx = n_states + j
        if idx >= len(axes):
            break
        ax = axes[idx]
        c_sim = np.asarray(ctrls_sim[k])
        if "steer" in k:
            c_sim = np.rad2deg(c_sim)
        ax.plot(t, c_sim, label="Sim", lw=1.5)

        if ctrls_real is not None:
            c_real = np.asarray(ctrls_real[k])
            if "steer" in k:
                c_real = np.rad2deg(c_real)
            ax.plot(t, c_real, label="Real", lw=1.0, alpha=0.8)

        ax.set_title(label, fontsize=9)
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(loc="best", fontsize=8)

    for ax in axes[total_plots:]:
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[plots] Saved state/control comparison to {out_path}")


# ---------------------------------------------------------------------
# 5) 1D histograms of observation channels (train vs real)
# ---------------------------------------------------------------------


def plot_obs_1d_hist_custom(
    train_obs: np.ndarray,
    real_obs: np.ndarray,
    obs_labels: Sequence[str],
    custom_ranges: Dict[str, tuple],
    out_path: Path,
    bins: int = 80,
):
    """
    Plot 1D histograms for each observation dimension comparing:

        - train_obs: (N_train, D)
        - real_obs:  (N_real, D)

    This is the modular version of your notebook's plot_obs_1d_hist_custom.

    Args:
        train_obs: (N1, D)
        real_obs:  (N2, D)
        obs_labels: list of length D with labels.
        custom_ranges: dict mapping label -> (xmin, xmax) for zoomed x-axis.
        out_path: PNG path to save figure.
        bins: histogram bins.
    """
    D = train_obs.shape[1]
    cols = 3
    rows = int(np.ceil(D / cols))

    _ensure_dir(out_path)
    plt.figure(figsize=(4 * cols, 3 * rows))

    for d in range(D):
        ax = plt.subplot(rows, cols, d + 1)
        label = obs_labels[d] if d < len(obs_labels) else f"obs_{d}"

        h_train = ax.hist(
            train_obs[:, d],
            bins=bins,
            density=True,
            alpha=0.6,
            label="Train",
        )
        h_real = ax.hist(
            real_obs[:, d],
            bins=bins,
            density=True,
            alpha=0.6,
            histtype="step",
            linewidth=1.2,
            label="Real",
        )

        ax.set_title(label)
        ax.grid(True, alpha=0.2)

        # Custom ranges
        if label in custom_ranges:
            xmin, xmax = custom_ranges[label]
            ax.set_xlim(xmin, xmax)

        # Special handling for yaw so blue is visible
        if d == 0:
            max_train = float(np.max(h_train[0]))
            ax.set_ylim(0, max_train * 1.3)
            ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[plots] Saved obs histograms to {out_path}")


def plot_prior_posterior_1d(
    prior_1d: np.ndarray,
    post_1d: np.ndarray,
    theta_ref: Optional[float],
    param_name: str,
    out_path: Path,
    bins: int = 100,
) -> None:
    """
    Plot 1D prior and posterior marginals for a single parameter, with an
    optional vertical line at a reference value theta_ref.

    Args:
        prior_1d:  (N_prior,) array of prior samples for this parameter.
        post_1d:   (N_post,) array of posterior samples for this parameter.
        theta_ref: scalar reference value (e.g. true θ); if None, no line is drawn.
        param_name: label for the parameter (e.g. "mu").
        out_path: output path for the PNG.
        bins: number of histogram bins.
    """
    prior_1d = np.asarray(prior_1d, dtype=np.float32)
    post_1d = np.asarray(post_1d, dtype=np.float32)

    _ensure_dir(out_path)

    plt.figure(figsize=(7, 4))

    # Prior in the background
    plt.hist(
        prior_1d,
        bins=bins,
        density=True,
        histtype="stepfilled",
        alpha=0.3,
        label="Prior",
    )

    # Posterior in the foreground
    plt.hist(
        post_1d,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2.0,
        label="Posterior",
    )

    # Optional vertical reference line
    if theta_ref is not None:
        plt.axvline(
            float(theta_ref),
            color="black",
            linestyle="--",
            linewidth=2.0,
            label="Reference θ",
        )

    plt.xlabel(param_name)
    plt.ylabel("Density")
    plt.title(f"Prior vs posterior for {param_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[plots] Saved 1D prior/posterior plot to {out_path}")
