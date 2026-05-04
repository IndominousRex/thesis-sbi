"""
Generate a training data example figure for the thesis.

Simulates three short segments with different friction coefficients
(µ ∈ {0.6, 0.9, 1.4}) using synthetic sine-steer + braking control inputs,
then plots yaw rate, v_x, and one tire channel for each µ value.

Run from the repository root:
    python code/scripts/plot_training_data_example.py

Output:
    text (Latex)/Figures/training_data_example.pdf
"""

import sys
import os
import pathlib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Paths and environment
# ---------------------------------------------------------------------------
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

OUT_DIR = REPO_ROOT / "text (Latex)" / "Figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "training_data_example.pdf"

# ---------------------------------------------------------------------------
# Import simulation modules
# ---------------------------------------------------------------------------
import jax
import jax.numpy as jnp
import jax.lax as lax

from simulation.VehicleModel import (
    default_params,
    vehicle_RK4x,
    vehicle_fy,
)
from simulation.simulation import u_sine

# ---------------------------------------------------------------------------
# Simulation settings
# ---------------------------------------------------------------------------
T_SEG = 1000       # 10 seconds at 100 Hz
DT = 0.01
t = np.arange(T_SEG) * DT
T_SWITCH = T_SEG // 2   # 5 s: engine torque → braking

# All three values are within the inference prior [0.50, 1.50]
MU_VALUES = [0.6, 0.9, 1.4]
MU_LABELS = [r"$\mu = 0.6$", r"$\mu = 0.9$", r"$\mu = 1.4$"]
MU_COLORS = ["#d6604d", "#f4a582", "#2166ac"]

# Synthetic control: gentle slalom + moderate braking halfway through
steer = u_sine(T_SEG, amp=np.deg2rad(6.0), freq_hz=0.4, dt=DT)
engine_torque = np.where(np.arange(T_SEG) < T_SWITCH, 800.0, 0.0).astype(np.float32)
brake_pressure = np.where(np.arange(T_SEG) >= T_SWITCH, 30.0, 0.0).astype(np.float32)
gear = np.full(T_SEG, 3.0, dtype=np.float32)

controls_jax = {
    "steer_ang": jnp.array(steer, dtype=jnp.float32),
    "engine_torque": jnp.array(engine_torque, dtype=jnp.float32),
    "break_torque": jnp.array(brake_pressure, dtype=jnp.float32),
    "gear_transmission": jnp.array(gear, dtype=jnp.float32),
}

# ---------------------------------------------------------------------------
# Simulate for each µ value
# ---------------------------------------------------------------------------
STATE_DIM = 10


@jax.jit
def simulate_one(p_inf):
    state0 = jnp.zeros(STATE_DIM, dtype=jnp.float32)

    def body_fn(state, t_idx):
        u_t = {k: v[t_idx] for k, v in controls_jax.items()}
        y_t = vehicle_fy(state, u_t, p_inf, **default_params)
        state_next = vehicle_RK4x(state, u_t, p_inf, **default_params)
        return state_next, y_t

    _, y_seq = lax.scan(body_fn, state0, jnp.arange(T_SEG, dtype=jnp.int32))
    return y_seq


print("Simulating three µ values ...")
trajectories = {}
for mu in MU_VALUES:
    p_inf = jnp.array([mu, 0.270, 1720.0], dtype=jnp.float32)
    y = np.asarray(simulate_one(p_inf))
    trajectories[mu] = y
    print(f"  µ={mu:.1f}  yaw_rate range: [{y[:,0].min():.3f}, {y[:,0].max():.3f}] rad/s")

# ---------------------------------------------------------------------------
# Plot: 3 rows (channels) × 3 columns (µ values)
# Channels: yaw_rate (0), v_x (1), tire_FR (6)
# ---------------------------------------------------------------------------
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

CH_INDICES = [0, 1, 6]
CH_NAMES   = ["Yaw rate [rad/s]", r"$v_x$ [m/s]", "Tire speed FR [rad/s]"]

t_switch = t[T_SWITCH]   # 5.0 s

fig, axes = plt.subplots(3, 3, figsize=(11, 6.5), sharex=True, sharey="row")

for col, (mu, label, color) in enumerate(zip(MU_VALUES, MU_LABELS, MU_COLORS)):
    y = trajectories[mu]
    for row, (ch, ch_name) in enumerate(zip(CH_INDICES, CH_NAMES)):
        ax = axes[row, col]
        ax.plot(t, y[:, ch], color=color, lw=1.0)

        # Shaded braking phase
        ax.axvspan(t_switch, t[-1], alpha=0.08, color="#888888")

        # Vertical dashed line marking the phase transition
        ax.axvline(t_switch, color="#555555", lw=0.8, ls="--", zorder=3)

        ax.grid(True, alpha=0.3, lw=0.5)

        if row == 0:
            ax.set_title(label, fontsize=11, fontweight="bold")
        if col == 0:
            ax.set_ylabel(ch_name, fontsize=9)
        if row == 2:
            ax.set_xlabel("Time [s]", fontsize=9)

# ---------------------------------------------------------------------------
# Phase annotations in the top row of the middle column (most readable space)
# Use the x-axis/axes-fraction blend transform: x in data coords, y in [0,1]
# ---------------------------------------------------------------------------
TOP_ROW = 0
for col in range(3):
    ax = axes[TOP_ROW, col]
    trans = ax.get_xaxis_transform()

    ax.text(
        t_switch / 2,           # x midpoint of engine-torque phase (data coords)
        0.93,                   # y near top (axes fraction)
        "Engine\ntorque",
        transform=trans,
        fontsize=7.5,
        ha="center", va="top",
        color="#333333",
        fontweight="semibold",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
    )
    ax.text(
        t_switch + (t[-1] - t_switch) / 2,   # x midpoint of braking phase
        0.93,
        "Braking",
        transform=trans,
        fontsize=7.5,
        ha="center", va="top",
        color="#333333",
        fontweight="semibold",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
    )

# ---------------------------------------------------------------------------
fig.suptitle(
    r"Simulated vehicle observations for three friction levels ($\mu \in \{0.6, 0.9, 1.4\}$, "
    r"$T_{\mathrm{seg}} = 1000$ steps, 10 s at 100 Hz)."
    "\nControl: 0.4 Hz sine-steer (±6°); engine torque (0–5 s); braking (5–10 s).",
    fontsize=9,
)
fig.tight_layout(rect=[0, 0, 1, 0.92], pad=0.8)
fig.savefig(OUT_PATH, bbox_inches="tight")
print(f"\nSaved: {OUT_PATH}")
