"""
Generate a training data example figure for the thesis.

Simulates three short segments with different friction coefficients
(µ ∈ {0.4, 0.8, 1.4}) using synthetic sine-steer + braking control inputs,
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
from simulation.simulation import u_sine, u_step, u_flat

# ---------------------------------------------------------------------------
# Simulation settings
# ---------------------------------------------------------------------------
T_SEG = 1000  # 10 seconds at 100 Hz
DT = 0.01
t = np.arange(T_SEG) * DT

MU_VALUES = [0.4, 0.8, 1.4]
MU_LABELS = [r"$\mu = 0.4$", r"$\mu = 0.8$", r"$\mu = 1.4$"]
MU_COLORS = ["#d6604d", "#f4a582", "#2166ac"]

# Synthetic control: gentle slalom + moderate braking halfway through
steer = u_sine(T_SEG, amp=np.deg2rad(6.0), freq_hz=0.4, dt=DT)  # slalom
engine_torque = np.where(np.arange(T_SEG) < T_SEG // 2, 800.0, 0.0).astype(np.float32)
brake_pressure = np.where(np.arange(T_SEG) >= T_SEG // 2, 30.0, 0.0).astype(np.float32)
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
    # p_inf = [mu, air_resistance (unused via default_params), mass (unused)]
    # The vehicle model accepts p_inf as [mu, cd_override, mass_override]
    p_inf = jnp.array([mu, 0.270, 1720.0], dtype=jnp.float32)
    y = np.asarray(simulate_one(p_inf))
    trajectories[mu] = y
    print(
        f"  µ={mu:.1f}  yaw_rate range: [{y[:,0].min():.3f}, {y[:,0].max():.3f}] rad/s"
    )

# ---------------------------------------------------------------------------
# Plot: 3 rows (channels) × 3 columns (µ values)
# Channels: yaw_rate (0), v_x (1), tire_FR (6)
# ---------------------------------------------------------------------------
matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    }
)

CH_INDICES = [0, 1, 6]
CH_NAMES = ["Yaw rate [rad/s]", r"$v_x$ [m/s]", "Tire FR [rad/s]"]

fig, axes = plt.subplots(3, 3, figsize=(11, 6.5), sharex=True)

for col, (mu, label, color) in enumerate(zip(MU_VALUES, MU_LABELS, MU_COLORS)):
    y = trajectories[mu]
    for row, (ch, ch_name) in enumerate(zip(CH_INDICES, CH_NAMES)):
        ax = axes[row, col]
        ax.plot(t, y[:, ch], color=color, lw=1.0)
        if row == 0:
            ax.set_title(label, fontsize=10)
        if col == 0:
            ax.set_ylabel(ch_name, fontsize=9)
        if row == 2:
            ax.set_xlabel("Time [s]", fontsize=9)
        ax.grid(True, alpha=0.3, lw=0.5)

# Shade the braking phase in each panel
for ax in axes.flat:
    ax.axvspan(t[T_SEG // 2], t[-1], alpha=0.06, color="grey", label="_")

# Annotate control inputs
axes[0, 1].annotate(
    "Engine torque",
    xy=(2.5, axes[0, 1].get_ylim()[1] * 0.85),
    fontsize=7,
    color="#555",
)
axes[0, 1].annotate(
    "Braking\n(grey)",
    xy=(6.0, axes[0, 1].get_ylim()[1] * 0.85),
    fontsize=7,
    color="#888",
)

fig.suptitle(
    r"Simulated vehicle observations for three friction levels ($\mu \in \{0.4, 0.8, 1.4\}$, "
    r"$T_{\mathrm{seg}}=1000$). "
    "\nControl: 0.4 Hz sine-steer (± 6°); engine torque (0–5 s); braking (5–10 s).",
    fontsize=9,
)
fig.tight_layout(rect=[0, 0, 1, 0.93], pad=0.8)
fig.savefig(OUT_PATH, bbox_inches="tight")
print(f"\nSaved: {OUT_PATH}")
