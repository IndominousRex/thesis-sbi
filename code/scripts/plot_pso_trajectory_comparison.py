"""
Generate PSO trajectory comparison figure for the thesis.

Produces a two-row figure for one representative recording:
  Row 1 (top)    : Default simulator vs. real measurement
  Row 2 (bottom) : PSO-calibrated simulator vs. real measurement

Only 4 channels are shown: yaw_rate, v_x, tire_FR, a_x.

Run from the repository root:
    python code/scripts/plot_pso_trajectory_comparison.py

Output:
    text (Latex)/Figures/pso/pso_trajectory_comparison.pdf
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import pathlib
import matplotlib
import matplotlib.pyplot as plt

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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp

from simulation.VehicleModel import vehicle_RK4x, vehicle_fy

JSON_PATH = (
    REPO_ROOT
    / "code"
    / "notebooks"
    / "experiments"
    / "pso_global_log_optimization_results.json"
)
DATA_DIR = REPO_ROOT / "data" / "measurements"
OUT_DIR = REPO_ROOT / "text (Latex)" / "Figures" / "pso"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "pso_trajectory_comparison.pdf"

# ---------------------------------------------------------------------------
# Default (nominal) simulator parameters — from vehicle model defaults
# These match the values in Tables A.1–A.3 of the thesis.
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = dict(
    c_1x=2.50e7,
    c_2x=3.60e6,
    c_1y=1.90e6,
    c_2y=1.35e5,
    C_x=1.42,
    C_y=1.90,
    E_x=-9.75,
    E_y=0.52,
    C_roll1=0.05,
    C_roll2=0.0,
    mass=1720.0,
    Inertia_z=2066.0,
    Inertia_tire=28.6,
    Inertia_engine=0.197,
    air_resistance=0.270,
    radius_tire=0.3116,
)
DEFAULT_MU = 0.85  # representative mid-range value

# ---------------------------------------------------------------------------
# Load PSO results
# ---------------------------------------------------------------------------
with open(JSON_PATH) as f:
    results = json.load(f)

gp = results["global_params"]
PSO_PARAMS = dict(
    c_1x=gp["c_1x"],
    c_2x=gp["c_2x"],
    c_1y=gp["c_1y"],
    c_2y=gp["c_2y"],
    C_x=gp["C_x"],
    C_y=gp["C_y"],
    E_x=gp["E_x"],
    E_y=gp["E_y"],
    C_roll1=gp["C_roll1"],
    C_roll2=gp["C_roll2"],
    mass=gp["mass"],
    Inertia_z=gp["Inertia_z"],
    Inertia_tire=gp["Inertia_tire"],
    Inertia_engine=gp["Inertia_engine"],
    air_resistance=gp["air_resistance"],
    radius_tire=gp["radius_tire"],
)

# Pick the PSO mu for the first Asphalt recording
mu_lookup = {
    tp["trajectory"]: tp["params"]["mu"] for tp in results["trajectory_params"]
}
asphalt_name = next(
    (
        tp["trajectory"]
        for tp in results["trajectory_params"]
        if tp["surface"] == "Asphalt"
    ),
    None,
)
PSO_MU = mu_lookup[asphalt_name] if asphalt_name else 1.4
START_IDX = results.get("start_idx", 50)
T_SEG = results.get("T_seg", 800)

# ---------------------------------------------------------------------------
# Load one representative recording (first Asphalt csv)
# ---------------------------------------------------------------------------
csv_files = sorted(DATA_DIR.glob("*.csv"))
asphalt_csv = next((f for f in csv_files if "asphalt" in f.stem.lower()), csv_files[0])
df = pd.read_csv(asphalt_csv)
T = min(T_SEG, len(df) - START_IDX)
sl = slice(START_IDX, START_IDX + T)

y_real = np.zeros((T, 9), dtype=np.float32)
y_real[:, 0] = df["Rate_Body_Z"].values[sl] * np.pi / 180.0
y_real[:, 1] = df["INS_Vel_Body_X"].values[sl]
y_real[:, 2] = df["INS_Vel_Body_Y"].values[sl]
y_real[:, 3] = df["Acc_Body_X"].values[sl]
y_real[:, 4] = df["Acc_Body_Y"].values[sl]
y_real[:, 5] = df["Tire_Rate_FL"].values[sl]
y_real[:, 6] = df["Tire_Rate_FR"].values[sl]
y_real[:, 7] = df["Tire_Rate_RL"].values[sl]
y_real[:, 8] = df["Tire_Rate_RR"].values[sl]

controls_arr = jnp.stack(
    [
        jnp.array(np.deg2rad(df["Steer_Angle"].values[sl]), dtype=jnp.float32),
        jnp.array(df["Engine_Torque"].values[sl], dtype=jnp.float32),
        jnp.array(df["Break_Pressure"].values[sl], dtype=jnp.float32),
        jnp.array(df["Gear_Transmission"].values[sl], dtype=jnp.float32),
    ],
    axis=1,
)

y0 = y_real[0]
state0 = jnp.array(
    [0.0, 0.0, 0.0, y0[0], y0[1], y0[2], y0[5], y0[6], y0[7], y0[8]],
    dtype=jnp.float32,
)
timesteps = jnp.arange(T)


# ---------------------------------------------------------------------------
# JIT simulator
# ---------------------------------------------------------------------------
@jax.jit
def simulate(p_inf, params_d, radius_val):
    params = dict(**params_d, radius_tire=radius_val, dt=0.01)

    def body_fn(state, t):
        u_t = dict(
            steer_ang=controls_arr[t, 0],
            engine_torque=controls_arr[t, 1],
            break_torque=controls_arr[t, 2],
            gear_transmission=controls_arr[t, 3],
        )
        y_t = vehicle_fy(state, u_t, p_inf, **params)
        state_next = vehicle_RK4x(state, u_t, p_inf, **params)
        return state_next, y_t

    _, y_seq = jax.lax.scan(body_fn, state0, timesteps)
    return y_seq


def make_params_arr(p):
    return jnp.array(
        [
            p["c_1x"],
            p["c_2x"],
            p["c_1y"],
            p["c_2y"],
            p["C_x"],
            p["C_y"],
            p["E_x"],
            p["E_y"],
            p["C_roll1"],
            p["C_roll2"],
            p["mass"],
            p["Inertia_z"],
            p["Inertia_tire"],
            p["Inertia_engine"],
            p["air_resistance"],
        ],
        dtype=jnp.float32,
    )


# Simulate
p_inf_default = jnp.array(
    [DEFAULT_MU, DEFAULT_PARAMS["air_resistance"], DEFAULT_PARAMS["mass"]],
    dtype=jnp.float32,
)
p_inf_pso = jnp.array(
    [PSO_MU, PSO_PARAMS["air_resistance"], PSO_PARAMS["mass"]], dtype=jnp.float32
)

print("Running default simulator...")
y_default = np.asarray(
    simulate(
        p_inf_default,
        {k: v for k, v in DEFAULT_PARAMS.items() if k != "radius_tire"},
        DEFAULT_PARAMS["radius_tire"],
    )
)

print("Running PSO-calibrated simulator...")
y_pso = np.asarray(
    simulate(
        p_inf_pso,
        {k: v for k, v in PSO_PARAMS.items() if k != "radius_tire"},
        PSO_PARAMS["radius_tire"],
    )
)

# ---------------------------------------------------------------------------
# Compute NRMSE for reporting in captions
# ---------------------------------------------------------------------------
OBS_REF = np.array(
    [0.05, 10.0, 0.5, 5.0, 2.0, 100.0, 100.0, 100.0, 100.0], dtype=np.float32
)
W = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)


def weighted_nrmse(y_hat, y_ref):
    per_ch = np.sqrt(np.mean((y_hat - y_ref) ** 2, axis=0))
    return float(np.sum(W * per_ch / OBS_REF) / W.sum())


nrmse_default = weighted_nrmse(y_default, y_real)
nrmse_pso = weighted_nrmse(y_pso, y_real)
print(f"NRMSE default : {nrmse_default:.4f}")
print(f"NRMSE PSO     : {nrmse_pso:.4f}")

# ---------------------------------------------------------------------------
# Plot: 2 rows × 4 channels
# ---------------------------------------------------------------------------
CHANNELS = [0, 1, 6, 3]  # yaw_rate, v_x, tire_FR, a_x
CHANNEL_NAMES = ["Yaw rate [rad/s]", r"$v_x$ [m/s]", "Tire FR [rad/s]", r"$a_x$ [m/s²]"]
t_axis = np.arange(T) * 0.01

fig, axes = plt.subplots(2, 4, figsize=(13, 5), sharex=True)

for col, (ch, ch_name) in enumerate(zip(CHANNELS, CHANNEL_NAMES)):
    # Row 0: default
    ax0 = axes[0, col]
    ax0.plot(t_axis, y_real[:, ch], color="#2166ac", lw=1.2, label="Real")
    ax0.plot(
        t_axis,
        y_default[:, ch],
        color="#d6604d",
        lw=1.0,
        ls="--",
        label=f"Default (NRMSE={nrmse_default:.3f})",
    )
    ax0.set_title(ch_name, fontsize=9)
    if col == 0:
        ax0.set_ylabel("Default params", fontsize=9)
        ax0.legend(loc="upper right", fontsize=7)

    # Row 1: PSO
    ax1 = axes[1, col]
    ax1.plot(t_axis, y_real[:, ch], color="#2166ac", lw=1.2, label="Real")
    ax1.plot(
        t_axis,
        y_pso[:, ch],
        color="#4dac26",
        lw=1.0,
        ls="--",
        label=f"PSO-calibrated (NRMSE={nrmse_pso:.3f})",
    )
    ax1.set_xlabel("Time [s]", fontsize=8)
    if col == 0:
        ax1.set_ylabel("PSO params", fontsize=9)
        ax1.legend(loc="upper right", fontsize=7)

fig.suptitle(
    f"Simulator trajectory comparison — Asphalt recording\n"
    f"Default NRMSE={nrmse_default:.3f}   PSO-calibrated NRMSE={nrmse_pso:.3f}",
    fontsize=10,
)

fig.tight_layout(pad=1.2)
fig.savefig(OUT_PATH, bbox_inches="tight")
print(f"\nSaved: {OUT_PATH}")
