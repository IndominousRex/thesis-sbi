# sbi_vehicle/simulation.py

from typing import Tuple, Dict, Any

import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax
from tqdm.auto import tqdm
import torch

from simulation.VehicleModel import (  # your existing module
    default_params,
    vehicle_RK4x,
    vehicle_fy,
    radius_tire,
)
from configs.config import ExperimentConfig, PARAMETER_ORDER


# --- Global-ish holders for config-dependent constants ---
DT: float = 0.01
T_SEG: int = 3000
STATE_DIM: int = 10


def expand_theta_to_full(theta_np: np.ndarray, cfg: ExperimentConfig) -> jnp.ndarray:
    """
    Map an active-parameter theta (dim=1/2/3) to the full (mu, cd, m) vector
    expected by the vehicle model, plugging in fixed defaults for inactive ones.
    """
    if theta_np.ndim != 2:
        raise ValueError(f"theta must be 2D, got shape {theta_np.shape}")

    if theta_np.shape[1] != len(cfg.active_parameters):
        raise ValueError(
            f"theta has dim {theta_np.shape[1]}, but cfg.active_parameters has "
            f"{len(cfg.active_parameters)} entries."
        )

    B = theta_np.shape[0]
    defaults = cfg.fixed_param_values()

    full = np.column_stack(
        [
            np.full(B, defaults["mu"], dtype=np.float32),
            np.full(B, defaults["cd"], dtype=np.float32),
            np.full(B, defaults["m"], dtype=np.float32),
        ]
    ).astype(np.float32)

    for idx, name in enumerate(cfg.active_parameters):
        full[:, PARAMETER_ORDER.index(name)] = theta_np[:, idx]

    return jnp.asarray(full, dtype=jnp.float32)


def init_simulation_from_config(cfg: ExperimentConfig) -> None:
    """
    Initialize module-level constants from the experiment config.
    """
    global DT, T_SEG, STATE_DIM
    DT = cfg.dt
    T_SEG = cfg.T_seg
    STATE_DIM = cfg.state_dim


@jax.jit
def rollout_with_states(p, controls, state0=None):
    """
    Single-trajectory rollout through the JAX vehicle model.

    Args:
        p:       parameter vector (θ) for one trajectory.
        controls: dict of control sequences (jnp arrays) of length T.
        state0:  initial state (10,) or None for zeros.

    Returns:
        state_seq: (T, state_dim)
        y_seq    : (T, d_obs)
    """
    T_local = controls["steer_ang"].shape[0]
    if state0 is None:
        state0 = jnp.zeros(STATE_DIM, dtype=jnp.float32)

    def body_fn(state, t):
        u_t = {k: v[t] for k, v in controls.items()}
        state_next = vehicle_RK4x(state, u_t, p, **default_params)
        y_t = vehicle_fy(state_next, u_t, p, **default_params)
        return state_next, (state_next, y_t)

    t_idx = jnp.arange(T_local, dtype=jnp.int32)
    _, (state_seq, y_seq) = lax.scan(body_fn, state0, t_idx)
    return state_seq, y_seq


def controls_to_array(ctrls):
    """Stack control channels into (T, 4) array in a fixed order."""
    return jnp.stack(
        [
            ctrls["steer_ang"],
            ctrls["engine_torque"],
            ctrls["break_torque"],
            ctrls["gear_transmission"],
        ],
        axis=-1,
    )  # (T, 4)


def u_flat(T, value=0.0, *, dtype=np.float32):
    x = np.empty(T, dtype=dtype)
    x.fill(value)
    return x


def u_step(T, value=1.0, t_on=0, t_off=None, *, dtype=np.float32):
    """
    Rectangular step: x=0 except x[t_on:t_off]=value.
    t_on / t_off are sample indices in [0, T].
    """
    x = np.zeros(T, dtype=dtype)
    if t_off is None:
        t_off = T
    t_on = int(np.clip(t_on, 0, T))
    t_off = int(np.clip(t_off, 0, T))
    if t_off > t_on:
        x[t_on:t_off] = value
    return x


def u_sine(T, amp=1.0, freq_hz=0.5, dt=0.01, phase=0.0, *, dtype=np.float32):
    t = np.arange(T, dtype=dtype) * dt
    return (amp * np.sin(2 * np.pi * freq_hz * t + phase)).astype(dtype)


def _mk_signal(kind, T, dt, **kw):
    if kind == "flat":
        return u_flat(T, **kw)
    if kind == "step":
        return u_step(T, **kw)
    if kind == "sine":
        return u_sine(T, dt=dt, **kw)
    raise ValueError(f"unknown kind {kind}")


def _prepare_step_kwargs(kw, T_block, dt):
    """
    Allow user-friendly step specs:
      - 'duty' as fraction of block
      - 't_on_s', 't_off_s' in seconds
      - or 't_on', 't_off' as indices
    """
    kw = dict(kw)
    if "duty" in kw:
        duty = float(kw.pop("duty"))
        on = 0
        off = max(1, int(round(duty * T_block)))
        kw["t_on"], kw["t_off"] = on, off
    else:
        if "t_on_s" in kw:
            kw["t_on"] = int(round(kw.pop("t_on_s") / dt))
        if "t_off_s" in kw:
            off_s = kw.pop("t_off_s")
            kw["t_off"] = None if off_s is None else int(round(off_s / dt))
    return kw


def controls_from_blocks(blocks, *, T_seg, dt, gear_default=3, ramp_s=0.0):
    """
    Build a long control sequence from a list of 'blocks' with different behaviours
    (accel, brake, coast). This is the same logic as your notebook.
    """
    segs = []
    left = T_seg

    def build_one(spec, default_kind, default_kw, T_b):
        kind, kw = spec if spec is not None else (default_kind, default_kw)
        if kind == "step":
            kw = _prepare_step_kwargs(kw, T_b, dt)
        return _mk_signal(kind, T_b, dt, **kw).astype(np.float32)

    for b in blocks:
        T_b = min(max(1, int(round(b["dur_s"] / dt))), left)
        steer = build_one(b.get("steer"), "flat", {"value": 0.0}, T_b)
        eng = build_one(b.get("engine"), "flat", {"value": 0.0}, T_b)
        brk = build_one(b.get("brake"), "flat", {"value": 0.0}, T_b)
        gear = build_one(b.get("gear"), "flat", {"value": gear_default}, T_b)

        segs.append((steer, eng, brk, gear))
        left -= T_b
        if left <= 0:
            break

    if left > 0:
        segs.append(
            (
                u_flat(left, 0.0),
                u_flat(left, 0.0),
                u_flat(left, 0.0),
                u_flat(left, gear_default),
            )
        )

    steer = np.concatenate([s for s, _, _, _ in segs], axis=0)[:T_seg]
    eng = np.concatenate([e for _, e, _, _ in segs], axis=0)[:T_seg]
    brk = np.concatenate([b for _, _, b, _ in segs], axis=0)[:T_seg]
    gear = np.concatenate([g for _, _, _, g in segs], axis=0)[:T_seg]

    # Optional ramps between blocks
    if ramp_s > 0.0:
        n = max(2, int(round(ramp_s / dt)))

        def ramp(vec):
            out = vec.copy()
            idx = 0
            for s, e, b, g in segs:
                L = len(s)
                if idx > 0:
                    a0, a1 = out[idx - 1], out[idx]
                    out[idx : idx + min(n, L)] = np.linspace(
                        a0, a1, num=min(n, L), dtype=np.float32
                    )
                idx += L
            return out

        steer = ramp(steer)
        eng = ramp(eng)
        brk = ramp(brk)

    return {
        "steer_ang": jnp.asarray(steer),
        "engine_torque": jnp.asarray(eng),
        "break_torque": jnp.asarray(brk),
        "gear_transmission": jnp.asarray(gear),
    }


def rand_block_accel(rng):
    eng = float(rng.uniform(300, 500))
    duty = float(rng.uniform(0.4, 1.0))
    steer_deg = float(rng.uniform(4, 12))
    f = float(rng.uniform(0.5, 1.5))
    dur = float(rng.uniform(1.5, 4.0))
    gear = int(rng.integers(2, 6))
    return {
        "dur_s": dur,
        "engine": ("step", {"value": eng, "duty": duty}),
        "brake": ("step", {"value": 0.0, "duty": 0.0}),
        "steer": ("sine", {"amp": np.deg2rad(steer_deg), "freq_hz": f, "phase": 0.0}),
        "gear": ("flat", {"value": gear}),
    }


def rand_block_brake(rng):
    brk = float(rng.uniform(180, 380))
    duty = float(rng.uniform(0.2, 1.0))
    dur = float(rng.uniform(0.6, 3.0))
    gear = int(rng.integers(2, 6))
    return {
        "dur_s": dur,
        "engine": ("step", {"value": 0.0, "duty": 0.0}),
        "brake": ("step", {"value": brk, "duty": duty}),
        "steer": ("flat", {"value": 0.0}),
        "gear": ("flat", {"value": gear}),
    }


def rand_block_coast(rng):
    dur = float(rng.uniform(0.8, 2.5))
    steer_deg = float(rng.uniform(-5, 5))
    gear = int(rng.integers(2, 6))
    return {
        "dur_s": dur,
        "engine": ("step", {"value": 0.0, "duty": 0.0}),
        "brake": ("step", {"value": 0.0, "duty": 0.0}),
        "steer": ("flat", {"value": np.deg2rad(steer_deg)}),
        "gear": ("flat", {"value": gear}),
    }


def make_shuffled_U(rng, total_T, dt):
    """
    Build a shuffled mixture of accel, brake, and coast blocks
    until total length reaches total_T * dt.
    """
    pieces = []
    for _ in range(12):
        pieces.append(rand_block_accel(rng))
        pieces.append(rand_block_brake(rng))
        pieces.append(rand_block_coast(rng))
    rng.shuffle(pieces)

    blocks, tsum = [], 0.0
    for b in pieces:
        blocks.append(b)
        tsum += b["dur_s"]
        if tsum >= total_T * dt:
            break
    return blocks


# --- Initial state sampling (S0) ---


KMH_TO_MS = 1.0 / 3.6
INIT_SPEEDS_MS = np.arange(10.0, 110.0, 10.0, dtype=np.float32) * KMH_TO_MS


def sample_initial_states(rng: np.random.Generator, B: int) -> jnp.ndarray:
    """
    Sample a batch of initial states following your chosen heuristics.
    """
    idx = rng.integers(0, len(INIT_SPEEDS_MS), size=B)
    v_x0 = INIT_SPEEDS_MS[idx]

    v_y0 = rng.uniform(-1.0, 1.0, size=B).astype(np.float32)
    yaw0 = rng.uniform(-0.05, 0.05, size=B).astype(np.float32)
    dyaw0 = rng.uniform(-0.2, 0.2, size=B).astype(np.float32)

    R_wheel = radius_tire
    base_omega = v_x0 / max(R_wheel, 1e-6)

    noise_scale = 0.05

    def noisy(base):
        return base * (1.0 + rng.normal(0.0, noise_scale, size=B).astype(np.float32))

    tire_fl = noisy(base_omega)
    tire_fr = noisy(base_omega)
    tire_rl = noisy(base_omega)
    tire_rr = noisy(base_omega)

    geo_pos_x = np.zeros(B, dtype=np.float32)
    geo_pos_y = np.zeros(B, dtype=np.float32)

    state_np = np.stack(
        [
            geo_pos_x,
            geo_pos_y,
            yaw0,
            dyaw0,
            v_x0,
            v_y0,
            tire_fl,
            tire_fr,
            tire_rl,
            tire_rr,
        ],
        axis=1,
    ).astype(np.float32)

    return jnp.asarray(state_np, dtype=jnp.float32)


def _one_rollout(p, s0, ctrls):
    return rollout_with_states(p, ctrls, state0=s0)[1]


_batched_rollout = jax.jit(jax.vmap(_one_rollout, in_axes=(0, 0, None)))


def vmapped_rollout_train(P, S0, ctrls):
    """Batched rollout over parameters and initial states."""
    return _batched_rollout(P, S0, ctrls)


def make_simulator(cfg: ExperimentConfig, device: torch.device):
    """
    Factory that returns a simulator(theta) -> torch.Tensor, matching the SBI API.

    The returned simulator:
      - samples a fresh control recipe per call,
      - samples initial states,
      - runs the JAX vehicle model,
      - concatenates controls to observations.
    """

    def simulator(theta: torch.Tensor) -> torch.Tensor:
        theta_np = theta.detach().cpu().numpy().astype(np.float32)
        B = theta_np.shape[0]

        rng = np.random.default_rng(
            getattr(simulator, "_batch_idx", 0) + cfg.random_seed
        )
        simulator._batch_idx = getattr(simulator, "_batch_idx", 0) + 1

        recipe = make_shuffled_U(rng, cfg.T_seg, cfg.dt)
        ctrls = controls_from_blocks(recipe, T_seg=cfg.T_seg, dt=cfg.dt, ramp_s=0.2)

        S0 = sample_initial_states(rng, B)
        P = expand_theta_to_full(theta_np, cfg)

        y_batch = vmapped_rollout_train(P, S0, ctrls)  # (B, T_seg, d_obs)

        c = controls_to_array(ctrls)  # (T_seg, 4)
        c_rep = jnp.broadcast_to(c, (B, c.shape[0], c.shape[1]))  # (B, T_seg, 4)
        yc = jnp.concatenate([y_batch, c_rep], axis=-1)  # (B, T_seg, D_in)

        yc_numpy = np.asarray(yc, dtype=np.float32)
        yc_numpy_copy = yc_numpy.copy()  # to ensure contiguous array

        return torch.from_numpy(yc_numpy_copy).to(device)

    return simulator


def generate_dataset(
    cfg: ExperimentConfig,
    prior,
    simulator,
    show_pbar: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate (theta, x) pairs for SBI using the provided simulator and prior.

    Returns:
        theta_all: (N, param_dim) on CPU
        x_all:     (N, T_event, D_in) on CPU
    """
    N = cfg.num_simulations
    batch = cfg.batch_sim

    Theta, X = [], []

    if cfg.jit_warmup:
        if show_pbar:
            tqdm.write("Compiling JAX (one-time JIT warmup)…")
        with torch.inference_mode():
            theta_w = prior.sample((min(8, N),)).to("cpu")
            _ = simulator(theta_w)

    pbar = (
        tqdm(total=N, unit="sims", desc="Generating sims", dynamic_ncols=True)
        if show_pbar
        else None
    )

    done = 0

    with torch.inference_mode():
        for i in range(0, N, batch):
            b = min(batch, N - i)
            theta_b = prior.sample((b,)).to("cpu")
            x_b = simulator(theta_b)

            Theta.append(theta_b)
            X.append(x_b.cpu())

            done += b
            if pbar is not None:
                elapsed = max(1e-6, (i + b) / max(1, done))  # cheap dummy ETA
                pbar.set_postfix_str(f"{done}/{N}")
                pbar.update(b)

    if pbar is not None:
        pbar.close()

    return torch.cat(Theta, 0), torch.cat(X, 0)
