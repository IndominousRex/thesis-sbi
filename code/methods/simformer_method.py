"""
Faithful Simformer-style method for vehicle parameter inference.

This implementation follows the reference `simformer-main` workflow much more
closely than the previous wrapper:
- no external PyTorch sequence encoder
- raw scalar observation values are used directly inside the JAX model
- all-conditional training with structured-random masks
- validation-based model selection with data-scaled training steps

For tractability on long vehicle trajectories, observations are represented as
channel-specific scalar timepoint nodes sampled from the dense trajectory,
matching the reference repo's unstructured time-series approach.
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax
import torch

if not hasattr(jax, "linear_util"):
    try:
        from jax.extend import linear_util

        jax.linear_util = linear_util
    except ImportError:
        from jax._src import linear_util

        jax.linear_util = linear_util

try:
    from jax._src import api_util as _jax_api_util
    if not hasattr(_jax_api_util, "shaped_abstractify"):
        from jax._src.core import shaped_abstractify as _shaped_abstractify

        _jax_api_util.shaped_abstractify = _shaped_abstractify
except Exception:
    pass

try:
    from jax.interpreters import batching as _jax_batching

    if not hasattr(_jax_batching, "spmd_axis_primitive_batchers") and hasattr(
        _jax_batching, "axis_primitive_batchers"
    ):
        _jax_batching.spmd_axis_primitive_batchers = (
            _jax_batching.axis_primitive_batchers
        )
except Exception:
    pass

try:
    if not hasattr(jax, "util"):
        from jax._src import util as _jax_util

        jax.util = _jax_util
except Exception:
    pass

_make_key = getattr(jrandom, "key", jrandom.PRNGKey)

if not hasattr(jrandom, "default_prng_impl"):

    class _StubPRNGImpl:
        @staticmethod
        def seed_and_split(seed):
            if hasattr(_make_key(0), "dtype"):
                k = _make_key(seed if isinstance(seed, int) else 0)
            else:
                k = jrandom.PRNGKey(seed if isinstance(seed, int) else 0)
            return k, _StubPRNGImpl.split(k)

        @staticmethod
        def split(key):
            return jrandom.split(key)[0]

    jrandom.default_prng_impl = _StubPRNGImpl


SIMFORMER_PATH = Path(__file__).parent.parent / "simformer-main" / "src"
if str(SIMFORMER_PATH) not in sys.path:
    sys.path.insert(0, str(SIMFORMER_PATH / "probjax"))
    sys.path.insert(0, str(SIMFORMER_PATH / "scoresbibm"))

from probjax.nn.loss_fn import denoising_score_matching_loss
from scoresbibm.methods.models import AllConditionalScoreModel
from scoresbibm.methods.neural_nets import scalar_transformer_model
from scoresbibm.methods.sde import init_sde_related
from scoresbibm.utils.condition_masks import get_condition_mask_fn

from .base import BaseMethod
from utils.normalization import Normalizer


class SimformerPosterior:
    """Torch-compatible posterior wrapper over the JAX all-conditional model."""

    def __init__(
        self,
        model: AllConditionalScoreModel,
        *,
        seq_len: int,
        input_dim: int,
        num_timepoints: int,
        node_id: jnp.ndarray,
        condition_mask: jnp.ndarray,
        meta_data: Optional[jnp.ndarray],
        num_steps: int,
        clip_low: Optional[jnp.ndarray] = None,
        clip_high: Optional[jnp.ndarray] = None,
    ) -> None:
        self.model = model
        self.seq_len = int(seq_len)
        self.input_dim = int(input_dim)
        self.num_timepoints = int(num_timepoints)
        self.node_id = node_id
        self.condition_mask = condition_mask
        self.meta_data = meta_data
        self.num_steps = int(num_steps)
        self.clip_low = clip_low
        self.clip_high = clip_high
        self.dense_time_grid = np.linspace(0.0, 1.0, self.seq_len, dtype=np.float32)
        self.eval_times = np.linspace(0.0, 1.0, self.num_timepoints, dtype=np.float32)

    def _flatten_conditioning_observation(self, x: torch.Tensor) -> jnp.ndarray:
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.shape[0] != 1:
            x = x[:1]

        x_np = x.detach().cpu().numpy().astype(np.float32)[0]
        channel_values = []
        for channel_idx in range(self.input_dim):
            channel_values.append(
                np.interp(self.eval_times, self.dense_time_grid, x_np[:, channel_idx])
            )
        flat = np.stack(channel_values, axis=0).reshape(-1)
        return jnp.asarray(flat, dtype=jnp.float32)

    def sample(
        self,
        shape: tuple[int, ...] | int,
        x: torch.Tensor,
        seed: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        num_samples = shape[0] if isinstance(shape, tuple) else int(shape)
        x_o = self._flatten_conditioning_observation(x)
        seed = int(seed) if seed is not None else int(time.time() * 1000) % (2**31)
        key = _make_key(seed)
        samples = self.model.sample(
            num_samples,
            x_o=x_o,
            rng=key,
            node_id=self.node_id,
            condition_mask=self.condition_mask,
            meta_data=self.meta_data,
            num_steps=self.num_steps,
            unique_nodes=False,
        )
        if self.clip_low is not None and self.clip_high is not None:
            samples = jnp.clip(samples, self.clip_low, self.clip_high)
        return torch.from_numpy(np.asarray(samples)).float()


class SimformerMethod(BaseMethod):
    """Reference-style Simformer adapted to long vehicle trajectories."""

    name = "simformer"
    model_filename = "simformer_model.pkl"

    def __init__(
        self,
        cfg,
        prior,
        device: torch.device,
        *,
        num_timepoints: int = 32,
        token_dim: int = 40,
        condition_token_dim: int = 10,
        condition_token_init_scale: float = 0.1,
        condition_token_init_mean: float = 0.0,
        condition_mode: str = "concat",
        time_embedding_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 6,
        attn_size: int = 10,
        widening_factor: int = 3,
        num_hidden_layers: int = 1,
        skip_connection_attn: bool = True,
        skip_connection_mlp: bool = True,
        layer_norm: bool = True,
        sigma_min: float = 0.01,
        sigma_max: float = 15.0,
        T_min: float = 0.02,
        T_max: float = 1.0,
        num_diffusion_steps: int = 500,
        learning_rate: float = 1e-3,
        min_learning_rate: float = 1e-6,
        clip_max_norm: float = 10.0,
        batch_size: int = 64,
        train_steps_scaling: int = 3,
        min_train_steps: int = 5000,
        max_train_steps: int = 100000,
        validation_fraction: float = 0.05,
        val_repeat: int = 5,
        val_every: int = 50,
        stop_early_count: int = 5,
        val_error_ratio: float = 1.1,
        condition_mask_name: str = "structured_random",
        condition_mask_kwargs: Optional[Dict[str, float]] = None,
        use_metadata: bool = True,
        rebalance_loss: bool = False,
    ) -> None:
        super().__init__(cfg, prior, device)
        self.num_timepoints = int(num_timepoints)
        self.token_dim = token_dim
        self.condition_token_dim = condition_token_dim
        self.condition_token_init_scale = condition_token_init_scale
        self.condition_token_init_mean = condition_token_init_mean
        self.condition_mode = condition_mode
        self.time_embedding_dim = time_embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.attn_size = attn_size
        self.widening_factor = widening_factor
        self.num_hidden_layers = num_hidden_layers
        self.skip_connection_attn = skip_connection_attn
        self.skip_connection_mlp = skip_connection_mlp
        self.layer_norm = layer_norm
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.T_min = T_min
        self.T_max = T_max
        self.num_diffusion_steps = int(num_diffusion_steps)
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.clip_max_norm = clip_max_norm
        self.batch_size = int(batch_size)
        self.train_steps_scaling = int(train_steps_scaling)
        self.min_train_steps = int(min_train_steps)
        self.max_train_steps = int(max_train_steps)
        self.validation_fraction = float(validation_fraction)
        self.val_repeat = int(val_repeat)
        self.val_every = int(val_every)
        self.stop_early_count = int(stop_early_count)
        self.val_error_ratio = float(val_error_ratio)
        self.condition_mask_name = condition_mask_name
        self.condition_mask_kwargs = condition_mask_kwargs or {}
        self.use_metadata = bool(use_metadata)
        self.rebalance_loss = bool(rebalance_loss)

        self.theta_dim: int | None = None
        self.obs_nodes: int | None = None
        self.total_nodes: int | None = None
        self.node_id: Optional[jnp.ndarray] = None
        self.posterior_condition_mask: Optional[jnp.ndarray] = None
        self.eval_meta_data: Optional[jnp.ndarray] = None
        self.eval_times: Optional[jnp.ndarray] = None
        self.dense_time_grid: Optional[jnp.ndarray] = None
        self.params = None
        self.model_fn = None
        self.model = None
        self.posterior = None
        self.sde = None
        self.weight_fn = None
        self._training_summary: Dict[str, Any] = {}
        self._sde_reference_data: Optional[np.ndarray] = None

    def build(self, input_dim: int, seq_len: int) -> None:
        self._input_dim = int(input_dim)
        self._seq_len = int(seq_len)
        self.theta_dim = len(self.cfg.active_parameters)
        self.obs_nodes = self._input_dim * self.num_timepoints
        self.total_nodes = self.theta_dim + self.obs_nodes
        self.dense_time_grid = jnp.linspace(0.0, 1.0, self._seq_len, dtype=jnp.float32)
        self.eval_times = jnp.linspace(0.0, 1.0, self.num_timepoints, dtype=jnp.float32)
        obs_node_ids = np.concatenate(
            [
                np.full(self.num_timepoints, self.theta_dim + channel_idx, dtype=np.int32)
                for channel_idx in range(self._input_dim)
            ]
        )
        self.node_id = jnp.asarray(
            np.concatenate([np.arange(self.theta_dim, dtype=np.int32), obs_node_ids]),
            dtype=jnp.int32,
        )
        self.posterior_condition_mask = jnp.asarray(
            [False] * self.theta_dim + [True] * self.obs_nodes,
            dtype=jnp.bool_,
        )
        if self.use_metadata:
            obs_meta = np.repeat(
                np.asarray(self.eval_times, dtype=np.float32)[None, :],
                self._input_dim,
                axis=0,
            ).reshape(-1)
            theta_meta = np.full(self.theta_dim, np.nan, dtype=np.float32)
            self.eval_meta_data = jnp.asarray(
                np.concatenate([theta_meta, obs_meta]), dtype=jnp.float32
            )
        else:
            self.eval_meta_data = None

        print(
            "[Simformer] Built raw-node all-conditional model "
            f"with {self.theta_dim} theta nodes + {self.obs_nodes} sampled observation nodes",
            flush=True,
        )

    def _subsample_dense_observations(
        self,
        x_dense: jnp.ndarray,
        sample_times: jnp.ndarray,
    ) -> tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """Interpolate dense trajectories at requested normalized time points."""

        dense_grid = self.dense_time_grid
        input_dim = self._input_dim

        def _interp_single(x_single: jnp.ndarray, times_single: jnp.ndarray):
            x_channels = jnp.swapaxes(x_single, 0, 1)
            obs = jax.vmap(
                lambda channel_values: jnp.interp(times_single, dense_grid, channel_values)
            )(x_channels)
            obs_flat = obs.reshape(-1)
            if self.use_metadata:
                meta_flat = jnp.repeat(times_single[None, :], input_dim, axis=0).reshape(-1)
                return obs_flat, meta_flat
            return obs_flat, None

        obs_nodes, obs_meta = jax.vmap(_interp_single)(x_dense, sample_times)
        return obs_nodes, obs_meta

    def _build_joint_batch(
        self,
        theta_batch: jnp.ndarray,
        x_dense_batch: jnp.ndarray,
        sample_times: jnp.ndarray,
    ) -> tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        obs_nodes, obs_meta = self._subsample_dense_observations(x_dense_batch, sample_times)
        joint = jnp.concatenate([theta_batch, obs_nodes], axis=1)[..., None]
        if not self.use_metadata:
            return joint, None

        theta_meta = jnp.full((theta_batch.shape[0], self.theta_dim), jnp.nan, dtype=jnp.float32)
        joint_meta = jnp.concatenate([theta_meta, obs_meta], axis=1)[..., None]
        return joint, joint_meta

    def _deterministic_joint_representation(
        self,
        theta_data: jnp.ndarray,
        x_dense_data: jnp.ndarray,
    ) -> tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        eval_times = jnp.repeat(self.eval_times[None, :], x_dense_data.shape[0], axis=0)
        return self._build_joint_batch(theta_data, x_dense_data, eval_times)

    def _build_model_fn(self):
        return scalar_transformer_model(
            self.total_nodes,
            token_dim=self.token_dim,
            condition_token_dim=self.condition_token_dim,
            condition_token_init_scale=self.condition_token_init_scale,
            condition_token_init_mean=self.condition_token_init_mean,
            condition_mode=self.condition_mode,
            time_embedding_dim=self.time_embedding_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            attn_size=self.attn_size,
            widening_factor=self.widening_factor,
            num_hidden_layers=self.num_hidden_layers,
            skip_connection_attn=self.skip_connection_attn,
            skip_connection_mlp=self.skip_connection_mlp,
            layer_norm=self.layer_norm,
            output_scale_fn=self._output_scale_fn,
        )

    def train(
        self,
        theta_train: torch.Tensor,
        x_train: torch.Tensor,
    ) -> Dict[str, Any]:
        if self.node_id is None or self.posterior_condition_mask is None:
            raise RuntimeError("Method not built. Call build() first.")

        train_start = time.time()
        theta_data = jnp.asarray(theta_train.detach().cpu().numpy(), dtype=jnp.float32)
        x_dense = jnp.asarray(x_train.detach().cpu().numpy(), dtype=jnp.float32)

        joint_reference, joint_meta_reference = self._deterministic_joint_representation(
            theta_data, x_dense
        )
        sde_ref_count = min(int(joint_reference.shape[0]), 4096)
        sde_reference = joint_reference[:sde_ref_count]
        self._sde_reference_data = np.asarray(sde_reference)

        self.sde, self.T_min, self.T_max, self.weight_fn, self._output_scale_fn = init_sde_related(
            sde_reference,
            name="vesde",
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            T_min=self.T_min,
            T_max=self.T_max,
        )

        init_fn, self.model_fn = self._build_model_fn()
        key = _make_key(int(self.cfg.train_seed or self.cfg.random_seed))
        init_count = min(10, int(joint_reference.shape[0]))
        key, key_init = jrandom.split(key)
        self.params = init_fn(
            key_init,
            jnp.ones((init_count,), dtype=jnp.float32),
            joint_reference[:init_count],
            self.node_id,
            jnp.zeros((init_count, self.total_nodes), dtype=jnp.bool_),
            meta_data=(
                joint_meta_reference[:init_count]
                if joint_meta_reference is not None
                else None
            ),
        )

        total_params = int(
            sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(self.params))
        )
        print(f"[Simformer] Model parameters: {total_params:,}", flush=True)

        train_count = int(theta_data.shape[0])
        val_count = 0
        if self.validation_fraction > 0.0 and train_count > 1:
            val_count = min(max(int(train_count * self.validation_fraction), 1), train_count - 1)

        if val_count > 0:
            theta_val, theta_data = theta_data[:val_count], theta_data[val_count:]
            x_val, x_dense = x_dense[:val_count], x_dense[val_count:]
        else:
            theta_val = x_val = None

        train_count = int(theta_data.shape[0])
        total_steps = int(
            np.clip(
                train_count * self.train_steps_scaling,
                self.min_train_steps,
                self.max_train_steps,
            )
        )
        print(
            f"[Simformer] Training schedule: total_steps={total_steps}, "
            f"batch_size={self.batch_size}, val_fraction={self.validation_fraction}",
            flush=True,
        )

        condition_mask_fn = get_condition_mask_fn(
            self.condition_mask_name, **self.condition_mask_kwargs
        )
        schedule = optax.linear_schedule(
            self.learning_rate,
            self.min_learning_rate,
            max(1, total_steps // 2),
            max(1, total_steps // 2),
        )
        optimizer = optax.chain(
            optax.adaptive_grad_clip(self.clip_max_norm),
            optax.adam(schedule),
        )
        opt_state = optimizer.init(self.params)

        @jax.jit
        def _sample_train_batch(
            theta_source: jnp.ndarray, x_source: jnp.ndarray, rng: jnp.ndarray
        ):
            key_idx, key_times = jrandom.split(rng)
            indices = jrandom.randint(
                key_idx,
                (self.batch_size,),
                minval=0,
                maxval=theta_source.shape[0],
            )
            theta_batch = theta_source[indices]
            x_batch = x_source[indices]
            times = jnp.sort(
                jrandom.uniform(
                    key_times,
                    (self.batch_size, self.num_timepoints),
                    minval=0.0,
                    maxval=1.0,
                ),
                axis=1,
            )
            joint_batch, joint_meta = self._build_joint_batch(theta_batch, x_batch, times)
            return joint_batch, joint_meta

        def _loss_core(
            params: Any,
            rng: jnp.ndarray,
            joint_batch: jnp.ndarray,
            joint_meta: Optional[jnp.ndarray],
        ):
            key_t, key_loss, key_mask = jrandom.split(rng, 3)
            batch_size = joint_batch.shape[0]
            diffusion_times = jrandom.uniform(
                key_t,
                (batch_size,),
                minval=self.T_min,
                maxval=self.T_max,
            )
            condition_mask = condition_mask_fn(
                key_mask, batch_size, self.theta_dim, self.obs_nodes
            )
            loss = denoising_score_matching_loss(
                params,
                key_loss,
                diffusion_times,
                joint_batch,
                loss_mask=condition_mask,
                model_fn=self.model_fn,
                mean_fn=self.sde.marginal_mean,
                std_fn=self.sde.marginal_stddev,
                weight_fn=self.weight_fn,
                rebalance_loss=self.rebalance_loss,
                data_id=self.node_id,
                condition_mask=condition_mask,
                meta_data=joint_meta,
                edge_mask=None,
            )
            return loss

        @jax.jit
        def _update(
            params: Any,
            opt_state: Any,
            rng: jnp.ndarray,
            theta_source: jnp.ndarray,
            x_source: jnp.ndarray,
        ):
            key_batch, key_loss = jrandom.split(rng)
            joint_batch, joint_meta = _sample_train_batch(theta_source, x_source, key_batch)
            loss, grads = jax.value_and_grad(_loss_core)(
                params, key_loss, joint_batch, joint_meta
            )
            updates, opt_state = optimizer.update(grads, opt_state, params=params)
            params = optax.apply_updates(params, updates)
            return loss, params, opt_state

        @jax.jit
        def _eval_loss_once(
            params: Any,
            rng: jnp.ndarray,
            theta_eval: jnp.ndarray,
            x_eval: jnp.ndarray,
        ):
            key_times, key_loss = jrandom.split(rng)
            batch_size = theta_eval.shape[0]
            eval_times = jnp.sort(
                jrandom.uniform(
                    key_times,
                    (batch_size, self.num_timepoints),
                    minval=0.0,
                    maxval=1.0,
                ),
                axis=1,
            )
            joint_eval, joint_meta = self._build_joint_batch(theta_eval, x_eval, eval_times)
            return _loss_core(params, key_loss, joint_eval, joint_meta)

        def _eval_loss_batched(
            params: Any,
            theta_eval: jnp.ndarray,
            x_eval: jnp.ndarray,
        ) -> float:
            val_batch_size = max(1, min(self.batch_size, int(theta_eval.shape[0])))
            batch_losses: list[float] = []
            for start in range(0, int(theta_eval.shape[0]), val_batch_size):
                stop = min(start + val_batch_size, int(theta_eval.shape[0]))
                key_local = jrandom.fold_in(key, start)
                batch_loss = _eval_loss_once(
                    params,
                    key_local,
                    theta_eval[start:stop],
                    x_eval[start:stop],
                )
                batch_losses.append(float(batch_loss))
            return float(np.mean(batch_losses))

        print("[Simformer] JIT compiling...", flush=True)
        key, key_warm = jrandom.split(key)
        warm_loss, self.params, opt_state = _update(
            self.params, opt_state, key_warm, theta_data, x_dense
        )
        _ = float(warm_loss)
        print("[Simformer] JIT compilation complete.", flush=True)

        print_every = max(1, total_steps // 10)
        val_interval = max(1, total_steps // max(1, self.val_every))
        train_ema = None
        best_val_loss = float("inf")
        best_validation_step = None
        best_params = self.params
        early_counter = 0
        train_loss_log: list[dict[str, float]] = []
        val_loss_log: list[dict[str, float]] = []

        for step in range(total_steps):
            key, key_step = jrandom.split(key)
            loss, self.params, opt_state = _update(
                self.params, opt_state, key_step, theta_data, x_dense
            )
            train_loss = float(loss)
            train_ema = train_loss if train_ema is None else (0.9 * train_ema + 0.1 * train_loss)

            if (step + 1) % print_every == 0 or step == 0:
                print(
                    f"  Step {step+1}/{total_steps}: loss = {train_ema:.4f}",
                    flush=True,
                )
                train_loss_log.append({"step": int(step + 1), "loss": float(train_ema)})

            if theta_val is not None and step > 50 and ((step + 1) % val_interval == 0):
                val_losses = []
                for _ in range(self.val_repeat):
                    key, _ = jrandom.split(key)
                    val_losses.append(_eval_loss_batched(self.params, theta_val, x_val))
                mean_val_loss = float(np.mean(val_losses))
                val_loss_log.append({"step": int(step + 1), "loss": mean_val_loss})
                print(
                    f"    Validation @ step {step+1}: {mean_val_loss:.4f}",
                    flush=True,
                )

                if mean_val_loss < best_val_loss:
                    best_val_loss = mean_val_loss
                    best_validation_step = int(step + 1)
                    best_params = self.params

                if train_ema is not None and mean_val_loss / max(train_ema, 1e-8) > self.val_error_ratio:
                    early_counter += 1
                else:
                    early_counter = 0

                if early_counter > self.stop_early_count:
                    print(
                        f"[Simformer] Early stopping at step {step+1} "
                        f"(best validation loss={best_val_loss:.4f})",
                        flush=True,
                    )
                    break

        self.params = best_params if best_validation_step is not None else self.params
        train_time = time.time() - train_start

        sde_init_params = {
            "data": sde_reference,
            "name": "vesde",
            "sigma_min": self.sigma_min,
            "sigma_max": self.sigma_max,
            "T_min": self.T_min,
            "T_max": self.T_max,
        }
        model_init_params = {
            "num_nodes": self.total_nodes,
            "token_dim": self.token_dim,
            "condition_token_dim": self.condition_token_dim,
            "condition_token_init_scale": self.condition_token_init_scale,
            "condition_token_init_mean": self.condition_token_init_mean,
            "condition_mode": self.condition_mode,
            "time_embedding_dim": self.time_embedding_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "attn_size": self.attn_size,
            "widening_factor": self.widening_factor,
            "num_hidden_layers": self.num_hidden_layers,
            "skip_connection_attn": self.skip_connection_attn,
            "skip_connection_mlp": self.skip_connection_mlp,
            "layer_norm": self.layer_norm,
            "use_output_scale_fn": True,
        }
        self.model = AllConditionalScoreModel(
            self.params,
            self.model_fn,
            self.sde,
            sde_init_params=sde_init_params,
            model_init_params=model_init_params,
            edge_mask_fn_params={"name": "none", "task": "vehicle_raw_time_series"},
            z_score_params=None,
        )
        self.model.set_default_condition_mask(self.posterior_condition_mask)
        self.model.set_default_node_id(self.node_id)
        self.model.set_default_edge_mask_fn(lambda *_args, **_kwargs: None)
        if self.eval_meta_data is not None:
            self.model.set_default_meta_data(self.eval_meta_data)
        self.model.set_default_sampling_kwargs(
            num_steps=self.num_diffusion_steps, sampling_method="sde"
        )

        optimizer_examples_seen = int(total_steps * self.batch_size)
        self._training_summary = {
            "train_loss": train_loss_log,
            "val_loss": val_loss_log,
            "final_loss": float(train_ema) if train_ema is not None else None,
            "best_validation_loss": (
                float(best_val_loss) if best_validation_step is not None else None
            ),
            "best_validation_step": best_validation_step,
            "train_time_s": float(train_time),
            "epochs_trained": None,
            "num_train_steps": int(total_steps),
            "training_batch_size": int(self.batch_size),
            "optimizer_examples_seen": optimizer_examples_seen,
            "optimizer_dataset_passes": float(
                optimizer_examples_seen / max(1, train_count)
            ),
            "total_params": total_params,
            "num_outer_epochs": None,
            "num_inner_epochs": None,
            "node_count": int(self.total_nodes),
            "observation_node_count": int(self.obs_nodes),
            "simformer_num_timepoints": int(self.num_timepoints),
            "embedding_trained": False,
        }
        print(
            f"[Simformer] Training complete in {train_time:.1f}s, "
            f"best validation step={best_validation_step}, "
            f"final train loss={self._training_summary['final_loss']:.4f}",
            flush=True,
        )
        return self._training_summary

    def _compute_normalized_clip_bounds(
        self, normalizer: Normalizer
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        bounds = self.cfg.param_bounds()
        low_phys = torch.tensor(
            [bounds[name][0] for name in self.cfg.active_parameters],
            dtype=torch.float32,
            device=normalizer.theta_mean.device,
        )
        high_phys = torch.tensor(
            [bounds[name][1] for name in self.cfg.active_parameters],
            dtype=torch.float32,
            device=normalizer.theta_mean.device,
        )
        clip_low = normalizer.normalize_theta(low_phys).detach().cpu().numpy()
        clip_high = normalizer.normalize_theta(high_phys).detach().cpu().numpy()
        return (
            jnp.asarray(clip_low.astype(np.float32)),
            jnp.asarray(clip_high.astype(np.float32)),
        )

    def build_posterior(
        self, normalizer: Optional[Normalizer] = None
    ) -> SimformerPosterior:
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        clip_low = clip_high = None
        if normalizer is not None:
            clip_low, clip_high = self._compute_normalized_clip_bounds(normalizer)

        self.posterior = SimformerPosterior(
            self.model,
            seq_len=self._seq_len,
            input_dim=self._input_dim,
            num_timepoints=self.num_timepoints,
            node_id=self.node_id,
            condition_mask=self.posterior_condition_mask,
            meta_data=self.eval_meta_data,
            num_steps=self.num_diffusion_steps,
            clip_low=clip_low,
            clip_high=clip_high,
        )
        return self.posterior

    def save(self, exp_dir: Path) -> Path:
        if self.params is None or self.model is None:
            raise RuntimeError("No model to save.")

        payload = {
            "params": self.params,
            "hyperparams": {
                "num_timepoints": self.num_timepoints,
                "token_dim": self.token_dim,
                "condition_token_dim": self.condition_token_dim,
                "condition_token_init_scale": self.condition_token_init_scale,
                "condition_token_init_mean": self.condition_token_init_mean,
                "condition_mode": self.condition_mode,
                "time_embedding_dim": self.time_embedding_dim,
                "num_heads": self.num_heads,
                "num_layers": self.num_layers,
                "attn_size": self.attn_size,
                "widening_factor": self.widening_factor,
                "num_hidden_layers": self.num_hidden_layers,
                "skip_connection_attn": self.skip_connection_attn,
                "skip_connection_mlp": self.skip_connection_mlp,
                "layer_norm": self.layer_norm,
                "sigma_min": self.sigma_min,
                "sigma_max": self.sigma_max,
                "T_min": self.T_min,
                "T_max": self.T_max,
                "num_diffusion_steps": self.num_diffusion_steps,
                "learning_rate": self.learning_rate,
                "min_learning_rate": self.min_learning_rate,
                "clip_max_norm": self.clip_max_norm,
                "batch_size": self.batch_size,
                "train_steps_scaling": self.train_steps_scaling,
                "min_train_steps": self.min_train_steps,
                "max_train_steps": self.max_train_steps,
                "validation_fraction": self.validation_fraction,
                "val_repeat": self.val_repeat,
                "val_every": self.val_every,
                "stop_early_count": self.stop_early_count,
                "val_error_ratio": self.val_error_ratio,
                "condition_mask_name": self.condition_mask_name,
                "condition_mask_kwargs": self.condition_mask_kwargs,
                "use_metadata": self.use_metadata,
                "rebalance_loss": self.rebalance_loss,
            },
            "dimensions": {
                "theta_dim": self.theta_dim,
                "obs_nodes": self.obs_nodes,
                "total_nodes": self.total_nodes,
                "input_dim": self._input_dim,
                "seq_len": self._seq_len,
            },
            "node_id": np.asarray(self.node_id),
            "posterior_condition_mask": np.asarray(self.posterior_condition_mask),
            "eval_meta_data": None
            if self.eval_meta_data is None
            else np.asarray(self.eval_meta_data),
            "sde_reference_data": self._sde_reference_data,
            "training_summary": self._training_summary,
        }
        model_path = exp_dir / self.model_filename
        with model_path.open("wb") as f:
            pickle.dump(payload, f)
        print(f"[Simformer] Saved model to {model_path}", flush=True)
        return model_path

    def load(self, exp_dir: Path) -> None:
        model_path = exp_dir / self.model_filename
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        with model_path.open("rb") as f:
            payload = pickle.load(f)

        self.params = payload["params"]
        for key, value in payload["hyperparams"].items():
            setattr(self, key, value)
        dims = payload["dimensions"]
        self.theta_dim = dims["theta_dim"]
        self.obs_nodes = dims["obs_nodes"]
        self.total_nodes = dims["total_nodes"]
        self._input_dim = dims["input_dim"]
        self._seq_len = dims["seq_len"]
        self.node_id = jnp.asarray(payload["node_id"], dtype=jnp.int32)
        self.posterior_condition_mask = jnp.asarray(
            payload["posterior_condition_mask"], dtype=jnp.bool_
        )
        eval_meta = payload.get("eval_meta_data")
        self.eval_meta_data = (
            None if eval_meta is None else jnp.asarray(eval_meta, dtype=jnp.float32)
        )
        self._sde_reference_data = payload.get("sde_reference_data")
        self._training_summary = payload.get("training_summary", {})
        self.dense_time_grid = jnp.linspace(0.0, 1.0, self._seq_len, dtype=jnp.float32)
        self.eval_times = jnp.linspace(0.0, 1.0, self.num_timepoints, dtype=jnp.float32)

        sde_reference = jnp.asarray(self._sde_reference_data, dtype=jnp.float32)
        self.sde, self.T_min, self.T_max, self.weight_fn, self._output_scale_fn = init_sde_related(
            sde_reference,
            name="vesde",
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            T_min=self.T_min,
            T_max=self.T_max,
        )
        _init_fn, self.model_fn = self._build_model_fn()
        self.model = AllConditionalScoreModel(
            self.params,
            self.model_fn,
            self.sde,
            sde_init_params={
                "data": sde_reference,
                "name": "vesde",
                "sigma_min": self.sigma_min,
                "sigma_max": self.sigma_max,
                "T_min": self.T_min,
                "T_max": self.T_max,
            },
            model_init_params={
                "num_nodes": self.total_nodes,
                "token_dim": self.token_dim,
                "condition_token_dim": self.condition_token_dim,
                "condition_token_init_scale": self.condition_token_init_scale,
                "condition_token_init_mean": self.condition_token_init_mean,
                "condition_mode": self.condition_mode,
                "time_embedding_dim": self.time_embedding_dim,
                "num_heads": self.num_heads,
                "num_layers": self.num_layers,
                "attn_size": self.attn_size,
                "widening_factor": self.widening_factor,
                "num_hidden_layers": self.num_hidden_layers,
                "skip_connection_attn": self.skip_connection_attn,
                "skip_connection_mlp": self.skip_connection_mlp,
                "layer_norm": self.layer_norm,
                "use_output_scale_fn": True,
            },
            edge_mask_fn_params={"name": "none", "task": "vehicle_raw_time_series"},
            z_score_params=None,
        )
        self.model.set_default_condition_mask(self.posterior_condition_mask)
        self.model.set_default_node_id(self.node_id)
        self.model.set_default_edge_mask_fn(lambda *_args, **_kwargs: None)
        if self.eval_meta_data is not None:
            self.model.set_default_meta_data(self.eval_meta_data)
        self.model.set_default_sampling_kwargs(
            num_steps=self.num_diffusion_steps, sampling_method="sde"
        )
        print(f"[Simformer] Loaded model from {model_path}", flush=True)

    @property
    def supports_lc2st(self) -> bool:
        return False
