"""
Simformer method implementation for vehicle parameter inference.

Uses score-based diffusion models with transformer architecture to learn
the joint distribution p(θ, x) and sample arbitrary conditionals.

This method is JAX-based (from simformer library) wrapped with a PyTorch-compatible interface.
"""

import sys
import time
import pickle
from pathlib import Path
from typing import Any, Dict, Optional
from functools import partial

import numpy as np
import torch
import torch.nn as nn

# JAX imports
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import Array

# Patch jax.linear_util for newer JAX (>=0.5) where it was moved to jax.extend
if not hasattr(jax, "linear_util"):
    try:
        from jax.extend import linear_util

        jax.linear_util = linear_util
    except ImportError:
        from jax._src import linear_util

        jax.linear_util = linear_util

# Use new-style typed PRNG keys (jax.random.key) when available (JAX >=0.4.16),
# falling back to jax.random.PRNGKey for older versions.
_make_key = getattr(jrandom, "key", jrandom.PRNGKey)

# Patch jax.random.default_prng_impl for JAX 0.8+ (where it was removed)
# haiku 0.0.16 calls this internally, so we need to provide a stub
if not hasattr(jrandom, "default_prng_impl"):

    class _StubPRNGImpl:
        @staticmethod
        def seed_and_split(seed):
            # Return a key that will work with both old and new haiku
            if hasattr(_make_key(0), "dtype"):
                # New typed key format
                k = _make_key(seed if isinstance(seed, int) else 0)
            else:
                # Old array format
                k = jrandom.PRNGKey(seed if isinstance(seed, int) else 0)
            return k, _StubPRNGImpl.split(k)

        @staticmethod
        def split(key):
            return jrandom.split(key)[0]

    jrandom.default_prng_impl = _StubPRNGImpl

# Simformer imports (add to path if needed)
SIMFORMER_PATH = Path(__file__).parent.parent / "simformer-main" / "src"
if str(SIMFORMER_PATH) not in sys.path:
    sys.path.insert(0, str(SIMFORMER_PATH / "probjax"))
    sys.path.insert(0, str(SIMFORMER_PATH / "scoresbibm"))

import haiku as hk
import optax

from probjax.nn.transformers import Transformer
from probjax.nn.helpers import GaussianFourierEmbedding
from probjax.nn.loss_fn import denoising_score_matching_loss
from probjax.distributions.sde import VESDE
from probjax.distributions import Independent
from probjax.distributions.discrete import Empirical
from probjax.utils.sdeint import sdeint

from .base import BaseMethod
from models.models import build_embedding


class EmbeddingWrapperJAX:
    """
    Wrapper to use PyTorch embedding network in a JAX-compatible way.
    Pre-embeds all data before training.
    """

    def __init__(self, embedding_net: nn.Module, device: torch.device):
        self.embedding_net = embedding_net
        self.device = device
        self._output_dim = None

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> np.ndarray:
        """Embed observations using PyTorch network, return numpy."""
        self.embedding_net.eval()
        x = x.to(self.device)
        embedded = self.embedding_net(x)
        self._output_dim = embedded.shape[-1]
        return embedded.cpu().numpy()

    @property
    def output_dim(self) -> int:
        if self._output_dim is None:
            raise RuntimeError("Embed data first to determine output dim")
        return self._output_dim


class SimformerPosterior:
    """
    Wrapper providing a sbi-compatible posterior interface for Simformer.

    Allows sampling via: posterior.sample((num_samples,), x=x_obs)
    """

    def __init__(
        self,
        params: Dict,
        model_fn,
        sde: VESDE,
        embedding_wrapper: EmbeddingWrapperJAX,
        theta_dim: int,
        embedding_dim: int,
        T_min: float,
        T_max: float,
        prior_bounds: Dict[str, tuple],
        active_parameters: tuple,
        num_steps: int = 500,
    ):
        self.params = params
        self.model_fn = model_fn
        self.sde = sde
        self.embedding_wrapper = embedding_wrapper
        self.theta_dim = theta_dim
        self.embedding_dim = embedding_dim
        self.T_min = T_min
        self.T_max = T_max
        self.num_steps = num_steps
        self.prior_bounds = prior_bounds
        self.active_parameters = active_parameters

        # Total nodes = theta_dim + embedding_dim
        self.total_nodes = theta_dim + embedding_dim
        self.node_ids = jnp.arange(self.total_nodes)

        # Condition mask: False for theta (to sample), True for observations (given)
        self.posterior_condition_mask = jnp.array(
            [False] * theta_dim + [True] * embedding_dim, dtype=jnp.bool_
        )

        # Pre-compute marginal statistics
        self.marginal_end_std = jnp.squeeze(sde.marginal_stddev(jnp.array([T_max])))
        self.marginal_end_mean = jnp.squeeze(sde.marginal_mean(jnp.array([T_max])))

        # Build prior bounds as JAX arrays for clipping
        self._prior_low = jnp.array([prior_bounds[p][0] for p in active_parameters])
        self._prior_high = jnp.array([prior_bounds[p][1] for p in active_parameters])

    def _init_backward_sde(self, x_o_embedded: Array):
        """Initialize backward SDE for sampling."""
        node_ids = self.node_ids
        condition_mask = self.posterior_condition_mask

        def drift_backward(t, x):
            t_broadcast = jnp.atleast_1d(self.T_max - t)
            x_full = x.reshape(-1, self.total_nodes, 1)

            # Set conditioned values
            x_full = x_full.at[:, self.theta_dim :, 0].set(x_o_embedded)

            score = self.model_fn(
                self.params, t_broadcast, x_full, node_ids, condition_mask
            )
            score = score.reshape(x.shape)

            t_actual = self.T_max - t
            drift = (
                self.sde.drift(t_actual, x)
                - self.sde.diffusion(t_actual, x) ** 2 * score
            )

            # Zero out drift for conditioned dimensions
            drift = drift.at[self.theta_dim :].set(0.0)
            return -drift

        def diffusion_backward(t, x):
            t_actual = self.T_max - t
            diff = self.sde.diffusion(t_actual, x)
            # Zero out diffusion for conditioned dimensions
            diff = diff.at[self.theta_dim :].set(0.0)
            return diff

        return drift_backward, diffusion_backward

    @partial(jax.jit, static_argnums=(0, 1, 4))
    def _sample_jax(
        self,
        num_samples: int,
        x_o_embedded: Array,
        key: jrandom.PRNGKey,
        num_steps: int,
    ) -> Array:
        """JIT-compiled sampling function."""
        key1, key2 = jrandom.split(key)

        # Initialize at noise distribution
        x_T = jax.random.normal(key1, (num_samples, self.total_nodes))
        x_T = x_T * self.marginal_end_std + self.marginal_end_mean

        # Set conditioned values
        x_T = x_T.at[:, self.theta_dim :].set(x_o_embedded)

        # Backward SDE
        drift, diffusion = self._init_backward_sde(x_o_embedded)
        keys = jrandom.split(key2, num_samples)

        ts = jnp.linspace(0.0, self.T_max - self.T_min, num_steps)

        ys = jax.vmap(
            lambda k, x0: sdeint(k, drift, diffusion, x0, ts, noise_type="diagonal"),
            in_axes=(0, 0),
        )(keys, x_T)

        # Extract theta samples (first theta_dim dimensions)
        theta_samples = ys[:, -1, : self.theta_dim]
        return theta_samples

    def sample(
        self,
        shape: tuple,
        x: torch.Tensor,
        seed: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Sample from posterior p(θ|x).

        Args:
            shape: Tuple like (num_samples,)
            x: Observation tensor (B, T, D) or (T, D)
            seed: Random seed

        Returns:
            Samples tensor (num_samples, theta_dim)
        """
        num_samples = shape[0] if isinstance(shape, tuple) else shape

        # Handle batched vs single observation
        if x.ndim == 2:
            x = x.unsqueeze(0)  # (1, T, D)

        # Embed observation using PyTorch network
        x_embedded = self.embedding_wrapper.embed(x)  # (1, embedding_dim)
        x_o_embedded = jnp.array(x_embedded[0])  # (embedding_dim,)

        # Sample using JAX
        seed = seed or int(time.time() * 1000) % (2**31)
        key = _make_key(seed)

        theta_samples = self._sample_jax(num_samples, x_o_embedded, key, self.num_steps)

        # Clip to prior bounds
        theta_samples = jnp.clip(theta_samples, self._prior_low, self._prior_high)

        # Convert to PyTorch
        return torch.from_numpy(np.array(theta_samples)).float()


class SimformerMethod(BaseMethod):
    """
    Simformer: All-in-one simulation-based inference using score transformers.

    Learns the joint distribution p(θ, z) where z is an embedded observation,
    then samples posteriors by conditioning on z.

    Key features:
    - Transformer-based score function
    - Variance Exploding SDE (VESDE) for diffusion
    - Learns to sample any conditional from joint
    """

    name = "simformer"

    @property
    def model_filename(self) -> str:
        """Return the filename used for saving the model."""
        return "simformer_model.pkl"

    def __init__(
        self,
        cfg,
        prior,
        device: torch.device,
        # Simformer-specific hyperparameters
        token_dim: int = 40,
        condition_token_dim: int = 10,
        time_embedding_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 6,
        attn_size: int = 10,
        widening_factor: int = 3,
        sigma_min: float = 0.01,
        sigma_max: float = 15.0,
        T_min: float = 0.02,
        T_max: float = 1.0,
        num_diffusion_steps: int = 500,
        learning_rate: float = 1e-3,
        num_train_steps: int = 50000,
        batch_size: int = 1024,
    ):
        super().__init__(cfg, prior, device)

        # Store hyperparameters
        self.token_dim = token_dim
        self.condition_token_dim = condition_token_dim
        self.time_embedding_dim = time_embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.attn_size = attn_size
        self.widening_factor = widening_factor
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.T_min = T_min
        self.T_max = T_max
        self.num_diffusion_steps = num_diffusion_steps
        self.learning_rate = learning_rate
        self.num_train_steps = num_train_steps
        self.batch_size = batch_size

        # To be set during build
        self.embedding_net = None
        self.embedding_wrapper = None
        self.theta_dim = None
        self.embedding_dim = None
        self.total_nodes = None
        self.params = None
        self.sde = None
        self._training_summary = {}

    def build(self, input_dim: int, seq_len: int) -> None:
        """Build Simformer model with embedding network."""
        self._input_dim = input_dim
        self._seq_len = seq_len
        self.theta_dim = len(self.cfg.active_parameters)

        # Build PyTorch embedding network (same as other methods)
        self.embedding_net = build_embedding(self.cfg, input_dim, seq_len, self.device)
        self.embedding_wrapper = EmbeddingWrapperJAX(self.embedding_net, self.device)

        # Embedding dimension will be determined when we first embed data
        self.embedding_dim = self.cfg.embedding_output_dim
        self.total_nodes = self.theta_dim + self.embedding_dim

        print(
            f"[Simformer] Built with {self.theta_dim} params + {self.embedding_dim} embedding = {self.total_nodes} nodes"
        )

    def _build_score_model(self):
        """Build the JAX score model (transformer)."""
        total_nodes = self.total_nodes
        token_dim = self.token_dim
        condition_token_dim = self.condition_token_dim
        time_embedding_dim = self.time_embedding_dim
        num_heads = self.num_heads
        num_layers = self.num_layers
        attn_size = self.attn_size
        widening_factor = self.widening_factor
        sde = self.sde

        def output_scale_fn(t, x):
            """Scale output by marginal std to improve training."""
            scale = jnp.clip(sde.marginal_stddev(t, jnp.ones_like(x)), 1e-2, None)
            return (1 / scale * x).reshape(x.shape)

        def model(t: Array, x: Array, node_ids: Array, condition_mask: Array):
            """Simformer score model."""
            batch_size, seq_len, _ = x.shape
            condition_mask = condition_mask.astype(jnp.bool_).reshape(-1, seq_len, 1)
            node_ids = node_ids.reshape(-1, seq_len)
            t = t.reshape(-1, 1, 1)

            # Time embedding
            embedding_time = GaussianFourierEmbedding(time_embedding_dim)
            time_embeddings = embedding_time(t)

            # Value embedding (simple repeat)
            embedding_net_value = lambda x: jnp.repeat(x, token_dim, axis=-1)

            # Node ID embedding
            embedding_net_id = hk.Embed(
                total_nodes, token_dim, w_init=hk.initializers.RandomNormal(stddev=3.0)
            )

            # Condition embedding
            condition_embedding = hk.get_parameter(
                "condition_embedding",
                shape=(1, 1, condition_token_dim),
                init=hk.initializers.RandomNormal(stddev=0.5),
            )
            condition_embedding = condition_embedding * condition_mask
            condition_embedding = jnp.broadcast_to(
                condition_embedding, (batch_size, seq_len, condition_token_dim)
            )

            # Embed inputs
            value_embeddings = embedding_net_value(x)
            id_embeddings = embedding_net_id(node_ids)
            value_embeddings, id_embeddings = jnp.broadcast_arrays(
                value_embeddings, id_embeddings
            )

            # Concatenate to form tokens
            x_encoded = jnp.concatenate(
                [value_embeddings, id_embeddings, condition_embedding], axis=-1
            )

            # Transformer
            transformer = Transformer(
                num_heads=num_heads,
                num_layers=num_layers,
                attn_size=attn_size,
                widening_factor=widening_factor,
            )

            h = transformer(x_encoded, context=time_embeddings)
            out = hk.Linear(1)(h)
            out = output_scale_fn(t, out)
            return out

        return model

    def train(
        self,
        theta_train: torch.Tensor,
        x_train: torch.Tensor,
    ) -> Dict[str, Any]:
        """Train Simformer on joint (θ, x) data."""
        if self.embedding_net is None:
            raise RuntimeError("Method not built. Call build() first.")

        train_start = time.time()

        # 1. Embed observations using PyTorch network
        print("[Simformer] Embedding observations...")
        x_embedded = self.embedding_wrapper.embed(x_train)  # (N, embedding_dim)
        self.embedding_dim = x_embedded.shape[1]
        self.total_nodes = self.theta_dim + self.embedding_dim

        # 2. Combine theta and embedded x into joint data
        theta_np = theta_train.cpu().numpy()  # (N, theta_dim)
        joint_data = np.concatenate([theta_np, x_embedded], axis=1)  # (N, total_nodes)
        joint_data = joint_data[:, :, np.newaxis]  # (N, total_nodes, 1)
        joint_data_jax = jnp.array(joint_data)

        print(f"[Simformer] Joint data shape: {joint_data_jax.shape}")

        # 3. Build SDE
        p0 = Independent(Empirical(joint_data_jax), 1)
        self.sde = VESDE(p0, sigma_min=self.sigma_min, sigma_max=self.sigma_max)

        # 4. Build and initialize model
        model_def = self._build_score_model()
        init_fn, model_fn = hk.without_apply_rng(hk.transform(model_def))

        node_ids = jnp.arange(self.total_nodes)
        key = _make_key(self.cfg.random_seed)

        # Initialize parameters
        dummy_condition = jnp.zeros(self.total_nodes, dtype=jnp.bool_)
        self.params = init_fn(
            key,
            jnp.ones(joint_data_jax.shape[0]),
            joint_data_jax,
            node_ids,
            dummy_condition,
        )
        self.model_fn = model_fn

        total_params = sum(x.size for x in jax.tree_util.tree_leaves(self.params))
        print(f"[Simformer] Model parameters: {total_params:,}")

        # 5. Define loss function
        def weight_fn(t):
            return jnp.clip(self.sde.diffusion(t, jnp.ones((1, 1, 1))) ** 2, 1e-4)

        def loss_fn(params, key, batch_size=self.batch_size):
            keys = jrandom.split(key, 5)

            # Sample batch
            indices = jrandom.randint(
                keys[0], (batch_size,), 0, joint_data_jax.shape[0]
            )
            batch_xs = joint_data_jax[indices]

            # Random times
            times = jrandom.uniform(
                keys[1], (batch_size, 1, 1), minval=self.T_min, maxval=self.T_max
            )

            # Random condition mask (for learning all conditionals)
            condition_mask = jrandom.bernoulli(
                keys[2], 0.333, shape=(batch_size, self.total_nodes)
            )
            all_conditioned = jnp.all(condition_mask, axis=-1, keepdims=True)
            condition_mask = (
                condition_mask * ~all_conditioned
            )  # Avoid conditioning on everything
            condition_mask = condition_mask[..., None]

            loss = denoising_score_matching_loss(
                params,
                keys[3],
                times,
                batch_xs,
                condition_mask,
                model_fn=model_fn,
                mean_fn=self.sde.marginal_mean,
                std_fn=self.sde.marginal_stddev,
                weight_fn=weight_fn,
                node_ids=node_ids,
                condition_mask=condition_mask,
            )
            return loss

        # 6. Training loop
        print(f"[Simformer] Training for {self.num_train_steps} steps...")
        optimizer = optax.adam(self.learning_rate)
        opt_state = optimizer.init(self.params)

        @jax.jit
        def update(params, opt_state, key):
            loss, grads = jax.value_and_grad(loss_fn)(params, key)
            updates, opt_state = optimizer.update(grads, opt_state, params=params)
            params = optax.apply_updates(params, updates)
            return loss, params, opt_state

        losses = []
        key = _make_key(self.cfg.train_seed or 0)
        print_every = max(1, self.num_train_steps // 20)

        for step in range(self.num_train_steps):
            key, subkey = jrandom.split(key)
            loss, self.params, opt_state = update(self.params, opt_state, subkey)
            losses.append(float(loss))

            if (step + 1) % print_every == 0:
                avg_loss = np.mean(losses[-print_every:])
                print(f"  Step {step+1}/{self.num_train_steps}: loss = {avg_loss:.4f}")

        train_time = time.time() - train_start

        self._training_summary = {
            "train_loss": losses,
            "final_loss": float(np.mean(losses[-100:])),
            "train_time_s": train_time,
            "total_params": total_params,
            "num_train_steps": self.num_train_steps,
            "embedding_dim": self.embedding_dim,
        }

        print(
            f"[Simformer] Training complete in {train_time:.1f}s, final loss = {self._training_summary['final_loss']:.4f}"
        )
        return self._training_summary

    def build_posterior(self) -> SimformerPosterior:
        """Build posterior object for sampling."""
        if self.params is None:
            raise RuntimeError("Model not trained. Call train() first.")

        self.posterior = SimformerPosterior(
            params=self.params,
            model_fn=self.model_fn,
            sde=self.sde,
            embedding_wrapper=self.embedding_wrapper,
            theta_dim=self.theta_dim,
            embedding_dim=self.embedding_dim,
            T_min=self.T_min,
            T_max=self.T_max,
            prior_bounds=self.cfg.param_bounds(),
            active_parameters=self.cfg.active_parameters,
            num_steps=self.num_diffusion_steps,
        )
        return self.posterior

    def save(self, exp_dir: Path) -> Path:
        """Save model and embedding network."""
        if self.params is None:
            raise RuntimeError("No model to save.")

        # Save JAX params and model info
        save_data = {
            "params": self.params,
            "hyperparams": {
                "token_dim": self.token_dim,
                "condition_token_dim": self.condition_token_dim,
                "time_embedding_dim": self.time_embedding_dim,
                "num_heads": self.num_heads,
                "num_layers": self.num_layers,
                "attn_size": self.attn_size,
                "widening_factor": self.widening_factor,
                "sigma_min": self.sigma_min,
                "sigma_max": self.sigma_max,
                "T_min": self.T_min,
                "T_max": self.T_max,
                "num_diffusion_steps": self.num_diffusion_steps,
            },
            "dimensions": {
                "theta_dim": self.theta_dim,
                "embedding_dim": self.embedding_dim,
                "total_nodes": self.total_nodes,
                "input_dim": self._input_dim,
                "seq_len": self._seq_len,
            },
            "training_summary": self._training_summary,
        }

        model_path = exp_dir / self.model_filename
        with open(model_path, "wb") as f:
            pickle.dump(save_data, f)

        # Save PyTorch embedding network
        emb_path = exp_dir / "simformer_embedding.pt"
        torch.save(self.embedding_net.state_dict(), emb_path)

        print(f"[Simformer] Saved model to {model_path}")
        return model_path

    def load(self, exp_dir: Path) -> None:
        """Load model and embedding network."""
        model_path = exp_dir / self.model_filename
        emb_path = exp_dir / "simformer_embedding.pt"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load JAX params
        with open(model_path, "rb") as f:
            save_data = pickle.load(f)

        self.params = save_data["params"]

        # Restore hyperparameters
        hp = save_data["hyperparams"]
        for key, val in hp.items():
            setattr(self, key, val)

        # Restore dimensions
        dims = save_data["dimensions"]
        self.theta_dim = dims["theta_dim"]
        self.embedding_dim = dims["embedding_dim"]
        self.total_nodes = dims["total_nodes"]
        self._input_dim = dims["input_dim"]
        self._seq_len = dims["seq_len"]

        self._training_summary = save_data.get("training_summary", {})

        # Rebuild embedding network and load weights
        self.embedding_net = build_embedding(
            self.cfg, self._input_dim, self._seq_len, self.device
        )
        if emb_path.exists():
            self.embedding_net.load_state_dict(
                torch.load(emb_path, map_location=self.device)
            )
        self.embedding_wrapper = EmbeddingWrapperJAX(self.embedding_net, self.device)

        # Rebuild model function and SDE
        # Need dummy data to rebuild SDE
        dummy_data = jnp.zeros((2, self.total_nodes, 1))
        p0 = Independent(Empirical(dummy_data), 1)
        self.sde = VESDE(p0, sigma_min=self.sigma_min, sigma_max=self.sigma_max)

        model_def = self._build_score_model()
        _, self.model_fn = hk.without_apply_rng(hk.transform(model_def))

        print(f"[Simformer] Loaded model from {model_path}")

    @property
    def supports_lc2st(self) -> bool:
        """Simformer does not support LC2ST-NF."""
        return False

    def sample_posterior(
        self, x_obs: torch.Tensor, num_samples: int, **kwargs
    ) -> torch.Tensor:
        """Sample from posterior using the Simformer model."""
        if self.posterior is None:
            self.build_posterior()
        return self.posterior.sample((num_samples,), x=x_obs, **kwargs)
