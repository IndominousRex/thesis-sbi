"""
FNPE (Flow-based Neural Posterior Estimation) method implementation.

Uses the MarkovSBI framework with JAX for score-based posterior estimation
with factorized/autoregressive score functions.
"""

import time
import pickle
from pathlib import Path
from typing import Any, Dict, Optional
import json

import numpy as np
import jax
import jax.numpy as jnp
import optax

from .base import BaseMethod

# MarkovSBI imports
from markovsbi.tasks import VehicleDynamicsTask
from markovsbi.utils.sde_utils import init_sde
from markovsbi.models.simple_scoremlp import build_score_mlp, precondition_functions
from markovsbi.models.train_utils import build_batch_sampler, build_loss_fn
from markovsbi.sampling.score_fn import FNPEScoreFn, UncorrectedScoreFn
from markovsbi.sampling.sample import Diffuser
from markovsbi.sampling.kernels import EulerMaruyama


class FNPEPosterior:
    """
    Wrapper to provide a consistent sampling interface for FNPE.

    Adapts the JAX-based FNPE sampler to the torch-like posterior interface
    used by NPE/NPSE.
    """

    def __init__(
        self,
        sampler: Diffuser,
        task: VehicleDynamicsTask,
        key: jax.random.PRNGKey,
    ):
        self.sampler = sampler
        self.task = task
        self.key = key

    def sample(self, sample_shape: tuple, x: Any, **kwargs) -> np.ndarray:
        """
        Sample from the posterior.

        Args:
            sample_shape: Tuple specifying number of samples (N,)
            x: Observation tensor/array (can be torch or numpy)

        Returns:
            Posterior samples as numpy array, shape (N, d_theta)
        """
        import torch

        num_samples = sample_shape[0]

        # Convert observation to JAX array
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.asarray(x)

        # Normalize observation
        x_norm = self.task.normalize_x(jnp.asarray(x_np.squeeze()))

        # Sample
        self.key, *sample_keys = jax.random.split(self.key, num_samples + 1)
        sample_keys = jnp.stack(sample_keys)

        samples_norm = jax.vmap(self.sampler.sample, in_axes=(0, None))(
            sample_keys, x_norm
        )
        samples_norm = jax.block_until_ready(samples_norm)

        # Unnormalize
        samples_phys = jax.vmap(self.task.unnormalize_theta)(samples_norm)

        return np.array(samples_phys)


class FNPEMethod(BaseMethod):
    """
    Flow-based Neural Posterior Estimation using MarkovSBI.

    This method uses JAX for efficient score network training and diffusion
    sampling with factorized score functions.
    """

    name = "fnpe"
    model_filename = "params.pkl"

    def __init__(
        self,
        cfg,
        prior,
        device,  # Note: FNPE uses JAX, device is mainly for compatibility
        hidden_dim: int = 128,
        num_hidden: int = 5,
        model_type: str = "gru",
        num_epochs: int = 20,
        steps_per_epoch: int = 10000,
        batch_size: int = 256,
        num_diffusion_steps: int = 500,
        score_fn_type: str = "fnpe",
    ):
        # Note: FNPE doesn't use torch prior/device directly
        super().__init__(cfg, prior, device)

        # FNPE-specific config
        self.hidden_dim = hidden_dim
        self.num_hidden = num_hidden
        self.model_type = model_type
        self.num_epochs = num_epochs
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.num_diffusion_steps = num_diffusion_steps
        self.score_fn_type = score_fn_type

        # Will be set during build/train
        self.task = None
        self.sde = None
        self.weight_fn = None
        self.score_net = None
        self.params = None
        self.sampler = None
        self._training_summary = {}
        self._key = jax.random.PRNGKey(cfg.random_seed)

    def build(self, input_dim: int, seq_len: int) -> None:
        """
        Build FNPE task and data structures.

        Note: FNPE builds its own task that includes simulator.
        """
        self._input_dim = input_dim
        self._seq_len = seq_len

        # Create task (this includes the simulator and normalization)
        self.task = VehicleDynamicsTask(
            cfg=self.cfg,
            normalize=True,
            seed=self.cfg.random_seed,
        )

    def train(
        self,
        theta_train=None,  # Ignored - FNPE generates its own data
        x_train=None,
        num_simulations: Optional[int] = None,
        T_obs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Train FNPE score network.

        Note: FNPE generates its own training data via the task.
        The theta_train/x_train arguments are ignored for consistency
        with the base interface.
        """
        if self.task is None:
            raise RuntimeError("Method not built. Call build() first.")

        num_sim = num_simulations or self.cfg.num_simulations
        t_obs = T_obs or self._seq_len

        print(f"[FNPE] Generating {num_sim} training trajectories (T={t_obs})...")

        # Generate data
        self._key, key_data = jax.random.split(self._key)
        data = self.task.get_data(key_data, num_sim, t_obs)

        print(
            f"[FNPE] Data shapes: thetas={data['thetas'].shape}, xs={data['xs'].shape}"
        )

        # Initialize SDE
        print("[FNPE] Initializing SDE...")
        self.sde, self.weight_fn = init_sde(data)

        # Train score network
        print(f"[FNPE] Training score network ({self.num_epochs} epochs)...")
        train_start = time.time()

        self.params, self.score_net, losses = self._train_score_network(data, t_obs)

        train_time = time.time() - train_start

        # Setup sampler
        print("[FNPE] Setting up sampler...")
        self._setup_sampler()

        # Store model reference
        self.model = self.params

        self._training_summary = {
            "train_loss": losses,
            "final_loss": losses[-1] if losses else None,
            "train_time_s": train_time,
            "num_simulations": num_sim,
            "T_obs": t_obs,
        }

        return self._training_summary

    def _train_score_network(self, data: Dict, window_size: int):
        """Internal method to train score network."""
        key = self._key
        key, key_init = jax.random.split(key)

        d = data["thetas"].shape[1]

        # Preconditioning
        c_in, c_noise, c_out = precondition_functions(self.sde)

        # Build score network
        init_fn, score_net = build_score_mlp(
            window_size=window_size,
            num_hidden=self.num_hidden,
            hidden_dim=self.hidden_dim,
            x_o_processing=self.model_type,
            c_in=c_in,
            c_noise=c_noise,
            c_out=c_out,
        )

        # Build batch sampler and loss
        batch_sampler = build_batch_sampler(data)
        loss_fn = build_loss_fn(
            "dsm", score_net, self.sde, self.weight_fn, control_variate=True
        )

        # Initialize
        theta_batch, x_batch = batch_sampler(key_init, self.batch_size)
        params = init_fn(key_init, jnp.ones((self.batch_size,)), theta_batch, x_batch)

        n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
        print(f"[FNPE] Score network: {n_params:,} parameters")

        # Optimizer
        total_steps = self.num_epochs * self.steps_per_epoch
        schedule = optax.cosine_onecycle_schedule(total_steps, self.cfg.learning_rate)
        optimizer = optax.chain(
            optax.adaptive_grad_clip(10.0),
            optax.adamw(schedule),
        )
        opt_state = optimizer.init(params)

        # JIT update
        @jax.jit
        def update(params, rng, opt_state, theta_batch, x_batch):
            loss, grads = jax.value_and_grad(loss_fn)(params, rng, theta_batch, x_batch)
            updates, opt_state = optimizer.update(grads, opt_state, params=params)
            params = optax.apply_updates(params, updates)
            return loss, params, opt_state

        # JIT warmup
        print("[FNPE] JIT compiling...")
        key, key_batch, key_loss = jax.random.split(key, 3)
        theta_batch, x_batch = batch_sampler(key_batch, self.batch_size)
        loss, params, opt_state = update(
            params, key_loss, opt_state, theta_batch, x_batch
        )
        _ = float(loss)  # Block until done

        # Training loop
        losses = []
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            for step in range(self.steps_per_epoch):
                key, key_batch, key_loss = jax.random.split(key, 3)
                theta_batch, x_batch = batch_sampler(key_batch, self.batch_size)
                loss, params, opt_state = update(
                    params, key_loss, opt_state, theta_batch, x_batch
                )
                epoch_loss += float(loss) / self.steps_per_epoch

            losses.append(epoch_loss)
            print(f"[FNPE] Epoch {epoch+1}/{self.num_epochs}: Loss = {epoch_loss:.6f}")

        self._key = key
        return params, score_net, losses

    def _setup_sampler(self):
        """Set up the diffusion sampler."""
        prior_norm = self.task.get_normalized_prior()

        if self.score_fn_type.lower() == "fnpe":
            score_fn = FNPEScoreFn(self.score_net, self.params, self.sde, prior_norm)
        else:
            score_fn = UncorrectedScoreFn(
                self.score_net, self.params, self.sde, prior_norm
            )

        kernel = EulerMaruyama(score_fn)
        time_grid = jnp.linspace(
            self.sde.T_min, self.sde.T_max, self.num_diffusion_steps
        )
        self.sampler = Diffuser(kernel, time_grid, self.task.input_shape)

    def build_posterior(self) -> FNPEPosterior:
        """Build posterior wrapper for sampling."""
        if self.sampler is None:
            raise RuntimeError("Model not trained. Call train() first.")

        self._key, key_posterior = jax.random.split(self._key)
        self.posterior = FNPEPosterior(self.sampler, self.task, key_posterior)
        return self.posterior

    def save(self, exp_dir: Path) -> Path:
        """Save FNPE artifacts."""
        if self.params is None:
            raise RuntimeError("No model to save.")

        # Save params
        model_path = exp_dir / self.model_filename
        with open(model_path, "wb") as f:
            pickle.dump(self.params, f)

        # Save normalization stats
        norm_stats = self.task.get_normalization_stats()
        norm_stats_json = {k: v.tolist() for k, v in norm_stats.items()}
        with open(exp_dir / "normalization_stats.json", "w") as f:
            json.dump(norm_stats_json, f, indent=2)

        return model_path

    def load(self, exp_dir: Path) -> None:
        """Load FNPE artifacts."""
        model_path = exp_dir / self.model_filename
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        with open(model_path, "rb") as f:
            self.params = pickle.load(f)

        self.model = self.params

        # Need to rebuild sampler - this requires the score_net architecture
        # which requires re-running build and partial train setup
        raise NotImplementedError(
            "FNPE loading requires re-building score network architecture. "
            "This is not yet fully supported."
        )

    @property
    def supports_lc2st(self) -> bool:
        """FNPE does not support LC2ST-NF."""
        return False
