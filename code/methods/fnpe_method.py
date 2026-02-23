"""
FNPE (Flow-based Neural Posterior Estimation) method implementation.

Uses the MarkovSBI framework with JAX for score-based posterior estimation
with factorized/autoregressive score functions.
"""

import sys
import time
import pickle
from pathlib import Path
from typing import Any, Dict, Optional
import json

import numpy as np
import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

from .base import BaseMethod

# MarkovSBI imports
from markovsbi.tasks import VehicleDynamicsTask
from markovsbi.utils.sde_utils import init_sde
from markovsbi.models.simple_scoremlp import build_score_mlp, precondition_functions
from markovsbi.models.train_utils import build_batch_sampler, build_loss_fn
from markovsbi.sampling.score_fn import (
    FNPEScoreFn,
    UncorrectedScoreFn,
    GaussCorrectedScoreFn,
)
from markovsbi.sampling.sample import Diffuser
from markovsbi.sampling.kernels import EulerMaruyama


class FNPEPosterior:
    """
    Wrapper to provide a consistent sampling interface for FNPE.

    Adapts the JAX-based FNPE sampler to the torch-like posterior interface
    used by NPE/NPSE. Returns torch tensors in NORMALIZED space by default,
    allowing seamless integration with existing NPE/NPSE evaluation pipelines.

    Uses the original FNPE score formula: (1-N)*prior_score + sum(likelihood_scores)
    """

    def __init__(
        self,
        sampler: Diffuser,
        task: VehicleDynamicsTask,
        key: jax.random.PRNGKey,
        normalizer=None,  # External normalizer for interface compatibility
        obs_dim: int = 9,  # Number of observation channels (without controls)
    ):
        self.sampler = sampler
        self.task = task
        self.key = key
        self.normalizer = normalizer  # Used to match NPE/NPSE interface
        self.obs_dim = obs_dim  # Observation dimension (controls appended separately)

    def sample(
        self,
        sample_shape: tuple,
        x,
        return_physical: bool = False,
        **kwargs,
    ):
        """
        Sample from the posterior.

        By default, this method matches NPE/NPSE interface:
        - Expects NORMALIZED observations (from normalizer.normalize_x)
        - Returns NORMALIZED samples as torch.Tensor

        Args:
            sample_shape: Tuple specifying number of samples (N,)
            x: Observation tensor/array (can be torch or numpy).
               Default: Expects NORMALIZED data (same as NPE/NPSE).
               If return_physical=True: Can be either normalized or physical.
            return_physical: If True, returns physical units as numpy (legacy mode).
                           If False (default), returns normalized torch tensor (NPE-compatible).

        Returns:
            If return_physical=True: numpy array, shape (N, d_theta) in PHYSICAL units
            If return_physical=False: torch.Tensor, shape (N, d_theta) in NORMALIZED units
        """
        import torch

        num_samples = sample_shape[0]

        # Convert observation to numpy
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.asarray(x)

        # Squeeze and ensure 2D: (T, D)
        x_squeezed = x_np.squeeze()
        if x_squeezed.ndim == 1:
            # Single feature time series
            x_squeezed = x_squeezed.reshape(-1, 1)

        # If we have an external normalizer, input is NORMALIZED - unnormalize first
        # This makes FNPE accept the same input as NPE/NPSE
        if self.normalizer is not None and not return_physical:
            x_tensor = torch.tensor(x_squeezed, dtype=torch.float32)
            if x_tensor.ndim == 2:
                x_tensor = x_tensor.unsqueeze(0)  # (1, T, D)

            if self.task.obs_only:
                obs_std_cpu = self.normalizer.obs_std.cpu()
                obs_mean_cpu = self.normalizer.obs_mean.cpu()
                obs_phys = (
                    x_tensor.cpu() * (obs_std_cpu + self.normalizer.eps) + obs_mean_cpu
                )
                x_squeezed = obs_phys.squeeze(0).numpy()
            else:
                expected_dim = self.obs_dim + getattr(self.task, "_ctrl_dim", 0)
                if x_tensor.shape[-1] != expected_dim:
                    raise ValueError(
                        f"FNPE expects {expected_dim} dims (obs+ctrl), "
                        f"got {x_tensor.shape[-1]}"
                    )
                norm_device = self.normalizer.obs_mean.device
                x_phys = self.normalizer.unnormalize_x(
                    x_tensor.to(norm_device), self.obs_dim
                )
                x_squeezed = x_phys.squeeze(0).detach().cpu().numpy()

        # FNPE expects PHYSICAL data and normalizes it using its own task stats
        x_jax = jnp.asarray(x_squeezed)

        # FNPE internal normalization (using task's own stats)
        x_norm = self.task.normalize_x(x_jax)

        # Check if score function requires hyperparameter estimation (e.g., GaussCorrectedScoreFn)
        score_fn = self.sampler.kernel.score_fn
        if (
            hasattr(score_fn, "requires_hyperparameters")
            and score_fn.requires_hyperparameters
        ):
            # Cache: skip re-estimation if observation hasn't changed
            x_hash = hash(x_norm.tobytes())
            if (
                not hasattr(self, "_hyper_cache_hash")
                or self._hyper_cache_hash != x_hash
            ):
                self.key, key_hyper = jax.random.split(self.key)
                score_fn.estimate_hyperparameters(
                    x_norm, self.sampler.theta_shape, key_hyper
                )
                self._hyper_cache_hash = x_hash

        # Sample from diffusion
        self.key, *sample_keys = jax.random.split(self.key, num_samples + 1)
        sample_keys = jnp.stack(sample_keys)

        samples_norm_jax = jax.vmap(self.sampler.sample, in_axes=(0, None))(
            sample_keys, x_norm
        )
        samples_norm_jax = jax.block_until_ready(samples_norm_jax)

        # Unnormalize using FNPE's task stats -> physical units
        samples_phys = jax.vmap(self.task.unnormalize_theta)(samples_norm_jax)
        samples_phys_np = np.array(samples_phys)

        if return_physical:
            # Legacy mode: return physical numpy array
            return samples_phys_np

        # Default mode: Convert to normalized torch tensor (same as NPE/NPSE output)
        if self.normalizer is not None:
            # Normalize on CPU, then move to same device as normalizer
            samples_phys_torch = torch.tensor(samples_phys_np, dtype=torch.float32)
            theta_mean_cpu = self.normalizer.theta_mean.cpu()
            theta_std_cpu = self.normalizer.theta_std.cpu()
            samples_norm_torch = (samples_phys_torch - theta_mean_cpu) / (
                theta_std_cpu + self.normalizer.eps
            )
            # Move to same device as normalizer
            return samples_norm_torch.to(self.normalizer.theta_mean.device)
        else:
            # No normalizer provided - return physical as torch (fallback)
            return torch.tensor(samples_phys_np, dtype=torch.float32)

    def sample_with_traces(
        self,
        sample_shape: tuple,
        x,
        return_physical: bool = True,
        **kwargs,
    ):
        """
        Sample from the posterior and return full diffusion traces.

        This method is useful for visualizing and diagnosing the diffusion
        sampling process. It returns the parameter values at each diffusion
        step, not just the final samples.

        Args:
            sample_shape: Tuple specifying number of samples (N,)
            x: Observation tensor/array.
            return_physical: If True, returns physical units (default for traces).

        Returns:
            traces: numpy array, shape (N, num_diffusion_steps, d_theta)
                    in PHYSICAL units. Each trace[i] shows how sample i
                    evolved during the reverse diffusion process.
        """
        import torch

        num_samples = sample_shape[0]

        # Convert observation to numpy (same preprocessing as sample())
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.asarray(x)

        x_squeezed = x_np.squeeze()
        if x_squeezed.ndim == 1:
            x_squeezed = x_squeezed.reshape(-1, 1)

        # Unnormalize if needed (same as sample())
        if self.normalizer is not None and not return_physical:
            x_tensor = torch.tensor(x_squeezed, dtype=torch.float32)
            if x_tensor.ndim == 2:
                x_tensor = x_tensor.unsqueeze(0)

            if self.task.obs_only:
                obs_std_cpu = self.normalizer.obs_std.cpu()
                obs_mean_cpu = self.normalizer.obs_mean.cpu()
                obs_phys = (
                    x_tensor.cpu() * (obs_std_cpu + self.normalizer.eps) + obs_mean_cpu
                )
                x_squeezed = obs_phys.squeeze(0).numpy()
            else:
                expected_dim = self.obs_dim + getattr(self.task, "_ctrl_dim", 0)
                if x_tensor.shape[-1] != expected_dim:
                    raise ValueError(
                        f"FNPE expects {expected_dim} dims (obs+ctrl), "
                        f"got {x_tensor.shape[-1]}"
                    )
                norm_device = self.normalizer.obs_mean.device
                x_phys = self.normalizer.unnormalize_x(
                    x_tensor.to(norm_device), self.obs_dim
                )
                x_squeezed = x_phys.squeeze(0).detach().cpu().numpy()

        # FNPE internal normalization
        x_jax = jnp.asarray(x_squeezed)
        x_norm = self.task.normalize_x(x_jax)

        # Check if score function requires hyperparameter estimation
        score_fn = self.sampler.kernel.score_fn
        if (
            hasattr(score_fn, "requires_hyperparameters")
            and score_fn.requires_hyperparameters
        ):
            x_hash = hash(x_norm.tobytes())
            if (
                not hasattr(self, "_hyper_cache_hash")
                or self._hyper_cache_hash != x_hash
            ):
                self.key, key_hyper = jax.random.split(self.key)
                score_fn.estimate_hyperparameters(
                    x_norm, self.sampler.theta_shape, key_hyper
                )
                self._hyper_cache_hash = x_hash

        # Sample with traces using sampler.simulate (returns full trajectory)
        self.key, *sample_keys = jax.random.split(self.key, num_samples + 1)
        sample_keys = jnp.stack(sample_keys)

        # sampler.simulate returns the full diffusion trajectory
        traces_norm_jax = jax.vmap(self.sampler.simulate, in_axes=(0, None))(
            sample_keys, x_norm
        )
        traces_norm_jax = jax.block_until_ready(traces_norm_jax)

        # traces_norm_jax shape: (num_samples, num_steps, d_theta)
        # Unnormalize each step to physical units
        def unnorm_trace(trace):
            return jax.vmap(self.task.unnormalize_theta)(trace)

        traces_phys = jax.vmap(unnorm_trace)(traces_norm_jax)
        traces_phys_np = np.array(traces_phys)

        return traces_phys_np


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
        window_size: int = 2,  # Markov window size - CRITICAL for performance
        num_epochs: int = 20,
        steps_per_epoch: int = 10000,
        batch_size: int = 512,
        num_diffusion_steps: int = 500,
        score_fn_type: str = "fnpe",
        stop_after_epochs: int = 30,
        validation_fraction: float = 0.1,
        proposal_type: str = "pred",  # "pred" (correct), "naive", or "trajectory" (old/wrong)
        pilot_fraction: float = 0.02,  # Fraction of num_sims for pilots (2%)
        pilot_length: int = 500,  # Length of each pilot trajectory
        proposal_noise: float = 0.03,  # Noise scale: noise = proposal_noise * std(pool)
        gauss_posterior_precission_scale: Optional[float] = None,
    ):
        # Note: FNPE doesn't use torch prior/device directly
        super().__init__(cfg, prior, device)

        # FNPE-specific config
        self.hidden_dim = hidden_dim
        self.num_hidden = num_hidden
        self.model_type = model_type
        self.window_size = window_size  # Small window, NOT full sequence!
        self.num_epochs = num_epochs
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.num_diffusion_steps = num_diffusion_steps
        self.score_fn_type = score_fn_type
        self.stop_after_epochs = stop_after_epochs
        self.validation_fraction = validation_fraction
        # Proposal type for training data generation
        # "pred" = correct FNPE (proposal from pilot sims)
        # "naive" = sample from initial state distribution
        # "trajectory" = OLD incorrect implementation (divide trajectories into pairs)
        self.proposal_type = proposal_type
        # Proposal hyperparameters
        self.pilot_fraction = pilot_fraction  # 2% of training sims for pilots
        self.pilot_length = pilot_length  # Length of pilot trajectories
        self.proposal_noise = proposal_noise  # Noise = proposal_noise * std(pool)
        self.gauss_posterior_precission_scale = gauss_posterior_precission_scale

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

        # Report JAX device
        devices = jax.devices()
        print(f"[FNPE] JAX devices: {devices}", flush=True)
        print(f"[FNPE] Default backend: {jax.default_backend()}", flush=True)

        # Create task (this includes the simulator and normalization)
        self.task = VehicleDynamicsTask(
            cfg=self.cfg,
            obs_only=False,
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

        IMPORTANT: Training data is generated with T=window_size (e.g., T=2),
        following the Markov assumption. At inference, longer sequences are
        handled via score factorization.
        """
        if self.task is None:
            raise RuntimeError("Method not built. Call build() first.")

        num_sim = num_simulations or self.cfg.num_simulations
        # Store full observation length for inference
        self._t_obs_full = T_obs or self._seq_len

        # For TRAINING, use window_size as T (like Lotka-Volterra: T=2)
        # This is the Markov assumption - we only need short windows for training
        t_train = self.window_size

        print(
            f"[FNPE] Generating {num_sim} training samples (T={t_train}, window_size={self.window_size})...",
            flush=True,
        )
        print(
            f"[FNPE] (Full observation length T_obs={self._t_obs_full} will be used at inference)",
            flush=True,
        )

        # Select proposal type for data generation
        if self.proposal_type == "trajectory":
            print(
                f"[FNPE] Using 'trajectory' proposal (OLD implementation - divides trajectories into pairs)",
                flush=True,
            )
            # For trajectory mode, we need longer T to get enough pairs
            t_train_actual = max(t_train, 10)  # At least 10 timesteps to get pairs
        else:
            print(
                f"[FNPE] Using '{self.proposal_type}' proposal (covering full state space)",
                flush=True,
            )
            t_train_actual = t_train

        # Generate data with T = window_size using selected PROPOSAL method
        self._key, key_data = jax.random.split(self._key)

        # Configure proposal parameters (only used for "pred" mode)
        # num_pilot_sims = pilot_fraction * num_sim (default 2%)
        num_pilot_sims = max(10, int(num_sim * self.pilot_fraction))
        T_pilot = self.pilot_length

        print(
            f"[FNPE] Proposal config: {num_pilot_sims} pilots (={self.pilot_fraction*100:.1f}% of {num_sim}), "
            f"T_pilot={T_pilot}, noise={self.proposal_noise}*std",
            flush=True,
        )

        data = self.task.get_data(
            key_data,
            num_sim,
            t_train_actual,
            proposal=self.proposal_type,  # Use configured proposal type
            num_pilot_sims=num_pilot_sims,
            T_pilot=T_pilot,
            proposal_noise=self.proposal_noise,  # Pass noise scale
        )

        print(
            f"[FNPE] Data shapes: thetas={data['thetas'].shape}, xs={data['xs'].shape}",
            flush=True,
        )

        # Initialize SDE
        t_min = getattr(self.cfg, "fnpe_t_min", 0.05)
        print(f"[FNPE] Initializing SDE (T_min={t_min})...", flush=True)
        self.sde, self.weight_fn = init_sde(data, T_min=t_min)

        # Train score network
        budget_epochs = self._compute_budget_epochs(num_sim)
        self.num_epochs = budget_epochs
        print(
            f"[FNPE] Training score network (max {self.num_epochs} epochs)...",
            flush=True,
        )
        train_start = time.time()

        # Pass window_size for network construction
        self.params, self.score_net, losses = self._train_score_network(
            data, self.window_size
        )

        train_time = time.time() - train_start

        # Setup sampler
        print("[FNPE] Setting up sampler...")
        self._setup_sampler()

        # Store model reference
        self.model = self.params

        # losses is now a dict with 'train' and 'val' keys
        self._training_summary = {
            "train_loss": losses.get("train", []),
            "final_train_loss": losses["train"][-1] if losses.get("train") else None,
            "epochs_trained": len(losses.get("train", [])),
            "train_time_s": train_time,
            "num_simulations": num_sim,
            "T_train": t_train,  # Training window size
            "T_obs_full": self._t_obs_full,  # Full observation length for inference
            "budget_epochs": budget_epochs,
        }

        return self._training_summary

    def _train_score_network(self, data: Dict, window_size: int):
        """Internal method to train score network without validation/early stopping."""
        key = self._key
        key, key_init = jax.random.split(key, 2)

        train_data = {
            "thetas": data["thetas"],
            "xs": data["xs"],
        }

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

        # Build batch sampler for training
        train_batch_sampler = build_batch_sampler(train_data)
        loss_fn = build_loss_fn(
            "dsm", score_net, self.sde, self.weight_fn, control_variate=True
        )

        # Initialize
        theta_batch, x_batch = train_batch_sampler(key_init, self.batch_size)
        params = init_fn(key_init, jnp.ones((self.batch_size,)), theta_batch, x_batch)

        n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
        print(f"[FNPE] Score network: {n_params:,} parameters", flush=True)

        # Optimizer - use max possible steps, early stopping will terminate early
        max_total_steps = self.num_epochs * self.steps_per_epoch
        schedule = optax.cosine_onecycle_schedule(
            max_total_steps, self.cfg.learning_rate
        )
        optimizer = optax.chain(
            optax.adaptive_grad_clip(self.cfg.clip_max_norm),
            optax.ema(
                0.01
            ),  # EMA on weights for smoother training (like MarkovSBI notebook)
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
        print("[FNPE] JIT compiling...", flush=True)
        key, key_batch, key_loss = jax.random.split(key, 3)
        theta_batch, x_batch = train_batch_sampler(key_batch, self.batch_size)
        loss, params, opt_state = update(
            params, key_loss, opt_state, theta_batch, x_batch
        )
        _ = float(loss)  # Block until done
        print("[FNPE] JIT compilation complete.", flush=True)

        # Training loop (no validation/early stopping)
        train_losses = []

        print(
            f"[FNPE] Starting training: {self.num_epochs} epochs, {self.steps_per_epoch} steps/epoch",
            flush=True,
        )

        for epoch in range(self.num_epochs):
            # Training - accumulate losses without blocking
            epoch_losses = []
            pbar = tqdm(
                range(self.steps_per_epoch),
                desc=f"Epoch {epoch+1}/{self.num_epochs}",
                leave=False,
                file=sys.stderr,
            )
            for step in pbar:
                key, key_batch, key_loss = jax.random.split(key, 3)
                theta_batch, x_batch = train_batch_sampler(key_batch, self.batch_size)
                loss, params, opt_state = update(
                    params, key_loss, opt_state, theta_batch, x_batch
                )
                epoch_losses.append(loss)
                # Only sync every 100 steps for progress display
                if step % 100 == 99:
                    # Block and compute mean of last 100 losses
                    recent_losses = jnp.stack(epoch_losses[-100:])
                    mean_loss = float(jnp.mean(recent_losses))
                    pbar.set_postfix({"loss": f"{mean_loss:.4f}"})
            pbar.close()

            # Compute epoch mean (single sync point)
            all_losses = jnp.stack(epoch_losses)
            epoch_train_loss = float(jnp.mean(all_losses))
            train_losses.append(epoch_train_loss)

            print(
                f"[FNPE] Epoch {epoch+1}/{self.num_epochs}: "
                f"Train={epoch_train_loss:.6f}",
                flush=True,
            )

        self._key = key

        return params, score_net, {"train": train_losses}

    def _setup_sampler(self):
        """Set up the diffusion sampler."""
        prior_norm = self.task.get_normalized_prior()

        if self.score_fn_type.lower() == "fnpe":
            # Basic FNPE score: (1-N)*prior + sum(local_scores)
            score_fn = FNPEScoreFn(
                self.score_net,
                self.params,
                self.sde,
                prior_norm,
            )
        elif self.score_fn_type.lower() == "gauss_corrected":
            # Gaussian-corrected FNPE score (more accurate at a > 0)
            # Uses covariance weighting as described in paper Section 6.2
            posterior_precission_est_fn = None
            if self.gauss_posterior_precission_scale is not None:
                scale = float(self.gauss_posterior_precission_scale)
                prior_var = prior_norm.var
                posterior_precission_est_fn = (
                    lambda *args, _scale=scale, _prior_var=prior_var: _scale
                    / _prior_var
                )
            score_fn = GaussCorrectedScoreFn(
                self.score_net,
                self.params,
                self.sde,
                prior_norm,
                posterior_precission_est_fn=posterior_precission_est_fn,
                window_size=self.window_size,
            )
        else:
            # Uncorrected: uses marginal prior score
            score_fn = UncorrectedScoreFn(
                self.score_net, self.params, self.sde, prior_norm
            )

        kernel = EulerMaruyama(score_fn)
        time_grid = jnp.linspace(
            self.sde.T_min, self.sde.T_max, self.num_diffusion_steps
        )

        # Optionally clip samples to normalized prior bounds during diffusion
        transform_state = None
        if getattr(self.cfg, "fnpe_clip_samples", False):
            # Compute normalized prior bounds for clipping
            bounds = self.cfg.param_bounds()
            low_list = [bounds[name][0] for name in self.cfg.active_parameters]
            high_list = [bounds[name][1] for name in self.cfg.active_parameters]
            low_phys = jnp.array(low_list, dtype=jnp.float32)
            high_phys = jnp.array(high_list, dtype=jnp.float32)

            if self.task.normalize and self.task._theta_mean is not None:
                low_norm = (low_phys - self.task._theta_mean) / self.task._theta_std
                high_norm = (high_phys - self.task._theta_mean) / self.task._theta_std
            else:
                low_norm = low_phys
                high_norm = high_phys

            from markovsbi.sampling.sample import clip_transform

            transform_state = clip_transform(low_norm, high_norm)
            print(
                f"[FNPE] Clipping diffusion samples to "
                f"[{np.array(low_norm)}, {np.array(high_norm)}] (normalized)",
                flush=True,
            )

        self.sampler = Diffuser(
            kernel,
            time_grid,
            self.task.input_shape,
            transform_state=transform_state,
        )

    def _compute_budget_epochs(self, num_simulations: int) -> int:
        """Compute FNPE epoch budget from simulation count, capped."""
        batch_size = max(1, int(self.cfg.training_batch_size))
        epochs = (num_simulations + batch_size - 1) // batch_size
        return min(int(self.cfg.fnpe_max_epochs), max(1, int(epochs)))

    def build_posterior(self, normalizer=None) -> FNPEPosterior:
        """Build posterior wrapper for sampling.

        Args:
            normalizer: Optional external normalizer for NPE-compatible interface.
                       If provided, posterior.sample() will accept normalized inputs
                       and return normalized torch tensors (same interface as NPE/NPSE).
        """
        if self.sampler is None:
            raise RuntimeError("Model not trained. Call train() first.")

        self._key, key_posterior = jax.random.split(self._key)
        self.posterior = FNPEPosterior(
            self.sampler,
            self.task,
            key_posterior,
            normalizer=normalizer,
            obs_dim=self.cfg.obs_dim,  # Pass obs_dim from config
        )
        return self.posterior

    def save(self, exp_dir: Path) -> Path:
        """Save FNPE artifacts."""
        if self.params is None:
            raise RuntimeError("No model to save.")

        # Save params
        model_path = exp_dir / self.model_filename
        with open(model_path, "wb") as f:
            pickle.dump(self.params, f)

        # Save normalization stats (values may be None when normalization is skipped)
        norm_stats = self.task.get_normalization_stats()
        norm_stats_json = {
            k: v.tolist() if v is not None else None for k, v in norm_stats.items()
        }
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
