# Simulation-Based Inference for Vehicle Parameter Estimation

Master's thesis — Leibniz Universität Hannover, Institute of Mechatronic Systems (IMES)
Supervisor: Jan-Hendrik Ewering

## Overview

Can amortized simulation-based inference (SBI) accurately recover physical vehicle parameters — friction coefficient (µ), air resistance (c_d), and mass (m) — directly from time-series driving data?

This project builds a quantitative benchmark comparing four neural posterior estimation methods on a JAX-based two-track vehicle dynamics simulator, then tests whether the conclusions hold on seven real-world braking-test recordings, outside the simulator's ground-truth-labeled distribution.

**Methods compared:**

| Method | Approach |
|---|---|
| NPE | Neural Posterior Estimation, normalizing flows (MAF) |
| NPSE | Neural Posterior Score Estimation, diffusion models |
| FNPSE | Factorized Neural Posterior Score Estimation via [MarkovSBI](https://github.com/mackelab/markovsbi) ([Gloeckler et al., 2024](https://arxiv.org/abs/2411.02728)) |
| Simformer | All-conditional score transformer ([Gloeckler et al., 2024](https://arxiv.org/abs/2404.09636)) |

## Key results

**432 simulation experiments say NPE wins — but 7 real-world recordings say it's not that simple.**

On the controlled simulation benchmark, **NPE consistently achieves the lowest posterior error** across every parameter configuration and simulation budget (e.g. W2 = 0.017 for single-parameter friction estimation at the highest budget, 4–20× lower than FNPSE and Simformer) while also being the fastest to train and sample (< 0.01s per posterior on GPU vs. up to 175s for Simformer). It's the only method whose accuracy scales reliably with more simulation budget across all three parameter dimensionalities.

But posterior accuracy and predictive robustness turned out to be different things. When the four trained methods were applied to **7 real vehicle braking-test recordings**, the simulative rankings partly **inverted**: FNPSE — the weakest method in simulation, prone to posterior mode-collapse — achieved the *lowest* mean prediction error on real data (17.7 vs. NPE's 18.8), because its broader, less-confident posteriors happened to absorb the sim-to-real gap better than NPE's narrow, overconfident ones. A parallel calibration study (Particle Swarm Optimization over 23 simulator parameters, fit to the 7 real recordings) confirmed the underlying vehicle simulator is structurally capable of matching real dynamics (in-sample NRMSE = 0.185), but does so only by compensating for missing physics (e.g. no dynamic load transfer) with joint parameter distortions — meaning the calibrated values aren't individually physically meaningful.

**Bottom line:** no method dominates on both criteria. NPE is the right default when the simulator is trustworthy and inference speed matters; when a large sim-to-real gap is unavoidable, the method that "wins" in simulation may not be the one that generalizes. This calibration-vs-accuracy trade-off, and its reversal under real-world distribution shift, is the central finding of the thesis.

| | NPE | NPSE | FNPSE | Simformer |
|---|---|---|---|---|
| Simulative W2 (µ-only, best budget) | **0.017** | 0.030 | 0.235 | 0.327 |
| Real-data PPC RMSE (mean, 7 recordings) | 18.8 | 21.5 | **17.7** | 18.5 |
| Sampling time / posterior | **< 0.01s** | 1–5s | 4–20s | 163–175s |

Full method-by-method analysis, all 12 evaluation metrics, and the qualitative failure modes behind these numbers (why FNPSE collapses, why Simformer's coverage is a diffuseness artifact rather than accuracy) are in the thesis.

## Repository contents

```
├── code/                 # Simulator, all four inference methods, training/eval pipeline, SLURM tooling
├── data/measurements/    # 7 real vehicle braking-test recordings (out-of-distribution validation)
├── presentations/        # Seminar and progress slides
└── thesis.pdf            # Final thesis document
```

Setup instructions, CLI reference, and steps to reproduce or extend the experiments are in **[TUTORIAL.md](TUTORIAL.md)**.

## Evaluation metrics

Wasserstein-2 distance, classifier two-sample test (C2ST), credible-interval coverage, one-step-ahead RMSE, and posterior predictive checks against real measurement trajectories. Full definitions in [TUTORIAL.md](TUTORIAL.md#evaluation-metrics).

## Built with

JAX · PyTorch · [`sbi`](https://github.com/sbi-dev/sbi) · [MarkovSBI](https://github.com/mackelab/markovsbi) · [Simformer](https://github.com/mackelab/simformer)

MarkovSBI and Simformer are used as external dependencies (cloned separately per [TUTORIAL.md](TUTORIAL.md)), not vendored in this repository.

## Citation

This project builds on the following methods — please cite them if referencing this work:

```bibtex
@misc{gloeckler2024compositional,
  title={Compositional simulation-based inference for time series},
  author={Gloeckler, Manuel and Toyota, Shoji and Fukumizu, Kenji and Macke, Jakob H.},
  year={2024}, eprint={2411.02728}, archivePrefix={arXiv}, primaryClass={cs.LG}
}

@misc{gloeckler2024simformer,
  title={All-in-one simulation-based inference},
  author={Gloeckler, Manuel and Deistler, Michael and Weilbach, Christian and Wood, Frank and Macke, Jakob H.},
  year={2024}, eprint={2404.09636}, archivePrefix={arXiv}, primaryClass={cs.LG}
}
```

## License

MIT — see [LICENSE](LICENSE). MarkovSBI and Simformer are governed by their own repositories' terms.
