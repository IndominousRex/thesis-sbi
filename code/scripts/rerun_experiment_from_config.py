#!/usr/bin/env python
"""Rerun a saved experiment config as a fresh train+eval run.

This loads a saved ``config.json`` from a previous experiment directory, keeps
its experiment hyperparameters and seeds, and launches a new run with:

- a fresh output directory
- no checkpoint reuse
- dataset caching/reuse disabled

The main use case is rerunning incomplete benchmark cells exactly from their
saved configs after copying those configs into a safe location.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from configs.config import ExperimentConfig
from inference.unified_experiment import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerun a saved experiment config as a fresh run."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a saved experiment config JSON file.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "experiments",
        help="Directory where the fresh rerun experiment folder should be created.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="cuda",
        help="Device override for the rerun. Defaults to cuda.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config_path = args.config.resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    results_root = args.results_root.resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    cfg = ExperimentConfig.load(str(config_path))

    # Preserve the saved experiment definition and seeds, but force a fresh run.
    cfg.device = args.device
    cfg.do_train = True
    cfg.do_eval = True
    cfg.output_dir = None
    cfg.checkpoint = None
    cfg.results_root = str(results_root)

    # Reruns must regenerate data deterministically rather than reading/writing
    # dataset cache artifacts.
    cfg.dataset_cache_dir = str((REPO_ROOT / "datasets").resolve())
    cfg.cache_dataset = False
    cfg.reuse_dataset = False
    cfg.dataset_id = None
    cfg.test_dataset_id = None

    print("=" * 80, flush=True)
    print("[RerunFromConfig] Fresh rerun from saved config", flush=True)
    print(f"  config:        {config_path}", flush=True)
    print(f"  results_root:  {results_root}", flush=True)
    print(f"  method:        {cfg.method}", flush=True)
    print(f"  exp_name:      {cfg.exp_name}", flush=True)
    print(f"  random_seed:   {cfg.random_seed}", flush=True)
    print(f"  sim_seed:      {cfg.sim_seed}", flush=True)
    print(f"  train_seed:    {cfg.train_seed}", flush=True)
    print(f"  num_sims:      {cfg.num_simulations}", flush=True)
    print(f"  T_seg:         {cfg.T_seg}", flush=True)
    print(f"  params:        {','.join(cfg.active_parameters)}", flush=True)
    print(f"  device:        {cfg.device}", flush=True)
    print("  cache_dataset: False", flush=True)
    print("  reuse_dataset: False", flush=True)
    print("=" * 80, flush=True)

    result = run_experiment(cfg)
    print(f"[RerunFromConfig] Finished: {result.get('exp_dir')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
