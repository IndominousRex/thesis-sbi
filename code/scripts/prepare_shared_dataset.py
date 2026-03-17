#!/usr/bin/env python
"""
Prepare the shared cached dataset used by comparison runs.

This generates the deterministic cached dataset once so separate SLURM jobs for
NPE, NPSE, and Simformer can all reuse the same data without a cache-generation
race.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.unified_experiment import get_or_generate_dataset
from models.models import build_prior
from scripts.run_simulation_comparison import create_config
from simulation.simulation import init_simulation_from_config, make_simulator
from utils.env_utils import get_device, setup_environment


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare the shared cached dataset for comparison runs."
    )
    parser.add_argument("--exp-name", type=str, default="sim_compare")
    parser.add_argument("--num-simulations", type=int, default=2000)
    parser.add_argument("--T-seg", type=int, default=1000)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="Device to use while materializing the dataset cache.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.quick and args.smoke:
        raise ValueError("Use either --quick or --smoke, not both.")

    cfg = create_config(
        method="npe",
        exp_name=args.exp_name,
        num_simulations=args.num_simulations,
        T_seg=args.T_seg,
        device=args.device,
        seed=args.seed,
        quick=args.quick,
        smoke=args.smoke,
        dataset_id=None,
        reuse_dataset=True,
    )

    print("=" * 70)
    print("Preparing shared comparison dataset")
    print("=" * 70)
    print(f"Dataset ID: {cfg.dataset_id}")
    print(f"Cache path: {cfg.get_dataset_cache_path()}")
    print(f"Device: {cfg.device}")
    print(f"Simulations: {cfg.num_simulations}, T_seg: {cfg.T_seg}")
    print("=" * 70)

    setup_environment(cfg.sim_seed)
    device = get_device(cfg.device)
    init_simulation_from_config(cfg)
    prior = build_prior(cfg, device)
    simulator = make_simulator(cfg, device)
    get_or_generate_dataset(cfg, prior, simulator, device)

    print(f"[DONE] Shared dataset ready at {cfg.get_dataset_cache_path()}")


if __name__ == "__main__":
    main()
