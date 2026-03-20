#!/usr/bin/env python
"""Launch or print a non-encoder simulation benchmark grid."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_PARAMS_GRID = ["mu", "mu,cd", "mu,cd,m"]
DEFAULT_TSEG_GRID = [1000, 1500, 2000, 2500, 3000]
DEFAULT_SEEDS = [42, 43, 44, 45, 46]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Launch a simulation benchmark sweep.")
    p.add_argument("--group-name", required=True, help="Benchmark group prefix.")
    p.add_argument("--num-simulations", type=int, required=True)
    p.add_argument("--methods", nargs="+", default=["npe", "npse", "fnpe", "simformer"])
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda", "auto"])
    p.add_argument("--params-grid", nargs="+", default=DEFAULT_PARAMS_GRID)
    p.add_argument("--tseg-grid", nargs="+", type=int, default=DEFAULT_TSEG_GRID)
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument(
        "--mode",
        choices=["print", "local", "slurm-submit"],
        default="print",
        help="Print commands, run locally, or submit with sbatch.",
    )
    p.add_argument(
        "--run-mode",
        choices=["no", "yes", "smoke"],
        default="no",
        help="Maps to the comparison runner mode.",
    )
    p.add_argument(
        "--comparison-script",
        default="scripts/run_simulation_comparison.py",
        help="Python comparison entrypoint for local mode.",
    )
    p.add_argument(
        "--slurm-script",
        default="scripts/run_simulation_comparison.sh",
        help="SLURM comparison entrypoint for slurm-submit mode.",
    )
    return p.parse_args()


def _param_label(params: str) -> str:
    return params.replace(",", "_")


def _build_exp_name(group_name: str, params: str, tseg: int, seed: int) -> str:
    return f"{group_name}_p{_param_label(params)}_t{tseg}_s{seed}"


def _local_cmd(
    python_exe: str,
    comparison_script: str,
    exp_name: str,
    num_simulations: int,
    tseg: int,
    device: str,
    methods: list[str],
    seed: int,
    params: str,
    run_mode: str,
) -> list[str]:
    cmd = [
        python_exe,
        comparison_script,
        "--exp-name",
        exp_name,
        "--num-simulations",
        str(num_simulations),
        "--T-seg",
        str(tseg),
        "--device",
        device,
        "--methods",
        *methods,
        "--seed",
        str(seed),
        "--params",
        params,
    ]
    if run_mode == "yes":
        cmd.append("--quick")
    elif run_mode == "smoke":
        cmd.append("--smoke")
    return cmd


def _slurm_cmd(
    slurm_script: str,
    exp_name: str,
    num_simulations: int,
    tseg: int,
    methods: list[str],
    seed: int,
    params: str,
    run_mode: str,
) -> list[str]:
    return [
        "sbatch",
        slurm_script,
        str(num_simulations),
        str(tseg),
        exp_name,
        run_mode,
        " ".join(methods),
        str(seed),
        params,
    ]


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    python_exe = sys.executable

    commands: list[list[str]] = []
    for params in args.params_grid:
        for tseg in args.tseg_grid:
            for seed in args.seeds:
                exp_name = _build_exp_name(args.group_name, params, tseg, seed)
                if args.mode == "slurm-submit":
                    commands.append(
                        _slurm_cmd(
                            args.slurm_script,
                            exp_name,
                            args.num_simulations,
                            tseg,
                            args.methods,
                            seed,
                            params,
                            args.run_mode,
                        )
                    )
                else:
                    commands.append(
                        _local_cmd(
                            python_exe,
                            args.comparison_script,
                            exp_name,
                            args.num_simulations,
                            tseg,
                            args.device,
                            args.methods,
                            seed,
                            params,
                            args.run_mode,
                        )
                    )

    for cmd in commands:
        rendered = subprocess.list2cmdline(cmd)
        print(rendered)
        if args.mode in {"local", "slurm-submit"}:
            subprocess.run(cmd, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()
