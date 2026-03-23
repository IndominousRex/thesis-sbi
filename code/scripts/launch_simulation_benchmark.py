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
DEFAULT_BUDGET_GRID = [10_000_000, 20_000_000, 40_000_000, 60_000_000]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Launch a simulation benchmark sweep.")
    p.add_argument("--group-name", required=True, help="Benchmark group prefix.")
    budget_group = p.add_mutually_exclusive_group(required=True)
    budget_group.add_argument("--num-simulations", type=int)
    budget_group.add_argument("--budget-grid", nargs="+", type=int)
    p.add_argument("--methods", nargs="+", default=["npe", "npse", "fnpe", "simformer"])
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda", "auto"])
    p.add_argument("--params-grid", nargs="+", default=DEFAULT_PARAMS_GRID)
    p.add_argument("--tseg-grid", nargs="+", type=int, default=DEFAULT_TSEG_GRID)
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument(
        "--min-num-simulations",
        type=int,
        default=3000,
        help="Skip budget-derived cells with fewer simulations than this threshold.",
    )
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


def _build_budget_exp_name(
    group_name: str, budget_steps: int, params: str, tseg: int, seed: int
) -> str:
    return f"{group_name}_b{budget_steps}_p{_param_label(params)}_t{tseg}_s{seed}"


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
    requested_budget_steps: int | None = None,
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
    if requested_budget_steps is not None:
        cmd.extend(["--requested-budget-steps", str(requested_budget_steps)])
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
    requested_budget_steps: int | None = None,
) -> list[str]:
    cmd = [
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
    if requested_budget_steps is not None:
        cmd.append(str(requested_budget_steps))
    return cmd


def _iter_cells(args: argparse.Namespace):
    if args.budget_grid:
        for budget_steps in args.budget_grid:
            for params in args.params_grid:
                for tseg in args.tseg_grid:
                    derived_num_simulations = budget_steps // tseg
                    if derived_num_simulations < args.min_num_simulations:
                        yield {
                            "skip": True,
                            "budget_steps": int(budget_steps),
                            "params": params,
                            "tseg": int(tseg),
                            "reason": (
                                "derived num_simulations "
                                f"{derived_num_simulations} < {args.min_num_simulations}"
                            ),
                        }
                        continue
                    for seed in args.seeds:
                        yield {
                            "skip": False,
                            "budget_steps": int(budget_steps),
                            "num_simulations": int(derived_num_simulations),
                            "params": params,
                            "tseg": int(tseg),
                            "seed": int(seed),
                        }
    else:
        for params in args.params_grid:
            for tseg in args.tseg_grid:
                for seed in args.seeds:
                    yield {
                        "skip": False,
                        "budget_steps": None,
                        "num_simulations": int(args.num_simulations),
                        "params": params,
                        "tseg": int(tseg),
                        "seed": int(seed),
                    }


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    python_exe = sys.executable

    commands: list[list[str]] = []
    skipped_cells: list[dict[str, object]] = []
    for cell in _iter_cells(args):
        if cell["skip"]:
            skipped_cells.append(cell)
            continue

        params = str(cell["params"])
        tseg = int(cell["tseg"])
        seed = int(cell["seed"])
        num_simulations = int(cell["num_simulations"])
        budget_steps = (
            int(cell["budget_steps"]) if cell["budget_steps"] is not None else None
        )
        exp_name = (
            _build_budget_exp_name(args.group_name, budget_steps, params, tseg, seed)
            if budget_steps is not None
            else _build_exp_name(args.group_name, params, tseg, seed)
        )
        if args.mode == "slurm-submit":
            commands.append(
                _slurm_cmd(
                    args.slurm_script,
                    exp_name,
                    num_simulations,
                    tseg,
                    args.methods,
                    seed,
                    params,
                    args.run_mode,
                    requested_budget_steps=budget_steps,
                )
            )
        else:
            commands.append(
                _local_cmd(
                    python_exe,
                    args.comparison_script,
                    exp_name,
                    num_simulations,
                    tseg,
                    args.device,
                    args.methods,
                    seed,
                    params,
                    args.run_mode,
                    requested_budget_steps=budget_steps,
                )
            )

    for cmd in commands:
        rendered = subprocess.list2cmdline(cmd)
        print(rendered)
        if args.mode in {"local", "slurm-submit"}:
            subprocess.run(cmd, cwd=repo_root, check=True)

    for skipped in skipped_cells:
        print(
            "# SKIP",
            f"budget={skipped['budget_steps']}",
            f"params={skipped['params']}",
            f"T_seg={skipped['tseg']}",
            f"reason={skipped['reason']}",
        )


if __name__ == "__main__":
    main()
