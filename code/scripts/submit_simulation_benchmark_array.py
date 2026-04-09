#!/usr/bin/env python
"""Build and submit a Slurm array manifest for simulation benchmarks."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from datetime import datetime
from pathlib import Path


DEFAULT_PARAMS_GRID = ["mu", "mu,cd", "mu,m", "mu,cd,m"]
DEFAULT_TSEG_GRID = [1000, 2000, 3000]
DEFAULT_SEEDS = [42, 43, 44, 45, 46]
DEFAULT_BUDGET_GRID = [20_000_000, 40_000_000, 60_000_000]
# NPE/NPSE/Simformer process the full trajectory in memory → need H200.
# FNPE factorises over short windows → any GPU is fine.
H200_METHODS = {"npe", "npse", "simformer"}
GPU_PROFILE_GRES = {
    "h200": "gpu:h200:1",
    "generic": "gpu:1",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build and submit a Slurm-native array benchmark manifest."
    )
    p.add_argument("--group-name", required=True, help="Benchmark group prefix.")
    budget_group = p.add_mutually_exclusive_group(required=True)
    budget_group.add_argument("--num-simulations", type=int)
    budget_group.add_argument("--budget-grid", nargs="+", type=int)
    p.add_argument(
        "--methods",
        nargs="+",
        default=["npe", "npse", "fnpe", "simformer"],
    )
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
        "--run-mode",
        choices=["no", "yes", "smoke"],
        default="no",
        help="Maps to the comparison runner mode.",
    )
    p.add_argument(
        "--mode",
        choices=["print", "slurm-submit"],
        default="print",
        help="Print the sbatch command or submit the array job.",
    )
    p.add_argument(
        "--max-concurrent",
        type=int,
        default=12,
        help="Maximum number of active array tasks at once.",
    )
    p.add_argument(
        "--manifest-path",
        default=None,
        help="Optional output manifest path. Defaults to experiments/manifests/<timestamp>.jsonl",
    )
    p.add_argument(
        "--array-script",
        default="scripts/run_simulation_benchmark_array.sh",
        help="Slurm array script that executes one manifest row.",
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


def _iter_cell_specs(args: argparse.Namespace):
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


def _build_manifest_entries(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    entries: list[dict] = []
    skipped: list[dict] = []
    for cell in _iter_cell_specs(args):
        if cell["skip"]:
            skipped.append(cell)
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
        for method in args.methods:
            entries.append(
                {
                    "group_name": args.group_name,
                    "exp_name": exp_name,
                    "method": method,
                    "num_simulations": num_simulations,
                    "tseg": tseg,
                    "seed": seed,
                    "params": params,
                    "device": args.device,
                    "run_mode": args.run_mode,
                    "requested_budget_steps": budget_steps,
                }
            )
    return entries, skipped


def _default_manifest_path(repo_root: Path, group_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = repo_root / "experiments" / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{group_name}_{timestamp}.jsonl"


def _render_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _gpu_profile_for_method(method: str) -> str:
    return "h200" if method in H200_METHODS else "generic"


def _manifest_path_for_profile(manifest_base: Path, profile: str) -> Path:
    return manifest_base.with_name(f"{manifest_base.stem}_{profile}{manifest_base.suffix}")


def _split_entries_by_gpu_profile(entries: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for entry in entries:
        grouped.setdefault(_gpu_profile_for_method(str(entry["method"])), []).append(entry)
    return grouped


def _allocate_concurrency(task_counts: dict[str, int], total_max: int) -> dict[str, int]:
    profiles = [profile for profile, count in task_counts.items() if count > 0]
    if not profiles:
        return {}

    total_max = max(1, int(total_max))
    if len(profiles) == 1:
        profile = profiles[0]
        return {profile: min(task_counts[profile], total_max)}

    base_alloc: dict[str, int] = {}
    total_tasks = sum(task_counts[profile] for profile in profiles)
    for profile in profiles:
        raw_share = (task_counts[profile] / total_tasks) * total_max if total_tasks else 0
        base_alloc[profile] = max(1, int(raw_share))

    current_total = sum(base_alloc.values())
    if current_total > total_max:
        for profile in sorted(profiles, key=lambda item: (base_alloc[item], task_counts[item]), reverse=True):
            while current_total > total_max and base_alloc[profile] > 1:
                base_alloc[profile] -= 1
                current_total -= 1
    elif current_total < total_max:
        remainders = sorted(
            profiles,
            key=lambda item: (
                ((task_counts[item] / total_tasks) * total_max if total_tasks else 0) - base_alloc[item],
                task_counts[item],
            ),
            reverse=True,
        )
        idx = 0
        while current_total < total_max and remainders:
            profile = remainders[idx % len(remainders)]
            if base_alloc[profile] < task_counts[profile]:
                base_alloc[profile] += 1
                current_total += 1
            idx += 1
            if idx > total_max * 4:
                break

    return {
        profile: min(task_counts[profile], max(1, base_alloc[profile]))
        for profile in profiles
    }


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    manifest_base_path = (
        Path(args.manifest_path).resolve()
        if args.manifest_path is not None
        else _default_manifest_path(repo_root, args.group_name)
    )
    manifest_base_path.parent.mkdir(parents=True, exist_ok=True)

    entries, skipped = _build_manifest_entries(args)
    if not entries:
        raise SystemExit("No runnable benchmark tasks were generated.")

    entries_by_profile = _split_entries_by_gpu_profile(entries)
    task_counts = {profile: len(profile_entries) for profile, profile_entries in entries_by_profile.items()}
    concurrency_by_profile = _allocate_concurrency(task_counts, args.max_concurrent)

    manifest_specs: list[dict] = []
    for profile, profile_entries in sorted(entries_by_profile.items()):
        manifest_path = _manifest_path_for_profile(manifest_base_path, profile)
        with manifest_path.open("w", encoding="utf-8") as f:
            for entry in profile_entries:
                f.write(json.dumps(entry, sort_keys=True) + "\n")

        log_dir = repo_root / "experiments" / "slurm"
        log_dir.mkdir(parents=True, exist_ok=True)
        job_name = f"simbench_{args.group_name}_{profile}"[:64]
        array_spec = f"0-{len(profile_entries) - 1}%{max(1, concurrency_by_profile.get(profile, 1))}"
        sbatch_cmd = [
            "sbatch",
            "--job-name",
            job_name,
            "--array",
            array_spec,
            "--gres",
            GPU_PROFILE_GRES[profile],
            "--output",
            str(log_dir / f"{job_name}_%A_%a.out"),
            "--error",
            str(log_dir / f"{job_name}_%A_%a.err"),
            args.array_script,
            str(manifest_path),
        ]
        manifest_specs.append(
            {
                "profile": profile,
                "gpu_gres": GPU_PROFILE_GRES[profile],
                "manifest_path": str(manifest_path),
                "num_tasks": len(profile_entries),
                "max_concurrent": int(concurrency_by_profile.get(profile, 1)),
                "sbatch_cmd": sbatch_cmd,
            }
        )

    metadata_path = manifest_base_path.with_suffix(".meta.json")
    metadata = {
        "group_name": args.group_name,
        "num_tasks": len(entries),
        "num_skipped_cells": len(skipped),
        "methods": args.methods,
        "params_grid": args.params_grid,
        "tseg_grid": args.tseg_grid,
        "seeds": args.seeds,
        "budget_grid": args.budget_grid,
        "run_mode": args.run_mode,
        "max_concurrent": args.max_concurrent,
        "manifest_base_path": str(manifest_base_path),
        "manifests": [
            {
                "profile": spec["profile"],
                "gpu_gres": spec["gpu_gres"],
                "manifest_path": spec["manifest_path"],
                "num_tasks": spec["num_tasks"],
                "max_concurrent": spec["max_concurrent"],
            }
            for spec in manifest_specs
        ],
        "skipped_cells": skipped,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Metadata: {metadata_path}")
    print(f"Total tasks: {len(entries)}")
    print(f"Skipped cells: {len(skipped)}")
    for spec in manifest_specs:
        print(
            f"Manifest ({spec['profile']}): {spec['manifest_path']}\n"
            f"  Tasks: {spec['num_tasks']}\n"
            f"  GPU request: {spec['gpu_gres']}\n"
            f"  Max concurrent: {spec['max_concurrent']}\n"
            f"  Array command:\n{_render_cmd(spec['sbatch_cmd'])}"
        )

    if skipped:
        for cell in skipped:
            print(
                "# SKIP",
                f"budget={cell['budget_steps']}",
                f"params={cell['params']}",
                f"T_seg={cell['tseg']}",
                f"reason={cell['reason']}",
            )

    if args.mode == "slurm-submit":
        for spec in manifest_specs:
            subprocess.run(spec["sbatch_cmd"], cwd=repo_root, check=True)


if __name__ == "__main__":
    main()
