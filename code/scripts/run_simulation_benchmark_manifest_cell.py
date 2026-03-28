#!/usr/bin/env python
"""Execute one manifest row for the Slurm-native benchmark array workflow."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one benchmark manifest cell.")
    p.add_argument("--manifest", required=True, help="Path to JSONL manifest.")
    p.add_argument("--index", type=int, required=True, help="0-based manifest row index.")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the derived comparison command without executing it.",
    )
    return p.parse_args()


def _load_entry(manifest_path: Path, index: int) -> dict | None:
    with manifest_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if line_idx == index:
                return json.loads(line)
    return None


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    repo_root = Path(__file__).resolve().parents[1]
    entry = _load_entry(manifest_path, args.index)
    if entry is None:
        print(
            f"[ArrayCell] No manifest entry at index {args.index}; exiting.",
            flush=True,
        )
        return

    cmd = [
        sys.executable,
        "scripts/run_simulation_comparison.py",
        "--exp-name",
        str(entry["exp_name"]),
        "--num-simulations",
        str(int(entry["num_simulations"])),
        "--T-seg",
        str(int(entry["tseg"])),
        "--device",
        str(entry.get("device", "cuda")),
        "--methods",
        str(entry["method"]),
        "--seed",
        str(int(entry["seed"])),
        "--params",
        str(entry["params"]),
        "--no-summary",
    ]
    requested_budget_steps = entry.get("requested_budget_steps")
    if requested_budget_steps is not None:
        cmd.extend(["--requested-budget-steps", str(int(requested_budget_steps))])
    run_mode = str(entry.get("run_mode", "no"))
    if run_mode == "yes":
        cmd.append("--quick")
    elif run_mode == "smoke":
        cmd.append("--smoke")

    print("=" * 70, flush=True)
    print("[ArrayCell] Benchmark manifest task", flush=True)
    print(f"  index: {args.index}", flush=True)
    print(f"  exp_name: {entry['exp_name']}", flush=True)
    print(f"  method: {entry['method']}", flush=True)
    print(f"  num_simulations: {entry['num_simulations']}", flush=True)
    print(f"  T_seg: {entry['tseg']}", flush=True)
    print(f"  seed: {entry['seed']}", flush=True)
    print(f"  params: {entry['params']}", flush=True)
    if requested_budget_steps is not None:
        print(f"  requested_budget_steps: {requested_budget_steps}", flush=True)
    print("  command:", " ".join(cmd), flush=True)
    print("=" * 70, flush=True)

    if args.dry_run:
        return

    subprocess.run(cmd, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()
