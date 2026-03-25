#!/usr/bin/env python
"""Recover eval metrics/plots for saved FNPE and Simformer benchmark runs.

Run this from inside the experiments directory, for example:

    python ../scripts/recover_saved_eval.py

It scans subdirectories whose names begin with:
    - fnpe_bench_budget_v2_mu
    - simformer_bench_budget_v2_mu

and selects only those that have a saved model checkpoint but are missing
metrics or figures. Each selected experiment is re-run in eval-only mode
using the saved checkpoint, and the eval artifacts are written back into the
same experiment directory.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


MODEL_FILES = {
    "fnpe": "params.pkl",
    "simformer": "simformer_model.pkl",
}

TIMESTAMP_SUFFIX_RE = re.compile(r"_(\d{8}-\d{6})$")
CELL_PATTERN_RE = re.compile(
    r"_b(?P<budget>\d+)_p(?P<params>[a-z_]+)_t(?P<tseg>\d+)_s(?P<seed>\d+)"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Recover eval-only metrics and plots for saved FNPE/Simformer runs."
    )
    p.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path.cwd(),
        help="Experiments directory to scan. Defaults to the current working directory.",
    )
    p.add_argument(
        "--match-prefix",
        type=str,
        default="bench_budget_v2_mu_s42",
        help="Folder-name prefix after the method name, e.g. bench_budget_v2_mu.",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        choices=["fnpe", "simformer"],
        default=["fnpe", "simformer"],
        help="Methods to include in the recovery scan.",
    )
    p.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="cuda",
        help="Device to use for eval recovery.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print matching folders without running evaluation.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of matching experiments to process.",
    )
    p.add_argument(
        "--non-recursive",
        action="store_true",
        help="Only scan direct children of the experiments directory.",
    )
    return p.parse_args()


def _has_plots(exp_dir: Path) -> bool:
    fig_dir = exp_dir / "figures"
    return fig_dir.exists() and any(fig_dir.rglob("*.png"))


def _matches_prefix(exp_dir: Path, methods: list[str], match_prefix: str) -> str | None:
    for method in methods:
        if exp_dir.name.startswith(f"{method}_{match_prefix}"):
            return method
    return None


def _is_candidate(exp_dir: Path, method: str) -> tuple[bool, str]:
    model_path = exp_dir / MODEL_FILES[method]
    if not exp_dir.is_dir():
        return False, "not a directory"
    if not model_path.exists():
        return False, f"missing {model_path.name}"

    metrics_missing = not (exp_dir / "metrics.json").exists()
    plots_missing = not _has_plots(exp_dir)
    if metrics_missing or plots_missing:
        reasons = []
        if metrics_missing:
            reasons.append("missing metrics.json")
        if plots_missing:
            reasons.append("missing plots")
        return True, ", ".join(reasons)
    return False, "already has metrics and plots"


def _load_original_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _restore_original_config(
    config_path: Path, original_config: dict[str, Any]
) -> None:
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(original_config, f, indent=2)


def _infer_fnpe_num_simulations_for_budget(
    requested_budget_steps: int,
    window_size: int,
    pilot_fraction: float,
    pilot_length: int,
) -> int:
    per_sim_step_cost = window_size + pilot_fraction * pilot_length
    approx = max(1, int(round(requested_budget_steps / max(per_sim_step_cost, 1e-9))))
    best = approx
    best_gap = abs(
        int(approx * window_size + round(approx * pilot_fraction) * pilot_length)
        - requested_budget_steps
    )
    for delta in range(-256, 257):
        candidate = approx + delta
        if candidate <= 0:
            continue
        gap = abs(
            int(
                candidate * window_size
                + round(candidate * pilot_fraction) * pilot_length
            )
            - requested_budget_steps
        )
        if gap < best_gap:
            best = candidate
            best_gap = gap
    return int(best)


def _parse_config_from_dir_name(exp_dir: Path, method: str, device: str) -> ExperimentConfig:
    from configs.config import ExperimentConfig

    name_wo_timestamp = TIMESTAMP_SUFFIX_RE.sub("", exp_dir.name)
    if not name_wo_timestamp.startswith(f"{method}_"):
        raise ValueError(f"Folder name does not start with method '{method}': {exp_dir.name}")

    exp_name = name_wo_timestamp[len(method) + 1 :]
    match = CELL_PATTERN_RE.search(name_wo_timestamp)
    if not match:
        raise ValueError(
            f"Could not infer budget/params/T_seg/seed from folder name: {exp_dir.name}"
        )

    requested_budget_steps = int(match.group("budget"))
    param_label = match.group("params")
    tseg = int(match.group("tseg"))
    seed = int(match.group("seed"))
    active_parameters = tuple(part for part in param_label.split("_") if part)
    derived_num_simulations = max(1, requested_budget_steps // tseg)

    cfg = ExperimentConfig(
        method=method,
        exp_name=exp_name,
        num_simulations=derived_num_simulations,
        requested_budget_steps=requested_budget_steps,
        derived_num_simulations=derived_num_simulations,
        T_seg=tseg,
        active_parameters=active_parameters,
        device=device,
        random_seed=seed,
        sim_seed=seed,
        train_seed=seed + 1,
        checkpoint=str(exp_dir.resolve()),
        output_dir=str(exp_dir.resolve()),
        results_root=str(exp_dir.parent.resolve()),
        dataset_cache_dir=str((REPO_ROOT / "datasets").resolve()),
        cache_dataset=True,
        reuse_dataset=True,
        run_sbc=False,
        no_plots=False,
        do_train=False,
        do_eval=True,
    )
    if method == "fnpe":
        cfg.fnpe_num_simulations = _infer_fnpe_num_simulations_for_budget(
            requested_budget_steps=requested_budget_steps,
            window_size=cfg.fnpe_window_size,
            pilot_fraction=cfg.fnpe_pilot_fraction,
            pilot_length=cfg.fnpe_pilot_length,
        )
    return cfg


def _prepare_eval_config(exp_dir: Path, device: str) -> ExperimentConfig:
    from configs.config import ExperimentConfig

    config_path = exp_dir / "config.json"
    if config_path.exists():
        cfg = ExperimentConfig.load(str(config_path))
    else:
        method = exp_dir.name.split("_", 1)[0]
        cfg = _parse_config_from_dir_name(exp_dir, method, device)

    cfg.device = device
    cfg.do_train = False
    cfg.do_eval = True
    cfg.checkpoint = str(exp_dir.resolve())
    cfg.output_dir = str(exp_dir.resolve())
    cfg.results_root = str(exp_dir.parent.resolve())
    cfg.dataset_cache_dir = str((REPO_ROOT / "datasets").resolve())
    cfg.cache_dataset = True
    cfg.reuse_dataset = True
    cfg.run_sbc = False
    cfg.no_plots = False
    return cfg


def main() -> int:
    args = parse_args()
    experiments_dir = args.experiments_dir.resolve()
    if not experiments_dir.exists():
        print(f"Experiments directory not found: {experiments_dir}", file=sys.stderr)
        return 2

    matches: list[tuple[Path, str, str]] = []
    skipped_prefix_matches: list[tuple[Path, str]] = []
    iterator = (
        sorted(experiments_dir.iterdir())
        if args.non_recursive
        else sorted(p for p in experiments_dir.rglob("*") if p.is_dir())
    )
    for child in iterator:
        method = _matches_prefix(child, args.methods, args.match_prefix)
        if method is None:
            continue
        include, reason = _is_candidate(child, method)
        if include:
            matches.append((child, method, reason))
        else:
            skipped_prefix_matches.append((child, reason))

    if args.limit is not None:
        matches = matches[: args.limit]

    if not matches:
        print("No matching recovery candidates found.")
        if skipped_prefix_matches:
            print("Matching-prefix folders were found but skipped:")
            for exp_dir, reason in skipped_prefix_matches[:20]:
                print(f"- {exp_dir} ({reason})")
        return 0

    print(f"Found {len(matches)} recovery candidate(s) in {experiments_dir}")
    for exp_dir, method, reason in matches:
        print(f"- {exp_dir.name} [{method}] ({reason})")

    if args.dry_run:
        return 0

    summary: dict[str, Any] = {
        "experiments_dir": str(experiments_dir),
        "match_prefix": args.match_prefix,
        "device": args.device,
        "processed": [],
        "failed": [],
    }

    for exp_dir, method, reason in matches:
        print("\n" + "=" * 80)
        print(f"Recovering eval for {exp_dir.name} [{method}]")
        print("=" * 80)

        config_path = exp_dir / "config.json"
        original_config = (
            _load_original_config(config_path) if config_path.exists() else None
        )
        cfg = _prepare_eval_config(exp_dir, args.device)

        try:
            from inference.unified_experiment import run_experiment

            result = run_experiment(cfg)
            summary["processed"].append(
                {
                    "exp_dir": str(exp_dir),
                    "method": method,
                    "reason": reason,
                    "metrics_path": str(exp_dir / "metrics.json"),
                    "result_exp_dir": str(result.get("exp_dir")),
                }
            )
        except Exception as exc:
            traceback.print_exc()
            summary["failed"].append(
                {
                    "exp_dir": str(exp_dir),
                    "method": method,
                    "reason": reason,
                    "error": repr(exc),
                }
            )
        finally:
            if original_config is not None:
                _restore_original_config(config_path, original_config)

    summary_path = experiments_dir / "recover_saved_eval_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote recovery summary to {summary_path}")

    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
