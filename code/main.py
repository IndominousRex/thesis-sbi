import argparse

from configs.config import ExperimentConfig, PARAMETER_ORDER
from inference.experiment import run_experiment


def active_param_type(value: str):
    """
    Argparse type that parses comma-separated parameter names (mu, cd, m).
    """
    allowed = set(PARAMETER_ORDER)
    parts = [p.strip().lower() for p in value.split(",") if p.strip()]

    if not parts:
        return PARAMETER_ORDER

    cleaned = []
    for p in parts:
        if p not in allowed:
            raise argparse.ArgumentTypeError(
                f"Unknown parameter '{p}'. Choose from {PARAMETER_ORDER}."
            )
        if p not in cleaned:
            cleaned.append(p)
    return tuple(cleaned)


def parse_args():
    p = argparse.ArgumentParser(description="Run NPE experiment for vehicle model.")
    p.add_argument("--exp-name", type=str, default="baseline_bigru_maf")
    p.add_argument("--num-sim", type=int, default=2000)
    p.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"]
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--T-seg", type=int, default=3000)
    p.add_argument(
        "--encoder-type",
        type=str,
        choices=["bigru", "causalcnn", "transformer"],
        default="bigru",
    )
    p.add_argument("--encoder-hidden", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-train", type=int, default=512)
    p.add_argument("--stop-after-epochs", type=int, default=20)
    p.add_argument("--num-lc2st-samples", type=int, default=1000)
    p.add_argument(
        "--params",
        type=active_param_type,
        default=PARAMETER_ORDER,
        help="Comma separated subset of parameters to infer (mu, cd, m). Order sets theta layout.",
    )
    p.add_argument(
        "--fixed-mu",
        type=float,
        default=1.0,
        help="Value of mu to plug in when it is not inferred.",
    )
    p.add_argument(
        "--fixed-cd",
        type=float,
        default=0.3,
        help="Value of cd to plug in when it is not inferred.",
    )
    p.add_argument(
        "--fixed-m",
        type=float,
        default=1700.0,
        help="Value of mass to plug in when it is not inferred.",
    )
    p.add_argument(
        "--real-data-csv",
        type=str,
        default=None,
        help="Optional path to real-data CSV to evaluate immediately after training.",
    )

    return p.parse_args()


def main():
    args = parse_args()

    cfg = ExperimentConfig(
        exp_name=args.exp_name,
        random_seed=args.seed,
        device=args.device,
        dt=args.dt,
        T_seg=args.T_seg,
        num_simulations=args.num_sim,
        encoder_type=args.encoder_type,
        encoder_hidden=args.encoder_hidden,
        learning_rate=args.lr,
        training_batch_size=args.batch_train,
        stop_after_epochs=args.stop_after_epochs,
        active_parameters=args.params,
        fixed_mu=args.fixed_mu,
        fixed_cd=args.fixed_cd,
        fixed_m=args.fixed_m,
        num_lc2st_samples=args.num_lc2st_samples,
        real_data_csv=args.real_data_csv,
    )

    run_experiment(cfg)


if __name__ == "__main__":
    main()
