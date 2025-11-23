# sbi_vehicle/models.py

import torch
import torch.nn as nn
from sbi.neural_nets import posterior_nn
from sbi import utils as sbi_utils
from sbi import inference as sbi_inference

from configs.config import ExperimentConfig


class BiGRUAttnEncoder(nn.Module):
    """
    Your BiGRU + attention encoder, unchanged from the notebook.
    """

    def __init__(self, input_dim: int, hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        self.att = nn.Linear(2 * hidden, 1)

    def forward(self, x):
        # x: (B, T, D_in)
        h, _ = self.gru(x)
        a = torch.softmax(self.att(h).squeeze(-1), dim=1)  # (B, T)
        emb = (h * a.unsqueeze(-1)).sum(dim=1)  # (B, 2H)
        return emb


def build_prior(cfg: ExperimentConfig, device: torch.device):
    """
    Build the BoxUniform prior over the active parameter subset.
    """
    bounds = cfg.param_bounds()
    low_list = [bounds[name][0] for name in cfg.active_parameters]
    high_list = [bounds[name][1] for name in cfg.active_parameters]

    low = torch.tensor(low_list, dtype=torch.float32, device=device)
    high = torch.tensor(high_list, dtype=torch.float32, device=device)
    return sbi_utils.BoxUniform(low=low, high=high)


def build_density_estimator(
    cfg: ExperimentConfig,
    input_dim: int,
    prior,
    device: torch.device,
):
    """
    Build encoder + MAF density estimator + NPE inference object.
    """
    if cfg.encoder_type == "bigru":
        encoder = BiGRUAttnEncoder(input_dim=input_dim, hidden=cfg.encoder_hidden).to(
            device
        )
        encoder.gru.flatten_parameters()
    else:
        raise NotImplementedError(f"Unknown encoder_type: {cfg.encoder_type}")

    density_estimator = posterior_nn(
        model="maf",
        hidden_features=cfg.maf_hidden_features,
        num_transforms=cfg.maf_num_transforms,
        embedding_net=encoder,
        z_score_x="independent",
        z_score_theta="independent",
    )

    inference = sbi_inference.NPE(
        prior=prior,
        density_estimator=density_estimator,
        device=device,
    )

    return encoder, density_estimator, inference
