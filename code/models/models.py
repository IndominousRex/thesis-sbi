import torch
import torch.nn as nn
from sbi.neural_nets import posterior_nn
from sbi.neural_nets import embedding_nets
from sbi import utils as sbi_utils
from sbi import inference as sbi_inference

from configs.config import ExperimentConfig


# -----------------------------------------------------------------------------
# GRU + ATTENTION ENCODER (time-domain)
# -----------------------------------------------------------------------------
class BiGRUAttnEncoder(nn.Module):
    """
    Bidirectional GRU + attention encoder for long temporal sequences.
    Output: (B, 2*hidden)
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
        a = torch.softmax(self.att(h).squeeze(-1), dim=1)  # (B,T)
        emb = (h * a.unsqueeze(-1)).sum(dim=1)
        return emb


# -----------------------------------------------------------------------------
# CausalCNN embedding wrapper
# -----------------------------------------------------------------------------
class ProjectedCausalCNN(nn.Module):
    """
    Wrapper to adapt multivariate time series (B,T,D) to CausalCNNEmbedding,
    which only supports 1D sequences.

    Steps:
      - apply a Linear over the feature dimension: (B,T,D) -> (B,T,1)
      - squeeze to (B,T)
      - feed into CausalCNNEmbedding configured with input_shape=(T,)
    """

    def __init__(self, input_dim: int, seq_len: int, cfg: ExperimentConfig):
        super().__init__()

        # Project D_in -> 1 at each time step
        self.proj = nn.Linear(input_dim, 1)

        # CausalCNNEmbedding expects 1D input: (T,)
        self.cnn = embedding_nets.CausalCNNEmbedding(
            input_shape=(seq_len,),  # NOTE: 1D shape!
            num_conv_layers=cfg.causalcnn_num_layers,
            kernel_size=cfg.causalcnn_kernel_size,
            pool_kernel_size=cfg.causalcnn_pool_kernel,
            output_dim=cfg.embedding_output_dim,
        )

    def forward(self, x):
        # x: (B, T, D_in)
        if x.ndim != 3:
            raise ValueError(f"Expected (B,T,D), got {x.shape}")

        B, T, D = x.shape
        x = self.proj(x).squeeze(-1)  # (B, T, 1) -> (B, T)
        # CausalCNNEmbedding supports batched (B, T) inputs
        return self.cnn(x)  # (B, output_dim)


# -----------------------------------------------------------------------------
# Transformer embedding wrapper
# -----------------------------------------------------------------------------
class ProjectedTransformer(nn.Module):
    """
    Wrapper for TransformerEmbedding.
    - Projects (B,T,D_in) to (B,T,d_model)
    - Runs transformer over sequence
    """

    def __init__(self, transformer: nn.Module, input_dim: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.transformer = transformer

    def forward(self, x):
        # x: (B, T, D_in)
        x = self.proj(x)  # (B, T, d_model)
        return self.transformer(x)  # (B, output_dim)


# -----------------------------------------------------------------------------
# Build PRIOR
# -----------------------------------------------------------------------------
def build_prior(cfg: ExperimentConfig, device: torch.device):
    """
    Build the BoxUniform prior over active parameters.
    """
    bounds = cfg.param_bounds()
    low_list = [bounds[name][0] for name in cfg.active_parameters]
    high_list = [bounds[name][1] for name in cfg.active_parameters]

    low = torch.tensor(low_list, dtype=torch.float32, device=device)
    high = torch.tensor(high_list, dtype=torch.float32, device=device)
    return sbi_utils.BoxUniform(low=low, high=high)


# -----------------------------------------------------------------------------
# Build Encoder / Embedding
# -----------------------------------------------------------------------------
def build_embedding(cfg: ExperimentConfig, input_dim: int, seq_len: int, device):
    """
    Build embedding net: bigru | causalcnn | transformer
    Returns embedding_net, embedding_output_dim
    """
    etype = cfg.encoder_type

    # === BIGRU ================================================================
    if etype == "bigru":
        encoder = BiGRUAttnEncoder(
            input_dim=input_dim,
            hidden=cfg.encoder_hidden,
        ).to(device)
        encoder.gru.flatten_parameters()
        # output_dim = 2 * hidden
        out_dim = 2 * cfg.encoder_hidden
        return encoder

    # === CAUSAL CNN ===========================================================
    elif etype == "causalcnn":
        embedding_net = ProjectedCausalCNN(
            input_dim=input_dim,
            seq_len=seq_len,
            cfg=cfg,
        ).to(device)
        return embedding_net

    # === TRANSFORMER ==========================================================
    elif etype == "transformer":
        # transformer config from cfg
        trans_cfg = dict(
            vit=False,
            feature_space_dim=cfg.transformer_feature_dim,
            sequence_length=seq_len,
            output_dim=cfg.embedding_output_dim,
            num_layers=cfg.transformer_layers,
            num_heads=cfg.transformer_heads,
            head_dim=cfg.transformer_head_dim,
            d_model=cfg.transformer_feature_dim,
        )

        base_trans = embedding_nets.TransformerEmbedding(trans_cfg).to(device)

        embedding_net = ProjectedTransformer(
            base_trans,
            input_dim=input_dim,
            d_model=cfg.transformer_feature_dim,
        ).to(device)

        return embedding_net

    else:
        raise NotImplementedError(f"Unknown encoder_type: {cfg.encoder_type}")


# -----------------------------------------------------------------------------
# Build density estimator + inference object
# -----------------------------------------------------------------------------
def build_density_estimator(
    cfg: ExperimentConfig,
    input_dim: int,
    prior,
    device: torch.device,
):
    """
    Builds:
      - embedding network (GRU / CNN / Transformer)
      - MAF flow with embedding_net
      - NPE inference object
    """

    # We need seq_len for CNN/Transformer embeddings
    seq_len = cfg.T_seg // cfg.decimate

    # Build embedding network
    embedding_net = build_embedding(cfg, input_dim, seq_len, device)

    # Build MAF density estimator
    density_estimator = posterior_nn(
        model="maf",
        embedding_net=embedding_net,
        hidden_features=cfg.maf_hidden_features,
        num_transforms=cfg.maf_num_transforms,
        z_score_x="independent",
        z_score_theta="independent",
    )

    # NPE inference object
    inference = sbi_inference.NPE(
        prior=prior,
        density_estimator=density_estimator,
        device=device,
    )

    return embedding_net, density_estimator, inference
