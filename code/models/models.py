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
    etype = cfg.encoder_type.lower()

    # === BIGRU ================================================================
    if etype == "bigru":
        encoder = BiGRUAttnEncoder(
            input_dim=input_dim,
            hidden=cfg.encoder_hidden,
        ).to(device)
        encoder.gru.flatten_parameters()
        # output_dim = 2 * hidden
        out_dim = 2 * cfg.encoder_hidden
        return encoder, out_dim

    # === CAUSAL CNN ===========================================================
    elif etype == "causalcnn":
        # input shape for CNN is (T, input_dim), but wrapped internally
        embedding_cnn = embedding_nets.CausalCNNEmbedding(
            input_shape=(seq_len, input_dim),
            num_conv_layers=cfg.causalcnn_num_layers,
            num_filters=cfg.causalcnn_num_filters,
            kernel_size=cfg.causalcnn_kernel_size,
            dilation_base=cfg.causalcnn_dilation_base,
            out_channels=cfg.causalcnn_out_channels,
            pool_kernel_size=cfg.causalcnn_pool_kernel,
            output_dim=cfg.embedding_output_dim,  # must exist in config
        ).to(device)
        return embedding_cnn, cfg.embedding_output_dim

    # === TRANSFORMER ==========================================================
    elif etype == "transformer":
        # transformer config from cfg
        trans_cfg = dict(
            vit=False,
            feature_space_dim=cfg.transformer_feature_dim,  # = d_model
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

        return embedding_net, cfg.embedding_output_dim

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
    seq_len = cfg.T_event  # assigned from config when building simulation

    # Build embedding network
    embedding_net, emb_dim = build_embedding(cfg, input_dim, seq_len, device)

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
