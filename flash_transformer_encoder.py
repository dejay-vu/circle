import torch
import torch.nn as nn
from flash_attn.modules.mha import MHA


class FlashTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = True,
        device="cuda",
        dtype=None,
    ):
        super().__init__()
        self.norm_first = norm_first

        self.self_attn = MHA(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            causal=True,
            rotary_emb_dim=d_model // nhead,
            use_flash_attn=True,
            device=device,
            dtype=dtype,
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def self_attn_block(self, x: torch.Tensor):
        x = self.self_attn(x)
        return self.dropout1(x)

    def feed_forward_block(self, x: torch.Tensor):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def forward(self, x: torch.Tensor):
        if self.norm_first:
            x = self.self_attn_block(self.norm1(x))
            x = self.feed_forward_block(self.norm2(x))
        else:
            x = self.norm1(x + self.self_attn_block(x))
            x = self.norm2(x + self.feed_forward_block(x))

        return x


class FlashTransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = True,
        device="cuda",
        dtype=None,
        norm: nn.Module = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                FlashTransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                    norm_first=norm_first,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = norm

    def forward(self, src: torch.Tensor):
        output = src
        for layer in self.layers:
            output = layer(output)

        if self.norm is not None:
            output = self.norm(output)

        return output
