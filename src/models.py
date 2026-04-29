"""
LSTM, GRU, and Transformer regressors.

Modern improvements over v1:
  - LayerNorm + GELU + 2-layer regression head (better gradient flow).
  - Causal mask in Transformer encoder (prevents look-ahead in attention).
  - Transformer exposes attention weights via `forward(x, return_attn=True)`
    for the interpretability heatmap (a key DL story-piece).
"""
from __future__ import annotations
import math
import sys, os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def _head(hidden: int, dropout: float) -> nn.Module:
    return nn.Sequential(
        nn.LayerNorm(hidden),
        nn.Dropout(dropout),
        nn.Linear(hidden, hidden // 2),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden // 2, 1),
    )


# ── LSTM ────────────────────────────────────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, input_size: int = len(config.FEATURE_COLS),
                 hidden: int = config.HIDDEN_SIZE,
                 num_layers: int = config.NUM_LAYERS,
                 dropout: float = config.DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = _head(hidden, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


# ── GRU ─────────────────────────────────────────────────────────────────
class GRUModel(nn.Module):
    def __init__(self, input_size: int = len(config.FEATURE_COLS),
                 hidden: int = config.HIDDEN_SIZE,
                 num_layers: int = config.NUM_LAYERS,
                 dropout: float = config.DROPOUT):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden, num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.head = _head(hidden, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out[:, -1, :]).squeeze(-1)


# ── Transformer ─────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class _AttnEncoderLayer(nn.Module):
    """Encoder layer that returns attention weights when asked."""
    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, return_attn=False):
        h = self.norm1(x)
        attn_out, attn_w = self.attn(h, h, h, attn_mask=attn_mask,
                                     need_weights=return_attn,
                                     average_attn_weights=True)
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ff(self.norm2(x)))
        return (x, attn_w) if return_attn else (x, None)


class TransformerModel(nn.Module):
    def __init__(self, input_size: int = len(config.FEATURE_COLS),
                 d_model: int = config.TRANSFORMER_DIM,
                 nhead: int = config.TRANSFORMER_HEADS,
                 num_layers: int = config.NUM_LAYERS,
                 dropout: float = config.DROPOUT,
                 seq_len: int = config.SEQUENCE_LENGTH):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len * 2)
        self.layers = nn.ModuleList([
            _AttnEncoderLayer(d_model, nhead, dropout) for _ in range(num_layers)
        ])
        self.head = _head(d_model, dropout)
        self.seq_len = seq_len

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device),
                          diagonal=1).bool()
        z = self.input_proj(x)
        z = self.pos_enc(z)
        attns = []
        for layer in self.layers:
            z, w = layer(z, attn_mask=mask, return_attn=return_attn)
            if return_attn:
                attns.append(w.detach().cpu())
        out = self.head(z[:, -1, :]).squeeze(-1)
        if return_attn:
            return out, attns  # list of (batch, seq, seq) per layer
        return out


# ── Factory ─────────────────────────────────────────────────────────────
MODEL_REGISTRY = {"lstm": LSTMModel, "gru": GRUModel, "transformer": TransformerModel}


def get_model(name: str, input_size: int | None = None) -> nn.Module:
    name = name.lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")
    if input_size is None:
        return MODEL_REGISTRY[name]()
    if name == "transformer":
        return TransformerModel(input_size=input_size)
    return MODEL_REGISTRY[name](input_size=input_size)
