"""
Attention weight extraction for the Transformer interpretability heatmap.

A reviewer's question for any Transformer model: 'what does it actually attend
to?'. This module loads a saved Transformer checkpoint, runs one batch of test
inputs, and returns the averaged attention weights for visualization.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.models import TransformerModel


def load_transformer(ticker: str, input_size: int) -> TransformerModel:
    path = os.path.join(config.SAVED_MODELS_DIR, f"transformer_{ticker}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved Transformer for {ticker}: {path}")
    model = TransformerModel(input_size=input_size)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def extract_attention(ticker: str, X_sample: np.ndarray) -> dict:
    """
    Returns dict with:
      - layers: list of (seq_len, seq_len) average-attention matrices
      - last_step: attention from last token position (what model "looks at"
                   when making the next-day prediction)
      - day_importance: marginal weight per input day (last layer, last step)
    """
    input_size = X_sample.shape[2]
    model = load_transformer(ticker, input_size)
    xb = torch.tensor(X_sample, dtype=torch.float32)
    with torch.no_grad():
        _, attns = model(xb, return_attn=True)
    # attns: list of (batch, seq, seq) tensors
    layer_avgs = [a.mean(dim=0).numpy() for a in attns]  # average over batch
    last_layer = layer_avgs[-1]
    last_step = last_layer[-1]  # attention from final timestep
    return {
        "layers": layer_avgs,
        "last_step": last_step,
        "day_importance": last_step / (last_step.sum() + 1e-12),
    }
