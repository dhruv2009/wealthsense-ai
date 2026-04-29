"""
Hyperparameter sensitivity analysis for GRU on AAPL.
"""
from __future__ import annotations

import json
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_pipeline import load_all_with_macro, prepare_ticker_data
from src.models import GRUModel
from src.train import metrics_on_prices, returns_to_prices, _seed


def _train_20_epochs(model: nn.Module, train_loader, val_loader, device) -> tuple[float, float]:
    """Train a model for 20 epochs and return (best_val_loss, final_val_loss)."""
    model.to(device)
    crit = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    best_val = float("inf")
    final_val = float("inf")

    for _ in range(20):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_total += crit(model(xb), yb).item() * len(xb)
        final_val = val_total / len(val_loader.dataset)
        best_val = min(best_val, final_val)

    return float(best_val), float(final_val)


def _evaluate_mape(model: nn.Module, prepared: dict, device) -> float:
    """Compute test MAPE in price space for a prepared dataset."""
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(prepared["X_test"], dtype=torch.float32, device=device)).cpu().numpy()
    last_p = float(prepared["feature_frame"]["Close"].iloc[-len(prepared["y_test"]) - 1])
    actual_p = returns_to_prices(last_p, prepared["y_test"])
    pred_p = returns_to_prices(last_p, preds)
    return float(metrics_on_prices(actual_p, pred_p)["MAPE_%"])


def run_hyperparam_sensitivity() -> dict:
    """
    Run one-factor-at-a-time hyperparameter sensitivity for GRU on AAPL.

    Parameters varied (others fixed to config defaults):
      - sequence_length: [15, 30, 60]
      - hidden_size: [32, 64, 128]
      - num_layers: [1, 2, 3]
      - dropout: [0.1, 0.3, 0.5]

    Returns:
      A results dict with per-run validation loss and test MAPE, saved to
      `artifacts/results/hyperparam_sensitivity.json`.
    """
    _seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw, macro = load_all_with_macro()
    ticker = "AAPL"

    grid = {
        "SEQUENCE_LENGTH": [15, 30, 60],
        "hidden_size": [32, 64, 128],
        "num_layers": [1, 2, 3],
        "dropout": [0.1, 0.3, 0.5],
    }

    out = {
        "ticker": ticker,
        "base_defaults": {
            "sequence_length": config.SEQUENCE_LENGTH,
            "hidden_size": config.HIDDEN_SIZE,
            "num_layers": config.NUM_LAYERS,
            "dropout": config.DROPOUT,
            "epochs_per_run": 20,
        },
        "results": {k: [] for k in grid},
    }

    original_seq_len = config.SEQUENCE_LENGTH
    try:
        for param, values in grid.items():
            for value in values:
                seq_len = original_seq_len
                hidden = config.HIDDEN_SIZE
                num_layers = config.NUM_LAYERS
                dropout = config.DROPOUT

                if param == "SEQUENCE_LENGTH":
                    seq_len = int(value)
                elif param == "hidden_size":
                    hidden = int(value)
                elif param == "num_layers":
                    num_layers = int(value)
                elif param == "dropout":
                    dropout = float(value)

                config.SEQUENCE_LENGTH = seq_len
                prepared = prepare_ticker_data(ticker, raw, macro)
                model = GRUModel(
                    input_size=prepared["X_train"].shape[2],
                    hidden=hidden,
                    num_layers=num_layers,
                    dropout=dropout,
                )

                best_val, final_val = _train_20_epochs(model, prepared["train_loader"], prepared["val_loader"], device)
                test_mape = _evaluate_mape(model, prepared, device)
                out["results"][param].append(
                    {
                        "value": value,
                        "best_val_loss": best_val,
                        "final_val_loss": final_val,
                        "test_mape": test_mape,
                    }
                )
                print(f"[{param}={value}] best_val={best_val:.6f} final_val={final_val:.6f} test_mape={test_mape:.4f}", flush=True)
    finally:
        config.SEQUENCE_LENGTH = original_seq_len

    # Impact ranking based on MAPE spread (max - min) per parameter.
    impact = []
    for param, runs in out["results"].items():
        mapes = [r["test_mape"] for r in runs]
        spread = max(mapes) - min(mapes) if mapes else 0.0
        impact.append({"parameter": param, "mape_spread": float(spread)})
    impact.sort(key=lambda x: x["mape_spread"], reverse=True)
    out["impact_ranking_by_mape_spread"] = impact

    out_path = os.path.join(config.RESULTS_DIR, "hyperparam_sensitivity.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_path}", flush=True)

    print("\nParameter impact on MAPE (higher spread = stronger impact):")
    print("Parameter | MAPE spread")
    for item in impact:
        print(f"{item['parameter']} | {item['mape_spread']:.4f}")

    return out


if __name__ == "__main__":
    run_hyperparam_sensitivity()
