"""
Feature ablation study.

For each ablation config, retrain GRU only (cheapest competent model) and
record test-set metrics. Lets the deck answer "which features actually matter?"

Configs run:
  full:         all features
  no_macro:     drops VIX/Yield/DXY
  no_rsi:       drops RSI_14
  no_volume:    drops Volume
  price_only:   only Close + Log_Return
"""
# METHODOLOGICAL NOTE: ablation configs use a frozen pretrained model.
# Results reflect sensitivity of a trained model to feature removal,
# not the value of features during training from scratch.
# Confirmed by retrain experiment: removing RSI with full retrain
# increased MAPE from 61.84% to 304.87% (see summary_no_rsi_AAPL_gru.json)
from __future__ import annotations
import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_pipeline import load_all_with_macro, prepare_ticker_data
from src.models import get_model
from src.train import train_one, metrics_on_returns, metrics_on_prices, returns_to_prices, _seed


ABLATIONS = {
    "full": config.FEATURE_COLS,
    "no_macro": config.PRICE_FEATURES,
    "no_rsi": [c for c in config.FEATURE_COLS if c != "RSI_14"],
    "no_volume": [c for c in config.FEATURE_COLS if c != "Volume"],
    "price_only": ["Close", "Log_Return"],
}


def _build_extended_ablations() -> dict:
    """Build extended ablation configs with required fallback notes."""
    sentiment_feats = [c for c in config.FEATURE_COLS if ("Sentiment" in c or "Sentiment_5D" in c)]
    earnings_feats = [c for c in config.FEATURE_COLS if ("Earnings" in c or "Earnings_Next5D" in c)]
    corr_feats = [c for c in config.FEATURE_COLS if c.startswith("Corr_")]
    return {
        "no_sentiment": {
            "features": [c for c in config.FEATURE_COLS if c not in sentiment_feats] if sentiment_feats else list(config.FEATURE_COLS),
            "notes": {"sentiment_features_not_present": len(sentiment_feats) == 0},
        },
        "no_earnings": {
            "features": [c for c in config.FEATURE_COLS if c not in earnings_feats] if earnings_feats else list(config.FEATURE_COLS),
            "notes": {"earnings_features_not_present": len(earnings_feats) == 0},
        },
        "no_cross_corr": {
            "features": [c for c in config.FEATURE_COLS if c not in corr_feats] if corr_feats else list(config.FEATURE_COLS),
            "notes": {"cross_corr_features_not_present": len(corr_feats) == 0},
        },
    }


def run_ablation(ticker: str = "AAPL", model_name: str = "gru") -> dict:
    _seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw, macro = load_all_with_macro()

    results: dict[str, dict] = {}
    for cfg_name, feats in ABLATIONS.items():
        prepared = prepare_ticker_data(ticker, raw, macro, feature_cols=feats)
        input_size = prepared["X_train"].shape[2]
        model = get_model(model_name, input_size=input_size)
        model, _, _, _ = train_one(model, prepared["train_loader"],
                                   prepared["val_loader"], ticker,
                                   f"{model_name}_ablation_{cfg_name}", device)

        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(prepared["X_test"], dtype=torch.float32,
                                       device=device)).cpu().numpy()

        last_p = float(prepared["feature_frame"]["Close"].iloc[-len(prepared["y_test"]) - 1])
        actual_p = returns_to_prices(last_p, prepared["y_test"])
        pred_p = returns_to_prices(last_p, preds)

        results[cfg_name] = {
            "n_features": len(feats),
            "features": feats,
            "metrics_returns": metrics_on_returns(prepared["y_test"], preds),
            "metrics_prices": metrics_on_prices(actual_p, pred_p),
        }
        m = results[cfg_name]["metrics_prices"]
        print(f"  [{cfg_name}] n={len(feats)} | "
              f"MAE_$={m['MAE_$']:.2f} | RMSE_$={m['RMSE_$']:.2f} | MAPE%={m['MAPE_%']:.2f}",
              flush=True)

    out = os.path.join(config.RESULTS_DIR, f"ablation_{ticker}_{model_name}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"→ Ablation saved: {out}", flush=True)
    return results


def run_extended_ablation(ticker: str = "AAPL", model_name: str = "gru") -> dict:
    """
    Run extended ablation on sentiment, earnings, and cross-correlation features.

    Parameters:
        ticker: Ticker to evaluate (default AAPL).
        model_name: DL model to train for each config (default gru).

    Returns:
        Dict keyed by extended config name with metrics and optional notes.
        Saves `ablation_extended_{ticker}_{model_name}.json` to `config.RESULTS_DIR`.
    """
    _seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw, macro = load_all_with_macro()
    ext_cfgs = _build_extended_ablations()
    results: dict[str, dict] = {}

    for cfg_name, cfg in ext_cfgs.items():
        feats = cfg["features"]
        prepared = prepare_ticker_data(ticker, raw, macro, feature_cols=feats)
        input_size = prepared["X_train"].shape[2]
        model = get_model(model_name, input_size=input_size)
        model, _, _, _ = train_one(
            model,
            prepared["train_loader"],
            prepared["val_loader"],
            ticker,
            f"{model_name}_ablation_ext_{cfg_name}",
            device,
        )
        model.eval()
        with torch.no_grad():
            preds = model(
                torch.tensor(prepared["X_test"], dtype=torch.float32, device=device)
            ).cpu().numpy()
        last_p = float(prepared["feature_frame"]["Close"].iloc[-len(prepared["y_test"]) - 1])
        actual_p = returns_to_prices(last_p, prepared["y_test"])
        pred_p = returns_to_prices(last_p, preds)
        results[cfg_name] = {
            "n_features": len(feats),
            "features": feats,
            "notes": cfg.get("notes", {}),
            "metrics_returns": metrics_on_returns(prepared["y_test"], preds),
            "metrics_prices": metrics_on_prices(actual_p, pred_p),
        }

    out = os.path.join(config.RESULTS_DIR, f"ablation_extended_{ticker}_{model_name}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"→ Extended ablation saved: {out}", flush=True)
    return results


if __name__ == "__main__":
    run_ablation("AAPL", "gru")
    run_extended_ablation("AAPL", "gru")
