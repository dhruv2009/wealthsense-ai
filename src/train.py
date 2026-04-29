"""
Training & evaluation pipeline.

Pipeline per ticker:
  1. Train LSTM, GRU, Transformer on log-return target.
  2. Extract MC Dropout intervals + calibration report.
  3. Reconstruct prices from log-returns for human-readable metrics.
  4. Run ARIMA + naive baselines for the same test window.
  5. Build dynamic ensemble.
  6. Persist everything to artifacts/results/.

Metrics design:
  - On log-returns:    MAE, RMSE, SMAPE, directional_accuracy
  - On reconstructed $: MAE_price, RMSE_price, MAPE_price (for the headline slide)

This is the principled fix to v2's broken billions-of-percent MAPE.
"""
from __future__ import annotations
import os
import sys
import json
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_pipeline import load_all_with_macro, prepare_ticker_data
from src.models import get_model
from src.uncertainty import (
    mc_dropout_predict,
    calibration_report,
    coverage as iv_coverage,
    regime_adjust,
    conformal_scale_factors,
    conformal_calibrated_report,
    calibrated_interval_from_conformal,
    save_conformal_scores,
)
from src.ensemble import rolling_inverse_rmse_weights, ensemble_predict
from src.baselines import run_baselines
from src.feature_audit import run_audit


# ── Reproducibility ─────────────────────────────────────────────────────
def _seed(s: int = config.SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ── Metrics ─────────────────────────────────────────────────────────────
def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE — finite even when y_true ≈ 0 (which log-returns are)."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred)
    return float(np.mean(np.where(denom == 0, 0.0, diff / denom)) * 100)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """% of next-day signs predicted correctly."""
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)) * 100)


def returns_to_prices(last_known_price: float, log_returns: np.ndarray) -> np.ndarray:
    """Reconstruct price path from log returns: P_t = P_{t-1} * exp(r_t)."""
    prices = [float(last_known_price)]
    for r in log_returns:
        prices.append(prices[-1] * float(np.exp(r)))
    return np.array(prices[1:], dtype=float)


def metrics_on_returns(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "SMAPE": smape(y_true, y_pred),
        "Dir_Acc": directional_accuracy(y_true, y_pred),
    }


def metrics_on_prices(true_p: np.ndarray, pred_p: np.ndarray) -> dict:
    return {
        "MAE_$": float(mean_absolute_error(true_p, pred_p)),
        "RMSE_$": float(np.sqrt(mean_squared_error(true_p, pred_p))),
        "MAPE_%": float(np.mean(np.abs((true_p - pred_p) / true_p)) * 100),
    }


# ── Train one model ─────────────────────────────────────────────────────
def train_one(model: nn.Module, train_loader, val_loader,
              ticker: str, model_name: str, device) -> tuple[nn.Module, list, list, int]:
    model.to(device)
    crit = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE,
                            weight_decay=config.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)

    best_val = float("inf")
    best_state = None
    patience_limit = 10
    patience_left = patience_limit
    train_losses, val_losses = [], []
    stop_epoch = config.EPOCHS

    print(f"  [{ticker} | {model_name}] training", flush=True)
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        tl = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            p = model(X)
            loss = crit(p, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item() * len(X)
        tl /= len(train_loader.dataset)

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                vl += crit(model(X), y).item() * len(X)
        vl /= len(val_loader.dataset)
        sched.step(vl)
        train_losses.append(tl)
        val_losses.append(vl)

        if vl < best_val:
            best_val = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience_limit
        else:
            patience_left -= 1
            if patience_left <= 0:
                stop_epoch = epoch
                print(
                    f"  [{ticker} | {model_name}] early stopping triggered at epoch {epoch} "
                    f"(best_val={best_val:.6f})",
                    flush=True,
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if stop_epoch == config.EPOCHS:
        print(
            f"  [{ticker} | {model_name}] early stopping not triggered; ran full {config.EPOCHS} epochs "
            f"(best_val={best_val:.6f})",
            flush=True,
        )
    save_path = os.path.join(config.SAVED_MODELS_DIR, f"{model_name}_{ticker}.pt")
    torch.save(model.state_dict(), save_path)
    return model.to(device), train_losses, val_losses, stop_epoch


# ── Per-ticker run ──────────────────────────────────────────────────────
def run_ticker(ticker: str, raw_data: dict, macro: pd.DataFrame, device) -> dict:
    print(f"\n{'='*60}\n  Ticker: {ticker}\n{'='*60}", flush=True)
    prepared = prepare_ticker_data(ticker, raw_data, macro)
    input_size = prepared["X_train"].shape[2]

    test_pred_map: dict[str, np.ndarray] = {}
    val_pred_map: dict[str, np.ndarray] = {}
    per_model: dict[str, dict] = {}
    train_curves: dict[str, dict] = {}

    last_known_price = float(prepared["feature_frame"]["Close"].iloc[
        -len(prepared["y_test"]) - 1
    ])
    actual_prices = returns_to_prices(last_known_price, prepared["y_test"])

    # ── DL models ──
    for name in ["lstm", "gru", "transformer"]:
        t0 = time.time()
        model = get_model(name, input_size=input_size)
        model, tloss, vloss, stop_ep = train_one(
            model, prepared["train_loader"], prepared["val_loader"],
            ticker, name, device,
        )

        # MC Dropout intervals + calibration on test set
        test_mean, test_lo, test_hi, test_samples = mc_dropout_predict(
            model, prepared["X_test"], device, n_samples=config.MC_DROPOUT_SAMPLES,
        )
        val_mean, _, _, val_samples = mc_dropout_predict(
            model, prepared["X_val"], device, n_samples=config.MC_DROPOUT_SAMPLES,
        )

        raw_calib = calibration_report(prepared["y_test"], test_samples)
        q_scales = conformal_scale_factors(prepared["y_val"], val_mean, val_samples)
        save_conformal_scores(ticker, name, q_scales)
        calib = conformal_calibrated_report(prepared["y_test"], test_mean, test_samples, q_scales)
        test_lo, test_hi = calibrated_interval_from_conformal(
            test_mean, test_samples, q_scale=q_scales.get("q_90", 1.0), level=0.9
        )
        cov_90 = iv_coverage(prepared["y_test"], test_lo, test_hi)

        # Metrics in return-space + price-space
        m_ret = metrics_on_returns(prepared["y_test"], test_mean)
        pred_prices = returns_to_prices(last_known_price, test_mean)
        m_price = metrics_on_prices(actual_prices, pred_prices)

        per_model[name] = {
            "metrics_returns": m_ret,
            "metrics_prices": m_price,
            "calibration": calib,
            "calibration_report": calib,
            "raw_calibration_report": raw_calib,
            "conformal_scales": q_scales,
            "interval_coverage_90": cov_90,
            "early_stop_epoch": stop_ep,
            "train_time_sec": round(time.time() - t0, 1),
        }
        train_curves[name] = {"train_loss": tloss, "val_loss": vloss}

        # Persist predictions for the dashboard
        np.savez(os.path.join(config.RESULTS_DIR, f"{name}_{ticker}_preds.npz"),
                 y_true=prepared["y_test"],
                 y_pred=test_mean,
                 y_pred_lower=test_lo,
                 y_pred_upper=test_hi,
                 actual_price=actual_prices,
                 pred_price=pred_prices,
                 train_losses=np.array(tloss),
                 val_losses=np.array(vloss),
                 test_dates=prepared["test_dates"].astype(str).values)

        test_pred_map[name] = test_mean
        val_pred_map[name] = val_mean
        print(f"    {name}: {m_ret} | Px MAPE {m_price['MAPE_%']:.2f}% | cov90 {cov_90:.2f}",
              flush=True)

    # ── Ensemble ──
    weights = rolling_inverse_rmse_weights(val_pred_map, prepared["y_val"], ticker=ticker)
    ens_test = ensemble_predict(test_pred_map, weights)
    ens_pred_prices = returns_to_prices(last_known_price, ens_test)
    per_model["ensemble"] = {
        "metrics_returns": metrics_on_returns(prepared["y_test"], ens_test),
        "metrics_prices": metrics_on_prices(actual_prices, ens_pred_prices),
        "weights": weights,
    }
    np.savez(os.path.join(config.RESULTS_DIR, f"ensemble_{ticker}_preds.npz"),
             y_true=prepared["y_test"],
             y_pred=ens_test,
             actual_price=actual_prices,
             pred_price=ens_pred_prices,
             test_dates=prepared["test_dates"].astype(str).values)
    print(f"    ensemble: {per_model['ensemble']['metrics_returns']} | "
          f"Px MAPE {per_model['ensemble']['metrics_prices']['MAPE_%']:.2f}% | "
          f"weights {weights}", flush=True)

    # ── Classical baselines ──
    print("    running ARIMA + naive baselines …", flush=True)
    base_preds = run_baselines(prepared)
    base_results: dict[str, dict] = {}
    for bname, bpred in base_preds.items():
        bp_price = returns_to_prices(last_known_price, bpred)
        base_results[bname] = {
            "metrics_returns": metrics_on_returns(prepared["y_test"], bpred),
            "metrics_prices": metrics_on_prices(actual_prices, bp_price),
        }
        np.savez(os.path.join(config.RESULTS_DIR, f"{bname}_{ticker}_preds.npz"),
                 y_true=prepared["y_test"],
                 y_pred=bpred,
                 actual_price=actual_prices,
                 pred_price=bp_price,
                 test_dates=prepared["test_dates"].astype(str).values)
        print(f"    {bname}: {base_results[bname]['metrics_returns']} | "
              f"Px MAPE {base_results[bname]['metrics_prices']['MAPE_%']:.2f}%", flush=True)

    return {
        "ticker": ticker,
        "models": per_model,
        "baselines": base_results,
        "ensemble_weights": weights,
        "training_curves": train_curves,
    }


# ── Master pipeline ─────────────────────────────────────────────────────
def run_full_pipeline() -> dict:
    _seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    audit = run_audit()
    for flagged in audit.get("flagged", []):
        if flagged.get("feature") in config.FEATURE_COLS:
            reasons = ", ".join(flagged.get("reasons", []))
            print(
                f"[train] warning: feature {flagged.get('feature')} for {flagged.get('ticker')} "
                f"flagged by audit ({reasons}). Continuing training.",
                flush=True,
            )

    raw_data, macro = load_all_with_macro()
    summary: dict[str, object] = {"per_ticker": {}, "config": {
        "tickers": config.TICKERS,
        "feature_cols": config.FEATURE_COLS,
        "target": config.TARGET_COL,
        "sequence_length": config.SEQUENCE_LENGTH,
        "hidden_size": config.HIDDEN_SIZE,
        "epochs": config.EPOCHS,
        "patience": config.PATIENCE,
    }}

    for t in config.TICKERS:
        summary["per_ticker"][t] = run_ticker(t, raw_data, macro, device)

    # Strip non-JSON-serializable bits before write (keep only summary metrics)
    serializable = json.loads(json.dumps(summary, default=lambda o: float(o)
                                          if isinstance(o, (np.floating,)) else str(o)))
    out = os.path.join(config.RESULTS_DIR, "summary.json")
    with open(out, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n→ Summary saved: {out}", flush=True)
    return summary


def run_quick_pipeline() -> dict:
    """
    Quick pipeline mode for smoke tests.

    Trains LSTM/GRU/Transformer on AAPL, saves model/predictions in standard artifact paths,
    and writes a minimal `summary.json` compatible with app/report loaders.
    """
    _seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (quick mode)", flush=True)

    audit = run_audit()
    for flagged in audit.get("flagged", []):
        if flagged.get("feature") in config.FEATURE_COLS:
            reasons = ", ".join(flagged.get("reasons", []))
            print(
                f"[train] warning: feature {flagged.get('feature')} for {flagged.get('ticker')} "
                f"flagged by audit ({reasons}). Continuing training.",
                flush=True,
            )

    raw_data, macro = load_all_with_macro()
    ticker = "AAPL"
    prepared = prepare_ticker_data(ticker, raw_data, macro)
    input_size = prepared["X_train"].shape[2]
    last_known_price = float(prepared["feature_frame"]["Close"].iloc[-len(prepared["y_test"]) - 1])
    actual_prices = returns_to_prices(last_known_price, prepared["y_test"])
    per_models = {}
    curves = {}
    for name in ["lstm", "gru", "transformer"]:
        model = get_model(name, input_size=input_size)
        model, tloss, vloss, stop_ep = train_one(
            model, prepared["train_loader"], prepared["val_loader"], ticker, name, device
        )
        test_mean, test_lo, test_hi, test_samples = mc_dropout_predict(
            model, prepared["X_test"], device, n_samples=config.MC_DROPOUT_SAMPLES
        )
        val_mean, _, _, val_samples = mc_dropout_predict(
            model, prepared["X_val"], device, n_samples=config.MC_DROPOUT_SAMPLES
        )
        raw_calib = calibration_report(prepared["y_test"], test_samples)
        q_scales = conformal_scale_factors(prepared["y_val"], val_mean, val_samples)
        save_conformal_scores(ticker, name, q_scales)
        calib = conformal_calibrated_report(prepared["y_test"], test_mean, test_samples, q_scales)
        test_lo, test_hi = calibrated_interval_from_conformal(
            test_mean, test_samples, q_scale=q_scales.get("q_90", 1.0), level=0.9
        )
        cov_90 = iv_coverage(prepared["y_test"], test_lo, test_hi)
        pred_prices = returns_to_prices(last_known_price, test_mean)
        m_ret = metrics_on_returns(prepared["y_test"], test_mean)
        m_price = metrics_on_prices(actual_prices, pred_prices)
        np.savez(
            os.path.join(config.RESULTS_DIR, f"{name}_{ticker}_preds.npz"),
            y_true=prepared["y_test"],
            y_pred=test_mean,
            y_pred_lower=test_lo,
            y_pred_upper=test_hi,
            actual_price=actual_prices,
            pred_price=pred_prices,
            train_losses=np.array(tloss),
            val_losses=np.array(vloss),
            test_dates=prepared["test_dates"].astype(str).values,
        )
        per_models[name] = {
            "metrics_returns": m_ret,
            "metrics_prices": m_price,
            "calibration": calib,
            "calibration_report": calib,
            "raw_calibration_report": raw_calib,
            "conformal_scales": q_scales,
            "interval_coverage_90": cov_90,
            "early_stop_epoch": stop_ep,
        }
        curves[name] = {"train_loss": tloss, "val_loss": vloss}

    summary = {
        "per_ticker": {
            ticker: {
                "ticker": ticker,
                "models": per_models,
                "baselines": {},
                "ensemble_weights": {},
                "training_curves": curves,
            }
        },
        "config": {
            "tickers": [ticker],
            "feature_cols": config.FEATURE_COLS,
            "target": config.TARGET_COL,
            "sequence_length": config.SEQUENCE_LENGTH,
            "hidden_size": config.HIDDEN_SIZE,
            "epochs": config.EPOCHS,
            "patience": config.PATIENCE,
            "quick_mode": True,
        },
    }
    serializable = json.loads(
        json.dumps(summary, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o))
    )
    out = os.path.join(config.RESULTS_DIR, "summary.json")
    with open(out, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n→ Quick summary saved: {out}", flush=True)
    return summary


if __name__ == "__main__":
    if "--quick" in sys.argv:
        run_quick_pipeline()
    else:
        run_full_pipeline()
