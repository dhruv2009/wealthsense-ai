"""
Walk-forward validation utilities for WealthSense AI.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.baselines import arima_forecast, naive_persistence_forecast
from src.data_pipeline import build_feature_frame, load_all_with_macro
from src.ensemble import ensemble_predict, rolling_inverse_rmse_weights
from src.models import get_model
from src.train import metrics_on_prices, metrics_on_returns, returns_to_prices


def _to_serializable(value):
    """Convert numpy scalars/arrays into JSON-serializable Python types."""
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    return value


def _seed_everything(seed: int = config.SEED) -> None:
    """Set deterministic seeds for numpy and torch operations."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(int(rng.integers(0, 1_000_000)))
    torch.cuda.manual_seed_all(int(rng.integers(0, 1_000_000)))


def _build_sequences(feats_scaled: np.ndarray, target: np.ndarray, seq_len: int):
    """Convert flat feature matrix/target into sliding-window sequences."""
    X, y = [], []
    for i in range(seq_len, len(feats_scaled)):
        X.append(feats_scaled[i - seq_len:i])
        y.append(target[i])
    return np.array(X), np.array(y)


def _build_fold_arrays(
    feature_frame: pd.DataFrame, train_end: pd.Timestamp, test_end: pd.Timestamp
) -> dict | None:
    """Create train/test arrays for a fold with train-only scaler fitting."""
    fold_df = feature_frame[feature_frame["Date"] <= test_end].copy()
    if fold_df.empty:
        return None

    feats = fold_df[config.FEATURE_COLS].astype(float).values
    target = fold_df[config.TARGET_COL].astype(float).values
    dates = pd.to_datetime(fold_df["Date"])
    train_mask = dates <= train_end
    test_mask = (dates > train_end) & (dates <= test_end)
    if train_mask.sum() <= config.SEQUENCE_LENGTH or test_mask.sum() == 0:
        return None

    scaler = StandardScaler()
    scaler.fit(feats[train_mask.values])
    feats_scaled = scaler.transform(feats)

    X_all, y_all = _build_sequences(feats_scaled, target, config.SEQUENCE_LENGTH)
    seq_dates = dates.iloc[config.SEQUENCE_LENGTH:].reset_index(drop=True)
    train_sel = (seq_dates <= train_end).values
    test_sel = ((seq_dates > train_end) & (seq_dates <= test_end)).values
    if train_sel.sum() == 0 or test_sel.sum() == 0:
        return None

    return {
        "X_train": X_all[train_sel],
        "y_train": y_all[train_sel],
        "X_test": X_all[test_sel],
        "y_test": y_all[test_sel],
        "seq_dates_test": seq_dates[test_sel].reset_index(drop=True),
        "fold_df": fold_df,
        "scaler": scaler,
    }


def _finetune_from_saved(
    model_name: str, ticker: str, X_train: np.ndarray, y_train: np.ndarray, input_size: int, device
):
    """Load saved DL weights and fine-tune for one fold with low learning rate."""
    model_path = os.path.join(config.SAVED_MODELS_DIR, f"{model_name}_{ticker}.pt")
    if not os.path.exists(model_path):
        print(f"[walk-forward] warning: missing saved model {model_path}, skipping.")
        return None

    model = get_model(model_name, input_size=input_size).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)

    ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=config.WF_FINETUNE_LR, weight_decay=config.WEIGHT_DECAY)
    crit = nn.MSELoss()
    model.train()
    for _ in range(config.WF_FINETUNE_EPOCHS):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    return model


def _predict_model(model, X_test: np.ndarray, device) -> np.ndarray:
    """Run model inference for a fold test matrix and return numpy predictions."""
    with torch.no_grad():
        xb = torch.tensor(X_test, dtype=torch.float32, device=device)
        pred = model(xb).detach().cpu().numpy()
    return pred.astype(float)


def _fold_schedule(feature_frame: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Build expanding-window fold boundaries using config walk-forward settings."""
    start_date = pd.to_datetime(feature_frame["Date"].min())
    end_date = pd.to_datetime(feature_frame["Date"].max())
    min_train_end = start_date + pd.DateOffset(years=config.WF_MIN_TRAIN_YEARS) - pd.Timedelta(days=1)
    train_end = max(min_train_end, pd.Timestamp(config.TRAIN_END))
    folds = []
    while train_end < end_date:
        test_start = train_end + pd.Timedelta(days=1)
        test_end = min(test_start + pd.DateOffset(months=config.WF_FOLD_SIZE_MONTHS) - pd.Timedelta(days=1), end_date)
        folds.append((train_end, test_start, test_end))
        train_end = test_end
    return folds


def walk_forward_validate(ticker: str, model_name: str) -> dict:
    """
    Run expanding-window walk-forward validation for one ticker/model.

    Parameters:
        ticker: Stock ticker symbol.
        model_name: One of lstm, gru, transformer, ensemble, arima, naive.

    Returns:
        Dict with fold-wise metrics, aggregate mean/std metrics, and metadata.
        Also saves `wf_{model_name}_{ticker}.json` to `config.RESULTS_DIR`.
    """
    _seed_everything(config.SEED)
    raw_data, macro = load_all_with_macro()
    if ticker not in raw_data:
        print(f"[walk-forward] warning: ticker {ticker} data not found, skipping.")
        return {"model": model_name, "ticker": ticker, "n_folds": 0, "per_fold": [], "mean_metrics": {}, "std_metrics": {}}

    feature_frame = build_feature_frame(ticker, raw_data[ticker], macro)
    folds = _fold_schedule(feature_frame)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    per_fold = []
    for fold_id, (train_end, test_start, test_end) in enumerate(folds, start=1):
        fold = _build_fold_arrays(feature_frame, train_end, test_end)
        if fold is None or len(fold["y_test"]) == 0:
            continue
        X_train, y_train = fold["X_train"], fold["y_train"]
        X_test, y_test = fold["X_test"], fold["y_test"]

        if model_name in {"lstm", "gru", "transformer"}:
            model = _finetune_from_saved(model_name, ticker, X_train, y_train, X_train.shape[2], device)
            if model is None:
                continue
            y_pred = _predict_model(model, X_test, device)
        elif model_name in {"naive", "arima"}:
            if model_name == "naive":
                y_pred = naive_persistence_forecast(y_train, y_test)
            else:
                y_pred = arima_forecast(y_train, y_test, order=(5, 0, 0))
        elif model_name == "ensemble":
            pred_map = {}
            train_tail_pred_map = {}
            for dl_name in ("lstm", "gru", "transformer"):
                model = _finetune_from_saved(dl_name, ticker, X_train, y_train, X_train.shape[2], device)
                if model is None:
                    pred_map = {}
                    break
                pred_map[dl_name] = _predict_model(model, X_test, device)
                tail = min(21, len(y_train))
                train_tail_pred_map[dl_name] = _predict_model(model, X_train[-tail:], device)
            if not pred_map:
                continue
            # Use recent training tail as local validation for fold-specific dynamic weights.
            tail = min(21, len(y_train))
            val_pred_map = {k: v[-tail:] for k, v in train_tail_pred_map.items()}
            val_true = y_train[-tail:]
            weights = rolling_inverse_rmse_weights(val_pred_map, val_true, window=tail)
            y_pred = ensemble_predict(pred_map, weights)
        else:
            print(f"[walk-forward] warning: unsupported model {model_name}, skipping.")
            return {"model": model_name, "ticker": ticker, "n_folds": 0, "per_fold": [], "mean_metrics": {}, "std_metrics": {}}

        test_rows = feature_frame[pd.to_datetime(feature_frame["Date"]) > train_end]
        if test_rows.empty:
            continue
        test_start_idx = int(test_rows.index[0])
        last_known_price = float(feature_frame["Close"].iloc[max(test_start_idx - 1, 0)])
        true_prices = returns_to_prices(last_known_price, y_test)
        pred_prices = returns_to_prices(last_known_price, y_pred)

        fold_result = {
            "fold_id": fold_id,
            "train_end": str(train_end.date()),
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
            "metrics_returns": metrics_on_returns(y_test, y_pred),
            "metrics_prices": metrics_on_prices(true_prices, pred_prices),
        }
        per_fold.append(fold_result)

    metric_keys = []
    if per_fold:
        metric_keys = list(per_fold[0]["metrics_returns"].keys()) + list(per_fold[0]["metrics_prices"].keys())
    mean_metrics, std_metrics = {}, {}
    for key in metric_keys:
        values = []
        for f in per_fold:
            if key in f["metrics_returns"]:
                values.append(float(f["metrics_returns"][key]))
            elif key in f["metrics_prices"]:
                values.append(float(f["metrics_prices"][key]))
        if values:
            mean_metrics[key] = float(np.mean(values))
            std_metrics[key] = float(np.std(values))

    result = {
        "model": model_name,
        "ticker": ticker,
        "n_folds": len(per_fold),
        "per_fold": per_fold,
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
    }
    out_path = os.path.join(config.RESULTS_DIR, f"wf_{model_name}_{ticker}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=_to_serializable)
    print(f"[walk-forward] saved {out_path}")
    return result


def run_all_walk_forward() -> dict:
    """
    Run walk-forward validation for all ticker/model combinations.

    Returns:
        Combined summary dict keyed by ticker and model. Saves
        `walk_forward_summary.json` to `config.RESULTS_DIR`.
    """
    models = ["lstm", "gru", "transformer", "ensemble", "arima", "naive"]
    combined = {"tickers": {}, "models": models}
    for ticker in config.TICKERS:
        combined["tickers"][ticker] = {}
        for model_name in models:
            print(f"Running walk-forward: {ticker} | {model_name}", flush=True)
            res = walk_forward_validate(ticker, model_name)
            combined["tickers"][ticker][model_name] = {
                "n_folds": res.get("n_folds", 0),
                "mean_metrics": res.get("mean_metrics", {}),
                "std_metrics": res.get("std_metrics", {}),
            }
    out = os.path.join(config.RESULTS_DIR, "walk_forward_summary.json")
    with open(out, "w") as f:
        json.dump(combined, f, indent=2, default=_to_serializable)
    print(f"[walk-forward] saved {out}")
    return combined


if __name__ == "__main__":
    run_all_walk_forward()
