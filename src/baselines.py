"""
Classical baselines: naive persistence + ARIMA.

Why this matters: a DL paper without a non-DL baseline can't claim deep learning
adds value. These two baselines let us say "DL beats persistence by X% and ARIMA
by Y%" — without that comparison, the project is incomplete.

Both baselines predict log returns on the same test split as the DL models so
metrics are directly comparable.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ── Naive persistence ───────────────────────────────────────────────────
def naive_persistence_forecast(train_returns: np.ndarray,
                               test_returns: np.ndarray) -> np.ndarray:
    """
    Predict next-day log return = today's log return.
    For test point t, prediction = actual return at t-1.
    The first prediction uses the last train return.
    """
    preds = np.empty_like(test_returns)
    preds[0] = train_returns[-1]
    preds[1:] = test_returns[:-1]
    return preds


# ── ARIMA ───────────────────────────────────────────────────────────────
def arima_forecast(train_returns: np.ndarray, test_returns: np.ndarray,
                   order: tuple = (5, 0, 0)) -> np.ndarray:
    """
    Walk-forward ARIMA forecast on log returns.
    Refits on each test step using train + observed test history.
    Returns array of test_returns shape predictions.

    `order=(p, d, q)` defaults to AR(5) which is standard for daily returns.
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError as e:
        raise ImportError("statsmodels is required for ARIMA baseline") from e

    history = list(train_returns)
    preds = np.empty(len(test_returns))
    for i, actual in enumerate(test_returns):
        try:
            model = ARIMA(history, order=order)
            fit = model.fit(method_kwargs={"warn_convergence": False})
            yhat = fit.forecast(steps=1)
            preds[i] = float(yhat[0])
        except Exception:
            # Fall back to last observation if ARIMA fails (rare on noisy data).
            preds[i] = history[-1]
        history.append(actual)
    return preds


# ── Convenience runner ──────────────────────────────────────────────────
def run_baselines(prepared: dict) -> dict[str, np.ndarray]:
    """
    Given prepared ticker data (from data_pipeline.prepare_ticker_data),
    return dict {model_name: predictions on test set} for each baseline.
    """
    # Reconstruct flat log-return series from windowed targets.
    train_y = prepared["y_train"]
    val_y = prepared["y_val"]
    test_y = prepared["y_test"]

    # ARIMA fits on full pre-test history
    pre_test = np.concatenate([train_y, val_y])
    return {
        "naive": naive_persistence_forecast(pre_test, test_y),
        "arima": arima_forecast(pre_test, test_y, order=(5, 0, 0)),
    }
