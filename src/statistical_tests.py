"""
Statistical significance tests for forecast comparison.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.stats import t as student_t

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def diebold_mariano_test(e1, e2, h: int = 1, crit: str = "MSE") -> dict:
    """
    Run the Diebold-Mariano predictive accuracy test with HLN correction.

    Parameters:
        e1: Forecast errors for model 1 (actual - predicted).
        e2: Forecast errors for model 2 (actual - predicted).
        h: Forecast horizon (h=1 for one-step-ahead).
        crit: Loss criterion, "MSE" or "MAE".

    Returns:
        Dictionary containing DM statistic, p-value, significance flags, and
        the better model label.
    """
    e1 = np.asarray(e1, dtype=float)
    e2 = np.asarray(e2, dtype=float)
    n = min(len(e1), len(e2))
    if n < 3:
        return {
            "dm_statistic": 0.0,
            "p_value": 1.0,
            "significant_5pct": False,
            "significant_1pct": False,
            "better_model": "indistinguishable",
        }
    e1, e2 = e1[:n], e2[:n]

    crit_up = crit.upper()
    if crit_up == "MAE":
        d = np.abs(e1) - np.abs(e2)
    else:
        d = (e1 ** 2) - (e2 ** 2)

    d_bar = float(np.mean(d))
    t_len = len(d)
    d_centered = d - d_bar

    if h <= 1:
        s = float(np.var(d_centered, ddof=1))
    else:
        gamma0 = float(np.var(d_centered, ddof=1))
        s = gamma0
        max_lag = min(h - 1, t_len - 1)
        for lag in range(1, max_lag + 1):
            cov = float(np.mean(d_centered[lag:] * d_centered[:-lag]))
            weight = 1.0 - lag / (max_lag + 1.0)
            s += 2.0 * weight * cov

    s = max(s, 1e-12)
    dm = d_bar / np.sqrt(s / t_len)
    hln = np.sqrt((t_len + 1 - 2 * h + (h * (h - 1) / t_len)) / t_len)
    dm_hln = float(dm * hln)
    p_value = float(2 * (1 - student_t.cdf(np.abs(dm_hln), df=t_len - 1)))

    better = "indistinguishable"
    if p_value < 0.05:
        better = "model1" if d_bar < 0 else "model2"

    return {
        "dm_statistic": dm_hln,
        "p_value": p_value,
        "significant_5pct": bool(p_value < 0.05),
        "significant_1pct": bool(p_value < 0.01),
        "better_model": better,
    }


def run_dm_tests(ticker: str) -> dict:
    """
    Run requested DM pairwise tests for one ticker using saved prediction files.

    Parameters:
        ticker: Stock ticker symbol.

    Returns:
        Dict keyed by "model1_vs_model2" with DM statistics. Also saves
        `dm_tests_{ticker}.json` to `config.RESULTS_DIR`.
    """
    pairs = [
        ("lstm", "gru"),
        ("lstm", "transformer"),
        ("lstm", "ensemble"),
        ("gru", "transformer"),
        ("gru", "ensemble"),
        ("transformer", "ensemble"),
        ("lstm", "arima"),
        ("gru", "arima"),
        ("transformer", "arima"),
        ("ensemble", "arima"),
        ("lstm", "naive"),
        ("ensemble", "naive"),
    ]

    errors = {}
    for model in sorted({m for p in pairs for m in p}):
        path = os.path.join(config.RESULTS_DIR, f"{model}_{ticker}_preds.npz")
        if not os.path.exists(path):
            print(f"[dm] warning: missing {path}, skipping model {model}.")
            continue
        d = np.load(path, allow_pickle=True)
        if "y_true" not in d.files or "y_pred" not in d.files:
            print(f"[dm] warning: malformed {path}, skipping model {model}.")
            continue
        errors[model] = d["y_true"] - d["y_pred"]

    out = {}
    for m1, m2 in pairs:
        key = f"{m1}_vs_{m2}"
        if m1 not in errors or m2 not in errors:
            print(f"[dm] warning: missing data for {key}, skipping.")
            continue
        out[key] = diebold_mariano_test(errors[m1], errors[m2], h=1, crit="MSE")

    out_path = os.path.join(config.RESULTS_DIR, f"dm_tests_{ticker}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[dm] saved {out_path}")
    return out


if __name__ == "__main__":
    for tk in config.TICKERS:
        run_dm_tests(tk)
