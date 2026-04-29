"""
Examiner-facing one-command summary for WealthSense AI results.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src.feature_audit import run_audit


def _load_json(path: str):
    """Load JSON file if it exists; otherwise print warning and return None."""
    if not os.path.exists(path):
        print(f"[results-summary] warning: missing {path}")
        return None
    with open(path) as f:
        return json.load(f)


def _print_results_table(summary: dict) -> None:
    """Print the main ticker x model metric table sorted by MAPE_%."""
    rows = []
    for ticker, blob in (summary.get("per_ticker") or {}).items():
        for model_name, mblob in (blob.get("models") or {}).items():
            r = mblob.get("metrics_returns") or {}
            p = mblob.get("metrics_prices") or {}
            rows.append(
                {
                    "Ticker": ticker,
                    "Model": model_name,
                    "SMAPE": r.get("SMAPE"),
                    "Dir_Acc": r.get("Dir_Acc"),
                    "MAE_$": p.get("MAE_$"),
                    "MAPE_%": p.get("MAPE_%"),
                }
            )
        for model_name, mblob in (blob.get("baselines") or {}).items():
            r = mblob.get("metrics_returns") or {}
            p = mblob.get("metrics_prices") or {}
            rows.append(
                {
                    "Ticker": ticker,
                    "Model": model_name,
                    "SMAPE": r.get("SMAPE"),
                    "Dir_Acc": r.get("Dir_Acc"),
                    "MAE_$": p.get("MAE_$"),
                    "MAPE_%": p.get("MAPE_%"),
                }
            )
    if not rows:
        print("[results-summary] no rows found in summary.")
        return
    df = pd.DataFrame(rows).sort_values(["Ticker", "MAPE_%"], ascending=[True, True])
    print("\n=== Main Results (sorted by MAPE_% within ticker) ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def _print_ablation_tables():
    """Print original and extended ablation tables when artifact files exist."""
    print("\n=== Ablation Results ===")
    orig = _load_json(os.path.join(config.RESULTS_DIR, "ablation_AAPL_gru.json"))
    ext = _load_json(os.path.join(config.RESULTS_DIR, "ablation_extended_AAPL_gru.json"))
    if orig:
        rows = []
        for cfg, res in orig.items():
            rows.append(
                {
                    "Config": cfg,
                    "n_features": res.get("n_features"),
                    "MAPE_%": (res.get("metrics_prices") or {}).get("MAPE_%"),
                    "MAE_$": (res.get("metrics_prices") or {}).get("MAE_$"),
                    "Dir_Acc": (res.get("metrics_returns") or {}).get("Dir_Acc"),
                }
            )
        print("\nOriginal ablation")
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if ext:
        rows = []
        for cfg, res in ext.items():
            rows.append(
                {
                    "Config": cfg,
                    "n_features": res.get("n_features"),
                    "MAPE_%": (res.get("metrics_prices") or {}).get("MAPE_%"),
                    "MAE_$": (res.get("metrics_prices") or {}).get("MAE_$"),
                    "Dir_Acc": (res.get("metrics_returns") or {}).get("Dir_Acc"),
                    "Notes": ",".join([f"{k}={v}" for k, v in (res.get("notes") or {}).items()]),
                }
            )
        print("\nExtended ablation")
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def _print_dm_tables():
    """Print Diebold-Mariano pairwise test tables for available tickers."""
    print("\n=== Diebold-Mariano Tests ===")
    for ticker in config.TICKERS:
        path = os.path.join(config.RESULTS_DIR, f"dm_tests_{ticker}.json")
        dm = _load_json(path)
        if not dm:
            continue
        rows = []
        for pair, res in dm.items():
            rows.append(
                {
                    "pair": pair,
                    "dm_stat": res.get("dm_statistic"),
                    "p_value": res.get("p_value"),
                    "sig_5pct": res.get("significant_5pct"),
                    "better_model": res.get("better_model"),
                }
            )
        print(f"\n[{ticker}]")
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.6f}"))


def _print_walk_forward():
    """Print walk-forward summary aggregates when available."""
    print("\n=== Walk-Forward Summary ===")
    wf = _load_json(os.path.join(config.RESULTS_DIR, "walk_forward_summary.json"))
    if not wf:
        return
    rows = []
    for ticker, models in (wf.get("tickers") or {}).items():
        for model_name, blob in models.items():
            mm = blob.get("mean_metrics") or {}
            ss = blob.get("std_metrics") or {}
            rows.append(
                {
                    "Ticker": ticker,
                    "Model": model_name,
                    "n_folds": blob.get("n_folds"),
                    "mean_MAPE_%": mm.get("MAPE_%"),
                    "std_MAPE_%": ss.get("MAPE_%"),
                }
            )
    if rows:
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def _print_calibration(summary: dict):
    """Print calibration levels for all DL models across tickers."""
    print("\n=== Calibration (DL models) ===")
    rows = []
    for ticker, blob in (summary.get("per_ticker") or {}).items():
        for model_name in ["lstm", "gru", "transformer"]:
            cal = ((blob.get("models") or {}).get(model_name) or {}).get("calibration") or {}
            if not cal:
                continue
            rows.append(
                {
                    "Ticker": ticker,
                    "Model": model_name,
                    "Actual_50pct": cal.get("actual_50pct"),
                    "Actual_80pct": cal.get("actual_80pct"),
                    "Actual_90pct": cal.get("actual_90pct"),
                }
            )
    if rows:
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def _final_paragraph(summary: dict):
    """Print one-paragraph plain-English final findings summary."""
    rows = []
    dl_vs_arima = []
    cal90 = []
    for ticker, blob in (summary.get("per_ticker") or {}).items():
        model_map = blob.get("models") or {}
        baseline_map = blob.get("baselines") or {}
        for m_name, mblob in model_map.items():
            mape = ((mblob.get("metrics_prices") or {}).get("MAPE_%"))
            if mape is not None:
                rows.append((ticker, m_name, float(mape)))
        arima = ((baseline_map.get("arima") or {}).get("metrics_prices") or {}).get("MAPE_%")
        if arima is not None:
            dl_mapes = [((model_map.get(n) or {}).get("metrics_prices") or {}).get("MAPE_%") for n in ["lstm", "gru", "transformer", "ensemble"]]
            dl_mapes = [float(v) for v in dl_mapes if v is not None]
            if dl_mapes:
                best_dl = min(dl_mapes)
                dl_vs_arima.append((float(arima) - best_dl) / float(arima) * 100.0)
        for n in ["lstm", "gru", "transformer"]:
            cal = ((model_map.get(n) or {}).get("calibration") or {}).get("actual_90pct")
            if cal is not None:
                cal90.append(float(cal))

    best_model = "N/A"
    if rows:
        _, best_model, _ = min(rows, key=lambda x: x[2])
    avg_dl_gain = float(np.mean(dl_vs_arima)) if dl_vs_arima else 0.0
    cal_level = float(np.mean(cal90) * 100) if cal90 else 0.0
    print(
        f"\nBest overall model: {best_model}. DL outperforms ARIMA baseline by "
        f"{avg_dl_gain:.2f}% MAPE on average across all tickers. Uncertainty intervals "
        f"are {cal_level:.2f}% calibrated at the 90% target level."
    )


def main():
    """Execute the full examiner-facing summary workflow in one command."""
    print("Running feature audit...")
    run_audit()

    summary_path = os.path.join(config.RESULTS_DIR, "summary.json")
    summary = _load_json(summary_path)
    if not summary:
        print("[results-summary] summary.json is required; run `python src/train.py` first.")
        return

    _print_results_table(summary)
    _print_ablation_tables()
    _print_dm_tables()
    _print_walk_forward()
    _print_calibration(summary)
    _final_paragraph(summary)


if __name__ == "__main__":
    main()
