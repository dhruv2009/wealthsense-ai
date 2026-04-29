"""
Feature quality audit for configured model inputs.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_pipeline import build_feature_frame, load_all_with_macro


def _feature_stats(values: np.ndarray) -> dict:
    """Compute NaN/zero rates and summary statistics for one feature array."""
    total = len(values)
    if total == 0:
        return {
            "pct_nan": 100.0,
            "pct_zero": 0.0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "is_constant": False,
        }
    nan_mask = np.isnan(values)
    clean = values[~nan_mask]
    if len(clean) == 0:
        return {
            "pct_nan": 100.0,
            "pct_zero": 0.0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "is_constant": False,
        }
    std = float(np.std(clean))
    return {
        "pct_nan": float(np.mean(nan_mask) * 100),
        "pct_zero": float(np.mean(clean == 0) * 100),
        "mean": float(np.mean(clean)),
        "std": std,
        "min": float(np.min(clean)),
        "max": float(np.max(clean)),
        "is_constant": bool(std < 1e-8),
    }


def run_audit() -> dict:
    """
    Audit each configured feature across all tickers and flag weak signals.

    Returns:
        Dict with per-ticker feature statistics and flagged features.
        Also saves `feature_audit.json` to `config.RESULTS_DIR`.
    """
    raw_data, macro = load_all_with_macro()
    report = {"tickers": {}, "flagged": []}
    print("\n=== Feature Audit Report ===")
    for ticker in config.TICKERS:
        if ticker not in raw_data:
            print(f"[feature-audit] warning: missing raw data for {ticker}.")
            continue
        frame = build_feature_frame(ticker, raw_data[ticker], macro)
        report["tickers"][ticker] = {}
        print(f"\n[{ticker}]")
        for col in config.FEATURE_COLS:
            if col not in frame.columns:
                print(f"  - {col}: MISSING")
                continue
            stats = _feature_stats(frame[col].astype(float).values)
            reasons = []
            if stats["pct_zero"] > 80:
                reasons.append("zeros_gt_80pct")
            if stats["is_constant"]:
                reasons.append("constant")
            if stats["pct_nan"] > 10:
                reasons.append("nan_gt_10pct")
            stats["flag_reasons"] = reasons
            report["tickers"][ticker][col] = stats
            print(
                f"  - {col}: nan={stats['pct_nan']:.2f}% zero={stats['pct_zero']:.2f}% "
                f"mean={stats['mean']:.4f} std={stats['std']:.4f} "
                f"min={stats['min']:.4f} max={stats['max']:.4f}"
            )
            if reasons:
                warning = {
                    "ticker": ticker,
                    "feature": col,
                    "reasons": reasons,
                }
                report["flagged"].append(warning)
                print(f"WARNING: {ticker}::{col} flagged for {', '.join(reasons)}")

    out = os.path.join(config.RESULTS_DIR, "feature_audit.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[feature-audit] saved {out}")
    return report


if __name__ == "__main__":
    run_audit()
