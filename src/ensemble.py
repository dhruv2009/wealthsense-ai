"""
Dynamic inverse-RMSE ensemble.

Models that perform better on the most recent validation window get more weight.
Per-ticker weights are persisted so you can show "the GRU gets 0.40 weight on
TSLA but only 0.30 on SPY" — a much richer story than naive averaging.
"""
from __future__ import annotations
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


_WF_CACHE: dict | None = None


def _load_walk_forward_summary() -> dict | None:
    """Load walk-forward summary JSON once per process."""
    global _WF_CACHE
    if _WF_CACHE is not None:
        return _WF_CACHE
    path = os.path.join(config.RESULTS_DIR, "walk_forward_summary.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            _WF_CACHE = json.load(f)
        return _WF_CACHE
    except Exception:
        return None


def _ticker_decision_from_walk_forward(ticker: str, names: list[str]) -> tuple[str | None, float | None, float | None, str]:
    """Return WF-based decision metadata for ticker-level ensemble behavior."""
    wf = _load_walk_forward_summary()
    if not wf:
        return None, None, None, "blend (no walk-forward summary)"
    tk_blob = ((wf.get("tickers") or {}).get(ticker) or {})
    if not tk_blob:
        return None, None, None, "blend (ticker missing in walk-forward summary)"

    best_name, best_mape = None, None
    for n in names:
        if n == "ensemble":
            continue
        mape = (((tk_blob.get(n) or {}).get("mean_metrics") or {}).get("MAPE_%"))
        if mape is None:
            continue
        if best_mape is None or float(mape) < best_mape:
            best_name, best_mape = n, float(mape)

    ens_mape = (((tk_blob.get("ensemble") or {}).get("mean_metrics") or {}).get("MAPE_%"))
    ens_mape = float(ens_mape) if ens_mape is not None else None
    if best_name is None or best_mape is None or ens_mape is None:
        return best_name, best_mape, ens_mape, "blend (insufficient walk-forward metrics)"
    if (ens_mape - best_mape) > 2.0:
        return best_name, best_mape, ens_mape, f"use {best_name} only"
    return best_name, best_mape, ens_mape, "blend (rolling inverse-RMSE)"


def rolling_inverse_rmse_weights(val_pred_map: dict[str, np.ndarray],
                                 val_true: np.ndarray,
                                 window: int = 21,
                                 ticker: str | None = None) -> dict[str, float]:
    """
    Weight each model by 1 / RMSE on the last `window` validation points.
    Falls back to equal weights if val length < window.
    If walk-forward summary exists and a single model beats ensemble by >2%
    MAPE for the given ticker, use that model alone.
    """
    names = list(val_pred_map.keys())
    if ticker:
        best_name, best_mape, ens_mape, decision = _ticker_decision_from_walk_forward(ticker, names)
        if best_name and decision.startswith("use "):
            weights = {n: (1.0 if n == best_name else 0.0) for n in names}
            print(
                "Ticker | Best individual model | Its WF MAPE | Ensemble WF MAPE | Decision",
                flush=True,
            )
            print(
                f"{ticker} | {best_name} | {best_mape:.4f} | {ens_mape:.4f} | {decision}",
                flush=True,
            )
            return weights
        print(
            "Ticker | Best individual model | Its WF MAPE | Ensemble WF MAPE | Decision",
            flush=True,
        )
        best_txt = f"{best_name}" if best_name else "N/A"
        best_mape_txt = f"{best_mape:.4f}" if best_mape is not None else "N/A"
        ens_mape_txt = f"{ens_mape:.4f}" if ens_mape is not None else "N/A"
        print(f"{ticker} | {best_txt} | {best_mape_txt} | {ens_mape_txt} | {decision}", flush=True)

    if len(val_true) < window:
        w = np.ones(len(names)) / len(names)
        return dict(zip(names, w.tolist()))

    rmse = []
    for n in names:
        recent_pred = val_pred_map[n][-window:]
        recent_true = val_true[-window:]
        rmse.append(float(np.sqrt(np.mean((recent_true - recent_pred) ** 2))))
    inv = np.array([1.0 / max(r, 1e-8) for r in rmse])
    w = inv / inv.sum()
    return dict(zip(names, w.tolist()))


def ensemble_predict(test_pred_map: dict[str, np.ndarray],
                     weights: dict[str, float]) -> np.ndarray:
    """Linear combination of test-set predictions using model weights."""
    out = np.zeros(len(next(iter(test_pred_map.values()))))
    for name, w in weights.items():
        out += w * test_pred_map[name]
    return out
