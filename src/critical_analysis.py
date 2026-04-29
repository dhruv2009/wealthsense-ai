"""
Automatic critical analysis generation from project summary outputs.
"""
from __future__ import annotations

from typing import Any


def _get_price_mape(model_blob: dict) -> float | None:
    """Extract price-domain MAPE_% from a model result blob when available."""
    prices = (model_blob or {}).get("metrics_prices") or {}
    return prices.get("MAPE_%")


def generate_analysis(summary: dict, ticker: str) -> dict:
    """
    Build a structured critical analysis report for one ticker.

    Parameters:
        summary: Parsed `summary.json` object from training pipeline.
        ticker: Ticker symbol to analyze in the Forecast View.

    Returns:
        Dict containing best/worst model diagnostics, calibration checks,
        directional-accuracy warnings, and mandatory limitations list.
    """
    per_ticker = (summary or {}).get("per_ticker") or {}
    ticker_blob = per_ticker.get(ticker) or {}
    models = ticker_blob.get("models") or {}
    baselines = ticker_blob.get("baselines") or {}
    all_models = {**models, **baselines}

    best_model = None
    best_mape = float("inf")
    for m_name, m_blob in all_models.items():
        mape = _get_price_mape(m_blob)
        if mape is not None and mape < best_mape:
            best_mape = mape
            best_model = m_name

    worst_ticker_for_best_model = None
    worst_ticker_mape = None
    if best_model is not None:
        for tk, tk_blob in per_ticker.items():
            source = (tk_blob.get("models") or {}) if best_model in (tk_blob.get("models") or {}) else (tk_blob.get("baselines") or {})
            mape = _get_price_mape(source.get(best_model))
            if mape is None:
                continue
            if worst_ticker_mape is None or mape > worst_ticker_mape:
                worst_ticker_mape = mape
                worst_ticker_for_best_model = tk

    directional_accuracy = {}
    directional_flags = []
    for m_name, m_blob in all_models.items():
        dacc = ((m_blob or {}).get("metrics_returns") or {}).get("Dir_Acc")
        if dacc is None:
            continue
        directional_accuracy[m_name] = float(dacc)
        if dacc < 52:
            directional_flags.append(
                f"{m_name}: near-random: model performs only {float(dacc) - 50:.2f}% better than coin flip"
            )

    calibration_verdict = {}
    calibration_flags = []
    for m_name in ["lstm", "gru", "transformer"]:
        cal = ((models.get(m_name) or {}).get("calibration") or {})
        actual_90 = cal.get("actual_90pct")
        if actual_90 is None:
            calibration_verdict[m_name] = "missing calibration data"
            continue
        diff = abs(float(actual_90) - 0.90)
        if diff <= 0.05:
            calibration_verdict[m_name] = f"well_calibrated: actual coverage {float(actual_90)*100:.2f}% vs target 90%"
        else:
            msg = f"miscalibrated: actual coverage {float(actual_90)*100:.2f}% vs target 90%"
            calibration_verdict[m_name] = msg
            calibration_flags.append(f"{m_name}: {msg}")

    ensemble_vs_best_individual = {}
    ensemble_blob = models.get("ensemble") or {}
    ensemble_mape = _get_price_mape(ensemble_blob)
    individual_mapes = {k: _get_price_mape(v) for k, v in models.items() if k != "ensemble" and _get_price_mape(v) is not None}
    if ensemble_mape is not None and individual_mapes:
        best_ind_name = min(individual_mapes, key=lambda k: individual_mapes[k])
        best_ind_mape = individual_mapes[best_ind_name]
        improvement = float(best_ind_mape - ensemble_mape)
        ensemble_vs_best_individual = {
            "best_individual_model": best_ind_name,
            "best_individual_mape": float(best_ind_mape),
            "ensemble_mape": float(ensemble_mape),
            "improvement_mape_pct_points": improvement,
            "verdict": (
                f"ensemble provides marginal improvement of only {improvement:.2f}% MAPE"
                if improvement <= 0.5
                else f"ensemble improves by {improvement:.2f}% MAPE vs best individual"
            ),
        }
    else:
        ensemble_vs_best_individual = {"verdict": "insufficient data"}

    high_volatility_ticker = {"ticker": "TSLA"}
    tsla_blob = per_ticker.get("TSLA") or {}
    spy_blob = per_ticker.get("SPY") or {}
    if best_model:
        tsla_source = (tsla_blob.get("models") or {}) if best_model in (tsla_blob.get("models") or {}) else (tsla_blob.get("baselines") or {})
        spy_source = (spy_blob.get("models") or {}) if best_model in (spy_blob.get("models") or {}) else (spy_blob.get("baselines") or {})
        tsla_mape = _get_price_mape(tsla_source.get(best_model))
        spy_mape = _get_price_mape(spy_source.get(best_model))
        if tsla_mape is not None and spy_mape not in (None, 0):
            ratio = float(tsla_mape / spy_mape)
            high_volatility_ticker.update(
                {
                    "best_model": best_model,
                    "tsla_mape": float(tsla_mape),
                    "spy_mape": float(spy_mape),
                    "tsla_vs_spy_ratio": ratio,
                    "note": f"TSLA error is {ratio:.2f}x SPY for {best_model}",
                }
            )
        else:
            high_volatility_ticker["note"] = "insufficient TSLA/SPY data"

    limitations = [
        "Results evaluated on a single 2023 test period — walk-forward validation provides more robust estimates",
        "Universe limited to 5 large-cap US equities — generalisability to small-cap or international markets is untested",
        "Transaction costs modelled as flat 5bps — does not account for market impact or bid-ask spread",
        "Log-return forecasts assume next-day price changes are predictable — efficient market hypothesis suggests systematic outperformance is unlikely to persist",
    ]
    limitations.extend(directional_flags)
    limitations.extend(calibration_flags)
    if "marginal improvement" in (ensemble_vs_best_individual.get("verdict") or ""):
        limitations.append(ensemble_vs_best_individual["verdict"])

    return {
        "best_model": best_model,
        "worst_ticker_for_best_model": worst_ticker_for_best_model,
        "directional_accuracy": directional_accuracy,
        "calibration_verdict": calibration_verdict,
        "ensemble_vs_best_individual": ensemble_vs_best_individual,
        "high_volatility_ticker": high_volatility_ticker,
        "limitations": limitations,
    }
