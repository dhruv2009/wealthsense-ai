"""
Gemini-API-backed chat with structured context injection + free fallback.

If GOOGLE_API_KEY is set, every user question is sent to Google Gemini along
with a compact summary of the current dashboard state (best models per
ticker, model outlook, latest goal plan). If the key is missing, falls
back to a keyword-based smart_reply engine that answers from model results.
"""
from __future__ import annotations
import os
import sys
import json
import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

load_dotenv()


# ═══════════════════════════════════════════════════════════════════════
#  Google Gemini API chat (used when GOOGLE_API_KEY is set)
# ═══════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are WealthSense AI, a financial research assistant built into a
Masters-level deep learning project. You have access to real model
results. Always answer in 2-3 sentences maximum unless asked to explain
something in detail. Always use the actual numbers provided in the
context. Never say 'I cannot provide financial advice' — instead say
'based on the model results' and give the specific answer.

Current project context:
- Models: LSTM, GRU, Transformer, Ensemble, ARIMA, Naive
- Tickers: AAPL, MSFT, NVDA, TSLA, SPY
- Evaluation: walk-forward validation across 4 folds
- Key finding: ensemble beats ARIMA significantly on MSFT, NVDA, SPY (p<0.01)
- Key finding: directional accuracy hovers near 50% — consistent with EMH
- Key finding: conformal calibration improved uncertainty coverage significantly"""


def _infer_ticker_from_text(text: str) -> str | None:
    """Infer ticker mention from free text for context routing."""
    text_upper = (text or "").upper()
    for t in config.TICKERS:
        if t in text_upper:
            return t
    name_map = {
        "APPLE": "AAPL",
        "MICROSOFT": "MSFT",
        "NVIDIA": "NVDA",
        "TESLA": "TSLA",
        "S&P": "SPY",
        "SP500": "SPY",
        "S&P500": "SPY",
        "INDEX": "SPY",
        "MARKET": "SPY",
    }
    for name, ticker in name_map.items():
        if name in text_upper:
            return ticker
    return None


def _load_walk_forward_summary() -> dict | None:
    """Load walk-forward summary JSON if available."""
    path = os.path.join(config.RESULTS_DIR, "walk_forward_summary.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _load_dm_tests(ticker: str) -> dict | None:
    """Load DM test results for a single ticker if available."""
    path = os.path.join(config.RESULTS_DIR, f"dm_tests_{ticker}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _load_default_summary() -> dict | None:
    """Load summary.json when caller doesn't pass summary explicitly."""
    path = os.path.join(config.RESULTS_DIR, "summary.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _fmt_pct(value: object) -> str:
    """Format percent-like values or return data-not-available marker."""
    try:
        v = float(value)
        if 0 <= v <= 1:
            v *= 100.0
        return f"{v:.2f}"
    except Exception:
        return "data not available"


def _fmt_float(value: object, digits: int = 2) -> str:
    """Format float with fixed digits or data-not-available marker."""
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "data not available"


def _extract_ticker_blob(summary: dict | None, ticker: str) -> dict:
    """Return per-ticker summary blob across supported summary formats."""
    if not summary:
        return {}
    if isinstance(summary.get("per_ticker"), dict):
        return summary.get("per_ticker", {}).get(ticker, {}) or {}
    return summary.get(ticker, {}) or {}


def _build_gemini_context(summary: dict | None, ticker: str) -> str:
    """Build concise, artifact-backed context block for Gemini prompts."""
    ticker_blob = _extract_ticker_blob(summary, ticker)
    models = ticker_blob.get("models", {}) if isinstance(ticker_blob, dict) else {}
    best_model = "data not available"
    best_mape = "data not available"
    dir_acc = "data not available"
    actual_90 = "data not available"

    best_name = None
    best_mape_val = float("inf")
    for model_name, model_blob in models.items():
        mape = (model_blob.get("metrics_prices") or {}).get("MAPE_%")
        try:
            mape_val = float(mape)
        except Exception:
            continue
        if mape_val < best_mape_val:
            best_mape_val = mape_val
            best_name = model_name
    if best_name:
        best_model = best_name
        best_mape = _fmt_pct(best_mape_val)
        dir_acc = _fmt_pct((models.get(best_name, {}).get("metrics_returns") or {}).get("Dir_Acc"))
        calibration = models.get(best_name, {}).get("calibration_report") or {}
        actual_90 = _fmt_pct(calibration.get("actual_90pct"))

    wf_mape = "data not available"
    wf_std = "data not available"
    wf_summary = _load_walk_forward_summary() or {}
    wf_blob = ((wf_summary.get("tickers") or {}).get(ticker) or {}).get(best_name or "", {})
    if wf_blob:
        wf_mape = _fmt_pct((wf_blob.get("mean_metrics") or {}).get("MAPE_%"))
        wf_std = _fmt_pct((wf_blob.get("std_metrics") or {}).get("MAPE_%"))

    dm_significant = "data not available"
    dm_pvalue = "data not available"
    dm_blob = _load_dm_tests(ticker) or {}
    dm_pair = dm_blob.get("ensemble_vs_arima") or {}
    if dm_pair:
        dm_significant = str(bool(dm_pair.get("significant_5pct")))
        dm_pvalue = _fmt_pct(dm_pair.get("p_value"))

    context = f"""
Current ticker: {ticker}
Best model for {ticker}: {best_model} (MAPE: {best_mape}%)
Directional accuracy: {dir_acc}%
Walk-forward mean MAPE: {wf_mape}% ± {wf_std}%
Ensemble vs ARIMA: significant={dm_significant}, p={dm_pvalue}
Calibration at 90%: {actual_90}% actual coverage (target 90%)
"""
    return context.strip()


def _gemini_chat(user_message: str, summary: dict | None = None,
                 outlook: dict | None = None,
                 goal_result: dict | None = None,
                 portfolio: dict | None = None,
                 history: list | None = None,
                 current_ticker: str | None = None) -> str:
    """Send a single message to Google Gemini with full dashboard context."""
    print("KEY LOADED:", os.getenv("GOOGLE_API_KEY", "NOT FOUND")[:20])
    print("GEMINI CALLED")
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        print("GEMINI FAILED: GOOGLE_API_KEY not found")
        return ""  # Signal caller to use fallback

    try:
        import google.generativeai as genai
    except ImportError as e:
        print(f"GEMINI FAILED: {e}")
        return ""  # Signal caller to use fallback

    genai.configure(api_key=api_key)
    selected_ticker = (current_ticker or _infer_ticker_from_text(user_message) or "AAPL")
    context = _build_gemini_context(summary, selected_ticker)

    messages = [
        {"role": "user", "parts": [f"{SYSTEM_PROMPT}\n\n{context}"]},
        {"role": "model", "parts": ["Understood."]},
    ]
    if history:
        # Last 3 exchanges => last 6 chat messages.
        for h in history[-6:]:
            role = "user" if h.get("role") == "user" else "model"
            messages.append({"role": role, "parts": [h.get("content", "")]})
    messages.append({"role": "user", "parts": [user_message]})

    try:
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        resp = model.generate_content(messages)
        if resp.text:
            return resp.text
        return ""
    except Exception as e:
        print(f"GEMINI FAILED: {e}")
        return ""  # Fall back to smart_reply on any error


# ═══════════════════════════════════════════════════════════════════════
#  Free rule-based fallback (used when no API key)
# ═══════════════════════════════════════════════════════════════════════

def _find_ticker(text: str) -> str | None:
    """Extract a ticker from user text, or return None."""
    text_upper = text.upper()
    for t in config.TICKERS:
        if t in text_upper:
            return t
    name_map = {"APPLE": "AAPL", "MICROSOFT": "MSFT", "NVIDIA": "NVDA",
                "TESLA": "TSLA", "S&P": "SPY", "SP500": "SPY", "S&P500": "SPY",
                "INDEX": "SPY", "MARKET": "SPY"}
    for name, ticker in name_map.items():
        if name in text_upper:
            return ticker
    return None


def _best_model(ticker: str, summary: dict | None):
    """Return (model_name, metrics_returns, metrics_prices) with lowest SMAPE."""
    if not summary:
        return None, None, None
    per_ticker = summary.get("per_ticker", {})
    info = per_ticker.get(ticker)
    if not info:
        return None, None, None
    models = info.get("models", {})
    best_name = None
    best_smape = float("inf")
    for name, mdata in models.items():
        if name == "ensemble":
            continue
        s = mdata.get("metrics_returns", {}).get("SMAPE", 1e9)
        if s < best_smape:
            best_smape = s
            best_name = name
    if best_name:
        m = models[best_name]
        return best_name, m.get("metrics_returns", {}), m.get("metrics_prices", {})
    return None, None, None


def _smart_reply(user_msg: str, summary: dict | None = None,
                 data: dict | None = None,
                 last_topic: str | None = None) -> tuple[str, str | None]:
    """
    Keyword-based reply engine. Returns (reply_text, new_topic).
    Adapted from v1's smart_reply to work with v3's summary.json format.
    """
    msg = user_msg.lower()
    ticker = _find_ticker(user_msg) or "AAPL"
    ticker_blob = _extract_ticker_blob(summary, ticker)
    models = ticker_blob.get("models", {}) if isinstance(ticker_blob, dict) else {}

    if "best model" in msg or "which model" in msg:
        bm, _, m_price = _best_model(ticker, summary)
        if bm and m_price:
            return (
                f"Based on the model results, the best model for {ticker} is {bm.upper()} with MAPE {m_price.get('MAPE_%', 0):.2f}%. "
                f"This is the lowest price error among the evaluated models for that ticker."
            ), "model"
        return f"Based on the model results, best-model data for {ticker} is not available yet.", "model"

    if "calibration" in msg or "uncertainty" in msg or "bands" in msg:
        for model_name in ("lstm", "gru", "transformer"):
            report = (models.get(model_name, {}) or {}).get("calibration_report") or {}
            if report:
                return (
                    f"Based on the model results for {ticker} {model_name.upper()}, 90% interval coverage is {_fmt_pct(report.get('actual_90pct'))}% "
                    f"against the 90% target, with 80%={_fmt_pct(report.get('actual_80pct'))}% and 50%={_fmt_pct(report.get('actual_50pct'))}%."
                ), "calibration"
        return f"Based on the model results, calibration data for {ticker} is not available yet.", "calibration"

    if "walk-forward" in msg:
        wf_summary = _load_walk_forward_summary() or {}
        wf_ticker = (wf_summary.get("tickers") or {}).get(ticker, {})
        if wf_ticker:
            best_model = min(
                wf_ticker.keys(),
                key=lambda m: ((wf_ticker.get(m, {}).get("mean_metrics") or {}).get("MAPE_%", float("inf"))),
            )
            best_blob = wf_ticker.get(best_model, {})
            return (
                f"Based on the model results, walk-forward mean MAPE for {ticker} is "
                f"{_fmt_float((best_blob.get('mean_metrics') or {}).get('MAPE_%'))}% ± "
                f"{_fmt_float((best_blob.get('std_metrics') or {}).get('MAPE_%'))}% on the best model ({best_model.upper()})."
            ), "walk-forward"
        return f"Based on the model results, walk-forward data for {ticker} is not available yet.", "walk-forward"

    if "limitation" in msg:
        limitations = []
        try:
            from src.critical_analysis import generate_analysis
            analysis = generate_analysis(summary or {}, ticker)
            limitations = analysis.get("limitations") or []
        except Exception:
            limitations = []
        if limitations:
            top3 = "; ".join(limitations[:3])
            return f"Based on the model results, the top limitations are: {top3}.", "limitations"
        return (
            "Three key limitations are: results are evaluated on a single 2023 test period, the universe is limited to five large-cap US equities, "
            "and directional accuracy stays near 50% which suggests weak exploitable signal under EMH."
        ), "limitations"

    if "arima" in msg:
        dm_blob = _load_dm_tests(ticker) or {}
        pair = dm_blob.get("ensemble_vs_arima") or {}
        if pair:
            return (
                f"Based on the model results for {ticker}, ensemble vs ARIMA has p-value {_fmt_float(pair.get('p_value'), 4)} "
                f"and significant_5pct={pair.get('significant_5pct', 'data not available')}, with better model={pair.get('better_model', 'data not available')}."
            ), "dm"
        return f"Based on the model results, DM test data for {ticker} vs ARIMA is not available yet.", "dm"

    if "direction" in msg or "50%" in msg or "emh" in msg:
        bm, m_ret, _ = _best_model(ticker, summary)
        if bm and m_ret:
            return (
                f"Based on the model results, directional accuracy for {ticker} on {bm.upper()} is {_fmt_float(m_ret.get('Dir_Acc'))}%, which is close to chance. "
                "That pattern is consistent with EMH: short-horizon direction is hard to predict consistently."
            ), "direction"
        return f"Based on the model results, directional-accuracy data for {ticker} is not available yet.", "direction"

    return (
        "Based on the model results, I can answer best model, calibration reliability, walk-forward consistency, ARIMA significance, project limitations, and directional-accuracy questions."
    ), None


# ═══════════════════════════════════════════════════════════════════════
#  Public API — tries Gemini first, falls back to smart_reply
# ═══════════════════════════════════════════════════════════════════════

def chat(user_message: str, summary: dict | None = None,
         outlook: dict | None = None,
         goal_result: dict | None = None,
         portfolio: dict | None = None,
         history: list | None = None,
         stock_data: dict | None = None,
         last_topic: str | None = None,
         current_ticker: str | None = None) -> tuple[str, str | None]:
    """
    Main entry point. Returns (reply_text, new_topic).

    Tries Google Gemini API first (if key is set). On failure or missing key,
    falls back to the free keyword-based engine.
    """
    # Try Gemini API first
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if summary is None:
        summary = _load_default_summary()
    if api_key:
        reply = _gemini_chat(user_message, summary, outlook, goal_result,
                             portfolio, history, current_ticker=current_ticker)
        if reply:
            return reply, last_topic  # Gemini doesn't track topic

    # Fallback to smart_reply
    return _smart_reply(user_message, summary=summary, data=stock_data,
                        last_topic=last_topic)
