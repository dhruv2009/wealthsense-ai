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


def _build_gemini_context(summary: dict | None, ticker: str,
                          outlook: dict | None = None,
                          goal_result: dict | None = None,
                          portfolio: dict | None = None) -> str:
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

    context = (
        f"Current ticker: {ticker}\n"
        f"Best model for {ticker}: {best_model} (MAPE: {best_mape}%)\n"
        f"Directional accuracy: {dir_acc}%\n"
        f"Walk-forward mean MAPE: {wf_mape}% +/- {wf_std}%\n"
        f"Ensemble vs ARIMA: significant={dm_significant}, p={dm_pvalue}\n"
        f"Calibration at 90%: {actual_90}% actual coverage (target 90%)"
    )

    if outlook:
        context += (
            f"\nModel outlook: {outlook.get('outlook_label', 'N/A')}"
            f"\nOutlook confidence: {outlook.get('confidence_score', 'N/A')}"
            f"\nReturn adjustment: {outlook.get('adjustment_pct', 0):+.2f}%"
        )

    if goal_result:
        context += (
            f"\nLatest goal simulation:"
            f"\n  Success rate: {goal_result.get('success_rate', 'N/A')}%"
            f"\n  Median final: {goal_result.get('median_final', 'N/A')} USD"
            f"\n  Recommended extra/month: {goal_result.get('recommended_extra', 'N/A')} USD"
        )

    if portfolio:
        context += f"\nPortfolio allocation: {json.dumps(portfolio)}"

    return context


def _gemini_chat(user_message: str, summary: dict | None = None,
                 outlook: dict | None = None,
                 goal_result: dict | None = None,
                 portfolio: dict | None = None,
                 history: list | None = None,
                 current_ticker: str | None = None) -> str:
    """Send a single message to Google Gemini with full dashboard context."""
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        return ""  # Signal caller to use fallback

    try:
        import google.generativeai as genai
    except ImportError:
        return ""  # Signal caller to use fallback

    genai.configure(api_key=api_key)
    selected_ticker = (current_ticker or _infer_ticker_from_text(user_message) or "AAPL")
    context = _build_gemini_context(summary, selected_ticker,
                                    outlook=outlook,
                                    goal_result=goal_result,
                                    portfolio=portfolio)

    gemini_history = []
    if history:
        for h in history[-6:]:
            role = "user" if h.get("role") == "user" else "model"
            gemini_history.append({"role": role, "parts": [h.get("content", "")]})

    try:
        model = genai.GenerativeModel(
            model_name=config.GEMINI_MODEL,
            system_instruction=SYSTEM_PROMPT,
        )
        chat_session = model.start_chat(history=gemini_history)
        prompt = (
            f"Current dashboard context (JSON):\n```json\n{context}\n```\n\n"
            f"User question: {user_message}"
        )
        resp = chat_session.send_message(prompt)
        if resp.text:
            return resp.text
        return ""
    except Exception:
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
                 last_topic: str | None = None,
                 outlook: dict | None = None,
                 goal_result: dict | None = None,
                 portfolio: dict | None = None) -> tuple[str, str | None]:
    """
    Keyword-based reply engine. Returns (reply_text, new_topic).
    Handles goal planning, model queries, stock summaries, risk analysis,
    trading strategy, walk-forward, calibration, and follow-up questions.
    """
    msg = user_msg.lower()
    ticker = _find_ticker(user_msg)

    # Follow-up detection
    is_followup = any(w in msg for w in [
        "why", "what does", "what do", "explain", "mean", "how does",
        "tell me more", "elaborate", "clarify", "more detail", "parameter",
        "what is", "how is", "why only", "why not", "can you explain",
    ])

    # ── Goal / house / retirement questions ──
    if any(w in msg for w in ["goal", "house", "home", "retire", "save", "saving",
                                "buy a", "down payment", "on track", "afford"]):
        from src.monte_carlo import simulate_goal, compute_portfolio_stats, get_model_outlook
        equal_w = {t: 0.2 for t in config.TICKERS}
        if data:
            ps = compute_portfolio_stats(data, equal_w, config.TICKERS)
            ol = get_model_outlook(config.RESULTS_DIR, config.TICKERS, equal_w)
            adj_ret = ps["ann_return"] + ol["return_adjustment"]
            sim_ret = adj_ret
            sim_vol = ps["ann_volatility"]
        else:
            ps = {"ann_return_pct": 8.0, "ann_volatility_pct": 18.0}
            ol = {"outlook_label": "Neutral", "adjustment_pct": 0.0}
            sim_ret = 0.08
            sim_vol = 0.18

        result = simulate_goal(
            current_savings=10000, monthly_contribution=1200,
            target_amount=60000, years=3,
            annual_return=sim_ret, annual_volatility=sim_vol,
        )
        extra_line = (f"To reach 90% confidence, increase monthly savings by "
                      f"**{result['recommended_extra']:,.0f} USD/month**.\n\n"
                      if result["recommended_extra"] > 0 else
                      "You're on track for 90%+ success!\n\n")
        return (
            "**Goal Analysis: Home Down Payment**\n\n"
            "Assuming 10,000 USD current savings and 1,200 USD/month contributions "
            "toward a 60,000 USD target over 3 years:\n\n"
            f"- **Success probability**: {result['success_rate']}%\n"
            f"- **Median final value**: {result['median_final']:,.0f} USD\n"
            f"- **5th percentile (worst case)**: {result['percentile_5']:,.0f} USD\n"
            f"- **95th percentile (best case)**: {result['percentile_95']:,.0f} USD\n\n"
            + extra_line
            + f"**Data-driven parameters used:**\n"
              f"- Historical portfolio return: {ps['ann_return_pct']}%/yr\n"
              f"- Historical volatility: {ps['ann_volatility_pct']}%/yr\n"
              f"- Model outlook: {ol['outlook_label']} ({ol['adjustment_pct']:+.1f}% adjustment)\n"
              f"- Adjusted return used: {round(sim_ret*100, 2)}%/yr\n\n"
              "Head to the **Goal Planner** tab to customize allocation and goal parameters.\n\n"
              "*Disclaimer: This is a simulation based on historical data, not financial advice.*"
        ), "goal"

    # ── Follow-up on goal topic ──
    if is_followup and last_topic == "goal":
        return (
            "**How the Goal Planner Uses Your Data:**\n\n"
            "**Step 1 — Historical Return & Volatility**\n"
            "We look at your portfolio allocation (which stocks, what %) and compute "
            "the actual annualized return and volatility from 2015-2023 daily price data. "
            "This replaces guesswork with real numbers from your chosen stocks.\n\n"
            "**Step 2 — Model Outlook Adjustment**\n"
            "Our trained models (LSTM/GRU/Transformer) predict whether each stock's price "
            "is likely to go up or down. We combine these signals weighted by your allocation "
            "into a Bullish/Neutral/Bearish outlook, which tilts the expected return by up to "
            "+-2%. This connects the deep learning forecasts to your goal planning.\n\n"
            "**Step 3 — Monte Carlo Simulation**\n"
            "Using the adjusted return and historical volatility, we simulate 5,000 random "
            "market scenarios. Each month: your portfolio grows (or shrinks) randomly, plus "
            "your monthly contribution is added. After all months, we count how many paths "
            "reached your target.\n\n"
            "**The recommended extra savings** is found by binary search — we keep increasing "
            "the monthly amount until 90% of the 5,000 scenarios succeed.\n\n"
            "**In the Goal Planner tab**, you can:\n"
            "- Change your stock allocation to see how it affects return/volatility\n"
            "- Toggle between data-driven and manual parameters\n"
            "- See per-stock model signals and accuracy\n"
            "- Customize savings, target, and timeline"
        ), "goal"

    # ── Best model question ──
    if any(w in msg for w in ["best model", "which model", "accurate", "recommend model"]):
        if ticker:
            bm, m_ret, m_price = _best_model(ticker, summary)
            if bm and m_ret and m_price:
                return (
                    f"**Best model for {ticker}: {bm.upper()}**\n\n"
                    f"| Metric | Value |\n|---|---|\n"
                    f"| MAE (USD) | {m_price.get('MAE_$', 0):.2f} |\n"
                    f"| RMSE (USD) | {m_price.get('RMSE_$', 0):.2f} |\n"
                    f"| MAPE | {m_price.get('MAPE_%', 0):.2f}% |\n"
                    f"| SMAPE (return) | {m_ret.get('SMAPE', 0):.2f}% |\n"
                    f"| Dir. Accuracy | {m_ret.get('Dir_Acc', 0):.1f}% |\n\n"
                    f"*Lower MAPE = better. Check the Forecast View tab for charts.*"
                ), "model"

        lines = ["**Best model per stock (by lowest SMAPE):**\n",
                 "| Stock | Best Model | MAPE (price) | MAE (USD) |", "|---|---|---|---|"]
        for t in config.TICKERS:
            bm, m_ret, m_price = _best_model(t, summary)
            if bm and m_price:
                lines.append(f"| {t} | {bm.upper()} | {m_price.get('MAPE_%', 0):.2f}% | {m_price.get('MAE_$', 0):.2f} |")
        return "\n".join(lines) + "\n\n*Check the Forecast View tab for detailed charts.*", "model"

    # ── Follow-up on model topic ──
    if is_followup and last_topic == "model":
        return (
            "**Understanding the Model Metrics:**\n\n"
            "- **MAE (Mean Absolute Error)**: Average difference between predicted and "
            "actual price in dollars. Lower = better.\n"
            "- **RMSE (Root Mean Squared Error)**: Like MAE but penalizes large errors "
            "more. A model with low RMSE makes fewer big mistakes.\n"
            "- **MAPE (Mean Absolute Percentage Error)**: Error as a percentage of the "
            "actual price. Reported on reconstructed prices (not log-returns). "
            "Below 5% is considered good.\n"
            "- **SMAPE (Symmetric MAPE)**: Used on log-return predictions because standard "
            "MAPE explodes when values are near zero. Finite and well-behaved.\n"
            "- **Directional Accuracy**: What % of the time the model correctly predicted "
            "whether the price would go up or down. Above 50% means it's better than random.\n\n"
            "**Our 3 DL models:**\n"
            "- **LSTM** — Classic recurrent network with memory gates.\n"
            "- **GRU** — Simpler variant of LSTM, fewer parameters, often trains faster.\n"
            "- **Transformer** — Uses attention mechanism instead of recurrence. "
            "State-of-the-art for many sequence tasks.\n\n"
            "**Plus classical baselines:** ARIMA(5,0,0) and naive persistence — these "
            "prove the DL models add value over traditional methods.\n\n"
            "All models use a 30-day sliding window of 12 features (including macro indicators) "
            "to predict the next day's log return."
        ), "model"

    # ── Calibration / uncertainty (INTEGRATED: uses conformal data) ──
    if "calibration" in msg or "uncertainty" in msg or "bands" in msg:
        t = ticker or "AAPL"
        ticker_blob = _extract_ticker_blob(summary, t)
        models = ticker_blob.get("models", {}) if isinstance(ticker_blob, dict) else {}
        for model_name in ("lstm", "gru", "transformer"):
            report = (models.get(model_name, {}) or {}).get("calibration_report") or {}
            if report:
                return (
                    f"Based on the model results for {t} {model_name.upper()}, 90% interval coverage is {_fmt_pct(report.get('actual_90pct'))}% "
                    f"against the 90% target, with 80%={_fmt_pct(report.get('actual_80pct'))}% and 50%={_fmt_pct(report.get('actual_50pct'))}%."
                ), "calibration"
        return f"Based on the model results, calibration data for {t} is not available yet.", "calibration"

    # ── Walk-forward (INTEGRATED) ──
    if "walk-forward" in msg or "walk forward" in msg or "consistency" in msg:
        t = ticker or "AAPL"
        wf_summary = _load_walk_forward_summary() or {}
        wf_ticker = (wf_summary.get("tickers") or {}).get(t, {})
        if wf_ticker:
            best_wf_model = min(
                wf_ticker.keys(),
                key=lambda m: ((wf_ticker.get(m, {}).get("mean_metrics") or {}).get("MAPE_%", float("inf"))),
            )
            best_blob = wf_ticker.get(best_wf_model, {})
            return (
                f"Based on the model results, walk-forward mean MAPE for {t} is "
                f"{_fmt_float((best_blob.get('mean_metrics') or {}).get('MAPE_%'))}% ± "
                f"{_fmt_float((best_blob.get('std_metrics') or {}).get('MAPE_%'))}% on the best model ({best_wf_model.upper()})."
            ), "walk-forward"
        return f"Based on the model results, walk-forward data for {t} is not available yet.", "walk-forward"

    # ── Limitations (INTEGRATED: uses critical_analysis) ──
    if "limitation" in msg:
        t = ticker or "AAPL"
        limitations = []
        try:
            from src.critical_analysis import generate_analysis
            analysis = generate_analysis(summary or {}, t)
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

    # ── ARIMA / Diebold-Mariano (INTEGRATED) ──
    if "arima" in msg:
        t = ticker or "AAPL"
        dm_blob = _load_dm_tests(t) or {}
        pair = dm_blob.get("ensemble_vs_arima") or {}
        if pair:
            return (
                f"Based on the model results for {t}, ensemble vs ARIMA has p-value {_fmt_float(pair.get('p_value'), 4)} "
                f"and significant_5pct={pair.get('significant_5pct', 'data not available')}, with better model={pair.get('better_model', 'data not available')}."
            ), "dm"
        return f"Based on the model results, DM test data for {t} vs ARIMA is not available yet.", "dm"

    # ── Directional accuracy / EMH (INTEGRATED) ──
    if "direction" in msg or "50%" in msg or "emh" in msg:
        t = ticker or "AAPL"
        bm, m_ret, _ = _best_model(t, summary)
        if bm and m_ret:
            return (
                f"Based on the model results, directional accuracy for {t} on {bm.upper()} is {_fmt_float(m_ret.get('Dir_Acc'))}%, which is close to chance. "
                "That pattern is consistent with EMH: short-horizon direction is hard to predict consistently."
            ), "direction"
        return f"Based on the model results, directional-accuracy data for {t} is not available yet.", "direction"

    # ── RSI ablation ──
    if "rsi" in msg or "ablation" in msg:
        t = ticker or "AAPL"
        ticker_blob = _extract_ticker_blob(summary, t)
        models_blob = ticker_blob.get("models", {}) if isinstance(ticker_blob, dict) else {}
        bm, _, m_price = _best_model(t, summary)
        return (
            "**RSI Ablation Finding:**\n\n"
            "When we remove RSI (Relative Strength Index) from the feature set and **freeze** "
            "the pretrained model weights, the MAPE barely changes. This initially suggests RSI "
            "is unimportant.\n\n"
            "**However**, when we **retrain from scratch** without RSI, the error increases "
            "dramatically (MAPE jumps from ~62% to ~305% on AAPL GRU). This means RSI is "
            "deeply embedded in the learned representations — the model relies on it during "
            "training even though removing it post-training doesn't immediately hurt.\n\n"
            "**Takeaway**: Frozen-model ablation underestimates feature importance. "
            "RSI is genuinely useful for training, but its contribution is entangled with "
            "other features in the learned weights."
        ), "model"

    # ── Specific stock question (bare ticker mention) ──
    if ticker:
        bm, m_ret, m_price = _best_model(ticker, summary)
        if bm and data and ticker in data:
            df = data[ticker]
            latest_price = df["Close"].iloc[-1]
            ytd_return = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
            return (
                f"**{ticker} Summary**\n\n"
                f"- **Latest close**: {latest_price:,.2f} USD\n"
                f"- **Dataset return (2015-2023)**: {ytd_return:,.1f}%\n"
                f"- **Best forecasting model**: {bm.upper()} (MAPE: {m_price.get('MAPE_%', 0):.2f}%)\n"
                f"- **MAE**: {m_price.get('MAE_$', 0):.2f} USD | **RMSE**: {m_price.get('RMSE_$', 0):.2f} USD\n"
                f"- **Directional Accuracy**: {m_ret.get('Dir_Acc', 0):.1f}%\n\n"
                f"All 3 models (LSTM, GRU, Transformer) plus an ensemble are trained on {ticker}. "
                f"Visit the **Forecast View** tab to see actual vs predicted charts.\n\n"
                f"*Disclaimer: Past performance does not guarantee future results.*"
            ), "stock"
        elif bm and m_price:
            return (
                f"**{ticker} — Best model: {bm.upper()}**\n\n"
                f"| Metric | Value |\n|---|---|\n"
                f"| MAE (USD) | {m_price.get('MAE_$', 0):.2f} |\n"
                f"| MAPE | {m_price.get('MAPE_%', 0):.2f}% |\n"
                f"| Dir. Accuracy | {m_ret.get('Dir_Acc', 0):.1f}% |\n\n"
                f"Visit the **Forecast View** tab for detailed charts."
            ), "stock"

    # ── Compare models ──
    if any(w in msg for w in ["compare", "comparison", "vs", "versus", "difference"]):
        if summary:
            per_ticker = summary.get("per_ticker", {})
            lines = ["**Model Comparison (2023 Test Set — all stocks):**\n"]
            for t in config.TICKERS:
                info = per_ticker.get(t)
                if not info:
                    continue
                lines.append(f"\n**{t}**\n")
                lines.append("| Model | SMAPE | Dir. Acc | MAPE (price) | MAE (USD) |")
                lines.append("|---|---|---|---|---|")
                for m_name in ["lstm", "gru", "transformer", "ensemble"]:
                    mdata = info.get("models", {}).get(m_name)
                    if mdata:
                        mr = mdata.get("metrics_returns", {})
                        mp = mdata.get("metrics_prices", {})
                        lines.append(
                            f"| {m_name.upper()} | {mr.get('SMAPE', 0):.2f}% | "
                            f"{mr.get('Dir_Acc', 0):.1f}% | "
                            f"{mp.get('MAPE_%', 0):.2f}% | {mp.get('MAE_$', 0):.2f} |"
                        )
                for bname, bdata in (info.get("baselines") or {}).items():
                    mr = bdata.get("metrics_returns", {})
                    mp = bdata.get("metrics_prices", {})
                    lines.append(
                        f"| {bname.upper()} (baseline) | {mr.get('SMAPE', 0):.2f}% | "
                        f"{mr.get('Dir_Acc', 0):.1f}% | "
                        f"{mp.get('MAPE_%', 0):.2f}% | {mp.get('MAE_$', 0):.2f} |"
                    )
            return "\n".join(lines), "model"
        return "No model results available. Run `python src/train.py` first.", "model"

    # ── Risk / Sharpe / portfolio ──
    if any(w in msg for w in ["risk", "sharpe", "sortino", "volatil", "drawdown", "portfolio"]):
        return (
            "**Portfolio Risk Analysis**\n\n"
            "Head to the **Portfolio Analytics** tab where you can:\n"
            "- Set custom allocation weights for AAPL, MSFT, NVDA, TSLA, SPY\n"
            "- See cumulative returns, Sharpe ratio, **Sortino ratio**, max drawdown\n"
            "- View annualized return and volatility\n\n"
            "The metrics are computed from historical daily returns (2015-2023) "
            "weighted by your chosen allocation.\n\n"
            "**Sharpe** measures risk-adjusted return (total vol). **Sortino** is "
            "similar but only penalizes downside volatility — more relevant for "
            "investors who care about losses, not upside swings."
        ), "risk"

    # ── Follow-up on risk topic ──
    if is_followup and last_topic == "risk":
        return (
            "**Risk Metrics Explained:**\n\n"
            "- **Sharpe Ratio**: Risk-adjusted return = (return - risk-free rate) / volatility. "
            "Above 1.0 is good, above 2.0 is excellent.\n"
            "- **Sortino Ratio**: Same idea but uses only downside volatility. Better for "
            "asymmetric return distributions.\n"
            "- **Max Drawdown**: The largest peak-to-trough drop. Shows worst-case loss "
            "if you bought at the peak and sold at the bottom.\n"
            "- **Annualized Volatility**: How much returns swing year to year. "
            "Higher volatility = more risk.\n"
            "- **Cumulative Return**: Total growth of your portfolio over the full period.\n\n"
            "All metrics use daily returns weighted by your portfolio allocation."
        ), "risk"

    # ── Invest / strategy ──
    if any(w in msg for w in ["invest", "strategy", "buy", "sell", "trade"]):
        lines = ["**Trading Strategy Results (2023 Test Set):**\n\n"
                 "Our models generate a simple signal: **go long** when the predicted "
                 "next-day return is positive, otherwise **stay in cash**. "
                 "A 5 bps (0.05%) transaction cost is subtracted on every position change.\n\n"
                 "| Stock | Best Model | MAPE (price) | Strategy |\n|---|---|---|---|"]
        for t in config.TICKERS:
            bm, m_ret, m_price = _best_model(t, summary)
            if bm and m_price:
                lines.append(f"| {t} | {bm.upper()} | {m_price.get('MAPE_%', 0):.2f}% | See Forecast View |")
        lines.append("\nVisit the **Forecast View** tab, then *Trading Strategy vs Buy-and-Hold* "
                     "section to see cumulative return charts and Sharpe/Sortino ratios.\n\n"
                     "*Disclaimer: This is not financial advice. Past performance does not equal future results.*")
        return "\n".join(lines), "strategy"

    # ── Follow-up on strategy topic ──
    if is_followup and last_topic == "strategy":
        return (
            "**How the Trading Strategy Works:**\n\n"
            "1. Each day, the model predicts tomorrow's log return.\n"
            "2. If predicted return > 0: **go long** (buy/hold the stock).\n"
            "3. If predicted return <= 0: **stay in cash** (no position).\n"
            "4. On every position change, a **5 bps transaction cost** is deducted.\n"
            "5. We compare this against **buy-and-hold** (simply holding the stock all year).\n\n"
            "This is intentionally simple — the goal is to show whether the model's "
            "directional predictions have practical value, not to build a production trading system.\n\n"
            "Check the **Forecast View** tab to see the cumulative return comparison chart."
        ), "strategy"

    # ── Generic follow-up (no specific topic) ──
    if is_followup:
        return (
            "Could you be more specific? Here are some things I can explain:\n\n"
            "- **\"What do the metrics mean?\"** — after asking about models\n"
            "- **\"What do these parameters mean?\"** — after a goal simulation\n"
            "- **\"How does the strategy work?\"** — after asking about investing\n"
            "- Or ask about a specific stock: **\"Tell me about AAPL\"**"
        ), last_topic

    # ── Help / what can you do ──
    if any(w in msg for w in ["help", "what can", "feature", "what do"]):
        return (
            "**I can help you with:**\n\n"
            "- **Stock info** — Ask about AAPL, MSFT, NVDA, TSLA, or SPY\n"
            "- **Model comparison** — \"Which model is best?\" or \"Compare models\"\n"
            "- **Goal planning** — \"Can I afford a house in 3 years?\"\n"
            "- **Investment strategy** — \"How should I invest?\"\n"
            "- **Risk analysis** — \"What's my portfolio risk?\"\n"
            "- **Walk-forward** — \"How consistent are the models?\"\n"
            "- **Calibration** — \"Are the uncertainty bands reliable?\"\n"
            "- **Follow-ups** — Ask \"why?\" or \"explain\" after any answer!\n\n"
            "Try: *\"Tell me about NVDA\"* or *\"Which model is most accurate?\"*"
        ), None

    # ── Default fallback ──
    return (
        "I'm WealthSense AI! Here's what I can answer:\n\n"
        "- **\"Tell me about AAPL\"** — stock summary and best model\n"
        "- **\"Which model is best?\"** — model comparison across stocks\n"
        "- **\"Compare models\"** — detailed LSTM vs GRU vs Transformer table\n"
        "- **\"I want to buy a house in 3 years\"** — goal planning simulation\n"
        "- **\"How should I invest?\"** — trading strategy results\n"
        "- **\"What's my portfolio risk?\"** — risk metrics guide\n"
        "- **\"Walk-forward consistency?\"** — cross-validation results\n"
        "- **\"Are my uncertainty bands reliable?\"** — calibration analysis\n\n"
        "You can also ask follow-up questions like \"why?\" or \"explain\" after any answer!\n\n"
        "Try one of the questions above!"
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
                        last_topic=last_topic, outlook=outlook,
                        goal_result=goal_result, portfolio=portfolio)
