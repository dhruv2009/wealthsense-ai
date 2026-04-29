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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ═══════════════════════════════════════════════════════════════════════
#  Google Gemini API chat (used when GOOGLE_API_KEY is set)
# ═══════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are WealthSense AI, a calm, plain-English financial coach.

You always answer based on the JSON context provided below — never invent stock
data, model accuracy figures, or goal results. If the context doesn't contain
something, say so honestly.

When users ask forecast-performance questions, prioritize the `current_ticker_context`
numbers (best model, directional accuracy, and walk-forward mean MAPE) in your answer.

Style:
  - Plain English; avoid jargon unless the user uses it.
  - Concise (4-8 sentences), with bullet points where helpful.
  - Always end advisory answers with: "*This is a simulation, not financial advice.*"
"""


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


def build_context(summary: dict | None, outlook: dict | None,
                  goal_result: dict | None, portfolio: dict | None,
                  current_ticker: str | None = None) -> str:
    """Compact JSON context — keeps token count low."""
    ctx: dict[str, object] = {}
    if summary:
        per_ticker = {}
        for t, info in (summary.get("per_ticker") or {}).items():
            models = info.get("models", {})
            best = None
            best_smape = float("inf")
            for name, mdata in models.items():
                if name == "ensemble":
                    continue
                s = mdata.get("metrics_returns", {}).get("SMAPE", 1e9)
                if s < best_smape:
                    best_smape = s
                    best = name
            if best:
                per_ticker[t] = {
                    "best_model": best,
                    "metrics_returns": models[best].get("metrics_returns"),
                    "metrics_prices": models[best].get("metrics_prices"),
                    "ensemble_metrics": models.get("ensemble", {}).get("metrics_returns"),
                    "calibration_90": models[best].get("calibration", {}).get("actual_90pct"),
                }
            base = info.get("baselines") or {}
            if base:
                per_ticker[t]["baselines"] = {
                    n: b.get("metrics_returns") for n, b in base.items()
                }
        ctx["model_results"] = per_ticker

        if current_ticker and current_ticker in (summary.get("per_ticker") or {}):
            info = (summary.get("per_ticker") or {}).get(current_ticker, {})
            models = info.get("models") or {}
            best_model = None
            best_mape = float("inf")
            best_dir_acc = None
            for model_name, mdata in models.items():
                mape = ((mdata.get("metrics_prices") or {}).get("MAPE_%"))
                if mape is None:
                    continue
                if float(mape) < best_mape:
                    best_mape = float(mape)
                    best_model = model_name
                    best_dir_acc = ((mdata.get("metrics_returns") or {}).get("Dir_Acc"))

            wf_mean_mape = None
            wf_summary = _load_walk_forward_summary()
            if wf_summary:
                wf_blob = (((wf_summary.get("tickers") or {}).get(current_ticker) or {}).get(best_model or "", {}))
                wf_mean_mape = ((wf_blob.get("mean_metrics") or {}).get("MAPE_%"))

            ctx["current_ticker_context"] = {
                "ticker": current_ticker,
                "best_model": best_model,
                "directional_accuracy_pct": best_dir_acc,
                "walk_forward_mean_mape_pct": wf_mean_mape,
            }
    if outlook:
        ctx["portfolio_outlook"] = {
            "label": outlook.get("outlook_label"),
            "tilt_pct": outlook.get("adjustment_pct"),
            "confidence": outlook.get("confidence_score"),
        }
    if goal_result:
        ctx["latest_goal_plan"] = {
            "success_rate": goal_result.get("success_rate"),
            "median_final": goal_result.get("median_final"),
            "recommended_extra_per_month": goal_result.get("recommended_extra"),
        }
    if portfolio:
        ctx["portfolio_allocation"] = portfolio

    return json.dumps(ctx, indent=2, default=str)


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
    selected_ticker = current_ticker or _infer_ticker_from_text(user_message)
    ctx_json = build_context(summary, outlook, goal_result, portfolio, selected_ticker)

    # Build conversation history for Gemini
    gemini_history = []
    if history:
        for h in history[-6:]:  # keep last 3 exchanges
            role = "user" if h["role"] == "user" else "model"
            gemini_history.append({"role": role, "parts": [h["content"]]})

    try:
        model = genai.GenerativeModel(
            model_name=config.GEMINI_MODEL,
            system_instruction=SYSTEM_PROMPT,
        )
        chat = model.start_chat(history=gemini_history)
        prompt = (
            f"Current dashboard context (JSON):\n```json\n{ctx_json}\n```\n\n"
            f"User question: {user_message}"
        )
        resp = chat.send_message(prompt)
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
                 last_topic: str | None = None) -> tuple[str, str | None]:
    """
    Keyword-based reply engine. Returns (reply_text, new_topic).
    Adapted from v1's smart_reply to work with v3's summary.json format.
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
            outlook = get_model_outlook(config.RESULTS_DIR, config.TICKERS, equal_w)
            adj_ret = ps["ann_return"] + outlook["return_adjustment"]
            sim_ret = adj_ret
            sim_vol = ps["ann_volatility"]
        else:
            ps = {"ann_return_pct": 8.0, "ann_volatility_pct": 18.0}
            outlook = {"outlook_label": "Neutral", "adjustment_pct": 0.0}
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
              f"- Model outlook: {outlook['outlook_label']} ({outlook['adjustment_pct']:+.1f}% adjustment)\n"
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

    # ── Specific stock question ──
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
                # Baselines
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
        "- **\"What's my portfolio risk?\"** — risk metrics guide\n\n"
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
    if api_key:
        reply = _gemini_chat(user_message, summary, outlook, goal_result,
                             portfolio, history, current_ticker=current_ticker)
        if reply:
            return reply, last_topic  # Gemini doesn't track topic

    # Fallback to smart_reply
    return _smart_reply(user_message, summary=summary, data=stock_data,
                        last_topic=last_topic)
