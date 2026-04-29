"""
WealthSense AI — Streamlit Dashboard (Merged Final).

Four tabs (v1 structure preserved):
  1. Portfolio Analytics — allocation, returns, Sharpe/Sortino, drawdown.
  2. Forecast View      — DL models + ARIMA/naive baselines + ensemble +
                          confidence bands + attention heatmap.
  3. Goal Planner       — Monte Carlo with model outlook tilt.
  4. AI Chat            — Google Gemini (if key set) + free fallback engine.

Backend: log-return target, modern model heads, MC Dropout uncertainty,
dynamic ensemble, ARIMA baseline, feature ablation.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src.data_pipeline import load_all_with_macro, build_feature_frame, prepare_ticker_data
from src.monte_carlo import (
    simulate_goal, portfolio_metrics, trading_strategy_returns,
    compute_portfolio_stats, get_model_outlook,
)
from src.chat import chat as merged_chat
from src.critical_analysis import generate_analysis


# ── Page config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WealthSense AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("WealthSense AI")
st.sidebar.markdown("*Personalized Financial Advisor*")
st.sidebar.caption(f"Tickers: {', '.join(config.TICKERS)} | Window: 2015-2023")

page = st.sidebar.radio(
    "Navigate",
    ["Portfolio Analytics", "Forecast View", "Walk-Forward Analysis", "Goal Planner", "AI Chat"],
)


# ── Cached loaders ──────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading market data ...")
def load_market_data():
    raw, macro = load_all_with_macro()
    enriched = {t: build_feature_frame(t, raw[t], macro) for t in config.TICKERS if t in raw}
    return enriched


@st.cache_data(show_spinner="Loading model results ...")
def load_summary():
    p = os.path.join(config.RESULTS_DIR, "summary.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def load_predictions(name: str, ticker: str):
    p = os.path.join(config.RESULTS_DIR, f"{name}_{ticker}_preds.npz")
    if not os.path.exists(p):
        return None
    return np.load(p, allow_pickle=True)


def load_ablation(ticker: str = "AAPL", model: str = "gru"):
    p = os.path.join(config.RESULTS_DIR, f"ablation_{ticker}_{model}.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def load_extended_ablation(ticker: str = "AAPL", model: str = "gru"):
    p = os.path.join(config.RESULTS_DIR, f"ablation_extended_{ticker}_{model}.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def load_walk_forward(ticker: str, model: str):
    p = os.path.join(config.RESULTS_DIR, f"wf_{model}_{ticker}.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def load_walk_forward_summary():
    p = os.path.join(config.RESULTS_DIR, "walk_forward_summary.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def load_dm_tests(ticker: str):
    p = os.path.join(config.RESULTS_DIR, f"dm_tests_{ticker}.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


# ════════════════════════════════════════════════════════════════════════
#  TAB 1 — Portfolio Analytics
# ════════════════════════════════════════════════════════════════════════
if page == "Portfolio Analytics":
    st.title("Portfolio Analytics Dashboard")

    data = load_market_data()
    if not data:
        st.error("No data found. Run `python src/data_pipeline.py` first.")
        st.stop()

    st.subheader("Define Your Portfolio")
    cols = st.columns(len(config.TICKERS))
    weights = {}
    for i, t in enumerate(config.TICKERS):
        with cols[i]:
            weights[t] = st.number_input(f"{t} %", 0, 100, 20, key=f"w_{t}")
    total_w = sum(weights.values())
    if total_w != 100:
        st.warning(f"Weights sum to {total_w}% -- please adjust to 100%.")

    returns_df = pd.DataFrame()
    for t in config.TICKERS:
        if t in data:
            returns_df[t] = data[t]["Close"].pct_change()
    returns_df = returns_df.dropna()
    w_arr = np.array([weights[t] / 100 for t in config.TICKERS])
    port_ret = returns_df[config.TICKERS].values @ w_arr

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Asset Allocation")
        fig = px.pie(names=config.TICKERS,
                     values=[weights[t] for t in config.TICKERS])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Cumulative Portfolio Return")
        cum = np.cumprod(1 + port_ret)
        fig = go.Figure(go.Scatter(x=returns_df.index, y=cum, name="Portfolio"))
        fig.update_layout(yaxis_title="Growth of 1 USD", xaxis_title="Date")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Risk Metrics")
    m = portfolio_metrics(port_ret)
    cols = st.columns(6)
    cols[0].metric("Cumulative", f"{m['cumulative_return']}%")
    cols[1].metric("Annualized", f"{m['annualized_return']}%")
    cols[2].metric("Volatility", f"{m['annualized_volatility']}%")
    cols[3].metric("Sharpe", m["sharpe_ratio"])
    cols[4].metric("Sortino", m["sortino_ratio"])
    cols[5].metric("Max DD", f"{m['max_drawdown']}%")

    st.subheader("Individual Stock Prices")
    sel = st.multiselect("Select stocks", config.TICKERS, default=config.TICKERS[:3])
    fig = go.Figure()
    for t in sel:
        if t in data:
            fig.add_trace(go.Scatter(x=data[t]["Date"], y=data[t]["Close"], name=t))
    fig.update_layout(yaxis_title="Close Price (USD)", xaxis_title="Date")
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════
#  TAB 2 — Forecast View
# ════════════════════════════════════════════════════════════════════════
elif page == "Forecast View":
    st.title("Deep Learning Forecast View")
    summary = load_summary()
    if summary is None:
        st.error("No trained models found. Run `python src/train.py` first.")
        st.stop()

    ticker = st.selectbox("Select Stock", config.TICKERS)
    st.session_state["current_ticker_selection"] = ticker
    info = summary["per_ticker"].get(ticker)
    if not info:
        st.warning(f"No results for {ticker}.")
        st.stop()

    # ── Comparison table: DL + baselines + ensemble ──
    st.subheader("Model Comparison (2023 Test Set)")
    rows = []
    for name, mdata in info["models"].items():
        r = mdata.get("metrics_returns") or {}
        p = mdata.get("metrics_prices") or {}
        rows.append({
            "Model": name.upper(),
            "MAE (return)": round(r.get("MAE", float("nan")), 5),
            "RMSE (return)": round(r.get("RMSE", float("nan")), 5),
            "SMAPE (%)": round(r.get("SMAPE", float("nan")), 2),
            "Dir. Acc (%)": round(r.get("Dir_Acc", float("nan")), 2),
            "MAE (USD)": round(p.get("MAE_$", float("nan")), 2),
            "RMSE (USD)": round(p.get("RMSE_$", float("nan")), 2),
            "MAPE (% of price)": round(p.get("MAPE_%", float("nan")), 2),
        })
    for name, bdata in (info.get("baselines") or {}).items():
        r = bdata.get("metrics_returns") or {}
        p = bdata.get("metrics_prices") or {}
        rows.append({
            "Model": f"{name.upper()} (baseline)",
            "MAE (return)": round(r.get("MAE", float("nan")), 5),
            "RMSE (return)": round(r.get("RMSE", float("nan")), 5),
            "SMAPE (%)": round(r.get("SMAPE", float("nan")), 2),
            "Dir. Acc (%)": round(r.get("Dir_Acc", float("nan")), 2),
            "MAE (USD)": round(p.get("MAE_$", float("nan")), 2),
            "RMSE (USD)": round(p.get("RMSE_$", float("nan")), 2),
            "MAPE (% of price)": round(p.get("MAPE_%", float("nan")), 2),
        })
    st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)
    st.caption("DL models predict log-returns. Price-domain metrics (USD) are computed on the reconstructed price path.")

    # ── Ensemble weights ──
    weights = info.get("ensemble_weights")
    if weights:
        st.markdown(f"**Ensemble weights (rolling inverse-RMSE):** "
                    + " | ".join([f"{n.upper()} {w:.2f}" for n, w in weights.items()]))

    # ── Forecast charts: actual vs predicted (with MC Dropout band) ──
    st.subheader("Actual vs Predicted (2023 Test Set)")
    tabs = st.tabs(["LSTM", "GRU", "Transformer", "Ensemble", "All Models"])
    for idx, name in enumerate(["lstm", "gru", "transformer", "ensemble"]):
        d = load_predictions(name, ticker)
        with tabs[idx]:
            if d is None:
                st.info(f"No predictions for {name.upper()} on {ticker}.")
                continue
            actual_p = d["actual_price"]
            pred_p = d["pred_price"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=actual_p, name="Actual", line=dict(color="black")))
            fig.add_trace(go.Scatter(y=pred_p, name="Predicted",
                                     line=dict(dash="dash", color="royalblue")))
            # Optional MC dropout band
            if "y_pred_lower" in d.files and "y_pred_upper" in d.files:
                last_p = float(actual_p[0]) / float(np.exp(d["y_true"][0]))
                lo_p = [last_p]
                hi_p = [last_p]
                for lo, hi in zip(d["y_pred_lower"], d["y_pred_upper"]):
                    lo_p.append(lo_p[-1] * float(np.exp(lo)))
                    hi_p.append(hi_p[-1] * float(np.exp(hi)))
                fig.add_trace(go.Scatter(y=hi_p[1:], name="Upper 95%",
                                         line=dict(width=0), showlegend=False))
                fig.add_trace(go.Scatter(y=lo_p[1:], name="Lower 5%",
                                         line=dict(width=0),
                                         fill="tonexty",
                                         fillcolor="rgba(65,105,225,0.18)",
                                         showlegend=True))
            fig.update_layout(title=f"{name.upper()} -- {ticker}",
                              xaxis_title="Test Day", yaxis_title="Price (USD)")
            st.plotly_chart(fig, use_container_width=True)

            if "train_losses" in d.files:
                fl = go.Figure()
                fl.add_trace(go.Scatter(y=d["train_losses"], name="Train"))
                fl.add_trace(go.Scatter(y=d["val_losses"], name="Val"))
                fl.update_layout(title="Training Curves",
                                 xaxis_title="Epoch", yaxis_title="MSE Loss")
                st.plotly_chart(fl, use_container_width=True)

    with tabs[4]:
        fig = go.Figure()
        first = True
        for name, color in [("lstm", "blue"), ("gru", "green"),
                            ("transformer", "red"), ("ensemble", "purple"),
                            ("arima", "orange"), ("naive", "gray")]:
            d = load_predictions(name, ticker)
            if d is None:
                continue
            if first:
                fig.add_trace(go.Scatter(y=d["actual_price"], name="Actual",
                                         line=dict(color="black")))
                first = False
            fig.add_trace(go.Scatter(y=d["pred_price"], name=name.upper(),
                                     line=dict(dash="dash", color=color)))
        fig.update_layout(title=f"All Models -- {ticker}",
                          xaxis_title="Test Day", yaxis_title="Price (USD)")
        st.plotly_chart(fig, use_container_width=True)

    # ── GRU deep-dive charts ──
    st.subheader("GRU Forecast Diagnostics")
    gru_pred = load_predictions("gru", ticker)
    if gru_pred is None:
        st.info("No GRU prediction file found for this ticker.")
    else:
        if "test_dates" in gru_pred.files:
            x_dates = pd.to_datetime(gru_pred["test_dates"])
        else:
            x_dates = np.arange(len(gru_pred["actual_price"]))
        actual_p = np.asarray(gru_pred["actual_price"], dtype=float)
        pred_p = np.asarray(gru_pred["pred_price"], dtype=float)

        # 1) Prediction vs Actual overlay with MC dropout band
        overlay = go.Figure()
        overlay.add_trace(go.Scatter(x=x_dates, y=actual_p, mode="lines",
                                     name="Actual close price", line=dict(color="blue")))
        overlay.add_trace(go.Scatter(x=x_dates, y=pred_p, mode="lines",
                                     name="Predicted price", line=dict(color="orange")))
        if "y_pred_lower" in gru_pred.files and "y_pred_upper" in gru_pred.files and "y_true" in gru_pred.files:
            last_p = float(actual_p[0]) / float(np.exp(gru_pred["y_true"][0]))
            lo_p = [last_p]
            hi_p = [last_p]
            for lo, hi in zip(gru_pred["y_pred_lower"], gru_pred["y_pred_upper"]):
                lo_p.append(lo_p[-1] * float(np.exp(lo)))
                hi_p.append(hi_p[-1] * float(np.exp(hi)))
            overlay.add_trace(go.Scatter(x=x_dates, y=hi_p[1:], name="Upper bound",
                                         line=dict(width=0), showlegend=False))
            overlay.add_trace(go.Scatter(x=x_dates, y=lo_p[1:], name="MC Dropout band",
                                         line=dict(width=0), fill="tonexty",
                                         fillcolor="rgba(255,165,0,0.2)"))
        overlay.update_layout(
            title=f"GRU forecast vs actual — {ticker} test period",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
        )
        st.plotly_chart(overlay, use_container_width=True)

        # 2) Residual plot with direction-correct coloring
        if "y_pred" in gru_pred.files and "y_true" in gru_pred.files:
            residual_usd = pred_p - actual_p
            direction_correct = np.sign(gru_pred["y_pred"]) == np.sign(gru_pred["y_true"])
            colors = np.where(direction_correct, "green", "red")
            residual_fig = go.Figure()
            residual_fig.add_trace(go.Scatter(
                x=x_dates,
                y=residual_usd,
                mode="markers",
                marker=dict(color=colors, size=7),
                name="Residual (Pred - Actual)",
            ))
            residual_fig.add_hline(y=0, line_dash="dash", line_color="black")
            residual_fig.update_layout(
                title="Residuals (USD) with direction correctness",
                xaxis_title="Date",
                yaxis_title="Predicted - Actual (USD)",
            )
            st.plotly_chart(residual_fig, use_container_width=True)

    # ── Calibration table ──
    st.subheader("Uncertainty Calibration (DL models)")
    cal_rows = []
    for name in ["lstm", "gru", "transformer"]:
        cal = (info["models"].get(name) or {}).get("calibration")
        if cal:
            cal_rows.append({
                "Model": name.upper(),
                "Target 50% / Actual": f"{cal.get('actual_50pct', 0):.2f}",
                "Target 80% / Actual": f"{cal.get('actual_80pct', 0):.2f}",
                "Target 90% / Actual": f"{cal.get('actual_90pct', 0):.2f}",
                "Error @ 90%": f"{cal.get('error_90pct', 0):.3f}",
            })
    if cal_rows:
        st.dataframe(pd.DataFrame(cal_rows).set_index("Model"), use_container_width=True)
        st.caption("Coverage closer to the target = better-calibrated intervals from MC Dropout.")

    # ── Attention heatmap (Transformer) ──
    with st.expander("Attention Heatmap (Transformer interpretability)"):
        try:
            from src.attention import extract_attention
            data_all = load_market_data()
            if ticker in data_all:
                raw, macro = load_all_with_macro()
                prepared = prepare_ticker_data(ticker, raw, macro)
                X_sample = prepared["X_test"][:32]
                att = extract_attention(ticker, X_sample)
                imp = att["day_importance"]
                fig = go.Figure(go.Bar(x=list(range(1, len(imp) + 1)), y=imp))
                fig.update_layout(title=f"What the Transformer attends to (avg over {len(X_sample)} test windows)",
                                  xaxis_title="Day in 30-day input window (1=oldest, 30=most recent)",
                                  yaxis_title="Attention weight (last layer, last position)")
                st.plotly_chart(fig, use_container_width=True)
                fig2 = px.imshow(att["layers"][-1],
                                 labels=dict(x="Key day", y="Query day", color="Attn"),
                                 title="Final-layer attention matrix")
                st.plotly_chart(fig2, use_container_width=True)
                # 3) Real heatmap view requested: x=day, y=attention head (proxy slots)
                head_like = np.asarray(att["layers"][-1], dtype=float)
                fig3 = px.imshow(
                    head_like,
                    labels=dict(x="Day in sequence (1-30)", y="Attention head", color="Attention weight"),
                    title="Transformer attention heatmap (x: day, y: attention head)",
                )
                st.plotly_chart(fig3, use_container_width=True)
        except FileNotFoundError:
            st.info("Train models first to enable attention visualization.")
        except Exception as e:
            st.info(f"Attention extraction unavailable: {e}")

    # ── Trading strategy ──
    st.subheader("Trading Strategy vs Buy-and-Hold (with 5 bps transaction cost)")
    model_choice = st.selectbox("Strategy model", ["ensemble", "lstm", "gru", "transformer"])
    d = load_predictions(model_choice, ticker)
    if d is not None and len(d["actual_price"]) > 1:
        bh, strat = trading_strategy_returns(d["actual_price"], d["pred_price"])
        bh_cum = np.cumprod(1 + bh)
        st_cum = np.cumprod(1 + strat)
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=bh_cum, name="Buy & Hold"))
        fig.add_trace(go.Scatter(y=st_cum, name=f"{model_choice.upper()} Strategy"))
        fig.update_layout(yaxis_title="Growth of 1 USD", xaxis_title="Test Day")
        st.plotly_chart(fig, use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Buy & Hold**")
            st.json(portfolio_metrics(bh))
        with c2:
            st.markdown(f"**{model_choice.upper()} Strategy**")
            st.json(portfolio_metrics(strat))

    # ── Ablation table ──
    abl = load_ablation(ticker, "gru")
    if abl is None:
        abl = load_ablation("AAPL", "gru")
    if abl:
        with st.expander("Feature Ablation Study (GRU on AAPL)"):
            rows = []
            for cfg, res in abl.items():
                rows.append({
                    "Config": cfg,
                    "# Features": res["n_features"],
                    "MAPE (% of price)": round(res["metrics_prices"]["MAPE_%"], 2),
                    "MAE (USD)": round(res["metrics_prices"]["MAE_$"], 2),
                    "Dir Acc (%)": round(res["metrics_returns"]["Dir_Acc"], 2),
                })
            st.dataframe(pd.DataFrame(rows).set_index("Config"), use_container_width=True)
            ext_abl = load_extended_ablation("AAPL", "gru")
            if ext_abl:
                st.markdown("**Extended feature ablation (Step 9 features)**")
                ext_rows = []
                for cfg, res in ext_abl.items():
                    notes = res.get("notes", {})
                    ext_rows.append({
                        "Config": cfg,
                        "# Features": res["n_features"],
                        "MAPE (% of price)": round(res["metrics_prices"]["MAPE_%"], 2),
                        "MAE (USD)": round(res["metrics_prices"]["MAE_$"], 2),
                        "Dir Acc (%)": round(res["metrics_returns"]["Dir_Acc"], 2),
                        "Notes": ", ".join([f"{k}={v}" for k, v in notes.items()]) if notes else "",
                    })
                st.dataframe(pd.DataFrame(ext_rows).set_index("Config"), use_container_width=True)
            st.caption("Run `python src/ablation.py` to regenerate.")

    with st.expander("Critical Analysis"):
        analysis = generate_analysis(summary, ticker)
        best_model = analysis.get("best_model")
        if best_model:
            st.success(f"Best model on {ticker}: {best_model.upper()}")
        verdict = (analysis.get("ensemble_vs_best_individual") or {}).get("verdict", "")
        if "improves by" in verdict:
            st.success(verdict)
        elif verdict:
            st.warning(verdict)

        st.markdown("**Directional accuracy**")
        for m_name, dacc in (analysis.get("directional_accuracy") or {}).items():
            if dacc < 52:
                st.error(f"{m_name.upper()}: {dacc:.2f}% (near-random)")
            else:
                st.success(f"{m_name.upper()}: {dacc:.2f}%")

        st.markdown("**Calibration verdict (90% target)**")
        for m_name, txt in (analysis.get("calibration_verdict") or {}).items():
            if "miscalibrated" in txt:
                st.error(f"{m_name.upper()}: {txt}")
            elif "well_calibrated" in txt:
                st.success(f"{m_name.upper()}: {txt}")
            else:
                st.warning(f"{m_name.upper()}: {txt}")

        hv = analysis.get("high_volatility_ticker") or {}
        if hv.get("tsla_vs_spy_ratio") is not None:
            ratio = hv["tsla_vs_spy_ratio"]
            if ratio > 1:
                st.warning(hv.get("note", "TSLA exhibits higher volatility-linked error than SPY."))
            else:
                st.success(hv.get("note", "TSLA performance is not worse than SPY for this model."))

        st.markdown("**Limitations**")
        for item in analysis.get("limitations", []):
            if "near-random" in item or "miscalibrated" in item:
                st.error(item)
            else:
                st.warning(item)


# ════════════════════════════════════════════════════════════════════════
#  TAB 3 — Walk-Forward Analysis
# ════════════════════════════════════════════════════════════════════════
elif page == "Walk-Forward Analysis":
    st.title("Walk-Forward Analysis")
    wf_summary = load_walk_forward_summary()
    if wf_summary is None:
        st.info("Run python src/train.py first to generate walk-forward results.")
        st.stop()

    ticker = st.selectbox("Select Stock", config.TICKERS, key="wf_ticker")
    st.session_state["current_ticker_selection"] = ticker
    model_names = ["lstm", "gru", "transformer", "ensemble", "arima", "naive"]

    wf_results = {}
    for m in model_names:
        d = load_walk_forward(ticker, m)
        if d is not None:
            wf_results[m] = d

    if not wf_results:
        st.info("No walk-forward results found. Run `python src/walk_forward.py` first.")
        st.stop()

    table_rows = []
    line_rows = []
    for m in model_names:
        data = wf_results.get(m)
        if data is None:
            continue
        mean_mape = (data.get("mean_metrics") or {}).get("MAPE_%")
        std_mape = (data.get("std_metrics") or {}).get("MAPE_%")
        if mean_mape is None or std_mape is None:
            continue
        table_rows.append({
            "Model": m.upper(),
            "Mean MAPE_% ± std": f"{mean_mape:.2f} ± {std_mape:.2f}",
        })
        for fold in data.get("per_fold", []):
            fold_mape = (fold.get("metrics_prices") or {}).get("MAPE_%")
            if fold_mape is not None:
                line_rows.append({
                    "Fold": int(fold["fold_id"]),
                    "Model": m.upper(),
                    "MAPE_%": float(fold_mape),
                })

    st.subheader("Model consistency table")
    if table_rows:
        st.dataframe(pd.DataFrame(table_rows).set_index("Model"), use_container_width=True)
    else:
        st.info("No valid MAPE_% metrics available in walk-forward files yet.")

    st.subheader("MAPE_% by fold")
    if line_rows:
        line_df = pd.DataFrame(line_rows).sort_values(["Fold", "Model"])
        fig = go.Figure()
        for m in sorted(line_df["Model"].unique()):
            model_df = line_df[line_df["Model"] == m].sort_values("Fold")
            model_key = m.lower()
            std_val = float((wf_results.get(model_key, {}).get("std_metrics") or {}).get("MAPE_%", 0.0))
            y = model_df["MAPE_%"].values.astype(float)
            x = model_df["Fold"].values.astype(int)
            upper = y + std_val
            lower = y - std_val
            fig.add_trace(go.Scatter(
                x=x, y=upper, mode="lines", line=dict(width=0), showlegend=False, name=f"{m} +1 std"
            ))
            fig.add_trace(go.Scatter(
                x=x, y=lower, mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor="rgba(100,149,237,0.12)",
                showlegend=False, name=f"{m} -1 std"
            ))
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="lines+markers", name=m
            ))
        fig.update_layout(xaxis_title="Fold number", yaxis_title="MAPE_%")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Per-fold series unavailable.")

    first_model = next(iter(wf_results.values()))
    n_folds = first_model.get("n_folds", 0)
    st.caption(
        f"Walk-forward validation uses an expanding window. Metrics are averaged across {n_folds} folds. "
        "Lower variance = more consistent model."
    )

    dm = load_dm_tests(ticker)
    if dm:
        st.subheader("Diebold-Mariano Test Results")
        dm_rows = []
        for pair, res in dm.items():
            better = res.get("better_model", "indistinguishable")
            m1, _, m2 = pair.partition("_vs_")
            if better == "model1":
                better_label = m1.upper()
            elif better == "model2":
                better_label = m2.upper()
            else:
                better_label = "indistinguishable"
            dm_rows.append(
                {
                    "Model Pair": pair,
                    "DM Statistic": round(float(res.get("dm_statistic", 0.0)), 4),
                    "p-value": round(float(res.get("p_value", 1.0)), 6),
                    "Significant @5%": "✓" if res.get("significant_5pct") else "✗",
                    "Better Model": better_label,
                }
            )
        st.dataframe(pd.DataFrame(dm_rows).set_index("Model Pair"), use_container_width=True)
        st.caption(
            "Diebold-Mariano test (Harvey-Leybourne-Newbold correction). "
            "H0: equal predictive accuracy. p<0.05 = one model is statistically better."
        )


# ════════════════════════════════════════════════════════════════════════
#  TAB 4 — Goal Planner
# ════════════════════════════════════════════════════════════════════════
elif page == "Goal Planner":
    st.title("Goal-Based Planning Engine")
    st.markdown("Define a goal. Return and volatility are computed from your "
                "portfolio's history, with a model-based outlook tilt.")

    data = load_market_data()
    st.subheader("1. Your Portfolio Allocation")
    cols = st.columns(len(config.TICKERS))
    gw = {}
    for i, t in enumerate(config.TICKERS):
        with cols[i]:
            gw[t] = st.number_input(f"{t} %", 0, 100, 20, key=f"gw_{t}")
    if sum(gw.values()) != 100:
        st.warning(f"Weights sum to {sum(gw.values())}% -- please adjust to 100%.")

    w_norm = {t: gw[t] / 100 for t in config.TICKERS}
    ps = compute_portfolio_stats(data, w_norm, config.TICKERS)
    outlook = get_model_outlook(config.RESULTS_DIR, config.TICKERS, w_norm)
    adj_ret = ps["ann_return"] + outlook["return_adjustment"]

    st.subheader("2. Data-Driven Parameters")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Hist. Return", f"{ps['ann_return_pct']}%/yr")
    p2.metric("Hist. Vol", f"{ps['ann_volatility_pct']}%/yr")
    p3.metric("Model Outlook", outlook["outlook_label"],
              delta=f"{outlook['adjustment_pct']:+.1f}% tilt")
    p4.metric("Adjusted Return", f"{round(adj_ret * 100, 2)}%/yr")

    with st.expander("Per-stock breakdown"):
        rows = []
        for t in config.TICKERS:
            row = {"Stock": t, "Weight": f"{gw[t]}%"}
            if t in ps["per_stock"]:
                row.update({
                    "Ann. Return": f"{ps['per_stock'][t]['ann_return']}%",
                    "Ann. Vol": f"{ps['per_stock'][t]['ann_volatility']}%",
                })
            if t in outlook["details"]:
                md = outlook["details"][t]
                row.update({
                    "Model": md["model"].upper(),
                    "Signal": md["signal"],
                    "Pred Up %": f"{md['pred_up_pct']}%",
                    "Recent Acc.": f"{md['recent_accuracy']}%",
                })
            rows.append(row)
        st.dataframe(pd.DataFrame(rows).set_index("Stock"), use_container_width=True)

    st.markdown("---")
    st.subheader("3. Your Financial Goal")
    c1, c2 = st.columns(2)
    with c1:
        goal_name = st.text_input("Goal Name", "Home Down Payment")
        current = st.number_input("Current Savings (USD)", 0, 10_000_000, 10_000, step=1000)
        monthly = st.number_input("Monthly Contribution (USD)", 0, 100_000, 1_200, step=100)
    with c2:
        target = st.number_input("Target Amount (USD)", 1_000, 100_000_000, 60_000, step=1000)
        years = st.slider("Time Horizon (years)", 1, 30, 3)
        use_dd = st.checkbox("Use data-driven return & volatility", value=True)

    if use_dd:
        sim_ret, sim_vol = adj_ret, ps["ann_volatility"]
        st.caption(f"Using {round(sim_ret*100, 2)}% return / {round(sim_vol*100, 2)}% vol "
                   f"(historical + model tilt)")
    else:
        sim_ret = st.slider("Expected Annual Return (%)", 1.0, 20.0, 8.0, 0.5) / 100
        sim_vol = st.slider("Expected Annual Volatility (%)", 5.0, 40.0, 18.0, 0.5) / 100

    if st.button("Run Simulation", type="primary"):
        with st.spinner("Running 5,000 Monte Carlo paths ..."):
            res = simulate_goal(current, monthly, target, years,
                                annual_return=sim_ret, annual_volatility=sim_vol)
        st.session_state["last_goal"] = res

        st.subheader(f"Results -- {goal_name}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Success Rate", f"{res['success_rate']}%")
        c2.metric("Median Final", f"{res['median_final']:,.0f} USD")
        c3.metric("5th %ile", f"{res['percentile_5']:,.0f} USD")
        c4.metric("95th %ile", f"{res['percentile_95']:,.0f} USD")

        if res["recommended_extra"] > 0:
            st.info(f"To reach **90% success**, increase monthly savings by "
                    f"**{res['recommended_extra']:,.0f} USD/month**.")
        else:
            st.success("On track for 90%+ success!")

        st.subheader("Simulated Portfolio Paths")
        paths = res["paths"]
        m_axis = np.arange(paths.shape[1])
        fig = go.Figure()
        idxs = np.random.choice(paths.shape[0], min(200, paths.shape[0]), replace=False)
        for i in idxs:
            fig.add_trace(go.Scatter(x=m_axis, y=paths[i],
                                     line=dict(width=0.3, color="rgba(100,149,237,0.15)"),
                                     showlegend=False))
        fig.add_trace(go.Scatter(x=m_axis, y=np.percentile(paths, 5, axis=0),
                                 name="5th %ile", line=dict(dash="dot", color="red")))
        fig.add_trace(go.Scatter(x=m_axis, y=np.median(paths, axis=0),
                                 name="Median", line=dict(color="blue", width=2)))
        fig.add_trace(go.Scatter(x=m_axis, y=np.percentile(paths, 95, axis=0),
                                 name="95th %ile", line=dict(dash="dot", color="green")))
        fig.add_hline(y=target, line_dash="dash", line_color="black",
                      annotation_text=f"Target {target:,.0f} USD")
        fig.update_layout(xaxis_title="Month", yaxis_title="Portfolio Value (USD)")
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════
#  TAB 5 — AI Chat (Claude API + free fallback)
# ════════════════════════════════════════════════════════════════════════
elif page == "AI Chat":
    st.title("AI Financial Assistant")

    has_api_key = bool(os.getenv("GOOGLE_API_KEY", "").strip())
    if has_api_key:
        st.markdown("Powered by **Google Gemini** with structured context from "
                    "your portfolio, model results, and most recent goal plan.")
    else:
        st.markdown("Powered by **WealthSense AI engine** -- ask about your portfolio, "
                    "forecasts, or financial goals. "
                    "Set `GOOGLE_API_KEY` in `.env` for Gemini-powered answers.")

    summary = load_summary()
    data = load_market_data()
    portfolio_alloc = {t: 20 for t in config.TICKERS}
    weights_norm = {t: 0.2 for t in config.TICKERS}
    outlook = get_model_outlook(config.RESULTS_DIR, config.TICKERS, weights_norm) if summary else None
    last_goal = st.session_state.get("last_goal")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "last_topic" not in st.session_state:
        st.session_state.last_topic = None

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about your portfolio, models, or goals ..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking ..."):
                reply, new_topic = merged_chat(
                    prompt,
                    summary=summary,
                    outlook=outlook,
                    goal_result=last_goal,
                    portfolio=portfolio_alloc,
                    history=st.session_state.chat_history[:-1],
                    stock_data=data,
                    last_topic=st.session_state.last_topic,
                    current_ticker=st.session_state.get("current_ticker_selection"),
                )
            st.markdown(reply)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        if new_topic is not None:
            st.session_state.last_topic = new_topic
