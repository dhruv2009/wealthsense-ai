# WealthSense AI

Deep-learning stock forecasting + goal-based financial planning, presented through a Streamlit dashboard.

**Live Demo:** [https://wealthsenseai.streamlit.app](https://wealthsenseai.streamlit.app/)

## What's in the box

| Layer | Module | What it does |
|---|---|---|
| Data | `src/data_pipeline.py` | Yahoo Finance OHLCV + macro (VIX, Yield_Spread, DXY) -> log-return target. Train-only-fit StandardScaler (no leakage). |
| Models | `src/models.py` | LSTM, GRU, Transformer with **LayerNorm + GELU** heads. Transformer uses **causal mask** + exposes **attention weights**. |
| Baselines | `src/baselines.py` | **ARIMA(5,0,0)** walk-forward + **naive persistence**. Lets us prove DL adds value. |
| Uncertainty | `src/uncertainty.py` | **MC Dropout** (100 forward passes) + 50/80/90% **calibration report**. |
| Ensemble | `src/ensemble.py` | **Rolling inverse-RMSE** weighted ensemble across LSTM/GRU/Transformer. |
| Training | `src/train.py` | Full pipeline. **SMAPE** on returns + **price-domain MAPE** (dual metrics). |
| Planning | `src/monte_carlo.py` | 5,000-path MC engine, model-outlook tilt, Sharpe + **Sortino** + max DD, transaction-cost-aware strategy backtest. |
| Ablation | `src/ablation.py` | Five feature configs (full / no_macro / no_rsi / no_volume / price_only). |
| Chat | `src/chat.py` | **Google Gemini** (free tier) with structured context injection + **free rule-based fallback** when no key. |
| Attention | `src/attention.py` | Day-importance bar chart + final-layer attention heatmap. |
| UI | `app.py` | 4-tab layout: Portfolio Analytics, Forecast View, Goal Planner, AI Chat. |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env       # add your GOOGLE_API_KEY (optional — free at aistudio.google.com/apikey)
```

## Run

```bash
# 1. Pull data + train all models (LSTM, GRU, Transformer, ensemble + ARIMA + naive)
python src/train.py

# 2. Generate examiner-facing consolidated report
python run_results_summary.py

# 3. Launch the dashboard
streamlit run app.py
```

## Research methodology

- Walk-forward validation with an expanding window across 6-month folds
- Diebold-Mariano testing (Harvey-Leybourne-Newbold correction) for pairwise significance
- Feature ablation (core and extended configurations) for sensitivity analysis
- Uncertainty calibration checks at 50%/80%/90% interval targets

## Reproducibility

- Seed pinned in `config.SEED = 42`.
- Frozen window: `START_DATE=2015-01-01`, `END_DATE=2023-12-31`.
- Train: 2015-2021 / Validation: 2022 / Test: 2023.
- Sliding window: 30 days. Target: log-return on day 31.

## Metrics design

The DL models predict **log-returns** (stationary). Reporting MAPE on log-returns is mathematically broken -- values divided by approximately 0 explode. We report:

- **On returns:** MAE, RMSE, **SMAPE** (symmetric, finite), Directional Accuracy
- **On reconstructed prices:** MAE (USD), RMSE (USD), MAPE (%) -- these are the human-readable headline metrics

ARIMA and naive baselines are evaluated under the same scheme on the same test split.

## Project structure

```
wealthsenseAI/
├── config.py
├── app.py                       # Streamlit dashboard (4-tab structure)
├── requirements.txt
├── .env.example
├── data/                        # cached CSVs (auto-populated)
├── artifacts/
│   ├── models/                  # *.pt + scalers
│   └── results/                 # summary.json + *_preds.npz + ablation_*.json
└── src/
    ├── data_pipeline.py
    ├── models.py
    ├── baselines.py             # ARIMA + naive
    ├── uncertainty.py           # MC Dropout + calibration
    ├── ensemble.py              # rolling inverse-RMSE
    ├── train.py                 # master pipeline
    ├── monte_carlo.py           # goal engine + portfolio analytics
    ├── ablation.py              # feature ablation
    ├── attention.py             # attention extraction
    └── chat.py                  # Google Gemini + free fallback
```

## Documentation

- [agent.md](agent.md) -- Developer/codebase guide
- [data.md](data.md) -- Dataset documentation
- [overview.md](overview.md) -- Non-technical project overview
