# WealthSense AI — Agent Guide

> **Purpose**: This file helps teammates (and Claude) understand the codebase,
> architecture, file layout, and how to extend the project.

---

## 1. Project Overview

WealthSense AI is a deep-learning financial advisor platform that:

1. **Downloads** historical stock data (2015-2023) from Yahoo Finance, including macro indicators (VIX, Treasury yields, US Dollar Index).
2. **Engineers 12 features** (9 price-based + 3 macro) from raw OHLCV data.
3. **Trains three DL models** (LSTM, GRU, Transformer) with modern heads (LayerNorm + GELU) to predict next-day log returns.
4. **Builds a dynamic ensemble** using rolling inverse-RMSE weights across all three models.
5. **Quantifies uncertainty** via MC Dropout (100 forward passes) with calibration diagnostics.
6. **Benchmarks against classical baselines** (ARIMA walk-forward + naive persistence) to prove DL adds value.
7. **Runs feature ablation** (5 configs) to answer "which features actually matter?"
8. **Extracts attention heatmaps** from the Transformer for interpretability.
9. **Simulates goal-based planning** with Monte Carlo (5,000 paths), transaction-cost-aware trading strategy, Sharpe + Sortino ratios.
10. **Serves everything** through a Streamlit dashboard with AI chat (Google Gemini when key set, free rule-based engine otherwise).

---

## 2. Directory Structure

```
wealthsenseAI/
├── app.py                  # Streamlit dashboard (entry point: streamlit run app.py)
├── config.py               # All hyperparameters, paths, constants
├── requirements.txt        # Python dependencies
├── .env.example            # Template for GOOGLE_API_KEY (optional)
├── agent.md                # THIS FILE — codebase guide for developers
├── data.md                 # Dataset documentation (features, splits, scaling)
├── overview.md             # Non-technical project overview for teammates
├── README.md               # Quick-start README
│
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py    # Yahoo Finance + macro download, features, StandardScaler, DataLoaders
│   ├── models.py           # LSTM, GRU, Transformer (LayerNorm+GELU heads, causal mask)
│   ├── train.py            # Full pipeline: train + MC Dropout + ensemble + baselines
│   ├── monte_carlo.py      # Monte Carlo goal engine + portfolio analytics + Sortino
│   ├── ensemble.py         # Rolling inverse-RMSE weighted ensemble
│   ├── uncertainty.py      # MC Dropout + calibration report
│   ├── baselines.py        # ARIMA(5,0,0) walk-forward + naive persistence
│   ├── ablation.py         # Feature ablation study (5 configs)
│   ├── attention.py        # Transformer attention heatmap extraction
│   └── chat.py             # Google Gemini chat + free rule-based fallback
│
├── data/                   # Downloaded CSV files (one per ticker + macro.csv)
│   ├── AAPL.csv
│   ├── MSFT.csv
│   ├── NVDA.csv
│   ├── TSLA.csv
│   ├── SPY.csv
│   └── macro.csv           # VIX, Treasury yields, DXY
│
└── artifacts/
    ├── models/             # Trained model weights (.pt) and scalers (.pkl)
    │   ├── lstm_AAPL.pt
    │   ├── gru_AAPL.pt
    │   ├── transformer_AAPL.pt
    │   ├── scaler_AAPL.pkl
    │   └── ...
    └── results/            # Predictions (.npz), summary.json, ablation JSON
        ├── lstm_AAPL_preds.npz
        ├── gru_AAPL_preds.npz
        ├── transformer_AAPL_preds.npz
        ├── ensemble_AAPL_preds.npz
        ├── arima_AAPL_preds.npz
        ├── naive_AAPL_preds.npz
        ├── summary.json
        ├── ablation_AAPL_gru.json
        └── ...
```

---

## 3. Key Files Explained

### `config.py`
Central configuration. **Every constant lives here.**

| Variable | Value | Description |
|---|---|---|
| `TICKERS` | `["AAPL","MSFT","NVDA","TSLA","SPY"]` | Stocks to download and model |
| `MACRO_TICKERS` | `{"^VIX": "VIX", "^TNX": "Yield_10Y", ...}` | Macro indicators |
| `FEATURE_COLS` | 12 features (9 price + 3 macro) | Input feature set |
| `TARGET_COL` | `"Log_Return"` | Stationary target (not raw Close) |
| `SEQUENCE_LENGTH` | `30` | Sliding window size |
| `HIDDEN_SIZE` | `128` | LSTM/GRU hidden dim |
| `TRANSFORMER_DIM` | `256` | Transformer embedding dim |
| `NUM_LAYERS` | `2` | Layers in all models |
| `EPOCHS` | `100` | Max training epochs |
| `PATIENCE` | `15` | Early stopping patience |
| `BATCH_SIZE` | `64` | Training batch size |
| `LEARNING_RATE` | `3e-4` | AdamW learning rate |
| `WEIGHT_DECAY` | `1e-5` | AdamW weight decay |
| `MC_DROPOUT_SAMPLES` | `100` | MC Dropout forward passes |
| `MC_SIMULATIONS` | `5000` | Monte Carlo paths |
| `GEMINI_MODEL` | `"gemini-2.0-flash"` | Google Gemini model for AI chat (free tier) |

### `src/data_pipeline.py`
- `load_all_with_macro()` — downloads all tickers + macro indicators, caches as CSV.
- `build_feature_frame(ticker, raw, macro)` — adds price features + merges macro.
- `prepare_ticker_data(ticker, raw, macro)` — full pipeline: features -> train-only StandardScaler -> chronological split -> sliding windows -> DataLoaders.
- `_download_macro()` — fetches VIX, Treasury yields, DXY; computes VIX_Zscore and Yield_Spread.

**Key improvement over v1**: StandardScaler fit on TRAIN data only (no data leakage).

**Input shape**: `(batch, 30, 12)` -> **Target**: next-day log return (scalar).

### `src/models.py`
Three model classes with shared `_head()`:

```python
def _head(hidden, dropout):
    return nn.Sequential(
        nn.LayerNorm(hidden),
        nn.Dropout(dropout),
        nn.Linear(hidden, hidden // 2),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden // 2, 1),
    )
```

1. **`LSTMModel`** — 2-layer LSTM -> LayerNorm+GELU head
2. **`GRUModel`** — 2-layer GRU -> LayerNorm+GELU head
3. **`TransformerModel`** — Input projection -> PositionalEncoding -> Custom `_AttnEncoderLayer` (with causal mask + attention weight extraction) -> LayerNorm+GELU head

The Transformer supports `forward(x, return_attn=True)` for interpretability heatmaps.

### `src/train.py`
- `train_one()` — AdamW optimizer, MSE loss, ReduceLROnPlateau, gradient clipping, early stopping.
- `run_ticker()` — full per-ticker pipeline: train 3 DL models -> MC Dropout -> ensemble -> baselines.
- `run_full_pipeline()` — runs all tickers, saves `summary.json`.

**Dual-domain metrics**:
- On log-returns: MAE, RMSE, **SMAPE** (finite when values near 0), directional accuracy
- On reconstructed prices: MAE (USD), RMSE (USD), MAPE (%)

### `src/ensemble.py`
- `rolling_inverse_rmse_weights()` — weights each model by 1/RMSE on last 21 validation points.
- `ensemble_predict()` — linear combination of test predictions.

### `src/uncertainty.py`
- `mc_dropout_predict()` — 100 forward passes with dropout active. Returns mean, 5th/95th percentiles.
- `calibration_report()` — checks if 50%/80%/90% intervals contain true values at advertised rates.
- `regime_adjust()` — widens intervals during high-VIX periods.

### `src/baselines.py`
- `naive_persistence_forecast()` — predict today's return = yesterday's return.
- `arima_forecast()` — walk-forward ARIMA(5,0,0) refitting on each test step.

### `src/ablation.py`
Five feature configs: full (12), no_macro (9), no_rsi (11), no_volume (11), price_only (2).
Retrains GRU for each and saves metrics to JSON.

### `src/attention.py`
- `extract_attention()` — loads saved Transformer, runs test samples, returns layer-averaged attention matrices and day-importance weights.

### `src/chat.py`
- `chat()` — main entry point. Tries Google Gemini first (if `GOOGLE_API_KEY` set), falls back to rule-based engine.
- `_gemini_chat()` — sends user message + compressed dashboard context JSON to Gemini.
- `_smart_reply()` — keyword-matching engine with topic tracking and follow-up detection.

### `src/monte_carlo.py`
- `simulate_goal()` — Monte Carlo savings simulation with binary search for recommended extra contribution.
- `portfolio_metrics()` — Sharpe, **Sortino** (downside-only vol), max drawdown.
- `trading_strategy_returns()` — long/cash strategy with **5 bps transaction cost**.
- `compute_portfolio_stats()` — per-stock and portfolio annualized return/volatility.
- `get_model_outlook()` — derives bullish/bearish tilt from model predictions (+-2% return adjustment).

### `app.py` (Streamlit Dashboard)
Four pages:

1. **Portfolio Analytics** — allocation weights, pie chart, cumulative return, Sharpe/Sortino/max drawdown, stock price charts.
2. **Forecast View** — model comparison table (DL + baselines + ensemble), actual vs predicted charts with MC Dropout confidence bands, training curves, calibration table, attention heatmap, trading strategy vs buy-and-hold, feature ablation table.
3. **Goal Planner** — data-driven return/volatility from portfolio, model outlook tilt, Monte Carlo simulation with path visualization.
4. **AI Chat** — Google Gemini (if key set) or free rule-based engine. History-aware with topic tracking.

---

## 4. How to Run

### First-time setup
```bash
pip install -r requirements.txt
cp .env.example .env       # add GOOGLE_API_KEY (optional — free at aistudio.google.com/apikey)
```

### Step 1: Train models
```bash
python src/train.py
```
Downloads data, trains 3 DL models x 5 tickers, runs MC Dropout, builds ensembles, runs ARIMA + naive baselines. Saves everything to `artifacts/`.

### Step 2: (Optional) Run ablation
```bash
python src/ablation.py
```
Retrains GRU under 5 feature configs for AAPL.

### Step 3: Launch dashboard
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`.

The AI Chat tab works without an API key (free rule-based engine). Set `GOOGLE_API_KEY` in `.env` for Gemini-powered answers (free at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)).

---

## 5. How to Extend

### Add a new ticker
1. Add to `config.TICKERS`.
2. Re-run `python src/train.py`.

### Add a new feature
1. Add computation in `_add_price_features()` in `data_pipeline.py`.
2. Add column name to `config.PRICE_FEATURES` or `config.MACRO_FEATURES`.
3. Retrain (input_size adjusts automatically).

### Add a new model
1. Create `nn.Module` class in `src/models.py` with the shared `_head()`.
2. Add to `MODEL_REGISTRY`.
3. Add the name to the loop in `train.py:run_ticker()`.

---

## 6. Data Flow

```
Yahoo Finance (yfinance)
    |
    v
data/*.csv (OHLCV) + data/macro.csv (VIX, yields, DXY)
    |
    v  build_feature_frame()
DataFrame with 12 features + Date column
    |
    v  StandardScaler (fit on TRAIN only)
Scaled numpy array
    |
    v  _build_sequences(seq_len=30)
X: (samples, 30, 12)   y: (samples,)  <- log-return target
    |
    v  StockDataset -> DataLoader
PyTorch DataLoaders (train/val/test)
    |
    v  train_one() x 3 models
Best weights -> artifacts/models/{model}_{ticker}.pt
    |
    v  mc_dropout_predict()
Mean + 5/95% intervals -> calibration_report()
    |
    v  rolling_inverse_rmse_weights() + ensemble_predict()
Ensemble predictions
    |
    v  run_baselines()
ARIMA + naive predictions on same test split
    |
    v  returns_to_prices()
Reconstructed price paths for human-readable metrics
    |
    v  All saved to artifacts/results/
*.npz files + summary.json + ablation JSON
    |
    v  app.py (Streamlit)
Interactive dashboard with 4 tabs
```

---

## 7. Model Architecture Details

### LSTM
```
Input(batch, 30, 12)
  -> LSTM(input=12, hidden=128, layers=2, dropout=0.2)
  -> take last time step -> (batch, 128)
  -> LayerNorm(128) -> Dropout -> Linear(128, 64) -> GELU -> Dropout -> Linear(64, 1)
  -> squeeze -> (batch,)
```

### GRU
```
Input(batch, 30, 12)
  -> GRU(input=12, hidden=128, layers=2, dropout=0.2)
  -> take last time step -> (batch, 128)
  -> LayerNorm(128) -> Dropout -> Linear(128, 64) -> GELU -> Dropout -> Linear(64, 1)
  -> squeeze -> (batch,)
```

### Transformer
```
Input(batch, 30, 12)
  -> Linear(12, 256)              # project to d_model
  -> PositionalEncoding(256)      # sinusoidal
  -> 2x _AttnEncoderLayer(
       MultiheadAttention(256, 4 heads, causal mask)
       + FeedForward(256 -> 1024 -> 256, GELU)
       + LayerNorm + residual connections
     )
  -> take last time step -> (batch, 256)
  -> LayerNorm(256) -> Dropout -> Linear(256, 128) -> GELU -> Dropout -> Linear(128, 1)
  -> squeeze -> (batch,)
```

The causal mask prevents the model from attending to future time steps.

---

## 8. Evaluation Metrics

### On log-returns (what the model predicts)
| Metric | Description |
|---|---|
| MAE | Mean absolute error on log-return predictions |
| RMSE | Root mean squared error on log-return predictions |
| SMAPE | Symmetric MAPE — finite even when log-returns are near 0 |
| Dir. Accuracy | % of days where predicted sign matches actual sign |

### On reconstructed prices (human-readable)
| Metric | Description |
|---|---|
| MAE (USD) | Average error in dollars |
| RMSE (USD) | Error penalizing large mistakes |
| MAPE (%) | Percentage error on price |

---

## 9. Important Notes

- **Log-return target**: Models predict `log(P_t / P_{t-1})` not raw price. This is stationary and well-behaved.
- **StandardScaler on train only**: Prevents data leakage. Scaler is saved and reused for val/test.
- **Causal mask**: Transformer cannot attend to future positions within the 30-day window.
- **MC Dropout**: 100 forward passes with dropout active to get prediction distributions.
- **Transaction costs**: Trading strategy deducts 5 bps (0.05%) on each position change.
- **Sortino ratio**: Uses only downside volatility, more relevant for investors.
- **Dollar sign rendering**: Streamlit renders `$` as LaTeX. All amounts use "USD" suffix.
- **Seeds pinned**: `config.SEED = 42` for reproducibility.
- **Frozen window**: 2015-2023 per proposal. No live data refresh.

---

## 10. Troubleshooting

| Issue | Fix |
|---|---|
| `yfinance` download fails | Check internet. Data is cached in `data/` after first download. |
| CUDA out of memory | Reduce `BATCH_SIZE` in `config.py`. |
| Dashboard shows "no models found" | Run `python src/train.py` first. |
| ARIMA is slow | Normal — walk-forward refits on each test step. |
| AI Chat gives generic answer | Use keywords: stock name, "compare", "goal", "invest", "risk" |
| Import errors | Run from project root directory. |

---

## 11. AI Chat Details

The chat system has two modes:

**1. Google Gemini (when `GOOGLE_API_KEY` is set — free tier, 15 req/min)**
- Sends user message + compressed dashboard context JSON to Gemini 2.0 Flash.
- Context includes: best model per ticker, ensemble metrics, calibration, baselines, portfolio outlook, latest goal result.
- Gemini answers based on actual data, never invents numbers.
- Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

**2. Free rule-based engine (when no API key)**
- Keyword matching with topic tracking (`last_topic` in session state).
- Follow-up detection: "why?", "explain", "what does" etc.
- Answers from actual model results (summary.json) and stock data.
- Runs Monte Carlo simulations inline for goal questions.

Both modes use "USD" instead of "$" to avoid Streamlit LaTeX rendering issues.
