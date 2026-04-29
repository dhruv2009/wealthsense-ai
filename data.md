# WealthSense AI — Dataset Documentation

> For a non-technical overview of the whole project, see [overview.md](overview.md).
> For the full developer/codebase guide, see [agent.md](agent.md).

---

## 1. Data Source

All data is sourced from **Yahoo Finance** via the `yfinance` Python library.
We download historical daily **OHLCV** data for a frozen period to ensure reproducibility.

| Parameter | Value |
|---|---|
| Source | Yahoo Finance (`yfinance`) |
| Period | January 1, 2015 -- December 31, 2023 |
| Frequency | Daily (trading days only) |
| Total rows per ticker | ~2,264 trading days |
| Storage format | CSV files in `data/` directory |

---

## 2. Tickers

### Stock tickers
| Ticker | Company | Sector | Role |
|---|---|---|---|
| AAPL | Apple Inc. | Technology | Core stock |
| MSFT | Microsoft Corporation | Technology | Core stock |
| NVDA | NVIDIA Corporation | Semiconductors | Core stock |
| TSLA | Tesla, Inc. | Automotive/EV | High-volatility stock |
| SPY | S&P 500 ETF | Index/Benchmark | Market benchmark |

### Macro indicators
| Yahoo Symbol | Column Name | Description |
|---|---|---|
| `^VIX` | VIX | CBOE Volatility Index (market fear gauge) |
| `^TNX` | Yield_10Y | 10-Year Treasury yield |
| `^IRX` | Yield_3M | 3-Month Treasury yield |
| `DX-Y.NYB` | DXY | US Dollar Index |

Derived macro features:
- **VIX_Zscore** = (VIX - rolling_252d_mean) / rolling_252d_std
- **Yield_Spread** = Yield_10Y - Yield_3M (inverted spread signals recession risk)

---

## 3. Raw Dataset Structure

Each stock CSV contains:

| Column | Type | Description |
|---|---|---|
| `Date` | datetime | Trading day |
| `Open` | float | Opening price |
| `High` | float | Highest intra-day price |
| `Low` | float | Lowest intra-day price |
| `Close` | float | Closing price |
| `Volume` | int | Shares traded |

Macro CSV (`data/macro.csv`) contains Date + VIX, Yield_10Y, Yield_3M, DXY, VIX_Zscore, Yield_Spread.

---

## 4. Feature Engineering

After loading raw data, we compute **12 features** (9 price-based + 3 macro). This is done by `_add_price_features()` and `build_feature_frame()` in `src/data_pipeline.py`.

### 4.1 Price Features (9)

| Feature | Formula / Method | Window | Description |
|---|---|---|---|
| `Close` | Raw closing price | - | Base price |
| `Volume` | Raw trading volume | - | Liquidity measure |
| `SMA_10` | Simple Moving Average | 10 days | Short-term trend |
| `SMA_30` | Simple Moving Average | 30 days | Medium-term trend |
| `EMA_12` | Exponential Moving Average | 12 days (span) | Responsive trend indicator |
| `RSI_14` | Relative Strength Index | 14 days | Momentum oscillator (0-100) |
| `Daily_Return` | `Close.pct_change()` | 1 day | Percentage price change |
| `Volatility_10` | Rolling std of Daily_Return | 10 days | Recent price volatility |
| `Log_Return` | `log(Close_t / Close_{t-1})` | 1 day | **This is also the prediction target** |

### 4.2 Macro Features (3)

| Feature | Source | Description |
|---|---|---|
| `VIX_Zscore` | Derived from ^VIX | Z-score of VIX over 252-day rolling window |
| `Yield_Spread` | ^TNX - ^IRX | 10Y-3M Treasury spread (recession signal) |
| `DXY` | DX-Y.NYB | US Dollar Index (raw) |

### 4.3 Feature Computation Details

**RSI (Relative Strength Index)**:
```
delta = Close.diff()
gain  = mean(positive deltas over 14 days)
loss  = mean(negative deltas over 14 days)
RS    = gain / loss
RSI   = 100 - (100 / (1 + RS))
```

**Log Return (prediction target)**:
```
Log_Return = log(Close_t / Close_{t-1})
```
Log returns are approximately stationary, making them a sound target for regression. Standard MAPE on log returns is undefined (values near 0), so we use **SMAPE** instead.

### 4.4 Handling NaN Values

Rolling window calculations produce NaN at the start. All rows with any NaN are **dropped** after feature engineering and macro merge. This removes ~30-50 rows.

---

## 5. Model Input and Output

### 5.1 Input Features (12 features fed to the model)

| # | Feature | Scale |
|---|---|---|
| 1 | Close | StandardScaler |
| 2 | Volume | StandardScaler |
| 3 | SMA_10 | StandardScaler |
| 4 | SMA_30 | StandardScaler |
| 5 | EMA_12 | StandardScaler |
| 6 | RSI_14 | StandardScaler |
| 7 | Daily_Return | StandardScaler |
| 8 | Volatility_10 | StandardScaler |
| 9 | Log_Return | StandardScaler |
| 10 | VIX_Zscore | StandardScaler |
| 11 | Yield_Spread | StandardScaler |
| 12 | DXY | StandardScaler |

**Input tensor shape**: `(batch_size, 30, 12)`
- `30` = sequence length (30 trading days of history)
- `12` = number of features

### 5.2 Output Field (Prediction target)

| Field | Description |
|---|---|
| `Log_Return` (day 31) | Next-day log return (unscaled) |

**Output tensor shape**: `(batch_size,)` -- a single scalar per sample.

The target is the **raw (unscaled) log return**. The scaler is only applied to the 12 input features, not the target. Price-domain metrics are computed by reconstructing: `P_t = P_{t-1} * exp(log_return_t)`.

### 5.3 Sliding Window Illustration

```
Day:  1   2   3   ... 29  30  | 31
      |--- Input (X) --------|  |- Target (y)
      [12 features] x 30 days -> predict Log_Return on day 31
```

---

## 6. Data Scaling

All 12 input features are standardized using `sklearn.preprocessing.StandardScaler`:

```
scaled_value = (value - mean) / std
```

**Critical**: The scaler is **fit on training data only** (dates <= 2021-12-31). Validation and test data are transformed using the training statistics. This prevents data leakage.

- **One scaler per ticker** -- each stock has its own feature distributions.
- Scalers are saved as `artifacts/models/scaler_{TICKER}.pkl`.

This replaces v1's MinMaxScaler-on-full-data approach, which leaked future statistics into training.

---

## 7. Train / Validation / Test Split

Chronological split (no shuffling):

| Split | Date Range | Years | Purpose | Approx. Rows |
|---|---|---|---|---|
| **Training** | 2015-01-01 -- 2021-12-31 | 7 years | Model weight optimization | ~1,730 |
| **Validation** | 2022-01-01 -- 2022-12-31 | 1 year | Early stopping, ensemble weights | ~250 |
| **Test** | 2023-01-01 -- 2023-12-31 | 1 year | Final held-out evaluation | ~250 |

### Why chronological?
Financial data is time-ordered. Random splitting leaks future information into training.

### Sequence generation
After scaling, sliding windows of length 30 are created. The split is applied to the date associated with each window's target (day 31), ensuring no window in the validation or test set includes any target from the training period.

---

## 8. Evaluation Metrics

### On log-returns (what the model predicts)

| Metric | Formula | Why used |
|---|---|---|
| **MAE** | `mean(|y - y_hat|)` | Standard error measure |
| **RMSE** | `sqrt(mean((y - y_hat)^2))` | Penalizes large errors |
| **SMAPE** | `mean(|y - y_hat| / ((|y| + |y_hat|) / 2)) * 100` | Finite when y near 0 (standard MAPE explodes) |
| **Dir. Accuracy** | `% where sign(y) == sign(y_hat)` | Trading relevance |

### On reconstructed prices (human-readable)

| Metric | Formula | Why used |
|---|---|---|
| **MAE (USD)** | `mean(|P_actual - P_predicted|)` | Error in dollars |
| **RMSE (USD)** | `sqrt(mean((P_actual - P_predicted)^2))` | Dollar error penalizing outliers |
| **MAPE (%)** | `mean(|P_actual - P_predicted| / P_actual) * 100` | Scale-independent percentage |

Price reconstruction: `P_t = P_{t-1} * exp(predicted_log_return_t)`, starting from the last known price before the test set.

---

## 9. Data Flow Summary

```
Yahoo Finance API
       |
       v
   Stock CSVs (data/{TICKER}.csv)  +  Macro CSV (data/macro.csv)
       |
       v  build_feature_frame()
   Merged DataFrame: 12 features + Date
       |
       v  StandardScaler (fit on TRAIN dates only)
   Scaled feature matrix
       |
       v  _build_sequences(seq_len=30)
   X: (samples, 30, 12)   y: (samples,) <- unscaled log-return
       |
       v  Chronological date-based split
   train/val/test DataLoaders
       |
       v  train_one() x 3 models (AdamW, MSE, grad clip, early stop)
   Best weights -> artifacts/models/*.pt
       |
       v  mc_dropout_predict() (100 forward passes)
   Mean + [5%, 95%] intervals -> calibration_report()
       |
       v  rolling_inverse_rmse_weights() + ensemble_predict()
   Dynamic ensemble -> artifacts/results/ensemble_*.npz
       |
       v  run_baselines() (ARIMA walk-forward + naive)
   Baseline predictions -> artifacts/results/{arima,naive}_*.npz
       |
       v  returns_to_prices()
   Reconstructed USD prices for MAE/RMSE/MAPE
       |
       v  summary.json (all metrics, all tickers)
   Dashboard reads this for tables and charts
```

---

## 10. Feature Ablation Configs

The ablation study (`src/ablation.py`) trains GRU under 5 feature subsets:

| Config | Features | Count | Purpose |
|---|---|---|---|
| full | All 12 | 12 | Baseline (best expected) |
| no_macro | Price features only | 9 | Do macro indicators help? |
| no_rsi | All except RSI_14 | 11 | Is RSI informative? |
| no_volume | All except Volume | 11 | Is Volume informative? |
| price_only | Close + Log_Return | 2 | Minimal baseline |

Results saved to `artifacts/results/ablation_{ticker}_{model}.json`.
