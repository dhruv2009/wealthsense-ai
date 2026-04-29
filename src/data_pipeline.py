"""
Data pipeline: Yahoo Finance OHLCV + macro features + log-return target.

Key fixes vs v1:
  - Predicts Log_Return (stationary) instead of raw Close (non-stationary).
  - StandardScaler fit on TRAIN ONLY (v1 fit on full data — leakage).
  - Adds macro features: VIX_Zscore, Yield_Spread, DXY.
"""
import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ── Indicators ──────────────────────────────────────────────────────────
def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean().replace(0, np.nan)
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def _add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_30"] = df["Close"].rolling(30).mean()
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["RSI_14"] = _rsi(df["Close"], 14)
    df["Daily_Return"] = df["Close"].pct_change()
    df["Volatility_10"] = df["Daily_Return"].rolling(10).std()
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    return df


# ── Download helpers ────────────────────────────────────────────────────
def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]) for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    if "Date" not in df.columns:
        df = df.reset_index().rename(columns={"index": "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).reset_index(drop=True)
    return df


def _download_ticker(ticker: str) -> pd.DataFrame:
    cache = os.path.join(config.DATA_DIR, f"{ticker}.csv")
    if os.path.exists(cache):
        df = pd.read_csv(cache)
        return _normalize(df)
    df = yf.download(ticker, start=config.START_DATE, end=config.END_DATE,
                     auto_adjust=False, progress=False).reset_index()
    df = _normalize(df)
    df.to_csv(cache, index=False)
    return df


def _download_macro() -> pd.DataFrame:
    cache = os.path.join(config.DATA_DIR, "macro.csv")
    if os.path.exists(cache):
        df = pd.read_csv(cache, parse_dates=["Date"])
        return df
    frames = []
    for tk, col in config.MACRO_TICKERS.items():
        try:
            df = yf.download(tk, start=config.START_DATE, end=config.END_DATE,
                             auto_adjust=False, progress=False).reset_index()
            df = _normalize(df)
            df = df[["Date", "Close"]].rename(columns={"Close": col})
            frames.append(df)
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    macro = frames[0]
    for f in frames[1:]:
        macro = macro.merge(f, on="Date", how="outer")
    macro = macro.sort_values("Date").ffill().bfill()
    if "Yield_10Y" in macro.columns and "Yield_3M" in macro.columns:
        macro["Yield_Spread"] = macro["Yield_10Y"] - macro["Yield_3M"]
    if "VIX" in macro.columns:
        roll = macro["VIX"].rolling(252)
        macro["VIX_Zscore"] = ((macro["VIX"] - roll.mean()) / roll.std()).fillna(0)
    macro.to_csv(cache, index=False)
    return macro


# ── Public API ──────────────────────────────────────────────────────────
def download_data() -> dict[str, pd.DataFrame]:
    """Return dict[ticker] -> raw OHLCV DataFrame."""
    return {t: _download_ticker(t) for t in config.TICKERS}


def build_feature_frame(ticker: str, raw: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    df = _add_price_features(raw)
    if not macro.empty:
        df = df.merge(macro[["Date"] + [c for c in config.MACRO_FEATURES if c in macro.columns]],
                      on="Date", how="left")
    df = df.dropna().reset_index(drop=True)
    return df


class StockDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def _build_sequences(features: np.ndarray, targets: np.ndarray, dates: pd.Series, seq_len: int):
    X, y, d = [], [], []
    for i in range(seq_len, len(features)):
        X.append(features[i - seq_len:i])
        y.append(targets[i])
        d.append(dates.iloc[i])
    return np.array(X), np.array(y), pd.Series(d)


def prepare_ticker_data(ticker: str, raw_data: dict, macro_df: pd.DataFrame,
                        feature_cols: list[str] | None = None):
    """
    Full pipeline: features -> train-only-fit scaler -> chronological split -> sliding windows.
    Returns dict with train/val/test loaders, scaler, dates, and the close-price series.
    """
    feature_cols = feature_cols or config.FEATURE_COLS
    fe = build_feature_frame(ticker, raw_data[ticker], macro_df)

    feats = fe[feature_cols].astype(float).values
    target = fe[config.TARGET_COL].astype(float).values
    dates = fe["Date"]

    # Scale features using TRAIN ONLY (no leakage)
    train_mask = dates <= config.TRAIN_END
    scaler = StandardScaler()
    scaler.fit(feats[train_mask.values])
    feats_scaled = scaler.transform(feats)

    X_all, y_all, d_all = _build_sequences(feats_scaled, target, dates, config.SEQUENCE_LENGTH)

    yr = pd.to_datetime(d_all).dt
    train_sel = (d_all <= pd.Timestamp(config.TRAIN_END)).values
    val_sel = ((d_all > pd.Timestamp(config.TRAIN_END)) &
               (d_all <= pd.Timestamp(config.VAL_END))).values
    test_sel = (d_all > pd.Timestamp(config.VAL_END)).values

    train_loader = DataLoader(StockDataset(X_all[train_sel], y_all[train_sel]),
                              batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(StockDataset(X_all[val_sel], y_all[val_sel]),
                            batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(StockDataset(X_all[test_sel], y_all[test_sel]),
                             batch_size=config.BATCH_SIZE, shuffle=False)

    joblib.dump(scaler, os.path.join(config.SAVED_MODELS_DIR, f"scaler_{ticker}.pkl"))

    return {
        "ticker": ticker,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "X_train": X_all[train_sel], "y_train": y_all[train_sel],
        "X_val": X_all[val_sel], "y_val": y_all[val_sel],
        "X_test": X_all[test_sel], "y_test": y_all[test_sel],
        "test_dates": d_all[test_sel].reset_index(drop=True),
        "scaler": scaler,
        "feature_frame": fe,
        "feature_cols": feature_cols,
    }


def load_all_with_macro():
    raw = download_data()
    macro = _download_macro()
    return raw, macro


if __name__ == "__main__":
    raw, macro = load_all_with_macro()
    for t in config.TICKERS:
        d = prepare_ticker_data(t, raw, macro)
        print(f"{t}: train={len(d['X_train'])} val={len(d['X_val'])} test={len(d['X_test'])}")
