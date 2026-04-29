"""
WealthSense AI — Central configuration.
Frozen 2015-2023 window per project proposal.
Hyperparameters tuned per v2 (modern heads, AdamW, log-return target).
"""
import os

# Paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")
SAVED_MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")
RESULTS_DIR = os.path.join(ARTIFACTS_DIR, "results")

for d in (DATA_DIR, ARTIFACTS_DIR, SAVED_MODELS_DIR, RESULTS_DIR):
    os.makedirs(d, exist_ok=True)

# Dataset (frozen for reproducibility per proposal)
TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "SPY"]
START_DATE = "2015-01-01"
END_DATE = "2023-12-31"

TRAIN_END = "2021-12-31"
VAL_END = "2022-12-31"
# Test: 2023-01-01 -> 2023-12-31

# Macro features (downloaded alongside ticker data)
MACRO_TICKERS = {
    "^VIX": "VIX",
    "^TNX": "Yield_10Y",
    "^IRX": "Yield_3M",
    "DX-Y.NYB": "DXY",
}

# Feature engineering — target is Log_Return (stationary, sound for finance)
PRICE_FEATURES = ["Close", "Volume", "SMA_10", "SMA_30", "EMA_12", "RSI_14",
                  "Daily_Return", "Volatility_10", "Log_Return"]
MACRO_FEATURES = ["VIX_Zscore", "Yield_Spread", "DXY"]
FEATURE_COLS = PRICE_FEATURES + MACRO_FEATURES
OPTIMISED_FEATURE_COLS = [c for c in FEATURE_COLS if "RSI" not in c]
TARGET_COL = "Log_Return"
SEQUENCE_LENGTH = 30

# Model hyperparameters (modern: bigger hidden, LayerNorm + GELU heads)
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2
TRANSFORMER_HEADS = 4
TRANSFORMER_DIM = 256

# Training
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 100
PATIENCE = 15
SEED = 42

# Uncertainty
MC_DROPOUT_SAMPLES = 100

# Monte Carlo goal engine
MC_SIMULATIONS = 5000
TRADING_DAYS_PER_YEAR = 252

# Walk-forward validation
WF_FOLD_SIZE_MONTHS = 6
WF_MIN_TRAIN_YEARS = 3
WF_FINETUNE_EPOCHS = 10
WF_FINETUNE_LR = 1e-5

# Chat (Google Gemini — free tier)
GEMINI_MODEL = "gemini-flash-latest"
