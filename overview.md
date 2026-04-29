# WealthSense AI — What This Project Does (Non-Technical Overview)

> This file explains the entire project in simple language.
> No coding knowledge needed. If you want the developer guide, see [agent.md](agent.md).
> For dataset details, see [data.md](data.md).

---

## What is WealthSense AI?

WealthSense AI is like a **smart financial assistant** that:

1. Looks at how stocks have performed over the past 9 years (2015 to 2023), plus economic indicators like market fear (VIX), interest rates, and the US dollar
2. Learns patterns from that history using three different AI models
3. Combines their predictions using a smart weighting system (ensemble)
4. Provides uncertainty estimates ("how confident is the prediction?")
5. Compares AI predictions against classical forecasting methods to prove they work
6. Helps you plan financial goals (like buying a house or saving for retirement)
7. Shows everything on a clean, interactive website you can click through

---

## The Big Picture -- Step by Step

### Step 1: Get the Data

We download 9 years of daily stock prices for 5 well-known companies:

- **Apple (AAPL)** -- iPhones, MacBooks
- **Microsoft (MSFT)** -- Windows, Office, Azure
- **NVIDIA (NVDA)** -- Graphics cards, AI chips
- **Tesla (TSLA)** -- Electric cars
- **SPY** -- Represents the entire US stock market (S&P 500)

We also download **economic indicators**:
- **VIX** -- the "fear index" (how nervous investors are)
- **Treasury Yields** -- interest rates on government bonds (10-year vs 3-month spread signals recession risk)
- **US Dollar Index (DXY)** -- strength of the dollar

### Step 2: Create Smart Features (Feature Engineering)

From raw prices and economic data, we calculate **12 features** that capture different aspects of the market:

**Price-based (9 features):**
- Closing price and trading volume
- Moving averages (10-day and 30-day trends)
- Exponential moving average (recent trend)
- RSI (is the stock overbought or oversold?)
- Daily return (how much did it change today?)
- Volatility (how wild have price swings been?)
- Log return (the mathematical way to measure price changes -- this is what our models predict)

**Macro-based (3 features):**
- VIX Z-score (is market fear unusually high?)
- Yield Spread (are interest rates signaling trouble?)
- Dollar Index (is the dollar strong or weak?)

### Step 3: Split the Data Fairly

We split the data by time (not randomly!) because in real life you can't use future data to predict the past:

- **Training (2015-2021)**: The AI learns patterns from 7 years of history
- **Validation (2022)**: We use this year to decide when to stop training and how to weight models
- **Testing (2023)**: The AI has **never seen** this year. This is our honest exam.

### Step 4: Train Three AI Models + Ensemble

We train three different types of AI, all with modern architecture (LayerNorm + GELU activation heads):

**1. LSTM (Long Short-Term Memory)**
Think of this as an AI with a **notepad**. As it reads through each day, it decides what to write down and what to forget. Good for time-based patterns.

**2. GRU (Gated Recurrent Unit)**
A **simplified LSTM** -- same idea, but with a smaller notepad. Faster to train and often works just as well.

**3. Transformer**
The same type of AI behind ChatGPT. Instead of reading data one day at a time, it can **look at all 30 days simultaneously** and figure out which days matter most. It uses a "causal mask" so it can't cheat by looking ahead.

**Dynamic Ensemble:**
Instead of picking one model, we combine all three. Models that performed better recently get more weight. This changes per stock -- the GRU might get more weight on Tesla while the Transformer leads on Apple.

### Step 5: Measure Uncertainty (MC Dropout)

A single prediction like "the stock will go up 1%" isn't useful without knowing **how confident** the model is. We run each model **100 times** with slight random variations (Monte Carlo Dropout) to get a range:

- "The stock will likely return between -0.5% and +2.5%"
- We then check if these ranges are accurate (calibration report)

### Step 6: Compare Against Classical Methods

To prove our AI models actually add value, we compare them against two traditional approaches:

- **Naive Persistence**: "Tomorrow's return = today's return" (simplest possible forecast)
- **ARIMA(5,0,0)**: A classical statistics model commonly used in finance

If our deep learning models can't beat these baselines, they're not worth the complexity.

### Step 7: Feature Ablation ("Which Features Matter?")

We retrain the GRU model under different feature combinations:
- All 12 features
- Without macro indicators (do economic factors help?)
- Without RSI (is momentum useful?)
- Without volume (does trading activity matter?)
- Only price + log return (minimal baseline)

This answers the question: "Are the extra features actually helping, or could we get away with less?"

### Step 8: Attention Visualization

The Transformer model can tell us **which days it pays most attention to** when making a prediction. We extract these attention weights and show them as a heatmap -- answering "what is the model actually looking at?"

### Step 9: Goal Planning with Monte Carlo Simulation

Say you want to buy a house in 3 years and need 60,000 USD. You have 10,000 USD saved and add 1,200 USD per month.

We run **5,000 imaginary futures** using your actual portfolio's historical return and volatility, plus a tilt from our AI predictions (if models predict the market will go up, we slightly increase the expected return).

If 4,000 out of 5,000 futures reach the goal, that's an **80% success rate**. We also calculate how much extra you'd need to save for 90% confidence.

### Step 10: The Dashboard

Everything comes together in a website with 4 pages.

---

## The Four Dashboard Pages

### Page 1: Portfolio Analytics
**What you do**: Pick how much of your money goes into each stock (e.g., 30% Apple, 20% Microsoft).
**What you see**:
- Pie chart of your allocation
- How your portfolio would have grown over time
- Risk metrics: Sharpe ratio (reward per risk), Sortino ratio (reward per downside risk), maximum drawdown (worst dip)

### Page 2: Forecast View
**What you see**:
- Table comparing all models (3 DL + ensemble + 2 baselines) with accuracy metrics
- Ensemble weights per model
- Charts: actual vs predicted prices with confidence bands (from MC Dropout)
- Training curves (did the model learn or overfit?)
- Calibration table (are the confidence bands accurate?)
- Transformer attention heatmap (which days does it focus on?)
- Trading strategy simulation: buy when model predicts up, cash when down (with realistic transaction costs)
- Feature ablation results

### Page 3: Goal Planner
**What you do**: Enter your goal (amount, timeline, monthly savings).
**What you see**:
- Your portfolio's actual return and volatility (computed from real data)
- AI model outlook (Bullish/Neutral/Bearish based on recent predictions)
- Success probability from Monte Carlo simulation
- Fan chart of thousands of possible futures
- How much extra to save for 90% confidence

### Page 4: AI Chat
**What you do**: Type questions in plain English.
**What you see**: Answers based on your actual model results and data.

Examples:
- "Tell me about AAPL" -- stock summary with best model
- "Which model is best?" -- comparison table
- "Compare models" -- detailed breakdown with baselines
- "I want to buy a house in 3 years" -- runs a goal simulation
- "Why?" or "Explain" -- explains the previous answer in detail

Works completely **free and offline** without any API key. Optionally, set a free Google Gemini API key for more natural, AI-powered conversations (get one at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)).

---

## How the Files Are Organized

```
wealthsenseAI/
|
|-- app.py              The website/dashboard. Run this to see everything.
|-- config.py           All settings in one place
|
|-- src/
|   |-- data_pipeline.py   Downloads stock + macro data, creates features
|   |-- models.py          The three AI models (LSTM, GRU, Transformer)
|   |-- train.py           Trains all models, runs uncertainty + ensemble + baselines
|   |-- monte_carlo.py     Goal planning simulation engine
|   |-- ensemble.py        Combines model predictions with smart weighting
|   |-- uncertainty.py     MC Dropout for confidence intervals
|   |-- baselines.py       ARIMA and naive persistence baselines
|   |-- ablation.py        Feature importance study
|   |-- attention.py       Transformer attention visualization
|   |-- chat.py            AI chat engine (Google Gemini + free fallback)
|
|-- data/               Downloaded stock price + macro CSV files
|-- artifacts/
|   |-- models/         Trained AI model files
|   |-- results/        Predictions and accuracy scores
|
|-- overview.md         THIS FILE
|-- agent.md            Developer guide
|-- data.md             Dataset documentation
|-- README.md           Quick-start guide
```

---

## How to Run It

```
pip install -r requirements.txt      (install libraries -- only once)
python src/train.py                  (train all AI models)
streamlit run app.py                 (open the dashboard website)
```

Optional:
```
python src/ablation.py               (run feature ablation study)
cp .env.example .env                 (add free Google Gemini API key)
```

---

## Common Questions

**Q: Does this actually predict the stock market?**
A: It predicts the next day's return with reasonable accuracy. But stock markets are inherently unpredictable -- this is an academic project demonstrating deep learning techniques, not a money-making tool.

**Q: Do I need a GPU?**
A: No. Training works on a regular laptop CPU. It uses GPU if available.

**Q: What makes this different from a simple prediction model?**
A: Several things that elevate it beyond a basic project:
- **Ensemble**: Combines three models with dynamic weighting
- **Uncertainty**: MC Dropout with calibration diagnostics
- **Baselines**: ARIMA + naive prove the DL models add value
- **Ablation**: Tests which features actually matter
- **Interpretability**: Attention heatmaps show what the Transformer focuses on
- **Dual metrics**: Reports both log-return metrics (SMAPE) and price metrics (MAPE)
- **Macro features**: VIX, yield spread, dollar index enrich the input
- **Transaction costs**: Trading strategy includes realistic costs (5 bps per trade)
- **Sortino ratio**: Measures risk-adjusted returns using only downside volatility

**Q: Why log returns instead of raw prices?**
A: Raw stock prices are non-stationary (they trend upward over time). Log returns are approximately stationary, which makes them much easier for neural networks to learn. We reconstruct prices afterward for human-readable metrics.

**Q: What's the "causal mask" in the Transformer?**
A: It prevents the model from cheating. Without a mask, the Transformer could look at day 30 when processing day 1. The causal mask ensures each position can only attend to itself and earlier positions, just like in real-time prediction.

**Q: Why not use GPT/ChatGPT for predictions?**
A: Language models aren't designed for numerical time series. Our models are specialized neural networks trained specifically on financial data with proper train/test splits.
