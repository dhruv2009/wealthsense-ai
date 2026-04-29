"""
Monte Carlo goal-planning engine + portfolio analytics.

Three big pieces:
  - simulate_goal: success probability for a savings goal under random walk.
  - portfolio_metrics: Sharpe / drawdown / cumulative for a return series.
  - get_model_outlook: turns recent model predictions into a bullish/bearish
    tilt that adjusts the simulation drift. This is the proposal's signature
    mechanism (DL → planning) — keep prominent.
"""
from __future__ import annotations
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def simulate_goal(current_savings: float, monthly_contribution: float,
                  target_amount: float, years: int,
                  annual_return: float = 0.08, annual_volatility: float = 0.18,
                  n_simulations: int = config.MC_SIMULATIONS):
    months = years * 12
    mu = annual_return / 12
    sigma = annual_volatility / np.sqrt(12)
    rng = np.random.default_rng(42)
    rets = rng.normal(mu, sigma, size=(n_simulations, months))

    paths = np.zeros((n_simulations, months + 1))
    paths[:, 0] = current_savings
    for m in range(months):
        paths[:, m + 1] = (paths[:, m] + monthly_contribution) * (1 + rets[:, m])

    final = paths[:, -1]
    success = final >= target_amount
    rate = float(np.mean(success) * 100)
    shortfall = float(np.mean(target_amount - final[~success])) if (~success).any() else 0.0

    extra = _binary_search_extra(current_savings, monthly_contribution, target_amount,
                                 years, annual_return, annual_volatility, 0.90)

    return {
        "success_rate": round(rate, 2),
        "median_final": round(float(np.median(final)), 2),
        "percentile_5": round(float(np.percentile(final, 5)), 2),
        "percentile_95": round(float(np.percentile(final, 95)), 2),
        "expected_shortfall": round(shortfall, 2),
        "paths": paths,
        "recommended_extra": round(extra, 2),
    }


def _binary_search_extra(curr, monthly, target, years, ret, vol, target_rate, n=2000):
    months = years * 12
    mu, sig = ret / 12, vol / np.sqrt(12)
    rng = np.random.default_rng(99)

    def rate(extra):
        rr = rng.normal(mu, sig, size=(n, months))
        p = np.zeros((n, months + 1))
        p[:, 0] = curr
        for m in range(months):
            p[:, m + 1] = (p[:, m] + monthly + extra) * (1 + rr[:, m])
        return np.mean(p[:, -1] >= target)

    if rate(0) >= target_rate:
        return 0.0
    lo, hi = 0.0, 50000.0
    for _ in range(30):
        mid = (lo + hi) / 2
        if rate(mid) >= target_rate:
            hi = mid
        else:
            lo = mid
    return hi


def portfolio_metrics(returns: np.ndarray, risk_free_rate: float = 0.02) -> dict:
    cum = np.cumprod(1 + returns)
    total = cum[-1] - 1
    ann_ret = (1 + total) ** (252 / len(returns)) - 1
    ann_vol = np.std(returns) * np.sqrt(252)
    sharpe = (ann_ret - risk_free_rate) / (ann_vol + 1e-8)
    peak = np.maximum.accumulate(cum)
    dd = (peak - cum) / peak
    max_dd = float(np.max(dd))
    return {
        "cumulative_return": round(float(total) * 100, 2),
        "annualized_return": round(float(ann_ret) * 100, 2),
        "annualized_volatility": round(float(ann_vol) * 100, 2),
        "sharpe_ratio": round(float(sharpe), 3),
        "sortino_ratio": round(_sortino(returns, risk_free_rate), 3),
        "max_drawdown": round(max_dd * 100, 2),
    }


def _sortino(returns: np.ndarray, rf: float = 0.02) -> float:
    """Sortino: penalizes only downside vol — better than Sharpe for asymmetric risk."""
    excess = returns - rf / 252
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf")
    dn_vol = np.std(downside) * np.sqrt(252)
    ann_excess = np.mean(excess) * 252
    return float(ann_excess / (dn_vol + 1e-8))


def trading_strategy_returns(actual_prices: np.ndarray, predicted_prices: np.ndarray,
                              transaction_cost_bps: float = 5.0):
    """
    Long when predicted_next > actual_today, else cash.
    Subtracts transaction cost (5 bps default = 0.05%) on each position change.
    Returns (buy_hold_returns, strategy_returns).
    """
    bh = np.diff(actual_prices) / actual_prices[:-1]
    signal = predicted_prices[1:] > actual_prices[:-1]
    flips = np.zeros_like(signal, dtype=bool)
    flips[0] = signal[0]
    flips[1:] = signal[1:] != signal[:-1]
    cost = (transaction_cost_bps / 10000.0) * flips.astype(float)
    strat = bh * signal.astype(float) - cost
    return bh, strat


def compute_portfolio_stats(stock_data: dict, weights: dict, tickers: list) -> dict:
    import pandas as pd
    rdf = pd.DataFrame()
    for t in tickers:
        if t in stock_data:
            rdf[t] = stock_data[t]["Close"].pct_change()
    rdf = rdf.dropna()
    w = np.array([weights.get(t, 0) for t in tickers])
    pr = rdf[tickers].values @ w

    per_stock = {}
    for t in tickers:
        if t in rdf.columns:
            r = rdf[t].values
            per_stock[t] = {
                "ann_return": round(float(np.mean(r) * 252) * 100, 2),
                "ann_volatility": round(float(np.std(r) * np.sqrt(252)) * 100, 2),
            }

    return {
        "ann_return": float(np.mean(pr) * 252),
        "ann_volatility": float(np.std(pr) * np.sqrt(252)),
        "ann_return_pct": round(float(np.mean(pr) * 252) * 100, 2),
        "ann_volatility_pct": round(float(np.std(pr) * np.sqrt(252)) * 100, 2),
        "per_stock": per_stock,
    }


def get_model_outlook(results_dir: str, tickers: list, weights: dict) -> dict:
    """
    Use ensemble (or best DL model) recent-window predictions to derive a
    bullish/bearish tilt. Returns a +/- 2% adjustment to expected return.
    """
    bullish = 0.0
    total_w = 0.0
    details: dict[str, dict] = {}

    for t in tickers:
        w = weights.get(t, 0)
        if w <= 0:
            continue

        y_true = y_pred = None
        used = None
        for name in ["ensemble", "transformer", "gru", "lstm"]:
            path = os.path.join(results_dir, f"{name}_{t}_preds.npz")
            if os.path.exists(path):
                d = np.load(path, allow_pickle=True)
                y_true = d["y_true"]
                y_pred = d["y_pred"]
                used = name
                break
        if y_true is None or len(y_true) < 21:
            continue

        # On log returns: positive predicted return = bullish
        pred_up = float(np.mean(y_pred[-20:] > 0))
        actual_dir = np.sign(y_true[-21:])
        pred_dir = np.sign(y_pred[-21:])
        recent_acc = float(np.mean(actual_dir[1:] == pred_dir[1:]))

        signal = 1.0 if pred_up > 0.5 else -1.0
        bullish += w * signal * recent_acc
        total_w += w
        details[t] = {
            "model": used,
            "pred_up_pct": round(pred_up * 100, 1),
            "recent_accuracy": round(recent_acc * 100, 1),
            "signal": "Bullish" if signal > 0 else "Bearish",
        }

    norm = bullish / total_w if total_w else 0.0
    adj = norm * 0.02
    label = "Bullish" if norm > 0.2 else "Bearish" if norm < -0.2 else "Neutral"
    return {
        "outlook_label": label,
        "return_adjustment": round(adj, 4),
        "adjustment_pct": round(adj * 100, 2),
        "confidence_score": round(float(norm), 3),
        "details": details,
    }
