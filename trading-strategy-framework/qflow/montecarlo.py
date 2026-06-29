"""
Monte-Carlo robustness analysis via trade-sequence bootstrapping.

Rather than assuming a return distribution, we resample the strategy's *actual*
realised trade R-multiples (with replacement) to build thousands of alternative
equity paths. This answers:

    * probability the strategy ends underwater
    * distribution of final returns
    * distribution of maximum drawdown
    * realistic worst-case (5th percentile) outcomes

A strategy is "robust" if its return distribution stays positive and its
worst-case drawdown is survivable; "fragile" if a reshuffle of the same trades
can easily wipe the account.
"""

from __future__ import annotations

import numpy as np


def simulate(r_multiples,
             n_paths: int = 5000,
             n_trades: int | None = None,
             risk_per_trade: float = 0.01,
             start_equity: float = 10_000.0,
             seed: int | None = 7) -> dict:
    """
    `r_multiples`: list of realised trade results in R units (pnl / initial risk).
    Each simulated trade moves equity by `risk_per_trade * R`.
    """
    r = np.asarray(r_multiples, dtype=float)
    if r.size == 0:
        raise ValueError("No trades to simulate.")
    n_trades = n_trades or r.size
    rng = np.random.default_rng(seed)

    sampled = rng.choice(r, size=(n_paths, n_trades), replace=True)
    # multiplicative equity path: equity *= (1 + risk*R) per trade
    growth = 1.0 + risk_per_trade * sampled
    growth = np.clip(growth, 0.01, None)        # cannot lose >100% on a trade
    paths = start_equity * np.cumprod(growth, axis=1)

    finals = paths[:, -1]
    final_returns = finals / start_equity - 1.0

    # max drawdown per path
    running_max = np.maximum.accumulate(paths, axis=1)
    drawdowns = (paths / running_max - 1.0).min(axis=1)

    pct = lambda a, q: float(np.percentile(a, q))
    return {
        "n_paths": n_paths,
        "n_trades": n_trades,
        "prob_loss": float((final_returns < 0).mean()),
        "prob_50pct_dd": float((drawdowns < -0.5).mean()),
        "median_return": float(np.median(final_returns)),
        "mean_return": float(np.mean(final_returns)),
        "return_p05": pct(final_returns, 5),
        "return_p25": pct(final_returns, 25),
        "return_p75": pct(final_returns, 75),
        "return_p95": pct(final_returns, 95),
        "median_max_dd": float(np.median(drawdowns)),
        "worst_max_dd": float(drawdowns.min()),
        "max_dd_p05": pct(drawdowns, 5),
    }


def verdict(result: dict) -> str:
    """Heuristic robust/fragile call from the simulation summary."""
    if result["prob_loss"] < 0.15 and result["max_dd_p05"] > -0.35:
        return "ROBUST — positive across most resamples, survivable worst case."
    if result["prob_loss"] < 0.35:
        return "MODERATE — edge present but path risk is meaningful; size down."
    return "FRAGILE — too easy to lose money on a reshuffle; rework the edge."


def report(result: dict) -> str:
    lines = [
        "Monte-Carlo (trade bootstrap)",
        "-" * 40,
        f"Paths / trades         : {result['n_paths']} / {result['n_trades']}",
        f"Probability of loss    : {result['prob_loss']*100:6.2f}%",
        f"P(>50% drawdown)       : {result['prob_50pct_dd']*100:6.2f}%",
        f"Median final return    : {result['median_return']*100:6.2f}%",
        f"Return  5th pctile     : {result['return_p05']*100:6.2f}%",
        f"Return 95th pctile     : {result['return_p95']*100:6.2f}%",
        f"Median max drawdown    : {result['median_max_dd']*100:6.2f}%",
        f"Worst-case max DD (p05): {result['max_dd_p05']*100:6.2f}%",
        "-" * 40,
        verdict(result),
    ]
    return "\n".join(lines)
