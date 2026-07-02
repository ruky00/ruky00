"""
Edge lab — hunt for *repeatable* daily patterns, not just profitable-on-average.

A pattern that "always works" does not exist in liquid markets (it would be
arbitraged away). What we can find is an edge that is:

    significant  : unlikely to be luck            (t-stat / p-value)
    consistent   : positive in most calendar years (year-by-year hit rate)
    persistent   : still works out-of-sample       (1st half discovers, 2nd confirms)
    tradeable    : survives transaction costs       (net Sharpe > 0)

`evaluate` scores any daily signal on all four. `scan` runs a battery of
candidate signals and ranks them by a combined robustness score, with a
multiple-testing warning (scanning many ideas inflates false positives).

All signals are *causal*: the position for a day is known at/*before* that day's
open, and we trade the open->close session — no look-ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# stats helpers (no scipy)
# --------------------------------------------------------------------------- #
def _erf(x):
    t = 1.0 / (1.0 + 0.3275911 * abs(x))
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return np.sign(x) * y


def _p_two_sided(t):
    return float(2 * (1 - 0.5 * (1 + _erf(abs(t) / np.sqrt(2)))))


def _intraday_leg(df: pd.DataFrame) -> pd.Series:
    """Open->close session return (the leg most daily patterns trade)."""
    return (df["close"] / df["open"] - 1.0)


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def evaluate(df: pd.DataFrame, signal: pd.Series, name: str = "signal",
             cost_bps: float = 4.0) -> dict:
    """
    Score a daily signal (-1/0/1, indexed like df) traded on the open->close leg.
    Returns significance, consistency, OOS persistence and after-cost metrics.
    """
    leg = _intraday_leg(df)
    sig = signal.reindex(df.index).fillna(0)
    rt = 2 * cost_bps / 1e4
    strat = (sig * leg - sig.abs() * rt).dropna()
    active = strat[sig.reindex(strat.index) != 0]

    n = int((sig != 0).sum())
    if n < 10:
        return {"name": name, "n_trades": n, "tradeable": False, "score": 0.0}

    mean = active.mean()
    se = active.std(ddof=1) / np.sqrt(len(active)) if len(active) > 1 else 0
    t = mean / se if se else 0.0
    ann = (1 + strat).prod() ** (TRADING_DAYS / len(strat)) - 1
    vol = strat.std(ddof=0) * np.sqrt(TRADING_DAYS)
    sharpe = (strat.mean() / strat.std(ddof=0) * np.sqrt(TRADING_DAYS)
              if strat.std(ddof=0) else 0.0)

    # consistency: fraction of calendar years with positive total return
    by_year = strat.groupby(strat.index.year).sum()
    consistency = float((by_year > 0).mean()) if len(by_year) else 0.0

    # persistence: split-half out-of-sample
    half = len(strat) // 2
    sh1 = (strat.iloc[:half].mean() / strat.iloc[:half].std(ddof=0)
           * np.sqrt(TRADING_DAYS)) if strat.iloc[:half].std(ddof=0) else 0.0
    sh2 = (strat.iloc[half:].mean() / strat.iloc[half:].std(ddof=0)
           * np.sqrt(TRADING_DAYS)) if strat.iloc[half:].std(ddof=0) else 0.0

    # combined robustness score: reward significance + consistency, demand that
    # the edge survives out-of-sample and is net positive.
    oos_ok = 1.0 if (sh1 > 0 and sh2 > 0) else 0.25
    score = max(0.0, sharpe) * consistency * oos_ok * (1.0 if ann > 0 else 0.0)

    return {
        "name": name,
        "n_trades": n,
        "mean_bps": round(mean * 1e4, 2),
        "t_stat": round(t, 2),
        "p_value": round(_p_two_sided(t), 4),
        "ann_return": round(ann, 4),
        "sharpe": round(sharpe, 2),
        "win_rate": round(float((active > 0).mean()), 3),
        "consistency": round(consistency, 2),     # share of years positive
        "years": int(len(by_year)),
        "oos_sharpe_1h": round(sh1, 2),
        "oos_sharpe_2h": round(sh2, 2),
        "tradeable": ann > 0,
        "score": round(score, 3),
    }


# --------------------------------------------------------------------------- #
# candidate daily signals (all causal, known at/before the open)
# --------------------------------------------------------------------------- #
def sig_weekday(df, weekday: int) -> pd.Series:
    """Long the session on a given weekday (0=Mon .. 4=Fri)."""
    return pd.Series(np.where(df.index.dayofweek == weekday, 1, 0), index=df.index)


def sig_turn_of_month(df, days_before: int = 1, days_after: int = 3) -> pd.Series:
    """Long around the turn of the month (last `days_before` + first `days_after`)."""
    s = pd.Series(0, index=df.index)
    month = df.index.to_period("M")
    for _, idx in pd.Series(df.index, index=df.index).groupby(month):
        days = idx.index
        for d in days[:days_after]:
            s.loc[d] = 1
        for d in days[-days_before:]:
            s.loc[d] = 1
    return s


def sig_gap_fade(df, threshold: float = 0.005) -> pd.Series:
    """Fade the opening gap: long after a gap-down, short after a gap-up."""
    overnight = df["open"] / df["close"].shift(1) - 1.0
    s = pd.Series(0, index=df.index)
    s[overnight < -threshold] = 1
    s[overnight > threshold] = -1
    return s


def sig_gap_follow(df, threshold: float = 0.005) -> pd.Series:
    """Opposite of fade: ride the gap (gap-up -> long, gap-down -> short)."""
    return -sig_gap_fade(df, threshold)


def sig_lead_lag(df, leader: pd.DataFrame, lag: int = 1, threshold: float = 0.0) -> pd.Series:
    led = leader["close"].pct_change().reindex(df.index).shift(lag)
    return pd.Series(np.where(led > threshold, 1, np.where(led < -threshold, -1, 0)),
                     index=df.index)


def default_candidates(df, leader: pd.DataFrame | None = None) -> dict:
    cands = {
        "gap_fade": sig_gap_fade(df),
        "gap_follow": sig_gap_follow(df),
        "turn_of_month": sig_turn_of_month(df),
        **{f"weekday_{d}": sig_weekday(df, i)
           for i, d in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri"])},
    }
    if leader is not None:
        cands["lead_lag_1d"] = sig_lead_lag(df, leader, lag=1, threshold=0.005)
    return cands


# --------------------------------------------------------------------------- #
# scan + report
# --------------------------------------------------------------------------- #
def scan(df: pd.DataFrame, candidates: dict | None = None,
         leader: pd.DataFrame | None = None, cost_bps: float = 4.0) -> list[dict]:
    candidates = candidates or default_candidates(df, leader)
    rows = [evaluate(df, sig, name, cost_bps) for name, sig in candidates.items()]
    return sorted(rows, key=lambda r: r.get("score", 0), reverse=True)


def report(rows: list[dict], name: str = "asset") -> str:
    n_tested = len(rows)
    bonf = 0.05 / max(1, n_tested)     # Bonferroni-corrected significance bar
    L = [f"{'='*78}",
         f"REPEATABLE-EDGE SCAN — {name}   ({n_tested} patterns tested)",
         f"significance bar after multiple-testing (Bonferroni): p < {bonf:.4f}",
         "=" * 78,
         f"{'pattern':<16}{'Sharpe':>7}{'t':>6}{'p':>8}{'consist':>9}"
         f"{'OOS1':>6}{'OOS2':>6}{'trades':>7}  verdict"]
    L.append("-" * 78)
    for r in rows:
        if not r.get("tradeable", False) and r.get("n_trades", 0) < 10:
            continue
        sig_flag = "robust" if (r["score"] > 0.3 and r["consistency"] >= 0.6
                                and r["oos_sharpe_2h"] > 0) else \
                   "weak" if r["score"] > 0 else "no edge"
        L.append(f"{r['name']:<16}{r.get('sharpe',0):>7.2f}{r.get('t_stat',0):>6.2f}"
                 f"{r.get('p_value',1):>8.3f}{r.get('consistency',0):>8.0%}"
                 f"{r.get('oos_sharpe_1h',0):>6.2f}{r.get('oos_sharpe_2h',0):>6.2f}"
                 f"{r.get('n_trades',0):>7}  {sig_flag}")
    L.append("-" * 78)
    L.append("consist = share of calendar years the pattern was positive (your "
             "'repeatability'). OOS1/OOS2 = Sharpe in the 1st/2nd half of history.")
    L.append("A pattern is worth paper-testing only if: consist high, BOTH OOS "
             "halves > 0, and p beats the Bonferroni bar. In-sample alone is not enough.")
    return "\n".join(L)
