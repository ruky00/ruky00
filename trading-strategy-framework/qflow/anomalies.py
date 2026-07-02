"""
Daily-edge / anomaly scanner.

Systematically tests a battery of well-known *exploitable* daily patterns on
real OHLCV data and reports which ones show a statistically meaningful effect
(|t-stat| > ~2). No scipy needed — t-stats and a normal-approx p-value are
computed by hand.

Patterns tested
---------------
1. Overnight vs intraday return decomposition
   (the close->open move vs the open->close move). The classic finding: most of
   the long-run drift lives *overnight*; the regular session is roughly flat.
2. Gap behaviour — when price gaps up/down at the open, does the rest of the day
   *continue* (gap-and-go) or *fill* (fade)?
3. Day-of-week seasonality.
4. Daily return autocorrelation (short-term momentum vs reversal).
5. Cross-market lead-lag — does asset A's move today predict asset B tomorrow?
   (Your "Spanish stock leads its US listing" idea, generalised to any pair.)

Everything here is descriptive statistics on historical data. A significant
in-sample t-stat is a *hypothesis*, not a guarantee — confirm out-of-sample and
after costs before trading it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _tstat(x: pd.Series) -> tuple[float, float, float]:
    """Return (mean, t-stat, two-sided normal-approx p) for a return series."""
    x = x.dropna()
    n = len(x)
    if n < 5:
        return 0.0, 0.0, 1.0
    mean = x.mean()
    se = x.std(ddof=1) / np.sqrt(n)
    t = mean / se if se > 0 else 0.0
    # two-sided p via normal approximation (erf-based, no scipy)
    p = 2 * (1 - _norm_cdf(abs(t)))
    return float(mean), float(t), float(p)


def _norm_cdf(z: float) -> float:
    return 0.5 * (1 + _erf(z / np.sqrt(2)))


def _erf(x: float) -> float:
    # Abramowitz & Stegun 7.1.26
    t = 1.0 / (1.0 + 0.3275911 * abs(x))
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return float(np.sign(x) * y)


def _components(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["overnight"] = df["open"] / df["close"].shift(1) - 1.0     # prev close -> open
    out["intraday"] = df["close"] / df["open"] - 1.0               # open -> close
    out["full"] = df["close"] / df["close"].shift(1) - 1.0         # close -> close
    return out.dropna()


# --------------------------------------------------------------------------- #
# 1. overnight vs intraday
# --------------------------------------------------------------------------- #
def overnight_vs_intraday(df: pd.DataFrame) -> dict:
    c = _components(df)
    res = {}
    for name in ("overnight", "intraday", "full"):
        mean, t, p = _tstat(c[name])
        ann = (1 + c[name]).prod() ** (252 / len(c)) - 1   # annualised geometric
        res[name] = {"mean_bps": mean * 1e4, "t": t, "p": p,
                     "annualised": ann, "hit_rate": float((c[name] > 0).mean())}
    return res


# --------------------------------------------------------------------------- #
# 2. gap behaviour
# --------------------------------------------------------------------------- #
def gap_behaviour(df: pd.DataFrame, threshold: float = 0.005) -> dict:
    c = _components(df)
    up = c[c["overnight"] > threshold]
    down = c[c["overnight"] < -threshold]
    res = {}
    for label, grp in (("gap_up", up), ("gap_down", down)):
        if len(grp) < 5:
            res[label] = {"n": len(grp)}
            continue
        mean, t, p = _tstat(grp["intraday"])     # what happens AFTER the gap, intraday
        res[label] = {"n": int(len(grp)),
                      "next_intraday_mean_bps": mean * 1e4, "t": t, "p": p,
                      "continue_rate": float((np.sign(grp["intraday"]) ==
                                              np.sign(grp["overnight"])).mean())}
    return res


# --------------------------------------------------------------------------- #
# 3. day-of-week
# --------------------------------------------------------------------------- #
def day_of_week(df: pd.DataFrame) -> dict:
    full = (df["close"].pct_change()).dropna()
    res = {}
    for i, name in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri"]):
        grp = full[full.index.dayofweek == i]
        mean, t, p = _tstat(grp)
        res[name] = {"mean_bps": mean * 1e4, "t": t, "p": p, "n": int(len(grp))}
    return res


# --------------------------------------------------------------------------- #
# 4. autocorrelation (momentum vs reversal)
# --------------------------------------------------------------------------- #
def autocorrelation(df: pd.DataFrame, lags=(1, 2, 3, 5)) -> dict:
    r = df["close"].pct_change().dropna()
    res = {}
    for lag in lags:
        ac = r.autocorr(lag)
        # t-stat of autocorr ~ ac * sqrt(n)
        res[f"lag_{lag}"] = {"autocorr": float(ac), "t": float(ac * np.sqrt(len(r)))}
    return res


# --------------------------------------------------------------------------- #
# 5. cross-market lead-lag
# --------------------------------------------------------------------------- #
def lead_lag(leader: pd.DataFrame, follower: pd.DataFrame, max_lag: int = 2) -> dict:
    """
    Does the LEADER's daily return predict the FOLLOWER's *future* return?
    Aligns on common dates. lag>0 means leader leads follower by `lag` days.

    Use for your example: leader = Spanish stock (closes earlier / different
    session), follower = its US listing or a US index.
    """
    a = leader["close"].pct_change().rename("leader")
    b = follower["close"].pct_change().rename("follower")
    j = pd.concat([a, b], axis=1).dropna()
    res = {}
    for lag in range(1, max_lag + 1):
        # leader at t-lag vs follower at t
        x = j["leader"].shift(lag)
        pair = pd.concat([x, j["follower"]], axis=1).dropna()
        if len(pair) < 10:
            res[f"lead_{lag}d"] = {"n": len(pair)}
            continue
        corr = pair["leader"].corr(pair["follower"])
        t = corr * np.sqrt(len(pair))
        res[f"lead_{lag}d"] = {"corr": float(corr), "t": float(t), "n": int(len(pair))}
    # contemporaneous for reference
    res["same_day"] = {"corr": float(j["leader"].corr(j["follower"])), "n": int(len(j))}
    return res


# --------------------------------------------------------------------------- #
# tradeable backtests built from the anomalies (cost-aware)
# --------------------------------------------------------------------------- #
def _equity_stats(daily_ret: pd.Series, capital: float) -> dict:
    from . import metrics
    eq = capital * (1 + daily_ret).cumprod()
    return {
        "equity": eq,
        "CAGR": metrics.cagr(eq),
        "Sharpe": metrics.sharpe(daily_ret),
        "Max Drawdown": metrics.max_drawdown(eq),
        "Total Return": metrics.total_return(eq),
        "trading_days": int((daily_ret != 0).sum()),
    }


def backtest_overnight(df: pd.DataFrame, cost_bps: float = 4.0,
                       capital: float = 10_000.0) -> dict:
    """Hold ONLY overnight: buy at the close, sell at the next open, every day.
    Two fills per day, so it is very cost-sensitive."""
    c = _components(df)
    rt_cost = 2 * cost_bps / 1e4
    ret = c["overnight"] - rt_cost
    return _equity_stats(ret, capital)


def backtest_gap_fade(df: pd.DataFrame, threshold: float = 0.005,
                      cost_bps: float = 4.0, capital: float = 10_000.0) -> dict:
    """
    Fade opening gaps: on a gap-down beyond `threshold`, go long the session
    (open->close); on a gap-up, go short the session. Only trades gap days, so
    far fewer fills than the overnight book.
    """
    c = _components(df)
    rt_cost = 2 * cost_bps / 1e4
    pos = pd.Series(0.0, index=c.index)
    pos[c["overnight"] < -threshold] = 1.0     # gap down -> buy the reversal
    pos[c["overnight"] > threshold] = -1.0     # gap up   -> short the fade
    ret = pos * c["intraday"] - (pos.abs() * rt_cost)
    return _equity_stats(ret, capital)


def backtest_lead_lag(leader: pd.DataFrame, follower: pd.DataFrame,
                      lag: int = 1, threshold: float = 0.0,
                      cost_bps: float = 4.0, capital: float = 10_000.0) -> dict:
    """
    Trade the FOLLOWER using the LEADER's lagged signal: if the leader rose
    `lag` days ago (beyond `threshold`), be long the follower today; if it fell,
    be short. This is the "Spanish stock leads US listing" trade, generalised.
    """
    a = leader["close"].pct_change().rename("leader")
    b = follower["close"].pct_change().rename("follower")
    j = pd.concat([a, b], axis=1).dropna()
    sig = np.where(j["leader"].shift(lag) > threshold, 1.0,
                   np.where(j["leader"].shift(lag) < -threshold, -1.0, 0.0))
    sig = pd.Series(sig, index=j.index).fillna(0.0)
    rt_cost = 2 * cost_bps / 1e4
    turn = sig.diff().abs().fillna(0.0)
    ret = sig * j["follower"] - turn * rt_cost / 2
    return _equity_stats(ret, capital)


# --------------------------------------------------------------------------- #
# scan + report
# --------------------------------------------------------------------------- #
def scan(df: pd.DataFrame, name: str = "asset") -> dict:
    return {
        "overnight_vs_intraday": overnight_vs_intraday(df),
        "gap_behaviour": gap_behaviour(df),
        "day_of_week": day_of_week(df),
        "autocorrelation": autocorrelation(df),
    }


def report(scan_result: dict, name: str = "asset") -> str:
    L = [f"{'='*64}", f"DAILY-EDGE SCAN — {name}", "=" * 64]

    oi = scan_result["overnight_vs_intraday"]
    L.append("\n[1] Overnight vs Intraday (annualised return | t-stat | hit%)")
    for k in ("overnight", "intraday", "full"):
        d = oi[k]
        flag = "  <-- significant" if abs(d["t"]) > 2 else ""
        L.append(f"    {k:<10} {d['annualised']*100:>7.1f}%   t={d['t']:>5.2f}   "
                 f"{d['hit_rate']*100:>4.0f}%{flag}")

    gb = scan_result["gap_behaviour"]
    L.append("\n[2] Gap behaviour (intraday move AFTER an opening gap)")
    for k in ("gap_up", "gap_down"):
        d = gb[k]
        if "t" not in d:
            L.append(f"    {k:<10} n={d.get('n',0)} (too few)")
            continue
        verdict = "CONTINUE" if d["continue_rate"] > 0.5 else "FADE"
        flag = "  <-- significant" if abs(d["t"]) > 2 else ""
        L.append(f"    {k:<10} n={d['n']:>4}  next_intraday={d['next_intraday_mean_bps']:>6.1f}bps  "
                 f"t={d['t']:>5.2f}  {verdict} {d['continue_rate']*100:.0f}%{flag}")

    dow = scan_result["day_of_week"]
    L.append("\n[3] Day-of-week (mean daily return)")
    for k, d in dow.items():
        flag = "  <-- significant" if abs(d["t"]) > 2 else ""
        L.append(f"    {k}  {d['mean_bps']:>6.1f}bps  t={d['t']:>5.2f}  n={d['n']}{flag}")

    ac = scan_result["autocorrelation"]
    L.append("\n[4] Return autocorrelation (>0 momentum, <0 reversal)")
    for k, d in ac.items():
        flag = "  <-- significant" if abs(d["t"]) > 2 else ""
        L.append(f"    {k}  ac={d['autocorr']:>6.3f}  t={d['t']:>5.2f}{flag}")

    L.append("=" * 64)
    L.append("Note: in-sample stats are hypotheses. Confirm OOS and after costs.")
    return "\n".join(L)
