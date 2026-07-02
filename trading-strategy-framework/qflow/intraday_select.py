"""
Autonomous intraday selection — let the bot pick its own (strategy, interval).

The lab (`examples/research/intraday_lab.py`) does this interactively; this module
packages the same logic so the *bot* can call it at startup and be self-sufficient:
for a symbol's recent 5-minute bars it sweeps the intraday strategies across
candle intervals, walk-forwards each out-of-sample, and returns the best
(strategy, interval) by OOS Sharpe — or nothing if no combination clears a
minimum bar (better to sit out than trade a non-edge).

    from qflow import feeds, intraday_select
    df = feeds.from_yahoo("NVDA", rng="60d", interval="5m")
    choice = intraday_select.select_best(df)
    # -> {"strategy": "vwap_reversion", "interval": "30m", "oos_sharpe": 0.4, ...}

Kept separate from intraday_strategies.py to avoid an import cycle (this imports
optimize, which imports strategies, which imports intraday_strategies).
"""

from __future__ import annotations

import json
import os
import time

from . import data, strategies, optimize

# interval label -> (resample rule or None for native 5m, bars/day for annualising)
INTERVALS = {"5m": (None, 78), "15m": ("15min", 26), "30m": ("30min", 13)}

STRATS = ["vwap_snap", "vwap_reversion", "opening_range", "intraday_momentum",
          "intraday_auto"]

# walk-forward grids (intraday_auto gets a small ADX grid)
GRIDS = dict(strategies.INTRADAY_GRIDS)
GRIDS["intraday_auto"] = {"adx_threshold": [20.0, 25.0, 30.0]}


def _bars(df, rule):
    return df if rule is None else data.resample_ohlcv(df, rule)


def evaluate(df, strats=None, intervals=None, capital=100_000.0, risk=0.004,
             stop_atr=1.5, target_atr=2.5, n_splits=3, train_frac=0.6,
             commission_bps=2.0, slippage_bps=2.0):
    """
    Walk-forward every (strategy, interval) on `df` (native 5m bars).

    Exits come from strategies.EXIT_PRESETS per strategy (falling back to the
    stop_atr/target_atr arguments), so what gets validated here is what the bot
    will actually trade. Pass FX-realistic costs (e.g. 0.3 + 0.2 bps) when the
    bars are FX from MT5 — stock-level costs (2+2 bps ~ 4-9 pips on EURUSD)
    wrongly kill high-frequency reversion edges.

    Returns a list of dicts sorted by OOS Sharpe (best first), each:
        {strategy, interval, oos_sharpe, oos_return, oos_maxdd, win_rate, folds}
    """
    strats = strats or STRATS
    intervals = intervals or list(INTERVALS)
    rows = []
    for name in strats:
        exits = strategies.EXIT_PRESETS.get(
            name, {"stop_atr": stop_atr, "target_atr": target_atr, "max_bars": 0})
        bt = {"risk_per_trade": risk, "flatten_eod": True,
              "commission_bps": commission_bps, "slippage_bps": slippage_bps,
              **exits}
        for label in intervals:
            rule, _ = INTERVALS[label]
            bars = _bars(df, rule)
            try:
                wf = optimize.walk_forward(bars, name, GRIDS[name],
                                           n_splits=n_splits, train_frac=train_frac,
                                           metric="sharpe", min_trades=3,
                                           capital=capital, bt_kwargs=bt)
            except Exception:
                continue
            st = wf.get("oos_stats")
            if not st:
                continue
            rows.append({
                "strategy": name,
                "interval": label,
                "oos_sharpe": float(st["OOS Sharpe"]),
                "oos_return": float(st["OOS Total Return"]),
                "oos_maxdd": float(st["OOS Max Drawdown"]),
                "folds": len(wf.get("folds", [])),
            })
    rows.sort(key=lambda r: r["oos_sharpe"], reverse=True)
    return rows


def select_best(df, min_sharpe=0.0, max_dd=-0.5, return_ranked=False, **kwargs):
    """
    Pick the single best (strategy, interval) for `df`, or None if nothing clears
    the bar. A choice must have OOS Sharpe >= `min_sharpe` and OOS max drawdown
    shallower than `max_dd` (e.g. -0.5 = don't accept worse than -50%).
    With ``return_ranked=True`` returns ``(choice, ranked)`` so callers can show
    the best rejected candidate when nothing clears.
    """
    ranked = evaluate(df, **kwargs)
    choice = None
    for r in ranked:
        if r["oos_sharpe"] >= min_sharpe and r["oos_maxdd"] >= max_dd:
            choice = r
            break
    return (choice, ranked) if return_ranked else choice


def select_cached(symbol, loader, cache_path="logs/select_cache.json",
                  max_age_hours=24.0, min_sharpe=0.0, **kwargs):
    """
    Cached selection: reuse a symbol's stored (strategy, interval) if it is fresh,
    else re-run the walk-forward and persist the result.

    Walk-forwarding every symbol on each startup is slow; caching lets the bot
    re-select only on a schedule. `loader()` returns the symbol's 5m DataFrame
    (called only on a cache miss / stale entry). Returns the choice dict (with a
    "cached" flag and "chosen_at" timestamp) or None if no edge cleared the bar.
    """
    cache = _load_cache(cache_path)
    entry = cache.get(symbol)
    now = time.time()
    if entry and (now - entry.get("chosen_at", 0)) < max_age_hours * 3600:
        entry = dict(entry); entry["cached"] = True
        return entry if entry.get("strategy") else None

    choice = select_best(loader(), min_sharpe=min_sharpe, **kwargs)
    record = dict(choice) if choice else {}
    record["chosen_at"] = now
    cache[symbol] = record
    _save_cache(cache_path, cache)
    if choice:
        choice = dict(choice); choice["cached"] = False; choice["chosen_at"] = now
    return choice


def _load_cache(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_cache(path, cache):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f, indent=2)
    os.replace(tmp, path)
