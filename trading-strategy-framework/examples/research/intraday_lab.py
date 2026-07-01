"""
Intraday strategy lab — pick the candle interval, then walk-forward the edge.

This is the intraday counterpart of the daily research pipeline. It takes the
*dedicated* intraday strategies (VWAP reversion, opening-range breakout,
intraday momentum, and the intraday_auto combiner) and does three things:

  1. INTERVAL SWEEP — backtests each strategy on 5m / 15m / 30m bars (resampling
     one 5-minute series so the comparison is apples-to-apples) with end-of-day
     flattening, and reports intraday-scaled Sharpe / trades-per-day / win%.
     This answers "qué intervalo de velas conviene".

  2. WALK-FORWARD — for the best interval, runs an out-of-sample walk-forward
     (optimise params on a training window, trade the next unseen window) so the
     number you trust is OOS, not the in-sample fit.

  3. COMBINE — reports intraday_auto, which switches between opening-range
     breakout (trending days, high ADX) and VWAP reversion (choppy days), i.e.
     the "combínalas" option.

    python examples/research/intraday_lab.py

Offline it uses a synthetic 5-minute series so it runs anywhere. On your own
machine, swap in real bars (up to ~60 days of 5m from Yahoo):

    from qflow import feeds
    df = feeds.from_yahoo("NVDA", rng="60d", interval="5m")

then resample to 15m/30m with data.resample_ohlcv. Synthetic numbers are for
illustrating the machinery only — validate on real bars before trusting an edge.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import warnings
warnings.filterwarnings("ignore")

import numpy as np

from qflow import data, strategies, backtest, metrics, optimize

CAPITAL = 100_000.0
RISK = 0.002
STOP_ATR, TARGET_ATR = 1.5, 2.5

# 5m -> None (native), else a pandas offset alias. bars/day drives annualisation.
INTERVALS = {
    "5m":  (None,     78),
    "15m": ("15min",  26),
    "30m": ("30min",  13),
}

STRATS = ["vwap_reversion", "opening_range", "intraday_momentum", "intraday_auto"]

# Walk-forward grids (intraday_auto gets a small ADX-threshold grid here).
GRIDS = dict(strategies.INTRADAY_GRIDS)
GRIDS["intraday_auto"] = {"adx_threshold": [20.0, 25.0, 30.0]}


def _bars_for(df, rule):
    return df if rule is None else data.resample_ohlcv(df, rule)


def _stats(res, bars_per_day):
    eq, rets = res.equity, res.returns
    ann = np.sqrt(252 * bars_per_day)
    sd = rets.std(ddof=0)
    sharpe = float(rets.mean() / sd * ann) if sd else 0.0
    n_days = max(1, len({t.date() for t in eq.index}))
    return {
        "ret": metrics.total_return(eq),
        "sharpe": sharpe,
        "maxdd": metrics.max_drawdown(eq),
        "trades": len(res.trades),
        "per_day": len(res.trades) / n_days,
        "win": metrics.win_rate(res.trade_pnls),
    }


def interval_sweep(df):
    """Backtest every strategy on every interval; return {(strat, interval): stats}."""
    print("=" * 74)
    print("1) INTERVAL SWEEP — same strategy, different candle (flat overnight)")
    print("=" * 74)
    hdr = f"{'strategy':<18}{'interval':<9}{'Return':>9}{'Sharpe':>8}{'MaxDD':>8}{'Trades':>8}{'/day':>7}{'Win%':>6}"
    out = {}
    for name in STRATS:
        print(hdr if name == STRATS[0] else "")
        print("-" * len(hdr))
        for label, (rule, bpd) in INTERVALS.items():
            bars = _bars_for(df, rule)
            sig = strategies.REGISTRY[name](bars)
            res = backtest.run_backtest(bars, sig.signal, sig.atr, capital=CAPITAL,
                                        risk_per_trade=RISK, stop_atr=STOP_ATR,
                                        target_atr=TARGET_ATR, flatten_eod=True)
            s = _stats(res, bpd)
            out[(name, label)] = s
            print(f"{name:<18}{label:<9}{s['ret']*100:>8.1f}%{s['sharpe']:>8.2f}"
                  f"{s['maxdd']*100:>7.1f}%{s['trades']:>8}{s['per_day']:>7.1f}"
                  f"{s['win']*100:>5.0f}%")
    return out


def best_interval(sweep):
    """Best (highest Sharpe) interval per strategy from the in-sample sweep."""
    best = {}
    for name in STRATS:
        cand = {lbl: sweep[(name, lbl)]["sharpe"] for lbl in INTERVALS}
        best[name] = max(cand, key=cand.get)
    return best


def walk_forward_best(df, best):
    """Out-of-sample walk-forward on each strategy's chosen interval."""
    print("\n" + "=" * 74)
    print("2) WALK-FORWARD — out-of-sample on each strategy's best interval")
    print("=" * 74)
    hdr = f"{'strategy':<18}{'interval':<9}{'OOS Sharpe':>12}{'OOS Return':>12}{'OOS MaxDD':>11}"
    print(hdr); print("-" * len(hdr))
    results = {}
    for name in STRATS:
        label = best[name]
        rule, _ = INTERVALS[label]
        bars = _bars_for(df, rule)
        wf = optimize.walk_forward(bars, name, GRIDS[name], n_splits=3,
                                   train_frac=0.6, metric="sharpe",
                                   min_trades=3, capital=CAPITAL,
                                   bt_kwargs={"risk_per_trade": RISK,
                                              "stop_atr": STOP_ATR,
                                              "target_atr": TARGET_ATR,
                                              "flatten_eod": True})
        results[name] = wf
        if "oos_stats" in wf:
            st = wf["oos_stats"]
            print(f"{name:<18}{label:<9}{st['OOS Sharpe']:>12.2f}"
                  f"{st['OOS Total Return']*100:>11.1f}%{st['OOS Max Drawdown']*100:>10.1f}%")
        else:
            print(f"{name:<18}{label:<9}{'n/a — ' + wf.get('error',''):>12}")
    return results


def main():
    df = data.synthetic_intraday(n_days=180, seed=11)
    print(f"Intraday lab — {len(df)} native 5-min bars over "
          f"{len({t.date() for t in df.index})} sessions\n")

    sweep = interval_sweep(df)
    best = best_interval(sweep)

    print("\nBest interval per strategy (by in-sample Sharpe):")
    for name in STRATS:
        print(f"  {name:<18} -> {best[name]}")

    walk_forward_best(df, best)

    print("\n" + "=" * 74)
    print("3) COMBINE — intraday_auto switches breakout (trend) vs VWAP-reversion "
          "(chop)")
    print("=" * 74)
    print("   See its row above: on trending days it trades the opening-range "
          "breakout,\n   on choppy days it fades VWAP — one strategy that adapts "
          "to the session.")

    print("\nNote: SYNTHETIC data — illustrates the pipeline, not a real edge. "
          "Re-run on\nreal 5m bars (feeds.from_yahoo interval='5m', resample_ohlcv "
          "for 15m/30m) for\nthe names the bot will trade, then trust the "
          "WALK-FORWARD (OOS) column only.")


if __name__ == "__main__":
    main()
