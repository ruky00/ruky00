"""
Backtest the INTRADAY strategies on 5-minute bars (flat overnight).

This is what validates the funded bot's edge: the bot trades qflow strategies on
5-minute bars, so we backtest those same strategies on intraday bars with
end-of-day flattening (nothing held overnight) and intraday-scaled metrics.

    python examples/research/backtest_intraday.py

Offline it uses a synthetic 5-minute series so it runs anywhere. On your machine,
swap in real 5m bars (up to ~60 days from Yahoo):
    from qflow import feeds
    df = feeds.from_yahoo("NVDA", rng="60d", interval="5m")
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import warnings
warnings.filterwarnings("ignore")

import numpy as np

from qflow import data, strategies, backtest, metrics

CAPITAL = 100_000.0
RISK = 0.002
BARS_PER_YEAR = 252 * 78          # 5-min bars in a trading year (annualisation)


def intraday_stats(res, n_bars):
    eq, rets = res.equity, res.returns
    ann = np.sqrt(BARS_PER_YEAR)
    sharpe = float(rets.mean() / rets.std(ddof=0) * ann) if rets.std(ddof=0) else 0.0
    n_days = len({t.date() for t in eq.index})
    return {
        "TotalReturn": metrics.total_return(eq),
        "Sharpe(ann)": sharpe,
        "MaxDD": metrics.max_drawdown(eq),
        "Trades": len(res.trades),
        "Trades/day": len(res.trades) / max(1, n_days),
        "Win%": metrics.win_rate(res.trade_pnls),
    }


def main():
    df = data.synthetic_intraday(n_days=120, seed=7)   # ~9360 5-min bars
    print(f"Intraday backtest — {len(df)} 5-min bars over "
          f"{len({t.date() for t in df.index})} sessions (flat overnight)\n")
    hdr = f"{'strategy':<20}{'Return':>9}{'Sharpe':>9}{'MaxDD':>9}{'Trades':>8}{'/day':>7}{'Win%':>7}"
    print(hdr); print("-" * len(hdr))
    for name in ["mean_reversion", "trend_following", "volatility_breakout", "auto"]:
        sig = strategies.REGISTRY[name](df)
        res = backtest.run_backtest(df, sig.signal, sig.atr, capital=CAPITAL,
                                    risk_per_trade=RISK, stop_atr=1.5, target_atr=2.5,
                                    flatten_eod=True)          # <-- intraday: no overnight
        s = intraday_stats(res, len(df))
        print(f"{name:<20}{s['TotalReturn']*100:>8.1f}%{s['Sharpe(ann)']:>9.2f}"
              f"{s['MaxDD']*100:>8.1f}%{s['Trades']:>8}{s['Trades/day']:>7.1f}"
              f"{s['Win%']*100:>6.0f}%")

    print("\nNote: synthetic data — for illustration of the machinery only. Run on "
          "real 5m bars (feeds.from_yahoo interval='5m') for the names the bot will "
          "trade, then walk-forward + select before trusting an intraday edge.")


if __name__ == "__main__":
    main()
