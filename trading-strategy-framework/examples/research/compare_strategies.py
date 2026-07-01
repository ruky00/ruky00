"""
Backtest and compare every strategy on REAL market data.

Loads bundled real daily OHLCV (AAPL, TSLA — fetched from public datasets and
cached under data/), runs all three strategies plus a buy & hold benchmark, and
ranks them so you can see which edge actually holds up out of sample.

    python examples/compare_strategies.py

On an open network you can swap in live data, e.g.:
    feeds.get("BTCUSDT", "binance", interval="1d")
    feeds.get("AAPL", "yahoo", rng="10y")
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from qflow import feeds, strategies, backtest, metrics

CAPITAL = 10_000.0
RISK = 0.01
ASSETS = ["AAPL", "TSLA"]          # bundled real samples (offline-safe)


def buy_and_hold(df, capital=CAPITAL):
    eq = capital * df["close"] / df["close"].iloc[0]
    eq.name = "equity"
    rets = eq.pct_change().fillna(0.0)
    return eq, rets


def run_asset(symbol):
    df = feeds.get(symbol, source="github")
    print(f"\n{'#'*70}\n# {symbol}  —  {len(df)} bars  "
          f"{df.index[0].date()} -> {df.index[-1].date()}\n{'#'*70}")

    rows = []

    # benchmark
    eq, rets = buy_and_hold(df)
    s = metrics.summary(eq, rets)
    rows.append(("buy_and_hold", s, 0))

    # strategies
    for name, fn in strategies.REGISTRY.items():
        sig = fn(df)
        res = backtest.run_backtest(df, sig.signal, sig.atr,
                                    capital=CAPITAL, risk_per_trade=RISK)
        rows.append((name, res.stats(), len(res.trades)))

    # comparison table
    hdr = f"{'strategy':<20}{'CAGR':>9}{'Sharpe':>9}{'Sortino':>9}{'MaxDD':>9}{'Win%':>8}{'PF':>7}{'Trades':>8}"
    print(hdr)
    print("-" * len(hdr))
    for name, st, ntr in rows:
        print(f"{name:<20}"
              f"{st.get('CAGR',0)*100:>8.1f}%"
              f"{st.get('Sharpe',0):>9.2f}"
              f"{st.get('Sortino',0):>9.2f}"
              f"{st.get('Max Drawdown',0)*100:>8.1f}%"
              f"{st.get('Win Rate',0)*100:>7.0f}%"
              f"{st.get('Profit Factor',0):>7.2f}"
              f"{ntr:>8}")

    # rank strategies (exclude benchmark) by a blended score: Sharpe + Calmar
    ranked = sorted(
        [(n, s) for n, s, _ in rows if n != "buy_and_hold"],
        key=lambda x: x[1].get("Sharpe", 0) + x[1].get("Calmar", 0),
        reverse=True,
    )
    best = ranked[0]
    print(f"\n>> Best on {symbol}: {best[0]}  "
          f"(Sharpe {best[1].get('Sharpe',0):.2f}, "
          f"MaxDD {best[1].get('Max Drawdown',0)*100:.1f}%)")
    return {symbol: {n: s for n, s, _ in rows}}


def main():
    print("Multi-strategy comparison on REAL data (capital $10k, 1% risk/trade)")
    all_stats = {}
    for sym in ASSETS:
        all_stats.update(run_asset(sym))

    # overall recommendation: average Sharpe across assets per strategy
    print(f"\n{'='*70}\nOVERALL — average Sharpe across assets\n{'='*70}")
    strat_names = list(strategies.REGISTRY.keys())
    avg = {}
    for name in strat_names:
        sh = [all_stats[a][name].get("Sharpe", 0) for a in ASSETS]
        avg[name] = sum(sh) / len(sh)
    for name, v in sorted(avg.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name:<22} avg Sharpe {v:>6.2f}")
    winner = max(avg, key=avg.get)
    print(f"\n>> Candidate strategy to forward-test on paper: {winner}")
    print("   Next step:  python examples/paper_trade.py --strategy", winner)


if __name__ == "__main__":
    main()
