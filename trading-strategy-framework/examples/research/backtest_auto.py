"""
Backtest the adaptive 'auto' strategy vs the individual strategies + buy & hold.

    python examples/backtest_auto.py

Offline it uses the bundled real samples (AAPL, TSLA). On your machine, edit
ASSETS and use live data for volatile names:

    from qflow import feeds
    df = feeds.get("NVDA", "yahoo", rng="10y")   # or TSLA, AMD, COIN, PLTR ...
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import warnings
warnings.filterwarnings("ignore")

from qflow import feeds, strategies, backtest, metrics

CAPITAL = 10_000.0
RISK = 0.01
ASSETS = ["AAPL", "TSLA"]          # bundled; swap for volatile names on your machine
SOURCE = "github"


def bh(df):
    eq = CAPITAL * df["close"] / df["close"].iloc[0]
    return metrics.summary(eq, eq.pct_change().fillna(0.0))


def main():
    for sym in ASSETS:
        df = feeds.get(sym, SOURCE)
        print(f"\n{'='*74}\n{sym}  —  {len(df)} bars  "
              f"{df.index[0].date()} -> {df.index[-1].date()}\n{'='*74}")
        hdr = f"{'strategy':<20}{'CAGR':>8}{'Sharpe':>8}{'Sortino':>8}{'MaxDD':>8}{'Win%':>7}{'Trades':>7}"
        print(hdr); print("-" * len(hdr))

        # buy & hold benchmark
        s = bh(df)
        print(f"{'buy_and_hold':<20}{s['CAGR']*100:>7.1f}%{s['Sharpe']:>8.2f}"
              f"{s['Sortino']:>8.2f}{s['Max Drawdown']*100:>7.1f}%{'-':>7}{'-':>7}")

        # individual strategies + auto
        for name in ["trend_following", "mean_reversion", "volatility_breakout", "auto"]:
            sig = strategies.REGISTRY[name](df)
            res = backtest.run_backtest(df, sig.signal, sig.atr,
                                        capital=CAPITAL, risk_per_trade=RISK)
            st = res.stats()
            tag = ">> " if name == "auto" else "   "
            print(f"{tag}{name:<17}{st['CAGR']*100:>7.1f}%{st['Sharpe']:>8.2f}"
                  f"{st['Sortino']:>8.2f}{st['Max Drawdown']*100:>7.1f}%"
                  f"{st.get('Win Rate',0)*100:>6.0f}%{st.get('Trades',0):>7}")

        # which strategy the adaptive model used
        chosen = strategies.adaptive(df).chosen.value_counts(normalize=True)
        usage = "  ".join(f"{k} {v*100:.0f}%" for k, v in chosen.items())
        print(f"\nauto strategy usage: {usage}")

    print("\nNote: 'auto' should be steadier than any single strategy across "
          "regimes, not necessarily the highest return. Validate on 10y data for "
          "the volatile names you actually intend to trade, then paper-test.")


if __name__ == "__main__":
    main()
