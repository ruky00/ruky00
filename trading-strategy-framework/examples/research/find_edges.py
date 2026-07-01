"""
Daily-edge research + optimisation, end to end on REAL data.

    python examples/find_edges.py

It does three things:
  1. Scans each asset for daily anomalies (overnight/intraday, gaps, day-of-week,
     autocorrelation) and flags statistically meaningful ones.
  2. Backtests the tradeable versions of those anomalies *net of costs*, plus a
     cross-market lead-lag trade (one market leading another by a day).
  3. Walk-forward optimises the mean-reversion strategy (out-of-sample).

Swap in your own instruments with feeds.get(...). The lead-lag block is the
generalisation of "stock A in one market leads stock B in another".
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import warnings
warnings.filterwarnings("ignore")

from qflow import feeds, anomalies as A, optimize, metrics

CAPITAL = 10_000.0


def fmt(name, s):
    return (f"{name:<28} CAGR {s['CAGR']*100:>6.1f}%  Sharpe {s['Sharpe']:>5.2f}  "
            f"MaxDD {s['Max Drawdown']*100:>6.1f}%  days {s['trading_days']}")


def main():
    aapl = feeds.get("AAPL", "github")
    tsla = feeds.get("TSLA", "github")

    # 1) descriptive scans
    for sym, df in (("AAPL", aapl), ("TSLA", tsla)):
        print(A.report(A.scan(df), sym))
        print()

    # 2) tradeable anomaly backtests (net of costs)
    print("=" * 64)
    print("TRADEABLE ANOMALY STRATEGIES (net of 4 bps/side)")
    print("=" * 64)
    results = {
        "TSLA gap-fade": A.backtest_gap_fade(tsla),
        "AAPL gap-fade": A.backtest_gap_fade(aapl),
        "TSLA overnight-only": A.backtest_overnight(tsla),
        "AAPL->TSLA lead-lag(1d)": A.backtest_lead_lag(aapl, tsla, lag=1, threshold=0.005),
        "TSLA->AAPL lead-lag(1d)": A.backtest_lead_lag(tsla, aapl, lag=1, threshold=0.005),
    }
    for name, s in sorted(results.items(), key=lambda kv: kv[1]["Sharpe"], reverse=True):
        print(fmt(name, s))

    best = max(results, key=lambda k: results[k]["Sharpe"])
    print(f"\n>> Strongest cost-aware edge on this sample: {best} "
          f"(Sharpe {results[best]['Sharpe']:.2f})")

    # 3) walk-forward optimisation of mean-reversion (out-of-sample)
    print("\n" + "=" * 64)
    print("WALK-FORWARD OPTIMISATION — mean_reversion on TSLA")
    print("=" * 64)
    grid = {"rsi_buy": [5, 10, 15], "rsi_exit": [50, 55, 65], "trend_window": [150, 200]}
    wf = optimize.walk_forward(tsla, "mean_reversion", grid, n_splits=3,
                               train_frac=0.5,
                               bt_kwargs=dict(capital=CAPITAL, risk_per_trade=0.01))
    print(optimize.report_walk_forward(wf))

    print("\nNOTE: in-sample edges are hypotheses. Forward-test the winner with "
          "paper money (examples/paper_trade.py) before risking capital.")


if __name__ == "__main__":
    main()
