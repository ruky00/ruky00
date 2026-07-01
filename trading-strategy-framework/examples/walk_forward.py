"""
Rolling calendar walk-forward — optimise each strategy year by year.

For each strategy it optimises the indicator settings on a trailing window of
years and trades the NEXT year with those fixed params, rolling forward:

    train 2013-2014 -> test 2015
    train 2014-2015 -> test 2016
    train 2015-2016 -> test 2017   ...

The chained out-of-sample years are the honest result. This is far more robust
than a single in-sample backtest.

    python examples/walk_forward.py

Offline it uses the bundled 5-year samples (NVDA, AMD, ...). On your machine use
10y data for the full 2014->2021 schedule:
    from qflow import feeds; feeds.get("NVDA", "yahoo", rng="10y")
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

from qflow import feeds, optimize

SYMBOLS = ["NVDA", "AMD", "AMZN"]
SOURCE = "github"
TRAIN_YEARS = 2                 # bundled data is ~5y; use 4 with 10y data
BT = dict(capital=10_000.0, risk_per_trade=0.01)

# parameter grids per strategy (the indicator settings being optimised)
GRIDS = {
    "trend_following": {"fast": [20, 50], "slow": [100, 200], "adx_threshold": [15, 20, 25]},
    "mean_reversion": {"rsi_buy": [5, 10, 15], "rsi_exit": [55, 65], "pct_b_buy": [0.02, 0.05]},
    "volatility_breakout": {"channel": [40, 55], "exit_channel": [10, 20],
                            "squeeze_lookback": [15, 25]},
}


def main():
    for sym in SYMBOLS:
        df = feeds.get(sym, SOURCE)
        print(f"\n{'#'*70}\n# {sym}  ({df.index[0].date()} -> {df.index[-1].date()})\n{'#'*70}")
        for strat, grid in GRIDS.items():
            wf = optimize.rolling_walk_forward(
                df, strat, grid, train_years=TRAIN_YEARS, test_years=1,
                metric="sharpe_calmar", bt_kwargs=BT)
            print(f"\n=== {strat} ===")
            print(optimize.report_rolling(wf))

    print("\nRead the OOS Sharpe / OOS max drawdown lines — those are out-of-sample. "
          "A strategy whose params keep changing wildly across folds, or whose OOS "
          "goes negative, is fragile. Validate on 10y + several names, then paper-test.")


if __name__ == "__main__":
    main()
