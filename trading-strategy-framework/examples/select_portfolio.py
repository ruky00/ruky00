"""
Build the final portfolio automatically — walk-forward-filter a basket.

Runs a rolling walk-forward for every (symbol, strategy) pair and keeps only the
robust ones (OOS Sharpe, OOS drawdown, parameter stability, positive-fold ratio),
then recommends the single best surviving strategy per symbol. Symbols where
nothing survives are dropped.

    python examples/select_portfolio.py

Offline it uses the bundled samples. On your machine, scan your real basket with
10y data:
    from qflow import portfolio_selector as sel
    res = sel.select(["TSLA","NVDA","AMD","COIN","PLTR","AMZN","NFLX"],
                     source="yahoo", train_years=4)
    print(sel.report(res))
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

from qflow import portfolio_selector as sel

BASKET = ["NVDA", "AMD", "AMZN", "NFLX", "MSFT", "GOOGL"]
SOURCE = "github"


def main():
    print(f"Scanning {len(BASKET)} symbols x 3 strategies with rolling walk-forward ...\n")
    res = sel.select(
        BASKET, source=SOURCE, train_years=2,          # use 4 with 10y data
        thresholds={"min_oos_sharpe": 0.5, "max_oos_drawdown": -0.20,
                    "min_stability": 0.5, "min_pos_ratio": 0.6, "min_folds": 3},
        bt_kwargs={"capital": 10_000.0, "risk_per_trade": 0.01},
        verbose=True,
    )
    print("\n" + sel.report(res))

    if res["portfolio"]:
        syms = ",".join(res["portfolio"])
        print(f"\nNext: paper-test the survivors, e.g.\n"
              f"  python examples/portfolio_run.py --symbols {syms} --source yahoo "
              f"--strategy auto --kill-switches --broker ibkr --ibkr-port 4002 --loop")


if __name__ == "__main__":
    main()
