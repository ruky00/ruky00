"""
Hunt for REPEATABLE daily edges — patterns that recur year after year.

    python examples/repeatable_edges.py

For each candidate pattern it reports significance (t/p with a multiple-testing
bar), consistency (share of calendar years positive), and out-of-sample
persistence (Sharpe in each half of history). A pattern is only worth
paper-testing if it is consistent AND survives both halves AND beats costs.

Swap in your own instruments + more history for stronger statistics:
    from qflow import feeds
    df     = feeds.get("AAPL",   "yahoo", rng="10y")
    leader = feeds.get("SAN.MC", "yahoo", rng="10y")   # your España -> US idea
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import warnings
warnings.filterwarnings("ignore")

from qflow import feeds, edge_lab


def main():
    aapl = feeds.get("AAPL", "github")
    tsla = feeds.get("TSLA", "github")

    print(edge_lab.report(edge_lab.scan(tsla, leader=aapl), "TSLA (leader AAPL)"))
    print()
    print(edge_lab.report(edge_lab.scan(aapl, leader=tsla), "AAPL (leader TSLA)"))

    print("\nTip: only ~2-4 years of bundled data here, so 'consistency' is "
          "coarse. Re-run on your own machine with 10y data (feeds.get(sym, "
          "'yahoo', rng='10y')) for statistics you can actually trust.")


if __name__ == "__main__":
    main()
