"""
Wide-universe + dual-listing edge hunt.

    python examples/scan_universe.py

OFFLINE here it can only reach the two bundled github samples, so it demos with
those. On your own machine (open network) point it at a real universe:

    from qflow import universe, dual_listing

    # (2) scan the whole IBEX-35 / S&P-500 for repeatable patterns
    res = universe.scan_universe(universe.IBEX35, source="yahoo")
    print(universe.report(res))

    # (3) the España <-> US dual-listing lead-lag (Santander, BBVA, Telefonica)
    dl = dual_listing.scan_dual_listings(source="yahoo")
    print(dual_listing.report(dl))
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

from qflow import feeds, universe, dual_listing

OPEN_NETWORK = False     # set True on your machine to use the real universes


def main():
    if OPEN_NETWORK:
        print("Scanning IBEX-35 ...")
        res = universe.scan_universe(universe.IBEX35, source="yahoo", verbose=True)
        print(universe.report(res))
        print("\nDual-listing España <-> US ...")
        print(dual_listing.report(dual_listing.scan_dual_listings(source="yahoo")))
        return

    # offline demo with the bundled samples
    print("(offline demo — bundled AAPL/TSLA. Set OPEN_NETWORK=True for real data.)\n")
    res = universe.scan_universe(["AAPL", "TSLA"], source="github",
                                 min_consistency=0.5)
    print(universe.report(res))

    print("\n# Dual-listing analysis (AAPL/TSLA used as a stand-in pair):")
    aapl = feeds.get("AAPL", "github")
    tsla = feeds.get("TSLA", "github")
    r = dual_listing.analyze_pair(aapl, tsla, "AAPL/TSLA")
    b = r["best"]
    print(f"  strongest direction: {b['name']}  Sharpe {b['sharpe']:.2f}  "
          f"consistency {b['consistency']:.0%}  OOS2 {b['oos_sharpe_2h']:.2f}")
    print("\n  -> On your machine, replace with real pairs:")
    print("     dual_listing.scan_dual_listings(source='yahoo')  # SAN/BBVA/TEF ...")


if __name__ == "__main__":
    main()
