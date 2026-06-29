"""
Paper-trading CLI — forward-test a strategy with virtual money.

Daily workflow (run once per day after the close, e.g. from cron):
    python examples/paper_trade.py --strategy mean_reversion --symbol AAPL --step
    python examples/paper_trade.py --strategy mean_reversion --symbol AAPL --report

Preview the whole 2-week workflow right now (replays recent history as if one
bar arrived per day):
    python examples/paper_trade.py --strategy mean_reversion --symbol AAPL --replay 15

Crypto / live data on an open network:
    python examples/paper_trade.py --strategy trend_following --symbol BTCUSDT \
        --source binance --interval 1d --step

State persists under data/paper/<symbol>_<strategy>/ between runs.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

from qflow.paper import PaperTrader
from qflow import strategies


def main():
    ap = argparse.ArgumentParser(description="qflow paper trading")
    ap.add_argument("--strategy", default="mean_reversion",
                    choices=list(strategies.REGISTRY))
    ap.add_argument("--symbol", default="AAPL")
    ap.add_argument("--source", default="github",
                    choices=["github", "binance", "stooq", "yahoo"])
    ap.add_argument("--interval", default=None, help="e.g. 1d, 1h (binance/yahoo)")
    ap.add_argument("--capital", type=float, default=10_000.0)
    ap.add_argument("--risk", type=float, default=0.01)
    ap.add_argument("--step", action="store_true", help="process the latest bar")
    ap.add_argument("--replay", type=int, default=0,
                    help="fast-forward the last N bars (one simulated day each)")
    ap.add_argument("--report", action="store_true", help="print account status")
    ap.add_argument("--reset", action="store_true", help="wipe paper state")
    args = ap.parse_args()

    feed_kwargs = {}
    if args.interval:
        feed_kwargs["interval"] = args.interval

    pt = PaperTrader(
        symbol=args.symbol, source=args.source, strategy=args.strategy,
        capital=args.capital, risk_per_trade=args.risk, feed_kwargs=feed_kwargs,
    )

    if args.reset:
        pt.reset()
        pt = PaperTrader(symbol=args.symbol, source=args.source,
                         strategy=args.strategy, capital=args.capital,
                         risk_per_trade=args.risk, feed_kwargs=feed_kwargs)
        print("State reset.")

    if args.replay:
        print(pt.replay(args.replay))
    if args.step:
        print(pt.step())

    # always show status at the end
    print(pt.report())


if __name__ == "__main__":
    main()
