"""
Run a strategy through Interactive Brokers — PAPER account by default.

PREREQUISITES (on your own machine — not in this sandbox):
  1. pip install ib_insync
  2. Open IB Gateway (or TWS), log in to your PAPER account.
  3. Enable the API:  Configure > Settings > API > Settings >
        [x] Enable ActiveX and Socket Clients
        [ ] Read-Only API   (leave UNCHECKED so it can place orders)
        Socket port = 4002   (IB Gateway paper)   # TWS paper = 7497
  4. Run this script.

PORTS:  IB Gateway paper 4002 · TWS paper 7497 · Gateway live 4001 · TWS live 7496

SAFETY
  * Defaults to the paper port 7497. The live ports (7496/4001) require
    allow_live=True — a deliberate friction.
  * Replay never sends orders; only the live `step()` path does.
  * Keep the risk governor (kill-switches) enabled.

This places native bracket orders: every entry carries a stop-loss and a
take-profit that live on IBKR's servers (honoured even if this script dies).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import warnings
warnings.filterwarnings("ignore")

from qflow.paper import PaperTrader
from qflow import broker, news


def main():
    # 1) the broker — IB Gateway PAPER port 4002 (use 7497 for TWS paper)
    ib = broker.IBKRBroker(host="127.0.0.1", port=4002, client_id=1)

    # 2) the strategy account, with kill-switches and a news overlay
    pt = PaperTrader(
        symbol="AAPL",
        source="yahoo",                 # live data on your machine
        strategy="mean_reversion",
        capital=10_000.0,
        risk_per_trade=0.005,           # start small: 0.5% per trade
        risk_limits={"max_daily_loss": 0.02, "max_drawdown": 0.08,
                     "max_portfolio_heat": 0.04},
        news_provider=news.RSSProvider(),
        broker=ib,                       # <-- real execution
    )

    # 3) one daily step (run this from cron after the close). It will:
    #    fetch data -> signal -> news gate -> kill-switch gate -> bracket order
    print(pt.step())
    print(pt.report())

    ib.disconnect()


if __name__ == "__main__":
    main()
