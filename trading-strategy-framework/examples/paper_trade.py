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

from qflow.paper import PaperTrader, ALL_STRATEGIES


def main():
    ap = argparse.ArgumentParser(description="qflow paper trading")
    ap.add_argument("--strategy", default="mean_reversion",
                    choices=sorted(ALL_STRATEGIES))
    ap.add_argument("--symbol", default="AAPL")
    ap.add_argument("--source", default="github",
                    choices=["github", "binance", "stooq", "yahoo"])
    ap.add_argument("--interval", default=None, help="e.g. 1d, 1h (binance/yahoo)")
    ap.add_argument("--leader-symbol", default="",
                    help="for lead_lag: the asset that moves first (e.g. SAN.MC)")
    ap.add_argument("--leader-source", default="",
                    help="data source for the leader (defaults to --source)")
    ap.add_argument("--news", default="none",
                    choices=["none", "sample", "rss", "finnhub", "newsapi"],
                    help="live news overlay applied at --step (not in replay)")
    ap.add_argument("--finbert", action="store_true",
                    help="use FinBERT for sentiment (needs transformers+torch)")
    ap.add_argument("--kill-switches", action="store_true",
                    help="enable the risk governor with default limits")
    ap.add_argument("--max-daily-loss", type=float, default=0.03)
    ap.add_argument("--max-drawdown", type=float, default=0.15)
    ap.add_argument("--max-heat", type=float, default=0.06)
    ap.add_argument("--broker", default="none", choices=["none", "paper", "ibkr"],
                    help="route real orders on --step (ibkr needs TWS/IB Gateway)")
    ap.add_argument("--ibkr-port", type=int, default=4002,
                    help="4002 IB Gateway paper (default), 7497 TWS paper, "
                         "4001/7496 = LIVE")
    ap.add_argument("--ibkr-host", default="127.0.0.1")
    ap.add_argument("--ibkr-allow-live", action="store_true",
                    help="required to connect to a LIVE port (real money)")
    ap.add_argument("--ibkr-tif", default="GTC", choices=["GTC", "DAY"],
                    help="exit-leg time-in-force; DAY if your preset forces DAY")
    ap.add_argument("--broker-symbol", default="",
                    help="IBKR ticker if it differs from the data symbol "
                         "(e.g. data SAN.MC on yahoo -> broker SAN)")
    ap.add_argument("--currency", default="USD", help="order currency: USD, EUR ...")
    ap.add_argument("--exchange", default="SMART")
    ap.add_argument("--primary", default="",
                    help="primary exchange, e.g. BM for Bolsa de Madrid")
    ap.add_argument("--capital", type=float, default=10_000.0)
    ap.add_argument("--risk", type=float, default=0.01)
    ap.add_argument("--step", action="store_true", help="process the latest bar")
    ap.add_argument("--loop", action="store_true",
                    help="run automatically: call --step every --loop-interval seconds")
    ap.add_argument("--loop-interval", type=int, default=3600,
                    help="seconds between steps in --loop mode (default 3600 = 1h)")
    ap.add_argument("--replay", type=int, default=0,
                    help="fast-forward the last N bars (one simulated day each)")
    ap.add_argument("--report", action="store_true", help="print account status")
    ap.add_argument("--reset", action="store_true", help="wipe paper state")
    args = ap.parse_args()

    feed_kwargs = {}
    if args.interval:
        feed_kwargs["interval"] = args.interval

    news_provider = None
    if args.news != "none":
        from qflow import news as newsmod
        if args.finbert:
            newsmod.set_scorer(newsmod.FinBERTScorer())
        news_provider = {
            "sample": newsmod.SampleProvider,
            "rss": newsmod.RSSProvider,
            "finnhub": newsmod.FinnhubProvider,
            "newsapi": newsmod.NewsAPIProvider,
        }[args.news]()

    risk_limits = None
    if args.kill_switches:
        risk_limits = {"max_daily_loss": args.max_daily_loss,
                       "max_drawdown": args.max_drawdown,
                       "max_portfolio_heat": args.max_heat}

    broker_obj = None
    if args.broker == "paper":
        from qflow import broker as brk
        broker_obj = brk.PaperBroker(cash=args.capital)
    elif args.broker == "ibkr":
        from qflow import broker as brk
        broker_obj = brk.IBKRBroker(host=args.ibkr_host, port=args.ibkr_port,
                                    allow_live=args.ibkr_allow_live,
                                    currency=args.currency, exchange=args.exchange,
                                    primary_exchange=args.primary, tif=args.ibkr_tif)

    kw = dict(symbol=args.symbol, source=args.source, strategy=args.strategy,
              capital=args.capital, risk_per_trade=args.risk,
              feed_kwargs=feed_kwargs, leader_symbol=args.leader_symbol,
              leader_source=args.leader_source, news_provider=news_provider,
              risk_limits=risk_limits, broker=broker_obj,
              broker_symbol=args.broker_symbol)

    if args.reset:
        PaperTrader(**kw).reset()
        print("State reset.")
    pt = PaperTrader(**kw)

    if args.replay:
        print(pt.replay(args.replay))

    if args.loop:
        import time
        from datetime import datetime
        print(f"AUTO mode: stepping every {args.loop_interval}s. Keep IB Gateway open. "
              "Ctrl+C to stop.")
        try:
            while True:
                stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                try:
                    print(f"[{stamp}] {pt.step()}")
                except Exception as e:      # never let one bad tick kill the loop
                    print(f"[{stamp}] step error: {type(e).__name__}: {e}")
                # ib.sleep keeps the IBKR connection alive during the wait
                w = max(30, args.loop_interval)
                if pt.broker is not None and getattr(pt.broker, "name", "") == "ibkr" \
                        and pt.broker.is_connected():
                    pt.broker.ib.sleep(w)
                else:
                    time.sleep(w)
        except KeyboardInterrupt:
            print("\nStopped by user.")
    elif args.step:
        print(pt.step())

    # always show status at the end
    print(pt.report())


if __name__ == "__main__":
    main()
