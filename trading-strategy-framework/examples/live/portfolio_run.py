"""
Grouped execution — run the bot across a BASKET of symbols at once.

One command trades several volatile names, each with the adaptive `auto` strategy
(which picks the best sub-strategy per candle), all sharing ONE IBKR session, with
per-symbol kill-switches plus a portfolio-level drawdown circuit breaker.

Preview offline (bundled samples, simulated broker):
    python examples/portfolio_run.py --symbols AAPL,TSLA --source github \
        --broker paper --reset --replay 300 --kill-switches

Live daily on your machine (IB Gateway paper, run after the US close or loop):
    python examples/portfolio_run.py --symbols TSLA,NVDA,AMD,COIN,PLTR --source yahoo \
        --strategy auto --risk 0.005 --kill-switches \
        --broker ibkr --ibkr-port 4002 --news rss --loop --loop-interval 3600
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import warnings
warnings.filterwarnings("ignore")

from qflow.portfolio_runner import PortfolioRunner


def main():
    ap = argparse.ArgumentParser(description="qflow multi-symbol portfolio runner")
    ap.add_argument("--symbols", default="TSLA,NVDA,AMD,COIN,PLTR",
                    help="comma-separated basket")
    ap.add_argument("--strategy", default="auto")
    ap.add_argument("--source", default="yahoo",
                    choices=["github", "binance", "stooq", "yahoo"])
    ap.add_argument("--interval", default=None, help="feed interval, e.g. 1d")
    ap.add_argument("--total-capital", type=float, default=10_000.0)
    ap.add_argument("--risk", type=float, default=0.005)
    ap.add_argument("--allocation", default="equal", choices=["equal", "inverse_vol"])
    # broker
    ap.add_argument("--broker", default="none", choices=["none", "paper", "ibkr"])
    ap.add_argument("--ibkr-port", type=int, default=4002)
    ap.add_argument("--ibkr-host", default="127.0.0.1")
    ap.add_argument("--ibkr-client-id", type=int, default=1,
                    help="unique per running bot (use 2,3,... to run several)")
    ap.add_argument("--ibkr-allow-live", action="store_true")
    ap.add_argument("--currency", default="USD")
    ap.add_argument("--exchange", default="SMART")
    ap.add_argument("--primary", default="")
    ap.add_argument("--ibkr-tif", default="GTC", choices=["GTC", "DAY"],
                    help="exit-leg time-in-force; use DAY if your account preset "
                         "forces DAY (avoids error 10349 cancellations)")
    # risk
    ap.add_argument("--kill-switches", action="store_true")
    ap.add_argument("--max-daily-loss", type=float, default=0.03)
    ap.add_argument("--max-drawdown", type=float, default=0.15)
    ap.add_argument("--max-heat", type=float, default=0.06)
    ap.add_argument("--portfolio-max-drawdown", type=float, default=0.15)
    ap.add_argument("--max-correlation", type=float, default=0.85,
                    help="block a new entry if it correlates above this with an open name")
    # news
    ap.add_argument("--news", default="none",
                    choices=["none", "sample", "rss", "finnhub", "newsapi"])
    ap.add_argument("--finbert", action="store_true")
    # actions
    ap.add_argument("--step", action="store_true")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--loop-interval", type=int, default=3600)
    ap.add_argument("--replay", type=int, default=0)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    feed_kwargs = {"interval": args.interval} if args.interval else {}

    news_provider = None
    if args.news != "none":
        from qflow import news as newsmod
        if args.finbert:
            newsmod.set_scorer(newsmod.FinBERTScorer())
        news_provider = {"sample": newsmod.SampleProvider, "rss": newsmod.RSSProvider,
                         "finnhub": newsmod.FinnhubProvider,
                         "newsapi": newsmod.NewsAPIProvider}[args.news]()

    broker_obj = None
    if args.broker == "paper":
        from qflow import broker as brk
        broker_obj = brk.PaperBroker(cash=args.total_capital)
    elif args.broker == "ibkr":
        from qflow import broker as brk
        broker_obj = brk.IBKRBroker(host=args.ibkr_host, port=args.ibkr_port,
                                    client_id=args.ibkr_client_id,
                                    allow_live=args.ibkr_allow_live,
                                    currency=args.currency, exchange=args.exchange,
                                    primary_exchange=args.primary, tif=args.ibkr_tif)

    risk_limits = None
    if args.kill_switches:
        risk_limits = {"max_daily_loss": args.max_daily_loss,
                       "max_drawdown": args.max_drawdown,
                       "max_portfolio_heat": args.max_heat}

    if args.reset:
        print("Portfolio state reset.")
    runner = PortfolioRunner(
        symbols, strategy=args.strategy, source=args.source,
        total_capital=args.total_capital, risk_per_trade=args.risk,
        allocation=args.allocation, broker=broker_obj, news_provider=news_provider,
        risk_limits=risk_limits, feed_kwargs=feed_kwargs,
        currency=args.currency, exchange=args.exchange, primary=args.primary,
        portfolio_max_drawdown=args.portfolio_max_drawdown,
        max_correlation=args.max_correlation, reset=args.reset,
    )

    if args.replay:
        for r in runner.replay_all(args.replay):
            print(r)

    if args.loop:
        import time
        from datetime import datetime
        print(f"AUTO portfolio mode: {len(symbols)} symbols every {args.loop_interval}s. "
              "Keep IB Gateway open. Ctrl+C to stop.")
        try:
            while True:
                stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                for r in runner.step_all():
                    print(f"[{stamp}] {r}")
                # ib.sleep pumps the IBKR event loop so the connection survives
                # the wait (plain time.sleep starves it -> timeouts)
                w = max(30, args.loop_interval)
                b = runner.broker
                if b is not None and getattr(b, "name", "") == "ibkr" and b.is_connected():
                    b.ib.sleep(w)
                else:
                    time.sleep(w)
        except KeyboardInterrupt:
            print("\nStopped by user.")
    elif args.step:
        for r in runner.step_all():
            print(r)

    print("\n" + runner.report())


if __name__ == "__main__":
    main()
