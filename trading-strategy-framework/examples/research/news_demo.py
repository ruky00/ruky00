"""
News + sentiment demo and how it gates trades.

    python examples/news_demo.py

Runs offline with the bundled SampleProvider. On your own machine, swap in a
live provider (no Bloomberg Terminal required):

    from qflow import news
    items = news.get_news("AAPL", providers=[news.RSSProvider()])          # free RSS
    items = news.get_news("AAPL", providers=[news.FinnhubProvider()])      # needs key
    # Bloomberg, only if you actually have a Terminal/B-PIPE entitlement:
    items = news.get_news("AAPL", providers=[news.BloombergProvider()])

And wire it into paper trading so adverse headlines veto/down-size entries:

    pt = PaperTrader("AAPL", strategy="mean_reversion",
                     news_provider=news.RSSProvider())
    pt.step()   # the overlay is consulted live before any new entry
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from qflow import news


def main():
    provider = news.SampleProvider()        # offline; swap for RSSProvider() live
    for sym in ["AAPL", "TSLA"]:
        items = provider.fetch(sym)
        print(news.report(sym, items))
        for direction, label in [(1, "LONG"), (-1, "SHORT")]:
            ov = news.news_overlay(items, direction)
            print(f"    if you wanted to go {label:<5}: size x{ov['size_multiplier']:<3} "
                  f"[{ov['action']}] — {ov['reason']}")
        print()

    print("How it plugs into trading:")
    print("  * size x1.0  -> trade normally")
    print("  * size x0.5  -> adverse tone or an event (earnings/FDA/Fed): half size")
    print("  * size x0.0  -> news strongly against the trade: skip it entirely")
    print("\nThe overlay only fires on the live --step path, never in historical")
    print("replay (there is no historical news feed), so backtests stay clean.")


if __name__ == "__main__":
    main()
