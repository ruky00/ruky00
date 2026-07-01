"""
Place ONE tiny bracket order on your IBKR PAPER account — a plumbing smoke test.

    python examples/ibkr_test_order.py --symbol AAPL --qty 1          # place it
    python examples/ibkr_test_order.py --flatten                      # close everything

It buys `qty` shares with a stop-loss (-5%) and take-profit (+10%) so you can SEE
the full bracket appear in IB Gateway (orders + position). Paper money only —
defaults to the paper port 4002 and refuses live ports.
"""

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="AAPL")
    ap.add_argument("--qty", type=int, default=1)
    ap.add_argument("--port", type=int, default=4002, help="4002 Gateway paper / 7497 TWS paper")
    ap.add_argument("--price", type=float, default=0.0,
                    help="reference price (else fetched as delayed data)")
    ap.add_argument("--flatten", action="store_true", help="cancel orders + close positions")
    args = ap.parse_args()

    # Python 3.13/3.14 import guard
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    from qflow import broker
    ib = broker.IBKRBroker(port=args.port)      # paper by default
    ib.connect()
    print(f"connected to {args.port} — accounts: {ib.ib.managedAccounts()}")

    if args.flatten:
        ib.cancel_all()
        ib.flatten()
        print("Cancelled all orders and flattened all positions.")
        ib.disconnect()
        return

    # get a reference price (delayed data is fine for a test)
    price = args.price
    if not price:
        from ib_insync import Stock
        c = Stock(args.symbol, "SMART", "USD")
        ib.ib.qualifyContracts(c)
        ib.ib.reqMarketDataType(3)              # 3 = delayed
        t = ib.ib.reqMktData(c, "", False, False)
        ib.ib.sleep(2.5)
        price = t.marketPrice()
        if price != price or price <= 0:        # NaN or invalid
            price = t.close or 0
    if not price or price <= 0:
        print("Could not fetch a price. Re-run with --price <approx price>.")
        ib.disconnect()
        return

    stop = round(price * 0.95, 2)
    target = round(price * 1.10, 2)
    print(f"{args.symbol} ~{price:.2f}  ->  bracket: BUY {args.qty}, "
          f"stop {stop}, target {target}")
    res = ib.place_bracket(args.symbol, qty=args.qty, side="BUY",
                           entry=price, stop=stop, target=target, entry_type="MKT")
    print(f"submitted parent order id {res.order_id}.")
    print("Look in IB Gateway (or Client Portal): you should see the entry fill "
          "plus a resting STOP and LIMIT (take-profit).")
    print("Clean up when done:  python examples/ibkr_test_order.py --flatten")
    ib.disconnect()


if __name__ == "__main__":
    main()
