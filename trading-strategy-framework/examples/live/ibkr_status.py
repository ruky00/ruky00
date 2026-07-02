"""
Show your IBKR account status via the API — orders, positions, balances.

    python examples/ibkr_status.py                 # IB Gateway paper (4002)
    python examples/ibkr_status.py --port 7497     # TWS paper

Because IBKR allows only one trading session at a time, the web Client Portal
goes read-only/delayed while IB Gateway holds the session. This reads straight
from the Gateway session, so it shows the REAL open orders and positions the bot
placed.
"""

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=4002,
                    help="4002 Gateway paper (default) / 7497 TWS paper")
    ap.add_argument("--client-id", type=int, default=2)   # different id from the bot
    args = ap.parse_args()

    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    from qflow import broker
    ib = broker.IBKRBroker(port=args.port)
    ib.connect()
    api = ib.ib
    print(f"Connected to {args.port} — accounts: {api.managedAccounts()}\n")

    # --- account balances ---
    summ = {v.tag: v.value for v in api.accountSummary()}
    print("ACCOUNT")
    print(f"  Net liquidation : {summ.get('NetLiquidation','?')} {summ.get('Currency','')}")
    print(f"  Cash            : {summ.get('TotalCashValue','?')}")
    print(f"  Buying power    : {summ.get('BuyingPower','?')}\n")

    # --- open orders (pulls orders from all API clients) ---
    api.reqAllOpenOrders()
    api.sleep(1.0)
    trades = api.openTrades()
    print(f"OPEN ORDERS ({len(trades)})")
    if not trades:
        print("  (none — nothing pending)")
    for t in trades:
        o, c, st = t.order, t.contract, t.orderStatus
        print(f"  #{o.orderId:<4} {o.action:<4} {o.totalQuantity:>5g} {c.symbol:<6}"
              f" {o.orderType:<5} @ {o.lmtPrice or o.auxPrice or 'MKT'}"
              f"  [{st.status}]")
    print()

    # --- positions ---
    positions = api.positions()
    print(f"POSITIONS ({len(positions)})")
    if not positions:
        print("  (flat — no open positions)")
    for p in positions:
        print(f"  {p.contract.symbol:<6} qty {p.position:>6g}  avg cost {p.avgCost:.2f}")

    ib.disconnect()
    print("\nTip: markets closed -> orders sit as 'PreSubmitted'/'Submitted' until "
          "the open. Cancel test orders with:  python examples/ibkr_test_order.py --flatten")


if __name__ == "__main__":
    main()
