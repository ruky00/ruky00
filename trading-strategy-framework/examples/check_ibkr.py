"""
Check the connection to IB Gateway / TWS — run this BEFORE trading.

    python examples/check_ibkr.py                 # IB Gateway paper (4002)
    python examples/check_ibkr.py --port 7497     # TWS paper

It tells you, in plain language, whether the API is reachable, which account is
logged in, and whether it's a paper or live account — without touching any menu.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=4002,
                    help="4002 Gateway paper (default) · 7497 TWS paper · 4001/7496 LIVE")
    ap.add_argument("--client-id", type=int, default=1)
    args = ap.parse_args()

    try:
        from ib_insync import IB
    except ImportError:
        print("[X] ib_insync is not installed. Run:  pip install ib_insync")
        return

    ib = IB()
    print(f"Connecting to {args.host}:{args.port} ...")
    try:
        ib.connect(args.host, args.port, clientId=args.client_id, timeout=8)
    except Exception as e:
        print(f"[X] Could not connect ({type(e).__name__}: {e})\n")
        print("Most likely one of these:")
        print("  1. IB Gateway/TWS is not open or not logged in.")
        print("  2. The API is off: Configuración → Settings → API → Settings →")
        print("     tick 'Enable ActiveX and Socket Clients'.")
        print(f"  3. Wrong port: this tried {args.port}. Gateway paper = 4002, "
              "TWS paper = 7497.")
        print("  4. Add 127.0.0.1 to 'Trusted IPs'.")
        return

    accounts = ib.managedAccounts()
    server_v = ib.client.serverVersion()
    is_paper = any(a.startswith("DU") for a in accounts)   # DU = paper, U = live
    print("[OK] Connected to the API.")
    print(f"     Server version : {server_v}")
    print(f"     Accounts       : {accounts}")
    print(f"     Mode           : {'PAPER (safe)' if is_paper else 'LIVE — real money!'}")
    try:
        summary = {v.tag: v.value for v in ib.accountSummary()}
        print(f"     Net liquidation: {summary.get('NetLiquidation', '?')} "
              f"{summary.get('Currency', '')}")
    except Exception:
        pass
    print("\nIf you see [OK] and PAPER, you're ready:  "
          "python examples/paper_trade.py --broker ibkr --ibkr-port %d --step" % args.port)
    ib.disconnect()


if __name__ == "__main__":
    main()
