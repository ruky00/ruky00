"""
Gap-fade routine (intraday, flat overnight) — two scheduled phases.

The gap-fade edge: on a volatile name that gaps DOWN at the open, buy and hold
the session (it tends to fill); on a gap UP, short it. Enter at the open, exit at
the close, nothing held overnight.

Run it as TWO scheduled tasks per day:
    ~15:35 Spain (US open):   python examples/gap_fade_routine.py --phase open  ...
    ~21:55 Spain (US close):  python examples/gap_fade_routine.py --phase close ...

Offline dry run (bundled samples, simulated broker):
    python examples/gap_fade_routine.py --phase open  --symbols AAPL,TSLA \
        --source github --broker paper
    python examples/gap_fade_routine.py --phase close --symbols AAPL,TSLA \
        --source github --broker paper

Live (IB Gateway paper):  add  --broker ibkr --ibkr-port 4002  --source yahoo
Honest caveat: this is the one intraday piece — it needs session data and a
reliable, always-on machine. Validate on paper first.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import warnings
warnings.filterwarnings("ignore")

from qflow import feeds, indicators as ind
from qflow.paper import DEFAULT_ROOT

STATE = os.path.join(DEFAULT_ROOT, "gapfade_state.json")


def make_broker(args):
    from qflow import broker as brk
    if args.broker == "paper":
        return brk.PaperBroker(cash=args.capital)
    return brk.IBKRBroker(host=args.ibkr_host, port=args.ibkr_port,
                          allow_live=args.ibkr_allow_live,
                          currency=args.currency, exchange=args.exchange,
                          primary_exchange=args.primary)


def load_state():
    if os.path.exists(STATE):
        with open(STATE) as f:
            return json.load(f)
    return {}


def save_state(s):
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    with open(STATE, "w") as f:
        json.dump(s, f, indent=2)


def phase_open(args, broker):
    entered = {}
    for sym in args.symbols:
        try:
            df = feeds.get(sym, source=args.source,
                           **({"interval": args.interval} if args.interval else {}))
        except Exception as e:
            print(f"  {sym}: data error ({type(e).__name__}); skip"); continue
        if len(df) < 20:
            print(f"  {sym}: not enough data; skip"); continue
        prev_close = float(df["close"].iloc[-2])
        today_open = float(df["open"].iloc[-1])
        gap = today_open / prev_close - 1.0
        if abs(gap) < args.threshold:
            print(f"  {sym}: gap {gap*100:+.2f}% < threshold; no trade"); continue
        side = "BUY" if gap < 0 else "SELL"          # fade the gap
        atr = float(ind.atr(df, 14).iloc[-1]) or (today_open * 0.03)
        risk_amt = args.capital / max(1, len(args.symbols)) * args.risk
        qty = max(1, int(risk_amt / (args.stop_atr * atr)))
        qty = min(qty, int((args.capital / len(args.symbols)) / today_open) or 1)
        res = broker.place_market(sym, qty, side, price=today_open)
        entered[sym] = {"side": side, "qty": qty, "entry": today_open,
                        "order_id": res.order_id}
        print(f"  {sym}: gap {gap*100:+.2f}% -> {side} {qty} @ ~{today_open:.2f} "
              f"(order {res.order_id})")
    save_state(entered)
    print(f"\nOpened {len(entered)} position(s). Run --phase close before the bell "
          "to flatten.")


def phase_close(args, broker):
    entered = load_state()
    if not entered:
        print("No open gap-fade positions recorded.")
        return
    for sym, info in entered.items():
        try:
            broker.flatten(sym)
            print(f"  {sym}: flattened ({info['side']} {info['qty']})")
        except Exception as e:
            print(f"  {sym}: flatten error ({type(e).__name__})")
    save_state({})
    print("\nAll gap-fade positions flattened. Flat overnight.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["open", "close"])
    ap.add_argument("--symbols", default="TSLA,NVDA,AMD,COIN,PLTR")
    ap.add_argument("--source", default="yahoo",
                    choices=["github", "binance", "stooq", "yahoo"])
    ap.add_argument("--interval", default=None)
    ap.add_argument("--threshold", type=float, default=0.005, help="min gap (0.5%)")
    ap.add_argument("--capital", type=float, default=10_000.0)
    ap.add_argument("--risk", type=float, default=0.005)
    ap.add_argument("--stop-atr", type=float, default=2.0)
    ap.add_argument("--broker", default="paper", choices=["paper", "ibkr"])
    ap.add_argument("--ibkr-port", type=int, default=4002)
    ap.add_argument("--ibkr-host", default="127.0.0.1")
    ap.add_argument("--ibkr-allow-live", action="store_true")
    ap.add_argument("--currency", default="USD")
    ap.add_argument("--exchange", default="SMART")
    ap.add_argument("--primary", default="")
    args = ap.parse_args()
    args.symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    broker = make_broker(args)
    broker.connect()
    print(f"[gap-fade {args.phase}] broker={broker.name} symbols={args.symbols}")
    if args.phase == "open":
        phase_open(args, broker)
    else:
        phase_close(args, broker)
    broker.disconnect()


if __name__ == "__main__":
    main()
