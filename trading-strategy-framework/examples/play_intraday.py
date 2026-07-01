"""
PLAY MODE — intraday 5-minute NASDAQ scalper on your IBKR PAPER account. 🎮

Just for fun on paper money. Every few seconds it pulls 5-minute bars for a few
liquid NASDAQ names, and when one gets short-term oversold (RSI) it buys a bracket
(stop + take-profit that live on IBKR); overbought -> short (optional). It only
acts on a NEW 5-minute bar, and the bracket handles the exit.

    python examples/play_intraday.py                 # default basket, until Ctrl+C
    python examples/play_intraday.py --minutes 90    # stop & flatten after 90 min
    python examples/play_intraday.py --allow-short   # also short overbought spikes

PREREQUISITES: IB Gateway (paper, port 4002) open + logged in, `pip install
ib_insync`, real-time (or delayed) NASDAQ market data. Uses tif=DAY so the
account's order preset doesn't cancel the bracket (error 10349).

NOTE: intraday from a home PC is NOT an edge — this is a toy to watch the machine
trade live. Paper only. Have fun, then go back to the daily swing system.
"""

import argparse
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

from qflow import feeds, indicators as ind, broker as brk


def size(capital, risk, atr, stop_atr, price):
    stop_dist = max(stop_atr * atr, 0.01)
    qty = int((capital * risk) / stop_dist)
    qty = min(qty, int((capital * 0.05) / max(price, 0.01)))   # <=5% notional/name
    return max(1, qty)


def main():
    ap = argparse.ArgumentParser(description="intraday 5m NASDAQ play (paper)")
    ap.add_argument("--symbols", default="AAPL,MSFT,NVDA,AMD,TSLA,QQQ")
    ap.add_argument("--port", type=int, default=4002)
    ap.add_argument("--capital", type=float, default=1_000_000.0)
    ap.add_argument("--risk", type=float, default=0.001)       # 0.1% per trade
    ap.add_argument("--rsi-window", type=int, default=14)
    ap.add_argument("--rsi-buy", type=float, default=30.0)
    ap.add_argument("--rsi-sell", type=float, default=70.0)
    ap.add_argument("--stop-atr", type=float, default=1.5)
    ap.add_argument("--target-atr", type=float, default=2.5)
    ap.add_argument("--allow-short", action="store_true")
    ap.add_argument("--poll", type=int, default=60, help="seconds between checks")
    ap.add_argument("--minutes", type=int, default=0, help="0 = until Ctrl+C")
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    broker = brk.IBKRBroker(port=args.port, tif="DAY")         # DAY avoids 10349
    broker.connect()
    print(f"🎮 PLAY intraday 5m | account {broker.ib.managedAccounts()} | {symbols}")
    print(f"   RSI<{args.rsi_buy:g} buy / RSI>{args.rsi_sell:g} "
          f"{'short' if args.allow_short else '(long-only)'} | poll {args.poll}s "
          f"| Ctrl+C to stop\n")

    last_bar = {}          # symbol -> last processed 5m timestamp (idempotency)
    start = time.time()
    try:
        while True:
            positions = {}
            try:
                positions = broker.positions()
            except Exception:
                pass
            rows = []
            for sym in symbols:
                try:
                    df = feeds.from_yahoo(sym, rng="5d", interval="5m")
                except Exception as e:
                    rows.append(f"  {sym:<5} data error ({type(e).__name__})")
                    continue
                if len(df) < 30:
                    rows.append(f"  {sym:<5} not enough bars"); continue

                ts = df.index[-1]
                rsi = ind.rsi(df["close"], args.rsi_window).iloc[-1]
                atr = float(ind.atr(df, 14).iloc[-1])
                price = float(df["close"].iloc[-1])
                held = sym in positions and abs(positions[sym]["qty"]) > 0
                tag = "held" if held else "flat"

                # act only on a NEW bar, only when flat
                new_bar = last_bar.get(sym) != ts
                action = ""
                try:
                    if new_bar and not held and atr > 0:
                        if rsi < args.rsi_buy:
                            q = size(args.capital, args.risk, atr, args.stop_atr, price)
                            broker.place_bracket(sym, q, "BUY", entry=price,
                                                 stop=price - args.stop_atr * atr,
                                                 target=price + args.target_atr * atr)
                            action = f"🟢 BUY {q}"
                        elif args.allow_short and rsi > args.rsi_sell:
                            q = size(args.capital, args.risk, atr, args.stop_atr, price)
                            broker.place_bracket(sym, q, "SELL", entry=price,
                                                 stop=price + args.stop_atr * atr,
                                                 target=price - args.target_atr * atr)
                            action = f"🔴 SHORT {q}"
                    if action:
                        broker.ib.sleep(1)      # let the order register
                except Exception as e:
                    action = f"⚠️ order error ({type(e).__name__})"
                last_bar[sym] = ts
                rows.append(f"  {sym:<5} {price:>8.2f}  RSI {rsi:>5.1f}  {tag:<4} {action}")

            try:
                eq = broker.account().get("equity", "?")
            except Exception:
                eq = "?"
            stamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{stamp}] equity {eq}  positions {len(positions)}")
            print("\n".join(rows) + "\n")

            if args.minutes and (time.time() - start) > args.minutes * 60:
                print("⏰ time's up — flattening everything.")
                broker.cancel_all(); broker.flatten()
                break
            # IMPORTANT: ib.sleep() pumps the ib_insync event loop so the IBKR
            # connection stays alive during the wait (a plain time.sleep starves
            # it and causes timeouts). Falls back to time.sleep if disconnected.
            wait = max(15, args.poll)
            if broker.is_connected():
                broker.ib.sleep(wait)
            else:
                broker.connect()
                time.sleep(5)
    except KeyboardInterrupt:
        print("\n👋 stopped. Open positions/brackets remain at IBKR — "
              "run  python examples/ibkr_test_order.py --flatten  to clear them.")
    finally:
        broker.disconnect()


if __name__ == "__main__":
    main()
