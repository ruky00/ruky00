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
    ap.add_argument("--client-id", type=int, default=7,
                    help="unique per running bot (avoid clashing with other scripts)")
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

    if sys.version_info[:2] >= (3, 13):
        print("=" * 70)
        print(f"⚠️  Python {sys.version_info.major}.{sys.version_info.minor} detected. "
              "ib_insync does NOT reliably process fills/positions on 3.13+,")
        print("   so orders may show '✗ not filled' and P&L stays 0 even if they")
        print("   execute at IBKR. Use a Python 3.12 venv:")
        print("     py -3.12 -m venv .venv && .venv\\Scripts\\activate")
        print("     pip install -r requirements.txt ib_insync")
        print("=" * 70)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    broker = brk.IBKRBroker(port=args.port, client_id=args.client_id, tif="DAY")  # DAY avoids 10349
    broker.connect()
    print(f"🎮 PLAY intraday 5m | account {broker.ib.managedAccounts()} | {symbols}")
    print(f"   RSI<{args.rsi_buy:g} buy / RSI>{args.rsi_sell:g} "
          f"{'short' if args.allow_short else '(long-only)'} | poll {args.poll}s "
          f"| Ctrl+C to stop\n")

    try:
        start_equity = float(broker.account().get("equity", args.capital))
    except Exception:
        start_equity = args.capital

    def summary():
        try:
            eq = float(broker.account().get("equity", start_equity))
        except Exception:
            eq = start_equity
        pos = {}
        try:
            pos = broker.positions()
        except Exception:
            pass
        print(f"\n📊 SUMMARY | entries {trades_placed} | "
              f"start ${start_equity:,.0f} -> equity ${eq:,.0f} | "
              f"P&L ${eq - start_equity:+,.0f}")
        if pos:
            print("   still open:", {s: v['qty'] for s, v in pos.items()})

    last_bar = {}          # symbol -> last processed 5m timestamp (idempotency)
    trades_placed = 0
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

                # use the last CLOSED 5-minute bar (drop the still-forming one),
                # so we act once per completed bar instead of spamming every poll
                closed = df.iloc[:-1]
                ts = closed.index[-1]
                rsi = ind.rsi(closed["close"], args.rsi_window).iloc[-1]
                atr = float(ind.atr(closed, 14).iloc[-1])
                price = float(df["close"].iloc[-1])          # latest price for the order
                held = sym in positions and abs(positions[sym]["qty"]) > 0
                tag = "held" if held else "flat"

                new_bar = last_bar.get(sym) != ts
                action = ""
                try:
                    side = None
                    if new_bar and not held and atr > 0:
                        if rsi < args.rsi_buy:
                            side = "BUY"
                        elif args.allow_short and rsi > args.rsi_sell:
                            side = "SELL"
                    if side:
                        q = size(args.capital, args.risk, atr, args.stop_atr, price)
                        stop = price - args.stop_atr * atr if side == "BUY" else price + args.stop_atr * atr
                        tgt = price + args.target_atr * atr if side == "BUY" else price - args.target_atr * atr
                        broker.place_bracket(sym, q, side, entry=price, stop=stop, target=tgt)
                        trades_placed += 1
                        broker.ib.sleep(2)                   # wait for the fill
                        got = broker.positions().get(sym, {})
                        filled = abs(got.get("qty", 0)) > 0
                        emoji = "🟢" if side == "BUY" else "🔴"
                        action = f"{emoji} {side} {q} " + ("✓filled" if filled
                                                           else "✗ not filled (cancel/borrow?)")
                except Exception as e:
                    action = f"⚠️ order error ({type(e).__name__})"
                last_bar[sym] = ts
                # unrealised P&L on any open position (mark to last price)
                upnl = ""
                if held:
                    p = positions[sym]
                    upnl = f"  uP&L {p['qty'] * (price - p['entry']):+,.0f}"
                rows.append(f"  {sym:<5} {price:>8.2f}  RSI {rsi:>5.1f}  {tag:<4} {action}{upnl}")

            try:
                eq = float(broker.account().get("equity", start_equity))
                pnl = f"{eq - start_equity:+,.0f}"
                eq = f"{eq:,.0f}"
            except Exception:
                eq, pnl = "?", "?"
            stamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{stamp}] equity ${eq}  P&L ${pnl}  positions {len(positions)}  "
                  f"entries {trades_placed}")
            print("\n".join(rows) + "\n")

            if args.minutes and (time.time() - start) > args.minutes * 60:
                print("⏰ time's up — flattening everything.")
                broker.cancel_all(); broker.flatten()
                summary()
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
        print("\n👋 stopped.")
        summary()
        print("   Open positions/brackets remain at IBKR — run "
              "python examples/ibkr_test_order.py --flatten  to clear them.")
    finally:
        broker.disconnect()


if __name__ == "__main__":
    main()
