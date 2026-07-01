"""
FUNDED-ACCOUNT INTRADAY BOT  🤖  (product 2 — the live earner)

A live intraday trader for a funded / paper IBKR account. It applies the SAME
qflow quant strategies as the research lab (default the regime-adaptive `auto`,
or mean_reversion / trend_following / volatility_breakout) but on **5-minute
bars**, sizes every trade by risk, and attaches an ATR stop-loss + take-profit
(bracket) that live on IBKR. It shows SL / TP / unrealised P&L per position and
enforces account-level kill-switches.

Difference vs the main software:
  * qflow / examples/research + examples/live = the LAB: backtest, walk-forward,
    select robust (symbol, strategy) pairs, and forward-test the DAILY swing book.
  * THIS bot = the funded-account INTRADAY earner: it takes the strategies the lab
    validated and trades them live intraday for recurring P&L.

Run (Python 3.12 venv, IB Gateway paper open on 4002):
    python bot/intraday_bot.py --strategy auto --symbols NVDA,AMD,TSLA,AAPL
    python bot/intraday_bot.py --strategy mean_reversion --allow-short --kill-switches
    python bot/intraday_bot.py --minutes 120     # stop & flatten after 2h

Paper first. Intraday from a home PC is hard — validate on the lab, then let this
run on the funded account only once it is consistently green.
"""

import argparse
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

from qflow import feeds, strategies, broker as brk
from qflow.risk_governor import RiskGovernor


def size(capital, risk, atr, stop_atr, price):
    """Shares so that hitting the stop loses `risk` of capital; capped at 5% notional."""
    stop_dist = max(stop_atr * atr, 0.01)
    qty = int((capital * risk) / stop_dist)
    qty = min(qty, int((capital * 0.05) / max(price, 0.01)))
    return max(1, qty)


def main():
    ap = argparse.ArgumentParser(description="funded-account intraday quant bot")
    ap.add_argument("--strategy", default="auto", choices=list(strategies.REGISTRY))
    ap.add_argument("--symbols", default="NVDA,AMD,TSLA,AAPL,MSFT")
    ap.add_argument("--interval", default="5m", help="bar size (5m, 15m, 1h)")
    ap.add_argument("--port", type=int, default=4002)
    ap.add_argument("--client-id", type=int, default=7)
    ap.add_argument("--capital", type=float, default=1_000_000.0)
    ap.add_argument("--risk", type=float, default=0.002)          # 0.2% per trade
    ap.add_argument("--stop-atr", type=float, default=1.5)
    ap.add_argument("--target-atr", type=float, default=2.5)
    ap.add_argument("--allow-short", action="store_true")
    ap.add_argument("--kill-switches", action="store_true",
                    help="account-level daily-loss / drawdown halts")
    ap.add_argument("--max-daily-loss", type=float, default=0.03)
    ap.add_argument("--max-drawdown", type=float, default=0.06)
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--minutes", type=int, default=0, help="0 = until Ctrl+C")
    args = ap.parse_args()

    if sys.version_info[:2] >= (3, 13):
        print("=" * 70)
        print(f"⚠️  Python {sys.version_info.major}.{sys.version_info.minor}: ib_insync is "
              "unreliable on 3.13+ (fills/positions may not register). Use a 3.12 venv.")
        print("=" * 70)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    fn = strategies.REGISTRY[args.strategy]
    broker = brk.IBKRBroker(port=args.port, client_id=args.client_id, tif="DAY")
    broker.connect()
    gov = RiskGovernor({"max_daily_loss": args.max_daily_loss,
                        "max_drawdown": args.max_drawdown}) if args.kill_switches else None

    print(f"🤖 INTRADAY BOT | {args.strategy} on {args.interval} | "
          f"account {broker.ib.managedAccounts()} | {symbols}")
    print(f"   risk {args.risk*100:.2f}%/trade | SL {args.stop_atr}xATR / TP {args.target_atr}xATR"
          f"{' | kill-switches ON' if gov else ''} | Ctrl+C to stop\n")

    try:
        start_equity = float(broker.account().get("equity", args.capital))
    except Exception:
        start_equity = args.capital
    book = {}              # sym -> {side, qty, entry, sl, tp} for display
    last_bar, trades = {}, 0
    start = time.time()

    def summary():
        try:
            eq = float(broker.account().get("equity", start_equity))
        except Exception:
            eq = start_equity
        print(f"\n📊 SUMMARY | strategy {args.strategy} | entries {trades} | "
              f"start ${start_equity:,.0f} -> ${eq:,.0f} | P&L ${eq - start_equity:+,.0f}")

    try:
        while True:
            try:
                positions = broker.positions()
            except Exception:
                positions = {}
            try:
                equity = float(broker.account().get("equity", start_equity))
            except Exception:
                equity = start_equity
            if gov:
                gov.start_day(datetime.now().strftime("%Y-%m-%d"), equity)
                gov.observe(equity)

            rows = []
            for sym in symbols:
                try:
                    df = feeds.from_yahoo(sym, rng="5d", interval=args.interval)
                except Exception as e:
                    rows.append(f"  {sym:<5} data error ({type(e).__name__})"); continue
                if len(df) < 210:
                    rows.append(f"  {sym:<5} warming up ({len(df)} bars)"); continue

                sig = fn(df)                                   # qflow quant strategy
                s = int(sig.signal.iloc[-2])                   # last CLOSED bar's signal
                ts = df.index[-2]
                atr = float(sig.atr.iloc[-2]) if sig.atr is not None else 0.0
                price = float(df["close"].iloc[-1])
                held = sym in positions and abs(positions[sym]["qty"]) > 0
                if not held:
                    book.pop(sym, None)                        # exited via SL/TP at IBKR

                action = ""
                try:
                    side = "BUY" if s > 0 else ("SELL" if (s < 0 and args.allow_short) else None)
                    if side and not held and atr > 0 and last_bar.get(sym) != ts:
                        risk_amt = args.capital * args.risk
                        blocked = gov.can_open(risk_amt, equity) if gov else (True, "")
                        if gov and not blocked[0]:
                            action = f"⛔ halt: {blocked[1]}"
                        else:
                            q = size(args.capital, args.risk, atr, args.stop_atr, price)
                            sl = price - args.stop_atr * atr if side == "BUY" else price + args.stop_atr * atr
                            tp = price + args.target_atr * atr if side == "BUY" else price - args.target_atr * atr
                            res = broker.place_bracket(sym, q, side, entry=price,
                                                       stop=round(sl, 2), target=round(tp, 2))
                            trades += 1
                            if gov:
                                gov.on_open(risk_amt)
                            book[sym] = {"side": side, "qty": q, "entry": price,
                                         "sl": round(sl, 2), "tp": round(tp, 2)}
                            emoji = "🟢" if side == "BUY" else "🔴"
                            action = f"{emoji} {side} {q} [{res.status}]"
                except Exception as e:
                    action = f"⚠️ {type(e).__name__}"
                last_bar[sym] = ts

                # per-symbol line with SL / TP / unrealised P&L
                if held:
                    p = positions[sym]
                    bk = book.get(sym, {})
                    upnl = p["qty"] * (price - p["entry"])
                    sltp = (f" SL {bk.get('sl','?')} TP {bk.get('tp','?')}" if bk else "")
                    posdesc = f"{'LONG' if p['qty']>0 else 'SHORT'} {abs(p['qty']):g}@{p['entry']:.2f}"
                    rows.append(f"  {sym:<5} {price:>8.2f} sig {s:+d}  {posdesc}{sltp}  uP&L {upnl:+,.0f}")
                else:
                    rows.append(f"  {sym:<5} {price:>8.2f} sig {s:+d}  flat  {action}")

            pnl = equity - start_equity
            halt = "  🛑 HALTED" if (gov and gov.state.halted) else ""
            print(f"[{datetime.now():%H:%M:%S}] equity ${equity:,.0f}  P&L ${pnl:+,.0f}  "
                  f"positions {len(positions)}  entries {trades}{halt}")
            print("\n".join(rows) + "\n")

            if args.minutes and (time.time() - start) > args.minutes * 60:
                print("⏰ time's up — flattening.")
                broker.cancel_all(); broker.flatten(); summary(); break
            wait = max(15, args.poll)
            if broker.is_connected():
                broker.ib.sleep(wait)
            else:
                broker.connect(); time.sleep(5)
    except KeyboardInterrupt:
        print("\n👋 stopped.")
        summary()
        print("   Positions/brackets remain at IBKR — flatten with "
              "python examples/live/ibkr_test_order.py --flatten")
    finally:
        broker.disconnect()


if __name__ == "__main__":
    main()
