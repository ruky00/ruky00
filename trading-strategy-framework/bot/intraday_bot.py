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

Autonomous funded-exam mode:
  * --auto-select : the bot picks its own (strategy, interval) per symbol by
    walk-forwarding the intraday strategies out-of-sample at startup — no manual
    --strategy needed.
  * --funded      : enforces a prop-firm challenge (profit target + hard
    daily-loss / total-drawdown limits) with *dynamic* sizing — greedy while
    there's cushion, throttling down as it nears a limit, stopping BEFORE it can
    breach, and locking the pass once the target is hit.

Run (Python 3.12 venv, IB Gateway paper open on 4002):
    python bot/intraday_bot.py --strategy auto --symbols NVDA,AMD,TSLA,AAPL
    python bot/intraday_bot.py --strategy mean_reversion --allow-short --kill-switches
    python bot/intraday_bot.py --minutes 120     # stop & flatten after 2h

    # fully autonomous funded-account run: self-select + exam rules on
    python bot/intraday_bot.py --auto-select --funded --allow-short \
        --profit-target 0.08 --max-daily-loss 0.05 --max-total-drawdown 0.10

    # FundedNext via MetaTrader 5 (two-way: real fills/positions/equity back).
    # Volume is in LOTS, so pass --fixed-qty; run the MT5 terminal logged in first:
    python bot/intraday_bot.py --broker mt5 --mt5-login 123456 --mt5-password ... \
        --mt5-server FundedNext-Server --fundednext stellar_2step_p1 \
        --symbols EURUSD,XAUUSD --fixed-qty 0.10 --journal logs/fn.csv

    # route to a Lucid (futures) demo via a TradersPost/CrossTrade webhook,
    # fixed 1 contract, dry-run first to inspect the payloads without sending:
    python bot/intraday_bot.py --broker webhook --webhook-url https://... \
        --lucid 50 --symbols MES,MNQ --fixed-qty 1 --journal logs/lucid.csv \
        --webhook-dry-run

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

from qflow import feeds, strategies, intraday_select, broker as brk
from qflow.risk_governor import RiskGovernor
from qflow.funded import FundedAccount, LUCID_PRESETS, FUNDEDNEXT_PRESETS
from qflow.journal import TradeJournal


def size(capital, risk, atr, stop_atr, price):
    """Shares so that hitting the stop loses `risk` of capital; capped at 5% notional."""
    stop_dist = max(stop_atr * atr, 0.01)
    qty = int((capital * risk) / stop_dist)
    qty = min(qty, int((capital * 0.05) / max(price, 0.01)))
    return max(1, qty)


def fetch_bars(broker, sym, interval, count=800, yahoo_rng="5d"):
    """Get bars from the broker's own feed (e.g. MT5 = FundedNext data) if it has
    one; otherwise fall back to Yahoo. Same data source as execution when possible."""
    df = broker.bars(sym, interval, count)
    if df is not None and len(df):
        return df
    return feeds.from_yahoo(sym, rng=yahoo_rng, interval=interval)


def build_plan(broker, symbols, default_name, default_interval, auto_select,
               cache_path=None, reselect_hours=24.0):
    """
    Decide, per symbol, which (strategy, interval) to trade.

    With --auto-select the bot is self-sufficient: it pulls ~60d of 5-minute bars
    and walk-forwards the intraday strategies across 5m/15m/30m, keeping the best
    out-of-sample (strategy, interval). With a cache path it reuses a fresh choice
    instead of re-running the walk-forward each startup. If nothing clears the bar
    (or data is missing) it falls back to the CLI default.
    Returns {sym: (name, fn, interval)}.
    """
    plan = {}
    for sym in symbols:
        name, interval = default_name, default_interval
        if auto_select:
            try:
                loader = lambda s=sym: fetch_bars(broker, s, "5m", count=8000,
                                                  yahoo_rng="60d")
                if cache_path:
                    choice = intraday_select.select_cached(
                        sym, loader, cache_path=cache_path,
                        max_age_hours=reselect_hours, min_sharpe=0.0)
                    tag = " (cached)" if (choice and choice.get("cached")) else ""
                else:
                    choice = intraday_select.select_best(loader(), min_sharpe=0.0)
                    tag = ""
                if choice:
                    name, interval = choice["strategy"], choice["interval"]
                    print(f"   auto-select {sym:<5} -> {name} @ {interval} "
                          f"(OOS Sharpe {choice['oos_sharpe']:+.2f}){tag}")
                else:
                    print(f"   auto-select {sym:<5} -> no intraday edge cleared the "
                          f"bar; using default {name} @ {interval}")
            except Exception as e:
                print(f"   auto-select {sym:<5} -> data/selection error "
                      f"({type(e).__name__}); using default {name} @ {interval}")
        plan[sym] = (name, strategies.REGISTRY[name], interval)
    return plan


def main():
    ap = argparse.ArgumentParser(description="funded-account intraday quant bot")
    ap.add_argument("--strategy", default="auto", choices=list(strategies.REGISTRY),
                    help="strategy to trade (ignored per-symbol when --auto-select)")
    ap.add_argument("--symbols", default="NVDA,AMD,TSLA,AAPL,MSFT")
    ap.add_argument("--interval", default="5m", help="bar size (5m, 15m, 30m, 1h)")
    ap.add_argument("--broker", default="ibkr", choices=["ibkr", "mt5", "webhook", "paper"],
                    help="execution venue: ibkr (stocks) · mt5 (FundedNext, two-way) · "
                         "webhook (Lucid via TradersPost) · paper (offline sim)")
    ap.add_argument("--webhook-url", default="",
                    help="webhook URL for --broker webhook (from TradersPost/CrossTrade)")
    ap.add_argument("--webhook-dry-run", action="store_true",
                    help="build the webhook payloads but don't POST (safe test)")
    ap.add_argument("--mt5-login", type=int, default=0, help="MT5 account number (FundedNext)")
    ap.add_argument("--mt5-password", default="", help="MT5 account password")
    ap.add_argument("--mt5-server", default="", help="MT5 server (e.g. FundedNext-Server)")
    ap.add_argument("--mt5-path", default="", help="path to terminal64.exe (optional)")
    ap.add_argument("--fixed-qty", type=float, default=0.0,
                    help="send a fixed quantity instead of the ATR/equity share sizer "
                         "(futures contracts or MT5 lots, e.g. 1 or 0.10)")
    ap.add_argument("--auto-select", action="store_true",
                    help="self-pick the best (strategy, interval) per symbol via "
                         "intraday walk-forward at startup (autonomous mode)")
    ap.add_argument("--select-cache", default="",
                    help="cache auto-select choices to this JSON (skip re-running "
                         "the walk-forward while fresh)")
    ap.add_argument("--reselect-hours", type=float, default=24.0,
                    help="re-run auto-select when the cached choice is older than this")
    ap.add_argument("--journal", default="",
                    help="append every entry/exit to this CSV (live win-rate/expectancy)")
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
    # --- funded-account exam mode (greedy but capped) ---
    ap.add_argument("--funded", action="store_true",
                    help="enforce a prop-firm challenge: profit target + hard "
                         "daily-loss / total-drawdown limits with dynamic sizing")
    ap.add_argument("--lucid", type=int, choices=sorted(LUCID_PRESETS),
                    help="use a Lucid Trading preset (account size 25/50/100/150): "
                         "sets target, EOD trailing drawdown and 50%% consistency")
    ap.add_argument("--fundednext", choices=sorted(FUNDEDNEXT_PRESETS),
                    help="use a FundedNext preset (stellar_1step / stellar_2step_p1 / "
                         "stellar_2step_p2 / express): sets target + daily/overall DD + min days")
    ap.add_argument("--profit-target", type=float, default=0.08)
    ap.add_argument("--max-total-drawdown", type=float, default=0.10)
    ap.add_argument("--drawdown-mode", choices=["static", "trailing", "eod"],
                    default="static", help="drawdown floor: fixed / intraday-peak / EOD-trailing")
    ap.add_argument("--consistency", type=float, default=1.0,
                    help="best-day <= this share of total profit (0.5 = Lucid eval; 1.0 = off)")
    ap.add_argument("--base-risk", type=float, default=0.004,
                    help="funded full-size risk per trade (scaled up/down by cushion)")
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--minutes", type=int, default=0, help="0 = until Ctrl+C")
    args = ap.parse_args()

    if args.broker == "ibkr" and sys.version_info[:2] >= (3, 13):
        print("=" * 70)
        print(f"⚠️  Python {sys.version_info.major}.{sys.version_info.minor}: ib_insync is "
              "unreliable on 3.13+ (fills/positions may not register). Use a 3.12 venv.")
        print("=" * 70)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if args.broker == "webhook":
        broker = brk.WebhookBroker(args.webhook_url, capital=args.capital,
                                   dry_run=args.webhook_dry_run)
    elif args.broker == "mt5":
        broker = brk.MT5Broker(login=args.mt5_login, password=args.mt5_password,
                               server=args.mt5_server, path=args.mt5_path)
    elif args.broker == "paper":
        broker = brk.PaperBroker(cash=args.capital)
    else:
        broker = brk.IBKRBroker(port=args.port, client_id=args.client_id, tif="DAY")
    broker.connect()
    gov = RiskGovernor({"max_daily_loss": args.max_daily_loss,
                        "max_drawdown": args.max_drawdown}) if args.kill_switches else None

    mode = "AUTO-SELECT (walk-forward per symbol)" if args.auto_select else args.strategy
    print(f"🤖 INTRADAY BOT | {mode} | {args.broker}:{broker.account_label()} | {symbols}")
    if args.auto_select:
        print("   picking the best (strategy, interval) per symbol out-of-sample...")
    plan = build_plan(broker, symbols, args.strategy, args.interval, args.auto_select,
                      cache_path=args.select_cache or None,
                      reselect_hours=args.reselect_hours)

    try:
        start_equity = float(broker.account().get("equity", args.capital))
    except Exception:
        start_equity = args.capital

    jrn = TradeJournal(args.journal) if args.journal else None
    if jrn:
        print(f"   📓 journaling entries/exits to {args.journal}")

    fund = None
    if args.fundednext:
        fund = FundedAccount.from_fundednext(args.fundednext, consistency=args.consistency,
                                             base_risk=args.base_risk)
        fund.set_anchor(start_equity)
        R = fund.rules
        print(f"   💰 FUNDEDNEXT {args.fundednext} | target +{R['profit_target']*100:.0f}% | "
              f"daily-loss {R['max_daily_loss']*100:.0f}% | overall-DD {R['max_total_drawdown']*100:.0f}% | "
              f"min-days {R['min_trading_days']} | base risk {R['base_risk']*100:.2f}% "
              f"(greedy, cushion-scaled)")
    elif args.lucid:
        fund = FundedAccount.from_lucid(args.lucid, consistency=args.consistency,
                                        base_risk=args.base_risk)
        fund.set_anchor(start_equity)     # trade the real account balance if it differs
        R = fund.rules
        dll = "none" if R["max_daily_loss"] >= 1 else f"{R['max_daily_loss']*100:.1f}%"
        print(f"   💰 LUCID {args.lucid}K EXAM | target +{R['profit_target']*100:.1f}% | "
              f"EOD-trailing DD {R['max_total_drawdown']*100:.1f}% | daily-loss {dll} | "
              f"consistency {R['consistency_pct']*100:.0f}% | "
              f"base risk {R['base_risk']*100:.2f}% (greedy, cushion-scaled)")
    elif args.funded:
        fund = FundedAccount({"profit_target": args.profit_target,
                              "max_daily_loss": args.max_daily_loss,
                              "max_total_drawdown": args.max_total_drawdown,
                              "drawdown_mode": args.drawdown_mode,
                              "consistency_pct": args.consistency,
                              "base_risk": args.base_risk},
                             start_equity=start_equity)
        print(f"   💰 FUNDED EXAM | target +{args.profit_target*100:.0f}% | "
              f"daily-loss {args.max_daily_loss*100:.0f}% | "
              f"total-DD {args.max_total_drawdown*100:.0f}% ({args.drawdown_mode}) | "
              f"base risk {args.base_risk*100:.2f}% (greedy, cushion-scaled)")
    else:
        print(f"   risk {args.risk*100:.2f}%/trade | SL {args.stop_atr}xATR / "
              f"TP {args.target_atr}xATR{' | kill-switches ON' if gov else ''}")
    print("   Ctrl+C to stop\n")
    book = {}              # sym -> {side, qty, entry, sl, tp, strategy, interval} for display
    last_bar, trades = {}, 0
    start = time.time()

    def summary():
        try:
            eq = float(broker.account().get("equity", start_equity))
        except Exception:
            eq = start_equity
        print(f"\n📊 SUMMARY | strategy {args.strategy} | entries {trades} | "
              f"start ${start_equity:,.0f} -> ${eq:,.0f} | P&L ${eq - start_equity:+,.0f}")
        if jrn:
            js = jrn.summary()
            print(f"   📓 journal ({args.journal}): {js['exits']} closed | "
                  f"win {js['win_rate']*100:.0f}% | avg win ${js['avg_win']:,.0f} / "
                  f"avg loss ${js['avg_loss']:,.0f} | expectancy ${js['expectancy']:,.0f}/trade")

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
            today = datetime.now().strftime("%Y-%m-%d")
            if gov:
                gov.start_day(today, equity)
                gov.observe(equity)
            if fund:
                fund.update(equity, today)
                if fund.done:                                  # exam finished this run
                    verdict = "✅ PASSED" if fund.state.passed else "❌ FAILED"
                    print(f"\n{verdict} — {fund.state.reason}. Flattening, no new risk.")
                    broker.cancel_all(); broker.flatten(); summary(); break

            rows = []
            for sym in symbols:
                name, fn, interval = plan[sym]
                try:
                    df = fetch_bars(broker, sym, interval, count=800, yahoo_rng="5d")
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
                if not held and sym in book:
                    bk = book.pop(sym)                         # exited via SL/TP at IBKR
                    if jrn:
                        signed = bk["qty"] if bk["side"] == "BUY" else -bk["qty"]
                        jrn.log_exit(ts, sym, price, signed * (price - bk["entry"]))

                # funded mode drives risk dynamically (greedy w/ cushion, throttled near limits)
                risk = fund.risk_fraction() if fund else args.risk

                action = ""
                try:
                    side = "BUY" if s > 0 else ("SELL" if (s < 0 and args.allow_short) else None)
                    if side and not held and atr > 0 and last_bar.get(sym) != ts:
                        risk_amt = args.capital * risk
                        blocked = gov.can_open(risk_amt, equity) if gov else (True, "")
                        fund_ok = fund.can_open() if fund else (True, "")
                        if gov and not blocked[0]:
                            action = f"⛔ halt: {blocked[1]}"
                        elif fund and not fund_ok[0]:
                            action = f"⛔ {fund_ok[1]}"
                        else:
                            q = args.fixed_qty or size(args.capital, risk, atr, args.stop_atr, price)
                            sl = price - args.stop_atr * atr if side == "BUY" else price + args.stop_atr * atr
                            tp = price + args.target_atr * atr if side == "BUY" else price - args.target_atr * atr
                            res = broker.place_bracket(sym, q, side, entry=price,
                                                       stop=round(sl, 2), target=round(tp, 2))
                            trades += 1
                            if gov:
                                gov.on_open(risk_amt)
                            if fund:
                                fund.on_open()
                            book[sym] = {"side": side, "qty": q, "entry": price,
                                         "sl": round(sl, 2), "tp": round(tp, 2)}
                            if jrn:
                                jrn.log_entry(ts, sym, side, q, price, round(sl, 2),
                                              round(tp, 2), risk * 100, name, interval)
                            emoji = "🟢" if side == "BUY" else "🔴"
                            action = f"{emoji} {side} {q} @{risk*100:.2f}% [{res.status}]"
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
            if fund:
                st = fund.status()
                cons = "" if st["consistency_ok"] else "  ⚠️ consistency"
                print(f"   💰 exam: profit {st['profit']:+.2f}%/{st['target']:.1f}%  "
                      f"day-loss {st['day_loss']:.2f}%  DD-floor ${st['dd_floor']:,.0f}  "
                      f"cushion day/total {st['daily_cushion']:.0f}%/{st['total_cushion']:.0f}%  "
                      f"days {st['trading_days']}/{st['min_days']}{cons}")
            print("\n".join(rows) + "\n")

            if args.minutes and (time.time() - start) > args.minutes * 60:
                print("⏰ time's up — flattening.")
                broker.cancel_all(); broker.flatten(); summary(); break
            wait = max(15, args.poll)
            if broker.is_connected():
                broker.sleep(wait)
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
