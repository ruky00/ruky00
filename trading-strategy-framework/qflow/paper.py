"""
Paper-trading engine — forward-test a strategy with virtual money.

This is the bridge between a backtest and a live account. You run it once per
bar (e.g. daily, after the close, via cron) for a week or two. It:

    * pulls the latest real data,
    * computes the strategy signal causally (only past bars),
    * simulates fills, commissions and slippage on a virtual $10k account,
    * sizes every trade by a fixed % risk and an ATR stop,
    * persists account state + a full trade journal + an equity curve to disk,
    * and grades go-live readiness against conservative gates.

State lives under ``data/paper/<symbol>_<strategy>/`` so each daily run resumes
exactly where the last one left off — a true forward test, not a re-run.

    >>> from qflow.paper import PaperTrader
    >>> pt = PaperTrader(symbol="AAPL", source="github", strategy="mean_reversion")
    >>> pt.step()        # process the latest bar (daily cron)
    >>> print(pt.report())

Use ``replay(n)`` to fast-forward the last n historical bars (one simulated day
each) so you can preview the whole workflow immediately.

NOTE: paper trading proves your *plumbing and discipline*. One or two weeks of
daily bars is far too little data to prove an edge statistically — keep the long
backtest and Monte-Carlo as your statistical evidence.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

from . import feeds, strategies, daily, indicators as ind, metrics
from .risk_governor import RiskGovernor

# every strategy the paper engine can run, across execution styles
ALL_STRATEGIES = (set(strategies.REGISTRY)
                  | set(daily.INTRADAY_REGISTRY)
                  | set(daily.LEADER_STRATEGIES))

DEFAULT_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "paper"
)


@dataclass
class Position:
    direction: int = 0           # +1 long, -1 short, 0 flat
    shares: float = 0.0
    entry: float = 0.0
    stop: float = 0.0
    target: float = 0.0
    risk_amt: float = 0.0
    entry_date: str = ""

    @property
    def is_open(self) -> bool:
        return self.direction != 0


@dataclass
class State:
    symbol: str
    source: str
    strategy: str
    params: dict
    capital: float
    risk_per_trade: float
    stop_atr: float
    target_atr: float
    fee: float
    balance: float                       # realised account value
    last_date: str = ""                  # last bar date processed (idempotency)
    execution: str = "swing"             # "swing" or "intraday"
    leader_symbol: str = ""              # for lead-lag strategies
    leader_source: str = ""
    governor: dict = field(default_factory=dict)   # {limits, state} or {}
    position: dict = field(default_factory=lambda: asdict(Position()))
    journal: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)   # [{date, equity}]


class PaperTrader:
    def __init__(self,
                 symbol: str = "AAPL",
                 source: str = "github",
                 strategy: str = "mean_reversion",
                 params: dict | None = None,
                 capital: float = 10_000.0,
                 risk_per_trade: float = 0.01,
                 stop_atr: float = 2.0,
                 target_atr: float = 4.0,
                 commission_bps: float = 2.0,
                 slippage_bps: float = 2.0,
                 root: str = DEFAULT_ROOT,
                 feed_kwargs: dict | None = None,
                 leader_symbol: str = "",
                 leader_source: str = "",
                 news_provider=None,
                 risk_limits: dict | None = None,
                 broker=None,
                 broker_symbol: str = ""):
        if strategy not in ALL_STRATEGIES:
            raise ValueError(f"Unknown strategy {strategy!r}. "
                             f"Choose from {sorted(ALL_STRATEGIES)}.")
        if strategy in daily.LEADER_STRATEGIES and not leader_symbol:
            raise ValueError(f"Strategy {strategy!r} needs a leader_symbol "
                             "(the asset that moves first).")
        self.symbol = symbol
        self.source = source
        self.strategy = strategy
        self.feed_kwargs = feed_kwargs or {}
        self.news_provider = news_provider     # optional live news overlay
        self._news_mult = 1.0                  # transient: applied only in step()
        self._news_reason = ""
        self.governor = None                   # set after state is ready
        self.broker = broker                   # optional real execution
        self.broker_symbol = broker_symbol or symbol   # IBKR ticker (may differ from data)
        self._live = False                     # True only inside step(); never in replay
        self._external_block = False           # set by a portfolio runner to veto new entries
        self.dir = os.path.join(root, f"{symbol}_{strategy}")
        os.makedirs(self.dir, exist_ok=True)
        self.state_path = os.path.join(self.dir, "state.json")

        if os.path.exists(self.state_path):
            self.state = self._load()
        else:
            self.state = State(
                symbol=symbol, source=source, strategy=strategy,
                params=params or {}, capital=capital,
                risk_per_trade=risk_per_trade, stop_atr=stop_atr,
                target_atr=target_atr,
                fee=(commission_bps + slippage_bps) / 1e4,
                balance=capital,
                leader_symbol=leader_symbol,
                leader_source=leader_source or source,
                governor={"limits": dict(RiskGovernor(risk_limits).limits),
                          "state": asdict(RiskGovernor(risk_limits).state)}
                if risk_limits is not None else {},
            )
            self._save()

        # allow enabling/retuning limits on an existing account
        if risk_limits is not None and self.state.governor.get("limits") != \
                {**RiskGovernor().limits, **risk_limits}:
            g = RiskGovernor(risk_limits, self.state.governor.get("state"))
            self.state.governor = g.to_dict()

        # live governor instance (None if no limits configured)
        self.governor = (RiskGovernor(self.state.governor["limits"],
                                      self.state.governor["state"])
                         if self.state.governor else None)

    # ----------------------- persistence ----------------------- #
    def _load(self) -> State:
        with open(self.state_path) as f:
            return State(**json.load(f))

    def _sync_governor(self) -> None:
        if self.governor is not None:
            self.state.governor = self.governor.to_dict()

    def _save(self) -> None:
        self._sync_governor()
        with open(self.state_path, "w") as f:
            json.dump(asdict(self.state), f, indent=2, default=str)
        # convenience CSV exports
        if self.state.journal:
            pd.DataFrame(self.state.journal).to_csv(
                os.path.join(self.dir, "journal.csv"), index=False)
        if self.state.equity_curve:
            pd.DataFrame(self.state.equity_curve).to_csv(
                os.path.join(self.dir, "equity.csv"), index=False)

    # ----------------------- data ------------------------------ #
    def _data(self) -> pd.DataFrame:
        return feeds.get(self.symbol, source=self.source, **self.feed_kwargs)

    def _signal(self, df: pd.DataFrame):
        params = self.state.params or {}
        if self.strategy in daily.LEADER_STRATEGIES:
            leader = feeds.get(self.state.leader_symbol,
                               source=self.state.leader_source or self.source,
                               **self.feed_kwargs)
            return daily.LEADER_STRATEGIES[self.strategy](df, leader, **params)
        if self.strategy in daily.INTRADAY_REGISTRY:
            return daily.INTRADAY_REGISTRY[self.strategy](df, **params)
        fn = strategies.REGISTRY[self.strategy]
        return fn(df, **params) if params else fn(df)

    # ----------------------- core step ------------------------- #
    def _equity(self, mark_close: float) -> float:
        p = Position(**self.state.position)
        if p.is_open:
            return self.state.balance + p.direction * (mark_close - p.entry) * p.shares
        return self.state.balance

    def _log(self, date, action, side, price, shares, reason, pnl, mark_close):
        self.state.journal.append({
            "date": str(date.date() if hasattr(date, "date") else date),
            "action": action, "side": side,
            "price": round(float(price), 4), "shares": round(float(shares), 4),
            "reason": reason, "pnl": round(float(pnl), 2),
            "balance": round(self.state.balance, 2),
            "equity": round(self._equity(mark_close), 2),
        })

    def _process_bar(self, df: pd.DataFrame, i: int, atr_series: pd.Series, sig) -> None:
        self.state.execution = sig.execution
        if sig.execution == "intraday":
            return self._process_bar_intraday(df, i, atr_series, sig)

        date = df.index[i]
        dstr = str(date.date() if hasattr(date, "date") else date)
        o, h, l, c = (float(df["open"].iloc[i]), float(df["high"].iloc[i]),
                      float(df["low"].iloc[i]), float(df["close"].iloc[i]))
        target_sig = int(sig.signal.iloc[i])
        atr = float(atr_series.iloc[i]) if not np.isnan(atr_series.iloc[i]) else 0.0
        fee = self.state.fee
        p = Position(**self.state.position)
        if self.governor:
            prev_eq = (self.state.equity_curve[-1]["equity"]
                       if self.state.equity_curve else self.state.balance)
            self.governor.start_day(dstr, prev_eq)

        # 1) manage an open position on this bar (stop/target intrabar)
        if p.is_open:
            exit_px, reason = None, ""
            if p.direction == 1:
                if l <= p.stop:
                    exit_px, reason = p.stop, "stop"
                elif h >= p.target:
                    exit_px, reason = p.target, "target"
            else:
                if h >= p.stop:
                    exit_px, reason = p.stop, "stop"
                elif l <= p.target:
                    exit_px, reason = p.target, "target"
            if exit_px is None and target_sig != p.direction:
                exit_px, reason = c, "signal"
            if exit_px is not None:
                fill = exit_px * (1 - fee * p.direction)
                pnl = p.direction * (fill - p.entry) * p.shares
                self.state.balance += pnl
                if self.governor:
                    self.governor.on_close(pnl, p.risk_amt, self._equity(c))
                p = Position()  # flat
                self.state.position = asdict(p)
                self._log(date, "CLOSE", "-", fill, 0, reason, pnl, c)

        # 2) open a new position if flat and signalled
        p = Position(**self.state.position)
        if not p.is_open and target_sig != 0 and atr > 0:
            direction = target_sig
            if self._external_block:                   # portfolio correlation veto
                self._log(date, "SKIP", "LONG" if direction == 1 else "SHORT",
                          c, 0, "correlation_block", 0.0, c)
                target_sig = 0
            elif self._news_mult <= 0.0:               # live news veto
                self._log(date, "SKIP", "LONG" if direction == 1 else "SHORT",
                          c, 0, f"news_veto: {self._news_reason}", 0.0, c)
                target_sig = 0
            else:
                entry = c * (1 + fee * direction)          # EOD fill at close
                stop_dist = self.state.stop_atr * atr
                equity_now = self._equity(c)
                risk_amt = equity_now * self.state.risk_per_trade
                shares = risk_amt / stop_dist
                shares = min(shares, equity_now / entry) * self._news_mult  # news sizing
                risk_amt = shares * stop_dist
                gov_ok, gov_why = (self.governor.can_open(risk_amt, equity_now)
                                   if self.governor else (True, ""))
                if not gov_ok:
                    self._log(date, "SKIP", "LONG" if direction == 1 else "SHORT",
                              c, 0, f"risk_halt: {gov_why}", 0.0, c)
                    target_sig = 0
                    p = Position(**self.state.position)   # stay flat
            if target_sig != 0 and self._news_mult > 0.0:
                p = Position(
                    direction=direction, shares=shares, entry=entry,
                    stop=entry - direction * stop_dist,
                    target=entry + direction * self.state.target_atr * atr,
                    risk_amt=risk_amt,
                    entry_date=str(date.date() if hasattr(date, "date") else date),
                )
                self.state.position = asdict(p)
                if self.governor:
                    self.governor.on_open(risk_amt)
                side = "LONG" if direction == 1 else "SHORT"
                reason = "entry" + (f" (news x{self._news_mult:g})"
                                    if self._news_mult < 1.0 else "")
                self._log(date, "OPEN", side, entry, shares, reason, 0.0, c)
                self._route_to_broker(date, direction, shares, entry, p.stop,
                                      p.target, c)

        # 3) record equity + advance the clock
        if self.governor:
            self.governor.observe(self._equity(c))
        self.state.equity_curve.append({
            "date": dstr, "equity": round(self._equity(c), 2),
        })
        self.state.last_date = dstr

    def _process_bar_intraday(self, df, i, atr_series, sig) -> None:
        """
        Intraday (open->close) execution: enter at the open on the signal, exit
        at the close the same day, flat overnight. ATR is used only to size the
        position to the 1%-risk budget; there is no overnight stop order.
        """
        date = df.index[i]
        o = float(df["open"].iloc[i])
        c = float(df["close"].iloc[i])
        target_sig = int(sig.signal.iloc[i])
        atr = float(atr_series.iloc[i]) if not np.isnan(atr_series.iloc[i]) else 0.0
        fee = self.state.fee
        dstr = str(date.date() if hasattr(date, "date") else date)
        if self.governor:
            prev_eq = (self.state.equity_curve[-1]["equity"]
                       if self.state.equity_curve else self.state.balance)
            self.governor.start_day(dstr, prev_eq)

        tradeable = target_sig != 0 and atr > 0 and o > 0
        risk_amt = self.state.balance * self.state.risk_per_trade
        gov_ok, gov_why = (self.governor.can_open(risk_amt, self.state.balance)
                           if (self.governor and tradeable) else (True, ""))

        if tradeable and self._external_block:
            self._log(date, "SKIP", "LONG" if target_sig == 1 else "SHORT",
                      o, 0, "correlation_block", 0.0, o)
        elif tradeable and self._news_mult <= 0.0:
            self._log(date, "SKIP", "LONG" if target_sig == 1 else "SHORT",
                      o, 0, f"news_veto: {self._news_reason}", 0.0, o)
        elif tradeable and not gov_ok:
            self._log(date, "SKIP", "LONG" if target_sig == 1 else "SHORT",
                      o, 0, f"risk_halt: {gov_why}", 0.0, o)
        elif tradeable:
            direction = target_sig
            entry = o * (1 + fee * direction)
            stop_dist = self.state.stop_atr * atr            # sizing proxy only
            equity_now = self.state.balance
            shares = min(risk_amt / stop_dist, equity_now / entry) * self._news_mult
            exit_fill = c * (1 - fee * direction)
            pnl = direction * (exit_fill - entry) * shares
            side = "LONG" if direction == 1 else "SHORT"
            reason = "open" + (f" (news x{self._news_mult:g})" if self._news_mult < 1.0 else "")
            self._log(date, "OPEN", side, entry, shares, reason, 0.0, o)
            self.state.balance += pnl
            self._log(date, "CLOSE", "-", exit_fill, 0, "close", pnl, c)
            if self.governor:
                self.governor.on_close(pnl, shares * stop_dist, self.state.balance)

        # always flat overnight; record equity = realised balance
        if self.governor:
            self.governor.observe(self.state.balance)
        self.state.equity_curve.append({"date": dstr, "equity": round(self.state.balance, 2)})
        self.state.last_date = dstr

    # ----------------------- news overlay ---------------------- #
    def _news_gate(self, direction: int) -> tuple[float, str]:
        """Live news risk overlay for a proposed trade (size multiplier, reason).
        Only consulted in step(); never during historical replay."""
        if not self.news_provider or direction == 0:
            return 1.0, ""
        try:
            from . import news
            items = self.news_provider.fetch(self.symbol, limit=20)
            ov = news.news_overlay(items, direction)
            return ov["size_multiplier"], ov["reason"]
        except Exception as e:
            return 1.0, f"news unavailable ({type(e).__name__})"

    def news_status(self) -> dict:
        """Current news tone for the symbol (live; requires a provider)."""
        if not self.news_provider:
            return {"enabled": False}
        try:
            from . import news
            items = self.news_provider.fetch(self.symbol, limit=20)
            return {"enabled": True, **news.summarise(items)}
        except Exception as e:
            return {"enabled": True, "error": f"{type(e).__name__}"}

    def _route_to_broker(self, date, direction, shares, entry, stop, target, mark):
        """Send a real bracket order — only on the LIVE step path, swing trades.
        Replay never reaches here (self._live stays False)."""
        if self.broker is None or not self._live:
            return
        side = "BUY" if direction == 1 else "SELL"
        qty = max(1, int(round(shares)))           # whole shares for equities
        try:
            if not self.broker.is_connected():
                self.broker.connect()
            res = self.broker.place_bracket(self.broker_symbol, qty=qty, side=side,
                                            entry=entry, stop=stop, target=target)
            self._log(date, "BROKER", side, entry, qty,
                      f"{self.broker.name} order {res.order_id} ({res.status})", 0.0, mark)
        except Exception as e:
            self._log(date, "BROKER_ERR", side, entry, 0,
                      f"{type(e).__name__}: {e}", 0.0, mark)

    # ----------------------- public API ------------------------ #
    def step(self) -> dict:
        """Process the single latest bar (idempotent). Use this in a daily cron."""
        df = self._data()
        atr = ind.atr(df, 14)
        sig = self._signal(df)
        i = len(df) - 1
        bar_date = str(df.index[i].date() if hasattr(df.index[i], "date") else df.index[i])
        if bar_date == self.state.last_date:
            return {"status": "no-new-bar", "last_date": self.state.last_date}
        # consult live news for the direction the strategy wants this bar
        self._news_mult, self._news_reason = self._news_gate(int(sig.signal.iloc[i]))
        news_note = self._news_reason or "n/a"
        self._live = True                                # enable real broker routing
        try:
            self._process_bar(df, i, atr, sig)
        finally:
            self._live = False
            self._news_mult, self._news_reason = 1.0, ""  # reset (no carry to replay)
        self._save()
        return {"status": "stepped", "date": bar_date, "news": news_note,
                "broker": self.broker.name if self.broker else "none",
                "equity": self.state.equity_curve[-1]["equity"]}

    def replay(self, n: int = 30) -> dict:
        """Fast-forward the last ``n`` historical bars (one simulated day each)."""
        df = self._data()
        atr = ind.atr(df, 14)
        sig = self._signal(df)
        start = max(0, len(df) - n)
        processed = 0
        for i in range(start, len(df)):
            bar_date = str(df.index[i].date() if hasattr(df.index[i], "date") else df.index[i])
            if self.state.last_date and bar_date <= self.state.last_date:
                continue
            self._process_bar(df, i, atr, sig)
            processed += 1
        self._save()
        return {"status": "replayed", "bars": processed,
                "equity": self.state.equity_curve[-1]["equity"] if self.state.equity_curve else self.state.capital}

    def reset(self) -> None:
        if os.path.exists(self.state_path):
            os.remove(self.state_path)
        for fn in ("journal.csv", "equity.csv"):
            p = os.path.join(self.dir, fn)
            if os.path.exists(p):
                os.remove(p)

    # ----------------------- reporting ------------------------- #
    def live_metrics(self) -> dict:
        if not self.state.equity_curve:
            return {}
        eq = pd.Series([e["equity"] for e in self.state.equity_curve],
                       index=pd.to_datetime([e["date"] for e in self.state.equity_curve]))
        rets = eq.pct_change().fillna(0.0)
        closed = [j["pnl"] for j in self.state.journal if j["action"] == "CLOSE"]
        out = {
            "days": len(eq),
            "equity": float(eq.iloc[-1]),
            "total_return": float(eq.iloc[-1] / self.state.capital - 1.0),
            "max_drawdown": metrics.max_drawdown(eq),
            "sharpe": metrics.sharpe(rets),
            "closed_trades": len(closed),
            "win_rate": metrics.win_rate(closed) if closed else 0.0,
            "profit_factor": metrics.profit_factor(closed) if closed else 0.0,
        }
        return out

    def readiness(self) -> dict:
        """Conservative go-live gate. Operational + minimal performance checks."""
        m = self.live_metrics()
        checks = []
        def chk(name, ok, detail):
            checks.append({"check": name, "pass": bool(ok), "detail": detail})

        chk("Ran >= 10 sessions", m.get("days", 0) >= 10,
            f"{m.get('days',0)} sessions")
        chk("No catastrophic drawdown (> -10%)", m.get("max_drawdown", -1) > -0.10,
            f"maxDD {m.get('max_drawdown',0)*100:.1f}%")
        chk("Account not in the red (> -2%)", m.get("total_return", -1) > -0.02,
            f"return {m.get('total_return',0)*100:.1f}%")
        chk("At least 3 closed trades", m.get("closed_trades", 0) >= 3,
            f"{m.get('closed_trades',0)} closed")
        chk("Profit factor >= 1 (if traded)",
            m.get("closed_trades", 0) < 3 or m.get("profit_factor", 0) >= 1.0,
            f"PF {m.get('profit_factor',0):.2f}")

        passed = sum(c["pass"] for c in checks)
        if passed == len(checks):
            verdict = "GO — plumbing works and paper results are sane. Start with MINIMUM size."
        elif passed >= len(checks) - 1:
            verdict = "ALMOST — one gate open; run a few more sessions before risking capital."
        else:
            verdict = "NOT READY — do not deploy real money yet."
        return {"verdict": verdict, "passed": passed, "total": len(checks),
                "checks": checks, "metrics": m}

    def report(self) -> str:
        m = self.live_metrics()
        p = Position(**self.state.position)
        lines = [
            "=" * 60,
            f"PAPER ACCOUNT  {self.symbol} / {self.strategy}",
            "=" * 60,
            f"Sessions processed : {m.get('days', 0)}",
            f"Equity             : ${m.get('equity', self.state.capital):,.2f}"
            f"   (start ${self.state.capital:,.0f})",
            f"Total return       : {m.get('total_return', 0)*100:+.2f}%",
            f"Max drawdown       : {m.get('max_drawdown', 0)*100:.2f}%",
            f"Sharpe (so far)    : {m.get('sharpe', 0):.2f}",
            f"Closed trades      : {m.get('closed_trades', 0)}  "
            f"(win {m.get('win_rate', 0)*100:.0f}%, PF {m.get('profit_factor', 0):.2f})",
        ]
        if p.is_open:
            side = "LONG" if p.direction == 1 else "SHORT"
            lines.append(f"OPEN position      : {side} {p.shares:.2f} @ {p.entry:.2f} "
                         f"stop {p.stop:.2f} target {p.target:.2f}")
        else:
            lines.append("OPEN position      : flat")
        if self.strategy == "auto":
            try:
                st = strategies.adaptive_status(self._data())
                lines.append(f"Active strategy    : {st['active_strategy']} "
                             f"(regime: {st['trend']}/{st['volatility']} vol, "
                             f"signal {st['signal']:+d})")
            except Exception:
                pass
        if self.broker is not None:
            conn = "connected" if self.broker.is_connected() else "not connected"
            lines.append(f"Broker             : {self.broker.name} ({conn}) "
                         "— real bracket orders on live --step")
        if self.governor:
            gs = self.governor.status()
            state = ("HALTED" if gs["halted"] else
                     f"cooldown {gs['cooldown_left']}d" if gs["cooldown_left"] else "active")
            lines.append(f"Risk governor      : {state}  "
                         f"(open risk ${gs['open_risk']:.0f}, "
                         f"streak {gs['consecutive_losses']})"
                         + (f"  — {gs['last_reason']}" if gs['last_reason'] else ""))
        if self.news_provider:
            ns = self.news_status()
            if ns.get("enabled") and "error" not in ns:
                lines.append(f"News overlay       : {ns.get('tone','?')}  "
                             f"(avg {ns.get('avg_sentiment',0):+.2f}, "
                             f"{ns.get('n',0)} items, {ns.get('n_events',0)} events)")
            else:
                lines.append(f"News overlay       : enabled ({ns.get('error','no data')})")
        r = self.readiness()
        lines.append("-" * 60)
        lines.append(f"Readiness: {r['passed']}/{r['total']} gates")
        for c in r["checks"]:
            mark = "PASS" if c["pass"] else "FAIL"
            lines.append(f"  [{mark}] {c['check']:<34} {c['detail']}")
        lines.append("-" * 60)
        lines.append(r["verdict"])
        lines.append("=" * 60)
        return "\n".join(lines)
