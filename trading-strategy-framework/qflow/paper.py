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

from . import feeds, strategies, indicators as ind, metrics

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
                 feed_kwargs: dict | None = None):
        if strategy not in strategies.REGISTRY:
            raise ValueError(f"Unknown strategy {strategy!r}. "
                             f"Choose from {list(strategies.REGISTRY)}.")
        self.symbol = symbol
        self.source = source
        self.strategy = strategy
        self.feed_kwargs = feed_kwargs or {}
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
            )
            self._save()

    # ----------------------- persistence ----------------------- #
    def _load(self) -> State:
        with open(self.state_path) as f:
            return State(**json.load(f))

    def _save(self) -> None:
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
        fn = strategies.REGISTRY[self.strategy]
        sig = fn(df, **self.state.params) if self.state.params else fn(df)
        return sig

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
        date = df.index[i]
        o, h, l, c = (float(df["open"].iloc[i]), float(df["high"].iloc[i]),
                      float(df["low"].iloc[i]), float(df["close"].iloc[i]))
        target_sig = int(sig.signal.iloc[i])
        atr = float(atr_series.iloc[i]) if not np.isnan(atr_series.iloc[i]) else 0.0
        fee = self.state.fee
        p = Position(**self.state.position)

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
                p = Position()  # flat
                self.state.position = asdict(p)
                self._log(date, "CLOSE", "-", fill, 0, reason, pnl, c)

        # 2) open a new position if flat and signalled
        p = Position(**self.state.position)
        if not p.is_open and target_sig != 0 and atr > 0:
            direction = target_sig
            entry = c * (1 + fee * direction)          # EOD fill at close
            stop_dist = self.state.stop_atr * atr
            equity_now = self._equity(c)
            risk_amt = equity_now * self.state.risk_per_trade
            shares = risk_amt / stop_dist
            shares = min(shares, equity_now / entry)   # no leverage
            risk_amt = shares * stop_dist
            p = Position(
                direction=direction, shares=shares, entry=entry,
                stop=entry - direction * stop_dist,
                target=entry + direction * self.state.target_atr * atr,
                risk_amt=risk_amt,
                entry_date=str(date.date() if hasattr(date, "date") else date),
            )
            self.state.position = asdict(p)
            side = "LONG" if direction == 1 else "SHORT"
            self._log(date, "OPEN", side, entry, shares, "entry", 0.0, c)

        # 3) record equity + advance the clock
        self.state.equity_curve.append({
            "date": str(date.date() if hasattr(date, "date") else date),
            "equity": round(self._equity(c), 2),
        })
        self.state.last_date = str(date.date() if hasattr(date, "date") else date)

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
        self._process_bar(df, i, atr, sig)
        self._save()
        return {"status": "stepped", "date": bar_date,
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
