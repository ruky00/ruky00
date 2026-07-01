"""
Trade journal — append-only CSV of every entry and exit for live review.

The bot's brackets live on IBKR, so exits (stop/target) happen there, not in the
script. This journal records what the bot *did*: one row per entry (with the
strategy, interval, risk%, SL/TP it chose) and one row per exit it detects when a
held position disappears (approximate P&L from the last mark). Read it back to
compute a live win-rate, average R, and expectancy — the evidence that decides
whether the edge is real before risking a funded account.

    from qflow.journal import TradeJournal
    j = TradeJournal("logs/bot_journal.csv")
    j.log_entry(ts, "NVDA", "BUY", qty=120, entry=1000.0, sl=990.0, tp=1020.0,
                risk_pct=0.6, strategy="vwap_reversion", interval="30m")
    ...
    j.log_exit(ts, "NVDA", exit_price=1020.0, pnl=2400.0, reason="target")
    print(j.summary())      # {'entries':.., 'exits':.., 'win_rate':.., 'expectancy':..}

Dependency-light: standard-library csv only; safe to call from the live loop.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime

FIELDS = ["time", "event", "symbol", "side", "qty", "price", "sl", "tp",
          "risk_pct", "strategy", "interval", "pnl", "reason"]


class TradeJournal:
    def __init__(self, path: str = "logs/bot_journal.csv"):
        self.path = path
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    def _append(self, row: dict) -> None:
        row.setdefault("time", datetime.now().isoformat(timespec="seconds"))
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(
                {k: row.get(k, "") for k in FIELDS})

    # ----- writers ----- #
    def log_entry(self, ts, symbol, side, qty, entry, sl, tp, risk_pct,
                  strategy="", interval="") -> None:
        self._append({"time": _fmt(ts), "event": "ENTRY", "symbol": symbol,
                      "side": side, "qty": qty, "price": round(entry, 4),
                      "sl": sl, "tp": tp, "risk_pct": round(risk_pct, 3),
                      "strategy": strategy, "interval": interval})

    def log_exit(self, ts, symbol, exit_price, pnl, reason="exit") -> None:
        self._append({"time": _fmt(ts), "event": "EXIT", "symbol": symbol,
                      "price": round(exit_price, 4), "pnl": round(pnl, 2),
                      "reason": reason})

    # ----- readback ----- #
    def rows(self) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        with open(self.path, newline="") as f:
            return list(csv.DictReader(f))

    def summary(self) -> dict:
        rows = self.rows()
        entries = [r for r in rows if r["event"] == "ENTRY"]
        exits = [r for r in rows if r["event"] == "EXIT" and r["pnl"] not in ("", None)]
        pnls = [float(r["pnl"]) for r in exits]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        n = len(pnls)
        win_rate = len(wins) / n if n else 0.0
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        expectancy = (sum(pnls) / n) if n else 0.0
        return {
            "entries": len(entries),
            "exits": n,
            "net_pnl": round(sum(pnls), 2),
            "win_rate": round(win_rate, 3),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "expectancy": round(expectancy, 2),
        }


def _fmt(ts) -> str:
    try:
        return ts.isoformat(timespec="seconds")
    except (AttributeError, TypeError):
        return str(ts)
