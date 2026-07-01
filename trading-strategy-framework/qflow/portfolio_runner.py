"""
Multi-symbol portfolio runner — group the bot's execution across a basket.

Runs one strategy (default the adaptive ``auto``) over several symbols at once:

    * one PaperTrader per symbol (each with its own persistent state),
    * ALL sharing a single broker connection (one IBKR session) and news overlay,
    * capital split across symbols (equal or inverse-volatility),
    * per-symbol kill-switches PLUS a portfolio-level drawdown circuit breaker.

This is the "grouped execution" entrypoint: instead of babysitting one ticker,
you point it at a basket of volatile names and it trades them all, letting the
``auto`` strategy pick the right sub-strategy per symbol from the candles.

Live orders only fire on ``step_all`` (never in ``replay_all``), inherited from
PaperTrader.
"""

from __future__ import annotations

import json
import os

import pandas as pd

from . import feeds, strategies, portfolio as port
from .paper import PaperTrader, DEFAULT_ROOT


class PortfolioRunner:
    def __init__(self,
                 symbols: list[str],
                 strategy: str = "auto",
                 source: str = "yahoo",
                 total_capital: float = 10_000.0,
                 risk_per_trade: float = 0.005,
                 allocation: str = "equal",          # "equal" | "inverse_vol"
                 broker=None,
                 news_provider=None,
                 risk_limits: dict | None = None,
                 feed_kwargs: dict | None = None,
                 broker_symbols: dict | None = None,
                 currency: str = "USD",
                 exchange: str = "SMART",
                 primary: str = "",
                 portfolio_max_drawdown: float = 0.15,
                 root: str = DEFAULT_ROOT,
                 name: str = "portfolio",
                 reset: bool = False):
        if not symbols:
            raise ValueError("Need at least one symbol.")
        self.symbols = symbols
        self.broker = broker
        self.portfolio_max_drawdown = portfolio_max_drawdown
        self.dir = os.path.join(root, name)
        os.makedirs(self.dir, exist_ok=True)
        self.state_path = os.path.join(self.dir, "portfolio.json")
        if reset:
            self._wipe_state(symbols, strategy, root)      # start fresh before building
        self._pstate = self._load_state()

        weights = self._weights(symbols, source, allocation, feed_kwargs or {})
        self.traders: dict[str, PaperTrader] = {}
        for sym in symbols:
            self.traders[sym] = PaperTrader(
                symbol=sym, source=source, strategy=strategy,
                capital=total_capital * weights[sym],
                risk_per_trade=risk_per_trade,
                broker=broker, news_provider=news_provider,
                risk_limits=dict(risk_limits) if risk_limits else None,
                feed_kwargs=feed_kwargs,
                broker_symbol=(broker_symbols or {}).get(sym, ""),
                root=root,
            )
        # apply order currency/exchange to the shared IBKR broker if present
        if broker is not None and hasattr(broker, "currency"):
            broker.currency, broker.exchange = currency, exchange
            broker.primary_exchange = primary

    # ----------------------- allocation ----------------------- #
    @staticmethod
    def _weights(symbols, source, allocation, feed_kwargs) -> dict:
        if allocation == "inverse_vol":
            closes = {}
            for s in symbols:
                try:
                    closes[s] = feeds.get(s, source=source, **feed_kwargs)["close"]
                except Exception:
                    pass
            if len(closes) == len(symbols):
                rets = pd.DataFrame(closes).pct_change().dropna()
                w = port.inverse_vol_weights(rets)
                return {s: float(w.get(s, 0.0)) for s in symbols}
        # equal weight (fallback)
        return {s: 1.0 / len(symbols) for s in symbols}

    def _wipe_state(self, symbols, strategy, root):
        """Delete persisted state so traders rebuild fresh (call before building)."""
        for fn in ("portfolio.json",):
            p = os.path.join(self.dir, fn)
            if os.path.exists(p):
                os.remove(p)
        for sym in symbols:
            d = os.path.join(root, f"{sym}_{strategy}")
            for fn in ("state.json", "journal.csv", "equity.csv"):
                p = os.path.join(d, fn)
                if os.path.exists(p):
                    os.remove(p)

    # ----------------------- portfolio state ------------------ #
    def _load_state(self) -> dict:
        if os.path.exists(self.state_path):
            with open(self.state_path) as f:
                return json.load(f)
        return {"peak_equity": 0.0, "halted": False}

    def _save_state(self):
        with open(self.state_path, "w") as f:
            json.dump(self._pstate, f, indent=2)

    def portfolio_equity(self) -> float:
        return float(sum(t.live_metrics().get("equity", t.state.capital)
                         for t in self.traders.values()))

    def _portfolio_guard(self) -> tuple[bool, str]:
        """Latch a halt if total equity falls too far below its peak."""
        eq = self.portfolio_equity()
        if eq > self._pstate["peak_equity"]:
            self._pstate["peak_equity"] = eq
        peak = self._pstate["peak_equity"] or eq
        dd = eq / peak - 1.0 if peak else 0.0
        if dd <= -self.portfolio_max_drawdown:
            self._pstate["halted"] = True
        self._save_state()
        if self._pstate["halted"]:
            return False, f"PORTFOLIO HALT: drawdown {dd*100:.1f}%"
        return True, "ok"

    # ----------------------- execution ------------------------ #
    def connect_broker(self):
        if self.broker is not None and not self.broker.is_connected():
            self.broker.connect()

    def step_all(self) -> list[dict]:
        ok, why = self._portfolio_guard()
        if not ok:
            return [{"status": "portfolio-halted", "reason": why}]
        self.connect_broker()
        results = []
        for sym, t in self.traders.items():
            try:
                res = t.step()
            except Exception as e:
                res = {"status": "error", "error": f"{type(e).__name__}: {e}"}
            results.append({"symbol": sym, **res})
        self._portfolio_guard()   # refresh peak/halt after stepping
        return results

    def replay_all(self, n: int = 30) -> list[dict]:
        results = []
        for sym, t in self.traders.items():
            results.append({"symbol": sym, **t.replay(n)})
        self._portfolio_guard()
        return results

    def reset(self):
        for t in self.traders.values():
            t.reset()
        if os.path.exists(self.state_path):
            os.remove(self.state_path)
        self._pstate = {"peak_equity": 0.0, "halted": False}

    # ----------------------- reporting ------------------------ #
    def report(self) -> str:
        lines = ["=" * 78, f"PORTFOLIO — {len(self.symbols)} symbols", "=" * 78,
                 f"{'symbol':<8}{'equity':>12}{'ret%':>8}{'maxDD%':>8}"
                 f"{'trades':>7}{'ready':>7}  active/position"]
        lines.append("-" * 78)
        total_cap = 0.0
        go = 0
        for sym, t in self.traders.items():
            m = t.live_metrics()
            total_cap += t.state.capital
            r = t.readiness()
            ok = "GO" if r["passed"] == r["total"] else f"{r['passed']}/{r['total']}"
            if r["passed"] == r["total"]:
                go += 1
            active = ""
            if t.strategy == "auto":
                try:
                    active = strategies.adaptive_status(t._data())["active_strategy"]
                except Exception:
                    active = "?"
            posd = t.state.position.get("direction", 0)
            pos = "flat" if not posd else ("LONG" if posd == 1 else "SHORT")
            lines.append(f"{sym:<8}{m.get('equity',t.state.capital):>12,.0f}"
                         f"{m.get('total_return',0)*100:>7.1f}%"
                         f"{m.get('max_drawdown',0)*100:>7.1f}%"
                         f"{m.get('closed_trades',0):>7}{ok:>7}  {active} {pos}")
        eq = self.portfolio_equity()
        peak = self._pstate["peak_equity"] or eq
        lines.append("-" * 78)
        lines.append(f"TOTAL equity ${eq:,.0f}  (start ${total_cap:,.0f}, "
                     f"return {(eq/total_cap-1)*100:+.1f}%, "
                     f"portfolio DD {(eq/peak-1)*100:.1f}%)  "
                     f"symbols at GO: {go}/{len(self.symbols)}")
        if self._pstate["halted"]:
            lines.append("!! PORTFOLIO HALTED by drawdown circuit breaker — no new entries.")
        lines.append("=" * 78)
        return "\n".join(lines)
