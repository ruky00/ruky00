"""
Funded-account rules engine — pass the prop-firm exam, greedy but not reckless.

A funded / prop-firm challenge is a set of hard constraints on an account:

    profit_target        : reach +X% -> challenge PASSED (lock it in, stop risking)
    max_daily_loss       : lose more than Y% from the day's start -> FAILED
    max_total_drawdown   : fall more than Z% below the anchor -> FAILED
    min_trading_days      : must trade at least N distinct days

This module turns those into a gate the bot must obey, plus a **dynamic position
sizer** that captures "sé codicioso pero sin exagerar las pérdidas":

  * When there is plenty of cushion to the daily / total loss limits, it trades
    at (a boosted) full size to press toward the profit target — greedy.
  * As equity approaches either loss limit, it throttles size down smoothly, and
    once a configurable fraction of the allowance is consumed it stops opening
    for the day (daily) or altogether (total) — so it de-risks *before* it can
    breach, never after.
  * Once the profit target is hit it stops opening new risk: a locked pass is
    worth more than a few extra dollars.

It is deliberately simple and serialisable (state survives cron/daily restarts).
`update()` marks equity each bar, `can_open()` is the gate, `risk_fraction()`
gives the size multiplier, `status()` reports where the challenge stands.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

DEFAULT_RULES = {
    "profit_target": 0.08,        # +8% -> pass
    "max_daily_loss": 0.05,       # -5% on the day -> fail
    "max_total_drawdown": 0.10,   # -10% from the anchor -> fail
    "min_trading_days": 3,
    "trailing_drawdown": False,   # False: anchor = start balance; True: anchor = peak
    # sizing / safety
    "base_risk": 0.004,           # 0.4% risk per trade at full size (the greedy base)
    "max_risk_mult": 1.5,         # up to 1.5x base when cushions are full
    "min_risk_mult": 0.25,        # never below 0.25x base while still trading
    "stop_buffer": 0.80,          # stop opening once 80% of an allowance is consumed
}


@dataclass
class FundedState:
    anchor: float = 0.0           # balance the total-drawdown limit is measured from
    peak: float = 0.0             # high-water mark (for trailing drawdown)
    day: str = ""
    day_start: float = 0.0        # equity at the start of the current day
    equity: float = 0.0           # latest observed equity
    trading_days: int = 0         # distinct days on which we opened a trade
    _traded_today: bool = False
    passed: bool = False          # profit target reached (latched)
    failed: bool = False          # a hard limit breached (latched)
    reason: str = ""


class FundedAccount:
    def __init__(self, rules: dict | None = None, state: dict | None = None,
                 start_equity: float | None = None):
        self.rules = {**DEFAULT_RULES, **(rules or {})}
        self.state = FundedState(**state) if state else FundedState()
        if start_equity is not None:
            self.set_anchor(start_equity)

    # ----- setup ----- #
    def set_anchor(self, equity: float) -> None:
        """Fix the starting balance (the total-drawdown / profit-target anchor)."""
        s = self.state
        if s.anchor == 0.0:
            s.anchor = s.peak = s.day_start = s.equity = float(equity)

    # ----- lifecycle ----- #
    def update(self, equity: float, day: str) -> None:
        """Mark equity for this bar; roll the daily anchor; latch pass/fail."""
        s, R = self.state, self.rules
        if s.anchor == 0.0:
            self.set_anchor(equity)
        s.equity = float(equity)
        if day != s.day:
            if s.day and s._traded_today:
                s.trading_days += 1
            s.day = day
            s.day_start = equity
            s._traded_today = False
        s.peak = max(s.peak, equity)

        # hard limits (latched)
        if not s.failed:
            if self.day_loss() >= R["max_daily_loss"]:
                s.failed, s.reason = True, f"daily loss {self.day_loss()*100:.1f}% >= {R['max_daily_loss']*100:.0f}%"
            elif self.total_drawdown() >= R["max_total_drawdown"]:
                s.failed, s.reason = True, f"total drawdown {self.total_drawdown()*100:.1f}% >= {R['max_total_drawdown']*100:.0f}%"
        if not s.passed and not s.failed and self.profit() >= R["profit_target"]:
            s.passed, s.reason = True, f"profit target reached (+{self.profit()*100:.1f}%)"

    def on_open(self) -> None:
        self.state._traded_today = True

    # ----- measurements ----- #
    def profit(self) -> float:
        s = self.state
        return (s.equity / s.anchor - 1.0) if s.anchor else 0.0

    def day_loss(self) -> float:
        s = self.state
        return max(0.0, 1.0 - s.equity / s.day_start) if s.day_start else 0.0

    def total_drawdown(self) -> float:
        s, R = self.state, self.rules
        base = s.peak if R["trailing_drawdown"] else s.anchor
        return max(0.0, 1.0 - s.equity / base) if base else 0.0

    def daily_cushion(self) -> float:
        """Fraction of today's loss allowance still available (1 = untouched)."""
        lim = self.rules["max_daily_loss"]
        return max(0.0, min(1.0, 1.0 - self.day_loss() / lim)) if lim else 1.0

    def total_cushion(self) -> float:
        """Fraction of the total-drawdown allowance still available."""
        lim = self.rules["max_total_drawdown"]
        return max(0.0, min(1.0, 1.0 - self.total_drawdown() / lim)) if lim else 1.0

    # ----- the gate ----- #
    def can_open(self) -> tuple[bool, str]:
        s, R = self.state, self.rules
        if s.passed:
            return False, "target reached — pass locked, no new risk"
        if s.failed:
            return False, "challenge failed — " + s.reason
        # stop BEFORE breaching: once stop_buffer of an allowance is used up
        floor = 1.0 - R["stop_buffer"]
        if self.daily_cushion() <= floor:
            return False, f"daily loss near limit ({self.day_loss()*100:.1f}%) — stop for today"
        if self.total_cushion() <= floor:
            return False, f"total drawdown near limit ({self.total_drawdown()*100:.1f}%) — stop"
        return True, "ok"

    # ----- dynamic greedy-but-throttled sizing ----- #
    def risk_fraction(self, base: float | None = None) -> float:
        """
        Per-trade risk fraction: scale the base risk by the *tighter* of the two
        loss cushions. Full/boosted size when both cushions are healthy (greedy),
        smoothly smaller as either loss limit approaches (sin exagerar pérdidas).
        """
        R = self.rules
        base = R["base_risk"] if base is None else base
        throttle = min(self.daily_cushion(), self.total_cushion())   # 0..1
        mult = R["min_risk_mult"] + (R["max_risk_mult"] - R["min_risk_mult"]) * throttle
        return base * mult

    # ----- reporting ----- #
    @property
    def done(self) -> bool:
        return self.state.passed or self.state.failed

    def to_dict(self) -> dict:
        return {"rules": self.rules, "state": asdict(self.state)}

    def status(self) -> dict:
        s = self.state
        return {
            "profit": round(self.profit() * 100, 2),
            "target": round(self.rules["profit_target"] * 100, 1),
            "day_loss": round(self.day_loss() * 100, 2),
            "total_dd": round(self.total_drawdown() * 100, 2),
            "daily_cushion": round(self.daily_cushion() * 100, 0),
            "total_cushion": round(self.total_cushion() * 100, 0),
            "trading_days": s.trading_days + (1 if s._traded_today else 0),
            "min_days": self.rules["min_trading_days"],
            "passed": s.passed,
            "failed": s.failed,
            "reason": s.reason,
        }
