"""
Risk governor — portfolio-level circuit breakers ("kill switches").

Per-trade sizing (qflow.risk) protects you on any single trade. The governor
protects the *account* by halting new risk when things go wrong at the
portfolio level:

    max_daily_loss        : stop opening if today's loss exceeds this fraction
    max_drawdown          : hard halt if equity falls this far below its peak
    max_portfolio_heat    : cap total open risk (sum of per-trade $risk / equity)
    max_consecutive_losses: cool down after a losing streak
    cooldown_days         : how long the streak cooldown lasts

It is intentionally simple, serialisable (state survives daily cron runs) and
strategy-agnostic. ``can_open()`` is the gate; ``on_open``/``on_close``/
``observe`` keep the state current.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field

DEFAULT_LIMITS = {
    "max_daily_loss": 0.03,         # -3% on the day -> no new entries today
    "max_drawdown": 0.15,           # -15% from peak -> hard halt
    "max_portfolio_heat": 0.06,     # <=6% of equity at risk across open trades
    "max_consecutive_losses": 5,
    "cooldown_days": 2,
}


@dataclass
class GovernorState:
    peak_equity: float = 0.0
    day: str = ""
    day_start_equity: float = 0.0
    open_risk: float = 0.0          # sum of $risk on open positions
    consecutive_losses: int = 0
    cooldown_left: int = 0          # days remaining in a streak cooldown
    halted: bool = False            # latched by a max_drawdown breach
    last_reason: str = ""


class RiskGovernor:
    def __init__(self, limits: dict | None = None, state: dict | None = None):
        self.limits = {**DEFAULT_LIMITS, **(limits or {})}
        self.state = GovernorState(**state) if state else GovernorState()

    # ----- lifecycle ----- #
    def start_day(self, day: str, equity: float) -> None:
        """Call at the start of each new bar/day before processing it."""
        if self.state.peak_equity == 0.0:
            self.state.peak_equity = equity
        if day != self.state.day:
            # a new calendar day: reset the daily loss anchor, decay cooldown
            self.state.day = day
            self.state.day_start_equity = equity
            if self.state.cooldown_left > 0:
                self.state.cooldown_left -= 1

    def observe(self, equity: float) -> None:
        """Update peak / drawdown halt after equity moves."""
        if equity > self.state.peak_equity:
            self.state.peak_equity = equity
        dd = equity / self.state.peak_equity - 1.0 if self.state.peak_equity else 0.0
        if dd <= -self.limits["max_drawdown"]:
            self.state.halted = True
            self.state.last_reason = (f"max drawdown breached ({dd*100:.1f}% "
                                      f"<= -{self.limits['max_drawdown']*100:.0f}%)")

    # ----- the gate ----- #
    def can_open(self, proposed_risk: float, equity: float) -> tuple[bool, str]:
        s, L = self.state, self.limits
        if s.halted:
            return False, "halted: " + (s.last_reason or "max drawdown")
        if s.cooldown_left > 0:
            return False, f"loss-streak cooldown active ({s.cooldown_left}d left)"
        if s.day_start_equity > 0:
            day_loss = 1.0 - equity / s.day_start_equity
            if day_loss >= L["max_daily_loss"]:
                return False, f"daily loss limit ({day_loss*100:.1f}% >= {L['max_daily_loss']*100:.0f}%)"
        heat = (s.open_risk + proposed_risk) / equity if equity > 0 else 1.0
        if heat > L["max_portfolio_heat"]:
            return False, f"portfolio heat cap ({heat*100:.1f}% > {L['max_portfolio_heat']*100:.0f}%)"
        return True, "ok"

    # ----- bookkeeping ----- #
    def on_open(self, risk_amt: float) -> None:
        self.state.open_risk += risk_amt

    def on_close(self, pnl: float, risk_amt: float, equity: float = 0.0) -> None:
        # NB: peak/drawdown is tracked via observe() once per bar (with the same
        # mark-to-close equity the equity curve records), so we do NOT update the
        # high-water mark here — that would anchor it to realised balance and
        # diverge from the reported curve.
        self.state.open_risk = max(0.0, self.state.open_risk - risk_amt)
        if pnl < 0:
            self.state.consecutive_losses += 1
            if self.state.consecutive_losses >= self.limits["max_consecutive_losses"]:
                self.state.cooldown_left = self.limits["cooldown_days"]
                self.state.consecutive_losses = 0
                self.state.last_reason = "consecutive-loss cooldown"
        else:
            self.state.consecutive_losses = 0

    # ----- io ----- #
    def to_dict(self) -> dict:
        return {"limits": self.limits, "state": asdict(self.state)}

    def status(self) -> dict:
        s = self.state
        return {
            "halted": s.halted,
            "cooldown_left": s.cooldown_left,
            "consecutive_losses": s.consecutive_losses,
            "open_risk": round(s.open_risk, 2),
            "peak_equity": round(s.peak_equity, 2),
            "last_reason": s.last_reason,
        }
