"""
Funded-account rules engine — pass the prop-firm exam, greedy but not reckless.

A funded / prop-firm challenge is a set of hard constraints on an account:

    profit_target        : reach +X% -> challenge PASSED (lock it in, stop risking)
    max_daily_loss       : lose more than Y% from the day's start -> FAILED
    max_total_drawdown   : fall more than Z% below the anchor -> FAILED
    consistency_pct      : your single best day must be <= this share of total profit
    min_trading_days      : must trade at least N distinct days

Drawdown can be measured three ways (`drawdown_mode`):
    "static"   : floor fixed at (start balance - allowance).
    "trailing" : floor trails the intraday equity peak.
    "eod"      : floor trails the highest END-OF-DAY balance — updates once a day
                 at the close, not on intraday swings (this is Lucid's model).

This turns those into a gate the bot must obey, plus a **dynamic position sizer**
that captures "sé codicioso pero sin exagerar las pérdidas":

  * Plenty of cushion to the loss limits -> trade at (boosted) full size to press
    toward the target — greedy.
  * As equity nears either loss limit, throttle size down smoothly, and once a
    configurable fraction of an allowance is consumed, stop opening for the day
    (daily) or altogether (total) — de-risk *before* it can breach, never after.
  * Once the profit target is hit (and consistency is satisfied), stop opening:
    a locked pass is worth more than a few extra dollars.
  * Consistency pacing: once today's profit already fills the allowed share of
    total profit, stop adding to it — spread the gains across more days so the
    "best day <= X% of total" rule is met.

Simple, serialisable (state survives cron/daily restarts). `update()` marks
equity each bar, `can_open()` is the gate, `risk_fraction()` gives the size
multiplier, `status()` reports where the challenge stands.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

DEFAULT_RULES = {
    "profit_target": 0.08,        # +8% -> pass
    "max_daily_loss": 0.05,       # -5% on the day -> fail (1.0 = "no daily limit")
    "max_total_drawdown": 0.10,   # -10% allowance below the (sliding) high-water base
    "drawdown_mode": "static",    # "static" | "trailing" | "eod"
    "consistency_pct": 1.0,       # best day must be <= this * total profit (1.0 = off)
    "min_trading_days": 3,
    # sizing / safety
    "base_risk": 0.004,           # 0.4% risk per trade at full size (the greedy base)
    "max_risk_mult": 1.5,         # up to 1.5x base when cushions are full
    "min_risk_mult": 0.25,        # never below 0.25x base while still trading
    "stop_buffer": 0.80,          # stop opening once 80% of an allowance is consumed
}

# Lucid Trading presets (futures, EOD trailing drawdown, 50% consistency on eval).
# Dollar amounts per account size; daily-loss figures marked (~) are approximate —
# confirm the exact DLL for your plan and override if needed. Target & trailing
# drawdown are well-sourced. See docs/FUNDED.md.
LUCID_PRESETS = {
    25:  {"balance": 25_000,  "profit_target": 1_250, "max_total_drawdown": 1_000, "max_daily_loss": None},
    50:  {"balance": 50_000,  "profit_target": 3_000, "max_total_drawdown": 2_000, "max_daily_loss": 1_200},
    100: {"balance": 100_000, "profit_target": 6_000, "max_total_drawdown": 3_000, "max_daily_loss": 2_000},
    150: {"balance": 150_000, "profit_target": 9_000, "max_total_drawdown": 4_500, "max_daily_loss": 2_700},
}

# FundedNext presets (CFD/Forex on MT4/MT5/cTrader). Percentage-based, static
# overall drawdown (from initial balance), daily loss from the day's start, no
# consistency rule by default, unlimited time. Confirm the exact target for your
# chosen model on the FundedNext dashboard and override if needed. See docs/FUNDED.md.
FUNDEDNEXT_PRESETS = {
    "stellar_1step":    {"profit_target": 0.10, "max_daily_loss": 0.03, "max_total_drawdown": 0.06, "min_days": 2},
    "stellar_2step_p1": {"profit_target": 0.08, "max_daily_loss": 0.05, "max_total_drawdown": 0.10, "min_days": 5},
    "stellar_2step_p2": {"profit_target": 0.05, "max_daily_loss": 0.05, "max_total_drawdown": 0.10, "min_days": 5},
    "express":          {"profit_target": 0.25, "max_daily_loss": 0.05, "max_total_drawdown": 0.10, "min_days": 10},
}


@dataclass
class FundedState:
    anchor: float = 0.0           # starting balance (target + allowances measured from it)
    peak: float = 0.0             # intraday equity high-water mark (trailing mode)
    eod_high: float = 0.0         # highest end-of-day balance (eod mode)
    day: str = ""
    day_start: float = 0.0        # equity at the start of the current day
    equity: float = 0.0           # latest observed equity
    other_days_best: float = 0.0  # best profit of any COMPLETED day (for consistency)
    trading_days: int = 0         # distinct days on which we opened a trade
    _traded_today: bool = False
    passed: bool = False          # profit target reached + consistency ok (latched)
    failed: bool = False          # a hard limit breached (latched)
    reason: str = ""


class FundedAccount:
    def __init__(self, rules: dict | None = None, state: dict | None = None,
                 start_equity: float | None = None):
        rules = dict(rules or {})
        # back-compat: trailing_drawdown=True -> drawdown_mode="trailing"
        if rules.pop("trailing_drawdown", False) and "drawdown_mode" not in rules:
            rules["drawdown_mode"] = "trailing"
        self.rules = {**DEFAULT_RULES, **rules}
        self.state = FundedState(**state) if state else FundedState()
        if start_equity is not None:
            self.set_anchor(start_equity)

    # ----- presets ----- #
    @classmethod
    def from_lucid(cls, size: int = 50, consistency: float = 0.5, **overrides):
        """Build a FundedAccount from a Lucid Trading account size (25/50/100/150)."""
        if size not in LUCID_PRESETS:
            raise ValueError(f"Lucid size {size} not in {sorted(LUCID_PRESETS)}")
        p = LUCID_PRESETS[size]
        bal = p["balance"]
        rules = {
            "profit_target": p["profit_target"] / bal,
            "max_total_drawdown": p["max_total_drawdown"] / bal,
            # no DLL -> 1.0 so the daily gate never binds
            "max_daily_loss": (p["max_daily_loss"] / bal) if p["max_daily_loss"] else 1.0,
            "drawdown_mode": "eod",          # Lucid uses EOD trailing drawdown
            "consistency_pct": consistency,  # eval: best day <= 50% of total profit
            "min_trading_days": 1,
        }
        return cls({**rules, **overrides}, start_equity=bal)

    @classmethod
    def from_fundednext(cls, model: str = "stellar_2step_p1", consistency: float = 1.0,
                        start_equity: float | None = None, **overrides):
        """Build a FundedAccount from a FundedNext model (percentage-based rules)."""
        key = model.lower()
        if key not in FUNDEDNEXT_PRESETS:
            raise ValueError(f"FundedNext model {model!r} not in {sorted(FUNDEDNEXT_PRESETS)}")
        p = FUNDEDNEXT_PRESETS[key]
        rules = {
            "profit_target": p["profit_target"],
            "max_daily_loss": p["max_daily_loss"],
            "max_total_drawdown": p["max_total_drawdown"],
            "drawdown_mode": "static",           # FundedNext overall DD is static from balance
            "consistency_pct": consistency,      # no consistency rule by default
            "min_trading_days": p["min_days"],
        }
        return cls({**rules, **overrides}, start_equity=start_equity)

    # ----- setup ----- #
    def set_anchor(self, equity: float) -> None:
        s = self.state
        if s.anchor == 0.0:
            s.anchor = s.peak = s.eod_high = s.day_start = s.equity = float(equity)

    # ----- lifecycle ----- #
    def update(self, equity: float, day: str) -> None:
        """Mark equity for this bar; roll daily/EOD anchors; latch pass/fail."""
        s, R = self.state, self.rules
        if s.anchor == 0.0:
            self.set_anchor(equity)
        prev_equity = s.equity
        if day != s.day:
            if s.day:                                    # a day just completed
                s.eod_high = max(s.eod_high, prev_equity)          # EOD high-water
                s.other_days_best = max(s.other_days_best,          # consistency book
                                        prev_equity - s.day_start)
                if s._traded_today:
                    s.trading_days += 1
            s.day = day
            s.day_start = equity
            s._traded_today = False
        s.equity = float(equity)
        s.peak = max(s.peak, equity)

        # hard limits (latched)
        if not s.failed:
            if self.day_loss() >= R["max_daily_loss"]:
                s.failed, s.reason = True, f"daily loss {self.day_loss()*100:.1f}% >= {R['max_daily_loss']*100:.0f}%"
            elif self.total_cushion() <= 0.0:
                s.failed, s.reason = True, f"total drawdown breached (floor {self.dd_floor():,.0f})"
        if (not s.passed and not s.failed
                and self.profit() >= R["profit_target"] and self.consistency_ok()):
            s.passed, s.reason = True, f"profit target reached (+{self.profit()*100:.1f}%), consistency ok"

    def on_open(self) -> None:
        self.state._traded_today = True

    # ----- measurements ----- #
    def profit(self) -> float:
        s = self.state
        return (s.equity / s.anchor - 1.0) if s.anchor else 0.0

    def day_loss(self) -> float:
        s = self.state
        return max(0.0, 1.0 - s.equity / s.day_start) if s.day_start else 0.0

    def _dd_base(self) -> float:
        s, mode = self.state, self.rules["drawdown_mode"]
        if mode == "trailing":
            return s.peak
        if mode == "eod":
            return s.eod_high
        return s.anchor                                   # static

    def dd_amount(self) -> float:
        """Drawdown allowance in dollars (fixed size, from the anchor)."""
        return self.rules["max_total_drawdown"] * self.state.anchor

    def dd_floor(self) -> float:
        """Equity level that fails the account (sliding high-water base - allowance)."""
        return self._dd_base() - self.dd_amount()

    def total_drawdown(self) -> float:
        """Drawdown from the sliding base, as a fraction of the anchor (reporting)."""
        s = self.state
        return max(0.0, (self._dd_base() - s.equity) / s.anchor) if s.anchor else 0.0

    def daily_cushion(self) -> float:
        lim = self.rules["max_daily_loss"]
        return max(0.0, min(1.0, 1.0 - self.day_loss() / lim)) if lim else 1.0

    def total_cushion(self) -> float:
        """Fraction of the drawdown allowance still available (0 = at the floor)."""
        amt = self.dd_amount()
        if amt <= 0:
            return 1.0
        return max(0.0, min(1.0, (self.state.equity - self.dd_floor()) / amt))

    # ----- consistency ----- #
    def _max_day_profit(self) -> float:
        s = self.state
        return max(s.other_days_best, s.equity - s.day_start)

    def consistency_ok(self) -> bool:
        """Best single day <= consistency_pct of total profit (trivially ok if off/flat)."""
        R, s = self.rules, self.state
        if R["consistency_pct"] >= 1.0:
            return True
        total = s.equity - s.anchor
        if total <= 0:
            return False                                  # no profit yet -> can't pass
        return self._max_day_profit() <= R["consistency_pct"] * total + 1e-9

    def _consistency_blocks_today(self) -> bool:
        """True if banking more profit TODAY would break the best-day rule."""
        R, s = self.rules, self.state
        if R["consistency_pct"] >= 1.0:
            return False
        total = s.equity - s.anchor
        today = s.equity - s.day_start
        return total > 0 and today > 0 and today >= R["consistency_pct"] * total

    # ----- the gate ----- #
    def can_open(self) -> tuple[bool, str]:
        s, R = self.state, self.rules
        if s.passed:
            return False, "target reached — pass locked, no new risk"
        if s.failed:
            return False, "challenge failed — " + s.reason
        floor = 1.0 - R["stop_buffer"]                    # stop BEFORE breaching
        if self.daily_cushion() <= floor:
            return False, f"daily loss near limit ({self.day_loss()*100:.1f}%) — stop for today"
        if self.total_cushion() <= floor:
            return False, f"drawdown near floor ({self.dd_floor():,.0f}) — stop"
        if self._consistency_blocks_today():
            return False, "consistency: big day — bank the rest another day"
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
            "dd_floor": round(self.dd_floor(), 0),
            "daily_cushion": round(self.daily_cushion() * 100, 0),
            "total_cushion": round(self.total_cushion() * 100, 0),
            "consistency_ok": self.consistency_ok(),
            "trading_days": s.trading_days + (1 if s._traded_today else 0),
            "min_days": self.rules["min_trading_days"],
            "passed": s.passed,
            "failed": s.failed,
            "reason": s.reason,
        }
