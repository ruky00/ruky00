"""
Position sizing and risk/reward helpers.

These are the building blocks behind disciplined risk management: never risk
more than a fixed fraction of equity, and know your reward-to-risk before you
enter.
"""

from __future__ import annotations


def position_size(equity: float,
                  risk_per_trade: float,
                  entry: float,
                  stop: float) -> dict:
    """
    Shares to trade so that hitting the stop loses exactly `risk_per_trade`
    of equity.
    """
    risk_amount = equity * risk_per_trade
    per_share_risk = abs(entry - stop)
    if per_share_risk == 0:
        raise ValueError("Entry and stop cannot be equal.")
    shares = risk_amount / per_share_risk
    notional = shares * entry
    return {
        "risk_amount": risk_amount,
        "per_share_risk": per_share_risk,
        "shares": shares,
        "notional": notional,
        "leverage": notional / equity,
    }


def reward_to_risk(entry: float, stop: float, target: float) -> float:
    risk = abs(entry - stop)
    reward = abs(target - entry)
    return reward / risk if risk else float("inf")


def expectancy(win_rate: float, avg_win_R: float, avg_loss_R: float = 1.0) -> float:
    """
    Expected R per trade. avg_loss_R is expressed as a positive number of R.
    Positive expectancy is the minimum bar for a tradeable edge.
    """
    return win_rate * avg_win_R - (1 - win_rate) * avg_loss_R


def kelly_fraction(win_rate: float, reward_to_risk: float) -> float:
    """
    Full-Kelly bet fraction. In practice trade a fraction of this (¼–½ Kelly)
    because inputs are estimated with error.
    """
    b = reward_to_risk
    p = win_rate
    q = 1 - p
    f = (b * p - q) / b
    return max(f, 0.0)


def atr_stop(entry: float, atr: float, direction: int = 1, mult: float = 2.0) -> float:
    """Stop price `mult` ATRs away from entry (direction +1 long / -1 short)."""
    return entry - direction * mult * atr
