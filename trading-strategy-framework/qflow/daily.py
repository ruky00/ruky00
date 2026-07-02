"""
Intraday (open->close) edge strategies, ready for the paper-trading engine.

These were discovered by ``qflow.anomalies`` and are expressed here in the same
``StrategySignal`` interface the rest of the framework uses, but flagged
``execution="intraday"``: the position is decided at (or before) the open, held
through the session, and closed at the bell — flat overnight.

    gap_fade  : fade an opening gap on volatile names
    lead_lag  : trade a follower off a *leader* asset's prior-day move
                (the "one market leads another the next session" trade)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .strategies import StrategySignal


def gap_fade(df: pd.DataFrame, threshold: float = 0.005) -> StrategySignal:
    """
    Signal is known at the open: if today gapped DOWN more than `threshold`
    versus yesterday's close, go long the session (expect the gap to fill);
    if it gapped UP, go short. Best on high-volatility instruments.
    """
    overnight = df["open"] / df["close"].shift(1) - 1.0
    signal = pd.Series(0, index=df.index)
    signal[overnight < -threshold] = 1
    signal[overnight > threshold] = -1
    signal = signal.fillna(0).astype(int)
    return StrategySignal(signal=signal, atr=None, execution="intraday",
                          params=dict(threshold=threshold))


def lead_lag(df: pd.DataFrame,
             leader: pd.DataFrame,
             lag: int = 1,
             threshold: float = 0.005) -> StrategySignal:
    """
    `df` is the FOLLOWER you trade; `leader` is the asset that moves first.
    If the leader's return `lag` days ago exceeded +threshold, go long the
    follower's session today; if it fell below -threshold, go short. The signal
    is known before today's open, so an open->close hold has no look-ahead.
    """
    leader_ret = leader["close"].pct_change().reindex(df.index)
    led = leader_ret.shift(lag)
    signal = pd.Series(
        np.where(led > threshold, 1, np.where(led < -threshold, -1, 0)),
        index=df.index,
    ).fillna(0).astype(int)
    return StrategySignal(signal=signal, atr=None, execution="intraday",
                          params=dict(lag=lag, threshold=threshold))


# Strategies that need only the traded symbol's own data.
INTRADAY_REGISTRY = {
    "gap_fade": gap_fade,
}

# Strategies that need a second (leader) series; handled specially by the
# paper engine, which fetches the leader symbol.
LEADER_STRATEGIES = {
    "lead_lag": lead_lag,
}
