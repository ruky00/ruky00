"""
A compact long/short backtester.

Design goals
------------
* Risk-based position sizing: each trade risks a fixed % of equity, with the
  distance to the stop (in ATR units) determining share count.
* Realistic frictions: commission + slippage in basis points.
* Trade ledger: every closed trade is recorded so win rate, profit factor and
  Monte-Carlo bootstrapping all work off real fills.

The engine consumes a *target signal* (-1 short / 0 flat / +1 long) produced by
a strategy, plus an ATR series used to place stops and take-profits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import metrics


@dataclass
class Trade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    direction: int          # +1 long, -1 short
    entry: float
    exit: float
    shares: float
    pnl: float
    r_multiple: float       # pnl measured in units of initial risk
    reason: str


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    trades: list = field(default_factory=list)

    @property
    def trade_pnls(self):
        return [t.pnl for t in self.trades]

    @property
    def r_multiples(self):
        return [t.r_multiple for t in self.trades]

    def stats(self) -> dict:
        return metrics.summary(self.equity, self.returns, self.trade_pnls)

    def report(self) -> str:
        return metrics.summary_table(self.stats())


def run_backtest(
    df: pd.DataFrame,
    signal: pd.Series,
    atr: pd.Series,
    capital: float = 10_000.0,
    risk_per_trade: float = 0.01,     # 1% of equity at risk
    stop_atr: float = 2.0,            # stop distance in ATR multiples
    target_atr: float = 4.0,          # take-profit distance in ATR multiples
    commission_bps: float = 2.0,
    slippage_bps: float = 2.0,
    allow_short: bool = True,
) -> BacktestResult:
    """
    Walk bar-by-bar. A position is opened when the target signal flips to
    +/-1 while flat. It is closed on stop, target, or signal reversal/exit.
    Fills happen at the next bar's open to avoid look-ahead.
    """
    close = df["close"].values
    open_ = df["open"].values
    high = df["high"].values
    low = df["low"].values
    sig = signal.reindex(df.index).fillna(0).values
    atr_v = atr.reindex(df.index).bfill().values

    fee = (commission_bps + slippage_bps) / 1e4

    equity = capital
    equity_curve = np.empty(len(df))
    pos = 0                # current direction
    shares = 0.0
    entry_px = 0.0
    stop_px = 0.0
    target_px = 0.0
    risk_amt = 0.0
    entry_idx = 0
    trades: list[Trade] = []

    for i in range(len(df)):
        # ---- manage an open position using this bar's range ----
        if pos != 0:
            exit_px = None
            reason = ""
            if pos == 1:
                if low[i] <= stop_px:
                    exit_px, reason = stop_px, "stop"
                elif high[i] >= target_px:
                    exit_px, reason = target_px, "target"
            else:  # short
                if high[i] >= stop_px:
                    exit_px, reason = stop_px, "stop"
                elif low[i] <= target_px:
                    exit_px, reason = target_px, "target"

            # Signal-driven exit (flip to flat or opposite) at this close
            if exit_px is None and sig[i] != pos:
                exit_px, reason = close[i], "signal"

            if exit_px is not None:
                fill = exit_px * (1 - fee * pos)  # slippage against us
                pnl = pos * (fill - entry_px) * shares
                equity += pnl
                r_mult = pnl / risk_amt if risk_amt else 0.0
                trades.append(
                    Trade(
                        entry_date=df.index[entry_idx],
                        exit_date=df.index[i],
                        direction=pos,
                        entry=entry_px,
                        exit=fill,
                        shares=shares,
                        pnl=pnl,
                        r_multiple=r_mult,
                        reason=reason,
                    )
                )
                pos = 0
                shares = 0.0

        # ---- open a new position if flat and signalled ----
        if pos == 0 and sig[i] != 0 and i + 1 < len(df):
            direction = int(sig[i])
            if direction == -1 and not allow_short:
                pass
            else:
                a = atr_v[i]
                if a > 0 and not np.isnan(a):
                    nxt_open = open_[i + 1]
                    entry_px = nxt_open * (1 + fee * direction)
                    stop_dist = stop_atr * a
                    risk_amt = equity * risk_per_trade
                    shares = risk_amt / stop_dist
                    # cap leverage at 1x notional of equity
                    max_shares = equity / entry_px
                    shares = min(shares, max_shares)
                    risk_amt = shares * stop_dist
                    stop_px = entry_px - direction * stop_dist
                    target_px = entry_px + direction * target_atr * a
                    pos = direction
                    entry_idx = i + 1

        equity_curve[i] = equity

    eq = pd.Series(equity_curve, index=df.index, name="equity")
    rets = eq.pct_change().fillna(0.0)
    return BacktestResult(equity=eq, returns=rets, trades=trades)
