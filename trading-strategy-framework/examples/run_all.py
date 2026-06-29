"""
End-to-end demonstration of the qflow framework.

Runs offline on synthetic regime-aware data and exercises every module:
backtest, risk/reward, regime detection, multi-factor, Monte-Carlo, drawdown
and portfolio construction.

    python examples/run_all.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from qflow import (
    data, strategies, backtest, metrics, regime,
    multifactor, montecarlo, portfolio, risk,
)

CAPITAL = 10_000.0
RISK = 0.01

pd.set_option("display.width", 120)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")


def section(title):
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def run():
    df = data.synthetic_ohlcv(n_days=2520, seed=42)
    print(f"Loaded {len(df)} bars  {df.index[0].date()} -> {df.index[-1].date()}")

    # ---- 1 & 2: strategy backtests -------------------------------------- #
    section("BACKTEST — three reference strategies (capital $10k, 1% risk)")
    results = {}
    for name, fn in strategies.REGISTRY.items():
        sig = fn(df)
        res = backtest.run_backtest(
            df, sig.signal, sig.atr,
            capital=CAPITAL, risk_per_trade=RISK,
        )
        results[name] = res
        print(f"\n# {name}   params={sig.params}")
        print(res.report())

    # ---- 3: risk / reward ---------------------------------------------- #
    section("RISK / REWARD — example long setup")
    entry, stop, target = 100.0, 96.0, 108.0
    sz = risk.position_size(CAPITAL, RISK, entry, stop)
    print(f"Entry {entry}  Stop {stop}  Target {target}")
    print(f"Reward:Risk        = {risk.reward_to_risk(entry, stop, target):.2f} : 1")
    print(f"Shares (1% risk)   = {sz['shares']:.1f}  notional ${sz['notional']:.0f}"
          f"  leverage {sz['leverage']:.2f}x")
    print(f"Kelly fraction     = {risk.kelly_fraction(0.45, 2.0):.3f} (use 1/4-1/2 of it)")

    # ---- 4: regime detection ------------------------------------------- #
    section("MARKET REGIME — most recent bar")
    rec = regime.current(df)
    for k, v in rec.items():
        print(f"  {k:<14}: {v}")

    # ---- 5: multi-factor model ----------------------------------------- #
    section("MULTI-FACTOR — cross-sectional model on a synthetic universe")
    universe = {}
    for i, tic in enumerate(["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]):
        universe[tic] = data.synthetic_ohlcv(2520, seed=100 + i)["close"]
    prices = pd.DataFrame(universe)
    weights = multifactor.build_portfolio(prices, top_n=3, rebalance="ME")
    eq = multifactor.backtest_panel(prices, weights, capital=CAPITAL)
    print(f"Factor weights: {multifactor.DEFAULT_WEIGHTS}")
    print("Latest target allocation:")
    print(weights.iloc[-1][weights.iloc[-1] > 0].to_string())
    print(metrics.summary_table(
        metrics.summary(eq, eq.pct_change().fillna(0.0))))

    # ---- 6: optimisation (grid search) on trend strategy --------------- #
    section("OPTIMISATION — grid search trend strategy (maximise Sharpe)")
    best = None
    base_sig = strategies.trend_following(df)
    base_res = backtest.run_backtest(df, base_sig.signal, base_sig.atr,
                                     capital=CAPITAL, risk_per_trade=RISK)
    base_sharpe = base_res.stats()["Sharpe"]
    for fast in (10, 20, 30):
        for slow in (50, 100, 150):
            if fast >= slow:
                continue
            for thr in (15, 20, 25):
                s = strategies.trend_following(df, fast=fast, slow=slow,
                                               adx_threshold=thr)
                r = backtest.run_backtest(df, s.signal, s.atr,
                                          capital=CAPITAL, risk_per_trade=RISK)
                sh = r.stats()["Sharpe"]
                if best is None or sh > best[0]:
                    best = (sh, fast, slow, thr, r)
    sh, fast, slow, thr, r = best
    print(f"Baseline  (20/50, adx>20)  Sharpe={base_sharpe:.2f}  "
          f"MaxDD={base_res.stats()['Max Drawdown']*100:.1f}%")
    print(f"Optimised ({fast}/{slow}, adx>{thr}) Sharpe={sh:.2f}  "
          f"MaxDD={r.stats()['Max Drawdown']*100:.1f}%")

    # ---- 9: Monte-Carlo on best strategy ------------------------------- #
    section("MONTE-CARLO — bootstrap of realised trades")
    # pick the strategy with the most trades for a meaningful resample
    name = max(results, key=lambda k: len(results[k].trades))
    rmults = results[name].r_multiples
    if len(rmults) >= 10:
        mc = montecarlo.simulate(rmults, n_paths=5000,
                                 risk_per_trade=RISK, start_equity=CAPITAL)
        print(f"(using '{name}', {len(rmults)} trades)")
        print(montecarlo.report(mc))
    else:
        print(f"Not enough trades to bootstrap ({len(rmults)}).")

    # ---- 10: drawdown analysis ----------------------------------------- #
    section("DRAWDOWN — trend strategy")
    eqc = results["trend_following"].equity
    dd = metrics.drawdown_series(eqc)
    print(f"Max drawdown        : {dd.min()*100:.2f}%")
    print(f"Avg recovery (days) : {metrics.avg_recovery_time(eqc):.1f}")
    print(f"Time in drawdown    : {(dd < 0).mean()*100:.1f}% of bars")

    # ---- 7: portfolio construction ------------------------------------- #
    section("PORTFOLIO CONSTRUCTION — inverse-vol, medium risk tolerance")
    port = portfolio.construct(prices, tolerance="medium")
    print("Weights:")
    print(port["weights"].to_string())
    print("\nExpected stats:")
    for k, v in port["stats"].items():
        print(f"  {k:<20}: {v*100:6.2f}%")

    section("DONE")
    print("All modules executed successfully on offline synthetic data.")


if __name__ == "__main__":
    run()
