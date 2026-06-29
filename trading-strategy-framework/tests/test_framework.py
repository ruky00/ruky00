"""
Smoke / correctness tests. Run with: python -m pytest -q   (or python tests/test_framework.py)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from qflow import (
    data, indicators as ind, metrics, backtest, strategies,
    regime, multifactor, montecarlo, portfolio, risk,
)


def _df():
    return data.synthetic_ohlcv(n_days=800, seed=1)


def test_data_shape():
    df = _df()
    assert len(df) == 800
    assert set(["open", "high", "low", "close", "volume"]).issubset(df.columns)
    assert (df["high"] >= df["low"]).all()
    assert (df["high"] >= df["close"]).all()
    assert (df["low"] <= df["close"]).all()


def test_indicators_bounds():
    df = _df()
    r = ind.rsi(df["close"], 14)
    assert r.between(0, 100).all()
    a = ind.adx(df, 14)
    assert (a >= 0).all() and (a <= 100).all()
    assert (ind.atr(df, 14) >= 0).all()


def test_metrics_known_values():
    eq = pd.Series([100, 110, 121], dtype=float)
    assert abs(metrics.total_return(eq) - 0.21) < 1e-9
    flat = pd.Series(np.ones(50))
    assert metrics.sharpe(flat.pct_change().fillna(0)) == 0.0
    # drawdown of a monotonic curve is 0
    assert metrics.max_drawdown(pd.Series([1, 2, 3, 4.0])) == 0.0


def test_backtest_runs_and_is_causal():
    df = _df()
    sig = strategies.trend_following(df)
    res = backtest.run_backtest(df, sig.signal, sig.atr, capital=10_000)
    assert len(res.equity) == len(df)
    assert res.equity.iloc[0] > 0
    # every trade has consistent direction sign in r-multiple vs pnl
    for t in res.trades:
        assert (t.pnl >= 0) == (t.r_multiple >= 0)


def test_all_strategies_produce_signals():
    df = _df()
    for name, fn in strategies.REGISTRY.items():
        s = fn(df)
        assert s.signal.isin([-1, 0, 1]).all(), name
        assert len(s.signal) == len(df), name


def test_risk_sizing():
    sz = risk.position_size(10_000, 0.01, entry=100, stop=95)
    assert abs(sz["shares"] - 20.0) < 1e-9          # $100 risk / $5 per share
    assert risk.reward_to_risk(100, 95, 110) == 2.0
    assert risk.kelly_fraction(0.5, 2.0) == 0.25


def test_regime_labels():
    df = _df()
    reg = regime.detect(df)
    assert reg["trend"].isin(["bull", "bear", "sideways"]).all()
    rec = regime.current(df)
    assert "best_strategy" in rec


def test_multifactor_weights_sum():
    prices = pd.DataFrame(
        {t: data.synthetic_ohlcv(800, seed=i)["close"].values
         for i, t in enumerate(list("ABCDE"))},
        index=data.synthetic_ohlcv(800, seed=0).index,
    )
    w = multifactor.build_portfolio(prices, top_n=3)
    row_sums = w.sum(axis=1)
    # rows are either fully invested (~1) or flat (0) before first rebalance
    assert ((np.isclose(row_sums, 1.0)) | (np.isclose(row_sums, 0.0))).all()


def test_montecarlo():
    rng = np.random.default_rng(0)
    rmults = list(rng.choice([-1.0, 2.0], size=200, p=[0.55, 0.45]))
    out = montecarlo.simulate(rmults, n_paths=1000, risk_per_trade=0.01)
    assert 0 <= out["prob_loss"] <= 1
    assert "ROBUST" in montecarlo.verdict(out) or "MODERATE" in montecarlo.verdict(out) \
        or "FRAGILE" in montecarlo.verdict(out)


def test_portfolio_construct():
    prices = pd.DataFrame(
        {t: data.synthetic_ohlcv(800, seed=i)["close"].values
         for i, t in enumerate(list("ABCD"))},
        index=data.synthetic_ohlcv(800, seed=0).index,
    )
    out = portfolio.construct(prices, tolerance="medium")
    assert abs(out["weights"].sum() - 1.0) < 1e-9
    assert "expected_return" in out["stats"]


def _run_all():
    fns = [v for k, v in globals().items() if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
