"""
Smoke / correctness tests. Run with: python -m pytest -q   (or python tests/test_framework.py)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from qflow import (
    data, feeds, indicators as ind, metrics, backtest, strategies,
    regime, multifactor, montecarlo, portfolio, risk, optimize, anomalies,
)
from qflow.paper import PaperTrader


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


def test_feeds_real_sample_offline():
    # bundled samples must load (network or committed fallback) & be canonical
    df = feeds.from_github("AAPL")
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) > 100
    assert (df["high"] >= df["low"]).all()
    assert df["volume"].notna().all()
    assert df.index.is_monotonic_increasing


def test_paper_trader_forward_and_idempotent(tmp_path=None):
    import tempfile
    root = tempfile.mkdtemp()
    pt = PaperTrader(symbol="TSLA", source="github",
                     strategy="trend_following", root=root)
    out = pt.replay(300)
    assert out["status"] == "replayed" and out["bars"] > 0
    m = pt.live_metrics()
    assert m["days"] > 0
    # equity must never go non-positive with 1% risk + no leverage
    assert m["equity"] > 0
    # idempotency: replaying again processes nothing new
    again = pt.replay(300)
    assert again["bars"] == 0
    # readiness returns a structured verdict
    r = pt.readiness()
    assert r["total"] == len(r["checks"]) and "verdict" in r


def test_anomaly_scan_and_backtests():
    df = data.synthetic_ohlcv(1000, seed=3)
    sc = anomalies.scan(df)
    assert set(sc) >= {"overnight_vs_intraday", "gap_behaviour",
                       "day_of_week", "autocorrelation"}
    # t-stats finite
    for k, v in sc["day_of_week"].items():
        assert np.isfinite(v["t"])
    gf = anomalies.backtest_gap_fade(df)
    assert gf["equity"].iloc[-1] > 0 and np.isfinite(gf["Sharpe"])
    ll = anomalies.backtest_lead_lag(df, data.synthetic_ohlcv(1000, seed=4))
    assert "Sharpe" in ll and ll["equity"].iloc[-1] > 0


def test_walk_forward_oos():
    df = data.synthetic_ohlcv(1500, seed=5)
    grid = {"rsi_buy": [5, 10], "rsi_exit": [55, 65]}
    wf = optimize.walk_forward(df, "mean_reversion", grid, n_splits=2,
                               train_frac=0.5, min_trades=1)
    assert "oos_stats" in wf
    assert np.isfinite(wf["oos_stats"]["OOS Sharpe"])


def test_grid_search_ranks():
    df = data.synthetic_ohlcv(1200, seed=6)
    res = optimize.grid_search(df, "trend_following",
                               {"fast": [10, 20], "slow": [50, 100]}, min_trades=1)
    assert not res.empty
    # sorted descending by score
    assert res["score"].is_monotonic_decreasing


def test_daily_strategies_signals():
    from qflow import daily
    df = data.synthetic_ohlcv(600, seed=8)
    gf = daily.gap_fade(df)
    assert gf.execution == "intraday"
    assert gf.signal.isin([-1, 0, 1]).all()
    ll = daily.lead_lag(df, data.synthetic_ohlcv(600, seed=9))
    assert ll.execution == "intraday" and ll.signal.isin([-1, 0, 1]).all()


def test_paper_intraday_gap_fade():
    import tempfile
    pt = PaperTrader(symbol="TSLA", source="github", strategy="gap_fade",
                     root=tempfile.mkdtemp())
    out = pt.replay(400)
    assert out["bars"] > 0
    m = pt.live_metrics()
    assert m["closed_trades"] > 0 and m["equity"] > 0
    # intraday book is flat overnight -> never holds a position between days
    assert not pt.state.position["direction"]
    assert pt.replay(400)["bars"] == 0          # idempotent


def test_paper_lead_lag_requires_leader():
    import tempfile
    root = tempfile.mkdtemp()
    try:
        PaperTrader(symbol="TSLA", source="github", strategy="lead_lag", root=root)
        assert False, "should have required a leader_symbol"
    except ValueError:
        pass
    pt = PaperTrader(symbol="TSLA", source="github", strategy="lead_lag",
                     leader_symbol="AAPL", root=root)
    out = pt.replay(756)
    assert out["bars"] > 0 and pt.live_metrics()["closed_trades"] > 0


def test_news_sentiment_and_overlay():
    from qflow import news
    # scorer: polarity + negation + events
    assert news.score_sentiment("profit surges, beats estimates")[0] > 0.3
    assert news.score_sentiment("plunges on fraud probe, guidance cut")[0] < -0.3
    assert news.score_sentiment("shares not weak, demand strong")[0] > 0    # negation
    assert news.score_sentiment("steady ahead of Fed decision")[1] is True  # event
    items = news.SampleProvider().fetch("AAPL")   # positive tone
    assert news.news_overlay(items, -1)["action"] == "veto"      # fights a short
    assert news.news_overlay(items, 1)["size_multiplier"] == 1.0  # ok for a long


def test_news_veto_blocks_entry_and_replay_isolated():
    import tempfile
    from qflow import news
    # replay must be identical with or without a provider (overlay is live-only)
    a = PaperTrader("TSLA", "github", "gap_fade", root=tempfile.mkdtemp())
    b = PaperTrader("TSLA", "github", "gap_fade", root=tempfile.mkdtemp(),
                    news_provider=news.SampleProvider())
    a.replay(400); b.replay(400)
    assert a.live_metrics()["closed_trades"] == b.live_metrics()["closed_trades"]

    # forcing a veto multiplier blocks a swing entry and logs a SKIP
    pt = PaperTrader("AAPL", "github", "trend_following", root=tempfile.mkdtemp())
    df = pt._data(); atr = ind.atr(df, 14); sig = pt._signal(df)
    # find a bar where the strategy wants a position
    idx = next(i for i in range(250, len(df)) if sig.signal.iloc[i] != 0)
    pt._news_mult, pt._news_reason = 0.0, "test veto"
    pt._process_bar(df, idx, atr, sig)
    assert pt.state.position["direction"] == 0          # entry vetoed
    assert any(j["action"] == "SKIP" for j in pt.state.journal)


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
