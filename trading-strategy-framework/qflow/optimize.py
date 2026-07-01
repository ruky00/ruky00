"""
Parameter optimisation with an emphasis on *not* fooling yourself.

Two tools:

    grid_search   : exhaustive in-sample sweep, ranked by a chosen metric.
    walk_forward  : the honest one. Repeatedly optimise on a training window and
                    evaluate on the *next* unseen window, then stitch the
                    out-of-sample (OOS) pieces into one equity curve. If the OOS
                    curve is good, the edge is more likely real and not curve-fit.

A parameter set that wins in-sample but collapses out-of-sample is overfit —
walk-forward is what exposes that.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from . import strategies, backtest, metrics


def _score(stats: dict, metric: str) -> float:
    if metric == "sharpe":
        return stats.get("Sharpe", 0.0)
    if metric == "calmar":
        return stats.get("Calmar", 0.0)
    if metric == "sharpe_calmar":
        return stats.get("Sharpe", 0.0) + stats.get("Calmar", 0.0)
    if metric == "return_dd":            # CAGR / |maxDD|
        dd = abs(stats.get("Max Drawdown", 0.0)) or 1e9
        return stats.get("CAGR", 0.0) / dd
    return stats.get(metric, 0.0)


def _expand(param_grid: dict):
    keys = list(param_grid)
    for combo in itertools.product(*(param_grid[k] for k in keys)):
        yield dict(zip(keys, combo))


def _run(df, strategy_name, params, bt_kwargs):
    fn = strategies.REGISTRY[strategy_name]
    sig = fn(df, **params)
    res = backtest.run_backtest(df, sig.signal, sig.atr, **bt_kwargs)
    return res


def grid_search(df: pd.DataFrame,
                strategy_name: str,
                param_grid: dict,
                metric: str = "sharpe_calmar",
                min_trades: int = 5,
                bt_kwargs: dict | None = None) -> pd.DataFrame:
    """Exhaustive in-sample sweep. Returns a ranked DataFrame of results."""
    bt_kwargs = bt_kwargs or {}
    rows = []
    for params in _expand(param_grid):
        try:
            res = _run(df, strategy_name, params, bt_kwargs)
        except Exception:
            continue
        st = res.stats()
        if st.get("Trades", 0) < min_trades:
            continue
        rows.append({**params,
                     "score": _score(st, metric),
                     "Sharpe": st["Sharpe"], "CAGR": st["CAGR"],
                     "MaxDD": st["Max Drawdown"], "Trades": st.get("Trades", 0)})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def walk_forward(df: pd.DataFrame,
                 strategy_name: str,
                 param_grid: dict,
                 n_splits: int = 4,
                 train_frac: float = 0.6,
                 metric: str = "sharpe_calmar",
                 min_trades: int = 5,
                 capital: float = 10_000.0,
                 bt_kwargs: dict | None = None) -> dict:
    """
    Anchored-then-rolling walk-forward.

    The series is cut into ``n_splits`` test windows. For each, parameters are
    optimised on the preceding ``train_frac`` slice and applied to the test
    slice. OOS test equity pieces are chained into a single curve.
    """
    bt_kwargs = dict(bt_kwargs or {})
    bt_kwargs.setdefault("capital", capital)

    n = len(df)
    test_size = int(n * (1 - train_frac) / n_splits)
    if test_size < 30:
        raise ValueError("Series too short for this split configuration.")

    folds = []
    oos_returns = []
    start_equity = capital
    for k in range(n_splits):
        test_end = n - (n_splits - 1 - k) * test_size
        test_start = test_end - test_size
        train = df.iloc[:test_start]
        test = df.iloc[max(0, test_start - 250):test_end]  # carry warmup for indicators
        if len(train) < 250:
            continue

        ranked = grid_search(train, strategy_name, param_grid, metric,
                             min_trades, bt_kwargs)
        if ranked.empty:
            continue
        best = {k_: ranked.iloc[0][k_] for k_ in param_grid}
        best = {k_: (int(v) if float(v).is_integer() else float(v))
                for k_, v in best.items()}

        res = _run(test, strategy_name, best, {**bt_kwargs, "capital": start_equity})
        # keep only the genuine test portion (drop the warmup head)
        test_only = res.equity.loc[df.index[test_start]:df.index[test_end - 1]]
        if len(test_only) < 2:
            continue
        oos_returns.append(test_only.pct_change().fillna(0.0))
        start_equity = float(test_only.iloc[-1])
        folds.append({"fold": k + 1,
                      "train_end": str(df.index[test_start].date()),
                      "test": f"{df.index[test_start].date()}..{df.index[test_end-1].date()}",
                      "params": best,
                      "oos_return": float(test_only.iloc[-1] / test_only.iloc[0] - 1.0)})

    if not oos_returns:
        return {"error": "no valid folds"}

    oos_ret = pd.concat(oos_returns)
    oos_equity = capital * (1 + oos_ret).cumprod()
    oos_stats = {
        "OOS CAGR": metrics.cagr(oos_equity),
        "OOS Sharpe": metrics.sharpe(oos_ret),
        "OOS Sortino": metrics.sortino(oos_ret),
        "OOS Max Drawdown": metrics.max_drawdown(oos_equity),
        "OOS Total Return": metrics.total_return(oos_equity),
    }
    return {"folds": folds, "oos_stats": oos_stats, "oos_equity": oos_equity}


def _cast_params(param_grid, best_row):
    out = {}
    for k in param_grid:
        v = best_row[k]
        if isinstance(v, bool):
            out[k] = bool(v)
        else:
            try:
                out[k] = int(v) if float(v).is_integer() else float(v)
            except (TypeError, ValueError):
                out[k] = v
    return out


def rolling_walk_forward(df: pd.DataFrame,
                         strategy_name: str,
                         param_grid: dict,
                         train_years: int = 4,
                         test_years: int = 1,
                         metric: str = "sharpe_calmar",
                         min_trades: int = 3,
                         warmup_bars: int = 260,
                         capital: float = 10_000.0,
                         bt_kwargs: dict | None = None) -> dict:
    """
    Rolling **calendar-year** walk-forward — the robust validation you described:

        train 2014-2017 -> test 2018
        train 2015-2018 -> test 2019
        train 2016-2019 -> test 2020   ... (roll forward one year at a time)

    Each fold optimises the parameters on the trailing ``train_years`` years and
    trades the next ``test_years`` year with those fixed params. The out-of-sample
    years are chained into one continuous equity curve — that is the number to
    trust, not the in-sample fit.
    """
    bt_kwargs = dict(bt_kwargs or {})
    years = sorted(int(y) for y in pd.unique(df.index.year))
    folds, oos_returns = [], []
    start_equity = capital

    for test_year in years:
        train = df[(df.index.year >= test_year - train_years) & (df.index.year < test_year)]
        test = df[(df.index.year >= test_year) & (df.index.year <= test_year + test_years - 1)]
        if len(train) < 200 or len(test) < 20:
            continue

        ranked = grid_search(train, strategy_name, param_grid, metric,
                             min_trades, bt_kwargs)
        if ranked.empty:
            continue
        best = _cast_params(param_grid, ranked.iloc[0])

        # evaluate on the test year with an indicator warmup carried in
        t0 = df.index.get_loc(test.index[0])
        wu = df.iloc[max(0, t0 - warmup_bars):df.index.get_loc(test.index[-1]) + 1]
        res = _run(wu, strategy_name, best, {**bt_kwargs, "capital": start_equity})
        test_eq = res.equity.loc[test.index[0]:test.index[-1]]
        if len(test_eq) < 2:
            continue
        oos_returns.append(test_eq.pct_change().fillna(0.0))
        start_equity = float(test_eq.iloc[-1])
        folds.append({
            "test_year": test_year,
            "train": f"{train.index[0].year}-{train.index[-1].year}",
            "params": best,
            "oos_return": float(test_eq.iloc[-1] / test_eq.iloc[0] - 1.0),
        })

    if not oos_returns:
        return {"error": "not enough calendar years for this configuration"}
    oos_ret = pd.concat(oos_returns)
    oos_equity = capital * (1 + oos_ret).cumprod()
    return {
        "folds": folds,
        "oos_stats": {
            "OOS CAGR": metrics.cagr(oos_equity),
            "OOS Sharpe": metrics.sharpe(oos_ret),
            "OOS Sortino": metrics.sortino(oos_ret),
            "OOS Max Drawdown": metrics.max_drawdown(oos_equity),
            "OOS Total Return": metrics.total_return(oos_equity),
        },
        "oos_equity": oos_equity,
    }


def walk_forward_params(df: pd.DataFrame,
                        strategy_name: str,
                        param_grid: dict,
                        train_years: int = 4,
                        metric: str = "sharpe_calmar",
                        min_trades: int = 3,
                        bt_kwargs: dict | None = None) -> dict:
    """Return {test_year: best_params} from a rolling walk-forward — the params
    to use for each calendar year, re-optimised on that year's trailing window."""
    wf = rolling_walk_forward(df, strategy_name, param_grid, train_years=train_years,
                              test_years=1, metric=metric, min_trades=min_trades,
                              bt_kwargs=bt_kwargs)
    if "error" in wf:
        return {}
    return {f["test_year"]: f["params"] for f in wf["folds"]}


def report_rolling(wf: dict) -> str:
    if "error" in wf:
        return f"rolling walk-forward: {wf['error']}"
    lines = ["Rolling walk-forward (train N years -> test next year)", "-" * 68]
    for f in wf["folds"]:
        lines.append(f"  train {f['train']} -> test {f['test_year']}  "
                     f"OOS {f['oos_return']*100:+6.2f}%   params={f['params']}")
    lines.append("-" * 68)
    for k, v in wf["oos_stats"].items():
        if "Return" in k or "CAGR" in k or "Drawdown" in k:
            lines.append(f"  {k:<20} {v*100:>8.2f}%")
        else:
            lines.append(f"  {k:<20} {v:>9.2f}")
    return "\n".join(lines)


def report_walk_forward(wf: dict) -> str:
    if "error" in wf:
        return f"walk-forward: {wf['error']}"
    lines = ["Walk-forward (out-of-sample) results", "-" * 52]
    for f in wf["folds"]:
        lines.append(f"  fold {f['fold']}  test {f['test']}  "
                     f"OOS {f['oos_return']*100:+6.2f}%   params={f['params']}")
    lines.append("-" * 52)
    for k, v in wf["oos_stats"].items():
        if "Return" in k or "CAGR" in k or "Drawdown" in k:
            lines.append(f"  {k:<20} {v*100:>8.2f}%")
        else:
            lines.append(f"  {k:<20} {v:>9.2f}")
    return "\n".join(lines)
