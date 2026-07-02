"""
Automatic universe / strategy selector.

Runs a rolling walk-forward for every (symbol, strategy) pair in a basket and
keeps only the pairs that are *robust*, judged on four axes — exactly the lessons
the walk-forward teaches:

    OOS Sharpe        : does it make risk-adjusted money out of sample?
    OOS max drawdown  : is the worst case survivable?
    parameter stability: do the optimised params stay put across folds (robust)
                         rather than jumping around (curve-fit / fragile)?
    positive-fold ratio: does it win in most test years, not one lucky year?

The output is a shortlist of survivors plus a recommended **final portfolio**:
the single best surviving strategy per symbol (symbols where nothing survives are
dropped). Feed that shortlist into paper trading — never trade the raw scan blind.
"""

from __future__ import annotations

from collections import Counter

from . import feeds, optimize, strategies

DEFAULT_THRESHOLDS = {
    "min_oos_sharpe": 0.5,
    "max_oos_drawdown": -0.25,   # OOS max DD must be >= this (i.e. not worse)
    "min_stability": 0.5,        # avg fraction of folds sharing the modal param value
    "min_pos_ratio": 0.6,        # share of test years with positive OOS return
    "min_folds": 3,
}


def _stability(folds) -> float:
    """Average, over params, of the fraction of folds sharing the modal value."""
    if not folds:
        return 0.0
    keys = folds[0]["params"].keys()
    scores = []
    for k in keys:
        vals = [f["params"][k] for f in folds]
        scores.append(max(Counter(vals).values()) / len(vals))
    return sum(scores) / len(scores) if scores else 0.0


def _modal_params(folds) -> dict:
    """The most common parameter value per key across folds (what to trade)."""
    keys = folds[0]["params"].keys()
    return {k: Counter(f["params"][k] for f in folds).most_common(1)[0][0] for k in keys}


def evaluate_pair(df, symbol, strategy, grid, train_years=4, bt_kwargs=None) -> dict | None:
    wf = optimize.rolling_walk_forward(df, strategy, grid, train_years=train_years,
                                       test_years=1, bt_kwargs=bt_kwargs)
    if "error" in wf or not wf["folds"]:
        return None
    folds, st = wf["folds"], wf["oos_stats"]
    pos_ratio = sum(1 for f in folds if f["oos_return"] > 0) / len(folds)
    return {
        "symbol": symbol,
        "strategy": strategy,
        "oos_sharpe": round(st["OOS Sharpe"], 2),
        "oos_cagr": round(st["OOS CAGR"], 4),
        "oos_maxdd": round(st["OOS Max Drawdown"], 4),
        "stability": round(_stability(folds), 2),
        "pos_ratio": round(pos_ratio, 2),
        "n_folds": len(folds),
        "params": _modal_params(folds),
    }


def _passes(r, th) -> bool:
    return (r["oos_sharpe"] >= th["min_oos_sharpe"]
            and r["oos_maxdd"] >= th["max_oos_drawdown"]
            and r["stability"] >= th["min_stability"]
            and r["pos_ratio"] >= th["min_pos_ratio"]
            and r["n_folds"] >= th["min_folds"])


def select(symbols,
           source="yahoo",
           grids: dict | None = None,
           strategies_to_test: list | None = None,
           train_years: int = 4,
           thresholds: dict | None = None,
           bt_kwargs: dict | None = None,
           feed_kwargs: dict | None = None,
           verbose: bool = False) -> dict:
    """
    Scan every (symbol, strategy) pair and keep the robust ones.

    Returns {"all", "survivors", "portfolio", "errors"} where `portfolio` maps
    each surviving symbol -> its best (symbol, strategy) row.
    """
    grids = grids or strategies.DEFAULT_WF_GRIDS
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    strat_list = strategies_to_test or list(grids)
    bt_kwargs = bt_kwargs or {"capital": 10_000.0, "risk_per_trade": 0.01}
    feed_kwargs = feed_kwargs or {}

    rows, errors = [], {}
    for sym in symbols:
        try:
            df = feeds.get(sym, source=source, **feed_kwargs)
        except Exception as e:
            errors[sym] = type(e).__name__
            continue
        for strat in strat_list:
            r = evaluate_pair(df, sym, strat, grids[strat], train_years, bt_kwargs)
            if r is None:
                continue
            r["pass"] = _passes(r, th)
            rows.append(r)
            if verbose:
                print(f"  {sym:<6} {strat:<20} Sharpe {r['oos_sharpe']:>5.2f} "
                      f"DD {r['oos_maxdd']*100:>6.1f}% stab {r['stability']:.2f} "
                      f"pos {r['pos_ratio']:.0%}  {'PASS' if r['pass'] else ''}")

    survivors = sorted([r for r in rows if r["pass"]],
                       key=lambda r: r["oos_sharpe"], reverse=True)
    # best surviving strategy per symbol -> the recommended portfolio
    portfolio = {}
    for r in survivors:
        if r["symbol"] not in portfolio:
            portfolio[r["symbol"]] = r
    return {"all": rows, "survivors": survivors, "portfolio": portfolio,
            "errors": errors, "thresholds": th}


def report(result: dict) -> str:
    th = result["thresholds"]
    L = [f"{'='*84}",
         "UNIVERSE / STRATEGY SELECTION (rolling walk-forward)",
         f"keep if: OOS Sharpe>={th['min_oos_sharpe']}, OOS maxDD>={th['max_oos_drawdown']*100:.0f}%, "
         f"stability>={th['min_stability']}, pos folds>={th['min_pos_ratio']:.0%}",
         "=" * 84,
         f"{'symbol':<7}{'strategy':<20}{'OOSsharpe':>10}{'OOScagr':>9}{'OOSmaxDD':>10}"
         f"{'stab':>6}{'pos':>6}  keep"]
    L.append("-" * 84)
    for r in sorted(result["all"], key=lambda r: (r["symbol"], -r["oos_sharpe"])):
        L.append(f"{r['symbol']:<7}{r['strategy']:<20}{r['oos_sharpe']:>10.2f}"
                 f"{r['oos_cagr']*100:>8.1f}%{r['oos_maxdd']*100:>9.1f}%"
                 f"{r['stability']:>6.2f}{r['pos_ratio']:>6.0%}  {'YES' if r['pass'] else ''}")
    L.append("-" * 84)
    if result["portfolio"]:
        L.append("RECOMMENDED PORTFOLIO (best surviving strategy per symbol):")
        for sym, r in result["portfolio"].items():
            L.append(f"  {sym:<7} -> {r['strategy']:<20} (OOS Sharpe {r['oos_sharpe']:.2f})  "
                     f"params={r['params']}")
    else:
        L.append("No (symbol, strategy) pair passed — loosen thresholds or widen the universe.")
    if result["errors"]:
        L.append(f"skipped: {', '.join(result['errors'])}")
    L.append("=" * 84)
    L.append("Shortlist only — paper-test the survivors before risking capital.")
    return "\n".join(L)
