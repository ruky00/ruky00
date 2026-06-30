"""
Wide-universe edge scanner.

Your "a stock that always reacts the same way" idea only shows up when you scan
*hundreds* of names, not two. This module runs the repeatable-edge lab
(`qflow.edge_lab`) across a whole universe and keeps only the patterns that pass
the robustness filters (consistency + out-of-sample + after costs), so you get a
short, ranked shortlist to forward-test.

Bundled universes (symbols use Yahoo Finance conventions):
    IBEX35      : the 35 Spanish blue chips ('.MC')
    SP500_LIQUID: a liquid large-cap US subset (extend with the full list)

In a locked-down network only the bundled github samples are reachable; the full
universes need an open network (source='yahoo' or 'stooq').
"""

from __future__ import annotations

import pandas as pd

from . import feeds, edge_lab

# --- Spanish blue chips (IBEX 35), Yahoo '.MC' tickers ---------------------- #
IBEX35 = [
    "SAN.MC", "BBVA.MC", "ITX.MC", "IBE.MC", "TEF.MC", "REP.MC", "AMS.MC",
    "FER.MC", "AENA.MC", "CLNX.MC", "ELE.MC", "NTGY.MC", "CABK.MC", "SAB.MC",
    "BKT.MC", "MAP.MC", "ACS.MC", "ENG.MC", "RED.MC", "GRF.MC", "ANA.MC",
    "ANE.MC", "MEL.MC", "COL.MC", "MRL.MC", "LOG.MC", "IDR.MC", "FDR.MC",
    "SCYR.MC", "SLR.MC", "UNI.MC", "IAG.MC", "ACX.MC", "PUIG.MC", "ROVI.MC",
]

# --- liquid US large caps (subset; add the rest of the S&P 500 as needed) --- #
SP500_LIQUID = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK-B", "JPM",
    "V", "UNH", "XOM", "JNJ", "WMT", "MA", "PG", "HD", "CVX", "KO", "PEP",
    "ABBV", "BAC", "AVGO", "COST", "MRK", "DIS", "ADBE", "NFLX", "CRM", "AMD",
]


def scan_universe(symbols: list[str],
                  source: str = "yahoo",
                  leader: pd.DataFrame | None = None,
                  cost_bps: float = 4.0,
                  min_trades: int = 20,
                  min_consistency: float = 0.6,
                  require_oos: bool = True,
                  feed_kwargs: dict | None = None,
                  refresh: bool = False,
                  verbose: bool = False) -> dict:
    """
    Run the repeatable-edge lab over every symbol and return:
        {"robust": [...passing patterns ranked...], "all": [...], "errors": {...}}

    A pattern is "robust" if it has enough trades, is positive in at least
    `min_consistency` of calendar years, survives both out-of-sample halves
    (when `require_oos`), and is net-positive after costs.
    """
    feed_kwargs = feed_kwargs or {}
    robust, allrows, errors = [], [], {}
    for sym in symbols:
        try:
            df = feeds.get(sym, source=source, refresh=refresh, **feed_kwargs)
        except Exception as e:  # unreachable / bad symbol -> skip, keep going
            errors[sym] = f"{type(e).__name__}"
            if verbose:
                print(f"  skip {sym}: {errors[sym]}")
            continue
        for r in edge_lab.scan(df, leader=leader, cost_bps=cost_bps):
            r = {**r, "symbol": sym}
            allrows.append(r)
            if (r.get("n_trades", 0) >= min_trades
                    and r.get("consistency", 0) >= min_consistency
                    and r.get("tradeable", False)
                    and (not require_oos
                         or (r.get("oos_sharpe_1h", 0) > 0 and r.get("oos_sharpe_2h", 0) > 0))):
                robust.append(r)
        if verbose:
            print(f"  scanned {sym}")
    robust.sort(key=lambda r: r["score"], reverse=True)
    allrows.sort(key=lambda r: r.get("score", 0), reverse=True)
    return {"robust": robust, "all": allrows, "errors": errors,
            "n_symbols": len(symbols), "n_scanned": len(symbols) - len(errors)}


def report(result: dict, top: int = 25) -> str:
    L = [f"{'='*82}",
         f"UNIVERSE SCAN — {result['n_scanned']}/{result['n_symbols']} symbols scanned, "
         f"{len(result['robust'])} robust patterns found",
         "=" * 82,
         f"{'symbol':<10}{'pattern':<15}{'Sharpe':>7}{'consist':>9}{'OOS1':>6}"
         f"{'OOS2':>6}{'ann%':>7}{'trades':>7}"]
    L.append("-" * 82)
    if not result["robust"]:
        L.append("  (no patterns passed the robustness filters — try more history "
                 "or loosen min_consistency)")
    for r in result["robust"][:top]:
        L.append(f"{r['symbol']:<10}{r['name']:<15}{r['sharpe']:>7.2f}"
                 f"{r['consistency']:>8.0%}{r['oos_sharpe_1h']:>6.2f}"
                 f"{r['oos_sharpe_2h']:>6.2f}{r['ann_return']*100:>6.1f}%{r['n_trades']:>7}")
    L.append("-" * 82)
    if result["errors"]:
        L.append(f"skipped {len(result['errors'])} symbols (unreachable/no data).")
    L.append("Shortlist these for paper trading — do NOT trade them blind: a wide "
             "scan inflates false positives, so re-confirm each on fresh data.")
    return "\n".join(L)
