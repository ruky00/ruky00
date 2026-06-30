"""
Cross-listing lead-lag — the structural España <-> US edge.

A handful of Spanish blue chips trade in BOTH Madrid and the US (as NYSE ADRs).
Because the sessions are offset — Madrid closes ~17:30 CET while the US ADR keeps
trading for hours afterward — information flows between the two listings with a
predictable timing:

    * the US ADR's *late* move (after Madrid has closed) tends to lead the Madrid
      listing's *next-day open*;
    * within the overlap, Madrid (which opens first) can lead the ADR's US session.

This is a *structural* lag (different trading hours), not a random correlation,
which is exactly the kind of repeatable effect you asked for. It is also small
and costs/borrow-sensitive, which is why it persists.

`analyze_pair` quantifies both directions; `scan_dual_listings` runs the curated
pairs. Symbols use Yahoo conventions (Madrid '.MC', US ADR plain). Liquid NYSE
ADRs (best for this) are flagged ``liquid_adr=True``; the others are thin OTC.
"""

from __future__ import annotations

import pandas as pd

from . import feeds, anomalies, edge_lab

# name -> (madrid '.MC', us ADR, liquid US listing?)
DUAL_LISTINGS = {
    "Santander":   ("SAN.MC",  "SAN",   True),    # NYSE, very liquid
    "BBVA":        ("BBVA.MC", "BBVA",  True),    # NYSE, very liquid
    "Telefonica":  ("TEF.MC",  "TEF",   True),    # NYSE
    "Repsol":      ("REP.MC",  "REPYY", False),   # OTC ADR (thin)
    "Iberdrola":   ("IBE.MC",  "IBDRY", False),   # OTC ADR (thin)
    "Inditex":     ("ITX.MC",  "IDEXY", False),   # OTC ADR (thin)
    "Aena":        ("AENA.MC", "ANYYY", False),   # OTC ADR (thin)
}


def analyze_pair(madrid: pd.DataFrame, us: pd.DataFrame,
                 name: str = "pair", cost_bps: float = 4.0) -> dict:
    """
    Quantify the lead-lag in both directions and the tradeable edge.

    Returns lead-lag correlation stats (us->madrid and madrid->us) plus the
    evaluated open->close trade for the stronger structural direction
    (US ADR leads the Madrid next-day session).
    """
    us_leads = anomalies.lead_lag(leader=us, follower=madrid, max_lag=2)
    madrid_leads = anomalies.lead_lag(leader=madrid, follower=us, max_lag=2)

    # tradeable: trade Madrid's session off the US ADR's prior-day move
    sig = edge_lab.sig_lead_lag(madrid, leader=us, lag=1, threshold=0.003)
    trade_madrid = edge_lab.evaluate(madrid, sig, f"{name}: US->MAD", cost_bps)
    # and the reverse: trade the ADR off Madrid's prior session
    sig2 = edge_lab.sig_lead_lag(us, leader=madrid, lag=1, threshold=0.003)
    trade_us = edge_lab.evaluate(us, sig2, f"{name}: MAD->US", cost_bps)

    best = max([trade_madrid, trade_us], key=lambda r: r.get("score", 0))
    return {
        "name": name,
        "us_leads_madrid": us_leads,
        "madrid_leads_us": madrid_leads,
        "trade_us_to_madrid": trade_madrid,
        "trade_madrid_to_us": trade_us,
        "best": best,
    }


def scan_dual_listings(source: str = "yahoo",
                       names: list[str] | None = None,
                       liquid_only: bool = True,
                       cost_bps: float = 4.0,
                       feed_kwargs: dict | None = None,
                       refresh: bool = False,
                       verbose: bool = False) -> dict:
    """Fetch and analyse the curated dual-listed pairs; rank by tradeable score."""
    feed_kwargs = feed_kwargs or {}
    names = names or list(DUAL_LISTINGS)
    results, errors = [], {}
    for nm in names:
        madrid_sym, us_sym, liquid = DUAL_LISTINGS[nm]
        if liquid_only and not liquid:
            continue
        try:
            mad = feeds.get(madrid_sym, source=source, refresh=refresh, **feed_kwargs)
            usd = feeds.get(us_sym, source=source, refresh=refresh, **feed_kwargs)
        except Exception as e:
            errors[nm] = f"{type(e).__name__}"
            if verbose:
                print(f"  skip {nm}: {errors[nm]}")
            continue
        results.append(analyze_pair(mad, usd, nm, cost_bps))
        if verbose:
            print(f"  analysed {nm}")
    results.sort(key=lambda r: r["best"].get("score", 0), reverse=True)
    return {"results": results, "errors": errors}


def report(scan_result: dict) -> str:
    L = [f"{'='*84}",
         "DUAL-LISTING LEAD-LAG — España <-> US ADRs",
         "=" * 84,
         f"{'pair':<14}{'direction':<14}{'corr(1d)':>9}{'t':>6}"
         f"{'Sharpe':>8}{'consist':>9}{'OOS2':>6}{'trades':>7}"]
    L.append("-" * 84)
    for r in scan_result["results"]:
        b = r["best"]
        # show the 1-day lead correlation for the US->Madrid direction
        ll = r["us_leads_madrid"].get("lead_1d", {})
        L.append(f"{r['name']:<14}{b['name'].split(':')[-1].strip():<14}"
                 f"{ll.get('corr',0):>9.3f}{ll.get('t',0):>6.2f}"
                 f"{b.get('sharpe',0):>8.2f}{b.get('consistency',0):>8.0%}"
                 f"{b.get('oos_sharpe_2h',0):>6.2f}{b.get('n_trades',0):>7}")
    L.append("-" * 84)
    if scan_result["errors"]:
        L.append(f"skipped: {', '.join(scan_result['errors'])} (unreachable here; "
                 "run on an open network with source='yahoo').")
    L.append("Structural edge: the US ADR trades after Madrid closes, so its late "
             "move leads Madrid's next open. Small, borrow/cost-sensitive -> "
             "confirm on fresh data and paper-test before trading.")
    return "\n".join(L)
