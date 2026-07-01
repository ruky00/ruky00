"""
Portfolio backtest of the adaptive 'auto' strategy across a basket.

Backtests 'auto' on each symbol, combines them into one equal-weight portfolio
equity curve, and reports per-symbol + aggregate stats PLUS an in-sample vs
out-of-sample split so you validate the basket instead of curve-fitting it.

    python examples/backtest_portfolio.py

Offline it uses the bundled samples (AAPL, TSLA). On your machine, edit BASKET
and use 10y data for the volatile names you actually intend to trade:
    from qflow import feeds; feeds.get("NVDA", "yahoo", rng="10y")
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from qflow import feeds, strategies, backtest, metrics

CAPITAL = 10_000.0
RISK = 0.005
BASKET = ["AAPL", "TSLA"]          # swap for TSLA,NVDA,AMD,COIN,PLTR on your machine
SOURCE = "github"


def strat_returns(df):
    """Daily return series of 'auto' on one symbol (per $1)."""
    sig = strategies.adaptive(df)
    res = backtest.run_backtest(df, sig.signal, sig.atr,
                                capital=CAPITAL, risk_per_trade=RISK)
    return res.returns, res.stats(), sig.chosen


def portfolio_stats(ret_df, label):
    port_ret = ret_df.mean(axis=1)               # equal weight, daily rebalance
    eq = CAPITAL * (1 + port_ret).cumprod()
    return {
        "label": label,
        "CAGR": metrics.cagr(eq),
        "Sharpe": metrics.sharpe(port_ret),
        "Sortino": metrics.sortino(port_ret),
        "MaxDD": metrics.max_drawdown(eq),
        "TotalReturn": metrics.total_return(eq),
    }


def main():
    print(f"Portfolio backtest — 'auto' on {BASKET} (equal weight, {RISK*100:.1f}% risk)\n")
    per_symbol_ret = {}
    usage_total = {}
    hdr = f"{'symbol':<8}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>8}{'trades':>7}"
    print(hdr); print("-" * len(hdr))
    for sym in BASKET:
        df = feeds.get(sym, SOURCE)
        ret, st, chosen = strat_returns(df)
        per_symbol_ret[sym] = ret
        for k, v in chosen.value_counts().items():
            usage_total[k] = usage_total.get(k, 0) + int(v)
        print(f"{sym:<8}{st['CAGR']*100:>7.1f}%{st['Sharpe']:>8.2f}"
              f"{st['Max Drawdown']*100:>7.1f}%{st.get('Trades',0):>7}")

    ret_df = pd.DataFrame(per_symbol_ret).fillna(0.0)

    # full-sample + out-of-sample split (anchored ~60/40)
    n = len(ret_df)
    split = int(n * 0.6)
    full = portfolio_stats(ret_df, "PORTFOLIO (full)")
    ins = portfolio_stats(ret_df.iloc[:split], "  in-sample (first 60%)")
    oos = portfolio_stats(ret_df.iloc[split:], "  out-of-sample (last 40%)")

    print("\n" + "=" * 58)
    print(f"{'window':<28}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>8}")
    print("-" * 58)
    for s in (full, ins, oos):
        print(f"{s['label']:<28}{s['CAGR']*100:>7.1f}%{s['Sharpe']:>8.2f}"
              f"{s['MaxDD']*100:>7.1f}%")
    print("=" * 58)

    tot = sum(usage_total.values()) or 1
    usage = "  ".join(f"{k} {v/tot*100:.0f}%" for k, v in usage_total.items())
    print(f"auto strategy usage across basket: {usage}")
    print("\nValidate: OOS Sharpe should stay positive and near in-sample. On this "
          "2-symbol / short sample it is only illustrative — use 10y and 5+ volatile "
          "names, then paper-test with examples/portfolio_run.py.")


if __name__ == "__main__":
    main()
