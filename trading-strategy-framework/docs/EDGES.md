# Daily Edges & Optimisation

> **Educational use only — not financial advice.** The numbers below are
> in-sample statistics on a short real-data sample (AAPL 2015–17, TSLA 2015–18).
> A good in-sample stat is a *hypothesis*, not a guarantee. Confirm out-of-sample
> and after real costs before trading anything. See [`DISCLAIMER.md`](DISCLAIMER.md).

This covers the two things you asked for: (a) **finding small, recurring daily
inefficiencies** and (b) **optimising a strategy honestly**. Reproduce with:

```bash
python examples/find_edges.py
```

---

## 1. How the scanner hunts for edges

`qflow/anomalies.py` runs a battery of tests and flags any with |t-stat| > 2:

| Test | Question it answers |
|------|---------------------|
| Overnight vs intraday | Is the drift in the *close→open* move or the *open→close* session? |
| Gap behaviour | After an opening gap, does price **continue** (gap-and-go) or **fade** (fill)? |
| Day-of-week | Are some weekdays systematically stronger/weaker? |
| Autocorrelation | Do daily returns trend (momentum) or snap back (reversal)? |
| **Lead-lag** | Does asset A's move **today** predict asset B **tomorrow**? |

The lead-lag test is the generalisation of your example — *"a stock in one
market moves and its listing/peer in another market follows the next session."*

---

## 2. What actually showed up (real data, net of costs)

```
TSLA gap-fade                CAGR  20.9%  Sharpe 0.77  MaxDD -32.2%   (408 trade days)
TSLA->AAPL lead-lag(1d)      CAGR  11.8%  Sharpe 0.65  MaxDD -15.4%
AAPL->TSLA lead-lag(1d)      CAGR  14.3%  Sharpe 0.57  MaxDD -21.8%
TSLA overnight-only          CAGR  -2.7%  Sharpe 0.03  MaxDD -46.6%   (dies after costs)
AAPL gap-fade                CAGR  -3.2%  Sharpe -0.14                 (calm name, no edge)
```

### Edge A — **Gap-fade on high-volatility names** (your "daily gap" idea)
- **Rule**: at the open, if price gapped **down** more than 0.5% vs yesterday's
  close, **buy** and hold to the close; if it gapped **up** >0.5%, **short** to
  the close. (`anomalies.backtest_gap_fade`)
- **Why it exists**: overnight gaps in volatile names often over-react to news;
  liquidity providers fade the open and price mean-reverts intraday.
- **Reality check**: it works on **TSLA** (Sharpe 0.77) but **not AAPL** (−0.14).
  Edge lives in high-volatility names. Costs matter — it only trades gap days.

### Edge B — **Cross-market lead-lag** (your España→US idea)
- **Finding**: AAPL's move **leads** TSLA by one day (corr 0.093, t≈1.71), while
  the reverse is ~zero (t=0.18) — a genuine asymmetry, not just shared beta.
- **Rule**: if the leader rose >0.5% yesterday, be **long** the follower today;
  if it fell, be **short**. (`anomalies.backtest_lead_lag`)
- **Why most miss it**: it requires watching a *different* instrument than the
  one you trade, and the effect is small per trade — boring, so under-arbitraged.
- **Your version**: set `leader =` the market that closes first (e.g. an IBEX 35
  name or the IBEX future) and `follower =` its US ADR or a US index, then test:
  ```python
  from qflow import feeds, anomalies
  leader   = feeds.get("SAN.MC", "yahoo", rng="5y")   # Santander, Madrid
  follower = feeds.get("SAN",    "yahoo", rng="5y")   # Santander ADR, NYSE
  print(anomalies.lead_lag(leader, follower, max_lag=2))
  print(anomalies.backtest_lead_lag(leader, follower, lag=1, threshold=0.005))
  ```

### Non-edge — **Overnight-only**
The overnight drift is real in raw returns (TSLA +19%/yr vs −12% intraday) but
**two fills per day** (≈8 bps round trip) eat a ~7.6 bps/day edge. Reported here
precisely because honest research includes the ideas that *don't* survive costs.

---

## 3. Optimising honestly — walk-forward

`qflow/optimize.py` provides `grid_search` (in-sample) and, more importantly,
`walk_forward`: optimise on a training window, score on the **next unseen**
window, repeat, and stitch the out-of-sample pieces together.

Mean-reversion on TSLA, 3 OOS folds:

```
fold 1  test 2017-04..2017-10  OOS +2.28%   params rsi_buy=15 rsi_exit=65 trend=150
fold 2  test 2017-10..2018-04  OOS +0.01%   params rsi_buy=10 rsi_exit=65 trend=200
fold 3  test 2018-04..2018-10  OOS -0.12%   params rsi_buy=10 rsi_exit=65 trend=200
------------------------------------------------------------
OOS Sharpe 0.58   OOS Max Drawdown -1.74%   OOS Total +2.16%
```

The optimiser consistently preferred a **later exit (RSI>65 vs the default 55)**
and stayed positive out-of-sample — evidence the parameter is robust, not just
curve-fit. Apply the tuned params:

```python
from qflow import feeds, strategies, backtest
df  = feeds.get("TSLA", "github")
sig = strategies.mean_reversion(df, rsi_buy=10, rsi_exit=65, trend_window=200)
res = backtest.run_backtest(df, sig.signal, sig.atr, capital=10_000, risk_per_trade=0.01)
print(res.report())
```

> **Anti-overfitting rules baked into the workflow**: judge on OOS not IS;
> prefer parameter *plateaus* over lone spikes; keep the grid small; and always
> forward-test the winner on paper before going live.

---

## 4. From edge → paper → live (your month)

1. Pick the edge with the best **OOS / cost-aware** Sharpe (here: TSLA gap-fade).
2. Re-scan it on the **exact instrument** you'll trade (`feeds.get(...)`).
3. Forward-test on paper for 1–2 weeks (`examples/paper_trade.py`,
   see [`PAPER_TRADING.md`](PAPER_TRADING.md)).
4. Go live only if the readiness gate passes — minimum size first.

Gap-fade and lead-lag are **once-a-day, at-the-open** decisions, so they slot
neatly into a single daily cron run.

---

## 5. Forward-testing the edges (now built into the paper engine)

Both edges are wired into the paper-trading engine as **intraday** strategies
(enter at the open, exit at the close, flat overnight). Forward-test them with
virtual money exactly like the swing strategies:

```bash
# gap-fade on a volatile name
python examples/paper_trade.py --strategy gap_fade --symbol TSLA --reset --replay 400

# cross-market lead-lag: trade TSLA off AAPL's prior-day move
python examples/paper_trade.py --strategy lead_lag --symbol TSLA \
    --leader-symbol AAPL --reset --replay 756

# your España -> US version (run on your own machine, open network):
python examples/paper_trade.py --strategy lead_lag --symbol SAN \
    --source yahoo --leader-symbol SAN.MC --leader-source yahoo --step
```

On the bundled samples this reproduces, in the paper account:

| Strategy | Paper result | Readiness |
|----------|--------------|-----------|
| `gap_fade` on TSLA | +4.5%, Sharpe 0.72, 224 trades, win 57% | **5/5 → GO** |
| `lead_lag` TSLA←AAPL | +4.4%, Sharpe 0.59, 215 trades, win 55% | **5/5 → GO** |

Sizing still respects the 1%-risk rule (ATR-based), the journal/equity files are
written as usual, and the same go-live readiness gate applies. For a live
intraday workflow you'd run twice a day (enter near the open, exit near the
close); on end-of-day daily bars a single run simulates the full round trip.
