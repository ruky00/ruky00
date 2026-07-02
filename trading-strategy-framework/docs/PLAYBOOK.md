# Quant Trading Playbook

> **Educational use only — not financial advice.** Every strategy, number and
> backtest in this document is for research and learning. Backtested or
> simulated results do not predict future returns. Markets can lose you money.
> Do your own due diligence and never risk capital you cannot afford to lose.
> See [`DISCLAIMER.md`](DISCLAIMER.md).

This playbook answers twelve common quant-desk questions. Each section is
backed by runnable code in the `qflow` package, so you can reproduce, change
settings, and re-test everything yourself.

**Standing assumptions** (unless a section says otherwise):

| Parameter        | Value                                   |
|------------------|-----------------------------------------|
| Markets          | Liquid crypto (BTC/ETH) & large-cap equities |
| Timeframe        | 1D (daily bars)                         |
| Capital          | $10,000                                 |
| Risk per trade   | 1% of equity ($100 at the start)        |
| Costs modelled   | 2 bps commission + 2 bps slippage / side |

Reproduce the headline numbers with:

```bash
pip install -r requirements.txt
python examples/research/run_all.py
```

---

## 1. Strategy Generation — three reference strategies

The three strategies are deliberately *uncorrelated in regime*: one needs
trends, one needs ranges, one needs volatility expansion. Together they cover
most market states. Code: [`qflow/strategies.py`](../qflow/strategies.py).

### Strategy A — Trend Following (EMA cross + ADX filter)
1. **Indicators**: EMA(20), EMA(50), ADX(14), ATR(14).
2. **Entry**: go **long** when `EMA20 > EMA50` **and** `ADX > 20`; **short**
   the mirror. The ADX gate is the edge — it keeps you out of flat tape where
   moving-average crosses whipsaw.
3. **Exit / stop**: stop = 2×ATR from entry, take-profit = 4×ATR (2:1 R),
   plus exit on the opposite EMA cross.
4. **Works best**: sustained trends — crypto bull runs, equity momentum
   regimes, commodity moves.
5. **Edge**: trend persistence / under-reaction is one of the most-documented
   anomalies (Moskowitz–Ooi–Pedersen "Time Series Momentum"). The ADX filter
   removes the whipsaw drag that kills naïve MA systems.

### Strategy B — Mean Reversion (2-period RSI inside a 200-SMA uptrend)
1. **Indicators**: RSI(2), SMA(200), Bollinger(20, 2σ), ATR(14).
2. **Entry**: only when `close > SMA200` (long-term uptrend) **and** `RSI(2) < 10`
   (short-term washout). Buy the dip, don't catch the knife.
3. **Exit**: `RSI(2) > 55` **or** `%B > 0.8`; hard stop 2×ATR.
4. **Works best**: ranging / mildly-up markets, index ETFs, blue chips.
5. **Edge**: short-term overreaction reverses (Connors 2-period RSI research).
   The 200-SMA trend filter stops you mean-reverting into a downtrend, which is
   where reversion systems blow up.

### Strategy C — Volatility Breakout (Donchian 55/20 + ATR vol filter)
1. **Indicators**: 55-day Donchian channel (entry), 20-day channel (exit),
   ATR(14), 100-day ATR median (vol filter).
2. **Entry**: new 55-day closing high (long) / low (short), **only** when
   `ATR > its 100-day median` so you trade expansion, not chop.
3. **Exit**: opposite 20-day channel; stop 2×ATR.
4. **Works best**: volatility-expansion regimes, new-highs momentum, crypto.
5. **Edge**: the classic Turtle breakout — captures fat-tailed trend bursts.
   Most traders fade breakouts; systematically *taking* them with strict risk
   control harvests the convexity.

> On the random-walk-like synthetic demo data, A is modestly profitable, B and
> C are roughly flat/negative — exactly what you'd expect when there is no real
> structure to exploit. That honesty is the point: **on real instruments with
> genuine regimes the spread between these widens.** Always re-validate on the
> instrument you actually intend to trade.

---

## 2. Backtesting

Engine: [`qflow/backtest.py`](../qflow/backtest.py) — bar-by-bar, next-bar-open
fills (no look-ahead), risk-based sizing, full trade ledger.

Metrics reported: Total Return, **CAGR, Sharpe, Sortino, Calmar, Max Drawdown**,
Avg Recovery, **Win Rate**, Profit Factor.

Example output (Trend Following, ~10y synthetic daily, $10k, 1% risk):

| Metric        | Value   |
|---------------|---------|
| CAGR          | ~1.2%   |
| Sharpe        | ~0.39   |
| Max Drawdown  | ~-7.6%  |
| Win Rate      | ~43%    |
| Profit Factor | ~1.38   |
| Trades        | 91      |

**Performs best**: clean, persistent trends (low whipsaw). **Performs worst**:
choppy, mean-reverting ranges where the EMA cross flips repeatedly.
**What breaks it**: sudden regime flips, gap risk (overnight/weekend in crypto),
and rising correlation across positions. Always test out-of-sample and across
multiple seeds/instruments before trusting a number.

```python
from qflow import data, strategies, backtest
df = data.synthetic_ohlcv(2520, seed=42)
sig = strategies.trend_following(df)
res = backtest.run_backtest(df, sig.signal, sig.atr, capital=10_000, risk_per_trade=0.01)
print(res.report())
```

---

## 3. Risk–Reward Analysis

Helpers: [`qflow/risk.py`](../qflow/risk.py).

- **Risk per trade**: fixed 1% of equity. With entry 100 / stop 96, that's
  $100 risk ÷ $4 per-share-risk = **25 shares** ($2,500 notional, 0.25× lev).
- **Reward:risk**: target 108 → (8/4) = **2.0 : 1**.
- **Expectancy** = `win% × avgWinR − loss% × avgLossR`. At 43% win and 2R wins
  vs 1R losses: `0.43·2 − 0.57·1 = +0.29R` per trade — positive edge.
- **Drawdown pattern**: trend systems draw down in ranges (many small losers),
  recover in bursts. Reversion systems have rare-but-deep drawdowns.

**3 ways to reduce risk**
1. **Volatility targeting** — scale position so each trade contributes equal
   risk; cut size when ATR% spikes.
2. **Correlation cap** — don't run 3 longs in correlated names; treat them as
   one risk unit.
3. **Trend filter on the equity curve** — stop trading a system when its own
   equity falls below its 50-day average (regime-of-the-strategy).

**2 ways to raise returns without raising risk**
1. **Diversify across the 3 strategies** — uncorrelated streams raise Sharpe
   for the same per-trade risk.
2. **Pyramiding into winners** with the *same* total risk budget (add only as
   the stop trails to break-even), improving the average R per trade.

---

## 4. Market Regime Detection

Code: [`qflow/regime.py`](../qflow/regime.py). Classifies every bar on three
axes and recommends a strategy *type*.

- **Trend**: bull / bear / sideways (200-SMA slope + price location + ADX).
- **Volatility**: low / normal / high (ATR% percentile vs its own year).
- **Volume**: expanding / contracting (vs 20-day average).

| Regime (trend, vol)   | Use                                  | Avoid                       |
|-----------------------|--------------------------------------|-----------------------------|
| bull / low            | Trend following, buy-the-dip         | Heavy shorting              |
| sideways / normal     | Mean reversion, range fade           | Trend following (whipsaw)   |
| bear / high           | Reduce size; tactical bounces only   | Leverage, breakout longs    |

```python
from qflow import data, regime
print(regime.current(data.synthetic_ohlcv(2520)))
```

**What to avoid right now**: run `regime.current()` on your live data feed; if
it prints `sideways`, the breakout system will bleed — switch to the
mean-reversion book and shrink size.

---

## 5. Multi-Factor Strategy

Code: [`qflow/multifactor.py`](../qflow/multifactor.py). Cross-sectional model
over a universe; pick the top names each rebalance.

| Factor      | Formula / logic                                  | Weight |
|-------------|--------------------------------------------------|--------|
| Momentum    | 12-1 month total return (skip last month)        | 35%    |
| Value       | −(price / 2-yr mean − 1)  (cheapness proxy)       | 20%    |
| Volatility  | −realised vol (63d, annualised) — low-vol tilt   | 20%    |
| Trend       | price / 200-day SMA − 1                           | 25%    |

Each factor is **z-scored cross-sectionally**, combined by weight, and the top
`N` names are held equal-weight. **Rebalance**: monthly (`ME`); use weekly for
faster signals at higher turnover/cost.

Example portfolio (6-name synthetic universe, top 3): holds the 3 highest
composite-score names, rebalanced month-end, ~6% CAGR / ~0.65 Sharpe on the
demo. Replace the value proxy with real fundamentals (E/P, B/P) for equities.

```python
from qflow import data, multifactor
import pandas as pd
prices = pd.DataFrame({t: data.synthetic_ohlcv(2520, seed=100+i)["close"]
                       for i, t in enumerate("ABCDEF")})
w  = multifactor.build_portfolio(prices, top_n=3, rebalance="ME")
eq = multifactor.backtest_panel(prices, w, capital=10_000)
```

---

## 6. Strategy Optimization

A small grid search over the trend strategy lives in `examples/research/run_all.py`.
It sweeps `EMA fast ∈ {10,20,30}`, `slow ∈ {50,100,150}`, `ADX thr ∈ {15,20,25}`
and keeps the **highest-Sharpe** configuration.

| | Settings | Sharpe | Max DD |
|--|----------|--------|--------|
| Before | 20/50, ADX>20 | baseline | baseline |
| After  | best of grid  | ≥ baseline | reported |

**How to optimise without overfitting**
1. **Walk-forward**: optimise on a rolling in-sample window, trade the next
   out-of-sample window, roll forward.
2. **Prefer plateaus, not peaks**: pick parameters surrounded by other good
   parameters (robust), not a lone spike (curve-fit).
3. **Add filters, don't add parameters**: a single trend/volume/volatility
   gate usually beats finely-tuned indicator lengths.
4. Penalise complexity; keep degrees of freedom low relative to trade count.

---

## 7. Portfolio Construction

Code: [`qflow/portfolio.py`](../qflow/portfolio.py).

- **Allocation**: inverse-volatility ("risk-parity lite") so each asset
  contributes similar risk; calmer assets get more weight.
- **Risk-tolerance overlay**: scale the risky sleeve vs cash —
  low → 40% invested, medium → 70%, high → 100%.
- **Outputs**: target weights, expected return, expected vol, historical max DD.

```python
from qflow import data, portfolio
import pandas as pd
prices = pd.DataFrame({t: data.synthetic_ohlcv(2520, seed=i)["close"]
                       for i, t in enumerate("ABCD")})
out = portfolio.construct(prices, tolerance="medium")
print(out["weights"]); print(out["stats"])
```

**Why each asset is included**: inverse-vol weighting includes every asset but
sizes it to its risk; a low-vol bond-like sleeve stabilises the curve, a
high-vol growth/crypto sleeve provides the return engine, and the CASH bucket
is your dry powder and drawdown buffer.

---

## 8. Trade Setup Generation

A high-probability setup = regime-aligned signal + defined risk. For each idea
specify **entry, stop, take-profit, R:R, reasoning**. Template (illustrative —
plug live prices in):

| # | Idea | Entry | Stop | Target | R:R | Rationale |
|---|------|-------|------|--------|-----|-----------|
| 1 | Trend pullback long | breakout retest | −2×ATR | +4×ATR | 2.0 | EMA20>EMA50, ADX>25, volume expanding |
| 2 | Range fade long | lower Bollinger | below band −1×ATR | mid band | ~1.5 | RSI(2)<10 in 200-SMA uptrend |
| 3 | Breakout long | new 55-day high | −2×ATR | trail 20-day | open | ATR>median, fresh momentum |

Generate them programmatically by running the strategies on your latest bar and
reading the entry/stop/target the backtester would use (entry ± `stop_atr`/`target_atr` × ATR).

---

## 9. Monte-Carlo Simulation

Code: [`qflow/montecarlo.py`](../qflow/montecarlo.py). Bootstraps the strategy's
**actual realised trade R-multiples** (resample with replacement, 5,000 paths).
No distributional assumption — it reshuffles the trades you really got.

Trend strategy example output:

```
Probability of loss    :  ~13%
P(>50% drawdown)       :   ~0%
Median final return    :  ~13%
Return  5th pctile     :  ~-5%
Worst-case max DD (p05): ~-13%
Verdict: ROBUST — positive across most resamples, survivable worst case.
```

**Robust vs fragile**: robust if the return distribution stays mostly positive
and the 5th-percentile drawdown is survivable; fragile if a mere reshuffle of
the *same* trades can wipe the account (→ size down or rework the edge).

---

## 10. Drawdown Analysis

Code: metrics in [`qflow/metrics.py`](../qflow/metrics.py)
(`drawdown_series`, `max_drawdown`, `avg_recovery_time`).

Trend strategy: **max DD ~-7.6%**, avg recovery ~215 days, ~77% of bars spent
below a prior peak (typical for trend systems — many shallow dips, few sharp
gains).

**3 ways to reduce drawdowns**
1. **Equity-curve trading** — stand down when the strategy's equity is below
   its own moving average.
2. **Volatility targeting** — cut size as realised vol rises; most deep
   drawdowns happen in high-vol clusters.
3. **Diversify across uncorrelated strategies/assets** — overlapping drawdowns
   are what create the deep ones.

**Position-sizing improvements**: fractional-Kelly (¼–½), per-trade risk cap,
and a portfolio heat cap (max total open risk, e.g. 3–5% of equity).

---

## 11. Macro-Based Strategy

A regime overlay driven by three macro factors (use real series: Fed funds /
2y yield, CPI YoY, ISM PMI / GDP nowcast).

| Factor          | Effect on positioning |
|-----------------|-----------------------|
| **Rates ↑**     | Headwind for long-duration/growth & gold; favour value, cash, short-duration. Rising-rate trend → reduce equity beta. |
| **Inflation ↑** | Tilt to real assets (commodities, energy, TIPS); trim long bonds. |
| **Growth ↑** (PMI>50 & rising) | Risk-on: overweight equities/cyclicals/crypto; **Growth ↓** → defensives, cash, treasuries. |

**Signals**: go risk-on when `PMI > 50 & rising` **and** `real rates falling`;
risk-off when `PMI < 50 & falling` **or** `policy tightening + inflation high`.
**Example trade**: PMI turns up through 50 while the Fed pauses → rotate the
risky sleeve from defensives into cyclicals/crypto; reverse on the next
inversion. Implement as a monthly multiplier on the equity weight in
`portfolio.construct`.

---

## 12. Alpha / Edge Detection

**Behavioral inefficiencies most traders miss**
- *Overnight/weekend drift in crypto* — retail-driven gaps create systematic
  edges around session boundaries.
- *Post-earnings / post-news drift* — under-reaction persists for days.
- *Round-number & liquidation clustering* — stops bunch at obvious levels.

**Market-structure gaps**
- *Funding-rate / basis arbitrage* in perps vs spot (carry without directional risk).
- *Index-rebalance and ETF-flow front-running* windows.

**Two unique strategies**
1. **Crypto funding-carry**: hold spot, short the perp when funding is richly
   positive — collect funding, hedge price. Edge: most discretionary traders
   chase price and ignore the carry. Execute: monitor funding across venues,
   size by basis, exit when funding normalises.
2. **Volatility-regime breakout switch**: trade breakouts *only* in the top
   ATR-percentile regime and mean-reversion in the bottom — i.e. let
   `qflow.regime` flip the book. Edge: most traders run one style in all
   regimes and give back profits in the wrong one. Execute: gate each
   strategy's signal by `regime.detect()`.

**Why most miss them**: they require patience (carry pays slowly), discipline
(only trade the right regime), and infrastructure (cross-venue data) rather
than a flashy indicator — so they stay under-exploited.

---

### Reproduce everything
```bash
python examples/research/run_all.py     # full walkthrough on offline synthetic data
python tests/test_framework.py # correctness checks
```
