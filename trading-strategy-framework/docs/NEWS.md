# News & Sentiment Layer (and the Bloomberg question)

> **Educational use only — not financial advice.** Sentiment scores are noisy
> heuristics from a small lexicon, not a research-grade NLP signal. See
> [`DISCLAIMER.md`](DISCLAIMER.md).

## Is a Bloomberg Terminal worth integrating? — honest verdict

**No, not for a retail-sized system.** A Bloomberg Terminal costs
**~$25,000–32,000 per year, per seat**, and its API (`blpapi` / B-PIPE) only
returns data when a licensed Terminal (or an enterprise Server API entitlement)
is actually running on the host. For a $10k account that is wildly
disproportionate — you'd spend 2–3× your trading capital on the data feed.

**But the underlying idea is right.** Real-time news + sentiment *is* a valuable
trading input. The correct engineering answer is not "bolt on Bloomberg" but a
**provider-agnostic news layer** where Bloomberg is one optional adapter and the
defaults are free/cheap sources. That's what `qflow/news.py` implements.

| Provider | Cost | Needs | Use |
|----------|------|-------|-----|
| `RSSProvider` | free | nothing (stdlib) | **default** — Yahoo Finance / Google News RSS |
| `FinnhubProvider` | free tier | `FINNHUB_API_KEY` | structured company news |
| `NewsAPIProvider` | free tier | `NEWSAPI_KEY` | broad article search |
| `BloombergProvider` | ~$25–30k/yr | `blpapi` + running Terminal | only if you already pay for it |
| `SampleProvider` | free | nothing | offline tests/demo |

All providers return the same `NewsItem` objects, so you can switch sources
without touching the rest of the system.

## What it does

1. **Fetch** headlines for a symbol from any provider.
2. **Score** each headline in `[-1, +1]` with a finance-tuned lexicon
   (Loughran-McDonald-inspired) including simple negation and event detection
   (earnings / FDA / Fed / M&A …). See `qflow/lexicon.py`.
3. **Overlay onto trades** — `news_overlay(items, direction)` returns a position
   size multiplier:
   - `x1.0` — trade normally,
   - `x0.5` — adverse tone **or** an active event (earnings etc.): half size,
   - `x0.0` — news strongly against the trade: **skip it**.

```bash
python examples/news_demo.py
```
```
NEWS — TSLA  (3 items, tone=negative, avg sentiment -0.22, 1 events)
  -1.00 EVENT Tesla plunges after disappointing delivery numbers and guidance cut
  -0.67       Tesla recalls vehicles, regulators open investigation
  +1.00       Tesla rebounds as margins beat and demand recovers
```

## Wiring it into trading

The overlay is built into the paper-trading engine and fires **only on the live
`--step` path** — never during historical replay, because there is no historical
news feed, so backtests stay clean and reproducible.

```bash
# live daily run with a free RSS news overlay
python examples/paper_trade.py --strategy mean_reversion --symbol AAPL \
    --news rss --step
```
```python
from qflow import news
from qflow.paper import PaperTrader

pt = PaperTrader("AAPL", strategy="mean_reversion",
                 news_provider=news.RSSProvider())     # or FinnhubProvider()
pt.step()        # before opening, checks news; vetoes/halves size on adverse tone
print(pt.report())   # shows a "News overlay" line with the current tone
```

When the overlay vetoes a trade, the journal records a `SKIP` row with the
reason, so you can audit every decision the news layer influenced.

## Honest limitations

- Headline lexicon sentiment is crude; it will mis-score sarcasm, complex
  sentences, and novel phrasing. For production, switch to **FinBERT** (built
  in) behind the same interface:
  ```python
  from qflow import news
  news.set_scorer(news.FinBERTScorer())   # needs: pip install transformers torch
  # ...all providers now score with FinBERT; CLI: --news rss --finbert
  ```
  FinBERT downloads a ~440 MB model on first use and is much slower than the
  lexicon, so it is opt-in. The lexicon stays the zero-dependency default.
- News is **latency-sensitive**: free RSS lags the market by minutes. It is best
  used as a **risk filter** ("don't fight fresh bad news", "halve size into
  earnings") rather than a primary alpha signal.
- Always confirm that a news-gated rule actually improves your paper results
  before relying on it live.
