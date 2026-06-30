"""
Real-time news + sentiment layer — provider-agnostic.

Design philosophy
-----------------
A Bloomberg Terminal (~$25-30k/yr, API only works with a live Terminal) is
overkill for a retail-sized system, so news here is an *interface* with
pluggable providers. Bloomberg is just one optional adapter; the defaults are
free/cheap sources:

    RSSProvider       : Yahoo Finance / Google News RSS  (no key, stdlib only)
    FinnhubProvider   : finnhub.io company-news API       (free tier, needs key)
    NewsAPIProvider   : newsapi.org                        (free tier, needs key)
    BloombergProvider : blpapi  (requires a running Bloomberg Terminal / B-PIPE)
    SampleProvider    : bundled offline headlines          (testing/demo)

Every provider returns ``list[NewsItem]``. A lightweight finance-tuned sentiment
scorer (qflow.lexicon) tags each headline in [-1, +1]. ``news_overlay`` turns
recent news into a *trading risk overlay*: veto or down-size entries that fight
strong adverse sentiment, and flag elevated event risk (earnings, FDA, Fed…).

Nothing here is investment advice; sentiment scores are noisy heuristics.
"""

from __future__ import annotations

import os
import re
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

from . import lexicon

_HEADERS = {"User-Agent": "qflow-news/0.1 (research)"}
_TOKEN = re.compile(r"[a-zA-Z'\-]+")


# --------------------------------------------------------------------------- #
# data model
# --------------------------------------------------------------------------- #
@dataclass
class NewsItem:
    symbol: str
    headline: str
    source: str
    url: str = ""
    summary: str = ""
    published: str = ""           # ISO timestamp if known
    sentiment: float = 0.0        # [-1, +1]
    is_event: bool = False        # high-attention event word present

    def to_dict(self):
        return asdict(self)


# --------------------------------------------------------------------------- #
# sentiment scoring
# --------------------------------------------------------------------------- #
def score_sentiment(text: str) -> tuple[float, bool]:
    """
    Return (sentiment in [-1,1], is_event). Lexicon-based with simple negation:
    a negator within 2 tokens flips the polarity of a sentiment word.
    """
    tokens = [t.lower() for t in _TOKEN.findall(text or "")]
    if not tokens:
        return 0.0, False
    score = 0
    is_event = False
    for i, tok in enumerate(tokens):
        if tok in lexicon.EVENT:
            is_event = True
        pol = 0
        if tok in lexicon.POSITIVE:
            pol = 1
        elif tok in lexicon.NEGATIVE:
            pol = -1
        if pol:
            window = tokens[max(0, i - 2):i]
            if any(w in lexicon.NEGATORS for w in window):
                pol = -pol
            score += pol
    # squash to [-1, 1] by hits relative to a soft cap
    n_hits = sum(1 for t in tokens if t in lexicon.POSITIVE or t in lexicon.NEGATIVE)
    if n_hits == 0:
        return 0.0, is_event
    norm = max(3, n_hits)
    return max(-1.0, min(1.0, score / norm)), is_event


class LexiconScorer:
    """Default fast scorer — the finance lexicon above. No dependencies."""
    name = "lexicon"

    def score(self, text: str) -> tuple[float, bool]:
        return score_sentiment(text)


class FinBERTScorer:
    """
    Optional research-grade scorer using FinBERT (ProsusAI/finbert) via
    HuggingFace `transformers` + `torch`. These are heavy dependencies and the
    model (~440 MB) downloads on first use, so it is loaded lazily and only if
    you explicitly select it:

        from qflow import news
        news.set_scorer(news.FinBERTScorer())     # then providers use FinBERT

    Sentiment = P(positive) - P(negative); the event flag still comes from the
    lexicon's EVENT words. Falls back with a clear error if transformers/torch
    are not installed.
    """
    name = "finbert"

    def __init__(self, model: str = "ProsusAI/finbert", device: int = -1):
        self.model = model
        self.device = device
        self._pipe = None

    def _ensure(self):
        if self._pipe is None:
            try:
                from transformers import pipeline
            except ImportError as e:
                raise RuntimeError(
                    "FinBERTScorer needs `transformers` and `torch`:\n"
                    "    pip install transformers torch\n"
                    "For a lightweight setup keep the default LexiconScorer."
                ) from e
            self._pipe = pipeline("sentiment-analysis", model=self.model,
                                  device=self.device, truncation=True)
        return self._pipe

    def score(self, text: str) -> tuple[float, bool]:
        text = (text or "").strip()
        if not text:
            return 0.0, False
        pipe = self._ensure()
        out = pipe(text[:512], top_k=None)          # all class scores
        probs = {d["label"].lower(): d["score"] for d in out}
        sentiment = probs.get("positive", 0.0) - probs.get("negative", 0.0)
        is_event = any(t.lower() in lexicon.EVENT for t in _TOKEN.findall(text))
        return float(max(-1.0, min(1.0, sentiment))), is_event


# Active scorer (swappable). Defaults to the dependency-free lexicon.
_SCORER = LexiconScorer()


def set_scorer(scorer) -> None:
    """Swap the sentiment scorer used by all providers (e.g. a FinBERTScorer)."""
    global _SCORER
    _SCORER = scorer


def get_scorer():
    return _SCORER


def _score_items(items: list[NewsItem]) -> list[NewsItem]:
    for it in items:
        s, ev = _SCORER.score(f"{it.headline}. {it.summary}")
        it.sentiment = round(s, 3)
        it.is_event = ev
    return items


# --------------------------------------------------------------------------- #
# providers
# --------------------------------------------------------------------------- #
class NewsProvider:
    """Interface: implement ``fetch(symbol, limit) -> list[NewsItem]``."""
    name = "base"

    def fetch(self, symbol: str, limit: int = 20) -> list[NewsItem]:
        raise NotImplementedError


def _http_get(url: str, retries: int = 2, timeout: int = 20) -> bytes:
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            last = e
            time.sleep(1.5 * (i + 1))
    raise ConnectionError(
        f"news fetch failed for {url!r}: {last}. In a restricted network this "
        "host may be blocked; use SampleProvider offline or run on an open network."
    )


class RSSProvider(NewsProvider):
    """Free RSS news, no API key. Defaults to Yahoo Finance per-symbol feed."""
    name = "rss"

    YAHOO = ("https://feeds.finance.yahoo.com/rss/2.0/headline"
             "?s={sym}&region=US&lang=en-US")
    GOOGLE = "https://news.google.com/rss/search?q={sym}+stock&hl=en-US&gl=US&ceid=US:en"

    def __init__(self, source: str = "yahoo"):
        self.source = source

    def fetch(self, symbol: str, limit: int = 20) -> list[NewsItem]:
        import xml.etree.ElementTree as ET
        url = (self.YAHOO if self.source == "yahoo" else self.GOOGLE).format(sym=symbol)
        raw = _http_get(url)
        root = ET.fromstring(raw)
        items = []
        for node in root.iter("item"):
            title = (node.findtext("title") or "").strip()
            link = (node.findtext("link") or "").strip()
            desc = re.sub("<[^>]+>", "", node.findtext("description") or "").strip()
            pub = (node.findtext("pubDate") or "").strip()
            if title:
                items.append(NewsItem(symbol=symbol, headline=title, source=self.name,
                                      url=link, summary=desc, published=pub))
            if len(items) >= limit:
                break
        return _score_items(items)


class FinnhubProvider(NewsProvider):
    """finnhub.io company news. Needs FINNHUB_API_KEY (free tier available)."""
    name = "finnhub"

    def __init__(self, api_key: str | None = None):
        self.key = api_key or os.environ.get("FINNHUB_API_KEY", "")

    def fetch(self, symbol: str, limit: int = 20) -> list[NewsItem]:
        if not self.key:
            raise ValueError("FinnhubProvider needs FINNHUB_API_KEY.")
        import json
        from datetime import timedelta
        today = datetime.now(timezone.utc).date()
        frm = today - timedelta(days=7)
        url = (f"https://finnhub.io/api/v1/company-news?symbol={symbol}"
               f"&from={frm}&to={today}&token={self.key}")
        data = json.loads(_http_get(url))
        items = [NewsItem(symbol=symbol, headline=d.get("headline", ""),
                          source=self.name, url=d.get("url", ""),
                          summary=d.get("summary", ""),
                          published=datetime.fromtimestamp(
                              d.get("datetime", 0), timezone.utc).isoformat())
                 for d in data[:limit]]
        return _score_items(items)


class NewsAPIProvider(NewsProvider):
    """newsapi.org. Needs NEWSAPI_KEY (free tier available)."""
    name = "newsapi"

    def __init__(self, api_key: str | None = None):
        self.key = api_key or os.environ.get("NEWSAPI_KEY", "")

    def fetch(self, symbol: str, limit: int = 20) -> list[NewsItem]:
        if not self.key:
            raise ValueError("NewsAPIProvider needs NEWSAPI_KEY.")
        import json
        url = (f"https://newsapi.org/v2/everything?q={symbol}&language=en"
               f"&sortBy=publishedAt&pageSize={limit}&apiKey={self.key}")
        data = json.loads(_http_get(url))
        items = [NewsItem(symbol=symbol, headline=a.get("title", ""),
                          source=self.name, url=a.get("url", ""),
                          summary=a.get("description", "") or "",
                          published=a.get("publishedAt", ""))
                 for a in data.get("articles", [])[:limit]]
        return _score_items(items)


class BloombergProvider(NewsProvider):
    """
    Bloomberg adapter — requires the `blpapi` package AND a running Bloomberg
    Terminal (or B-PIPE / Server API entitlement) on the host. There is no way
    to use Bloomberg data without that paid entitlement, so this provider raises
    an explanatory error if the environment isn't present. The request scaffold
    below is the real shape of a Bloomberg news/reference request for when you
    do have a Terminal.
    """
    name = "bloomberg"

    def __init__(self, host: str = "localhost", port: int = 8194):
        self.host, self.port = host, port

    def fetch(self, symbol: str, limit: int = 20) -> list[NewsItem]:
        try:
            import blpapi  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "BloombergProvider requires the 'blpapi' package and a running "
                "Bloomberg Terminal/B-PIPE entitlement (~$25-30k/yr). For a "
                "retail-sized system use RSSProvider or FinnhubProvider instead."
            ) from e
        # --- real Terminal path (only reachable with blpapi + entitlement) ---
        session = blpapi.Session(self._opts())          # pragma: no cover
        if not session.start() or not session.openService("//blp/refdata"):
            raise ConnectionError("Could not connect to the Bloomberg session.")
        # A production implementation issues a //blp/mktnews subscription or a
        # ReferenceDataRequest for news fields here and maps results to NewsItem.
        raise NotImplementedError(
            "Connected to Bloomberg, but news mapping is left as an entitlement-"
            "specific implementation. Fill in per your //blp/mktnews schema."
        )

    def _opts(self):                                     # pragma: no cover
        import blpapi
        opts = blpapi.SessionOptions()
        opts.setServerHost(self.host)
        opts.setServerPort(self.port)
        return opts


class SampleProvider(NewsProvider):
    """Bundled offline headlines so the pipeline runs/tests without a network."""
    name = "sample"

    _DB = {
        "AAPL": ["Apple shares rally as iPhone demand beats expectations",
                 "Analysts upgrade Apple, raise price target on strong growth",
                 "Apple faces antitrust probe in Europe over App Store"],
        "TSLA": ["Tesla plunges after disappointing delivery numbers and guidance cut",
                 "Tesla recalls vehicles, regulators open investigation",
                 "Tesla rebounds as margins beat and demand recovers"],
        "_default": ["Company reports record quarterly profit and raises dividend",
                     "Shares fall on weak outlook and earnings miss",
                     "Stock steady ahead of Fed rate decision"],
    }

    def fetch(self, symbol: str, limit: int = 20) -> list[NewsItem]:
        heads = self._DB.get(symbol.upper(), self._DB["_default"])
        items = [NewsItem(symbol=symbol, headline=h, source=self.name)
                 for h in heads[:limit]]
        return _score_items(items)


# --------------------------------------------------------------------------- #
# aggregation + trading overlay
# --------------------------------------------------------------------------- #
def get_news(symbol: str, providers=None, limit: int = 20) -> list[NewsItem]:
    """Aggregate scored news across providers; providers that fail are skipped."""
    providers = providers or [RSSProvider()]
    out: list[NewsItem] = []
    for p in providers:
        try:
            out.extend(p.fetch(symbol, limit))
        except Exception:
            continue
    return out


def summarise(items: list[NewsItem]) -> dict:
    if not items:
        return {"n": 0, "avg_sentiment": 0.0, "n_positive": 0, "n_negative": 0,
                "n_events": 0, "tone": "neutral"}
    sents = [it.sentiment for it in items]
    avg = sum(sents) / len(sents)
    tone = "positive" if avg > 0.15 else "negative" if avg < -0.15 else "neutral"
    return {
        "n": len(items),
        "avg_sentiment": round(avg, 3),
        "n_positive": sum(1 for s in sents if s > 0.1),
        "n_negative": sum(1 for s in sents if s < -0.1),
        "n_events": sum(1 for it in items if it.is_event),
        "tone": tone,
    }


def news_overlay(items: list[NewsItem], direction: int,
                 veto_threshold: float = 0.4,
                 caution_threshold: float = 0.2) -> dict:
    """
    Convert recent news into a trading risk overlay for a proposed trade.

    Returns {size_multiplier, action, reason}:
      * fighting strong adverse sentiment -> size 0 (veto)
      * mild adverse sentiment or an active event -> size 0.5 (caution)
      * otherwise -> size 1.0
    `direction`: +1 for a long, -1 for a short.
    """
    s = summarise(items)
    adverse = -direction * s["avg_sentiment"]      # >0 means news fights the trade
    if s["n"] == 0:
        return {"size_multiplier": 1.0, "action": "proceed", "reason": "no news"}
    if adverse >= veto_threshold:
        return {"size_multiplier": 0.0, "action": "veto",
                "reason": f"news strongly against {('long' if direction>0 else 'short')} "
                          f"(avg {s['avg_sentiment']:+.2f}, {s['n']} items)"}
    if adverse >= caution_threshold or s["n_events"] > 0:
        return {"size_multiplier": 0.5, "action": "reduce",
                "reason": f"adverse/elevated event risk (avg {s['avg_sentiment']:+.2f}, "
                          f"{s['n_events']} event headlines)"}
    return {"size_multiplier": 1.0, "action": "proceed",
            "reason": f"news ok (avg {s['avg_sentiment']:+.2f}, {s['n']} items)"}


def report(symbol: str, items: list[NewsItem]) -> str:
    s = summarise(items)
    L = [f"NEWS — {symbol}  ({s['n']} items, tone={s['tone']}, "
         f"avg sentiment {s['avg_sentiment']:+.2f}, {s['n_events']} events)",
         "-" * 64]
    for it in items[:10]:
        tag = "EVENT " if it.is_event else "      "
        L.append(f"  {it.sentiment:+.2f} {tag}{it.headline[:70]}")
    return "\n".join(L)
