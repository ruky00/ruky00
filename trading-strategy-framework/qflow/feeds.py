"""
Real market-data feeds (stdlib only — no extra dependencies).

Sources
-------
* ``binance``  : Binance public klines  — crypto, no API key (BTCUSDT, ETHUSDT, ...)
* ``stooq``    : Stooq CSV               — stocks / ETFs / FX, no key (aapl.us, eurusd, ...)
* ``yahoo``    : Yahoo Finance chart API — stocks / ETFs / crypto, no key (AAPL, BTC-USD)
* ``github``   : raw CSV on GitHub       — bundled real sample datasets (works everywhere)

Everything is normalised to the framework's canonical OHLCV frame:
    index = DatetimeIndex (named 'date')
    columns = [open, high, low, close, volume]

A local CSV cache (``data/`` by default) means you fetch once and re-use, which
also lets the offline test environment replay real data that was pulled earlier.

Note: in a locked-down/sandboxed network only the bundled ``github`` samples may
be reachable; ``binance``/``stooq``/``yahoo`` work from an ordinary machine.
"""

from __future__ import annotations

import io
import json
import os
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone

import pandas as pd

CANONICAL = ["open", "high", "low", "close", "volume"]
DEFAULT_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Bundled real sample datasets (daily OHLCV) hosted on GitHub.
# value = (url, colmap, name_filter). name_filter selects one ticker's rows from a
# multi-ticker file (the S&P-500 5-year set). Newer names (COIN, PLTR) and 10-year
# history are not available offline here — use source="yahoo" on an open network.
_SP500 = "https://raw.githubusercontent.com/plotly/datasets/master/all_stocks_5yr.csv"
GITHUB_SAMPLES = {
    "AAPL": ("https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv",
             {"date": "Date", "open": "AAPL.Open", "high": "AAPL.High",
              "low": "AAPL.Low", "close": "AAPL.Close", "volume": "AAPL.Volume"}, None),
    "TSLA": ("https://raw.githubusercontent.com/plotly/datasets/master/tesla-stock-price.csv",
             {"date": "date", "open": "open", "high": "high",
              "low": "low", "close": "close", "volume": "volume"}, None),
    # S&P-500 five-year set (2013-2018), full OHLCV, selected by ticker name
    "NVDA": (_SP500, None, "NVDA"),
    "AMD":  (_SP500, None, "AMD"),
    "NFLX": (_SP500, None, "NFLX"),
    "AMZN": (_SP500, None, "AMZN"),
    "MSFT": (_SP500, None, "MSFT"),
    "GOOGL": (_SP500, None, "GOOGL"),
}

_HEADERS = {"User-Agent": "qflow/0.1 (research)"}


# --------------------------------------------------------------------------- #
# low-level fetch with retry
# --------------------------------------------------------------------------- #
def _http_get(url: str, retries: int = 3, timeout: int = 30) -> bytes:
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            last = e
            time.sleep(2 ** i)
    raise ConnectionError(
        f"Failed to fetch {url!r} after {retries} tries: {last}. "
        "If you are in a restricted/sandboxed network, this host may be blocked "
        "by policy — try the bundled 'github' samples or run from an open network."
    )


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #
def _normalise(df: pd.DataFrame, colmap: dict | None = None) -> pd.DataFrame:
    """Coerce an arbitrary OHLCV frame into the canonical layout."""
    if colmap:
        rename = {v: k for k, v in colmap.items()}
        df = df.rename(columns=rename)
    else:
        # auto-map by suffix (handles 'AAPL.Open', 'Adj Close', etc.)
        low = {c: c.lower().strip() for c in df.columns}
        mapping = {}
        for c, lc in low.items():
            for field in ("open", "high", "low", "close", "volume"):
                if lc == field or lc.endswith("." + field) or lc.endswith("_" + field):
                    mapping.setdefault(field, c)
            if lc in ("date", "time", "datetime", "timestamp"):
                mapping.setdefault("date", c)
        df = df.rename(columns={v: k for k, v in mapping.items()})

    if "date" not in df.columns:
        df = df.reset_index().rename(columns={df.index.name or "index": "date"})

    # Drop junk rows whose date string has no 4-digit year (e.g. a stray
    # "11:34" header artifact that dateutil would silently map to "today").
    date_str = df["date"].astype(str)
    df = df[date_str.str.contains(r"\d{4}", na=False)].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()

    for col in CANONICAL:
        if col not in df.columns:
            if col == "volume":
                df[col] = 0.0          # some indices have no volume
            else:
                raise ValueError(f"Source is missing required column '{col}'.")
        # strip thousands separators before numeric coercion
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", "", regex=False), errors="coerce"
        )
    df = df[CANONICAL].dropna(subset=["open", "high", "low", "close"])
    df.index.name = "date"
    return df


# --------------------------------------------------------------------------- #
# per-source loaders
# --------------------------------------------------------------------------- #
def from_github(symbol: str) -> pd.DataFrame:
    """Load a bundled real sample. Prefers the committed copy under
    data/samples/<SYM>.csv (offline-first, and avoids re-downloading the 30 MB
    multi-ticker file); only fetches from GitHub when the local file is missing,
    applying the per-ticker name filter."""
    sym = symbol.upper()
    if sym not in GITHUB_SAMPLES:
        raise KeyError(f"No bundled sample for {symbol!r}. Available: {list(GITHUB_SAMPLES)}")
    local = os.path.join(DEFAULT_CACHE, "samples", f"{sym}.csv")
    if os.path.exists(local):
        df = pd.read_csv(local, parse_dates=["date"]).set_index("date")
        return df[CANONICAL]

    url, colmap, name_filter = GITHUB_SAMPLES[sym]
    raw = _http_get(url)
    df = pd.read_csv(io.BytesIO(raw))
    if name_filter is not None:
        df = df[df["Name"] == name_filter]
    return _normalise(df, colmap)


def from_binance(symbol: str = "BTCUSDT", interval: str = "1d", limit: int = 1000) -> pd.DataFrame:
    """Binance public klines. interval: 1m,5m,15m,1h,4h,1d,1w. limit<=1000."""
    url = (f"https://api.binance.com/api/v3/klines"
           f"?symbol={symbol.upper()}&interval={interval}&limit={int(limit)}")
    raw = json.loads(_http_get(url))
    rows = [{
        "date": datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc).replace(tzinfo=None),
        "open": k[1], "high": k[2], "low": k[3], "close": k[4], "volume": k[5],
    } for k in raw]
    return _normalise(pd.DataFrame(rows))


def from_stooq(symbol: str = "aapl.us", interval: str = "d") -> pd.DataFrame:
    """Stooq CSV. symbol e.g. 'aapl.us', 'spy.us', 'eurusd'. interval: d/w/m."""
    url = f"https://stooq.com/q/d/l/?s={symbol.lower()}&i={interval}"
    raw = _http_get(url)
    df = pd.read_csv(io.BytesIO(raw))
    return _normalise(df)


def yahoo_symbol(symbol: str) -> str:
    """Map a broker symbol to Yahoo's convention (EURUSD -> EURUSD=X)."""
    from .broker import is_fx_pair
    if is_fx_pair(symbol) and not symbol.upper().endswith("=X"):
        return symbol.upper() + "=X"
    return symbol


def from_yahoo(symbol: str = "AAPL", rng: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """Yahoo Finance chart API. rng: 1y,2y,5y,10y,max. interval: 1d,1wk,1h.
    FX pairs (EURUSD) and European suffixes (SAN.MC) are accepted directly."""
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol(symbol)}"
           f"?range={rng}&interval={interval}")
    data = json.loads(_http_get(url))
    res = data["chart"]["result"][0]
    ts = res["timestamp"]
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame({
        "date": [datetime.fromtimestamp(t, tz=timezone.utc).replace(tzinfo=None) for t in ts],
        "open": q["open"], "high": q["high"], "low": q["low"],
        "close": q["close"], "volume": q["volume"],
    })
    return _normalise(df)


_DISPATCH = {
    "github": lambda sym, **kw: from_github(sym),
    "binance": lambda sym, **kw: from_binance(sym, **kw),
    "stooq": lambda sym, **kw: from_stooq(sym, **kw),
    "yahoo": lambda sym, **kw: from_yahoo(sym, **kw),
}


# --------------------------------------------------------------------------- #
# public API with caching
# --------------------------------------------------------------------------- #
def get(symbol: str,
        source: str = "github",
        cache_dir: str | None = DEFAULT_CACHE,
        refresh: bool = False,
        **kwargs) -> pd.DataFrame:
    """
    Fetch (and cache) an OHLCV series.

        get("AAPL", "github")                    # bundled real sample, offline-safe
        get("BTCUSDT", "binance", interval="1d") # crypto (open network)
        get("aapl.us", "stooq")                  # equities/FX (open network)
        get("AAPL", "yahoo", rng="10y")          # 10y daily (open network)

    Cached CSVs live in ``cache_dir`` and are reused unless ``refresh=True``.
    """
    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        safe = symbol.replace("/", "_").replace(".", "_")
        cache_path = os.path.join(cache_dir, f"{source}_{safe}.csv")
        if os.path.exists(cache_path) and not refresh:
            df = pd.read_csv(cache_path, parse_dates=["date"]).set_index("date")
            return df[CANONICAL]

    if source not in _DISPATCH:
        raise ValueError(f"Unknown source {source!r}. Use one of {list(_DISPATCH)}.")
    df = _DISPATCH[source](symbol, **kwargs)

    if cache_path:
        df.reset_index().to_csv(cache_path, index=False)
    return df


def save_csv(df: pd.DataFrame, path: str) -> None:
    df.reset_index().to_csv(path, index=False)
