"""
Broker execution layer — the bridge from signals to real orders.

The whole framework decides *what* to trade; a ``BrokerAdapter`` decides *where*
the order actually goes. Two implementations share one interface:

    PaperBroker : in-memory simulation (default, no dependencies, fully testable)
    IBKRBroker  : Interactive Brokers via `ib_insync` + a running TWS/IB Gateway

The key method is ``place_bracket``: it submits an entry plus a **stop-loss and
take-profit as one linked (OCO) order**. With IBKR those exit legs live on the
broker's servers, so your stop is honoured even if your script crashes or your
machine goes offline.

SAFETY: IBKRBroker defaults to the **paper-trading port (7497)**. Connecting to a
live port requires passing ``allow_live=True`` explicitly — a deliberate friction
so you never go live by accident.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


def _ensure_event_loop():
    """Work around ib_insync/eventkit failing to import on Python >= 3.13/3.14.

    eventkit calls ``asyncio.get_event_loop()`` at import time; on newer Pythons
    that raises when no loop exists. Create one first so the import succeeds.
    """
    import asyncio
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


@dataclass
class BracketResult:
    order_id: str
    symbol: str
    side: str            # BUY / SELL
    qty: float
    entry: float
    stop: float
    target: float
    status: str = "submitted"
    broker: str = ""

    def to_dict(self):
        return asdict(self)


class BrokerAdapter:
    """Interface every broker must implement."""
    name = "base"

    def connect(self): ...
    def disconnect(self): ...
    def is_connected(self) -> bool: return True
    def account(self) -> dict: raise NotImplementedError
    def positions(self) -> dict: raise NotImplementedError
    def place_bracket(self, symbol, qty, side, entry, stop, target,
                      entry_type="MKT") -> BracketResult: raise NotImplementedError
    def place_market(self, symbol, qty, side) -> BracketResult:
        """Plain market order (no bracket) — used by the intraday gap-fade routine."""
        raise NotImplementedError
    def cancel_all(self, symbol=None): ...
    def flatten(self, symbol=None): ...

    # Broker-agnostic helpers so the live loop never touches a broker-specific
    # attribute (e.g. IBKR's .ib). Overridden where a broker can do better.
    def sleep(self, seconds: float):
        """Wait, pumping the broker's event loop if it has one."""
        import time
        time.sleep(seconds)

    def account_label(self) -> str:
        """Short id for the header line (account number / broker name)."""
        return self.name

    def bars(self, symbol: str, interval: str = "5m", count: int = 800):
        """Return recent OHLCV bars for `symbol` from the broker's own feed, or
        None if this broker has no data feed (caller then falls back to feeds.*)."""
        return None

    def size_for_risk(self, symbol: str, risk_amount: float, stop_dist: float,
                      price: float = 0.0):
        """Position size (in the broker's units) so that hitting the stop loses
        ~`risk_amount`, or None if this broker can't size itself (caller then uses
        its own share sizer). `stop_dist` is the stop distance in price."""
        return None


# --------------------------------------------------------------------------- #
# Paper broker — in-memory, no dependencies (default / testing)
# --------------------------------------------------------------------------- #
class PaperBroker(BrokerAdapter):
    """
    Simulates a broker: tracks cash, positions and the bracket levels in memory.
    Fills the entry immediately at the requested price; the engine's own logic
    (or you) decides when stop/target are hit. Useful to exercise the full live
    code path without any external connection.
    """
    name = "paper"

    def __init__(self, cash: float = 10_000.0):
        self._cash = cash
        self._positions: dict[str, dict] = {}
        self._orders: list[BracketResult] = []
        self._connected = False
        self._seq = 0

    def connect(self): self._connected = True
    def disconnect(self): self._connected = False
    def is_connected(self): return self._connected

    def account(self) -> dict:
        mkt = sum(p["qty"] * p["entry"] for p in self._positions.values())
        return {"cash": round(self._cash, 2), "positions_value": round(mkt, 2),
                "equity": round(self._cash + mkt, 2)}

    def positions(self) -> dict:
        return dict(self._positions)

    def place_bracket(self, symbol, qty, side, entry, stop, target,
                      entry_type="MKT") -> BracketResult:
        self._seq += 1
        oid = f"PB-{self._seq}"
        direction = 1 if side.upper() == "BUY" else -1
        self._positions[symbol] = {"qty": qty * direction, "entry": entry,
                                   "stop": stop, "target": target}
        res = BracketResult(order_id=oid, symbol=symbol, side=side.upper(),
                            qty=qty, entry=entry, stop=stop, target=target,
                            status="filled", broker=self.name)
        self._orders.append(res)
        return res

    def place_market(self, symbol, qty, side, price=0.0) -> BracketResult:
        self._seq += 1
        oid = f"PM-{self._seq}"
        direction = 1 if side.upper() == "BUY" else -1
        # net into any existing position
        cur = self._positions.get(symbol, {"qty": 0.0, "entry": price})
        new_qty = cur["qty"] + qty * direction
        if abs(new_qty) < 1e-9:
            self._positions.pop(symbol, None)
        else:
            self._positions[symbol] = {"qty": new_qty, "entry": price or cur["entry"],
                                       "stop": 0.0, "target": 0.0}
        res = BracketResult(order_id=oid, symbol=symbol, side=side.upper(),
                            qty=qty, entry=price, stop=0.0, target=0.0,
                            status="filled", broker=self.name)
        self._orders.append(res)
        return res

    def cancel_all(self, symbol=None):
        if symbol:
            self._positions.pop(symbol, None)
        else:
            self._positions.clear()

    def flatten(self, symbol=None):
        self.cancel_all(symbol)

    @property
    def orders(self):
        return list(self._orders)


# --------------------------------------------------------------------------- #
# Interactive Brokers — real execution via ib_insync
# --------------------------------------------------------------------------- #
# Common API ports: 7497 TWS-paper, 7496 TWS-live, 4002 Gateway-paper, 4001 live
_LIVE_PORTS = {7496, 4001}


class IBKRBroker(BrokerAdapter):
    """
    Interactive Brokers adapter. Requires:
        pip install ib_insync
        a running TWS or IB Gateway with the API enabled.

    Defaults to the paper port 7497; connecting to a live port needs
    allow_live=True. Submits native bracket orders (parent + OCO stop + target).
    """
    name = "ibkr"

    def __init__(self, host="127.0.0.1", port=7497, client_id=1,
                 allow_live=False, exchange="SMART", currency="USD",
                 primary_exchange="", tif="GTC", outside_rth=True,
                 market_data_type=3):
        if port in _LIVE_PORTS and not allow_live:
            raise ValueError(
                f"Port {port} is a LIVE trading port. Pass allow_live=True to "
                "trade real money — or use the paper port 7497.")
        self.host, self.port, self.client_id = host, port, client_id
        self.exchange, self.currency = exchange, currency
        # 1 = real-time (needs a paid subscription), 3 = DELAYED (free, ~15 min),
        # 4 = delayed-frozen. Without a real-time subscription, paper orders sit
        # in PreSubmitted forever unless we tell IBKR to use delayed data.
        self.market_data_type = market_data_type
        # e.g. 'BM' (Bolsa de Madrid) for Spanish stocks with SMART routing
        self.primary_exchange = primary_exchange
        # Exit-leg time-in-force. If your IBKR account has an order preset that
        # forces DAY, GTC exits trigger warning 10349 and the bracket is
        # CANCELLED — set tif="DAY" to match the preset (or remove the preset in
        # Gateway to keep GTC stops that survive overnight).
        self.tif = tif
        self.outside_rth = outside_rth        # let orders rest before the open
        self.ib = None

    def connect(self):
        _ensure_event_loop()               # Python 3.13/3.14 compatibility
        try:
            from ib_insync import IB
        except ImportError as e:
            raise RuntimeError(
                "IBKRBroker needs `ib_insync` and a running TWS/IB Gateway:\n"
                "    pip install ib_insync\n"
                "Open TWS/Gateway, enable API (Configure > API > Settings), then "
                "connect to the paper port 7497.") from e
        except RuntimeError as e:
            raise RuntimeError(
                f"ib_insync failed to import ({e}). This usually means Python is "
                "too new for ib_insync (3.14). Use Python 3.11 or 3.12 in a venv:\n"
                "    py -3.12 -m venv .venv && .venv\\Scripts\\activate\n"
                "    pip install -r requirements.txt ib_insync") from e
        self.ib = IB()
        try:
            self.ib.connect(self.host, self.port, clientId=self.client_id, timeout=15)
        except (TimeoutError, Exception) as e:
            if isinstance(e, RuntimeError):
                raise
            raise RuntimeError(
                f"Could not connect to IBKR on {self.host}:{self.port} "
                f"(clientId={self.client_id}): {type(e).__name__}. Common causes:\n"
                "  1. Another script is already using this clientId — give each bot "
                "a different --ibkr-client-id (2, 3, ...), or close the other one.\n"
                "  2. Python 3.14 is not supported by ib_insync — use a 3.12 venv.\n"
                "  3. IB Gateway not fully logged in / API not enabled (port 4002)."
            ) from e
        # Use FREE delayed data if there's no real-time subscription, so paper
        # orders actually fill instead of hanging in PreSubmitted.
        try:
            self.ib.reqMarketDataType(self.market_data_type)
        except Exception:
            pass
        return self

    def disconnect(self):
        if self.ib is not None:
            self.ib.disconnect()

    def is_connected(self):
        return self.ib is not None and self.ib.isConnected()

    def sleep(self, seconds: float):
        self.ib.sleep(seconds)                 # pump the ib_insync event loop

    def account_label(self) -> str:
        try:
            return ",".join(self.ib.managedAccounts()) or self.name
        except Exception:
            return self.name

    def _contract(self, symbol):
        from ib_insync import Stock
        if self.primary_exchange:
            return Stock(symbol, self.exchange, self.currency,
                         primaryExchange=self.primary_exchange)
        return Stock(symbol, self.exchange, self.currency)

    def account(self) -> dict:
        vals = {v.tag: v.value for v in self.ib.accountSummary()}
        return {"cash": float(vals.get("TotalCashValue", 0) or 0),
                "equity": float(vals.get("NetLiquidation", 0) or 0)}

    def positions(self) -> dict:
        return {p.contract.symbol: {"qty": p.position, "entry": p.avgCost}
                for p in self.ib.positions()}

    def place_bracket(self, symbol, qty, side, entry, stop, target,
                      entry_type="MKT") -> BracketResult:
        """Submit a parent entry + OCO take-profit/stop-loss as one group."""
        from ib_insync import MarketOrder, LimitOrder, StopOrder
        contract = self._contract(symbol)
        self.ib.qualifyContracts(contract)
        # warm up a (delayed, free) price stream so the paper engine can fill
        try:
            self.ib.reqMktData(contract, "", False, False)
            self.ib.sleep(1.5)
        except Exception:
            pass
        action = side.upper()
        exit_action = "SELL" if action == "BUY" else "BUY"

        parent = (MarketOrder(action, qty) if entry_type == "MKT"
                  else LimitOrder(action, qty, entry))
        parent.transmit = False
        parent.outsideRth = self.outside_rth
        parent.tif = self.tif        # match the account preset (else 10349 cancels it)
        # exit-leg TIF (default GTC so a swing stop/target survive overnight). If
        # the account preset forces DAY, GTC triggers error 10349 + cancellation —
        # pass tif="DAY" to match the preset (see IBKRBroker docstring).
        tp = LimitOrder(exit_action, qty, target)
        tp.parentId = 0          # set after parent has an id
        tp.tif = self.tif
        tp.outsideRth = self.outside_rth
        tp.transmit = False
        sl = StopOrder(exit_action, qty, stop)
        sl.parentId = 0
        sl.tif = self.tif
        sl.outsideRth = self.outside_rth
        sl.transmit = True       # last leg transmits the whole group

        oca = f"oca-{symbol}-{int(entry*100)}"
        tp.ocaGroup = sl.ocaGroup = oca
        tp.ocaType = sl.ocaType = 1

        trade_parent = self.ib.placeOrder(contract, parent)
        pid = trade_parent.order.orderId
        tp.parentId = sl.parentId = pid
        self.ib.placeOrder(contract, tp)
        self.ib.placeOrder(contract, sl)

        # poll the real order status for a few seconds so callers see the truth
        # (Filled / Cancelled / PreSubmitted / Submitted) instead of guessing
        status = "Submitted"
        for _ in range(8):
            self.ib.sleep(0.5)
            status = trade_parent.orderStatus.status or status
            if status in ("Filled", "Cancelled", "ApiCancelled", "Inactive"):
                break
        return BracketResult(order_id=str(pid), symbol=symbol, side=action,
                             qty=qty, entry=entry, stop=stop, target=target,
                             status=status, broker=self.name)

    def place_market(self, symbol, qty, side, price=0.0) -> BracketResult:
        from ib_insync import MarketOrder
        contract = self._contract(symbol)
        self.ib.qualifyContracts(contract)
        order = MarketOrder(side.upper(), qty)
        order.tif = self.tif                 # match account preset (avoid 10349)
        order.outsideRth = self.outside_rth
        trade = self.ib.placeOrder(contract, order)
        return BracketResult(order_id=str(trade.order.orderId), symbol=symbol,
                             side=side.upper(), qty=qty, entry=price, stop=0.0,
                             target=0.0, status="submitted", broker=self.name)

    def cancel_all(self, symbol=None):
        self.ib.reqGlobalCancel()

    def flatten(self, symbol=None):
        for p in self.ib.positions():
            if symbol and p.contract.symbol != symbol:
                continue
            action = "SELL" if p.position > 0 else "BUY"
            from ib_insync import MarketOrder
            self.ib.placeOrder(p.contract, MarketOrder(action, abs(p.position)))


# --------------------------------------------------------------------------- #
# Webhook broker — the easiest live connector (routes orders over HTTP)
# --------------------------------------------------------------------------- #
class WebhookBroker(BrokerAdapter):
    """
    Route orders to any webhook that accepts a JSON alert — the simplest way to
    reach a **futures prop firm like Lucid** without a native API integration.

    The typical path is:  bot  ->  WebhookBroker (HTTP POST)  ->  TradersPost /
    CrossTrade  ->  Tradovate  ->  Lucid. You paste the bridge's webhook URL and
    the bot fires TradersPost-compatible payloads (buy / sell / exit with an
    optional stop-loss + take-profit that then live at the broker).

    IMPORTANT — this is a **one-way order router**: a plain webhook cannot report
    fills, positions or equity back. So this adapter *shadow-tracks* what it sent
    (like PaperBroker) purely to keep the bot loop, journal and funded engine
    running; the real account truth lives on the Lucid dashboard until you add a
    two-way API adapter (Tradovate). Stops/targets are enforced by the broker, not
    here, so the shadow book won't auto-close on a stop — treat its equity as an
    estimate. Uses only the standard library (urllib); no extra dependency.
    """
    name = "webhook"

    def __init__(self, webhook_url: str, capital: float = 50_000.0,
                 timeout: float = 10.0, dry_run: bool = False):
        if not webhook_url and not dry_run:
            raise ValueError("WebhookBroker needs a webhook_url (or dry_run=True).")
        self.url = webhook_url
        self.timeout = timeout
        self.dry_run = dry_run
        self._capital = capital
        self._positions: dict[str, dict] = {}
        self._connected = False
        self._seq = 0

    # ----- lifecycle ----- #
    def connect(self):
        self._connected = True
        return self

    def disconnect(self):
        self._connected = False

    def is_connected(self):
        return self._connected

    def account_label(self) -> str:
        return "webhook" + (" (dry-run)" if self.dry_run else "")

    # ----- readback (shadow, estimated) ----- #
    def account(self) -> dict:
        mkt = sum(p["qty"] * p["entry"] for p in self._positions.values())
        return {"cash": round(self._capital, 2), "positions_value": round(mkt, 2),
                "equity": round(self._capital + mkt, 2), "estimated": True}

    def positions(self) -> dict:
        return {s: {"qty": p["qty"], "entry": p["entry"]}
                for s, p in self._positions.items()}

    # ----- HTTP ----- #
    def _post(self, payload: dict) -> str:
        """POST the JSON alert; return a short status string (never raises)."""
        if self.dry_run:
            return "dry-run"
        import json
        import urllib.request
        data = json.dumps(payload).encode()
        req = urllib.request.Request(self.url, data=data,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return f"sent[{r.status}]"
        except Exception as e:                       # network / HTTP error
            return f"error[{type(e).__name__}]"

    # ----- orders ----- #
    def place_bracket(self, symbol, qty, side, entry, stop, target,
                      entry_type="MKT") -> BracketResult:
        self._seq += 1
        payload = {
            "ticker": symbol,
            "action": "buy" if side.upper() == "BUY" else "sell",
            "price": round(entry, 4),
            "quantity": qty,
            "takeProfit": {"limitPrice": round(target, 4)},
            "stopLoss": {"type": "stop", "stopPrice": round(stop, 4)},
        }
        status = self._post(payload)
        direction = 1 if side.upper() == "BUY" else -1
        self._positions[symbol] = {"qty": qty * direction, "entry": entry,
                                   "stop": stop, "target": target}
        return BracketResult(order_id=f"WH-{self._seq}", symbol=symbol,
                             side=side.upper(), qty=qty, entry=entry, stop=stop,
                             target=target, status=status, broker=self.name)

    def place_market(self, symbol, qty, side, price=0.0) -> BracketResult:
        self._seq += 1
        status = self._post({"ticker": symbol,
                             "action": "buy" if side.upper() == "BUY" else "sell",
                             "quantity": qty, "price": round(price, 4)})
        direction = 1 if side.upper() == "BUY" else -1
        cur = self._positions.get(symbol, {"qty": 0.0, "entry": price})
        new_qty = cur["qty"] + qty * direction
        if abs(new_qty) < 1e-9:
            self._positions.pop(symbol, None)
        else:
            self._positions[symbol] = {"qty": new_qty, "entry": price or cur["entry"],
                                       "stop": 0.0, "target": 0.0}
        return BracketResult(order_id=f"WH-{self._seq}", symbol=symbol,
                             side=side.upper(), qty=qty, entry=price, stop=0.0,
                             target=0.0, status=status, broker=self.name)

    def cancel_all(self, symbol=None):
        self.flatten(symbol)

    def flatten(self, symbol=None):
        syms = [symbol] if symbol else list(self._positions)
        for s in syms:
            self._post({"ticker": s, "action": "exit"})
            self._positions.pop(s, None)


# --------------------------------------------------------------------------- #
# MetaTrader 5 broker — two-way connector for FundedNext (and other MT5 firms)
# --------------------------------------------------------------------------- #
class MT5Broker(BrokerAdapter):
    """
    MetaTrader 5 adapter — the native, **two-way** path to a FundedNext (or any
    MT5) account: it places orders *and* reads real positions / balance / equity
    back, so the funded engine tracks the true account.

    Requires the official package and a running MT5 terminal logged into the
    challenge account:
        pip install MetaTrader5          # Windows (or Wine); the terminal must be installed

    Symbols are broker symbols (e.g. "EURUSD", "XAUUSD", "US30") and **volume is
    in lots** (0.01 step), not shares — use --fixed-qty (e.g. 0.10) since the
    equity/ATR share-sizer doesn't apply to lots. Stop-loss / take-profit are
    attached to the market order and enforced by the broker.

    Tests inject a fake `mt5` module; live code imports MetaTrader5 lazily.
    """
    name = "mt5"

    def __init__(self, login: int = 0, password: str = "", server: str = "",
                 path: str = "", magic: int = 555_000, deviation: int = 20, mt5=None):
        self.login_id, self.password, self.server = login, password, server
        self.path = path
        self.magic, self.deviation = magic, deviation
        self._mt5 = mt5           # optional injected module (tests)
        self._connected = False

    # ----- lifecycle ----- #
    def connect(self):
        m = self._mt5
        if m is None:
            try:
                import MetaTrader5 as m           # noqa: N813
            except ImportError as e:
                raise RuntimeError(
                    "MT5Broker needs the MetaTrader5 package and a running MT5 "
                    "terminal:\n    pip install MetaTrader5\n"
                    "Install MT5, log into your FundedNext account, then run again. "
                    "(MetaTrader5 is Windows-only; on Linux/Mac run under Wine.)") from e
        kw = {"path": self.path} if self.path else {}
        if not m.initialize(**kw):
            raise RuntimeError(f"MT5 initialize() failed: {m.last_error()}. Is the "
                               "terminal open and 'Algo Trading' enabled?")
        if self.login_id:
            if not m.login(self.login_id, password=self.password, server=self.server):
                raise RuntimeError(f"MT5 login failed for {self.login_id}@{self.server}: "
                                   f"{m.last_error()}")
        self._mt5 = m
        self._connected = True
        return self

    def disconnect(self):
        if self._mt5 is not None and self._connected:
            try:
                self._mt5.shutdown()
            except Exception:
                pass
        self._connected = False

    def is_connected(self):
        return self._connected

    def account_label(self) -> str:
        return f"{self.login_id}@{self.server}" if self.login_id else "mt5"

    # ----- market data (same feed as execution) ----- #
    def bars(self, symbol: str, interval: str = "5m", count: int = 800):
        """Recent OHLCV from MT5's own feed (FundedNext data) — no Yahoo needed."""
        import pandas as pd
        m = self._mt5
        tf = {
            "1m": m.TIMEFRAME_M1, "5m": m.TIMEFRAME_M5, "15m": m.TIMEFRAME_M15,
            "30m": m.TIMEFRAME_M30, "1h": m.TIMEFRAME_H1, "4h": m.TIMEFRAME_H4,
            "1d": m.TIMEFRAME_D1,
        }.get(interval.lower())
        if tf is None:
            raise ValueError(f"unsupported MT5 interval {interval!r}")
        if not (m.symbol_info(symbol) or (m.symbol_select(symbol, True) and None)):
            pass                                    # ensure the symbol is in Market Watch
        m.symbol_select(symbol, True)
        rates = m.copy_rates_from_pos(symbol, tf, 0, count)
        if rates is None or len(rates) == 0:
            return None
        df = pd.DataFrame(rates)
        df.index = pd.to_datetime(df["time"], unit="s")
        df.index.name = "date"
        vol = df["real_volume"] if "real_volume" in df and df["real_volume"].any() \
            else df.get("tick_volume", 0)
        return pd.DataFrame({"open": df["open"], "high": df["high"], "low": df["low"],
                             "close": df["close"], "volume": vol}, index=df.index)

    def size_for_risk(self, symbol, risk_amount, stop_dist, price=0.0):
        """
        Lots so a stop `stop_dist` away loses ~`risk_amount` (account currency).

        Uses the symbol's tick economics from MT5:
            loss for 1.0 lot = (stop_dist / trade_tick_size) * trade_tick_value
            lots = risk_amount / loss_per_lot   (rounded down to volume_step)
        clamped to [volume_min, volume_max]. This is the risk-based sizing the
        funded engine needs — every trade risks a fixed fraction of the real
        equity, whatever the instrument's tick value.
        """
        import math
        m = self._mt5
        info = m.symbol_info(symbol)
        if info is None or stop_dist <= 0:
            return None
        tick_val = getattr(info, "trade_tick_value", 0.0) or 0.0
        tick_size = getattr(info, "trade_tick_size", 0.0) or 0.0
        vmin = getattr(info, "volume_min", 0.01) or 0.01
        vmax = getattr(info, "volume_max", 100.0) or 100.0
        vstep = getattr(info, "volume_step", 0.01) or 0.01
        if tick_val <= 0 or tick_size <= 0:
            return vmin                              # can't size — smallest allowed
        loss_per_lot = (stop_dist / tick_size) * tick_val
        if loss_per_lot <= 0:
            return vmin
        lots = math.floor((risk_amount / loss_per_lot) / vstep) * vstep
        lots = max(vmin, min(vmax, lots))
        return round(lots, 8)

    # ----- readback (real, two-way) ----- #
    def account(self) -> dict:
        info = self._mt5.account_info()
        if info is None:
            return {"balance": 0.0, "equity": 0.0}
        return {"balance": float(info.balance), "equity": float(info.equity),
                "currency": getattr(info, "currency", "")}

    def positions(self) -> dict:
        m = self._mt5
        out: dict[str, dict] = {}
        for p in (m.positions_get() or []):
            signed = p.volume if p.type == m.POSITION_TYPE_BUY else -p.volume
            if p.symbol in out:
                out[p.symbol]["qty"] += signed
            else:
                out[p.symbol] = {"qty": signed, "entry": p.price_open}
        return out

    # ----- orders ----- #
    def _filling(self, symbol):
        return self._mt5.ORDER_FILLING_IOC

    def place_bracket(self, symbol, qty, side, entry, stop, target,
                      entry_type="MKT") -> BracketResult:
        m = self._mt5
        info = m.symbol_info(symbol)
        if info is None:
            return BracketResult(order_id="", symbol=symbol, side=side.upper(), qty=qty,
                                 entry=entry, stop=stop, target=target,
                                 status="error[unknown symbol]", broker=self.name)
        if not getattr(info, "visible", True):
            m.symbol_select(symbol, True)
        tick = m.symbol_info_tick(symbol)
        is_buy = side.upper() == "BUY"
        price = (tick.ask if is_buy else tick.bid) if tick else entry
        request = {
            "action": m.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(qty),
            "type": m.ORDER_TYPE_BUY if is_buy else m.ORDER_TYPE_SELL,
            "price": price,
            "sl": float(stop),
            "tp": float(target),
            "deviation": self.deviation,
            "magic": self.magic,
            "comment": "qflow-bot",
            "type_time": m.ORDER_TIME_GTC,
            "type_filling": self._filling(symbol),
        }
        res = m.order_send(request)
        ok = res is not None and res.retcode == m.TRADE_RETCODE_DONE
        status = "filled" if ok else f"error[{getattr(res, 'retcode', '?')}]"
        return BracketResult(order_id=str(getattr(res, "order", "") or ""), symbol=symbol,
                             side=side.upper(), qty=qty, entry=price, stop=stop,
                             target=target, status=status, broker=self.name)

    def place_market(self, symbol, qty, side, price=0.0) -> BracketResult:
        # a bracket with SL/TP disabled (0.0 = MT5 treats as none)
        return self.place_bracket(symbol, qty, side, entry=price, stop=0.0, target=0.0)

    def cancel_all(self, symbol=None):
        m = self._mt5
        for o in (m.orders_get() or []):           # pending orders
            if symbol and o.symbol != symbol:
                continue
            m.order_send({"action": m.TRADE_ACTION_REMOVE, "order": o.ticket})
        self.flatten(symbol)

    def flatten(self, symbol=None):
        m = self._mt5
        for p in (m.positions_get() or []):
            if symbol and p.symbol != symbol:
                continue
            is_buy = p.type == m.POSITION_TYPE_BUY
            tick = m.symbol_info_tick(p.symbol)
            price = (tick.bid if is_buy else tick.ask) if tick else p.price_open
            m.order_send({
                "action": m.TRADE_ACTION_DEAL,
                "symbol": p.symbol,
                "volume": p.volume,
                "type": m.ORDER_TYPE_SELL if is_buy else m.ORDER_TYPE_BUY,
                "position": p.ticket,
                "price": price,
                "deviation": self.deviation,
                "magic": self.magic,
                "comment": "qflow-flat",
                "type_time": m.ORDER_TIME_GTC,
                "type_filling": self._filling(p.symbol),
            })
