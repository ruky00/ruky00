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
