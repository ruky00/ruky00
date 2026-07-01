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
    def cancel_all(self, symbol=None): ...
    def flatten(self, symbol=None): ...


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
                 allow_live=False, exchange="SMART", currency="USD"):
        if port in _LIVE_PORTS and not allow_live:
            raise ValueError(
                f"Port {port} is a LIVE trading port. Pass allow_live=True to "
                "trade real money — or use the paper port 7497.")
        self.host, self.port, self.client_id = host, port, client_id
        self.exchange, self.currency = exchange, currency
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
        self.ib.connect(self.host, self.port, clientId=self.client_id)
        return self

    def disconnect(self):
        if self.ib is not None:
            self.ib.disconnect()

    def is_connected(self):
        return self.ib is not None and self.ib.isConnected()

    def _contract(self, symbol):
        from ib_insync import Stock
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
        action = side.upper()
        exit_action = "SELL" if action == "BUY" else "BUY"

        parent = (MarketOrder(action, qty) if entry_type == "MKT"
                  else LimitOrder(action, qty, entry))
        parent.transmit = False
        tp = LimitOrder(exit_action, qty, target)
        tp.parentId = 0          # set after parent has an id
        tp.transmit = False
        sl = StopOrder(exit_action, qty, stop)
        sl.parentId = 0
        sl.transmit = True       # last leg transmits the whole group

        oca = f"oca-{symbol}-{int(entry*100)}"
        tp.ocaGroup = sl.ocaGroup = oca
        tp.ocaType = sl.ocaType = 1

        trade_parent = self.ib.placeOrder(contract, parent)
        pid = trade_parent.order.orderId
        tp.parentId = sl.parentId = pid
        self.ib.placeOrder(contract, tp)
        self.ib.placeOrder(contract, sl)

        return BracketResult(order_id=str(pid), symbol=symbol, side=action,
                             qty=qty, entry=entry, stop=stop, target=target,
                             status="submitted", broker=self.name)

    def cancel_all(self, symbol=None):
        self.ib.reqGlobalCancel()

    def flatten(self, symbol=None):
        for p in self.ib.positions():
            if symbol and p.contract.symbol != symbol:
                continue
            action = "SELL" if p.position > 0 else "BUY"
            from ib_insync import MarketOrder
            self.ib.placeOrder(p.contract, MarketOrder(action, abs(p.position)))
