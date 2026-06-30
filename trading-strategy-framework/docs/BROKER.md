# Broker Execution — from simulation to Interactive Brokers

> **Educational use only — not financial advice.** Live trading risks real money.
> Start on a paper account, size minimally, and keep the kill-switches on. See
> [`DISCLAIMER.md`](DISCLAIMER.md).

The framework decides *what* to trade; a **broker adapter** decides *where the
order goes*. Same engine, same kill-switches, same news overlay — only the
execution layer changes.

| Adapter | What it is | Needs |
|---------|-----------|-------|
| `PaperBroker` | in-memory simulation (default) | nothing |
| `IBKRBroker` | Interactive Brokers, native bracket orders | `ib_insync` + running TWS/IB Gateway |

## How it plugs in

```python
from qflow.paper import PaperTrader
from qflow import broker

ib = broker.IBKRBroker(port=7497)          # 7497 = PAPER (default), 7496 = LIVE
pt = PaperTrader("AAPL", source="yahoo", strategy="mean_reversion",
                 risk_limits={"max_drawdown": 0.08}, broker=ib)
pt.step()        # the ONLY path that sends real orders
```

The broker is consulted **only on the live `step()` path**. `replay()` (backtest)
**never** sends an order — a hard safety boundary, enforced and tested.

## Stop-loss & take-profit (bracket orders)

Every entry is submitted as a **bracket**: parent entry + a stop-loss + a
take-profit, linked as **OCO** (one-cancels-other). With IBKR these exit legs
live on the broker's servers, so:

> Your stop-loss is honoured even if your script crashes, your internet drops,
> or your machine is off. If the take-profit fills, the stop auto-cancels.

The engine already computes `entry`, `stop` (2×ATR) and `target` (4×ATR) under
the 1%-risk rule; the adapter just translates them into the bracket.

## Setup (Interactive Brokers)

1. `pip install ib_insync`
2. Open **TWS** or **IB Gateway**, log in to your **paper** account.
3. *Configure → API → Settings*: enable *ActiveX and Socket Clients*; set the
   socket port to **7497** (paper).
4. `python examples/live_ibkr.py`

### Safety rails
- `IBKRBroker` defaults to the paper port 7497. The live ports (7496/4001)
  **raise unless you pass `allow_live=True`** — you cannot go live by accident.
- Keep `risk_limits` (kill-switches) enabled; they gate entries before any order.
- Start at `risk_per_trade=0.005` (0.5%) and minimum size.

## What to expect

- **Execution levels:** signals-only → semi-auto → full-auto. Don't jump to
  full-auto on real money; spend weeks on the paper account first.
- **Slippage / partial fills:** your fill won't exactly match the backtest;
  liquid names are close, thin names aren't.
- **Market data:** IBKR charges small monthly fees for real-time data; without
  it you get delayed quotes.
- **Reconcile daily:** compare IBKR's fills against the bot's `journal.csv`
  (`BROKER` rows record each order id). Any drift → stop and investigate.
- **Account rules:** frequent intraday/short trading may trigger PDT or borrow
  constraints depending on your account/jurisdiction.

## Other brokers

The same `BrokerAdapter` interface fits any venue. To add one (e.g. **Alpaca**
for US equities, or **ccxt** for crypto), subclass `BrokerAdapter` and implement
`connect`, `account`, `positions`, `place_bracket`, `flatten`. The engine doesn't
change.
