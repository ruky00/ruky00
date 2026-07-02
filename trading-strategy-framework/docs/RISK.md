# Risk Governor — kill-switches before you go live

> **Educational use only — not financial advice.** See [`DISCLAIMER.md`](DISCLAIMER.md).

Per-trade sizing (the 1%-risk rule) protects any single trade. The **risk
governor** (`qflow/risk_governor.py`) protects the **account** by halting new
risk when things go wrong at the portfolio level. These are the circuit breakers
you want *on* before any real money is involved.

| Kill-switch | Default | What it does |
|-------------|---------|--------------|
| `max_daily_loss` | 3% | No new entries once today's loss hits the limit |
| `max_drawdown` | 15% | **Hard halt** if equity falls this far below its peak |
| `max_portfolio_heat` | 6% | Cap total open risk (sum of per-trade $risk ÷ equity) |
| `max_consecutive_losses` | 5 | Trigger a cooldown after a losing streak |
| `cooldown_days` | 2 | Length of that streak cooldown |

When a switch fires, the engine **skips the entry and journals a `risk_halt`
row** with the reason, so every blocked trade is auditable. The governor's state
(peak equity, streak, halt latch) **persists across daily runs**.

## Use it

```bash
# enable with defaults
python examples/live/paper_trade.py --strategy trend_following --symbol TSLA \
    --kill-switches --step

# tune the limits
python examples/live/paper_trade.py --strategy trend_following --symbol TSLA \
    --kill-switches --max-drawdown 0.08 --max-daily-loss 0.02 --max-heat 0.05 --step
```
```python
from qflow.paper import PaperTrader
pt = PaperTrader("TSLA", strategy="trend_following",
                 risk_limits={"max_drawdown": 0.08, "max_daily_loss": 0.02,
                              "max_portfolio_heat": 0.05})
pt.step()
print(pt.report())   # shows a "Risk governor" line: active / cooldown / HALTED
```

Standalone (for a multi-symbol portfolio runner):

```python
from qflow.risk_governor import RiskGovernor
g = RiskGovernor({"max_drawdown": 0.10})
g.start_day("2026-06-30", equity)
ok, why = g.can_open(proposed_risk=100, equity=equity)
if ok:
    g.on_open(100)          # ... place trade ...
    g.on_close(pnl, 100)    # on exit
g.observe(equity)           # once per bar; latches the drawdown halt
```

## What to expect

On the bundled TSLA trend test with a tight `max_drawdown=0.05`, the governor
**latches a halt the moment the equity curve crosses −5%** and blocks every
subsequent entry — capping the account instead of letting the drawdown run. That
is the whole point: it trades worse returns for survivability. Set the limits to
levels you could actually stomach in real life, then confirm in paper that they
fire when you expect.

> A halt is a *floor on disaster*, not a profit engine. If the governor halts
> often, the strategy itself is too risky for the limit — fix the strategy or
> the size, don't just loosen the switch.
