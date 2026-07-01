# IB Gateway — start-to-first-trade checklist

> **Educational use only — not financial advice.** Start on a **paper** account,
> keep the kill-switches on, size minimally. See [`DISCLAIMER.md`](DISCLAIMER.md).

How the pieces fit:

```
your bot (this repo)  →  IB Gateway (same PC, port 4002)  →  IBKR servers  →  market
```

IB Gateway is the lightweight bridge (no charts) that stays logged in and relays
your bot's orders to Interactive Brokers. It must be **running and logged in**
whenever the bot trades.

## 1. Accounts
1. Create an Interactive Brokers account (or a free **paper-only** account from
   their website).
2. In Client Portal → **Settings → Account Settings → Paper Trading Account**,
   enable it. You'll get separate paper login credentials.

## 2. Install IB Gateway
- Download **IB Gateway** (the "stable" build) from the IBKR website and install
  it on the computer that will run the bot.
- **Use Python 3.11 or 3.12** for the bot — `ib_insync` does **not** support
  Python 3.14 (it fails to import with a `RuntimeError` about the event loop).
  Create a dedicated environment:
  ```powershell
  py -3.12 -m venv .venv
  .venv\Scripts\activate
  pip install -r requirements.txt ib_insync
  ```

## 3. Enable the API in IB Gateway
1. Launch IB Gateway → log in choosing **Paper Trading** mode.
2. **Configure → Settings → API → Settings**:
   - ☑️ **Enable ActiveX and Socket Clients**
   - ⬜ **Read-Only API** — leave **unchecked** (so it can place orders)
   - **Socket port = 4002**  (IB Gateway paper)
   - **Trusted IPs**: add `127.0.0.1`
3. Apply / OK. Leave IB Gateway running.

> Port cheat-sheet: **IB Gateway paper 4002** · TWS paper 7497 · Gateway live
> 4001 · TWS live 7496. The bot defaults to 4002 and refuses live ports unless
> you pass `--ibkr-allow-live`.

## 4. Pull the repo and run
```bash
git clone -b claude/trading-strategy-framework-9jndzj https://github.com/ruky00/ruky00.git
cd ruky00/trading-strategy-framework
pip install -r requirements.txt ib_insync

# forward-test through IB Gateway paper (port 4002), kill-switches on:
python examples/paper_trade.py --symbol AAPL --source yahoo \
    --strategy mean_reversion --risk 0.005 \
    --kill-switches --max-drawdown 0.08 \
    --broker ibkr --ibkr-port 4002 --news rss --step
```
The first time, IB Gateway may pop up an **"Accept incoming connection"** — accept it.

## 5. Watch it work
When the bot opens a position you'll see it, in real time:
- in **IB Gateway / Client Portal** (orders, positions) — entry + stop + target;
- in the bot's **`data/paper/<symbol>_<strategy>/journal.csv`** — `BROKER` rows
  carry the IBKR order id, so you can reconcile the two every day.

## 6. Run it daily (cron)
```cron
# weekdays 22:10 (after the US close, Europe time). IB Gateway must be running.
10 22 * * 1-5  cd /path/to/trading-strategy-framework && \
  /usr/bin/python3 examples/paper_trade.py --symbol AAPL --source yahoo \
  --strategy mean_reversion --risk 0.005 --kill-switches \
  --broker ibkr --ibkr-port 4002 --news rss --step \
  >> data/paper/ibkr_cron.log 2>&1
```

## Gotchas to know
- **Daily restart:** IB Gateway logs out once a day for maintenance (~midnight)
  and needs to log back in. For unattended 24/7 running, add **IBC**
  (IBController) to auto-login/restart — a later step.
- **PC must stay on:** if IB Gateway is closed, the bot can't open new trades.
  But any **stop-loss / take-profit already placed stays live on IBKR's servers**
  (that's the point of bracket orders).
- **Market data:** if you see "delayed" prices, subscribe to the relevant
  market-data package in Client Portal (a few €/month), or enable delayed data.
- **Reconcile daily:** compare IB Gateway fills vs `journal.csv`. Any mismatch →
  stop and investigate before the next session.

## When to go live
Only after weeks on paper with the readiness gate at **GO**. Then:
`--ibkr-port 4001 --ibkr-allow-live`, start at `--risk 0.0025` and minimum size,
kill-switches on.
