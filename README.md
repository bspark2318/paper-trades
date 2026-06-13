# patterns

Empirically test whether short-term **intraday chart patterns** predict the next
few minutes of returns on **1-minute QQQ bars**, behind a deliberately
conservative validation layer — then paper-trade only the rules that survive.

**The validation layer is the product.** Most configurations you try will be
**REJECTED**, and that is the system working: it is built to make it hard to fool
yourself, not to manufacture a strategy. A rule has to beat a time-of-day-matched
random baseline at a Bonferroni-corrected threshold, stay consistent between train
and test, and clear a one-shot test gate that is spent the instant you peek — only
then is it allowed to place (paper) orders.

> Paper fills at intraday speed are optimistic. Every number this system prints is
> an **upper bound**. Default costs are a pessimistic 5 bps/side on purpose.

---

## What's inside

Three pluggable **signal sources**, one referee:

| source | hypothesis | reads |
| --- | --- | --- |
| `knn_shape` | the next move rhymes with the k most similar historical shapes | close path, or full OHLC |
| `template` | named multi-bar chart shapes (double bottom, bull flag, …) | the close *path* |
| `candles` | candlestick anatomy (hammer, engulfing, …) in trend context | the *bars themselves* |

Each declares the config fields it reads; those plus the shared experiment fields
form its **identity hash**, the unit of the multiple-testing ledger. Tweaking a
knob a source doesn't read does **not** mint a new hypothesis.

---

## Setup

Requires a recent Python (built and tested on Homebrew **Python 3.14**). `alpaca-py`
wheels also work on 3.12 if 3.14 lags.

```bash
/opt/homebrew/opt/python@3.14/bin/python3.14 -m venv .venv
source .venv/bin/activate
pip install -e .            # installs the `patterns` CLI
pip install pytest mypy     # dev
```

Minute-bar history needs an authenticated Alpaca data API — a **free paper
account** is enough. Keys come from the environment only; nothing is read from
disk or committed.

```bash
cp .env.example .env        # then edit:
export ALPACA_API_KEY=...
export ALPACA_SECRET_KEY=...
```

The system is **paper-only by construction** — the broker adapter refuses any
non-paper endpoint and there is no code path to live trading.

---

## The pipeline

Everything is one CLI. Config lives in `config.yaml`; override any field inline
with `--set key=value`.

```bash
# 1. cache full-history 1-minute bars (incremental, idempotent)
patterns data refresh
patterns data status

# 2. look at one moment: what happened after the k most similar past shapes?
patterns match --asof "2026-06-10 14:30"          # writes overlay + histogram PNGs

# 3. walk-forward backtest on TRAIN data only (auto-records in the ledger;
#    a crashed run still counts toward the multiple-testing N)
patterns backtest
patterns backtest --set signal_source=candles --set candle_trend_lookback=0

# 4. see every hypothesis ever tried and the current bar for belief
patterns ledger

# 5. spend THE one test-set evaluation for a promising config.
#    This is irreversible — the counter row is committed before any compute,
#    so even a crash spends it. You retype the hash to confirm.
patterns evaluate <config_hash>

# 6. paper-trade a SURVIVOR during market hours (refuses non-survivors)
patterns trade

# 7. check the account, and digest the week
patterns status
patterns report weekly
```

### What "survivor" means

`patterns evaluate` writes a verdict. A config **SURVIVED** only if, on the
untouched test set, it: makes money per trade, beats the random baseline at
`0.05 / N`, and stays consistent with train (test mean ≥ ¼ train mean, Sharpe same
sign). `patterns trade` arms **only** for a SURVIVED config — the gate that guards
the test set also guards the trigger finger.

---

## Running the live loop on a schedule (macOS)

The loop trades during regular hours and is safe to kill and restart at any
instant: it re-reads positions and open orders from the broker each minute and
believes them, so a restart never double-enters or orphans a position. Sizing is
5% of live equity, one position at a time; exits are a time-stop at `horizon` bars
and a hard force-flat 5 minutes before the close (never overnight).

`~/Library/LaunchAgents/com.patterns.trade.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key>            <string>com.patterns.trade</string>
  <key>WorkingDirectory</key> <string>/path/to/paper-trades</string>
  <key>ProgramArguments</key>
  <array>
    <string>/path/to/paper-trades/.venv/bin/patterns</string>
    <string>trade</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>ALPACA_API_KEY</key>    <string>...</string>
    <key>ALPACA_SECRET_KEY</key> <string>...</string>
  </dict>
  <!-- start a few minutes before the open; the loop sleeps until RTH and exits
       cleanly after the close. Re-launched each weekday morning. -->
  <key>StartCalendarInterval</key>
  <dict><key>Hour</key><integer>9</integer><key>Minute</key><integer>25</integer></dict>
  <key>StandardOutPath</key>  <string>/tmp/patterns-trade.log</string>
  <key>StandardErrorPath</key><string>/tmp/patterns-trade.err</string>
</dict></plist>
```

```bash
launchctl load ~/Library/LaunchAgents/com.patterns.trade.plist
```

The loop self-gates to RTH, so the schedule only needs to start it each morning.
**cron** alternative (loop sleeps until the open):

```
25 9 * * 1-5  cd /path/to/paper-trades && .venv/bin/patterns trade >> /tmp/patterns-trade.log 2>&1
```

---

## Reproduce from the database

Everything is keyed by **config hash**. `db/patterns.db` holds the bars, every
backtest run, the ledger, the (one-shot) test evaluations, and the live order /
snapshot journal. Given the db you can replay any `backtest`/`evaluate` and get the
same verdict; the strategy itself is deterministic and all baseline randomness is
seeded (`seed` in `config.yaml`).

---

## Honesty rules baked in

- **No lookahead**, proven twice: a matcher unit test and a walk-forward
  perturbation property test (changing bars after time *t* leaves every signal at
  ≤ *t* bit-identical). Windows and horizons never cross the overnight gap.
- **The test set is sacred.** One evaluation per config, spent before compute,
  counted forever. Peeking is paid for in advance.
- **No LLM sentiment for historical dates, ever** — the model knows what happened
  next; that is invisible lookahead and is banned outright. (A forward-only
  sentiment recorder is a deferred, post-Phase-3 idea.)
- **Paper-fill caveat** on every report.

## Development

```bash
pytest          # full suite
mypy patterns   # strict
```
