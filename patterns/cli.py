"""CLI entry points. Commands are added as their modules are built."""

from __future__ import annotations

import typer

from patterns.config import load_config, parse_set_overrides

app = typer.Typer(no_args_is_help=True, add_completion=False)
data_app = typer.Typer(no_args_is_help=True)
app.add_typer(data_app, name="data", help="Bar acquisition and storage")


@data_app.command("refresh")
def data_refresh(
    config_path: str = typer.Option("config.yaml", "--config"),
) -> None:
    """Incrementally sync minute bars from Alpaca for all configured symbols."""
    from patterns import db as dbm
    from patterns.data import store

    cfg = load_config(config_path)
    conn = dbm.connect(cfg.db_path)
    for symbol in cfg.symbols:
        result = store.refresh(conn, symbol)
        typer.echo(
            f"{symbol}: fetched {result['fetched']} bars, stored {result['stored_rth']} RTH bars, "
            f"{result['sessions']} sessions"
        )


@data_app.command("status")
def data_status(
    config_path: str = typer.Option("config.yaml", "--config"),
) -> None:
    """Show bar coverage per symbol."""
    from patterns import db as dbm
    from patterns.data import store

    cfg = load_config(config_path)
    conn = dbm.connect(cfg.db_path)
    for symbol in cfg.symbols:
        cov = store.coverage(conn, symbol)
        if cov["bars"] == 0:
            typer.echo(f"{symbol}: no bars stored — run `patterns data refresh`")
            continue
        typer.echo(
            f"{symbol}: {cov['bars']:,} bars across {cov['sessions']:,} sessions "
            f"({cov['first_date']} → {cov['last_date']}), "
            f"{cov['short_or_gappy_sessions']} short/gappy sessions"
        )


@app.command()
def match(
    asof: str = typer.Option(..., "--asof", help="Window end, e.g. '2026-06-10 14:30' (NY time if no tz)"),
    symbol: str = typer.Option("", "--symbol", help="Defaults to first configured symbol"),
    set_: list[str] = typer.Option([], "--set", help="Override key=value"),
    config_path: str = typer.Option("config.yaml", "--config"),
) -> None:
    """What happened after the k most similar historical windows?"""
    import pandas as pd

    from patterns import db as dbm
    from patterns import plotting
    from patterns.data import store
    from patterns.engine import matcher
    from patterns.engine.windows import build_windows

    cfg = load_config(config_path, parse_set_overrides(set_))
    sym = (symbol or cfg.symbols[0]).upper()
    conn = dbm.connect(cfg.db_path)
    typer.echo(dbm.report_banner(conn))

    bars = store.load_bars(conn, sym)
    if bars.empty:
        typer.echo(f"{sym}: no bars stored — run `patterns data refresh`", err=True)
        raise typer.Exit(1)

    ts = pd.Timestamp(asof)
    if ts.tzinfo is None:
        ts = ts.tz_localize("America/New_York")

    ws = build_windows(bars, cfg.window, cfg.horizon,
                       normalization=cfg.normalization, features=cfg.features)
    out = matcher.query(ws, ts, k=cfg.k, dedup_gap=cfg.dedup_gap)

    s = out.stats()
    typer.echo(f"\n{sym} @ {out.query_ts:%Y-%m-%d %H:%M} UTC — "
               f"window {cfg.window}m, horizon {cfg.horizon}m, features {cfg.features}")
    typer.echo(f"eligible candidates: {out.n_candidates:,} | kept after dedup/top-k: {out.n}")
    if out.n == 0:
        typer.echo("no matches — not enough prior history at this timestamp")
        raise typer.Exit(0)
    typer.echo(f"fwd return: mean {s['mean']:+.4%} | median {s['median']:+.4%} | "
               f"positive {s['pct_positive']:.0%}\n")

    typer.echo("top matches (best first):")
    for ts_m, dist, fwd in list(zip(out.match_ts, out.distance, out.fwd_ret))[:10]:
        typer.echo(f"  {ts_m:%Y-%m-%d %H:%M}  dist {dist:7.3f}  fwd {fwd:+.4%}")

    stamp = f"{sym}_{out.query_ts:%Y%m%d_%H%M}"
    p1 = plotting.plot_match_overlay(ws, out, f"{cfg.reports_dir}/match_{stamp}.png")
    p2 = plotting.plot_fwd_histogram(out, f"{cfg.reports_dir}/match_{stamp}_hist.png")
    typer.echo(f"\nsaved: {p1}\nsaved: {p2}")


@app.command()
def backtest(
    set_: list[str] = typer.Option([], "--set", help="Override key=value"),
    config_path: str = typer.Option("config.yaml", "--config"),
    resamples: int = typer.Option(1000, "--resamples", help="Random-baseline resamples"),
) -> None:
    """Walk-forward backtest on TRAIN data only. Registers in the ledger first —
    a crashed run still counts toward N."""
    from patterns import db as dbm
    from patterns import plotting
    from patterns.data import store
    from patterns.validate import baselines, ledger
    from patterns.validate.walkforward import run_walkforward
    from patterns.strategy.base import Direction

    cfg = load_config(config_path, parse_set_overrides(set_))
    sym = cfg.symbols[0]
    conn = dbm.connect(cfg.db_path)
    typer.echo(dbm.report_banner(conn))

    bars = store.load_bars(conn, sym, end_ts=f"{cfg.split_date}T23:59:59Z")
    if bars.empty:
        typer.echo(f"{sym}: no train bars (split {cfg.split_date}) — run `patterns data refresh`", err=True)
        raise typer.Exit(1)

    # On the record BEFORE any computation.
    dbm.register_config(conn, cfg.config_hash, cfg.identity_json())
    run_id = dbm.start_run(conn, "backtest", cfg.config_hash, seed=cfg.seed)

    try:
        res = run_walkforward(cfg, bars)
        m = dict(res.metrics)
        decisions = [s.asof for s in res.signals if s.direction is Direction.LONG]
        if decisions:
            rb = baselines.random_baseline(
                bars, decisions, m["mean_net_ret"], cfg.horizon, cfg.cost_bps,
                cfg.min_history_bars, n_resamples=resamples, seed=cfg.seed,
            )
            m["p_random"] = rb.p_value
        bh = baselines.buy_and_hold(bars, cfg.cost_bps)
        m["bh_total_return"] = bh.total_return
        m["bh_sharpe"] = bh.sharpe
    except BaseException:
        dbm.finish_run(conn, run_id, status="crashed")
        raise
    ledger.save_trades(conn, run_id, sym, res.trades)
    dbm.finish_run(conn, run_id, status="ok", metrics=m)

    n = dbm.n_configs_tried(conn)
    a = ledger.alpha(n)
    png = plotting.plot_equity(res.equity_ts, res.equity,
                               f"{cfg.reports_dir}/equity_{cfg.config_hash}.png",
                               title=f"walk-forward equity — {cfg.config_hash} (train)")

    typer.echo(f"\nconfig {cfg.config_hash} | TRAIN ≤ {cfg.split_date} | "
               f"{m['n_sessions']} sessions | run #{run_id}")
    typer.echo(f"signals: {m['n_signals']:,} queried, {m['n_long_signals']} LONG")
    typer.echo(f"trades:  {m['n_trades']} | hit rate {m['hit_rate']:.0%} | "
               f"mean net/trade {m['mean_net_ret']:+.4%}" if m["n_trades"]
               else "trades:  0 — nothing cleared the rule")
    if m["n_trades"]:
        typer.echo(f"total return {m['total_return']:+.2%} | sharpe {m['sharpe']:.2f} | "
                   f"max DD {m['max_drawdown']:.2%}")
        typer.echo(f"vs random (TOD-matched, {resamples} resamples): p = {m.get('p_random', float('nan')):.4f}")
    typer.echo(f"vs buy-and-hold: {m['bh_total_return']:+.2%} (sharpe {m['bh_sharpe']:.2f})")
    typer.echo(f"\nledger: N = {n} configs tried → survivor needs p < {a:.2e}")
    typer.echo(f"NOTE: paper fills are optimistic — read all numbers as upper bounds.")
    typer.echo(f"saved: {png}")


@app.command()
def evaluate(
    config_hash_arg: str = typer.Argument(..., metavar="CONFIG_HASH", help="Hash from `patterns ledger`"),
    yes: bool = typer.Option(False, "--yes", help="Skip the retype confirmation"),
    resamples: int = typer.Option(1000, "--resamples"),
    config_path: str = typer.Option("config.yaml", "--config"),
) -> None:
    """Run THE one test-set evaluation for a config. Spends a permanent
    evaluation counter row before computing — crashes count."""
    import json

    import pandas as pd

    from patterns import db as dbm
    from patterns.config import with_overrides
    from patterns.data import store
    from patterns.strategy.base import Direction
    from patterns.validate import baselines, evaluate as ev, ledger as ledger_mod
    from patterns.validate.walkforward import run_walkforward

    base_cfg = load_config(config_path)
    conn = dbm.connect(base_cfg.db_path)
    typer.echo(dbm.report_banner(conn))

    row = conn.execute(
        "SELECT config_json FROM configs WHERE config_hash = ?", (config_hash_arg,)
    ).fetchone()
    if row is None:
        typer.echo(f"unknown config hash {config_hash_arg!r} — backtest it first", err=True)
        raise typer.Exit(1)
    cfg = with_overrides(base_cfg, **json.loads(row["config_json"]))
    if cfg.config_hash != config_hash_arg:
        typer.echo("stored identity does not reproduce this hash — config drift?", err=True)
        raise typer.Exit(1)

    train_metrics = ev.latest_train_metrics(conn, cfg.config_hash)
    if train_metrics is None:
        typer.echo("no completed train backtest for this hash — run `patterns backtest` first", err=True)
        raise typer.Exit(1)

    sym = cfg.symbols[0]
    bars = store.load_bars(conn, sym)
    test_mask = bars["ts"] > pd.Timestamp(f"{cfg.split_date}T23:59:59Z")
    if not test_mask.any():
        typer.echo(f"no bars after split {cfg.split_date} — nothing to evaluate", err=True)
        raise typer.Exit(1)
    first_test_idx = int(test_mask.idxmax())

    n_before = dbm.n_test_evaluations(conn)
    typer.echo(f"\nThis permanently spends test evaluation #{n_before + 1} for {cfg.config_hash}.")
    if not yes:
        typed = typer.prompt("Retype the config hash to proceed")
        if typed.strip() != cfg.config_hash:
            typer.echo("hash mismatch — aborted, nothing spent")
            raise typer.Exit(1)

    # ---- the gate: spend first, compute second ----
    run_id = dbm.start_run(conn, "evaluate", cfg.config_hash, seed=cfg.seed)
    eval_id = ev.spend_evaluation(conn, cfg.config_hash, run_id)

    try:
        res = run_walkforward(cfg, bars, min_query_idx=first_test_idx)
        m = dict(res.metrics)
        decisions = [s.asof for s in res.signals if s.direction is Direction.LONG]
        p_random = None
        if decisions:
            rb = baselines.random_baseline(
                bars, decisions, m["mean_net_ret"], cfg.horizon, cfg.cost_bps,
                first_test_idx, n_resamples=resamples, seed=cfg.seed,
            )
            p_random = rb.p_value
            m["p_random"] = p_random
    except BaseException:
        dbm.finish_run(conn, run_id, status="crashed")
        raise  # the spent CRASHED row stays

    a = ledger_mod.alpha(dbm.n_configs_tried(conn))
    verdict = ev.decide(train_metrics, m, p_random, a)
    ev.record_verdict(conn, eval_id, verdict, m)
    dbm.finish_run(conn, run_id, status="ok", metrics=m)

    typer.echo(f"\n{'=' * 60}\n  VERDICT: {verdict.label}\n{'=' * 60}")
    if verdict.reasons:
        typer.echo("failed criteria:")
        for r in verdict.reasons:
            typer.echo(f"  - {r}")
    typer.echo(f"\n{'':14}{'train':>12}{'test':>12}")
    for key in ("n_trades", "mean_net_ret", "hit_rate", "sharpe"):
        tr, te = train_metrics.get(key), m.get(key)
        fmt = (lambda v: f"{v:+.4%}" if key == "mean_net_ret" else
               f"{v:.2%}" if key == "hit_rate" else
               f"{v:.2f}" if key == "sharpe" else str(v))
        typer.echo(f"{key:14}{fmt(tr) if tr is not None else '—':>12}"
                   f"{fmt(te) if te is not None else '—':>12}")
    if p_random is not None:
        typer.echo(f"\ntest p_random = {p_random:.4f} vs corrected alpha = {a:.2e}")
    typer.echo("NOTE: paper fills are optimistic — read all numbers as upper bounds.")
    typer.echo(dbm.report_banner(conn))


@app.command()
def ledger(
    config_path: str = typer.Option("config.yaml", "--config"),
) -> None:
    """Every hypothesis ever tried, and what the bar for belief now is."""
    from patterns import db as dbm
    from patterns.validate import ledger as ledger_mod

    cfg = load_config(config_path)
    conn = dbm.connect(cfg.db_path)
    typer.echo(dbm.report_banner(conn))

    rows = ledger_mod.ledger_rows(conn)
    if not rows:
        typer.echo("ledger empty — no backtests recorded yet")
        return
    typer.echo(f"\n{'hash':12}  {'runs':>4}  {'trades':>6}  {'mean net':>9}  {'p_random':>8}  candidate")
    for r in rows:
        mean = f"{r.mean_net_ret:+.4%}" if r.mean_net_ret is not None else "—"
        p = f"{r.p_random:.4f}" if r.p_random is not None else "—"
        trades = str(r.n_trades) if r.n_trades is not None else "—"
        typer.echo(f"{r.config_hash:12}  {r.n_runs:>4}  {trades:>6}  {mean:>9}  {p:>8}  "
                   f"{'YES' if r.candidate else 'no'}")


@app.command()
def config(
    set_: list[str] = typer.Option([], "--set", help="Override key=value"),
    config_path: str = typer.Option("config.yaml", "--config"),
) -> None:
    """Show the resolved config and its identity hash."""
    cfg = load_config(config_path, parse_set_overrides(set_))
    for key, value in cfg.identity_dict().items():
        typer.echo(f"{key:>18}: {value}")
    typer.echo(f"{'config_hash':>18}: {cfg.config_hash}")


if __name__ == "__main__":
    app()
