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
