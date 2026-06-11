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
