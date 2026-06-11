"""CLI entry points. Commands are added as their modules are built."""

from __future__ import annotations

import typer

from patterns.config import load_config, parse_set_overrides

app = typer.Typer(no_args_is_help=True, add_completion=False)
data_app = typer.Typer(no_args_is_help=True)
app.add_typer(data_app, name="data", help="Bar acquisition and storage")


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
