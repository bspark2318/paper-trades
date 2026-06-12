"""SignalSource: the plugin boundary between pattern definitions and the referee.

Everything downstream (walk-forward, baselines, ledger, evaluate gate, live
loop) consumes this protocol and never knows which engine produced a signal.
A source declares the config fields it reads; those + the shared fields form
its ledger identity. Registration asserts the declaration matches the
canonical table in patterns.config so the two cannot drift silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Protocol, runtime_checkable

import pandas as pd

from patterns.config import SOURCE_IDENTITY_FIELDS, Config

LONG = "LONG"
SHORT = "SHORT"
NO_TRADE = "NO_TRADE"


@dataclass(frozen=True)
class Signal:
    asof: pd.Timestamp
    symbol: str
    direction: str                      # LONG | SHORT | NO_TRADE
    diagnostics: dict = field(default_factory=dict)


@runtime_checkable
class SignalSource(Protocol):
    name: ClassVar[str]
    identity_fields: ClassVar[tuple[str, ...]]

    def prepare(self, bars: pd.DataFrame) -> None:
        """One-time precomputation over history (e.g. window matrix)."""
        ...

    def signal_at(self, asof: pd.Timestamp) -> Signal:
        """Decision using only information available at `asof`."""
        ...


SOURCES: dict[str, type] = {}


def register_source(cls: type) -> type:
    declared = SOURCE_IDENTITY_FIELDS.get(cls.name)
    if declared is None:
        raise KeyError(f"Source {cls.name!r} missing from config.SOURCE_IDENTITY_FIELDS")
    if tuple(cls.identity_fields) != declared:
        raise ValueError(
            f"Source {cls.name!r} identity fields {cls.identity_fields} "
            f"!= config declaration {declared}"
        )
    SOURCES[cls.name] = cls
    return cls


def make_source(cfg: Config) -> SignalSource:
    if cfg.signal_source not in SOURCES:
        raise KeyError(f"Unknown signal_source {cfg.signal_source!r}; available: {sorted(SOURCES)}")
    return SOURCES[cfg.signal_source](cfg)
