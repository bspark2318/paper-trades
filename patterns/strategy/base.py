"""SignalSource: the plugin boundary between pattern definitions and the referee.

Everything downstream (walk-forward, baselines, ledger, evaluate gate, live
loop) consumes this protocol and never knows which engine produced a signal.
A source declares the config fields it reads; those + the shared fields form
its ledger identity. Registration asserts the declaration matches the
canonical table in patterns.config so the two cannot drift silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Callable, ClassVar, Protocol, TypedDict, runtime_checkable

import pandas as pd

from patterns.config import SOURCE_IDENTITY_FIELDS, Config


class Direction(StrEnum):
    LONG = "LONG"
    SHORT = "SHORT"
    NO_TRADE = "NO_TRADE"


# Aliases so call sites read `LONG`, not `Direction.LONG`.
LONG = Direction.LONG
SHORT = Direction.SHORT
NO_TRADE = Direction.NO_TRADE


class Diagnostics(TypedDict, total=False):
    """Generic evidence behind a signal — only what every source can promise.
    Each source extends this with its own typed fields (see KnnDiagnostics)."""

    reason: str             # why this direction, e.g. "rule" | "too_few_matches"


@dataclass(frozen=True)
class Signal:
    asof: pd.Timestamp
    symbol: str
    direction: Direction
    diagnostics: Diagnostics = field(default_factory=lambda: Diagnostics())


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


# Values are constructors: Config -> SignalSource. (Plain `type` for the decorator
# arg — Protocol classes can't be instantiated via type[Protocol] under mypy.)
SOURCES: dict[str, Callable[[Config], SignalSource]] = {}


def register_source(cls: type) -> type:
    name: str = getattr(cls, "name")
    identity_fields: tuple[str, ...] = tuple(getattr(cls, "identity_fields"))
    declared = SOURCE_IDENTITY_FIELDS.get(name)
    if declared is None:
        raise KeyError(f"Source {name!r} missing from config.SOURCE_IDENTITY_FIELDS")
    if identity_fields != declared:
        raise ValueError(
            f"Source {name!r} identity fields {identity_fields} "
            f"!= config declaration {declared}"
        )
    SOURCES[name] = cls
    return cls


def make_source(cfg: Config) -> SignalSource:
    if cfg.signal_source not in SOURCES:
        raise KeyError(f"Unknown signal_source {cfg.signal_source!r}; available: {sorted(SOURCES)}")
    return SOURCES[cfg.signal_source](cfg)
