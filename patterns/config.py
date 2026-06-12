"""Configuration loading and identity hashing.

The config hash is the unit of the multiple-testing ledger: every distinct
combination of IDENTITY_FIELDS ever backtested counts toward the Bonferroni
correction. Execution plumbing (sizing, paths, seeds) is deliberately excluded
so that re-running the same hypothesis with different plumbing does not inflate N.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields, replace
from pathlib import Path

import yaml

# Fields that define a hypothesis. Changing any of these is a new config in the ledger.
IDENTITY_FIELDS = (
    "symbols",
    "timeframe",
    "window",
    "horizon",
    "k",
    "dedup_gap",
    "p_threshold",
    "t_multiplier",
    "min_matches",
    "min_history_bars",
    "features",
    "normalization",
    "enable_shorts",
    "cost_bps",
    "split_date",
    "query_stride",
)


@dataclass(frozen=True)
class Config:
    symbols: tuple[str, ...] = ("QQQ",)
    timeframe: str = "1Min"
    window: int = 30
    horizon: int = 15
    k: int = 50
    dedup_gap: int = 15
    p_threshold: float = 0.65
    t_multiplier: float = 1.5
    min_matches: int = 20
    min_history_bars: int = 35000
    features: str = "close"
    normalization: str = "logret_zscore"
    enable_shorts: bool = False
    cost_bps: float = 5.0
    split_date: str = "2022-12-31"
    query_stride: int = 1

    # Plumbing — excluded from identity.
    position_size: float = 0.10
    force_flat_minutes_before_close: int = 5
    seed: int = 42
    db_path: str = "db/patterns.db"
    reports_dir: str = "reports"
    block_size: int = 1024
    candidate_chunk: int = 65536

    def identity_dict(self) -> dict:
        d = {}
        for f in IDENTITY_FIELDS:
            v = getattr(self, f)
            d[f] = list(v) if isinstance(v, tuple) else v
        return d

    @property
    def config_hash(self) -> str:
        canonical = json.dumps(self.identity_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]

    def identity_json(self) -> str:
        return json.dumps(self.identity_dict(), sort_keys=True)


_FIELD_TYPES = {f.name: f.type for f in fields(Config)}


def _coerce(name: str, value):
    if name == "symbols":
        if isinstance(value, str):
            value = [value]
        return tuple(str(s).upper() for s in value)
    default = getattr(Config, name)
    if isinstance(default, bool):
        if isinstance(value, str):
            return value.lower() in ("1", "true", "yes", "on")
        return bool(value)
    if isinstance(default, int) and not isinstance(default, bool):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return str(value)


def load_config(path: str | Path | None = "config.yaml", overrides: dict | None = None) -> Config:
    """Load config.yaml (if present) and apply --set style overrides on top."""
    raw: dict = {}
    if path is not None and Path(path).exists():
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
    raw.update(overrides or {})
    unknown = set(raw) - set(_FIELD_TYPES)
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    coerced = {k: _coerce(k, v) for k, v in raw.items()}
    return Config(**coerced)


def parse_set_overrides(pairs: list[str]) -> dict:
    """Parse repeated --set key=value CLI flags."""
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--set expects key=value, got: {pair!r}")
        key, value = pair.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def with_overrides(cfg: Config, **kwargs) -> Config:
    return replace(cfg, **{k: _coerce(k, v) for k, v in kwargs.items()})
