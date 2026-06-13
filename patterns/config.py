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
from typing import Any

import yaml

# A hypothesis = shared experiment conditions + the active signal source's own knobs.
# Fields a source does NOT read are excluded from its hash: tweaking `k` must not
# mint a new ledger entry for a source that never looks at `k`.
SHARED_IDENTITY_FIELDS = (
    "symbols",
    "timeframe",
    "horizon",
    "min_history_bars",
    "enable_shorts",
    "cost_bps",
    "split_date",
    "query_stride",
    "signal_source",
)

# Canonical per-source identity declarations. Signal source classes in
# patterns/strategy declare the same tuple and registration asserts they match —
# config stays import-cycle-free while drift fails loudly.
SOURCE_IDENTITY_FIELDS: dict[str, tuple[str, ...]] = {
    "knn_shape": (
        "window",
        "k",
        "dedup_gap",
        "p_threshold",
        "t_multiplier",
        "min_matches",
        "features",
        "normalization",
    ),
    "template": (
        "window",
        "normalization",
        "template_patterns",
        "template_threshold",
    ),
    "candles": (
        "candle_patterns",
        "candle_trend_lookback",
    ),
}


def identity_fields_for(source: str) -> tuple[str, ...]:
    if source not in SOURCE_IDENTITY_FIELDS:
        raise KeyError(f"Unknown signal_source {source!r}; available: {sorted(SOURCE_IDENTITY_FIELDS)}")
    return SHARED_IDENTITY_FIELDS + SOURCE_IDENTITY_FIELDS[source]


@dataclass(frozen=True)
class Config:
    signal_source: str = "knn_shape"
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
    # template source knobs (ignored by knn_shape; see SOURCE_IDENTITY_FIELDS)
    template_patterns: tuple[str, ...] = (
        "double_bottom", "triple_bottom", "inverse_head_shoulders", "rounding_bottom",
        "cup_with_handle", "v_reversal", "double_top", "triple_top", "head_shoulders",
        "spike_top", "bull_flag", "high_tight_flag", "ascending_triangle", "falling_wedge",
        "ascending", "bear_flag", "descending_triangle", "rising_wedge",
    )
    template_threshold: float = 3.5
    # candles source knobs (ignored by other sources; see SOURCE_IDENTITY_FIELDS)
    candle_patterns: tuple[str, ...] = (
        "hammer", "shooting_star", "bullish_engulfing", "bearish_engulfing",
        "piercing_line", "dark_cloud_cover", "morning_star", "evening_star",
        "three_white_soldiers", "three_black_crows",
    )
    candle_trend_lookback: int = 10     # bars of preceding trend a reversal needs; 0 = pure anatomy
    enable_shorts: bool = False
    cost_bps: float = 5.0
    split_date: str = "2022-12-31"
    query_stride: int = 1

    # Plumbing — excluded from identity.
    position_size: float = 0.05
    force_flat_minutes_before_close: int = 5
    seed: int = 42
    db_path: str = "db/patterns.db"
    reports_dir: str = "reports"
    block_size: int = 1024
    candidate_chunk: int = 65536

    def identity_dict(self) -> dict:
        d = {}
        for f in identity_fields_for(self.signal_source):
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


def _coerce(name: str, value: Any) -> Any:
    if name == "symbols":
        if isinstance(value, str):
            value = [value]
        return tuple(str(s).upper() for s in value)
    if name in ("template_patterns", "candle_patterns"):
        if isinstance(value, str):
            value = [value]
        return tuple(str(s) for s in value)
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


def with_overrides(cfg: Config, **kwargs: Any) -> Config:
    return replace(cfg, **{k: _coerce(k, v) for k, v in kwargs.items()})
