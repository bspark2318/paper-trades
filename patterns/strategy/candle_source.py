"""candles: the candlestick-anatomy folklore family as a SignalSource.

At each query bar it checks the last 1-3 bars against each enabled candlestick
detector (the pattern's full length). A detector firing is necessary but not
sufficient: candlestick patterns are reversal signals, so the folklore only
counts them in context — a hammer means nothing mid-rally. `candle_trend_lookback`
bars before the pattern must show the setup move (down before a bullish pattern,
up before a bearish one); set it to 0 to test pure anatomy with no context.

Multi-bar patterns and the lookback must lie within one session — no candle
spans the overnight gap. Direction resolution mirrors the template source: if the
fired patterns disagree, abstain; shorts resolve to NO_TRADE unless enabled.

No-lookahead is automatic — every bar inspected is at or before asof.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from patterns.config import Config
from patterns.strategy.base import LONG, NO_TRADE, SHORT, Diagnostics, Direction, Signal, register_source
from patterns.strategy.candlesticks import Candle, CandlePattern, load_candles

NY = "America/New_York"


@dataclass
class _Series:
    ts: np.ndarray          # UTC-naive datetime64, for asof lookup
    o: np.ndarray
    h: np.ndarray
    low: np.ndarray
    c: np.ndarray
    sess: np.ndarray        # NY session date per bar


class CandleDiagnostics(Diagnostics, total=False):
    """Which candlestick(s) fired at this bar."""

    pattern: str                # headline (first fired)
    matched: tuple[str, ...]    # all that fired (and passed the context gate)
    n_matched: int


@register_source
class CandleSource:
    name = "candles"
    identity_fields = ("candle_patterns", "candle_trend_lookback")

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.symbol = cfg.symbols[0]
        self.patterns: list[CandlePattern] = load_candles(cfg.candle_patterns)
        self.lookback = cfg.candle_trend_lookback
        self._s: _Series | None = None

    def prepare(self, bars: pd.DataFrame) -> None:
        self._s = _Series(
            ts=bars["ts"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(),
            o=bars["open"].to_numpy(dtype=np.float64),
            h=bars["high"].to_numpy(dtype=np.float64),
            low=bars["low"].to_numpy(dtype=np.float64),
            c=bars["close"].to_numpy(dtype=np.float64),
            sess=bars["ts"].dt.tz_convert(NY).dt.date.to_numpy(),
        )

    def signal_at(self, asof: pd.Timestamp) -> Signal:
        s = self._s
        if s is None:
            raise RuntimeError("signal_at() before prepare()")
        idx = self._row(s, asof)
        no_match = self._signal(asof, NO_TRADE, reason="no_pattern", matched=(), n_matched=0)
        if idx < 0:
            return no_match

        fired: list[tuple[str, Direction]] = []
        for p in self.patterns:
            lo = idx - p.n_bars + 1
            if lo < 0 or s.sess[lo] != s.sess[idx]:
                continue
            candles = [Candle(s.o[j], s.h[j], s.low[j], s.c[j]) for j in range(lo, idx + 1)]
            if p.detect(candles) and self._trend_ok(s, lo, idx, p.direction):
                fired.append((p.name, p.direction))

        if not fired:
            return no_match
        matched = tuple(name for name, _ in fired)
        dirs = {d for _, d in fired}
        base: dict[str, object] = dict(pattern=matched[0], matched=matched, n_matched=len(matched))

        if LONG in dirs and SHORT in dirs:
            return self._signal(asof, NO_TRADE, reason="conflict", **base)
        direction = LONG if LONG in dirs else SHORT
        if direction is SHORT and not self.cfg.enable_shorts:
            return self._signal(asof, NO_TRADE, reason="shorts_disabled", **base)
        return self._signal(asof, direction, reason="rule", **base)

    # ---- internals ----

    def _row(self, s: _Series, asof: pd.Timestamp) -> int:
        asof = pd.Timestamp(asof)
        if asof.tzinfo is not None:
            asof = asof.tz_convert("UTC").tz_localize(None)
        return int(np.searchsorted(s.ts, np.datetime64(asof), side="right") - 1)

    def _trend_ok(self, s: _Series, lo: int, idx: int, direction: Direction) -> bool:
        """The lookback bars before the pattern must show the setup move, in-session:
        down before a bullish reversal, up before a bearish one. lookback=0 disables."""
        if self.lookback <= 0:
            return True
        start = lo - self.lookback
        if start < 0 or s.sess[start] != s.sess[idx]:
            return False        # context not available within this session
        before, ref = float(s.c[lo - 1]), float(s.c[start])
        return before < ref if direction is LONG else before > ref

    def _signal(self, asof: pd.Timestamp, direction: Direction, **extra: object) -> Signal:
        diagnostics: CandleDiagnostics = {**extra}  # type: ignore[typeddict-item]
        return Signal(
            asof=pd.Timestamp(asof),
            symbol=self.symbol,
            direction=direction,
            diagnostics=diagnostics,
        )
