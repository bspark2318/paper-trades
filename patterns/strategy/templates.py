"""Hand-drawn folklore shapes, and how to put them in the matcher's space.

Each template is a sketch of an idealized *close path* — a handful of relative
price levels at evenly-spaced control points. `template_vector` stretches that
sketch to exactly W bars (`np.interp`), turns it into log-returns, and z-scores
it through the SAME normalizer real windows use. Template and live window then
live in one space, so plain Euclidean distance between them is meaningful.

These shapes are deliberately naive guesses at the folklore; whether any of them
actually predicts a forward move is the referee's job, never asserted here. A
template carries the direction the folklore claims it precedes — double bottom →
up, double top → down — and the signal layer resolves SHORT to NO_TRADE when
shorts are disabled.

Templates are close-path sketches only: there is no honest way to hand-draw a
candle's wick, so the template source refuses features='ohlc' upstream.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from jaxtyping import Float

from patterns.engine.normalize import normalize
from patterns.strategy.base import Direction


@dataclass(frozen=True)
class Template:
    name: str
    anchors: tuple[float, ...]   # relative close levels at evenly-spaced control points
    direction: Direction         # forward move the folklore claims this shape precedes


# The library. Anchor numbers are idealized folklore sketches, not fitted values —
# the validation layer decides whether they pay, and rejection is the expected result.
#
# Shape selection guided by Bulkowski's measured failure-rank ordering (thepatternsite.com):
# the classics traders treat as highest-signal that also render cleanly as a CLOSE path.
# Bulkowski's rates are daily-bar / 5%-breakout over months and do NOT transfer to
# 1-min/time-stop — they pick the shapes; our referee re-measures from scratch.
#
# Deliberately omitted, because a single close path cannot carry what defines them:
#  - symmetric triangle: breaks either way, so it has no single direction to emit;
#  - rectangles: a flat consolidation window is near-zero-variance, so z-scoring amplifies
#    noise and the shape matches everything;
#  - wedges (rising/falling) and bear flag: their bull/bear direction lives in the high/low
#    ENVELOPE (sloping boundary lines), not the close line — as close paths a falling wedge
#    is indistinguishable from a descending triangle, and a bear flag from a bottom's recovery,
#    so they collide with OPPOSITE-direction patterns (<1.5 apart) and the trade direction
#    becomes a coin flip. Same reason wicks need OHLC; these belong to k-NN(ohlc), not here.
_TEMPLATES: tuple[Template, ...] = (
    # --- reversals: bottoms (bullish) ---
    Template("double_bottom",          (1.00, 0.62, 0.80, 0.60, 1.00), Direction.LONG),
    Template("triple_bottom",          (1.00, 0.60, 0.85, 0.60, 0.85, 0.60, 1.00), Direction.LONG),
    Template("inverse_head_shoulders", (1.00, 0.82, 0.98, 0.68, 0.98, 0.82, 1.08), Direction.LONG),
    Template("rounding_bottom",        (1.00, 0.70, 0.50, 0.42, 0.50, 0.70, 1.00), Direction.LONG),
    Template("cup_with_handle",        (1.00, 0.62, 0.46, 0.52, 0.74, 0.96, 1.00, 0.90, 0.97), Direction.LONG),
    Template("v_reversal",             (1.00, 0.70, 0.45, 0.70, 1.00), Direction.LONG),
    # --- reversals: tops (bearish) ---
    Template("double_top",             (1.00, 1.38, 1.20, 1.40, 1.00), Direction.SHORT),
    Template("triple_top",             (1.00, 1.40, 1.15, 1.40, 1.15, 1.40, 1.00), Direction.SHORT),
    Template("head_shoulders",         (1.00, 1.18, 1.02, 1.32, 1.02, 1.18, 0.92), Direction.SHORT),
    Template("spike_top",              (1.00, 1.30, 1.55, 1.30, 1.00), Direction.SHORT),
    # --- continuations (bullish) ---
    Template("bull_flag",              (0.55, 0.85, 1.05, 1.00, 0.97), Direction.LONG),
    Template("high_tight_flag",        (0.30, 0.68, 1.00, 1.05, 1.02, 1.05), Direction.LONG),
    Template("ascending_triangle",     (0.50, 1.00, 0.72, 1.00, 0.86, 1.00, 0.95), Direction.LONG),
    Template("ascending",             (0.80, 0.86, 0.92, 0.97, 1.03), Direction.LONG),
    # --- continuations (bearish) ---
    Template("descending_triangle",    (1.00, 0.50, 0.84, 0.50, 0.70, 0.50, 0.60), Direction.SHORT),
)

TEMPLATES: dict[str, Template] = {t.name: t for t in _TEMPLATES}


def load_templates(names: tuple[str, ...]) -> list[Template]:
    """Resolve enabled pattern names to Templates; unknown names fail loudly."""
    if not names:
        raise ValueError("template source needs at least one pattern in template_patterns")
    unknown = [n for n in names if n not in TEMPLATES]
    if unknown:
        raise ValueError(f"Unknown template patterns {unknown}; available: {sorted(TEMPLATES)}")
    return [TEMPLATES[n] for n in names]


def template_vector(anchors: tuple[float, ...], window: int,
                    normalization: str) -> Float[np.ndarray, " W"]:
    """Sketch -> W-dim z-scored shape vector, in the matcher's window space.

    Interpolate the anchor levels to W+1 closes, take the W consecutive
    log-returns, z-score them exactly as build_windows does. The result is
    directly comparable to a normalized WindowSet row (features='close').
    """
    levels = np.interp(
        np.linspace(0.0, 1.0, window + 1),
        np.linspace(0.0, 1.0, len(anchors)),
        np.asarray(anchors, dtype=np.float64),
    )
    logret = np.log(levels[1:] / levels[:-1])[None, :]   # (1, W)
    Z, valid = normalize(normalization, logret)
    if not bool(valid[0]):
        raise ValueError("degenerate template: a flat sketch has no shape to z-score")
    vector: Float[np.ndarray, " W"] = Z[0]
    return vector
