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
_TEMPLATES: tuple[Template, ...] = (
    Template("double_bottom", (1.00, 0.62, 0.80, 0.60, 1.00), Direction.LONG),
    Template("double_top",    (1.00, 1.38, 1.20, 1.40, 1.00), Direction.SHORT),
    Template("v_reversal",    (1.00, 0.70, 0.45, 0.70, 1.00), Direction.LONG),
    Template("spike_top",     (1.00, 1.30, 1.55, 1.30, 1.00), Direction.SHORT),
    Template("bull_flag",     (0.55, 0.85, 1.05, 1.00, 0.97), Direction.LONG),
    Template("head_shoulders", (1.00, 1.18, 1.02, 1.32, 1.02, 1.18, 0.92), Direction.SHORT),
    Template("ascending",     (0.80, 0.86, 0.92, 0.97, 1.03), Direction.LONG),
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
