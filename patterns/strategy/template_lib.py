"""template: the chart-pattern-folklore hypothesis as a SignalSource.

Rule: at each query bar, take the normalized current window and measure its
Euclidean distance to every enabled template vector. If the nearest template is
within `template_threshold`, emit that template's direction; else NO_TRADE.

Unlike knn_shape this source consults no history to decide — the templates are
fixed shapes, so the only data it reads is the current window (bars up to asof).
No-lookahead is therefore automatic. Whether the shapes pay is left entirely to
the same referee (walk-forward, TOD baseline, ledger, evaluate gate).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from jaxtyping import Float

from patterns.config import Config
from patterns.engine.windows import WindowSet, build_windows
from patterns.strategy.base import LONG, NO_TRADE, SHORT, Diagnostics, Direction, Signal, register_source
from patterns.strategy.templates import Template, load_templates, template_vector


class TemplateDiagnostics(Diagnostics, total=False):
    """Template-specific evidence: which shape was nearest and how near."""

    pattern: str            # nearest enabled template
    distance: float         # Euclidean distance to it, in z-scored window space
    threshold: float        # bar the distance had to clear


@register_source
class TemplateLib:
    name = "template"
    identity_fields = (
        "window",
        "normalization",
        "template_patterns",
        "template_threshold",
    )

    def __init__(self, cfg: Config):
        if cfg.features != "close":
            raise ValueError(
                "template source supports features='close' only: templates sketch a "
                f"close path and cannot draw candle wicks; got features={cfg.features!r}"
            )
        self.cfg = cfg
        self.symbol = cfg.symbols[0]
        self.ws: WindowSet | None = None
        # Template vectors are pure config (no bars) — build them once, up front.
        self.templates: list[Template] = load_templates(cfg.template_patterns)
        self.vectors: Float[np.ndarray, "P W"] = np.stack(
            [template_vector(t.anchors, cfg.window, cfg.normalization) for t in self.templates]
        )

    def prepare(self, bars: pd.DataFrame) -> None:
        # build_windows gives the normalized current-window lookup (Z[q_row]); the
        # forward returns it also computes are unused — templates don't consult them.
        self.ws = build_windows(
            bars,
            window=self.cfg.window,
            horizon=self.cfg.horizon,
            normalization=self.cfg.normalization,
            features="close",
        )

    def signal_at(self, asof: pd.Timestamp) -> Signal:
        cfg = self.cfg
        ws = self.ws
        if ws is None:
            raise RuntimeError("signal_at() before prepare()")

        q_row = ws.row_for_ts(asof)
        if not ws.valid[q_row]:
            return self._signal(asof, NO_TRADE, reason="degenerate_window")

        diff = self.vectors - ws.Z[q_row]
        dist = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        best = int(np.argmin(dist))
        best_dist = float(dist[best])
        tmpl = self.templates[best]

        if best_dist <= cfg.template_threshold:
            direction = tmpl.direction
            if direction is SHORT and not cfg.enable_shorts:
                direction = NO_TRADE
            reason = "rule"
        else:
            direction = NO_TRADE
            reason = "too_far"

        return self._signal(
            asof, direction, reason=reason,
            pattern=tmpl.name, distance=best_dist, threshold=cfg.template_threshold,
        )

    def _signal(self, asof: pd.Timestamp, direction: Direction, **extra: object) -> Signal:
        diagnostics: TemplateDiagnostics = {**extra}  # type: ignore[typeddict-item]
        return Signal(
            asof=pd.Timestamp(asof),
            symbol=self.symbol,
            direction=direction,
            diagnostics=diagnostics,
        )
