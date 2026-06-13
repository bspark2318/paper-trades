import dataclasses

import numpy as np
import pandas as pd
import pytest

from patterns.config import Config
from patterns.strategy.base import Direction, make_source
from patterns.strategy.templates import TEMPLATES, load_templates, template_vector
from tests.conftest import make_session_bars, rebuild_ohlc_from_closes

W = 10


def tmpl_cfg(**over) -> Config:
    base = Config(
        signal_source="template", window=W, horizon=3, features="close",
        template_patterns=tuple(TEMPLATES), template_threshold=3.5,
        min_history_bars=0, cost_bps=0.0,
    )
    return dataclasses.replace(base, **over)


def plant_template_at_tail(bars: pd.DataFrame, name: str) -> pd.DataFrame:
    """Overwrite the last W+1 closes with a template's exact interpolated level
    path (rescaled to sit near the prior price), then rederive OHLC. The final
    window then reproduces the template's z-scored shape to within float error."""
    from patterns.strategy.templates import TEMPLATES as LIB

    anchors = LIB[name].anchors
    levels = np.interp(np.linspace(0, 1, W + 1), np.linspace(0, 1, len(anchors)),
                       np.asarray(anchors, dtype=np.float64))
    out = bars.copy()
    close = out["close"].to_numpy().copy()
    base = close[-(W + 2)]                      # continue from the bar before the planted run
    close[-(W + 1):] = base * levels / levels[0]
    out["close"] = close
    return rebuild_ohlc_from_closes(out)


# ---------- template_vector geometry ----------

def test_template_vector_is_unit_variance_shape():
    v = template_vector(TEMPLATES["double_bottom"].anchors, W, "logret_zscore")
    assert v.shape == (W,)
    # z-scored W-vector: sum of squares == W (unit variance), mean ~ 0
    assert float(np.sum(v ** 2)) == pytest.approx(W, rel=1e-4)
    assert float(np.mean(v)) == pytest.approx(0.0, abs=1e-5)


def test_distinct_templates_are_distinguishable():
    db = template_vector(TEMPLATES["double_bottom"].anchors, W, "logret_zscore")
    dt = template_vector(TEMPLATES["double_top"].anchors, W, "logret_zscore")
    assert np.linalg.norm(db - dt) > 1.0      # bottom and top are not the same shape


@pytest.mark.parametrize("name", list(TEMPLATES))
def test_every_template_is_a_valid_unit_shape(name):
    v = template_vector(TEMPLATES[name].anchors, W, "logret_zscore")
    assert v.shape == (W,)
    assert float(np.sum(v ** 2)) == pytest.approx(W, rel=1e-4)


@pytest.mark.parametrize("name", list(TEMPLATES))
def test_every_template_is_its_own_nearest(name):
    """Plant each shape; it must be the nearest match (distance ~0). Proves no two
    library shapes are accidentally identical. Direction wiring is covered by the
    targeted long/short/conflict tests below — a few shape pairs (wedge vs triangle)
    are close enough that planting one legitimately triggers a conflict abstain."""
    bars = make_session_bars("2024-03-04", n_bars=30)
    bars = plant_template_at_tail(bars, name)
    src = make_source(tmpl_cfg(enable_shorts=True, template_threshold=0.5))
    src.prepare(bars)
    sig = src.signal_at(pd.Timestamp(bars["ts"].iloc[-1]))
    assert sig.diagnostics["pattern"] == name
    assert sig.diagnostics["distance"] < 1e-3


def test_conflicting_templates_abstain():
    """A close path that matches BOTH a long and a short shape must NO_TRADE, not
    coin-flip on which is marginally nearer. Falling wedge and descending triangle
    sit 0.80 apart with opposite directions; a loose-enough threshold catches both."""
    bars = make_session_bars("2024-03-04", n_bars=30)
    bars = plant_template_at_tail(bars, "falling_wedge")
    src = make_source(tmpl_cfg(enable_shorts=True, template_threshold=1.0))
    src.prepare(bars)
    sig = src.signal_at(pd.Timestamp(bars["ts"].iloc[-1]))
    assert sig.direction is Direction.NO_TRADE
    assert sig.diagnostics["reason"] == "conflict"
    assert sig.diagnostics["n_matched"] >= 2
    assert {"falling_wedge", "descending_triangle"} <= set(sig.diagnostics["matched"])


# ---------- signal_at ----------

def test_exact_double_bottom_fires_long():
    bars = make_session_bars("2024-03-04", n_bars=30)
    bars = plant_template_at_tail(bars, "double_bottom")
    src = make_source(tmpl_cfg(template_threshold=0.5))
    src.prepare(bars)
    sig = src.signal_at(pd.Timestamp(bars["ts"].iloc[-1]))
    assert sig.direction is Direction.LONG
    assert sig.diagnostics["pattern"] == "double_bottom"
    assert sig.diagnostics["distance"] < 1e-3          # planted shape ~ exact


def test_noise_window_does_not_fire():
    # random walk tail → far from every idealized template at a tight threshold
    bars = make_session_bars("2024-03-04", n_bars=30, seed=7)
    src = make_source(tmpl_cfg(template_threshold=0.5))
    src.prepare(bars)
    sig = src.signal_at(pd.Timestamp(bars["ts"].iloc[-1]))
    assert sig.direction is Direction.NO_TRADE
    assert sig.diagnostics["reason"] == "too_far"


def test_bearish_template_needs_shorts_enabled():
    bars = make_session_bars("2024-03-04", n_bars=30)
    bars = plant_template_at_tail(bars, "double_top")
    asof = pd.Timestamp(bars["ts"].iloc[-1])

    flat = make_source(tmpl_cfg(enable_shorts=False, template_threshold=0.5))
    flat.prepare(bars)
    s1 = flat.signal_at(asof)
    assert s1.direction is Direction.NO_TRADE
    assert s1.diagnostics["reason"] == "shorts_disabled"
    assert s1.diagnostics["pattern"] == "double_top"   # recognized, just not actionable

    shorting = make_source(tmpl_cfg(enable_shorts=True, template_threshold=0.5))
    shorting.prepare(bars)
    s2 = shorting.signal_at(asof)
    assert s2.direction is Direction.SHORT


# ---------- construction guards ----------

def test_ohlc_features_rejected():
    with pytest.raises(ValueError, match="features='close' only"):
        make_source(tmpl_cfg(features="ohlc"))


def test_unknown_pattern_rejected():
    with pytest.raises(ValueError, match="Unknown template patterns"):
        load_templates(("not_a_real_pattern",))


def test_empty_pattern_list_rejected():
    with pytest.raises(ValueError, match="at least one pattern"):
        load_templates(())


# ---------- ledger identity ----------

def test_template_threshold_is_an_identity_field():
    a = tmpl_cfg(template_threshold=3.0)
    b = tmpl_cfg(template_threshold=4.0)
    assert a.config_hash != b.config_hash


def test_knn_knobs_do_not_affect_template_hash():
    # k / dedup_gap / p_threshold belong to knn_shape; under signal_source=template
    # they must not mint a new ledger entry.
    a = tmpl_cfg(k=50)
    b = tmpl_cfg(k=999, dedup_gap=99, p_threshold=0.99)
    assert a.config_hash == b.config_hash


def test_pattern_selection_is_an_identity_field():
    a = tmpl_cfg(template_patterns=("double_bottom",))
    b = tmpl_cfg(template_patterns=("double_bottom", "v_reversal"))
    assert a.config_hash != b.config_hash
