import dataclasses

import pytest

from patterns.config import (
    SOURCE_IDENTITY_FIELDS,
    Config,
    identity_fields_for,
    load_config,
    parse_set_overrides,
    with_overrides,
)


def test_hash_stable_default():
    assert Config().config_hash == Config().config_hash
    assert len(Config().config_hash) == 12


def test_hash_insensitive_to_plumbing():
    base = Config()
    for field, value in [("seed", 7), ("position_size", 0.5), ("db_path", "x.db"), ("block_size", 64)]:
        assert dataclasses.replace(base, **{field: value}).config_hash == base.config_hash, field


@pytest.mark.parametrize("field", [f for f in identity_fields_for("knn_shape") if f != "signal_source"])
def test_hash_sensitive_to_every_identity_field(field):
    base = Config()
    value = getattr(base, field)
    if isinstance(value, bool):
        new = not value
    elif isinstance(value, int):
        new = value + 1
    elif isinstance(value, float):
        new = value + 0.01
    elif isinstance(value, tuple):
        new = value + ("SPY",)
    else:
        new = value + "_x"
    assert dataclasses.replace(base, **{field: new}).config_hash != base.config_hash


def test_hash_ignores_fields_foreign_to_active_source(monkeypatch):
    """A knob the active source never reads must not mint a new ledger entry."""
    monkeypatch.setitem(SOURCE_IDENTITY_FIELDS, "dummy", ("window",))
    base = dataclasses.replace(Config(), signal_source="dummy")
    assert dataclasses.replace(base, k=999).config_hash == base.config_hash          # k foreign
    assert dataclasses.replace(base, window=60).config_hash != base.config_hash      # window declared
    assert dataclasses.replace(base, cost_bps=9.9).config_hash != base.config_hash   # shared always counts


def test_unknown_signal_source_rejected():
    with pytest.raises(KeyError, match="nonsense"):
        _ = dataclasses.replace(Config(), signal_source="nonsense").config_hash


def test_load_config_with_overrides(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("window: 40\nsymbols: [qqq]\n")
    cfg = load_config(path, {"k": "25"})
    assert cfg.window == 40
    assert cfg.symbols == ("QQQ",)
    assert cfg.k == 25


def test_load_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("not_a_real_key: 1\n")
    with pytest.raises(ValueError, match="not_a_real_key"):
        load_config(path)


def test_parse_set_overrides():
    assert parse_set_overrides(["k=25", "enable_shorts=true"]) == {"k": "25", "enable_shorts": "true"}
    with pytest.raises(ValueError):
        parse_set_overrides(["nonsense"])


def test_with_overrides_coerces_types():
    cfg = with_overrides(Config(), window="45", enable_shorts="true")
    assert cfg.window == 45 and cfg.enable_shorts is True
