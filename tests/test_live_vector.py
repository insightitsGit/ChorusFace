"""From-scratch live_vector package — NWR-first (no ownership seal)."""

from __future__ import annotations

from chorusface.live_vector.driver import LiveVectorDriver
from chorusface.live_vector.features import rms_history_features
from chorusface.live_vector.schema import FEATURE_DIM, LiveControlVector


def test_feature_dim() -> None:
    assert rms_history_features([0.1, 0.2]).shape == (FEATURE_DIM,)


def test_open_vowel_table_floor() -> None:
    driver = LiveVectorDriver()
    driver.push_rms(0.0)
    out = driver.resolve(phoneme="AH", phoneme_jaw=1.0)
    assert out.jaw_n >= 1.0
    assert isinstance(out, LiveControlVector)


def test_pp_follows_table_not_forced_lock() -> None:
    """PP jaw table is 0; driver must not invent a Path-A ownership lock."""
    driver = LiveVectorDriver()
    driver.push_rms(0.0)
    out = driver.resolve(phoneme="PP", phoneme_jaw=0.0)
    assert out.jaw_n == 0.0


def test_try_load_missing(tmp_path) -> None:
    driver = LiveVectorDriver.try_load(tmp_path)
    assert not driver.using_ml
