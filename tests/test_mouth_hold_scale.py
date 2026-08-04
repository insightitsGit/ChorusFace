"""Realtime mouth-hold slider mapping."""

from __future__ import annotations

from chorusface.mouth_speed import clamp_hold_scale, hold_scale_to_params


def test_hold_scale_monotone() -> None:
    d0, b0, m0 = hold_scale_to_params(0.0)
    d1, b1, m1 = hold_scale_to_params(1.0)
    assert d0 < d1 and b0 < b1 and m0 < m1
    assert d0 >= 0.03 and d1 <= 0.45


def test_clamp_hold_scale() -> None:
    assert clamp_hold_scale(-1.0) == 0.0
    assert clamp_hold_scale(2.0) == 1.0
    assert clamp_hold_scale(0.45) == 0.45
