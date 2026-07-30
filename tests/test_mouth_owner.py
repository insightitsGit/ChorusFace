"""NWR-first mouth policy — Path A seals removed."""

from __future__ import annotations

from aiface.mouth_owner import (
    plate_amount_for_openness,
    resolve_mouth_ownership,
)


def test_always_allows_jaw_muscle_field() -> None:
    own = resolve_mouth_ownership(
        openness=0.0, emotion="NEUTRAL", phoneme="REST", speaking=False
    )
    assert own.muscle_warp
    assert own.jaw
    assert own.field_velocity
    assert not own.dark_cavity
    assert own.as_dict()["policy"] == "nwr-first-no-path-a-seal"


def test_pp_does_not_block_field() -> None:
    """Path A used to seal PP; that locked the mouth — removed."""
    own = resolve_mouth_ownership(
        openness=0.02, emotion="NEUTRAL", phoneme="PP", speaking=True
    )
    assert own.field_velocity
    assert own.jaw


def test_openness_enables_plates() -> None:
    own = resolve_mouth_ownership(
        openness=0.55, emotion="NEUTRAL", phoneme="AH", speaking=True
    )
    assert own.plate_atlas
    assert own.plate_amount == 1.0


def test_happy_enables_smile_plate() -> None:
    own = resolve_mouth_ownership(
        openness=0.1, emotion="HAPPY", phoneme="EH", speaking=True
    )
    assert own.smile_plate


def test_plate_amount_ramp() -> None:
    assert plate_amount_for_openness(0.0) == 0.0
    mid = plate_amount_for_openness(0.25)
    assert 0.0 < mid < 1.0
    assert plate_amount_for_openness(0.5) == 1.0
