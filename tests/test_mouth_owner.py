"""NWR-first mouth policy — Path A seals removed."""

from __future__ import annotations

from aiface.mouth_owner import (
    commit_plate_amount,
    hold_speech_viseme,
    look_field_gain_scale,
    mute_smile_under_open,
    plate_amount_for_openness,
    resolve_mouth_ownership,
    snap_smile_drive,
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


def test_plate_amount_hard_snap_is_binary() -> None:
    assert plate_amount_for_openness(0.05, hard_snap=True) == 0.0
    assert plate_amount_for_openness(0.25, hard_snap=True) == 1.0
    assert plate_amount_for_openness(0.5, hard_snap=True) == 1.0


def test_mute_smile_under_open() -> None:
    assert mute_smile_under_open(0.55, 0.0) == 0.55
    assert mute_smile_under_open(0.55, 0.12) == 0.55
    assert mute_smile_under_open(0.55, 0.32) == 0.0
    mid = mute_smile_under_open(0.55, 0.22)
    assert 0.0 < mid < 0.55


def test_hold_speech_viseme_while_open() -> None:
    viseme, held = hold_speech_viseme("AH", "REST", open_n=0.8, jaw_n=0.4)
    assert viseme == "AH" and held == "AH"
    viseme, held = hold_speech_viseme("REST", "AH", open_n=0.8, jaw_n=0.4)
    assert viseme == "AH" and held == "AH"
    viseme, held = hold_speech_viseme("REST", "AH", open_n=0.0, jaw_n=0.0)
    assert viseme == "REST" and held == "REST"


def test_tight_lips_cancel_open_hold() -> None:
    """CLOSED/PP must not keep OH parked over closing lips."""
    viseme, held = hold_speech_viseme("CLOSED", "OH", open_n=0.8, jaw_n=0.4)
    assert viseme == "CLOSED" and held == "CLOSED"
    viseme, held = hold_speech_viseme("PP", "OH", open_n=0.8, jaw_n=0.4)
    assert viseme == "PP" and held == "PP"


def test_snap_smile_drive_binary() -> None:
    assert snap_smile_drive(0.55, hard_snap=True) == 1.0
    assert snap_smile_drive(0.4, hard_snap=True) == 0.0
    assert snap_smile_drive(0.55, hard_snap=False) == 0.55


def test_look_field_gain_mutes_live_speech() -> None:
    assert look_field_gain_scale(
        mouth_state="OPEN", plate_open=0.5, live_speech=True
    ) == 0.0


def test_look_field_gain_mutes_opening_midband() -> None:
    assert look_field_gain_scale(
        mouth_state="OPENING", plate_open=0.35, open_vel=1.2
    ) == 0.0
    assert look_field_gain_scale(
        mouth_state="CLOSING", plate_open=0.40, open_vel=-1.0
    ) == 0.0


def test_look_field_gain_tiny_at_steady_open() -> None:
    assert look_field_gain_scale(
        mouth_state="OPEN", plate_open=0.7, live_speech=False
    ) == 0.02


def test_look_field_gain_full_at_rest() -> None:
    assert look_field_gain_scale(
        mouth_state="REST", plate_open=0.0, live_speech=False
    ) == 1.0


def test_commit_plate_amount_boosts_transition() -> None:
    boosted = commit_plate_amount(0.20, "OPENING")
    assert boosted > 0.20
    assert commit_plate_amount(0.02, "OPENING") == 0.02
