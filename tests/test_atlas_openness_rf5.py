"""RF5 — atlas plate openness must not collapse vowels to 0."""

from __future__ import annotations

from aiface.mouth_owner import snap_midband_openness
from aiface.plates import OPEN_TOOTH_VISEMES, VISEME_OPENNESS


def _atlas_openness_for_tag(tag: str, measured: float) -> float:
    """Mirror capture._write_plate_atlas RF5 policy."""
    if tag in {"CLOSED", "PP", "REST", "MM"}:
        return 0.0
    if tag in OPEN_TOOTH_VISEMES or float(VISEME_OPENNESS.get(tag, 0.0)) >= 0.9:
        return 1.0
    if tag:
        return snap_midband_openness(
            max(measured, float(VISEME_OPENNESS.get(tag, 0.35)))
        )
    return snap_midband_openness(measured)


def test_rf5_vowel_plates_not_zeroed_by_midband_snap() -> None:
    # Capture floor often lands AA measured ~0.25 — old path snap→0.
    assert snap_midband_openness(0.25) == 0.0
    assert _atlas_openness_for_tag("AA", 0.25) == 1.0
    assert _atlas_openness_for_tag("OH", 0.18) == 1.0
    assert _atlas_openness_for_tag("EE", 0.22) == 1.0


def test_rf5_closed_plates_stay_sealed() -> None:
    assert _atlas_openness_for_tag("CLOSED", 0.20) == 0.0
    assert _atlas_openness_for_tag("PP", 0.30) == 0.0


def test_rf5_capture_source_has_policy() -> None:
    from pathlib import Path

    text = Path("src/aiface/capture.py").read_text(encoding="utf-8")
    assert "RF5: plate.openness must follow viseme ladder" in text
    assert "tag in OPEN_TOOTH_VISEMES" in text
    plates = Path("src/aiface/plates.py").read_text(encoding="utf-8")
    assert 'viseme_to_plate["REST"] = int(viseme_to_plate["CLOSED"])' in plates
