"""Incremental render-fidelity ladder — Step QA gates."""

from __future__ import annotations

from pathlib import Path

from aiface.tickfeed.timeline_io import SOURCE_MEASURED, SOURCE_SYNTH


def test_step1_l5_gap_never_on_measured() -> None:
    """Step 1 contract: gap prior only when provenance is not measured."""

    def allow_gap(src_code: int) -> bool:
        # Must match TickFeedDriver.push_drives policy.
        return int(src_code) != SOURCE_MEASURED

    assert allow_gap(SOURCE_MEASURED) is False
    assert allow_gap(SOURCE_SYNTH) is True


def test_step1_driver_source_matches_policy() -> None:
    """Guard against accidental re-introduction of measured L5 blend."""
    text = Path("src/aiface/tickfeed/driver.py").read_text(encoding="utf-8")
    assert "allow_gap = src_code != SOURCE_MEASURED" in text
    # Old policy blended on low conf even when measured — must stay gone.
    assert "mean_conf < 90.0\n                or src_code != SOURCE_MEASURED" not in text


def test_step2_fidelity_hud_flag_and_snapshot_wired() -> None:
    """Step 2: HUD is opt-in overlay only — no mouth/shader fidelity knobs."""
    app_text = Path("src/aiface/app.py").read_text(encoding="utf-8")
    demo_text = Path("scripts/run_tickfeed_demo.py").read_text(encoding="utf-8")
    assert '"--fidelity-hud"' in app_text
    assert '"--fidelity-hud"' in demo_text
    assert "def _fidelity_snapshot(self)" in app_text
    assert 'f"FIDELITY viseme=' in app_text
    assert "self._fidelity_hud = not bool" in app_text
    # Must not re-land failed render knobs with the HUD step.
    assert "mouth_muscles" not in app_text
    assert "mouth_motion" not in app_text
    assert "_tickfeed_jaw_residual" not in app_text
    frag = Path("src/aiface/shaders/avatar.frag").read_text(encoding="utf-8")
    assert "teeth_mask" not in frag
    assert "tongue_mask" not in frag


def test_step2_fidelity_snapshot_keys() -> None:
    """Snapshot shape is stable for status/HUD consumers."""
    from aiface.app import AvatarFaceApp

    class _HudHost:
        _plate_pair = (3, 4)
        _plate_blend_current = (0.25, 0.8)
        _held_speech_viseme = "AA"
        _mouth_transition = "OPEN"
        _plate_openness_current = 0.7
        _field_gain_eff = 0.1
        _tickfeed_look_authority = True
        _fidelity_snapshot = AvatarFaceApp._fidelity_snapshot

    snap = _HudHost()._fidelity_snapshot()
    assert snap["viseme"] == "AA"
    assert snap["transition"] == "OPEN"
    assert snap["plate_index"] == 3
    assert snap["plate_pair"] == [3, 4]
    assert snap["provenance"] == "measured"
    assert "fidelity_hud" not in snap  # toggle lives on app, not snapshot
