"""Incremental render-fidelity ladder — Step QA gates."""

from __future__ import annotations

from pathlib import Path

from chorusface.tickfeed.timeline_io import SOURCE_MEASURED, SOURCE_SYNTH


def test_step1_l5_gap_never_on_measured() -> None:
    """Step 1 contract: gap prior only when provenance is not measured."""

    def allow_gap(src_code: int) -> bool:
        # Must match TickFeedDriver.push_drives policy.
        return int(src_code) != SOURCE_MEASURED

    assert allow_gap(SOURCE_MEASURED) is False
    assert allow_gap(SOURCE_SYNTH) is True


def test_step1_driver_source_matches_policy() -> None:
    """Guard against accidental re-introduction of measured L5 blend."""
    text = Path("src/chorusface/tickfeed/driver.py").read_text(encoding="utf-8")
    assert "allow_gap = src_code != SOURCE_MEASURED" in text
    # Old policy blended on low conf even when measured — must stay gone.
    assert "mean_conf < 90.0\n                or src_code != SOURCE_MEASURED" not in text


def test_step2_fidelity_hud_flag_and_snapshot_wired() -> None:
    """HUD is opt-in overlay; parked P-A knobs must stay out of app/shader."""
    app_text = Path("src/chorusface/app.py").read_text(encoding="utf-8")
    demo_text = Path("scripts/run_tickfeed_demo.py").read_text(encoding="utf-8")
    assert '"--fidelity-hud"' in app_text
    assert '"--fidelity-hud"' in demo_text
    assert "def _fidelity_snapshot(self)" in app_text
    assert 'f"FIDELITY viseme=' in app_text
    assert "phase={snap['phase']}" in app_text
    assert "self._fidelity_hud = not bool" in app_text
    # P3 MouthMotionState is allowed; parked P-A must stay out.
    assert "MouthMotionState" in app_text
    assert "mouth_muscles" not in app_text
    assert "_tickfeed_jaw_residual" not in app_text
    frag = Path("src/chorusface/shaders/avatar.frag").read_text(encoding="utf-8")
    assert "teeth_mask" not in frag
    assert "tongue_mask" not in frag


def test_dense_kit_script_has_tongue_th_and_blink() -> None:
    """Dense calibration contract: tongue TH + deliberate BLINK lid window."""
    from chorusface.tickfeed.calibration import calibration_script_payload
    from chorusface.tickfeed.schema import BeatId

    script = calibration_script_payload()
    assert int(BeatId.TONGUE_TH) == 7
    assert int(BeatId.BLINK) == 8
    assert any(b["id"] == "TONGUE_TH" for b in script["beats"])
    assert any(b["id"] == "BLINK" for b in script["beats"])
    prompt = Path("docs/AvatarCalibrationPrompt.md").read_text(encoding="utf-8")
    assert "TONGUE_TH" in prompt
    assert "BLINK" in prompt
    assert "think" in prompt.lower()


def test_step4_plate_b_mirrors_a_when_mix_zero() -> None:
    """Step 4: upload binds plate_b=plate_a only when mix is already ~0."""
    text = Path("src/chorusface/app.py").read_text(encoding="utf-8")
    assert "if mix_ab <= 1e-6:" in text
    assert "plate_b = plate_a" in text
    # Must not force mix=0 / always-nearest (failed fidelity bundle).
    assert "Nearest measured plate only" not in text
    assert "mix_upload = 0.0" not in text


def test_step3_ownership_hard_snap_wired() -> None:
    """Step 3: refresh path passes hard_snap; resolver matches GPU commit."""
    app_text = Path("src/chorusface/app.py").read_text(encoding="utf-8")
    owner_text = Path("src/chorusface/mouth_owner.py").read_text(encoding="utf-8")
    assert "hard_snap=hard," in app_text
    assert "mouth_state=str(getattr(self, \"_mouth_transition\", \"REST\"))" in app_text
    assert "commit_plate_amount(open_n, mouth_state)" in owner_text
    from chorusface.mouth_owner import commit_plate_amount, resolve_mouth_ownership

    own = resolve_mouth_ownership(
        openness=0.40,
        phoneme="AH",
        mouth_state="OPEN",
        hard_snap=True,
    )
    assert own.plate_amount == commit_plate_amount(0.40, "OPEN")


def test_step2_fidelity_snapshot_keys() -> None:
    """Snapshot shape is stable for status/HUD consumers."""
    from chorusface.app import AvatarFaceApp

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
    assert snap["phase"] == "REST"  # no MouthMotionState on host → default
    assert snap["plate_index"] == 3
    assert snap["plate_pair"] == [3, 4]
    assert snap["provenance"] == "measured"
    assert snap["open_png"] is True
    assert snap["occlusion"] == {"teeth_visible": False, "tongue_visible": False}
    assert snap["muscles"] == {}
    assert float(snap["jaw_gpu"]) == 0.0  # look authority → hard-zero
    assert "fidelity_hud" not in snap  # toggle lives on app, not snapshot
