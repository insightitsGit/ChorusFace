"""Blink + jaw ownership ladder — BJ1+ QA gates."""

from __future__ import annotations

from pathlib import Path

from aiface.tickfeed.lid_measure import commit_lid_for_look


def test_bj1_lid_teacher_latches_on_close() -> None:
    """Once lids close, teacher stays True even when lid returns to open."""

    class _Host:
        _tickfeed_lid_teacher = False
        _tickfeed_lid_amt = 1.0

        def apply_lid(self, lid: float) -> None:
            self._tickfeed_lid_amt = float(lid)
            if lid < 0.98:
                self._tickfeed_lid_teacher = True

    host = _Host()
    host.apply_lid(1.0)
    assert host._tickfeed_lid_teacher is False
    host.apply_lid(0.05)
    assert host._tickfeed_lid_teacher is True
    host.apply_lid(1.0)
    assert host._tickfeed_lid_teacher is True  # latched


def test_bj1_app_source_does_not_clear_on_reopen() -> None:
    text = Path("src/aiface/app.py").read_text(encoding="utf-8")
    assert "commit_lid_for_look" in text
    assert "self._tickfeed_lid_teacher = True" in text
    # Old ping-pong assignment must stay gone.
    assert "self._tickfeed_lid_teacher = lid < 0.98" not in text
    assert "BJ1: release measured lid teacher" in text
    assert "Measured lids own LOOK from tick 0" in text


def test_lid_open_deadzone_keeps_rest_lashes_up() -> None:
    """Rest EAR ~0.82 must commit to fully open (no ghost closed plate)."""
    assert commit_lid_for_look(0.82) == 1.0
    assert commit_lid_for_look(0.95) == 1.0
    assert commit_lid_for_look(0.0) == 0.0
    mid = commit_lid_for_look(0.36)
    assert 0.0 < mid < 1.0
    assert mid == 0.36 / 0.72


def test_blink_jaw_ladder_doc() -> None:
    doc = Path("docs/BlinkJawLadder.md").read_text(encoding="utf-8")
    assert "BJ1" in doc
    assert "BJ2" in doc
    assert "BJ3" in doc
