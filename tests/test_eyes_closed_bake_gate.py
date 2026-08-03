"""eyes_closed plate bake must refuse open-eye takes (Alt C fallback)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "bake_eyes_closed_plate.py"


def _load_bake_module():
    spec = importlib.util.spec_from_file_location("bake_eyes_closed_plate", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_blink_evidence_rejects_flat_scores() -> None:
    mod = _load_bake_module()
    ok, best, median = mod._has_blink_evidence(
        [0.62, 0.61, 0.63, 0.60, 0.62],
        relative_max=0.55,
        abs_max=0.12,
    )
    assert ok is False
    assert best == 0.60
    assert 0.60 < median < 0.63


def test_blink_evidence_accepts_near_closed() -> None:
    mod = _load_bake_module()
    ok, best, _median = mod._has_blink_evidence(
        [0.55, 0.52, 0.08, 0.54, 0.51],
        relative_max=0.55,
        abs_max=0.12,
    )
    assert ok is True
    assert best == 0.08


def test_shader_declares_eye_closed_plate() -> None:
    source = (ROOT / "src/aiface/shaders/avatar.frag").read_text(encoding="utf-8")
    assert "avatar_eye_closed_plate" in source
    assert "avatar_eye_closed_ready" in source
