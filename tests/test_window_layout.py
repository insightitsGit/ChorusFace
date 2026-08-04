"""Window aspect lock — resize keeps product ratio."""

from __future__ import annotations

from chorusface.window_layout import (
    DEFAULT_ASPECT,
    DEFAULT_WINDOW_SIZE,
    FACE_ONLY_ASPECT,
    aspect_for_layout,
    snap_size_to_aspect,
)


def test_default_aspect_matches_window_size() -> None:
    w, h = DEFAULT_WINDOW_SIZE
    assert abs((w / h) - DEFAULT_ASPECT) < 1e-9


def test_snap_keeps_ratio_when_width_dragged() -> None:
    # Width-edge drag: width moves a lot, height barely.
    w, h = snap_size_to_aspect(
        1200,
        1320,
        DEFAULT_ASPECT,
        prev_width=1024,
        prev_height=1320,
    )
    assert abs((w / h) - DEFAULT_ASPECT) < 0.01
    assert w == 1200


def test_snap_keeps_ratio_when_height_dragged() -> None:
    w, h = snap_size_to_aspect(
        1024,
        1400,
        DEFAULT_ASPECT,
        prev_width=1024,
        prev_height=1320,
    )
    assert abs((w / h) - DEFAULT_ASPECT) < 0.01
    assert h == 1400


def test_face_only_aspect_is_square() -> None:
    assert aspect_for_layout(chat_box_visible=False) == FACE_ONLY_ASPECT
    w, h = snap_size_to_aspect(800, 600, FACE_ONLY_ASPECT, prev_width=1024, prev_height=1024)
    assert w == h


def test_app_declares_resizable_aspect_lock() -> None:
    from pathlib import Path

    text = Path("src/chorusface/app.py").read_text(encoding="utf-8")
    assert "resizable = True" in text
    assert "aspect_ratio = DEFAULT_ASPECT" in text
    assert "def _configure_window_aspect" in text
    assert "snap_size_to_aspect" in text
