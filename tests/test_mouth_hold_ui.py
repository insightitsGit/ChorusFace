"""Hold scrollbar hit-testing uses top-down overlay pixels (mglw convention)."""

from __future__ import annotations

from aiface.chatbox import frame_layout, hit_test
from aiface.mouth_speed import clamp_hold_scale


def test_hold_track_hit_in_panel_header() -> None:
    """Clicks on the bottom chat band must land on mouth_hold_track."""
    width, height = 1280, 720
    _uv, panel = frame_layout(width, height)
    px, py, pw, ph = panel
    assert py > height // 2  # panel is the bottom band in top-down pixels

    # Simulate a painted track in the header row (same layout as chatbox).
    track = (px + 90, py, 200, 28)
    hits = {
        "mouth_button": (px + pw - 140, py + 4, 120, 22),
        "mouth_hold_track": track,
    }
    # Center of the track — top-down mouse y (mglw already flipped pyglet).
    mx = track[0] + track[2] // 2
    my = track[1] + track[3] // 2
    assert hit_test(hits, mx, my) == "mouth_hold_track"

    # Double-flipped y (the old bug) lands near the top — must miss.
    flipped = height - my - 1
    assert flipped < height // 2
    assert hit_test(hits, mx, flipped) is None


def test_hold_scale_from_track_x() -> None:
    tx, tw, pad = 100, 200, 8
    visual_x = tx + pad
    visual_w = tw - pad * 2
    left = clamp_hold_scale((visual_x - visual_x) / visual_w)
    mid = clamp_hold_scale(((visual_x + visual_w * 0.5) - visual_x) / visual_w)
    right = clamp_hold_scale(((visual_x + visual_w) - visual_x) / visual_w)
    assert left == 0.0
    assert 0.49 <= mid <= 0.51
    assert right == 1.0
