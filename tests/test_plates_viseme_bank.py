"""AMIN step 12/13 — hard plate snap + viseme→frame bank."""

from __future__ import annotations

from dataclasses import dataclass

from aiface.plates import (
    HARD_SNAP_THRESHOLD,
    AtlasPlate,
    PlateAtlas,
    match_visemes_to_frames,
    select_viseme_atlas_frames,
)
from aiface.runtime.recipe import DisplayRecipe


@dataclass
class _Metrics:
    mouth_open: float
    smile_width: float
    teeth: float = 0.0
    sharpness: float = 20.0


@dataclass
class _Frame:
    index: int
    metrics: _Metrics


def _frames() -> list[_Frame]:
    return [
        _Frame(0, _Metrics(0.00, 0.30)),
        _Frame(1, _Metrics(0.02, 0.32)),
        _Frame(2, _Metrics(0.08, 0.40)),
        _Frame(3, _Metrics(0.12, 0.45)),
        _Frame(4, _Metrics(0.16, 0.50)),
        _Frame(5, _Metrics(0.20, 0.55)),
        _Frame(6, _Metrics(0.22, 0.70)),  # smile-ish
        _Frame(7, _Metrics(0.25, 0.40)),  # open
    ]


def test_hard_snap_returns_single_plate() -> None:
    plates = tuple(
        AtlasPlate(
            index=i,
            path=f"plates/plate_{i:02d}.png",
            openness=i * 0.05,
            smile_width=0.3,
            frame_index=i,
            time_seconds=float(i),
        )
        for i in range(5)
    )
    atlas = PlateAtlas(
        plates=plates,
        viseme_openness={"AH": 1.0, "PP": 0.0},
        viseme_to_plate={"AH": 4, "PP": 0},
    )
    ia, ib, mix = atlas.pair_for_viseme("AH", hard_snap=True)
    assert ia == ib == 4
    assert mix == 0.0
    ia, ib, mix = atlas.pair_for_viseme("PP", hard_snap=True)
    assert ia == ib == 0
    assert mix == 0.0


def test_pair_for_openness_hard_snap_no_mid_blend() -> None:
    plates = tuple(
        AtlasPlate(
            index=i,
            path=f"p{i}.png",
            openness=float(i),
            smile_width=0.3,
            frame_index=i,
            time_seconds=0.0,
        )
        for i in range(3)
    )
    atlas = PlateAtlas(plates=plates, viseme_openness={})
    ia, ib, mix = atlas.pair_for_openness(0.5, hard_snap=True)
    assert ia == ib
    assert mix == 0.0


def test_viseme_bank_assigns_all_canonical() -> None:
    frames = _frames()
    matched = match_visemes_to_frames(frames)
    assert "AH" in matched and "PP" in matched and "EE" in matched
    chosen, mapping = select_viseme_atlas_frames(frames, max_plates=8)
    assert chosen
    assert len(mapping) >= 10
    # Closed sounds should not pick a more-open plate than AH when alternatives exist.
    assert (
        chosen[mapping["PP"]].metrics.mouth_open
        <= chosen[mapping["AH"]].metrics.mouth_open
    )


def test_recipe_default_hard_snap_threshold() -> None:
    assert DisplayRecipe().plate_sharpness >= HARD_SNAP_THRESHOLD
