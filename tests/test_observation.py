"""Avatar observations — measured smile/open GPU vectors from the world."""

from __future__ import annotations

import json
from pathlib import Path

from chorusface.observation.extract import (
    extract_avatar_observations,
    load_avatar_observations,
    save_avatar_observations,
)
from chorusface.observation.schema import OBS_SCHEMA


def test_extract_smile_vector_from_avatar_world() -> None:
    world = Path("output/worlds/avatar")
    if not (world / "smile.png").is_file() or not (
        world / "expression_catalog.json"
    ).is_file():
        return  # skip when world artifacts absent in CI
    obs = extract_avatar_observations(world)
    smile = obs.look("smile")
    open_ = obs.look("open")
    rest = obs.look("rest")
    assert smile is not None and open_ is not None and rest is not None
    assert smile.gpu.smile_drive >= 0.85
    assert smile.gpu.plate_texture.endswith("smile.png")
    assert open_.gpu.open_drive >= 0.85
    assert len(obs.smile_vector) == 8
    # Smile widens corners vs rest (delta width / corner > 0).
    assert obs.smile_vector[2] > 0.0 or obs.smile_vector[5] > 0.0
    assert obs.cells.mouth_cell_count > 0

    path = save_avatar_observations(world, obs)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == OBS_SCHEMA
    assert "avatar_mouth_pose.w" in payload["looks"][1]["gpu_uniforms"]
    loaded = load_avatar_observations(world)
    assert loaded is not None
    assert loaded.look("smile") is not None
    assert loaded.look("smile").gpu.smile_drive >= 0.85


def test_behavior_driver_uses_observed_smile(tmp_path: Path) -> None:
    world = Path("output/worlds/avatar")
    if not (world / "expression_catalog.json").is_file():
        return
    from chorusface.behavior.driver import BehaviorDriver
    from chorusface.observation.extract import (
        extract_avatar_observations,
        save_avatar_observations,
    )

    # Copy minimal observation package into tmp with model optional.
    obs = extract_avatar_observations(world)
    save_avatar_observations(tmp_path, obs)
    # Point driver at tmp (no ML) — must still resolve observed_smile.
    driver = BehaviorDriver(observations=obs)
    state = driver.resolve(phoneme="REST", smile_amount=0.9, open_amount=0.0)
    assert state.source == "observed_smile"
    assert state.width_n >= 0.5 or state.corner_dx >= 0.5
