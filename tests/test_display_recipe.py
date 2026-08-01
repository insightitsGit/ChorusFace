"""AMIN Step 8/10 — the GPU display recipe is real and consumed at play."""

from __future__ import annotations

import json
from pathlib import Path

from aiface.biomechanics.intent import PHONEME_JAW_TARGET
from aiface.runtime.recipe import (
    DISPLAY_PATH,
    RECIPE_NAME,
    RECIPE_SCHEMA,
    DisplayRecipe,
    load_condition_jaw,
    load_display_recipe,
)
from amin_loop.gpu_recipe import build_gpu_recipe, write_gpu_recipe
from amin_loop.mapping import write_condition_maps

SHADER = Path(__file__).resolve().parents[1] / "src/aiface/shaders/avatar.frag"


def test_recipe_payload_round_trip() -> None:
    recipe = DisplayRecipe(open_jaw_full=0.5, smile_happy_floor=0.7)
    payload = recipe.to_payload()
    assert payload["schema"] == RECIPE_SCHEMA
    assert payload["display_path"] == list(DISPLAY_PATH)
    restored = DisplayRecipe.from_payload(payload)
    assert restored == recipe


def test_train_writes_the_runtime_recipe(tmp_path: Path) -> None:
    path = write_gpu_recipe(tmp_path)
    assert path.name == RECIPE_NAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    # L-codes: capture before cavity before atlas (display_layers hierarchy).
    steps = payload["display_path"]
    smile = next(s for s in steps if "capture_smile" in s)
    open_ = next(s for s in steps if "capture_open" in s)
    cavity = next(s for s in steps if "cavity_fill" in s)
    atlas = next(s for s in steps if "atlas_viseme" in s)
    assert steps.index(smile) < steps.index(open_) < steps.index(cavity) < steps.index(
        atlas
    )
    assert "path_a_mouth_ownership_seals" in payload["forbidden"]
    # Play loads exactly what train wrote.
    assert load_display_recipe(tmp_path) == DisplayRecipe()


def test_play_loads_tuned_knobs(tmp_path: Path) -> None:
    tuned = DisplayRecipe(atlas_strength=0.4, closed_openness_cap=0.2)
    write_gpu_recipe(
        tmp_path, build_gpu_recipe(world=tmp_path / "avatar_face.bds", recipe=tuned)
    )
    loaded = load_display_recipe(tmp_path / "avatar_face.bds")
    assert loaded == tuned
    assert loaded.shader_knobs == tuned.shader_knobs


def test_missing_recipe_falls_back_to_defaults(tmp_path: Path) -> None:
    assert load_display_recipe(tmp_path) == DisplayRecipe()
    assert load_condition_jaw(tmp_path) == {}


def test_condition_maps_carry_jaw_table(tmp_path: Path) -> None:
    write_condition_maps(tmp_path)
    jaw = load_condition_jaw(tmp_path)
    assert jaw, "digested condition maps must expose the viseme jaw table"
    for viseme, value in jaw.items():
        assert value == PHONEME_JAW_TARGET.get(viseme, 0.1)
    assert jaw["AH"] >= 0.9
    assert jaw["PP"] == 0.0


def test_shader_reads_recipe_uniform() -> None:
    source = SHADER.read_text(encoding="utf-8")
    assert "uniform vec4 avatar_recipe;" in source
    for component in ("avatar_recipe.x", "avatar_recipe.y",
                      "avatar_recipe.z", "avatar_recipe.w"):
        assert component in source, f"shader must use {component}"


def test_shader_warps_from_nwr_field() -> None:
    """AMIN steps 1-3: the GPU field's 2D velocity vectors must move pixels."""
    source = SHADER.read_text(encoding="utf-8")
    assert "uniform float avatar_field_gain;" in source
    assert "field_displacement" in source
    # The field term must be part of the visible deformation, not debug-only.
    total = source.split("vec2 total_displacement", 1)[1].split("}", 1)[0]
    assert "field_displacement" in total
    assert "muscles + jaw + field" in total
    assert DisplayRecipe().field_warp_gain > 0.0


def test_shader_snaps_plates_with_recipe_knob() -> None:
    """AMIN step 12: the plate-snap knob is a real uniform used by the blend."""
    source = SHADER.read_text(encoding="utf-8")
    assert "uniform float avatar_plate_sharpness;" in source
    # Loaded once into `snap`, then applied to capture drives + atlas matte.
    assert "float snap = clamp(avatar_plate_sharpness" in source
    assert "smoothstep(0.18, 0.82, open_drive), snap)" in source
    assert "harden_matte(max(pa.a, pb.a), snap)" in source
    recipe = DisplayRecipe()
    assert 0.0 < recipe.plate_sharpness <= 1.0
    # The knob rides the same serialized contract as every other knob.
    restored = DisplayRecipe.from_payload(
        DisplayRecipe(plate_sharpness=0.9).to_payload()
    )
    assert restored.plate_sharpness == 0.9


def test_capture_exposes_display_resolution() -> None:
    """AMIN step 11: display plates are re-cut above the 256² field grid."""
    from aiface.capture import DISPLAY_SIZE, resample_frames_hires
    from aiface.runtime.bds import GRID_HEIGHT

    assert DISPLAY_SIZE >= 4 * GRID_HEIGHT
    # No frames → no work, no crash (worlds captured before step 11 keep 256).
    assert resample_frames_hires("missing.mp4", []) == {}
