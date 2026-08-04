"""Display layer hierarchy — order, evaluate, recipe contract."""

from __future__ import annotations

from pathlib import Path

from chorusface.display_layers import (
    DISPLAY_PATH,
    LAYER_SPECS,
    DisplayLayer,
    assert_display_order,
    evaluate_frame_layers,
    hierarchy_table,
)
from chorusface.runtime.recipe import DISPLAY_PATH as RECIPE_PATH, UNIFORM_MAP

SHADER = Path(__file__).resolve().parents[1] / "src/chorusface/shaders/avatar.frag"


def test_layer_codes_are_contiguous_bottom_to_top() -> None:
    codes = [int(spec.code) for spec in LAYER_SPECS]
    assert codes == list(range(len(LAYER_SPECS)))
    assert LAYER_SPECS[0].code is DisplayLayer.IDENTITY_LOCK
    assert LAYER_SPECS[-1].code is DisplayLayer.HUD_OVERLAY


def test_planes_never_interleave() -> None:
    last_plane = -1
    for spec in LAYER_SPECS:
        plane = int(spec.plane)
        assert plane >= last_plane, f"{spec.name} dropped below prior plane"
        last_plane = plane


def test_assert_display_order_accepts_subsequence() -> None:
    assert_display_order(["identity_lock", "capture_smile", "atlas_viseme", "eyes_lids"])


def test_assert_display_order_rejects_swap() -> None:
    try:
        assert_display_order(["atlas_viseme", "capture_open"])
    except ValueError as exc:
        assert "order violated" in str(exc)
    else:
        raise AssertionError("expected ValueError for atlas before capture")


def test_evaluate_atlas_owns_skips_capture_and_cavity() -> None:
    state = evaluate_frame_layers(
        phoneme="AH",
        plate_open_amount=0.9,
        smile_amount=0.4,
        atlas_strength=0.85,
        cavity_strength=0.8,
        field_gain=0.2,
        expr_blend=0.0,
        brow_raise=0.1,
        speaking_plate=True,
        cell_plan_steps=12,
        hard_snap=True,
        chat_visible=False,
    )
    assert state.is_on("atlas_viseme")
    assert state.is_on("cell_groups")
    assert not state.is_on("capture_open")
    assert not state.is_on("capture_smile")
    assert not state.is_on("cavity_fill")
    assert "atlas_owns_speech" in " ".join(state.notes)
    assert_display_order(state.ordered_active())


def test_evaluate_rest_skips_look_plane() -> None:
    state = evaluate_frame_layers(
        phoneme="REST",
        plate_open_amount=0.0,
        smile_amount=0.0,
        atlas_strength=0.7,
        cavity_strength=0.5,
        field_gain=0.2,
        expr_blend=0.0,
        brow_raise=0.0,
        speaking_plate=False,
        cell_plan_steps=0,
        hard_snap=True,
        chat_visible=True,
    )
    assert state.is_on("identity_lock")
    assert state.is_on("muscle_jaw_warp")
    assert state.is_on("eyes_lids")
    assert state.is_on("hud_overlay")
    assert not state.is_on("cell_groups")
    assert not state.is_on("atlas_viseme")
    assert not state.is_on("capture_open")


def test_recipe_shares_display_path() -> None:
    assert RECIPE_PATH == DISPLAY_PATH
    assert any(k.startswith("L07:") for k in UNIFORM_MAP)


def test_hierarchy_table_matches_specs() -> None:
    table = hierarchy_table()
    assert len(table) == len(LAYER_SPECS)
    assert table[4]["name"] == "capture_smile"
    assert table[7]["code"] == "L07"


def test_shader_documents_layer_codes() -> None:
    source = SHADER.read_text(encoding="utf-8")
    assert "L04/L05" in source or "L04" in source
    assert "L07" in source
    assert "L10" in source
