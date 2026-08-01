"""Display layer hierarchy — one ordered stack from field → pixels.

Layers used to be ad-hoc mixes in ``avatar.frag`` + timeline + cell plan.
This module is the **single coding** for names, order, authority, and
realtime skip rules. CPU (timeline / groups / recipe) and GPU (shader
composite) must agree on these codes.

Hierarchy (bottom → top)
------------------------
Plane A — FIELD (moves tissue; never rewrites albedo identity)
  L0  identity_lock     Master Lock / locked skull (immutable)
  L1  field_velocity    NWR ch0/1 ±4 + neighbor couple
  L2  muscle_jaw_warp   Analytic muscle + jaw inverse-warp of photo
  L3  cell_groups       Word-timed lip/teeth/cavity cell drives → L1

Plane B — LOOK (photographed plates; billboards on UV)
  L4  capture_smile     smile.png
  L5  capture_open      open.png (muted when atlas owns)
  L6  cavity_fill       Synthetic/photo cavity (gated off under atlas)
  L7  atlas_viseme      Viseme plate bank (speech authority under hard snap)
  L8  expr_upper        surprise / brow plate

Plane C — PRESENTATION
  L9  eyes_lids         Globe / blink composite
  L10 brow_procedural   Display brow lift (not lock-gated)
  L11 hud_overlay       Chat / Hold UI (separate pass)

Realtime rule: evaluate ``FrameLayerState`` once per tick; skip drives and
texture binds for layers marked inactive. Never reorder Plane B relative to
this table — mid-stack swaps cause blur / gap regressions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Final, Iterable, Mapping


class LayerPlane(IntEnum):
    """Authority plane — field under looks under presentation."""

    FIELD = 0
    LOOK = 1
    PRESENTATION = 2


class DisplayLayer(IntEnum):
    """Stable layer codes. Integer value == composite order (bottom → top)."""

    IDENTITY_LOCK = 0
    FIELD_VELOCITY = 1
    MUSCLE_JAW_WARP = 2
    CELL_GROUPS = 3
    CAPTURE_SMILE = 4
    CAPTURE_OPEN = 5
    CAVITY_FILL = 6
    ATLAS_VISEME = 7
    EXPR_UPPER = 8
    EYES_LIDS = 9
    BROW_PROCEDURAL = 10
    HUD_OVERLAY = 11


@dataclass(frozen=True, slots=True)
class LayerSpec:
    """One layer in the display stack."""

    code: DisplayLayer
    name: str
    plane: LayerPlane
    owner: str
    description: str
    #: Shader / CPU stage hint (documentation + probes).
    stage: str
    #: When True, inactive frames skip work for realtime.
    skippable: bool = True


#: Canonical ordered stack — do not reorder without a deliberate shader change.
LAYER_SPECS: Final[tuple[LayerSpec, ...]] = (
    LayerSpec(
        DisplayLayer.IDENTITY_LOCK,
        "identity_lock",
        LayerPlane.FIELD,
        "master_lock_ch31",
        "Locked skull / albedo identity — AI velocity rejected",
        "constraint.comp + photo base",
        skippable=False,
    ),
    LayerSpec(
        DisplayLayer.FIELD_VELOCITY,
        "field_velocity",
        LayerPlane.FIELD,
        "cell_cluster + constraint",
        "Unlocked soft-cell ±4 velocity + Moore neighbor couple",
        "constraint.comp → avatar.frag field_displacement",
    ),
    LayerSpec(
        DisplayLayer.MUSCLE_JAW_WARP,
        "muscle_jaw_warp",
        LayerPlane.FIELD,
        "biomechanics",
        "Muscle + jaw inverse-warp of the identity photo",
        "avatar.frag inverse_warp",
        skippable=False,
    ),
    LayerSpec(
        DisplayLayer.CELL_GROUPS,
        "cell_groups",
        LayerPlane.FIELD,
        "mouth_cell_plan + mouth_groups",
        "Word-timed upper/lower lip, corners, teeth, cavity cell drives",
        "MouthCellPlan → ±4 into FIELD_VELOCITY",
    ),
    LayerSpec(
        DisplayLayer.CAPTURE_SMILE,
        "capture_smile",
        LayerPlane.LOOK,
        "mouth_timeline + emotion",
        "smile.png capture look",
        "avatar.frag capture smile mix",
    ),
    LayerSpec(
        DisplayLayer.CAPTURE_OPEN,
        "capture_open",
        LayerPlane.LOOK,
        "mouth_timeline",
        "open.png capture look (muted when atlas owns)",
        "avatar.frag capture open mix",
    ),
    LayerSpec(
        DisplayLayer.CAVITY_FILL,
        "cavity_fill",
        LayerPlane.LOOK,
        "recipe.cavity_strength",
        "Gated cavity fill — bows out under atlas/open",
        "avatar.frag cavity_color",
    ),
    LayerSpec(
        DisplayLayer.ATLAS_VISEME,
        "atlas_viseme",
        LayerPlane.LOOK,
        "mouth_timeline + plate_atlas",
        "Viseme plate bank — speech look authority under hard snap",
        "avatar.frag atlas plate mix",
    ),
    LayerSpec(
        DisplayLayer.EXPR_UPPER,
        "expr_upper",
        LayerPlane.LOOK,
        "expression_catalog",
        "surprise.png / upper-face plate",
        "avatar.frag expr plate mix",
    ),
    LayerSpec(
        DisplayLayer.EYES_LIDS,
        "eyes_lids",
        LayerPlane.PRESENTATION,
        "biomechanics.eyes",
        "Globe / blink / widen composite",
        "avatar.frag eye aperture",
        skippable=False,
    ),
    LayerSpec(
        DisplayLayer.BROW_PROCEDURAL,
        "brow_procedural",
        LayerPlane.PRESENTATION,
        "expression + emotion",
        "Procedural brow lift (display-only, not lock-gated)",
        "avatar.frag brow resample",
    ),
    LayerSpec(
        DisplayLayer.HUD_OVERLAY,
        "hud_overlay",
        LayerPlane.PRESENTATION,
        "chatbox",
        "Chat panel / Hold slider / Mouth speed",
        "hud.frag",
    ),
)

LAYER_BY_NAME: Final[dict[str, LayerSpec]] = {spec.name: spec for spec in LAYER_SPECS}
LAYER_BY_CODE: Final[dict[DisplayLayer, LayerSpec]] = {
    spec.code: spec for spec in LAYER_SPECS
}

#: Human-readable display_path strings for gpu_display_recipe.json (ordered).
DISPLAY_PATH: Final[tuple[str, ...]] = tuple(
    f"L{int(spec.code):02d}:{spec.name} — {spec.description}" for spec in LAYER_SPECS
)

#: Mouth groups live under L3; listed here for hierarchy probes.
MOUTH_GROUP_LAYER: Final = DisplayLayer.CELL_GROUPS


@dataclass(slots=True)
class FrameLayerState:
    """Per-tick active flags — evaluate once, drive CPU + log consistently."""

    active: dict[str, bool] = field(default_factory=dict)
    amounts: dict[str, float] = field(default_factory=dict)
    phoneme: str = "REST"
    notes: list[str] = field(default_factory=list)

    def is_on(self, name: str) -> bool:
        return bool(self.active.get(name, False))

    def amount(self, name: str) -> float:
        return float(self.amounts.get(name, 0.0))

    def ordered_active(self) -> list[str]:
        return [
            spec.name
            for spec in LAYER_SPECS
            if self.is_on(spec.name)
        ]

    def snapshot(self) -> dict[str, Any]:
        return {
            "phoneme": self.phoneme,
            "active": [name for name in self.ordered_active()],
            "amounts": {k: round(v, 3) for k, v in self.amounts.items() if v > 1e-4},
            "notes": list(self.notes),
            "hierarchy": hierarchy_table(),
        }


def hierarchy_table() -> list[dict[str, Any]]:
    """Stable table for docs / GET /cells / probes."""
    rows: list[dict[str, Any]] = []
    for spec in LAYER_SPECS:
        rows.append(
            {
                "order": int(spec.code),
                "code": f"L{int(spec.code):02d}",
                "name": spec.name,
                "plane": spec.plane.name,
                "owner": spec.owner,
                "stage": spec.stage,
                "skippable": spec.skippable,
                "description": spec.description,
            }
        )
    return rows


def assert_display_order(names: Iterable[str]) -> None:
    """Raise if ``names`` is not a subsequence of the canonical order."""
    order = {spec.name: int(spec.code) for spec in LAYER_SPECS}
    last = -1
    for name in names:
        if name not in order:
            raise KeyError(f"unknown display layer {name!r}")
        idx = order[name]
        if idx < last:
            raise ValueError(
                f"display layer order violated: {name!r} at {idx} after {last}"
            )
        last = idx


def evaluate_frame_layers(
    *,
    phoneme: str,
    plate_open_amount: float,
    smile_amount: float,
    atlas_strength: float,
    cavity_strength: float,
    field_gain: float,
    expr_blend: float,
    brow_raise: float,
    speaking_plate: bool,
    cell_plan_steps: int,
    hard_snap: bool,
    chat_visible: bool,
) -> FrameLayerState:
    """Resolve which layers work this tick (realtime skip map)."""
    state = FrameLayerState(phoneme=str(phoneme or "REST"))
    open_amt = float(plate_open_amount)
    smile = float(smile_amount)
    atlas = float(atlas_strength) * open_amt
    # L0–L2 always conceptually present while deforming.
    state.active["identity_lock"] = True
    state.active["muscle_jaw_warp"] = True
    state.amounts["field_velocity"] = float(field_gain)
    state.active["field_velocity"] = float(field_gain) > 1e-4
    state.active["cell_groups"] = int(cell_plan_steps) > 0
    state.amounts["cell_groups"] = float(cell_plan_steps)

    # Look plane — atlas owns speech under hard snap.
    atlas_owns = bool(speaking_plate and hard_snap and atlas > 0.35)
    state.active["atlas_viseme"] = open_amt > 0.001 and atlas > 0.001
    state.amounts["atlas_viseme"] = atlas
    state.active["capture_open"] = open_amt > 0.001 and not atlas_owns
    state.amounts["capture_open"] = 0.0 if atlas_owns else open_amt
    state.active["capture_smile"] = smile > 0.001 and not atlas_owns
    state.amounts["capture_smile"] = 0.0 if atlas_owns else smile
    # Cavity only when something is parting and atlas does not own the hole.
    state.active["cavity_fill"] = (
        float(cavity_strength) > 0.01 and open_amt > 0.05 and not atlas_owns
    )
    state.amounts["cavity_fill"] = (
        float(cavity_strength) if state.active["cavity_fill"] else 0.0
    )
    state.active["expr_upper"] = float(expr_blend) > 0.001
    state.amounts["expr_upper"] = float(expr_blend)
    state.active["eyes_lids"] = True
    state.active["brow_procedural"] = float(brow_raise) > 0.04 and float(expr_blend) < 0.15
    state.amounts["brow_procedural"] = float(brow_raise)
    state.active["hud_overlay"] = bool(chat_visible)

    if atlas_owns:
        state.notes.append("atlas_owns_speech: capture_open/smile + cavity skipped")
    assert_display_order(state.ordered_active())
    return state


def layer_uniforms() -> Mapping[str, str]:
    """Uniform ownership hints aligned with LAYER_SPECS."""
    return {
        "L00:identity_lock": "HUMAN_LOCK_CHANNEL / Master Lock",
        "L01:field_velocity": "avatar_field_gain, world ch0/1",
        "L02:muscle_jaw_warp": "muscle_drive[], avatar_jaw",
        "L03:cell_groups": "MouthCellPlan / mouth_groups recipes",
        "L04:capture_smile": "avatar_smile_plate, avatar_mouth_pose.w",
        "L05:capture_open": "avatar_open_plate, avatar_plate_blend.y",
        "L06:cavity_fill": "avatar_recipe.w (cavity_strength)",
        "L07:atlas_viseme": "avatar_plate_a/b, avatar_plate_blend, avatar_recipe.z",
        "L08:expr_upper": "avatar_expr_plate, avatar_expr_state",
        "L09:eyes_lids": "avatar_eye_state / tissue.a",
        "L10:brow_procedural": "avatar_expr_state.y",
        "L11:hud_overlay": "hud texture pass",
    }


__all__ = [
    "DISPLAY_PATH",
    "LAYER_BY_CODE",
    "LAYER_BY_NAME",
    "LAYER_SPECS",
    "MOUTH_GROUP_LAYER",
    "DisplayLayer",
    "FrameLayerState",
    "LayerPlane",
    "LayerSpec",
    "assert_display_order",
    "evaluate_frame_layers",
    "hierarchy_table",
    "layer_uniforms",
]
