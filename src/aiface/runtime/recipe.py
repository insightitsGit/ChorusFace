"""GPU display recipe — the SAME contract at digest (train) and playback.

AMIN_DESIGN Step 8: digestion must learn *how the GPU shows* a look and the
runtime must load and drive that exact recipe. This module is the single
source of truth: `amin_loop.gpu_recipe` serializes it into
`gpu_display_recipe.json` next to the world, and `AvatarFaceApp` loads it
back and feeds the knobs into the shader / uniform path every frame.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Final

RECIPE_NAME: Final = "gpu_display_recipe.json"
RECIPE_SCHEMA: Final = "aiface.gpu_display_recipe.v2"
MAPPING_NAME: Final = "condition_maps.json"

# The real fragment-shader composite order (avatar.frag deform path).
DISPLAY_PATH: Final[tuple[str, ...]] = (
    "identity_photo + tissue_warp (muscles + jaw, undamped jaw)",
    "capture plates open.png / smile.png painted over the mouth matte",
    "optional cavity fill when the jaw actually parts",
    "atlas plate memory (finer viseme shapes)",
    "upper-face expression plate (surprise)",
    "Master Lock ch31 rejects illegal cell writes",
)

# What actually drives each look at playback (uniform contract).
UNIFORM_MAP: Final[dict[str, str]] = {
    "avatar_jaw.z": "jaw angle from viseme table (words own jaw timing)",
    "avatar_plate_blend.y": "open.png / atlas amount from eased plate openness",
    "avatar_mouth_pose.w": "smile.png drive from HAPPY emotion or live width_n",
    "avatar_mouth_pose.y": "mouth openness for cavity weighting",
    "avatar_expr_state.z": "surprise.png blend (upper face)",
    "avatar_recipe": "shader knobs: open_jaw_full, smile_open_overlap, "
    "atlas_strength, cavity_strength",
    "avatar_open_plate / avatar_smile_plate": "capture look textures",
    "avatar_plate_a / avatar_plate_b": "atlas plate pair",
    "avatar_expr_plate": "upper-face expression texture",
}

FORBIDDEN: Final[tuple[str, ...]] = (
    "generative_face_rgb",
    "invented_teeth",
    "path_a_mouth_ownership_seals",
    "jaw_from_raw_rms_energy",
    "plate_warp_damping_0.18",
)

LIVE_CONTROL_CHANNELS: Final[tuple[str, ...]] = (
    "openness_n",
    "jaw_n",
    "width_n",
    "plate_gate",
)


@dataclass(frozen=True, slots=True)
class DisplayRecipe:
    """Numeric knobs of the display recipe, shared by train and play.

    Shader-side (uploaded as the `avatar_recipe` vec4):
      open_jaw_full      jaw angle at which open.png reaches full drive
      smile_open_overlap how much a full open plate suppresses smile.png
      atlas_strength     atlas plate memory opacity ceiling
      cavity_strength    cavity fill opacity when the jaw parts

    Python-side (used by AvatarFaceApp when building targets):
      smile_happy_floor    minimum smile.png drive while emotion is HAPPY
      smile_width_start    live width_n where a NEUTRAL smile starts to show
      smile_width_span     width_n span mapping to full NEUTRAL smile
      closed_openness_cap  plate openness cap for closed visemes (PP/MM/...)
      openness_plate_boost plate openness -> shader mouth-openness boost
      plate_open_floor     eased openness where plates start to fade in
      plate_open_full      eased openness where plates are fully on
    """

    open_jaw_full: float = 0.40
    smile_open_overlap: float = 0.55
    atlas_strength: float = 0.65
    cavity_strength: float = 0.85
    smile_happy_floor: float = 0.55
    smile_width_start: float = 0.12
    smile_width_span: float = 0.35
    closed_openness_cap: float = 0.15
    openness_plate_boost: float = 12.0
    plate_open_floor: float = 0.04
    plate_open_full: float = 0.32

    @property
    def shader_knobs(self) -> tuple[float, float, float, float]:
        """The `avatar_recipe` vec4 uploaded to avatar.frag every frame."""
        return (
            float(self.open_jaw_full),
            float(self.smile_open_overlap),
            float(self.atlas_strength),
            float(self.cavity_strength),
        )

    def to_payload(
        self,
        *,
        world_name: str = "avatar_face.bds",
        plates: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        return {
            "schema": RECIPE_SCHEMA,
            "world": world_name,
            "display_path": list(DISPLAY_PATH),
            "uniforms": dict(UNIFORM_MAP),
            "knobs": asdict(self),
            "plates": dict(plates or {}),
            "live_control_channels": list(LIVE_CONTROL_CHANNELS),
            "forbidden": list(FORBIDDEN),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "DisplayRecipe":
        knobs = payload.get("knobs")
        if not isinstance(knobs, dict):
            return cls()
        names = {f.name for f in fields(cls)}
        kwargs: dict[str, float] = {}
        for key, value in knobs.items():
            if key in names:
                try:
                    kwargs[key] = float(value)
                except (TypeError, ValueError):
                    continue
        return cls(**kwargs)


def world_dir(world: Path | str) -> Path:
    world = Path(world)
    return world if world.is_dir() else world.parent


def recipe_path(world: Path | str) -> Path:
    return world_dir(world) / RECIPE_NAME


def load_display_recipe(world: Path | str) -> DisplayRecipe:
    """Load the digested recipe beside the world; defaults when absent."""
    path = recipe_path(world)
    if not path.is_file():
        print("DisplayRecipe: no gpu_display_recipe.json — built-in defaults")
        return DisplayRecipe()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"DisplayRecipe: load failed ({exc}) — built-in defaults")
        return DisplayRecipe()
    recipe = DisplayRecipe.from_payload(payload)
    print(f"DisplayRecipe: loaded {path}")
    return recipe


def load_condition_jaw(world: Path | str) -> dict[str, float]:
    """Viseme → jaw target table from the digested condition_maps.json.

    AMIN_DESIGN Step 6/10: known words come from the trained tables, so the
    runtime consumes the digested map when present and falls back to the
    built-in `PHONEME_JAW_TARGET` per entry otherwise.
    """
    path = world_dir(world) / MAPPING_NAME
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"ConditionMaps: load failed ({exc}) — built-in tables")
        return {}
    table = payload.get("viseme_table")
    if not isinstance(table, dict):
        return {}
    jaw: dict[str, float] = {}
    for key, entry in table.items():
        if isinstance(entry, dict) and "jaw" in entry:
            try:
                jaw[str(key)] = max(0.0, min(1.0, float(entry["jaw"])))
            except (TypeError, ValueError):
                continue
    if jaw:
        print(f"ConditionMaps: loaded {len(jaw)} viseme jaw targets from {path}")
    return jaw


__all__ = [
    "DISPLAY_PATH",
    "FORBIDDEN",
    "LIVE_CONTROL_CHANNELS",
    "MAPPING_NAME",
    "RECIPE_NAME",
    "RECIPE_SCHEMA",
    "UNIFORM_MAP",
    "DisplayRecipe",
    "load_condition_jaw",
    "load_display_recipe",
    "recipe_path",
    "world_dir",
]
