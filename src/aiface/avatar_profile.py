"""Avatar adoption contract — any qualifying world dir plugs into the same GPU path.

Digestion learns how each unlocked cell couples to the GPU from the user's
uploaded images/video (``.bds`` + plates + ``region_catalog``). Runtime must
not assume one hard-coded face under ``output/worlds/avatar/``.

This module is the **abstraction layer**:

* ``AvatarProfile`` — side-car JSON describing one adopted world
* ``validate_avatar_profile`` — hard requirements for playable coupling
* ``open_avatar`` — typed bundle ``AvatarFaceApp`` / train / verify share

Global contracts stay outside the profile: display layer order (L00–L11),
viseme flow tables, Master Lock. The profile only points at per-avatar
geometry, plates, and cell coupling artifacts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Mapping

from aiface.paths import WORLDS
from aiface.runtime.recipe import (
    DisplayRecipe,
    load_condition_jaw,
    load_display_recipe,
    world_dir as recipe_world_dir,
)

PROFILE_NAME: Final = "avatar_profile.json"
PROFILE_SCHEMA: Final = "aiface.avatar_profile.v1"

#: Dense field + identity + capture looks — without these, no GPU coupling.
REQUIRED_FILES: Final[tuple[str, ...]] = (
    "avatar_face.bds",
    "source_face.png",
    "open.png",
    "smile.png",
)

#: Learned cell→GPU side-cars. Missing → WARN at adopt, FAIL if mouth absent.
COUPLING_FILES: Final[tuple[str, ...]] = (
    "region_catalog.json",
    "gpu_display_recipe.json",
    "condition_maps.json",
)

#: Optional quality / speech cover.
OPTIONAL_FILES: Final[tuple[str, ...]] = (
    "surprise.png",
    "plate_atlas.json",
    "expression_catalog.json",
    "face_parts.npy",
    "face_tissue.npy",
    "capture_meta.json",
    "live_vector_model.joblib",
    "live_vector_model.meta.json",
    "cell_transition_track.npz",
    "behavior_model.joblib",
)

MIN_MOUTH_CELLS: Final = 32


class AvatarAdoptionError(RuntimeError):
    """World does not meet the adoption requirements."""


@dataclass(slots=True)
class AvatarGeometry:
    """Per-avatar registration learned at digest (grid space, y-up)."""

    face_box: dict[str, float] = field(
        default_factory=lambda: {
            "x": 0.0,
            "y": 0.0,
            "width": 256.0,
            "height": 256.0,
        }
    )
    mouth_center_grid: tuple[float, float] = (128.0, 64.0)
    mouth_cell_count: int = 0
    grid_width: int = 256
    grid_height: int = 256

    def as_dict(self) -> dict[str, Any]:
        return {
            "face_box": dict(self.face_box),
            "mouth_center_grid": [
                float(self.mouth_center_grid[0]),
                float(self.mouth_center_grid[1]),
            ],
            "mouth_cell_count": int(self.mouth_cell_count),
            "grid": [int(self.grid_width), int(self.grid_height)],
        }


@dataclass(slots=True)
class AvatarValidation:
    """Result of requirement checks."""

    ok: bool
    missing: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    mouth_unlocked: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "missing": list(self.missing),
            "warnings": list(self.warnings),
            "mouth_unlocked": bool(self.mouth_unlocked),
            "details": dict(self.details),
        }


@dataclass(slots=True)
class AvatarProfile:
    """Portable description of one adopted avatar world directory."""

    id: str
    world: str = "avatar_face.bds"
    source_face: str = "source_face.png"
    plates: dict[str, str] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    geometry: AvatarGeometry = field(default_factory=AvatarGeometry)
    capture_priors: dict[str, float] = field(default_factory=dict)
    validation: AvatarValidation = field(
        default_factory=lambda: AvatarValidation(ok=False)
    )
    root: Path | None = None

    def path(self, name: str) -> Path:
        if self.root is None:
            raise AvatarAdoptionError("profile has no root directory")
        return self.root / name

    def world_path(self) -> Path:
        return self.path(self.world)

    def source_path(self) -> Path:
        return self.path(self.source_face)

    def plate_path(self, role: str) -> Path | None:
        name = self.plates.get(role)
        if not name:
            return None
        path = self.path(name)
        return path if path.is_file() else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": PROFILE_SCHEMA,
            "id": self.id,
            "world": self.world,
            "source_face": self.source_face,
            "plates": dict(self.plates),
            "artifacts": dict(self.artifacts),
            "geometry": self.geometry.as_dict(),
            "capture_priors": {k: float(v) for k, v in self.capture_priors.items()},
            "validation": self.validation.as_dict(),
            "display_layers": "aiface.display_layers (global L00–L11)",
            "coupling": (
                "Unlocked mouth cells (permeability≥0.5) → ±4 velocity → "
                "constraint.comp → avatar.frag field_displacement + plates"
            ),
        }


@dataclass(slots=True)
class AvatarBundle:
    """Opened avatar — everything runtime needs to adopt a face."""

    profile: AvatarProfile
    recipe: DisplayRecipe
    condition_jaw: dict[str, float]
    world_path: Path
    source_face: Path
    root: Path

    @property
    def ok(self) -> bool:
        return bool(self.profile.validation.ok)

    def require(self) -> "AvatarBundle":
        if not self.ok:
            missing = ", ".join(self.profile.validation.missing) or "unknown"
            raise AvatarAdoptionError(
                f"avatar {self.profile.id!r} does not meet requirements: {missing}"
            )
        return self


def resolve_world_dir(world: Path | str) -> Path:
    """Accept a ``.bds`` path or a world directory."""
    path = Path(world)
    if path.is_dir():
        return path.resolve()
    return recipe_world_dir(path).resolve()


def resolve_world_file(world: Path | str) -> Path:
    """Resolve the dense ``.bds`` for a world path or directory."""
    path = Path(world)
    if path.is_file() and path.suffix.lower() == ".bds":
        return path.resolve()
    root = resolve_world_dir(path)
    candidate = root / "avatar_face.bds"
    if candidate.is_file():
        return candidate
    bds = sorted(root.glob("*.bds"))
    if bds:
        return bds[0].resolve()
    return candidate


def _discover_plates(root: Path) -> dict[str, str]:
    plates: dict[str, str] = {}
    for role in ("open", "smile", "surprise", "rest"):
        name = f"{role}.png"
        if (root / name).is_file():
            plates[role] = name
    recipe_path = root / "gpu_display_recipe.json"
    if recipe_path.is_file():
        try:
            payload = json.loads(recipe_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = {}
        found = payload.get("plates")
        if isinstance(found, dict):
            for key, value in found.items():
                if isinstance(value, str) and (root / value).is_file():
                    plates[str(key)] = value
    return plates


def _discover_artifacts(root: Path) -> dict[str, str]:
    names = (
        "region_catalog.json",
        "condition_maps.json",
        "gpu_display_recipe.json",
        "plate_atlas.json",
        "expression_catalog.json",
        "face_parts.npy",
        "face_tissue.npy",
        "capture_meta.json",
        "live_vector_model.joblib",
        "live_vector_model.meta.json",
        "amin_data_store.json",
    )
    return {name: name for name in names if (root / name).is_file()}


def _mouth_from_catalog(root: Path) -> tuple[tuple[float, float], int]:
    catalog = root / "region_catalog.json"
    if not catalog.is_file():
        return (128.0, 64.0), 0
    try:
        payload = json.loads(catalog.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return (128.0, 64.0), 0
    best_cells: list | None = None
    best_count = 0
    for region in payload.get("regions") or []:
        if not isinstance(region, dict) or region.get("name") != "mouth_unlocked":
            continue
        cells = region.get("cells") or region.get("cells_sample") or []
        count = int(region.get("cell_count", len(cells) if cells else 0))
        if count > best_count:
            best_count = count
            best_cells = cells if cells else None
    if best_count <= 0:
        return (128.0, 64.0), 0
    if best_cells:
        xs = [float(c[0]) for c in best_cells]
        ys = [float(c[1]) for c in best_cells]
        return (sum(xs) / len(xs) + 0.5, sum(ys) / len(ys) + 0.5), best_count
    return (128.0, 64.0), best_count


def _face_box_from_world(world_path: Path, grid_w: int, grid_h: int) -> dict[str, float]:
    default = {
        "x": 0.0,
        "y": 0.0,
        "width": float(grid_w),
        "height": float(grid_h),
    }
    if not world_path.is_file():
        return default
    try:
        from aiface.runtime.bds import load_bds

        header, _grid = load_bds(world_path)
    except (OSError, ValueError, ImportError):
        return default
    meta = header.get("application_metadata", {})
    box = meta.get("avatar_seed", {}).get("face_box") or {}
    if not isinstance(box, dict):
        return default
    return {
        "x": float(box.get("x", 0.0)),
        "y": float(box.get("y", 0.0)),
        "width": float(box.get("width", grid_w)),
        "height": float(box.get("height", grid_h)),
    }


def _capture_priors(root: Path) -> dict[str, float]:
    meta_path = root / "capture_meta.json"
    if not meta_path.is_file():
        return {}
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    priors = payload.get("travel_priors") or payload.get("priors") or {}
    if not isinstance(priors, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in priors.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _grid_size(world_path: Path) -> tuple[int, int]:
    if not world_path.is_file():
        return 256, 256
    try:
        from aiface.runtime.bds import load_bds

        _header, grid = load_bds(world_path)
        return int(grid.shape[1]), int(grid.shape[0])
    except (OSError, ValueError, ImportError):
        return 256, 256


def validate_avatar_root(root: Path) -> AvatarValidation:
    """Hard gates: files + mouth_unlocked coupling from digest."""
    root = Path(root)
    missing: list[str] = []
    warnings: list[str] = []
    for name in REQUIRED_FILES:
        if not (root / name).is_file():
            # Allow alternate .bds name when avatar_face.bds missing.
            if name == "avatar_face.bds" and list(root.glob("*.bds")):
                continue
            missing.append(name)
    for name in COUPLING_FILES:
        if not (root / name).is_file():
            warnings.append(f"missing optional coupling file: {name}")
    for name in OPTIONAL_FILES:
        if not (root / name).is_file():
            warnings.append(f"missing optional: {name}")

    center, mouth_count = _mouth_from_catalog(root)
    mouth_ok = mouth_count >= MIN_MOUTH_CELLS
    if not mouth_ok:
        # Catalog may sample — try live .bds cluster if catalog thin/absent.
        world = resolve_world_file(root)
        if world.is_file():
            try:
                from aiface.cell_cluster import CellClusterIndex

                index = CellClusterIndex.from_world(world)
                mouth = index.primary_mouth()
                if mouth is not None:
                    mouth_count = int(mouth.cell_count)
                    center = mouth.centroid()
                    mouth_ok = mouth_count >= MIN_MOUTH_CELLS
            except (OSError, ValueError):
                pass
    if not mouth_ok:
        missing.append(
            f"mouth_unlocked<{MIN_MOUTH_CELLS} cells "
            f"(got {mouth_count}) — retake / re-digest"
        )

    world_path = resolve_world_file(root)
    box = _face_box_from_world(world_path, 256, 256)
    if box["width"] < 8.0 or box["height"] < 8.0:
        missing.append("face_box too small or missing in BDS avatar_seed")

    ok = not missing
    return AvatarValidation(
        ok=ok,
        missing=missing,
        warnings=warnings,
        mouth_unlocked=mouth_ok,
        details={
            "mouth_cell_count": mouth_count,
            "mouth_center_grid": [float(center[0]), float(center[1])],
            "face_box": box,
            "plates": _discover_plates(root),
        },
    )


def synthesize_avatar_profile(
    world: Path | str,
    *,
    avatar_id: str | None = None,
) -> AvatarProfile:
    """Build a profile from side-cars (works before avatar_profile.json exists)."""
    root = resolve_world_dir(world)
    world_path = resolve_world_file(root)
    grid_w, grid_h = _grid_size(world_path)
    center, mouth_count = _mouth_from_catalog(root)
    if mouth_count < MIN_MOUTH_CELLS and world_path.is_file():
        try:
            from aiface.cell_cluster import CellClusterIndex

            mouth = CellClusterIndex.from_world(world_path).primary_mouth()
            if mouth is not None:
                mouth_count = int(mouth.cell_count)
                center = mouth.centroid()
        except (OSError, ValueError):
            pass
    geometry = AvatarGeometry(
        face_box=_face_box_from_world(world_path, grid_w, grid_h),
        mouth_center_grid=(float(center[0]), float(center[1])),
        mouth_cell_count=int(mouth_count),
        grid_width=grid_w,
        grid_height=grid_h,
    )
    validation = validate_avatar_root(root)
    aid = avatar_id or root.name
    plates = _discover_plates(root)
    artifacts = _discover_artifacts(root)
    return AvatarProfile(
        id=aid,
        world=world_path.name if world_path.is_file() else "avatar_face.bds",
        source_face="source_face.png",
        plates=plates,
        artifacts=artifacts,
        geometry=geometry,
        capture_priors=_capture_priors(root),
        validation=validation,
        root=root,
    )


def profile_from_payload(payload: Mapping[str, Any], *, root: Path) -> AvatarProfile:
    geo_raw = payload.get("geometry") if isinstance(payload.get("geometry"), dict) else {}
    face_box = geo_raw.get("face_box") if isinstance(geo_raw.get("face_box"), dict) else {}
    mouth = geo_raw.get("mouth_center_grid") or [128.0, 64.0]
    grid = geo_raw.get("grid") or [256, 256]
    geometry = AvatarGeometry(
        face_box={
            "x": float(face_box.get("x", 0.0)),
            "y": float(face_box.get("y", 0.0)),
            "width": float(face_box.get("width", grid[0])),
            "height": float(face_box.get("height", grid[1])),
        },
        mouth_center_grid=(float(mouth[0]), float(mouth[1])),
        mouth_cell_count=int(geo_raw.get("mouth_cell_count", 0)),
        grid_width=int(grid[0]),
        grid_height=int(grid[1]),
    )
    val_raw = payload.get("validation") if isinstance(payload.get("validation"), dict) else {}
    validation = AvatarValidation(
        ok=bool(val_raw.get("ok", False)),
        missing=list(val_raw.get("missing") or []),
        warnings=list(val_raw.get("warnings") or []),
        mouth_unlocked=bool(val_raw.get("mouth_unlocked", False)),
        details=dict(val_raw.get("details") or {}),
    )
    priors_raw = payload.get("capture_priors")
    priors: dict[str, float] = {}
    if isinstance(priors_raw, dict):
        for key, value in priors_raw.items():
            try:
                priors[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
    plates = {
        str(k): str(v)
        for k, v in (payload.get("plates") or {}).items()
        if isinstance(v, str)
    }
    artifacts = {
        str(k): str(v)
        for k, v in (payload.get("artifacts") or {}).items()
        if isinstance(v, str)
    }
    return AvatarProfile(
        id=str(payload.get("id") or root.name),
        world=str(payload.get("world") or "avatar_face.bds"),
        source_face=str(payload.get("source_face") or "source_face.png"),
        plates=plates,
        artifacts=artifacts,
        geometry=geometry,
        capture_priors=priors,
        validation=validation,
        root=root,
    )


def write_avatar_profile(
    world: Path | str,
    *,
    avatar_id: str | None = None,
) -> Path:
    """Synthesize + persist ``avatar_profile.json`` beside the world."""
    profile = synthesize_avatar_profile(world, avatar_id=avatar_id)
    # Re-validate against disk after synthesize so ok flag is current.
    assert profile.root is not None
    profile.validation = validate_avatar_root(profile.root)
    path = profile.root / PROFILE_NAME
    path.write_text(json.dumps(profile.as_dict(), indent=2), encoding="utf-8")
    return path


def load_avatar_profile(world: Path | str) -> AvatarProfile:
    """Load profile.json or synthesize from side-cars."""
    root = resolve_world_dir(world)
    path = root / PROFILE_NAME
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                profile = profile_from_payload(payload, root=root)
                # Always refresh validation against current disk.
                profile.validation = validate_avatar_root(root)
                # Refresh mouth geometry if catalog grew since write.
                center, count = _mouth_from_catalog(root)
                if count > 0:
                    profile.geometry.mouth_center_grid = (
                        float(center[0]),
                        float(center[1]),
                    )
                    profile.geometry.mouth_cell_count = int(count)
                return profile
        except (OSError, ValueError):
            pass
    return synthesize_avatar_profile(root)


def open_avatar(
    world: Path | str,
    *,
    require: bool = False,
    avatar_id: str | None = None,
) -> AvatarBundle:
    """Open any qualifying world dir into a runtime bundle."""
    root = resolve_world_dir(world)
    profile = load_avatar_profile(root)
    if avatar_id:
        profile.id = avatar_id
    world_path = resolve_world_file(root)
    if profile.world != world_path.name and world_path.is_file():
        profile.world = world_path.name
    source = root / profile.source_face
    if not source.is_file():
        alt = world_path.with_suffix(".png")
        if alt.is_file():
            source = alt
            profile.source_face = alt.name
    recipe = load_display_recipe(world_path if world_path.is_file() else root)
    jaw = load_condition_jaw(world_path if world_path.is_file() else root)
    bundle = AvatarBundle(
        profile=profile,
        recipe=recipe,
        condition_jaw=jaw,
        world_path=world_path,
        source_face=source,
        root=root,
    )
    if require:
        bundle.require()
    return bundle


def list_avatars(worlds_root: Path | str | None = None) -> list[AvatarProfile]:
    """Scan ``output/worlds/*`` (or custom root) for adoptable directories."""
    base = Path(worlds_root) if worlds_root is not None else WORLDS
    if not base.is_dir():
        return []
    found: list[AvatarProfile] = []
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        if not list(child.glob("*.bds")) and not (child / "source_face.png").is_file():
            continue
        found.append(synthesize_avatar_profile(child))
    return found


def meets_requirements(world: Path | str) -> bool:
    return validate_avatar_root(resolve_world_dir(world)).ok


__all__ = [
    "COUPLING_FILES",
    "MIN_MOUTH_CELLS",
    "OPTIONAL_FILES",
    "PROFILE_NAME",
    "PROFILE_SCHEMA",
    "REQUIRED_FILES",
    "AvatarAdoptionError",
    "AvatarBundle",
    "AvatarGeometry",
    "AvatarProfile",
    "AvatarValidation",
    "list_avatars",
    "load_avatar_profile",
    "meets_requirements",
    "open_avatar",
    "resolve_world_dir",
    "resolve_world_file",
    "synthesize_avatar_profile",
    "validate_avatar_root",
    "write_avatar_profile",
]
