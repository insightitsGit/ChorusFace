"""Build measured avatar observations from world plates + catalog + .bds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from aiface.behavior.schema import CONTROL_DIM, CONTROL_NAMES, landmarks_to_controls
from aiface.observation.schema import (
    LOOK_ROLES,
    AvatarObservationSet,
    CellGeometryObs,
    GpuLookVector,
    LookObservation,
    PlateDelta,
    obs_json_path,
    obs_npz_path,
)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _mouth_roi_stats(rgba: np.ndarray) -> tuple[tuple[float, float, float], float]:
    """Mean RGB + edge energy on the lower-mid face (mouth band)."""
    if rgba.ndim != 3 or rgba.shape[2] < 3:
        return (0.0, 0.0, 0.0), 0.0
    h, w = int(rgba.shape[0]), int(rgba.shape[1])
    y0, y1 = int(h * 0.55), int(h * 0.88)
    x0, x1 = int(w * 0.25), int(w * 0.75)
    patch = rgba[y0:y1, x0:x1, :3].astype(np.float64)
    if patch.size == 0:
        return (0.0, 0.0, 0.0), 0.0
    mean = tuple(float(v) for v in patch.reshape(-1, 3).mean(axis=0))
    # Simple horizontal gradient energy (lip edges show up here).
    gx = np.abs(np.diff(patch, axis=1)).mean() if patch.shape[1] > 1 else 0.0
    gy = np.abs(np.diff(patch, axis=0)).mean() if patch.shape[0] > 1 else 0.0
    energy = float(gx + gy)
    return mean, energy  # type: ignore[return-value]


def _load_plate_rgba(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        from PIL import Image

        image = Image.open(path).convert("RGBA")
        return np.asarray(image, dtype=np.float32) / 255.0
    except (OSError, ValueError):
        return None


def _plate_delta(role_rgba: np.ndarray | None, rest_rgba: np.ndarray | None) -> PlateDelta:
    if role_rgba is None:
        return PlateDelta()
    mean, energy = _mouth_roi_stats(role_rgba)
    if rest_rgba is None:
        return PlateDelta(mean_rgb=mean, mouth_energy=energy)
    rest_mean, _ = _mouth_roi_stats(rest_rgba)
    delta = tuple(float(a - b) for a, b in zip(mean, rest_mean, strict=True))
    luma = float(0.299 * delta[0] + 0.587 * delta[1] + 0.114 * delta[2])
    return PlateDelta(
        mean_rgb=mean,
        delta_rgb=delta,  # type: ignore[arg-type]
        delta_luma=luma,
        mouth_energy=energy,
    )


def _normalize_pair(value: float, rest: float, peak: float) -> float:
    span = max(float(peak) - float(rest), 1e-4)
    return float(np.clip((float(value) - float(rest)) / span, 0.0, 1.0))


def _gpu_for_role(
    role: str,
    *,
    open_n: float,
    smile_n: float,
    brow: float,
    plate_name: str,
) -> GpuLookVector:
    """Map a measured look to the uniforms avatar.frag actually reads."""
    if role == "smile":
        return GpuLookVector(
            smile_drive=max(smile_n, 0.85),
            open_drive=0.0,
            jaw=0.0,
            atlas_amount=0.0,
            expr_blend=0.0,
            brow_raise=float(brow),
            plate_role="smile",
            plate_texture=plate_name or "smile.png",
        )
    if role == "open":
        return GpuLookVector(
            smile_drive=0.0,
            open_drive=max(open_n, 0.85),
            jaw=max(open_n, 0.85),
            atlas_amount=max(open_n, 0.85),
            expr_blend=0.0,
            brow_raise=float(brow),
            plate_role="open",
            plate_texture=plate_name or "open.png",
        )
    if role == "surprise":
        return GpuLookVector(
            smile_drive=0.0,
            open_drive=float(open_n) * 0.5,
            jaw=float(open_n) * 0.5,
            atlas_amount=0.0,
            expr_blend=max(float(brow), 0.7),
            brow_raise=max(float(brow), 0.55),
            plate_role="surprise",
            plate_texture=plate_name or "surprise.png",
        )
    return GpuLookVector(
        smile_drive=0.0,
        open_drive=0.0,
        jaw=0.0,
        atlas_amount=0.0,
        expr_blend=0.0,
        brow_raise=0.0,
        plate_role="rest",
        plate_texture=plate_name or "source_face.png",
    )


def _cell_geometry(world: Path) -> CellGeometryObs:
    bds = world / "avatar_face.bds"
    if not bds.is_file():
        matches = list(world.glob("*.bds"))
        bds = matches[0] if matches else bds
    if not bds.is_file():
        return CellGeometryObs()
    try:
        from aiface.cell_cluster import CellClusterIndex
        from aiface.mouth_cell_plan import detect_mouth_cells
        from aiface.mouth_groups import build_mouth_group_plan

        index = CellClusterIndex.from_world(bds)
        mouth = index.primary_mouth()
        if mouth is None:
            return CellGeometryObs()
        cells = detect_mouth_cells(mouth)
        groups = build_mouth_group_plan(cells)
        counts = groups.groups.counts()
        cx, cy = mouth.centroid()
        return CellGeometryObs(
            mouth_cell_count=int(mouth.cell_count),
            centroid=(float(cx), float(cy)),
            group_counts=counts,
        )
    except (OSError, ValueError, ImportError) as exc:
        print(f"observation: cell geometry skipped ({exc})")
        return CellGeometryObs()


def extract_avatar_observations(world: Path | str) -> AvatarObservationSet:
    """Read plates + expression catalog + .bds → measured look vectors."""
    root = Path(world)
    root = root if root.is_dir() else root.parent
    catalog = _load_json(root / "expression_catalog.json")
    roles = catalog.get("roles") if isinstance(catalog.get("roles"), dict) else {}
    capture = _load_json(root / "capture_meta.json")
    video = str(catalog.get("source") or capture.get("video") or "")

    rest_meta = roles.get("rest") if isinstance(roles.get("rest"), dict) else {}
    rest_width = float(rest_meta.get("smile_width", 0.35))
    rest_open = float(rest_meta.get("mouth_open", 0.0))
    # Peaks from roles for normalization.
    peak_width = rest_width
    peak_open = rest_open
    for name in LOOK_ROLES:
        entry = roles.get(name)
        if isinstance(entry, dict):
            peak_width = max(peak_width, float(entry.get("smile_width", 0.0)))
            peak_open = max(peak_open, float(entry.get("mouth_open", 0.0)))

    rest_rgba = _load_plate_rgba(root / "source_face.png")
    looks: list[LookObservation] = []
    rest_controls: list[float] | None = None

    for role in LOOK_ROLES:
        entry = roles.get(role) if isinstance(roles.get(role), dict) else {}
        plate_name = str(entry.get("plate") or {
            "rest": "source_face.png",
            "smile": "smile.png",
            "open": "open.png",
            "surprise": "surprise.png",
        }.get(role, "source_face.png"))
        mouth_open = float(entry.get("mouth_open", 0.0))
        smile_width = float(entry.get("smile_width", rest_width))
        teeth = float(entry.get("teeth", 0.0))
        brow = float(entry.get("brow_raise", 0.0))
        lid = float(entry.get("eye_widen", entry.get("lid_open", 0.0)))
        open_n = _normalize_pair(mouth_open, rest_open, peak_open) if role != "rest" else 0.0
        smile_n = _normalize_pair(smile_width, rest_width, peak_width) if role != "rest" else 0.0
        if role == "smile" and smile_n < 0.5:
            smile_n = 0.85  # capture selected this frame — treat as full smile look
        if role == "open" and open_n < 0.5:
            open_n = 0.85
        controls = landmarks_to_controls(
            openness_n=open_n if role != "smile" else 0.0,
            width_n=smile_n if role == "smile" else (
                _normalize_pair(smile_width, rest_width, max(peak_width, rest_width + 0.05))
                if role != "rest"
                else 0.35
            ),
            teeth_n=float(np.clip(teeth * 4.0, 0.0, 1.0)) if role == "open" else (
                open_n * 0.65
            ),
        )
        if role == "smile":
            # Closed-lip smile: width/corners dominate; open/cavity stay near 0.
            controls = landmarks_to_controls(
                openness_n=0.0, width_n=max(smile_n, 0.85), teeth_n=float(teeth)
            )
            controls[5] = max(controls[5], 0.85)  # corner_dx
        if role == "rest":
            controls = landmarks_to_controls(openness_n=0.0, width_n=0.35, teeth_n=0.0)
            controls[0] = controls[1] = controls[3] = controls[4] = 0.0
            controls[6] = controls[7] = 0.0
            rest_controls = list(controls)

        rgba = _load_plate_rgba(root / plate_name)
        plate = _plate_delta(rgba, rest_rgba)
        gpu = _gpu_for_role(
            role,
            open_n=open_n,
            smile_n=smile_n,
            brow=brow,
            plate_name=plate_name,
        )
        delta = [0.0] * CONTROL_DIM
        if rest_controls is not None and role != "rest":
            delta = [float(a - b) for a, b in zip(controls, rest_controls, strict=True)]
        looks.append(
            LookObservation(
                role=role,
                time_seconds=float(entry.get("time_seconds", 0.0)),
                frame_index=int(entry.get("frame_index", -1)),
                mouth_open=mouth_open,
                smile_width=smile_width,
                teeth=teeth,
                brow_raise=brow,
                lid_open=lid,
                gpu=gpu,
                plate=plate,
                controls=tuple(controls),
                delta_from_rest=tuple(delta),
                notes=str(entry.get("notes") or f"measured {role} look"),
            )
        )

    # Append talk_series samples as time-keyed observations (landmark truth).
    talk = capture.get("talk_series") if isinstance(capture.get("talk_series"), list) else []
    if rest_controls is None:
        rest_controls = landmarks_to_controls(openness_n=0.0, width_n=0.35, teeth_n=0.0)
    for row in talk[:120]:
        if not isinstance(row, dict):
            continue
        mouth_open = float(row.get("mouth_open", 0.0))
        smile_width = float(row.get("smile_width", rest_width))
        teeth = float(row.get("teeth", 0.0))
        brow = float(row.get("brow_raise", 0.0))
        lid = float(row.get("lid_open", 0.0))
        open_n = _normalize_pair(mouth_open, rest_open, max(peak_open, rest_open + 0.05))
        smile_n = _normalize_pair(smile_width, rest_width, max(peak_width, rest_width + 0.05))
        controls = landmarks_to_controls(
            openness_n=open_n, width_n=max(smile_n, 0.35), teeth_n=float(np.clip(teeth * 4.0, 0.0, 1.0))
        )
        delta = [float(a - b) for a, b in zip(controls, rest_controls, strict=True)]
        # GPU vector follows measured open/smile at this talk sample.
        gpu = GpuLookVector(
            smile_drive=float(smile_n) if open_n < 0.15 else 0.0,
            open_drive=float(open_n),
            jaw=float(open_n),
            atlas_amount=float(open_n),
            expr_blend=0.0,
            brow_raise=float(brow),
            plate_role="talk",
            plate_texture="atlas" if open_n > 0.2 else "source_face.png",
        )
        looks.append(
            LookObservation(
                role="talk",
                time_seconds=float(row.get("t", row.get("time_seconds", 0.0))),
                frame_index=int(row.get("index", row.get("frame_index", -1))),
                mouth_open=mouth_open,
                smile_width=smile_width,
                teeth=teeth,
                brow_raise=brow,
                lid_open=lid,
                gpu=gpu,
                plate=PlateDelta(),
                controls=tuple(controls),
                delta_from_rest=tuple(delta),
                notes="talk_series landmark sample",
            )
        )

    cells = _cell_geometry(root)
    smile_look = next((x for x in looks if x.role == "smile"), None)
    open_look = next((x for x in looks if x.role == "open"), None)
    return AvatarObservationSet(
        looks=looks,
        cells=cells,
        smile_vector=list(smile_look.delta_from_rest) if smile_look else [],
        open_vector=list(open_look.delta_from_rest) if open_look else [],
        video=video,
        root=root,
    )


def save_avatar_observations(
    world: Path | str, obs: AvatarObservationSet | None = None
) -> Path:
    """Persist observations JSON + NPZ beside the world."""
    root = Path(world)
    root = root if root.is_dir() else root.parent
    obs = obs or extract_avatar_observations(root)
    path = obs_json_path(root)
    path.write_text(json.dumps(obs.as_dict(), indent=2), encoding="utf-8")

    # Compact arrays for ML / probes.
    keyframes = [look for look in obs.looks if look.role in LOOK_ROLES]
    roles = [look.role for look in keyframes]
    controls = np.asarray(
        [list(look.controls) if look.controls else [0.0] * CONTROL_DIM for look in keyframes],
        dtype=np.float64,
    )
    deltas = np.asarray(
        [
            list(look.delta_from_rest)
            if look.delta_from_rest
            else [0.0] * CONTROL_DIM
            for look in keyframes
        ],
        dtype=np.float64,
    )
    gpu = np.asarray([look.gpu.as_vector() for look in keyframes], dtype=np.float64)
    talk = [look for look in obs.looks if look.role == "talk"]
    talk_t = np.asarray([look.time_seconds for look in talk], dtype=np.float64)
    talk_ctrl = np.asarray(
        [list(look.controls) if look.controls else [0.0] * CONTROL_DIM for look in talk],
        dtype=np.float64,
    ) if talk else np.zeros((0, CONTROL_DIM), dtype=np.float64)
    talk_gpu = np.asarray(
        [look.gpu.as_vector() for look in talk], dtype=np.float64
    ) if talk else np.zeros((0, 6), dtype=np.float64)

    np.savez_compressed(
        obs_npz_path(root),
        roles=np.asarray(roles),
        controls=controls,
        deltas=deltas,
        gpu_vectors=gpu,
        smile_vector=np.asarray(obs.smile_vector, dtype=np.float64),
        open_vector=np.asarray(obs.open_vector, dtype=np.float64),
        talk_times=talk_t,
        talk_controls=talk_ctrl,
        talk_gpu=talk_gpu,
        control_names=np.asarray(list(CONTROL_NAMES)),
        mouth_cell_count=np.asarray([obs.cells.mouth_cell_count]),
    )
    print(
        f"observation: wrote {path.name} "
        f"(looks={len(keyframes)} talk={len(talk)} "
        f"smile_vector={len(obs.smile_vector)} cells={obs.cells.mouth_cell_count})"
    )
    return path


def load_avatar_observations(world: Path | str) -> AvatarObservationSet | None:
    path = obs_json_path(world)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    # Reconstruct minimal set used at runtime (keyframes + vectors).
    looks: list[LookObservation] = []
    for row in payload.get("looks") or []:
        if not isinstance(row, dict):
            continue
        gpu_raw = row.get("gpu") if isinstance(row.get("gpu"), dict) else {}
        plate_raw = row.get("plate") if isinstance(row.get("plate"), dict) else {}
        lm = row.get("landmarks") if isinstance(row.get("landmarks"), dict) else {}
        looks.append(
            LookObservation(
                role=str(row.get("role") or "rest"),
                time_seconds=float(row.get("time_seconds", 0.0)),
                frame_index=int(row.get("frame_index", -1)),
                mouth_open=float(lm.get("mouth_open", 0.0)),
                smile_width=float(lm.get("smile_width", 0.0)),
                teeth=float(lm.get("teeth", 0.0)),
                brow_raise=float(lm.get("brow_raise", 0.0)),
                lid_open=float(lm.get("lid_open", 0.0)),
                gpu=GpuLookVector(
                    smile_drive=float(gpu_raw.get("smile_drive", 0.0)),
                    open_drive=float(gpu_raw.get("open_drive", 0.0)),
                    jaw=float(gpu_raw.get("jaw", 0.0)),
                    atlas_amount=float(gpu_raw.get("atlas_amount", 0.0)),
                    expr_blend=float(gpu_raw.get("expr_blend", 0.0)),
                    brow_raise=float(gpu_raw.get("brow_raise", 0.0)),
                    plate_role=str(gpu_raw.get("plate_role") or "rest"),
                    plate_texture=str(gpu_raw.get("plate_texture") or ""),
                ),
                plate=PlateDelta(
                    mean_rgb=tuple(plate_raw.get("mean_rgb") or (0.0, 0.0, 0.0)),  # type: ignore[arg-type]
                    delta_rgb=tuple(plate_raw.get("delta_rgb") or (0.0, 0.0, 0.0)),  # type: ignore[arg-type]
                    delta_luma=float(plate_raw.get("delta_luma", 0.0)),
                    mouth_energy=float(plate_raw.get("mouth_energy", 0.0)),
                ),
                controls=tuple(float(v) for v in (row.get("controls") or [])),
                delta_from_rest=tuple(float(v) for v in (row.get("delta_from_rest") or [])),
                notes=str(row.get("notes") or ""),
            )
        )
    cells_raw = payload.get("cells") if isinstance(payload.get("cells"), dict) else {}
    centroid = cells_raw.get("centroid") or [0.0, 0.0]
    return AvatarObservationSet(
        looks=looks,
        cells=CellGeometryObs(
            mouth_cell_count=int(cells_raw.get("mouth_cell_count", 0)),
            centroid=(float(centroid[0]), float(centroid[1])),
            group_counts=dict(cells_raw.get("group_counts") or {}),
        ),
        smile_vector=[float(v) for v in (payload.get("smile_vector") or [])],
        open_vector=[float(v) for v in (payload.get("open_vector") or [])],
        video=str(payload.get("video") or ""),
        root=path.parent,
    )


__all__ = [
    "extract_avatar_observations",
    "load_avatar_observations",
    "save_avatar_observations",
]
