"""Expression plate atlas — memory of captured mouth shapes for speech.

AMIN step 13: one real video keyframe per canonical viseme (landmark-matched).
Hard snap (step 12): speech shows a single plate, not a 50/50 ghost blend.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import numpy as np

from aiface.speech import CANONICAL_VISEMES, canonical_viseme

PLATE_ATLAS_DIR: Final = "plates"
PLATE_ATLAS_META: Final = "plate_atlas.json"
MAX_ATLAS_PLATES: Final = 16
HARD_SNAP_THRESHOLD: Final = 0.75

# Target mouth openness [0,1] per viseme — indexes into the captured atlas.
VISEME_OPENNESS: Final[dict[str, float]] = {
    "REST": 0.00,
    "CLOSED": 0.00,
    "PP": 0.00,
    "FF": 0.18,
    "TH": 0.22,
    "DD": 0.28,
    "KK": 0.30,
    "CH": 0.40,
    "SS": 0.32,
    "NN": 0.26,
    "RR": 0.42,
    "IH": 0.55,
    "EH": 0.68,
    "EE": 0.72,
    "OH": 0.88,
    "OU": 0.82,
    "AA": 1.00,
    "AH": 1.00,
}

# Relative smile width target in the capture's observed smile range [0,1].
VISEME_SMILE: Final[dict[str, float]] = {
    "REST": 0.15,
    "CLOSED": 0.10,
    "PP": 0.10,
    "FF": 0.20,
    "TH": 0.25,
    "DD": 0.25,
    "KK": 0.25,
    "CH": 0.30,
    "SS": 0.35,
    "NN": 0.25,
    "RR": 0.30,
    "IH": 0.45,
    "EH": 0.50,
    "EE": 0.85,
    "OH": 0.35,
    "OU": 0.30,
    "AA": 0.40,
    "AH": 0.40,
}

OPEN_TOOTH_VISEMES: Final[frozenset[str]] = frozenset(
    {"AA", "AH", "OH", "OU", "EH", "EE", "IH"}
)


@dataclass(frozen=True, slots=True)
class AtlasPlate:
    index: int
    path: str
    openness: float
    smile_width: float
    frame_index: int
    time_seconds: float
    viseme: str = ""


@dataclass(frozen=True, slots=True)
class PlateAtlas:
    plates: tuple[AtlasPlate, ...]
    viseme_openness: dict[str, float]
    viseme_to_plate: dict[str, int] = field(default_factory=dict)

    def target_openness(self, viseme: str) -> float:
        key = canonical_viseme(viseme)
        return float(self.viseme_openness.get(key, VISEME_OPENNESS.get(key, 0.0)))

    def plate_index_for_viseme(self, viseme: str) -> int | None:
        key = canonical_viseme(viseme)
        if key in self.viseme_to_plate:
            idx = int(self.viseme_to_plate[key])
            if 0 <= idx < len(self.plates):
                return idx
        return None

    def pair_for_openness(
        self,
        target: float,
        *,
        hard_snap: bool = False,
    ) -> tuple[int, int, float]:
        """Return (index_a, index_b, mix) for a 0..1 openness target."""
        if not self.plates:
            return 0, 0, 0.0
        opens = [plate.openness for plate in self.plates]
        lo = min(opens)
        hi = max(opens)
        span = max(hi - lo, 1e-6)
        norm = [(o - lo) / span for o in opens]
        goal = float(np.clip(target, 0.0, 1.0))
        if goal <= norm[0]:
            return 0, 0, 0.0
        if goal >= norm[-1]:
            last = len(self.plates) - 1
            return last, last, 0.0
        for i in range(len(norm) - 1):
            if norm[i] <= goal <= norm[i + 1]:
                local = (goal - norm[i]) / max(norm[i + 1] - norm[i], 1e-6)
                if hard_snap:
                    nearest = i if local < 0.5 else i + 1
                    return nearest, nearest, 0.0
                return i, i + 1, float(local)
        last = len(self.plates) - 1
        return last, last, 0.0

    def pair_for_viseme(
        self,
        viseme: str,
        *,
        hard_snap: bool = True,
    ) -> tuple[int, int, float]:
        """Viseme → plate pair. Hard snap uses assigned bank index (mix=0)."""
        assigned = self.plate_index_for_viseme(viseme)
        if assigned is not None and hard_snap:
            return assigned, assigned, 0.0
        ia, ib, mix = self.pair_for_openness(
            self.target_openness(viseme), hard_snap=hard_snap
        )
        return ia, ib, mix


def default_atlas_dir(world: str | Path) -> Path:
    return Path(world).with_name(PLATE_ATLAS_DIR)


def default_atlas_meta_path(world: str | Path) -> Path:
    return Path(world).with_name(PLATE_ATLAS_META)


def teeth_visibility_score(
    image_bgr: Any,
    mouth_xy: tuple[float, float],
    face_width: float,
    face_height: float,
) -> float:
    """Fraction of bright, low-saturation pixels in the mouth band (teeth proxy)."""
    from aiface.seed import _cv2

    cv2 = _cv2()
    height, width = image_bgr.shape[:2]
    mx, my = mouth_xy
    half_w = max(4, int(face_width * 0.16))
    half_h = max(3, int(face_height * 0.07))
    x0, x1 = max(0, int(mx) - half_w), min(width, int(mx) + half_w)
    y0, y1 = max(0, int(my) - half_h), min(height, int(my) + half_h)
    patch = image_bgr[y0:y1, x0:x1]
    if patch.size == 0:
        return 1.0
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).astype(np.float32)
    value = hsv[..., 2]
    sat = hsv[..., 1]
    teeth = (value > 165.0) & (sat < 70.0)
    return float(np.mean(teeth.astype(np.float32)))


def _normalized_metrics(frames: Sequence[Any]) -> tuple[float, float, float, float]:
    opens = [float(f.metrics.mouth_open) for f in frames]
    smiles = [float(f.metrics.smile_width) for f in frames]
    return (
        min(opens),
        max(opens),
        min(smiles),
        max(smiles),
    )


def score_frame_for_viseme(
    frame: Any,
    viseme: str,
    *,
    open_lo: float,
    open_hi: float,
    smile_lo: float,
    smile_hi: float,
) -> float:
    """Lower is better — landmark distance to the viseme's target look."""
    key = canonical_viseme(viseme)
    open_span = max(open_hi - open_lo, 1e-6)
    smile_span = max(smile_hi - smile_lo, 1e-6)
    open_n = (float(frame.metrics.mouth_open) - open_lo) / open_span
    smile_n = (float(frame.metrics.smile_width) - smile_lo) / smile_span
    open_n = float(np.clip(open_n, 0.0, 1.0))
    smile_n = float(np.clip(smile_n, 0.0, 1.0))
    target_o = float(VISEME_OPENNESS.get(key, 0.35))
    target_s = float(VISEME_SMILE.get(key, 0.3))
    teeth = float(getattr(frame.metrics, "teeth", 0.0))
    sharp = float(getattr(frame.metrics, "sharpness", 0.0))
    teeth_bonus = 0.0
    if key in OPEN_TOOTH_VISEMES:
        teeth_bonus = -0.10 * teeth
    elif key == "TH":
        # Prefer a slight aperture + teeth hint (tongue/teeth edge) over DD.
        teeth_bonus = -0.12 * teeth
        if 0.08 <= open_n <= 0.35:
            teeth_bonus -= 0.08
    elif key == "FF":
        if 0.05 <= open_n <= 0.28:
            teeth_bonus -= 0.06
    # Prefer non-smiling closed frames for REST/PP — identity take is soft-smile.
    smile_penalty = 0.0
    if key in {"REST", "CLOSED", "PP"}:
        smile_penalty = 0.55 * max(0.0, smile_n - 0.35)
        # Strongly prefer the flattest closed aperture available.
        smile_penalty += 0.40 * open_n
    return (
        abs(open_n - target_o) * 1.45
        + abs(smile_n - target_s) * 0.95
        + smile_penalty
        + teeth_bonus
        - 0.0006 * sharp
    )


def match_visemes_to_frames(
    frames: Sequence[Any],
) -> dict[str, Any]:
    """Best landmark-matched frame per canonical viseme (may reuse frames)."""
    if not frames:
        return {}
    open_lo, open_hi, smile_lo, smile_hi = _normalized_metrics(frames)
    mapping: dict[str, Any] = {}
    for viseme in sorted(CANONICAL_VISEMES):
        best = min(
            frames,
            key=lambda f, v=viseme: score_frame_for_viseme(
                f,
                v,
                open_lo=open_lo,
                open_hi=open_hi,
                smile_lo=smile_lo,
                smile_hi=smile_hi,
            ),
        )
        mapping[viseme] = best
    return mapping


# Must-keep lip-reading shapes when the atlas is capped.
PRIORITY_ATLAS_VISEMES: Final[tuple[str, ...]] = (
    "CLOSED",
    "PP",
    "FF",
    "TH",
    "SS",
    "EE",
    "OH",
    "AA",
)


def select_viseme_atlas_frames(
    frames: Sequence[Any],
    *,
    max_plates: int = MAX_ATLAS_PLATES,
) -> tuple[list[Any], dict[str, int]]:
    """AMIN step 13: unique frames for visemes, capped, with viseme→index map."""
    matched = match_visemes_to_frames(frames)
    if not matched:
        return [], {}

    # Prefer unique frames ordered by openness; keep REST/closed first.
    by_index: dict[int, Any] = {}
    for viseme, frame in matched.items():
        by_index.setdefault(int(frame.index), frame)

    unique = sorted(
        by_index.values(),
        key=lambda f: (f.metrics.mouth_open, f.metrics.smile_width),
    )
    # Reserve priority lip-reading visemes on *distinct* frames when possible.
    open_lo, open_hi, smile_lo, smile_hi = _normalized_metrics(frames)
    keep: dict[int, Any] = {}
    priority_frame: dict[str, Any] = {}
    for viseme in PRIORITY_ATLAS_VISEMES:
        ranked = sorted(
            frames,
            key=lambda f, v=viseme: score_frame_for_viseme(
                f,
                v,
                open_lo=open_lo,
                open_hi=open_hi,
                smile_lo=smile_lo,
                smile_hi=smile_hi,
            ),
        )
        chosen = None
        for cand in ranked:
            if int(cand.index) not in keep:
                chosen = cand
                break
        if chosen is None and ranked:
            chosen = ranked[0]
        if chosen is not None:
            keep[int(chosen.index)] = chosen
            priority_frame[viseme] = chosen
        if len(keep) >= max_plates:
            break
    if len(keep) < max_plates:
        for frame in unique:
            keep.setdefault(int(frame.index), frame)
            if len(keep) >= max_plates:
                break
    unique = sorted(
        keep.values(),
        key=lambda f: (f.metrics.mouth_open, f.metrics.smile_width),
    )

    index_by_frame = {int(f.index): i for i, f in enumerate(unique)}
    viseme_to_plate: dict[str, int] = {}
    for viseme, frame in priority_frame.items():
        if int(frame.index) in index_by_frame:
            viseme_to_plate[viseme] = index_by_frame[int(frame.index)]
    for viseme, frame in matched.items():
        if viseme in viseme_to_plate:
            continue
        if int(frame.index) in index_by_frame:
            viseme_to_plate[viseme] = index_by_frame[int(frame.index)]
            continue
        best_i = min(
            range(len(unique)),
            key=lambda i: score_frame_for_viseme(
                unique[i],
                viseme,
                open_lo=open_lo,
                open_hi=open_hi,
                smile_lo=smile_lo,
                smile_hi=smile_hi,
            ),
        )
        viseme_to_plate[viseme] = best_i
    # Explicit aliases for runtime canonicalisation.
    if "PP" in viseme_to_plate:
        viseme_to_plate.setdefault("MM", int(viseme_to_plate["PP"]))
    if "CLOSED" in viseme_to_plate and "REST" not in viseme_to_plate:
        viseme_to_plate["REST"] = int(viseme_to_plate["CLOSED"])
    return unique, viseme_to_plate


def select_atlas_frames(
    frames: Sequence[Any],
    *,
    count: int = MAX_ATLAS_PLATES,
) -> list[Any]:
    """Legacy openness-bin picker; prefer ``select_viseme_atlas_frames``."""
    chosen, _mapping = select_viseme_atlas_frames(frames, max_plates=count)
    if chosen:
        return chosen
    if not frames:
        return []
    ordered = sorted(frames, key=lambda f: f.metrics.mouth_open)
    return list(ordered[:count])


def build_viseme_openness_table() -> dict[str, float]:
    return {name: VISEME_OPENNESS.get(name, 0.35) for name in sorted(CANONICAL_VISEMES)}


def write_plate_atlas_meta(
    path: str | Path,
    plates: Sequence[AtlasPlate],
    *,
    source: str,
    viseme_to_plate: Mapping[str, int] | None = None,
) -> Path:
    destination = Path(path)
    mapping = (
        {str(k): int(v) for k, v in viseme_to_plate.items()}
        if viseme_to_plate is not None
        else {}
    )
    if not mapping and plates:
        # Infer from plate.viseme tags when provided.
        for plate in plates:
            if plate.viseme:
                mapping[canonical_viseme(plate.viseme)] = int(plate.index)
    payload: dict[str, Any] = {
        "version": "plate-atlas-1.1",
        "source": source,
        "plates": [
            {
                "index": plate.index,
                "file": plate.path,
                "openness": plate.openness,
                "smile_width": plate.smile_width,
                "frame_index": plate.frame_index,
                "time_seconds": plate.time_seconds,
                "viseme": plate.viseme,
            }
            for plate in plates
        ],
        "viseme_openness": build_viseme_openness_table(),
        "viseme_to_plate": mapping,
    }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination


def load_plate_atlas(world: str | Path) -> PlateAtlas | None:
    meta_path = default_atlas_meta_path(world)
    if not meta_path.is_file():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    raw_plates = payload.get("plates") or []
    plates: list[AtlasPlate] = []
    for item in raw_plates:
        if not isinstance(item, Mapping):
            continue
        plates.append(
            AtlasPlate(
                index=int(item.get("index", len(plates))),
                path=str(item.get("file", "")),
                openness=float(item.get("openness", 0.0)),
                smile_width=float(item.get("smile_width", 0.0)),
                frame_index=int(item.get("frame_index", 0)),
                time_seconds=float(item.get("time_seconds", 0.0)),
                viseme=str(item.get("viseme", "") or ""),
            )
        )
    if not plates:
        return None
    viseme = payload.get("viseme_openness")
    table = (
        {str(k): float(v) for k, v in viseme.items()}
        if isinstance(viseme, Mapping)
        else build_viseme_openness_table()
    )
    raw_map = payload.get("viseme_to_plate") or {}
    viseme_to_plate: dict[str, int] = {}
    if isinstance(raw_map, Mapping):
        for key, value in raw_map.items():
            try:
                viseme_to_plate[canonical_viseme(str(key))] = int(value)
            except (TypeError, ValueError):
                continue
    return PlateAtlas(
        plates=tuple(plates),
        viseme_openness=table,
        viseme_to_plate=viseme_to_plate,
    )


__all__ = [
    "HARD_SNAP_THRESHOLD",
    "MAX_ATLAS_PLATES",
    "OPEN_TOOTH_VISEMES",
    "PLATE_ATLAS_DIR",
    "PLATE_ATLAS_META",
    "VISEME_OPENNESS",
    "VISEME_SMILE",
    "AtlasPlate",
    "PlateAtlas",
    "build_viseme_openness_table",
    "default_atlas_dir",
    "default_atlas_meta_path",
    "load_plate_atlas",
    "match_visemes_to_frames",
    "score_frame_for_viseme",
    "select_atlas_frames",
    "select_viseme_atlas_frames",
    "teeth_visibility_score",
    "write_plate_atlas_meta",
]
