"""Expression plate atlas — memory of captured mouth shapes for speech.

Plates are real frames from ``aiface-capture``. At runtime the current viseme
picks two neighbours by openness and the shader shows/blends them over the
mouth region. This is not generative fill and not NWR terrain conversion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from aiface.speech import CANONICAL_VISEMES, canonical_viseme

PLATE_ATLAS_DIR: Final = "plates"
PLATE_ATLAS_META: Final = "plate_atlas.json"
MAX_ATLAS_PLATES: Final = 8

# Target mouth openness [0,1] per viseme — indexes into the captured atlas.
# Spaced so neighbouring speech sounds land on distinct plates.
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

#: Visemes that must dwell on the open/teeth end of the atlas.
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


@dataclass(frozen=True, slots=True)
class PlateAtlas:
    plates: tuple[AtlasPlate, ...]
    viseme_openness: dict[str, float]

    def target_openness(self, viseme: str) -> float:
        key = canonical_viseme(viseme)
        return float(self.viseme_openness.get(key, VISEME_OPENNESS.get(key, 0.0)))

    def pair_for_openness(self, target: float) -> tuple[int, int, float]:
        """Return (index_a, index_b, mix) for a 0..1 openness target."""
        if not self.plates:
            return 0, 0, 0.0
        opens = [plate.openness for plate in self.plates]
        lo = min(opens)
        hi = max(opens)
        span = max(hi - lo, 1e-6)
        # Map atlas openness into 0..1 using the captured range.
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
                return i, i + 1, float(local)
        last = len(self.plates) - 1
        return last, last, 0.0

    def pair_for_viseme(self, viseme: str) -> tuple[int, int, float]:
        """Return (index_a, index_b, mix) for the current viseme."""
        return self.pair_for_openness(self.target_openness(viseme))


def default_atlas_dir(world: str | Path) -> Path:
    return Path(world).with_name(PLATE_ATLAS_DIR)


def default_atlas_meta_path(world: str | Path) -> Path:
    return Path(world).with_name(PLATE_ATLAS_META)


def teeth_visibility_score(
    image_bgr: npt.NDArray[np.uint8],
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


def select_atlas_frames(
    frames: Sequence[Any],
    *,
    count: int = MAX_ATLAS_PLATES,
) -> list[Any]:
    """Pick diverse openness samples; keep closed + toothy open extremes.

    Closed slots prefer low smile_width so the atlas does not bake a smirk into
    every phoneme. Open slots prefer teeth / jaw drop over smile width.
    """
    if not frames:
        return []
    ordered = sorted(
        frames,
        key=lambda f: (f.metrics.mouth_open, getattr(f.metrics, "teeth", 0.0)),
    )
    if len(ordered) <= count:
        # Still demote smiling closed frames to later indices when possible.
        closed = [f for f in ordered if f.metrics.mouth_open <= 0.04]
        if closed:
            best_closed = min(
                closed,
                key=lambda f: (
                    f.metrics.smile_width,
                    f.metrics.teeth,
                    -f.metrics.sharpness,
                ),
            )
            rest = [f for f in ordered if f.index != best_closed.index]
            ordered = [best_closed, *rest]
        return list(ordered)

    picks: list[Any] = []
    used: set[int] = set()

    def take(frame: Any) -> None:
        if frame.index in used:
            return
        picks.append(frame)
        used.add(frame.index)

    # Most neutral closed mouth (not the widest "closed smile").
    closed_pool = [f for f in ordered if f.metrics.mouth_open <= 0.04] or ordered[:3]
    take(
        min(
            closed_pool,
            key=lambda f: (
                f.metrics.smile_width,
                f.metrics.teeth,
                -f.metrics.sharpness,
            ),
        )
    )
    take(ordered[-1])  # most open
    # Highest teeth among the open half — enamel, not a smile beat.
    open_half = ordered[len(ordered) // 2 :]
    toothy = max(
        open_half,
        key=lambda f: (
            getattr(f.metrics, "teeth", 0.0),
            f.metrics.mouth_open,
            -f.metrics.smile_width,
        ),
    )
    take(toothy)

    for i in range(count):
        if len(picks) >= count:
            break
        t = i / max(count - 1, 1)
        idx = int(round(t * (len(ordered) - 1)))
        candidate = ordered[idx]
        # Skip smile-heavy closed frames when filling mid slots.
        if (
            candidate.metrics.mouth_open <= 0.04
            and candidate.metrics.smile_width > 0.42
            and len(picks) < count - 1
        ):
            continue
        take(candidate)

    for frame in ordered:
        if len(picks) >= count:
            break
        if frame.metrics.mouth_open <= 0.04 and frame.metrics.smile_width > 0.45:
            continue
        take(frame)

    # Last resort fill.
    for frame in ordered:
        if len(picks) >= count:
            break
        take(frame)

    return sorted(picks, key=lambda f: f.metrics.mouth_open)


def build_viseme_openness_table() -> dict[str, float]:
    table = {name: VISEME_OPENNESS.get(name, 0.35) for name in sorted(CANONICAL_VISEMES)}
    return table


def write_plate_atlas_meta(
    path: str | Path,
    plates: Sequence[AtlasPlate],
    *,
    source: str,
) -> Path:
    destination = Path(path)
    payload: dict[str, Any] = {
        "version": "plate-atlas-1.0",
        "source": source,
        "plates": [
            {
                "index": plate.index,
                "file": plate.path,
                "openness": plate.openness,
                "smile_width": plate.smile_width,
                "frame_index": plate.frame_index,
                "time_seconds": plate.time_seconds,
            }
            for plate in plates
        ],
        "viseme_openness": build_viseme_openness_table(),
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
    return PlateAtlas(plates=tuple(plates), viseme_openness=table)


__all__ = [
    "MAX_ATLAS_PLATES",
    "OPEN_TOOTH_VISEMES",
    "PLATE_ATLAS_DIR",
    "PLATE_ATLAS_META",
    "VISEME_OPENNESS",
    "AtlasPlate",
    "PlateAtlas",
    "build_viseme_openness_table",
    "default_atlas_dir",
    "default_atlas_meta_path",
    "load_plate_atlas",
    "select_atlas_frames",
    "teeth_visibility_score",
    "write_plate_atlas_meta",
]
