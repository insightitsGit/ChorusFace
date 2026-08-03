"""Side A ingest debug — dump every TickPackage posted to the GPU master.

Enable with ``AIFACE_TICKFEED_DEBUG=1`` or ``--tickfeed-debug``. Writes JSONL to
``output/previews/tickfeed_side_a.jsonl`` so blur/ghost frames can be matched to
exact LOOK labels + FIELD patch stats + effective warp gain.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from aiface.paths import PREVIEWS
from aiface.tickfeed.package import TickPackage
from aiface.tickfeed.schema import DeltaEncoding, PackageKind

DEFAULT_LOG = PREVIEWS / "tickfeed_side_a.jsonl"


def _kind_name(pkg: TickPackage | None) -> str:
    if pkg is None:
        return "MISS"
    try:
        return PackageKind(int(pkg.kind)).name
    except ValueError:
        return f"KIND_{int(pkg.kind)}"


def _enc_name(pkg: TickPackage | None) -> str:
    if pkg is None:
        return "NONE"
    try:
        return DeltaEncoding(int(pkg.delta_encoding)).name
    except ValueError:
        return f"ENC_{int(pkg.delta_encoding)}"


def field_patch_stats(
    values: NDArray[np.floating] | None,
) -> dict[str, float | int]:
    """Compact FIELD metrics for one face patch (H,W,2)."""
    if values is None:
        return {
            "n": 0,
            "max": 0.0,
            "mean": 0.0,
            "p95": 0.0,
            "mouth_max": 0.0,
            "upper_vy_mean": 0.0,
            "lower_vy_mean": 0.0,
            "sep_l_minus_u": 0.0,
            "nonzero_frac": 0.0,
        }
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] < 2:
        return field_patch_stats(None)
    h, w, _ = arr.shape
    mag = np.linalg.norm(arr[..., :2], axis=-1)
    my = int(h * 0.58)
    mouth = arr[max(0, my - 10) : my + 22, w // 2 - 28 : w // 2 + 28]
    if mouth.size == 0:
        mouth = arr
    mid = mouth.shape[0] // 2
    upper = mouth[:mid, :, 1]
    lower = mouth[mid:, :, 1]
    flat = mag.reshape(-1)
    return {
        "n": int(flat.size),
        "max": float(flat.max()) if flat.size else 0.0,
        "mean": float(flat.mean()) if flat.size else 0.0,
        "p95": float(np.percentile(flat, 95)) if flat.size else 0.0,
        "mouth_max": float(np.linalg.norm(mouth[..., :2], axis=-1).max())
        if mouth.size
        else 0.0,
        "upper_vy_mean": float(upper.mean()) if upper.size else 0.0,
        "lower_vy_mean": float(lower.mean()) if lower.size else 0.0,
        "sep_l_minus_u": float(lower.mean() - upper.mean())
        if upper.size and lower.size
        else 0.0,
        "nonzero_frac": float((flat > 0.02).mean()) if flat.size else 0.0,
    }


def blur_risk(
    *,
    plate_open: float,
    smile: float,
    field_max: float,
    field_gain_eff: float,
    muscles: int,
) -> list[str]:
    """Heuristic flags for known blurry / ghost-lip failure modes."""
    flags: list[str] = []
    travel = float(field_max) * float(field_gain_eff)
    if plate_open >= 0.45 and travel >= 0.08:
        flags.append("plate+field_stack")
    # Mid-open transitions are the smear zone: plate half-on + FIELD still hot.
    if 0.12 <= plate_open < 0.45 and travel >= 0.12:
        flags.append("transition_plate+field")
    if plate_open >= 0.35 and smile >= 0.2:
        flags.append("open+smile_stack")
    if muscles > 0 and (plate_open >= 0.2 or travel >= 0.05):
        flags.append("muscle_stack")
    if plate_open >= 0.6 and field_gain_eff >= 0.15:
        flags.append("open_plate_field_not_muted")
    if field_max >= 0.4 and field_gain_eff < 0.08 and plate_open < 0.2:
        flags.append("field_hot_gain_muted_no_plate")
    if plate_open >= 0.5 and field_max < 0.05:
        flags.append("plate_only_open")
    return flags


@dataclass
class SideADebugLog:
    """Append-only JSONL logger + in-memory ring for bridge/status."""

    path: Path = field(default_factory=lambda: DEFAULT_LOG)
    enabled: bool = False
    every_n: int = 1
    ring_size: int = 240
    _handle: Any = None
    _count: int = 0
    _ring: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=240))
    started_at: float = field(default_factory=time.time)

    def open(self) -> None:
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Fresh file each enable so a capture session is self-contained.
        self._handle = self.path.open("w", encoding="utf-8")
        self._count = 0
        self._ring = deque(maxlen=int(self.ring_size))
        self.started_at = time.time()
        header = {
            "type": "session",
            "t": 0.0,
            "path": str(self.path),
            "note": "Side A TickPackage ingest debug",
        }
        self._handle.write(json.dumps(header) + "\n")
        self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def record(
        self,
        *,
        master_tick: int,
        package: TickPackage | None,
        live_speech: bool,
        live_mode: str,
        plate_open: float,
        smile: float,
        viseme: str,
        field_gain_recipe: float,
        field_gain_eff: float,
        muscles: int,
        gpu_peak: float,
        gpu_mean: float,
        gpu_cells: int,
        presence: str,
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        self._count += 1
        if self.every_n > 1 and (self._count % self.every_n) != 0:
            return None

        labels = getattr(package, "labels", None) if package is not None else None
        values = getattr(package, "values", None) if package is not None else None
        # Sparse packages: rebuild a dense abs proxy from sparse if needed.
        if package is not None and values is None and package.sparse_delta is not None:
            face = package.face
            dense = np.zeros((face.h, face.w, 2), dtype=np.float32)
            idx = np.asarray(package.sparse_idx, dtype=np.int64)
            delta = np.asarray(package.sparse_delta, dtype=np.float32)
            if idx.size and delta.size:
                dense.reshape(-1, 2)[idx] = delta.reshape(-1, 2)
            values = dense

        stats = field_patch_stats(values)
        lab_open = float(getattr(labels, "open_amt", 0.0) or 0.0) if labels else 0.0
        lab_smile = float(getattr(labels, "smile_amt", 0.0) or 0.0) if labels else 0.0
        lab_vis = int(getattr(labels, "viseme_id", -1) or -1) if labels else -1
        row = {
            "type": "ingest",
            "t": round(time.time() - self.started_at, 4),
            "master_tick": int(master_tick),
            "pkg_tick": int(package.tick) if package is not None else -1,
            "kind": _kind_name(package),
            "encoding": _enc_name(package),
            "live_speech": bool(live_speech),
            "live_mode": str(live_mode or ""),
            "presence": str(presence or ""),
            "labels": {
                "open": lab_open,
                "smile": lab_smile,
                "viseme_id": lab_vis,
                "plate_open": float(plate_open),
                "ui_smile": float(smile),
                "viseme": str(viseme or ""),
            },
            "field": stats,
            "gain": {
                "recipe": float(field_gain_recipe),
                "effective": float(field_gain_eff),
                "travel": float(stats["max"]) * float(field_gain_eff),
            },
            "gpu": {
                "peak": float(gpu_peak),
                "mean": float(gpu_mean),
                "cells": int(gpu_cells),
                "muscles": int(muscles),
            },
            "blur_flags": blur_risk(
                plate_open=float(plate_open),
                smile=float(smile),
                field_max=float(stats["max"]),
                field_gain_eff=float(field_gain_eff),
                muscles=int(muscles),
            ),
        }
        self._ring.append(row)
        if self._handle is not None:
            self._handle.write(json.dumps(row) + "\n")
            if self._count % 8 == 0:
                self._handle.flush()
        return row

    def recent(self, n: int = 30) -> list[dict[str, Any]]:
        items = list(self._ring)
        return items[-max(0, int(n)) :] if n else items

    def summary(self) -> dict[str, Any]:
        rows = [r for r in self._ring if r.get("type") == "ingest"]
        if not rows:
            return {"n": 0, "path": str(self.path)}
        flag_counts: dict[str, int] = {}
        for r in rows:
            for f in r.get("blur_flags") or []:
                flag_counts[f] = flag_counts.get(f, 0) + 1
        travels = [float((r.get("gain") or {}).get("travel") or 0) for r in rows]
        opens = [float((r.get("labels") or {}).get("plate_open") or 0) for r in rows]
        return {
            "n": len(rows),
            "path": str(self.path),
            "open_max": max(opens) if opens else 0.0,
            "travel_max": max(travels) if travels else 0.0,
            "blur_flag_counts": flag_counts,
            "kinds": {
                k: sum(1 for r in rows if r.get("kind") == k)
                for k in sorted({str(r.get("kind")) for r in rows})
            },
        }


__all__ = [
    "DEFAULT_LOG",
    "SideADebugLog",
    "blur_risk",
    "field_patch_stats",
]
