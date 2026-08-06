"""Play a PulseChunk into TickFeed-facing packages / LOOK drives."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from chorusface.tickfeed.package import FaceBox, TickPackage, encode
from chorusface.vowel.expand import author_w_from_catalog, load_catalog, load_wexpand
from chorusface.vowel.pipeline import ComposeResult, compose_utterance
from chorusface.vowel.pulsechunk import PulseChunk, decode_pulsechunk
from chorusface.vowel.tick_emit import EmitConfig, emit_tick_packages


@dataclass
class VowelRuntime:
    face: FaceBox = field(default_factory=lambda: FaceBox(64, 64, 128, 128))
    W: np.ndarray | None = None
    cells: list[tuple[int, int]] | None = None
    model_dir: Path | None = None
    world_dir: Path | None = None

    @classmethod
    def from_world(cls, world_dir: str | Path) -> VowelRuntime:
        world = Path(world_dir)
        rt = cls(model_dir=world / "vowel", world_dir=world)
        wexp = world / "vowel" / "expand_matrix_v1.wexpand"
        catalog = world / "region_catalog.json"
        if wexp.is_file():
            W, cells, _ = load_wexpand(wexp)
            rt.W, rt.cells = W, cells
        elif catalog.is_file():
            W, cells = author_w_from_catalog(load_catalog(catalog))
            rt.W, rt.cells = W, cells
        else:
            W, cells = author_w_from_catalog({})
            rt.W, rt.cells = W, cells
        # face box from cell extents
        xs = [c[0] for c in rt.cells]
        ys = [c[1] for c in rt.cells]
        x0, y0 = min(xs), min(ys)
        x1, y1 = max(xs), max(ys)
        rt.face = FaceBox(x0, y0, max(1, x1 - x0 + 1), max(1, y1 - y0 + 1))
        return rt

    def compose(self, payload: dict) -> ComposeResult:
        return compose_utterance(payload, model_dir=self.model_dir)

    def packages_for(self, chunk: PulseChunk) -> list[TickPackage]:
        cfg = EmitConfig(face=self.face, W=self.W, cells=self.cells)
        return emit_tick_packages(chunk, cfg)

    def push_to_transport(
        self,
        chunk: PulseChunk,
        transport: Any,
    ) -> int:
        """Emit TPK1 stream onto TickFeedTransport (Fabric / spool lane B)."""
        n = 0
        for pkg in self.packages_for(chunk):
            raw = encode(pkg)
            kind = int(getattr(pkg, "kind", 0) or 0)
            transport.push_package_bytes(int(pkg.tick), raw, kind=kind)
            n += 1
        return n

    def iter_packages(self, payload: dict) -> Iterator[TickPackage]:
        result = self.compose(payload)
        yield from self.packages_for(result.chunk)

    def load_chunk(self, data: bytes, utterance_id: str | None = None) -> PulseChunk:
        return decode_pulsechunk(data, utterance_id=utterance_id)
