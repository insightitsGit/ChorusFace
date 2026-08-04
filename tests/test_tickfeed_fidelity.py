"""Fidelity gates: measured pass, provenance, no fake live override."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from chorusface.tickfeed.calibration import write_calibration_script
from chorusface.tickfeed.driver import TickFeedDriver
from chorusface.tickfeed.package import FaceBox, TickLabels
from chorusface.tickfeed.schema import FLAG_VS_REST, PackageKind
from chorusface.tickfeed.timeline_io import (
    SOURCE_MEASURED,
    SOURCE_SYNTH,
    write_face_cell_timeline,
)


def _world_with_timeline(tmp_path: Path, *, n: int = 30, source: int = 0) -> Path:
    face = FaceBox(0, 0, 8, 6)
    vel = np.zeros((n, face.h, face.w, 2), dtype=np.float32)
    for t in range(n):
        vel[t, :, :, 1] = 0.2 * (t / max(n - 1, 1))
    conf = np.full((n, face.n_cells), 220, dtype=np.uint8)
    source = np.full(n, source, dtype=np.uint8)
    write_calibration_script(tmp_path)
    write_face_cell_timeline(
        tmp_path,
        face=face,
        velocity=vel,
        conf=conf,
        video_name="test",
        source=source,
        open_curve=[0.0] * n,
        smile_curve=[0.0] * n,
        lid_curve=[1.0 - 0.5 * (t / max(n - 1, 1)) for t in range(n)],
    )
    return tmp_path


def test_source_gate_lowers_synth_conf(tmp_path: Path) -> None:
    # OPEN beat (~2.1s) so rest-still gate does not overwrite conf to 255.
    _world_with_timeline(tmp_path, n=200, source=SOURCE_SYNTH)
    face = FaceBox(0, 0, 8, 6)
    drv = TickFeedDriver.try_load_timeline(tmp_path, face, mouth_uv=(4.0, 4.0))
    assert drv.timeline_source[130] == SOURCE_SYNTH
    pkg = drv.push_drives(tick=130, open_amt=0.0, smile_amt=0.0, live_speech=False)
    assert pkg.conf is not None
    assert int(pkg.conf.max()) <= 120


def test_measured_timeline_used_when_not_live(tmp_path: Path) -> None:
    # Need ticks past REST (0–1s) so rest-still gate does not erase FIELD.
    _world_with_timeline(tmp_path, n=200, source=SOURCE_MEASURED)
    face = FaceBox(0, 0, 8, 6)
    drv = TickFeedDriver.try_load_timeline(tmp_path, face, mouth_uv=(4.0, 4.0))
    pkg = drv.push_drives(tick=130, open_amt=0.99, smile_amt=0.99, live_speech=False)
    assert pkg.values is not None
    # Measured y displacement from timeline (not zero / not synth open)
    assert float(np.abs(pkg.values[..., 1]).max()) > 0.05


def test_look_drive_lid_reaches_labels(tmp_path: Path) -> None:
    _world_with_timeline(tmp_path, n=40, source=SOURCE_MEASURED)
    face = FaceBox(0, 0, 8, 6)
    drv = TickFeedDriver.try_load_timeline(tmp_path, face, mouth_uv=(4.0, 4.0))
    assert "lid" in drv.look_by_tick[30]
    pkg = drv.push_drives(tick=30, open_amt=0.0, smile_amt=0.0, live_speech=False)
    assert pkg.labels is not None
    assert pkg.labels.lid_amt < 0.95


def test_last_applied_labels_freeze_on_miss() -> None:
    face = FaceBox(0, 0, 4, 4)
    drv = TickFeedDriver.create(face, mouth_uv=(2.0, 2.0))
    pkg = drv.push_drives(tick=0, open_amt=0.7, smile_amt=0.0, phoneme="AA")
    popped = drv.pop_for_master(0)
    assert popped is not None
    assert drv.last_applied_labels is not None
    assert abs(float(drv.last_applied_labels.open_amt) - 0.7) < 1e-3
    # Miss does not clear applied labels
    miss = drv.pop_for_master(1)
    assert miss is None
    assert abs(float(drv.last_applied_labels.open_amt) - 0.7) < 1e-3


def test_wire_loop_source_defaults_to_package() -> None:
    face = FaceBox(0, 0, 4, 4)
    drv = TickFeedDriver.create(face, mouth_uv=(2.0, 2.0))
    assert drv.wire_loop_source == "package"


def test_hello_disp_vs_rest_and_flag() -> None:
    from chorusface.tickfeed.package import build_hello, negotiate_hello

    face = FaceBox(1, 2, 8, 8)
    hello = build_hello(face, world_id="tickfeed")
    assert hello.hello is not None
    assert hello.hello.apply_mode == "disp_vs_rest"
    assert hello.flags & FLAG_VS_REST
    ack = negotiate_hello(hello)
    assert ack.hello is not None
    assert ack.hello.apply_mode == "disp_vs_rest"
