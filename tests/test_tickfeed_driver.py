"""TickFeedDriver + synth — live package path."""

from __future__ import annotations

import numpy as np

from aiface.tickfeed.driver import TickFeedDriver
from aiface.tickfeed.package import FaceBox, decode, encode
from aiface.tickfeed.schema import PackageKind
from aiface.tickfeed.synth import synthesize_velocity


def test_synth_and_driver_key_then_delta() -> None:
    face = FaceBox(10, 20, 32, 24)
    drv = TickFeedDriver.create(face, mouth_uv=(26.0, 35.0))
    k0 = drv.push_drives(tick=0, open_amt=0.0, smile_amt=0.0, phoneme="REST")
    assert k0.kind == PackageKind.KEYFRAME
    d1 = drv.push_drives(tick=1, open_amt=0.8, smile_amt=0.1, phoneme="AH")
    assert d1.kind == PackageKind.DELTA
    blob = encode(d1)
    back = decode(blob)
    assert back.face.w == 32
    vel = synthesize_velocity(face, open_amt=0.8, smile_amt=0.1)
    assert vel.shape == (24, 32, 2)
    assert float(np.abs(vel).max()) > 0.0
