"""TickFeed — full-face KEY/DELTA packages @ 60 Hz (design: TickFeedDesign.md)."""

from aiface.tickfeed.collect import prepare_face_timeline
from aiface.tickfeed.driver import TickFeedDriver, face_box_from_profile
from aiface.tickfeed.package import (
    FaceBox,
    TickLabels,
    TickPackage,
    apply_to_state,
    build_delta,
    build_keyframe,
    decode,
    encode,
)
from aiface.tickfeed.ring import FaceVelocityState, LockstepPlayer, TickRingBuffer
from aiface.tickfeed.schema import (
    CHANNEL_MASK_VELOCITY,
    TICK_DT,
    TICK_RATE_HZ,
    VELOCITY_MISS_DAMP,
    BeatId,
    DeltaEncoding,
    EmotionId,
    PackageKind,
    ValueDtype,
)

__all__ = [
    "CHANNEL_MASK_VELOCITY",
    "TICK_DT",
    "TICK_RATE_HZ",
    "VELOCITY_MISS_DAMP",
    "BeatId",
    "DeltaEncoding",
    "EmotionId",
    "FaceBox",
    "FaceVelocityState",
    "LockstepPlayer",
    "PackageKind",
    "TickFeedDriver",
    "TickLabels",
    "TickPackage",
    "TickRingBuffer",
    "ValueDtype",
    "apply_to_state",
    "build_delta",
    "build_keyframe",
    "decode",
    "encode",
    "face_box_from_profile",
    "prepare_face_timeline",
]
