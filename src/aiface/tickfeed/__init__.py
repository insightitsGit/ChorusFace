"""TickFeed — full-face KEY/DELTA packages @ 60 Hz (design: TickFeedDesign.md)."""

from aiface.tickfeed.calibration import (
    load_calibration_script,
    validate_calibration_take,
    write_calibration_script,
)
from aiface.tickfeed.collect import prepare_face_timeline
from aiface.tickfeed.cosmetics import load_cosmetic_prefs, write_cosmetic_prefs
from aiface.tickfeed.driver import TickFeedDriver, face_box_from_profile
from aiface.tickfeed.ml import TickFeedMLStack, fit_all_layers
from aiface.tickfeed.package import (
    FaceBox,
    HelloPayload,
    TickLabels,
    TickPackage,
    apply_to_state,
    build_delta,
    build_hello,
    build_hello_ack,
    build_keyframe,
    decode,
    encode,
    negotiate_hello,
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
    "HelloPayload",
    "LockstepPlayer",
    "PackageKind",
    "TickFeedDriver",
    "TickFeedMLStack",
    "TickLabels",
    "TickPackage",
    "TickRingBuffer",
    "ValueDtype",
    "apply_to_state",
    "build_delta",
    "build_hello",
    "build_hello_ack",
    "build_keyframe",
    "decode",
    "encode",
    "face_box_from_profile",
    "fit_all_layers",
    "load_calibration_script",
    "load_cosmetic_prefs",
    "negotiate_hello",
    "prepare_face_timeline",
    "validate_calibration_take",
    "write_calibration_script",
    "write_cosmetic_prefs",
]
