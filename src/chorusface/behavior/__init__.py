"""Avatar behavior: measured cell/group transitions + ML fill for gaps."""

from chorusface.behavior.schema import BehaviorState, CONTROL_NAMES, TRACK_NPZ
from chorusface.behavior.track import TransitionTrack, load_transition_track

# Driver / train pipeline imported lazily by callers that need them to avoid
# amin_loop ↔ chorusface.behavior circular imports at package init.


def __getattr__(name: str):
    if name == "BehaviorDriver":
        from chorusface.behavior.driver import BehaviorDriver

        return BehaviorDriver
    if name == "train_behavior_from_video":
        from chorusface.behavior.pipeline import train_behavior_from_video

        return train_behavior_from_video
    raise AttributeError(name)


__all__ = [
    "CONTROL_NAMES",
    "TRACK_NPZ",
    "BehaviorDriver",
    "BehaviorState",
    "TransitionTrack",
    "load_transition_track",
    "train_behavior_from_video",
]
