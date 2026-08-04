"""From-scratch live control vector pipeline (branch live-vector-from-video).

Video truth → control vectors → train → GPU display recipe.
Does not invent face RGB.
"""

from chorusface.live_vector.driver import LiveVectorDriver
from chorusface.live_vector.pipeline import train_avatar_from_video
from chorusface.live_vector.schema import LiveControlVector, MODEL_NAME

__all__ = [
    "LiveControlVector",
    "LiveVectorDriver",
    "MODEL_NAME",
    "train_avatar_from_video",
]
