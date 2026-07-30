"""AIFace — a chat-driven photoreal avatar face.

Four layers, deliberately separable:

``aiface.speech``
    Pure text → viseme → mouth-pose translation and the chat backend.
``aiface.tts`` / ``aiface.audio``
    Speech synthesis, waveform analysis, playback, and viseme timing measured
    from the audio rather than guessed from the letters.
``aiface.biomechanics``
    Deterministic muscle, jaw, eye, breathing, and emotion simulation.
``aiface.runtime``
    A minimal 32-channel GPU field substrate that renders an immutable
    photograph and enforces the Master Lock on identity.

The photograph never mutates. Speech only adds velocity to unlocked mouth
tissue, and the renderer displaces labelled anatomical pieces of the original
pixels, so the face stays the person it started as.
"""

from aiface.paths import DEFAULT_AVATAR_FACE, DEFAULT_AVATAR_SOURCE

__version__ = "0.1.0"

__all__ = ["DEFAULT_AVATAR_FACE", "DEFAULT_AVATAR_SOURCE", "__version__"]
