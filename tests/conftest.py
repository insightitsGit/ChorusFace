"""Shared fixtures, including a headless OpenGL avatar app."""

from __future__ import annotations

import argparse
import contextlib
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import pytest

from aiface.paths import DEFAULT_AVATAR_FACE

FAKE_SPEECH_RATE = 24_000


def fake_speech(
    text: str,
    *,
    seconds_per_viseme: float = 0.09,
    sample_rate: int = FAKE_SPEECH_RATE,
    lead: float = 0.20,
    tail: float = 0.25,
    honour_punctuation: bool = True,
) -> "object":
    """Audio shaped like speech, without needing a synthesiser installed.

    One voiced burst per word, sized by that word's viseme count, with real
    silence where the punctuation says the speaker stops. That is exactly the
    structure the streaming channel keys off — onsets, level, and gaps — so the
    timing tests are meaningful while staying deterministic and offline.

    Set ``honour_punctuation`` false for a voice that reads straight through its
    own commas, which real synthesisers do and which the channel has to survive.
    """
    from aiface.audio import AudioClip
    from aiface.speech import tokenize_speech

    rng = np.random.default_rng(7)
    pieces = [np.zeros(int(lead * sample_rate), dtype=np.float32)]
    for token in tokenize_speech(text):
        if token.is_word:
            count = int(len(token.visemes) * seconds_per_viseme * sample_rate)
            clock = np.arange(count) / sample_rate
            carrier = np.sin(2.0 * np.pi * 130.0 * clock).astype(np.float32)
            shape = np.hanning(count).astype(np.float32) if count else carrier
            noise = rng.normal(0.0, 0.02, count).astype(np.float32)
            pieces.append(shape * (carrier * 0.5 + noise))
        else:
            paused = honour_punctuation and token.kind in ("pause", "stop")
            gap = 0.22 if paused else 0.02
            pieces.append(np.zeros(int(gap * sample_rate), dtype=np.float32))
    pieces.append(np.zeros(int(tail * sample_rate), dtype=np.float32))
    return AudioClip(np.concatenate(pieces), sample_rate)


@pytest.fixture
def spoken() -> Callable[..., object]:
    """Factory for :func:`fake_speech` clips."""
    return fake_speech


def _default_namespace(config_class: type, **overrides: object) -> argparse.Namespace:
    """Build the argv namespace ``moderngl_window`` would normally supply."""
    parser = argparse.ArgumentParser()
    config_class.add_arguments(parser)
    namespace = parser.parse_args([])
    # Headless tests must not open speakers or wait on SAPI synthesis.
    if "tts" not in overrides:
        overrides = {**overrides, "tts": False}
    for key, value in overrides.items():
        setattr(namespace, key, value)
    return namespace


@pytest.fixture(scope="session")
def synthetic_world(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A deterministic synthetic seed bundle: world, part atlas, and portrait."""
    pytest.importorskip("cv2")
    from aiface.seed import build_avatar_seed, write_seed_bundle

    directory = tmp_path_factory.mktemp("avatar")
    written = write_seed_bundle(
        build_avatar_seed(synthetic=True), directory / DEFAULT_AVATAR_FACE.name
    )
    return written["world"]


@pytest.fixture
def headless_app(synthetic_world: Path) -> Iterator[object]:
    """A fully constructed :class:`AvatarFaceApp` on a headless GL 4.3 context.

    Skips rather than fails when the machine has no OpenGL 4.3 context, so the
    pure-Python suite still runs in CI containers.
    """
    moderngl_window = pytest.importorskip("moderngl_window")
    from moderngl_window.context.headless import Window

    from aiface.app import AvatarFaceApp

    try:
        window = Window(gl_version=(4, 3), size=(512, 512), title="aiface-test")
    except Exception as exc:  # noqa: BLE001 - any driver failure means "no GPU here"
        pytest.skip(f"no headless OpenGL 4.3 context: {exc}")

    moderngl_window.activate_context(window=window)
    AvatarFaceApp.argv = _default_namespace(
        AvatarFaceApp,
        world=synthetic_world,
        no_chat=True,
        face_image=synthetic_world.with_name("source_face.png"),
    )
    application = AvatarFaceApp(ctx=window.ctx, wnd=window, timer=None)
    window.config = application
    try:
        yield application
    finally:
        with contextlib.suppress(Exception):
            application.on_close()
        with contextlib.suppress(Exception):
            window.destroy()
