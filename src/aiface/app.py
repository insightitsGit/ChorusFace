"""Chat-driven avatar face window.

Ties the layers together:

* :mod:`aiface.speech` turns a reply into a timed viseme stream.
* :mod:`aiface.stream` locks that stream to audio arriving from somewhere else —
  the product path, since a voice model already has a voice and what it lacks is
  a face that agrees with it. Callers push audio at ``/voice/pcm``.
* :mod:`aiface.tts` is the fixture behind that: it can speak a reply locally, and
  its full-lookahead alignment is the reference :mod:`aiface.sync` measures the
  streaming channel against. Off unless ``--tts`` asks for it.
* :mod:`aiface.biomechanics` turns each viseme into muscle impulses and
  integrates them into continuous jaw, lip, eye, and brow state.
* :mod:`aiface.runtime` applies the resulting velocity impulses to unlocked
  mouth cells and renders the photograph with its parts displaced.

Three clocks can drive the mouth, in descending order of authority: audio
arriving on the sync channel, a clip this process synthesised, and — with no
audio at all — timing estimated from the written words.

Speech never paints pixels and never rewrites identity. Impulses travel through
the same SSBO command rows as any other write — operation ``±4`` adds
``(V_x, V_y)`` inside a disc — and the constraint shader drops the write on
Master-Locked cells, so the skull, cheeks, and eyes cannot drift.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Final, Sequence

import moderngl
import numpy as np

from aiface.audio import (
    AudioError,
    AudioSink,
    default_sink_preference,
    open_audio_sink,
    rms_envelope,
)
from aiface.biomechanics import (
    DEFAULT_FACE_DEFINITION,
    BiomechanicalFace,
    FaceRenderState,
    FieldImpulseSpec,
)
from aiface.chatbox import (
    SPEAKER_FACE,
    SPEAKER_SYSTEM,
    SPEAKER_YOU,
    ChatBox,
    frame_layout,
    hit_test,
    paint_panel,
)
from aiface.mouth_owner import (
    MouthOwnership,
    resolve_mouth_ownership,
)
from aiface.live_vector import LiveVectorDriver
from aiface.behavior import BehaviorDriver
from aiface.mouth_speed import (
    DEFAULT_MOUTH_SPEED,
    HOLD_SCALE_DEFAULT,
    MOUTH_SPEED_PRESETS,
    clamp_hold_scale,
    hold_scale_to_params,
    next_preset_key,
    preset_by_key,
)
from aiface.parts import (
    build_face_part_atlas,
    default_parts_path,
    load_face_part_atlas,
    save_face_part_atlas,
)
from aiface.avatar_profile import open_avatar
from aiface.paths import DEFAULT_AVATAR_FACE, DEFAULT_AVATAR_SOURCE
from aiface.runtime.bds import (
    HARD_SURFACE_CHANNEL,
    HUMAN_LOCK_CHANNEL,
    PRIORITY_LEVELS,
    TICK_RATE_HZ,
    BDSFormatError,
    load_bds,
)
from aiface.runtime.commands import PaintCommand
from aiface.runtime.field import FieldRuntime
from aiface.cell_cluster import (
    CellClusterIndex,
    distribute_to_nearby_cells,
    parse_drive_request,
    to_commands,
)
from aiface.display_layers import FrameLayerState, evaluate_frame_layers
from aiface.tickfeed.driver import TickFeedDriver, face_box_from_profile
from aiface.service import DEFAULT_HOST, DEFAULT_PORT, FaceBridge, new_token
from aiface.skinning import (
    MAX_ACTIVE_MUSCLES,
    build_tissue_maps,
    default_tissue_path,
    jaw_pose_from_definition,
    load_tissue_maps,
    pack_muscle_uniforms,
    save_tissue_maps,
)
from aiface.mouth_timeline import LayerCommand, MouthLayerTimeline
from aiface.speech import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    EMOTION_IMPULSES,
    ConversationSession,
    MouthPose,
    VisemeEvent,
    extract_states,
    llm_reply,
    mouth_pose,
    schedule_spans,
    schedule_visemes,
    strip_tags,
)
from aiface.stream import (
    DEFAULT_LOOKAHEAD,
    DEFAULT_STREAM_RATE,
    StreamConfig,
    StreamedSpan,
    VoiceStream,
    stream_visemes,
)
from aiface.tts import (
    ALIGN_ENERGY,
    ALIGNMENTS,
    DEFAULT_SPEECH_MODEL,
    DEFAULT_SPEECH_VOICE,
    DEFAULT_TRANSCRIBE_MODEL,
    PreparedSpeech,
    SpeechSynthesizer,
    TTSError,
    build_synthesizer,
    local_voice_available,
)

if TYPE_CHECKING:
    from aiface.seed import FaceBox

MOUTH_RADIUS: Final = 14.0
# AMIN step 11: display textures are decoupled from the 256² cell grid — the
# shader samples by UV, so photo/plates render at capture resolution up to this
# cap while the field keeps its grid size.
MAX_DISPLAY_TEXTURE: Final = 1024
# Audio-locked speech schedules a viseme per articulated sound, so the queue has
# to hold a whole paragraph without dropping the beginning of it.
MAX_PHONEME_QUEUE: Final = 512
TELEMETRY_WINDOW: Final = 30
DEBUG_VIEW_NAMES: Final[dict[int, str]] = {
    0: "portrait",
    1: "density",
    2: "velocity",
    3: "muscle",
    4: "emotion",
    5: "jaw",
    6: "locks",
    7: "impulse-heat",
    8: "parts",
    9: "displacement",
    10: "mobility",
    11: "tissue-gates",
}


@dataclass(slots=True)
class VectorTelemetry:
    """Rolling mouth-field statistics for the HUD."""

    mean_speed: float = 0.0
    peak_speed: float = 0.0
    active_cells: int = 0
    last_phoneme: str = "REST"
    last_emotion: str = "NEUTRAL"
    impulses_fired: int = 0
    gpu_frame_ms: float = 0.0
    command_latency_ms: float = 0.0
    alignment: str = ""
    clip_seconds: float = 0.0


@dataclass(slots=True)
class SpokenReply:
    """A finished chat reply, with its synthesised voice when there is one."""

    text: str
    speech: PreparedSpeech | None = None


def _environment_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


# Demo face presence — "0 state" is idle closed lips + natural blink.
PRESENCE_ZERO: Final = "zero"
PRESENCE_HEARING: Final = "hearing"
PRESENCE_SPEAKING: Final = "speaking"
# Idle LOOK variants while presence stays zero (switchable).
# neutral = no impression; smile = closed-lip smile plate; waiting = attentive.
ZERO_MOOD_NEUTRAL: Final = "neutral"
ZERO_MOOD_SMILE: Final = "smile"
ZERO_MOOD_WAITING: Final = "waiting"
ZERO_MOODS: Final = (
    ZERO_MOOD_NEUTRAL,
    ZERO_MOOD_SMILE,
    ZERO_MOOD_WAITING,
)


def _default_tts_enabled() -> bool:
    """Whether to synthesise speech in-process. Off unless asked.

    Owning a voice is not this program's job: anything with something to say
    already has one, and it pushes the audio to the sync channel. Local synthesis
    stays as a fixture — a demo without a client, and the clip source for the
    offline oracle — so it is opt-in through ``--tts`` or ``AIFACE_TTS``.
    """
    return _environment_flag("AIFACE_TTS")


class AvatarFaceApp(FieldRuntime):
    """A Master-Locked talking face driven by chat."""

    title = "AIFace - Avatar Chat"
    # Uncap the swap interval so the window can clear 85+ FPS when the GPU can.
    vsync = False
    # Taller than it is wide: the portrait keeps a square frame and the chat
    # panel takes the band underneath. Free aspect so resizing works.
    window_size = (1024, 1320)
    aspect_ratio = None

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        super().add_arguments(parser)
        parser.add_argument(
            "--demo",
            action="store_true",
            help="Play a built-in phoneme loop without waiting for chat",
        )
        parser.add_argument(
            "--gpu-log",
            action="store_true",
            default=_environment_flag("AIFACE_GPU_LOG"),
            help=(
                "Log GPU recipe objects once per 60 Hz simulation tick "
                "(mouth object, plates, jaw, viseme). Also AIFACE_GPU_LOG=1"
            ),
        )
        parser.add_argument(
            "--gpu-log-verbose",
            action="store_true",
            help="Print every tick to stdout (default: file + 5 Hz summary)",
        )
        parser.add_argument(
            "--tickfeed-debug",
            action="store_true",
            default=_environment_flag("AIFACE_TICKFEED_DEBUG"),
            help=(
                "Dump Side A TickPackage ingest JSONL "
                "(labels/FIELD/gain/blur flags). Also AIFACE_TICKFEED_DEBUG=1"
            ),
        )
        parser.add_argument(
            "--no-chat",
            action="store_true",
            help="Disable the stdin chat thread (window + demo only)",
        )
        parser.add_argument(
            "--llm-base-url",
            default=os.environ.get("AIFACE_LLM_BASE_URL", DEFAULT_BASE_URL),
            help="OpenAI-compatible chat endpoint root",
        )
        parser.add_argument(
            "--llm-model",
            default=os.environ.get("AIFACE_LLM_MODEL", DEFAULT_MODEL),
            help="Chat model name",
        )
        parser.add_argument(
            "--llm-api-key",
            default=os.environ.get(
                "AIFACE_LLM_API_KEY",
                os.environ.get("OPENAI_API_KEY", ""),
            ),
            help="API key; without one the deterministic local reply is used",
        )
        parser.add_argument(
            "--tts",
            action=argparse.BooleanOptionalAction,
            default=_default_tts_enabled(),
            help=(
                "Synthesise replies here and lock the lips to that audio. Off by "
                "default: normally the voice lives elsewhere and pushes audio to "
                "/voice/pcm. Useful for a demo with no client, and as the fixture "
                "aiface-sync measures the streaming channel against"
            ),
        )
        parser.add_argument(
            "--tts-backend",
            choices=("auto", "openai", "sapi", "command"),
            default=os.environ.get("AIFACE_TTS_BACKEND", "auto"),
            help="Synthesiser: hosted, Windows SAPI, local command, or auto",
        )
        parser.add_argument(
            "--tts-model",
            default=os.environ.get("AIFACE_TTS_MODEL", DEFAULT_SPEECH_MODEL),
            help="Hosted speech model name",
        )
        parser.add_argument(
            "--tts-voice",
            default=os.environ.get("AIFACE_TTS_VOICE", DEFAULT_SPEECH_VOICE),
            help="Hosted speech voice name",
        )
        parser.add_argument(
            "--tts-speed",
            type=float,
            default=float(os.environ.get("AIFACE_TTS_SPEED", "1.0")),
            help="Hosted speech rate multiplier",
        )
        parser.add_argument(
            "--speech-pace",
            type=float,
            default=float(os.environ.get("AIFACE_SPEECH_PACE", "0")),
            help=(
                "Slow audio+visemes together for clearer mouth shapes "
                "(1.0=realtime, 1.12=+12%%). 0 = use recipe speech_pace"
            ),
        )
        parser.add_argument(
            "--tts-instructions",
            default=os.environ.get("AIFACE_TTS_INSTRUCTIONS", ""),
            help="Delivery notes for speech models that accept them",
        )
        parser.add_argument(
            "--tts-command",
            default=os.environ.get("AIFACE_TTS_COMMAND", ""),
            help=(
                "Local synthesiser reading text on stdin and writing wav to "
                'stdout, e.g. "espeak-ng --stdout -s 165"'
            ),
        )
        parser.add_argument(
            "--tts-align",
            choices=ALIGNMENTS,
            default=os.environ.get("AIFACE_TTS_ALIGN")
            or (
                ALIGN_WORDS
                if (
                    os.environ.get("OPENAI_API_KEY")
                    or os.environ.get("AIFACE_LLM_API_KEY")
                )
                else ALIGN_ENERGY
            ),
            help=(
                "Viseme timing source: transcribed word timestamps (default "
                "when OPENAI_API_KEY is set), acoustic energy, or uniform stretch"
            ),
        )
        parser.add_argument(
            "--tts-transcribe-model",
            default=os.environ.get("AIFACE_TTS_TRANSCRIBE_MODEL", DEFAULT_TRANSCRIBE_MODEL),
            help="Model used for word timestamps when --tts-align words",
        )
        parser.add_argument(
            "--tts-warp",
            type=float,
            default=float(os.environ.get("AIFACE_TTS_WARP", "0.65")),
            help="How strongly acoustic energy bends the viseme schedule (0-1)",
        )
        parser.add_argument(
            "--tts-latency",
            type=float,
            default=float(os.environ.get("AIFACE_TTS_LATENCY", "0.0")),
            help=(
                "Extra seconds to delay the lips relative to the audio; "
                "negative moves them earlier"
            ),
        )
        parser.add_argument(
            "--voice-rate",
            type=int,
            default=int(os.environ.get("AIFACE_VOICE_RATE", DEFAULT_STREAM_RATE)),
            help=(
                "Sample rate assumed for audio pushed to /voice/pcm when the "
                "caller does not declare one"
            ),
        )
        parser.add_argument(
            "--voice-lookahead",
            type=float,
            default=float(os.environ.get("AIFACE_VOICE_LOOKAHEAD", DEFAULT_LOOKAHEAD)),
            help=(
                "Seconds of arrived audio the sync channel holds back before "
                "judging it; higher is steadier and later"
            ),
        )
        parser.add_argument(
            "--voice-trim",
            type=float,
            default=float(os.environ.get("AIFACE_VOICE_TRIM", "0.0")),
            help=(
                "Seconds to shift streamed visemes against the caller's audio "
                "clock; positive moves the lips later"
            ),
        )
        parser.add_argument(
            "--audio-backend",
            default=default_sink_preference(),
            help=(
                "Playback backend: auto, sounddevice, winsound, command, or "
                "null for measured timing without sound"
            ),
        )
        parser.add_argument(
            "--show-locks",
            action="store_true",
            help="Show magenta Master-Lock boundaries over the natural face",
        )
        parser.add_argument(
            "--face-image",
            type=Path,
            default=None,
            help=(
                "Optional override photograph. Default: this world's "
                "source_face.png from adoption (not a hard-coded avatar path)"
            ),
        )
        parser.add_argument(
            "--face-definition",
            type=Path,
            default=DEFAULT_FACE_DEFINITION,
            help="JSON muscle/character definition for the biomechanical layer",
        )
        parser.add_argument(
            "--capture",
            type=Path,
            default=None,
            help="Write rendered PNG frames into this directory and exit",
        )
        parser.add_argument(
            "--capture-frames",
            type=int,
            default=60,
            help="How many frames --capture writes before closing",
        )
        parser.add_argument(
            "--no-chat-box",
            action="store_true",
            help="Hide the in-window chat panel and render the portrait fullscreen",
        )
        parser.add_argument(
            "--bridge",
            action="store_true",
            help="Open an NWR-style loopback HTTP face bridge (speak/preview/status)",
        )
        parser.add_argument(
            "--bridge-host",
            default=os.environ.get("AIFACE_BRIDGE_HOST", DEFAULT_HOST),
            help="Face bridge bind host; loopback unless --allow-remote-bind",
        )
        parser.add_argument(
            "--allow-remote-bind",
            action="store_true",
            help="Permit a non-loopback --bridge-host, exposing the face off-machine",
        )
        parser.add_argument(
            "--bridge-port",
            type=int,
            default=int(os.environ.get("AIFACE_BRIDGE_PORT", str(DEFAULT_PORT))),
            help="Face bridge bind port",
        )
        parser.add_argument(
            "--bridge-token",
            default=os.environ.get("AIFACE_BRIDGE_TOKEN", ""),
            help="Bearer token; generated and printed when --bridge is set without one",
        )
        parser.add_argument(
            "--bridge-direct-speak",
            action="store_true",
            help="POST /speak TTS the text directly (skip LLM) — lab calibration",
        )
        parser.add_argument(
            "--wire-loop",
            action=argparse.BooleanOptionalAction,
            default=_environment_flag("AIFACE_WIRE_LOOP"),
            help=(
                "TickFeed master consumes from transport (c_t L4 expand or lane-B "
                "package decode) instead of the local produce→ring path. Proves the "
                "bandwidth claim. Also AIFACE_WIRE_LOOP=1"
            ),
        )
        parser.add_argument(
            "--wire-loop-source",
            choices=("code", "package"),
            default=os.environ.get("AIFACE_WIRE_LOOP_SOURCE", "code"),
            help="Wire-loop feed: lane-A c_t (default) or lane-B TickPackage bytes",
        )

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)

        world = Path(getattr(self.argv, "world", DEFAULT_AVATAR_FACE))
        # Adoption layer: any world dir that meets requirements shares this path.
        self._avatar_bundle = open_avatar(world)
        world = self._avatar_bundle.world_path
        self._chat_queue: queue.Queue[str] = queue.Queue()
        self._reply_queue: queue.Queue[SpokenReply] = queue.Queue()
        self._visemes: deque[VisemeEvent] = deque()
        self._telemetry = VectorTelemetry()
        self._gpu_times: deque[float] = deque(maxlen=TELEMETRY_WINDOW)
        self._clock0 = time.perf_counter()
        self._telemetry_age = 0
        self._mouth_pose = mouth_pose("REST", "NEUTRAL")
        self._target_mouth_pose = self._mouth_pose
        self._show_locks = bool(getattr(self.argv, "show_locks", False))
        self._avatar_source = self._resolve_avatar_source(world)
        self._part_anchors: dict[str, tuple[float, float]] = {}
        self._use_parts = False
        self._use_tissue = False
        self._use_plates = False
        self._active_muscle_count = 0
        self._capture_priors: dict[str, float] = {}
        self._plate_atlas = None
        self._atlas_textures: list[moderngl.Texture] = []
        self._plate_blend = (0.0, 0.0)
        self._plate_blend_current = (0.0, 0.0)
        self._plate_openness_current = 0.0
        self._plate_pair = (0, 0)
        self._held_speech_viseme = "REST"
        self._mouth_timeline = MouthLayerTimeline()
        self._layer_command = LayerCommand(
            phoneme="REST",
            atlas_viseme="REST",
            open_amount=0.0,
            smile_amount=0.0,
            jaw_target=0.0,
            plate_openness=0.0,
            source="timeline-rest",
            active_until=0.0,
        )
        self._expression_catalog = None
        self._expr_eye_widen = 0.0
        self._expr_brow_raise = 0.0
        self._expr_plate_blend = 0.0
        self._expr_target_widen = 0.0
        self._expr_target_brow = 0.0
        self._expr_target_blend = 0.0
        self._expr_role_name = "rest"
        self._mouth_ownership = resolve_mouth_ownership(
            openness=0.0, emotion="NEUTRAL", phoneme="REST", speaking=False
        )
        # Branch live-vector-from-video: new driver is authority (not old patches).
        self._live_vector = LiveVectorDriver.try_load(world)
        # Measured transitions + ML fill for missing in-betweens / live speech.
        self._behavior = BehaviorDriver.try_load(world)
        # AMIN Step 8/10: recipe + jaw from the adopted avatar bundle.
        self._display_recipe = self._avatar_bundle.recipe
        cli_pace = float(getattr(self.argv, "speech_pace", 0.0) or 0.0)
        if cli_pace > 0.0:
            from dataclasses import replace as _dc_replace

            self._display_recipe = _dc_replace(
                self._display_recipe,
                speech_pace=max(0.85, min(1.60, cli_pace)),
            )
            print(
                f"Speech pace: {float(self._display_recipe.speech_pace):.3f} "
                f"(CLI --speech-pace; 1.0s → "
                f"{float(self._display_recipe.speech_pace):.3f}s)"
            )
        else:
            print(
                f"Speech pace: {float(self._display_recipe.speech_pace):.3f} "
                f"(recipe; 1.0s → "
                f"{float(self._display_recipe.speech_pace):.3f}s)"
            )
        self._condition_jaw = dict(self._avatar_bundle.condition_jaw)
        self._open_close_envelope = None
        self._open_close_start: float | None = None
        self._open_close_peak = float(self._live_vector.peak_hint)
        self._ml_openness = 0.0
        self._ml_jaw = 0.0
        self._ml_width = 0.0
        self._ml_smile = 0.0
        self._ml_plate_gate = 0.0
        self._open_close_source = "heuristic"
        self._mouth_object_cells = 0
        self._cell_clusters: CellClusterIndex | None = None
        self._tickfeed: TickFeedDriver | None = None
        self._tickfeed_look_authority = False
        # Live chat/TTS overlay for TickFeed (bypasses MouthLayerTimeline LOOK).
        self._tickfeed_live: dict[str, float | str] | None = None
        self._tickfeed_last_pkg = None
        self._tickfeed_last_live = False
        self._tickfeed_last_live_mode = ""
        self._field_gain_eff = 0.0
        # Isolation modes for feed-vs-NWR calibrate (bridge POST /calibrate).
        # normal | plate_only (gain=0) | field_only (plates forced closed).
        self._calibrate_mode = "normal"
        # Plate openness hysteresis — hold shapes so mid-blend flicker softens.
        self._plate_open_hyst = 0.0
        # Transition ownership (ChatGPT roadmap): track d(open)/dt for FIELD gate
        # and atlas snap. States: REST | OPENING | OPEN | CLOSING.
        self._plate_open_prev = 0.0
        self._plate_open_vel = 0.0
        self._mouth_transition = "REST"
        from aiface.tickfeed.side_a_debug import SideADebugLog

        self._side_a_debug = SideADebugLog(
            enabled=bool(getattr(self.argv, "tickfeed_debug", False))
        )
        if self._side_a_debug.enabled:
            self._side_a_debug.open()
            print(f"TickFeed Side A debug: {self._side_a_debug.path}")
        # 0-state / hearing / speaking — chat presence for the demo face.
        self._presence: str = PRESENCE_ZERO
        # Idle face within 0-state: neutral | smile | waiting (Z cycles).
        self._zero_mood: str = ZERO_MOOD_NEUTRAL
        try:
            from aiface.tickfeed.cosmetics import load_cosmetic_prefs

            self._cosmetics = load_cosmetic_prefs(self._avatar_bundle.root)
        except Exception:  # noqa: BLE001
            self._cosmetics = None
        self._frame_layers = FrameLayerState()
        self._pending_cell_commands: list = []
        self._gpu_log = bool(getattr(self.argv, "gpu_log", False))
        self._gpu_log_verbose = bool(getattr(self.argv, "gpu_log_verbose", False))
        self._gpu_log_last_tick = -1
        self._gpu_log_path: Path | None = None
        self._gpu_log_handle = None
        if self._gpu_log:
            from aiface.paths import PREVIEWS, ensure_output_tree

            ensure_output_tree()
            self._gpu_log_path = PREVIEWS / "gpu_tick.log"
            self._gpu_log_handle = self._gpu_log_path.open("w", encoding="utf-8")
            print(f"GPU tick log (60 Hz): {self._gpu_log_path}")
        # Muscle anchors are authored in face-box UV, so the box has to be
        # known before any texture or uniform derived from it is built.
        self._face_box = self._load_face_box(world)
        self._biomech = BiomechanicalFace.from_file(
            getattr(self.argv, "face_definition", DEFAULT_FACE_DEFINITION),
            seed=17,
        )
        self._apply_capture_priors()
        (
            self._avatar_base_texture,
            self._avatar_parts_texture,
        ) = self._create_avatar_base_texture()
        self._avatar_tissue_texture = self._create_tissue_texture()
        (
            self._avatar_open_plate_texture,
            self._avatar_smile_plate_texture,
        ) = self._create_expression_plate_textures()
        self._load_plate_atlas_textures()
        self._avatar_expr_plate_texture = self._create_expression_catalog_texture()
        # Mouth centre after part anchors load so lip pieces stay registered.
        self._mouth_center = self._estimate_mouth_center()
        self._load_cell_clusters()
        self._derive_face_geometry()
        self._render_state = FaceRenderState()
        self._debug_view = 0
        self._cpu_frame_ms = 0.0
        self._update_avatar_uniforms()

        self._chat_thread: threading.Thread | None = None
        self._llm_thread: threading.Thread | None = None
        self._shutdown = threading.Event()
        self._conversation = ConversationSession()
        self._speech: SpeechSynthesizer | None = None
        self._audio: AudioSink | None = None
        self._speech_trim = float(getattr(self.argv, "tts_latency", 0.0))
        # Wall-clock anchor for audio media_time → viseme due_at (playback sync).
        self._audio_anchor: float | None = None
        self._start_speech_engine()
        self._bridge: FaceBridge | None = None
        self._open_voice_channel()
        self._chatbox = ChatBox()
        self._chat_box_visible = not bool(getattr(self.argv, "no_chat_box", False))
        self._panel_rect = (0, 0, 0, 0)
        self._panel_fonts: tuple[object, object] | None = None
        self._ui_hits: dict[str, tuple[int, int, int, int]] = {}
        self._mouth_menu_open = False
        self._mouth_hold_dragging = False
        self._mouth_hold_scale = float(HOLD_SCALE_DEFAULT)
        self._mouth_speed = preset_by_key(DEFAULT_MOUTH_SPEED)
        self._apply_mouth_speed_preset(self._mouth_speed, announce=False)
        self._relayout_frame()
        if self._chat_box_visible:
            # Escape is the window's default exit key; the chat box needs it to
            # toggle typing, and closing mid-sentence is never what you meant.
            with contextlib.suppress(AttributeError, TypeError):
                self.wnd.exit_key = None
            self._chatbox.add(
                SPEAKER_SYSTEM,
                "0-state: closed lips + blink. Typing → hearing look. "
                "Enter to chat; face returns to 0-state when done. "
                "Hold slider / Mouth dropdown or M for plate timing.",
            )

        self._capture_directory = getattr(self.argv, "capture", None)
        self._capture_budget = max(int(getattr(self.argv, "capture_frames", 60)), 0)
        self._captured = 0
        if self._capture_directory is not None:
            Path(self._capture_directory).mkdir(parents=True, exist_ok=True)

        if bool(getattr(self.argv, "demo", False)):
            if self._tickfeed_look_authority:
                # TickFeed-native demo: idle plays measured REST→SMILE→OPEN…;
                # do not boot with old hard-snap open-mouth greeting.
                print(
                    "TickFeed demo: LOOK from package labels "
                    "(one 8s calibration pass, then 0-state: closed lips + blink; "
                    "typing = hearing look; chat end → 0-state)."
                )
                self._enter_zero_state(blink=True)
            else:
                self._speak_without_chat(
                    "[EMOTION:HAPPY] Ah, oh — hello! My smile and lips follow the capture."
                )
        if not bool(getattr(self.argv, "no_chat", False)):
            self._start_chat_thread()
        if bool(getattr(self.argv, "bridge", False)):
            self._start_face_bridge()

        print(self._avatar_help())
        profile = self._avatar_bundle.profile
        adopt = "OK" if self._avatar_bundle.ok else "INCOMPLETE"
        print(
            f"Avatar adopt [{adopt}]: id={profile.id} "
            f"mouth_cells={profile.geometry.mouth_cell_count} "
            f"root={self._avatar_bundle.root}"
        )
        if not self._avatar_bundle.ok:
            print(
                "  missing: "
                + ", ".join(profile.validation.missing[:6])
            )
        print(
            f"Mouth centre ~ ({self._mouth_center[0]:.1f}, {self._mouth_center[1]:.1f})"
        )
        if self._avatar_source is not None:
            print(f"Immutable face photo: {self._avatar_source}")

    @staticmethod
    def _avatar_help() -> str:
        return (
            "AIFace - biomechanical face driven by a continuous muscle "
            "displacement field. Type into the chat box under the portrait and "
            "press Enter; the terminal chat> prompt still works too. "
            "Push audio to /voice/pcm to lock the lips to another voice, or add "
            "--tts to synthesise one here. "
            "Type 'quit' to close. "
            "Keys: Esc toggles typing, Up/Down recall, F1-F11 debug "
            "(F9 displacement, F10 mobility, F11 tissue gates), "
            "H locks, Space pause."
        )

    # ------------------------------------------------------------- geometry

    def _load_face_box(self, world: Path) -> dict[str, float]:
        """Read the seed's face rectangle so muscle UVs land on the right pixels."""
        try:
            header, _grid = load_bds(world)
        except (OSError, BDSFormatError, ValueError):
            box = {}
        else:
            metadata = header.get("application_metadata", {})
            box = metadata.get("avatar_seed", {}).get("face_box") or {}
        return {
            "x": float(box.get("x", 0.0)),
            "y": float(box.get("y", 0.0)),
            "width": float(box.get("width", self.grid_width)),
            "height": float(box.get("height", self.grid_height)),
        }

    def _uv_to_grid(self, uv: tuple[float, float]) -> tuple[float, float]:
        """Map face-definition UV (image y-down) into grid coordinates (y-up)."""
        box = self._face_box
        image_x = box["x"] + uv[0] * box["width"]
        image_y = box["y"] + uv[1] * box["height"]
        grid_x = float(np.clip(image_x, 1.0, self.grid_width - 1.0))
        grid_y = float(
            np.clip(self.grid_height - image_y, 1.0, self.grid_height - 1.0)
        )
        return grid_x, grid_y

    def _resolve_avatar_source(self, world: Path) -> Path | None:
        """Prefer this world's adopted photograph (never a stale hard-coded path)."""
        bundle = getattr(self, "_avatar_bundle", None)
        override = getattr(self.argv, "face_image", None)
        candidates = [
            Path(override) if override else None,
            bundle.source_face if bundle is not None else None,
            world.with_name("source_face.png"),
            world.with_suffix(".png"),
            # Last resort only when no world photo exists
            DEFAULT_AVATAR_SOURCE,
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            path = Path(candidate)
            if path.is_file():
                return path
        return None

    def _mouth_center_from_regions(self) -> tuple[float, float] | None:
        """AMIN steps 5+7 — the digested mouth *object* is the cell address.

        Digestion already clustered the unlocked mouth cells into a region
        object (``region_catalog.json``). Using its centroid means the runtime
        feeds impulses to the exact cells that object owns instead of
        re-deriving the address from grid statistics every launch.
        """
        catalog = self.world_path.with_name("region_catalog.json")
        if not catalog.is_file():
            return None
        try:
            payload = json.loads(catalog.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        best_cells: list | None = None
        best_count = 0
        for region in payload.get("regions") or []:
            if not isinstance(region, dict) or region.get("name") != "mouth_unlocked":
                continue
            cells = region.get("cells") or region.get("cells_sample") or []
            count = int(region.get("cell_count", len(cells)))
            if cells and count > best_count:
                best_cells = cells
                best_count = count
        if not best_cells:
            return None
        xs = np.asarray([cell[0] for cell in best_cells], dtype=np.float64)
        ys = np.asarray([cell[1] for cell in best_cells], dtype=np.float64)
        x = float(xs.mean()) + 0.5
        y = float(ys.mean()) + 0.5
        self._mouth_object_cells = int(best_count)
        print(
            f"Mouth object address from region catalog: {best_count} cells "
            f"@ ({x:.1f}, {y:.1f})"
        )
        return (
            float(np.clip(x, 1.0, self.grid_width - 1.0)),
            float(np.clip(y, 1.0, self.grid_height - 1.0)),
        )

    def _load_cell_clusters(self) -> None:
        """Load face ROI + TickFeed driver (replaces legacy ±4 MouthCellPlan)."""
        try:
            index = CellClusterIndex.from_world(self.world_path)
            self._cell_clusters = index
            mouth = index.primary_mouth()
            if mouth is not None:
                self._mouth_object_cells = int(mouth.cell_count)
                cx, cy = mouth.centroid()
                print(
                    f"Cell clusters: {len(index.clusters)} regions; "
                    f"mouth_unlocked={mouth.cell_count} cells "
                    f"@ ({cx:.1f}, {cy:.1f})"
                )
            else:
                print(
                    f"Cell clusters: {len(index.clusters)} regions (no mouth_unlocked)"
                )
        except (OSError, ValueError, BDSFormatError) as exc:
            print(f"Cell clusters: unavailable ({exc})")
            self._cell_clusters = None
        try:
            face = face_box_from_profile(
                self.world_path, self.grid_width, self.grid_height
            )
            mouth_uv = (
                float(self._mouth_center[0]),
                float(self._mouth_center[1]),
            )
            self._tickfeed = TickFeedDriver.try_load_timeline(
                self.world_path, face, mouth_uv
            )
            if self._tickfeed is not None and bool(
                getattr(self.argv, "wire_loop", False)
            ):
                self._tickfeed.wire_loop = True
                self._tickfeed.wire_loop_source = str(
                    getattr(self.argv, "wire_loop_source", "code") or "code"
                )
            self._tickfeed_look_authority = bool(
                self._tickfeed is not None and self._tickfeed.enabled
            )
            wire = (
                f"wire-loop={self._tickfeed.wire_loop_source}"
                if self._tickfeed is not None and self._tickfeed.wire_loop
                else "local-ring"
            )
            print(
                f"TickFeed: full-face ROI {face.w}x{face.h} @ ({face.x},{face.y}) "
                f"— KEY/DELTA ingest (legacy ±4 cell plan disabled); "
                f"LOOK authority={'tickfeed-labels' if self._tickfeed_look_authority else 'mouth-timeline'}; "
                f"master={wire}"
            )
        except Exception as exc:  # noqa: BLE001 — adopt must not kill launch
            print(f"TickFeed: unavailable ({exc})")
            self._tickfeed = None

    def _estimate_mouth_center(self) -> tuple[float, float]:
        """Locate the mouth from the digested region object, seed metadata,
        part anchors, or unlocked tissue — in that order of authority."""
        from_regions = self._mouth_center_from_regions()
        if from_regions is not None:
            return from_regions
        try:
            header, _grid = load_bds(self.world_path)
            meta = header.get("application_metadata", {}).get("avatar_seed", {})
            mouth = meta.get("mouth_center_image")
            face = meta.get("face_box")
            if mouth and face:
                # Image y points down; grid y points up.
                x = float(mouth["x"]) + 0.5
                y = float(self.grid_height) - float(mouth["y"]) - 0.5
                return (
                    float(np.clip(x, 1.0, self.grid_width - 1.0)),
                    float(np.clip(y, 1.0, self.grid_height - 1.0)),
                )
            if face:
                x = float(face["x"]) + float(face["width"]) * 0.50
                y_image = float(face["y"]) + float(face["height"]) * 0.78
                y = float(self.grid_height) - y_image
                return (
                    float(np.clip(x, 1.0, self.grid_width - 1.0)),
                    float(np.clip(y, 1.0, self.grid_height - 1.0)),
                )
        except (OSError, BDSFormatError, ValueError, KeyError, TypeError):
            pass

        # Anatomical part anchors are the next best source when an older save
        # stripped the seed metadata: they keep lip pieces on the photograph.
        upper = self._part_anchors.get("upper_lip")
        lower = self._part_anchors.get("lower_lip")
        if upper is not None and lower is not None:
            return (
                float((upper[0] + lower[0]) * 0.5),
                float((upper[1] + lower[1]) * 0.5),
            )
        cavity = self._part_anchors.get("mouth_cavity")
        if cavity is not None:
            return float(cavity[0]), float(cavity[1])

        grid = self._read_world()
        unlocked = grid[..., HUMAN_LOCK_CHANNEL] < 0.5
        soft = (grid[..., 3] > 0.2) & unlocked & (grid[..., HARD_SURFACE_CHANNEL] < 0.5)
        # Prefer the lower half of the world (mouth/jaw after the y-flip).
        lower_mask = soft.copy()
        lower_mask[self.grid_height // 2 :, :] = False
        if not lower_mask.any():
            lower_mask = soft
        if not lower_mask.any():
            return (self.grid_width * 0.5, self.grid_height * 0.35)
        rows, columns = np.nonzero(lower_mask)
        # Density-weighted centroid, biased toward the horizontal middle.
        weights = grid[rows, columns, 3].astype(np.float64) + 0.05
        weights *= 1.0 / (1.0 + np.abs(columns - self.grid_width * 0.5))
        x = float(np.average(columns, weights=weights)) + 0.5
        y = float(np.average(rows, weights=weights)) + 0.5
        return x, y

    # --------------------------------------------------------------- render

    def _create_avatar_base_texture(self) -> tuple[moderngl.Texture, moderngl.Texture]:
        """Upload the photograph and the anatomical part-id map.

        AMIN step 11 split them: the photo renders at capture resolution with
        mipmaps (LINEAR minification — hi-res without cache thrash on iGPUs),
        while discrete part ids stay a small NEAREST texture at grid size.
        Packing ids into the photo's alpha forced NEAREST + no mips on both.
        """
        rgba: np.ndarray | None = None
        parts_file = default_parts_path(self.world_path)
        if parts_file.is_file():
            try:
                rgba = load_face_part_atlas(parts_file)
                meta = parts_file.with_suffix(".json")
                if meta.is_file():
                    payload = json.loads(meta.read_text(encoding="utf-8"))
                    anchors = payload.get("anchors", {})
                    self._part_anchors = {
                        str(key): (float(value[0]), float(value[1]))
                        for key, value in anchors.items()
                    }
                self._use_parts = True
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                print(f"warning: could not load face parts ({exc})")
                rgba = None

        if rgba is None and self._avatar_source is not None:
            try:
                from aiface.seed import load_face_image

                image, face = load_face_image(
                    self._avatar_source, self.grid_width, self.grid_height
                )
                atlas = build_face_part_atlas(image, face=face)
                save_face_part_atlas(parts_file, atlas)
                rgba = atlas.rgba
                self._part_anchors = dict(atlas.anchors)
                self._use_parts = True
                print(f"Built face part atlas at {parts_file}")
            except (OSError, ValueError, RuntimeError) as exc:
                print(f"warning: face part atlas unavailable ({exc})")

        if rgba is None:
            rgb = (
                self._load_display_portrait_rgb()
                if self._avatar_source is not None
                else None
            )
            if rgb is None:
                rgb = np.clip(self._read_world()[..., 8:11], 0.0, 1.0)
            rgba = np.empty((*rgb.shape[:2], 4), dtype=np.float32)
            rgba[..., :3] = rgb
            rgba[..., 3] = 0.1  # everything reads as static face
            self._use_parts = False

        # Photo texture at display resolution (same registered crop as seed).
        photo_rgb = self._load_display_portrait_rgb()
        if photo_rgb is None or (
            photo_rgb.shape[0] <= rgba.shape[0] and photo_rgb.shape[1] <= rgba.shape[1]
        ):
            photo_rgb = np.asarray(rgba[..., :3], dtype=np.float32)
        else:
            print(
                f"Avatar base texture: {photo_rgb.shape[1]}x{photo_rgb.shape[0]} "
                f"display res (field grid stays {self.grid_width}x{self.grid_height})"
            )
        photo = np.empty((*photo_rgb.shape[:2], 4), dtype=np.uint8)
        photo[..., :3] = np.rint(np.clip(photo_rgb, 0.0, 1.0) * 255.0)
        photo[..., 3] = 255
        photo_texture = self.ctx.texture(
            (photo.shape[1], photo.shape[0]),
            components=4,
            data=np.ascontiguousarray(photo).tobytes(),
            dtype="f1",
        )
        # No mipmaps: warped / rest-aligned samples at non-integer UVs were
        # pulling soft mip levels and reading as mouth blur next to plates.
        photo_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        photo_texture.repeat_x = False
        photo_texture.repeat_y = False

        # Part ids stay grid-sized, NEAREST, un-mipmapped — discrete labels.
        ids = np.ascontiguousarray(
            np.rint(np.clip(rgba[..., 3], 0.0, 1.0) * 255.0), dtype=np.uint8
        )
        parts_texture = self.ctx.texture(
            (ids.shape[1], ids.shape[0]),
            components=1,
            data=ids.tobytes(),
            dtype="f1",
        )
        parts_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        parts_texture.repeat_x = False
        parts_texture.repeat_y = False
        return photo_texture, parts_texture

    def _load_display_portrait_rgb(self) -> np.ndarray | None:
        """``source_face.png`` at capture resolution (≤ cap), world y-up."""
        if self._avatar_source is None:
            return None
        try:
            from PIL import Image

            image_rgb = Image.open(self._avatar_source).convert("RGB")
        except (OSError, ValueError) as exc:
            print(f"warning: display-res portrait unavailable ({exc})")
            return None
        if max(image_rgb.size) > MAX_DISPLAY_TEXTURE:
            scale = MAX_DISPLAY_TEXTURE / max(image_rgb.size)
            image_rgb = image_rgb.resize(
                (
                    max(round(image_rgb.size[0] * scale), 1),
                    max(round(image_rgb.size[1] * scale), 1),
                ),
                Image.Resampling.LANCZOS,
            )
        return np.flipud(np.asarray(image_rgb, dtype=np.float32) / 255.0)

    def _create_tissue_texture(self) -> moderngl.Texture:
        """Load or bake the per-cell deformation maps the warp reads."""
        rgba: np.ndarray | None = None
        tissue_file = default_tissue_path(self.world_path)
        if tissue_file.is_file():
            try:
                rgba = load_tissue_maps(tissue_file)
            except (OSError, ValueError) as exc:
                print(f"warning: could not load tissue maps ({exc})")
            else:
                self._adopt_tissue_face_box(tissue_file.with_suffix(".json"))

        if rgba is None:
            tissue = build_tissue_maps(
                self.grid_height,
                self.grid_width,
                self._face_box_rect(),
                self._biomech.definition,
                landmarks=self._landmarks_from_seed(),
            )
            rgba = tissue.rgba
            try:
                save_tissue_maps(tissue_file, tissue, face=self._face_box_rect())
            except OSError as exc:
                print(f"warning: could not cache tissue maps ({exc})")
            else:
                print(f"Baked tissue maps at {tissue_file}")

        expected = (self.grid_height, self.grid_width)
        if rgba.shape[:2] != expected:
            raise ValueError(
                f"Tissue maps are {rgba.shape[:2]}, runtime grid is {expected}"
            )

        texture = self.ctx.texture(
            (self.grid_width, self.grid_height),
            components=4,
            data=np.ascontiguousarray(rgba, dtype=np.float32).tobytes(),
            dtype="f4",
        )
        # Linear: these are continuous fields, and interpolating them is what
        # keeps the warp smooth between cells.
        texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        texture.repeat_x = False
        texture.repeat_y = False
        self._use_tissue = True
        return texture

    def _create_expression_plate_textures(
        self,
    ) -> tuple[moderngl.Texture, moderngl.Texture]:
        """Load smile/open plates from aiface-capture, or bind dark placeholders."""
        from aiface.capture import default_open_plate_path, default_smile_plate_path

        open_path = default_open_plate_path(self.world_path)
        smile_path = default_smile_plate_path(self.world_path)
        open_rgba = self._load_plate_rgba(open_path)
        smile_rgba = self._load_plate_rgba(smile_path)
        self._use_plates = open_rgba is not None and smile_rgba is not None
        if open_rgba is None:
            open_rgba = np.zeros(
                (self.grid_height, self.grid_width, 4), dtype=np.float32
            )
        if smile_rgba is None:
            smile_rgba = np.zeros(
                (self.grid_height, self.grid_width, 4), dtype=np.float32
            )
        if self._use_plates:
            print(
                f"Expression plates: {open_path.name} + {smile_path.name} "
                "(real capture interior)"
            )
        return (
            self._upload_rgba_texture(open_rgba),
            self._upload_rgba_texture(smile_rgba),
        )

    def _create_expression_catalog_texture(self) -> moderngl.Texture:
        """Load expression_catalog.json + surprise plate as the upper-face DB."""
        from aiface.expression_catalog import load_expression_catalog

        blank = np.zeros((self.grid_height, self.grid_width, 4), dtype=np.float32)
        catalog = load_expression_catalog(self.world_path)
        self._expression_catalog = catalog
        if catalog is None:
            return self._upload_rgba_texture(blank)
        role = catalog.roles.get("surprise") or catalog.role_for_emotion("SURPRISED")
        rgba = None
        if role is not None and role.plate:
            plate_path = self.world_path.with_name(Path(role.plate).name)
            if not plate_path.is_file():
                plate_path = self.world_path.parent / role.plate
            rgba = self._load_plate_rgba(plate_path)
        if rgba is None:
            print(
                "Expression catalog loaded "
                f"({len(catalog.roles)} roles) — no surprise plate yet"
            )
            return self._upload_rgba_texture(blank)
        print(
            f"Expression catalog: {len(catalog.roles)} roles "
            f"(surprise plate {role.plate if role else '?'})"
        )
        return self._upload_rgba_texture(rgba)

    def _active_emotion(self) -> str:
        """Prefer a non-NEUTRAL mood; telemetry defaults to NEUTRAL (truthy)."""
        tel = (self._telemetry.last_emotion or "").strip().upper()
        conv = (self._conversation.last_emotion or "").strip().upper()
        if tel and tel != "NEUTRAL":
            return tel
        if conv:
            return conv
        return tel or "NEUTRAL"

    def _sync_expression_from_emotion(self) -> None:
        """Drive eye widen / brow raise / upper plate from catalog + biomech."""
        catalog = self._expression_catalog
        emotion = self._active_emotion()
        role = catalog.role_for_emotion(emotion) if catalog is not None else None
        biomech_widen = float(getattr(self._render_state, "eye_widen", 0.0))
        biomech_brow = float(getattr(self._render_state, "brow_raise", 0.0))
        phoneme = str(
            getattr(self._mouth_timeline, "phoneme", None)
            or getattr(self._render_state, "last_phoneme", "REST")
            or "REST"
        ).upper()
        speaking = bool(getattr(self._render_state, "speaking", False)) or bool(
            self._visemes
        )
        if role is None:
            self._expr_target_widen = biomech_widen
            self._expr_target_brow = biomech_brow
            self._expr_target_blend = 0.0
            self._expr_role_name = "rest"
        else:
            self._expr_role_name = role.name
            # Catalog params are learned from video; biomech adds live muscle drive.
            self._expr_target_widen = max(float(role.eye_widen), biomech_widen)
            self._expr_target_brow = max(float(role.brow_raise), biomech_brow)
            # Upper-face plate only when ownership allows (SURPRISED).
            if role.name == "surprise" and (
                self._mouth_ownership.upper_expr_plate
                or emotion in {"SURPRISED", "SURPRISE"}
            ):
                self._expr_target_blend = max(
                    0.55, float(role.eye_widen), float(role.brow_raise)
                )
            elif role.name == "smile":
                self._expr_target_blend = 0.0
                self._expr_target_widen = max(self._expr_target_widen, 0.12)
                # Catalog smile brow is 0 — floor so HAPPY speech lifts brows.
                self._expr_target_brow = max(self._expr_target_brow, 0.30)
            else:
                self._expr_target_blend = 0.0
            if not self._mouth_ownership.upper_expr_plate:
                self._expr_target_blend = 0.0

        # Speech / emphasis brow so NEUTRAL talk still moves the forehead.
        open_brows = {"AH", "AA", "OH", "EH", "EE", "OU"}
        if emotion in {"SURPRISED", "SURPRISE"}:
            self._expr_target_brow = max(self._expr_target_brow, 0.55)
            self._expr_target_widen = max(self._expr_target_widen, 0.50)
        elif emotion == "HAPPY":
            self._expr_target_brow = max(self._expr_target_brow, 0.28)
            self._expr_target_widen = max(self._expr_target_widen, 0.10)
        if speaking and phoneme in open_brows:
            self._expr_target_brow = max(self._expr_target_brow, 0.24)
            self._expr_target_widen = max(self._expr_target_widen, 0.08)
        elif speaking and phoneme not in {"REST", "CLOSED", "PP", "MM"}:
            self._expr_target_brow = max(self._expr_target_brow, 0.14)

        # Hearing / waiting look — closed lips, attentive brows + soft widen.
        waiting_look = (
            self._presence == PRESENCE_HEARING
            or self._tickfeed_live_mode() == "hearing"
            or (
                self._presence == PRESENCE_ZERO
                and str(getattr(self, "_zero_mood", "")) == ZERO_MOOD_WAITING
            )
        )
        if waiting_look:
            self._expr_target_brow = max(self._expr_target_brow, 0.45)
            self._expr_target_widen = max(self._expr_target_widen, 0.30)
            if not self._tickfeed_look_authority:
                self._expr_target_blend = max(self._expr_target_blend, 0.18)
        elif (
            self._presence == PRESENCE_ZERO
            and str(getattr(self, "_zero_mood", "")) == ZERO_MOOD_NEUTRAL
        ):
            # No-impression idle — flatten residual brow/widen from prior speech.
            self._expr_target_brow = 0.0
            self._expr_target_widen = 0.0
            self._expr_target_blend = 0.0

    def _ease_expression_state(self, frame_time: float) -> None:
        rate = 4.5
        amount = 1.0 - math.exp(-max(frame_time, 0.0) * rate)
        self._expr_eye_widen += (self._expr_target_widen - self._expr_eye_widen) * amount
        self._expr_brow_raise += (self._expr_target_brow - self._expr_brow_raise) * amount
        if self._tickfeed_look_authority:
            # B4: TickPackage surprise_amt already set _expr_plate_blend — do not
            # ease it back toward the emotion-catalog target.
            return
        self._expr_plate_blend += (
            self._expr_target_blend - self._expr_plate_blend
        ) * amount

    def _load_plate_atlas_textures(self) -> None:
        """Load the viseme plate bank (movement memory) next to the world."""
        from aiface.plates import load_plate_atlas

        atlas = load_plate_atlas(self.world_path)
        self._plate_atlas = atlas
        self._atlas_textures = []
        if atlas is None:
            # Placeholders so the samplers always bind.
            blank = np.zeros(
                (self.grid_height, self.grid_width, 4), dtype=np.float32
            )
            self._atlas_textures = [
                self._upload_rgba_texture(blank),
                self._upload_rgba_texture(blank),
            ]
            return
        root = self.world_path.parent
        for plate in atlas.plates:
            path = root / plate.path
            rgba = self._load_plate_rgba(path)
            if rgba is None:
                rgba = np.zeros(
                    (self.grid_height, self.grid_width, 4), dtype=np.float32
                )
            self._atlas_textures.append(self._upload_rgba_texture(rgba))
        if not self._atlas_textures:
            blank = np.zeros(
                (self.grid_height, self.grid_width, 4), dtype=np.float32
            )
            self._atlas_textures = [self._upload_rgba_texture(blank)]
        self._use_plates = True
        print(
            f"Plate atlas: {len(self._atlas_textures)} mouth shapes "
            "(viseme-timed plate memory)"
        )

    def _sync_plate_blend_from_phoneme(self) -> None:
        """Upload atlas pair + open amount from TickFeed labels or LayerCommand."""
        atlas = self._plate_atlas
        if atlas is None or not self._atlas_textures:
            self._plate_blend = (0.0, 0.0)
            self._plate_blend_current = (0.0, 0.0)
            self._plate_openness_current = 0.0
            self._plate_pair = (0, 0)
            self._held_speech_viseme = "REST"
            return
        from aiface.plates import HARD_SNAP_THRESHOLD

        # Design B4: when TickFeed is live, package labels own LOOK amounts.
        # Plate identity = viseme (hard snap); intensity = openness weight.
        if (
            self._tickfeed_look_authority
            and self._tickfeed is not None
            and self._tickfeed.last_labels is not None
        ):
            phoneme = str(self._held_speech_viseme or "REST")
            open_amt = float(getattr(self, "_plate_open_hyst", 0.0))
            # Early commitment during OPENING/CLOSING — no soft A/B ghost.
            snap = True
            ia, ib, mix = atlas.pair_for_viseme(phoneme, hard_snap=snap)
            ia = max(0, min(ia, len(self._atlas_textures) - 1))
            ib = max(0, min(ib, len(self._atlas_textures) - 1))
            self._plate_pair = (ia, ib)
            self._plate_openness_current = open_amt
            self._plate_blend = (float(mix), open_amt)
            self._plate_blend_current = (float(mix), open_amt)
            return

        cmd = self._layer_command
        phoneme = cmd.atlas_viseme
        self._held_speech_viseme = phoneme
        hard = float(self._display_recipe.plate_sharpness) >= HARD_SNAP_THRESHOLD
        if hard:
            ia, ib, mix = atlas.pair_for_viseme(phoneme, hard_snap=True)
        else:
            ia, ib, mix = atlas.pair_for_openness(
                float(cmd.plate_openness), hard_snap=False
            )
        ia = max(0, min(ia, len(self._atlas_textures) - 1))
        ib = max(0, min(ib, len(self._atlas_textures) - 1))
        self._plate_pair = (ia, ib)
        amount = float(cmd.open_amount)
        self._plate_openness_current = float(cmd.plate_openness)
        self._plate_blend = (float(mix), amount)
        self._plate_blend_current = (float(mix), amount)

    def _tickfeed_live_active(self) -> bool:
        live = self._tickfeed_live
        if live is None:
            return False
        now = self._speech_now()
        return now <= float(live.get("until", 0.0))

    def _tickfeed_live_mode(self) -> str:
        live = self._tickfeed_live
        if live is None or not self._tickfeed_live_active():
            return ""
        return str(live.get("mode") or "speech").strip().lower()

    def _enter_zero_state(self, *, blink: bool = False) -> None:
        """Idle 0-state: closed lips, soft gaze, optional blink."""
        was = self._presence
        self._presence = PRESENCE_ZERO
        live = self._tickfeed_live
        if live is not None and str(live.get("mode") or "") in {
            "hearing",
            "speech",
            "zero",
        }:
            # Clear speech/hearing overlay; zero-mood drives reapply each tick.
            if str(live.get("mode") or "") != "zero":
                self._tickfeed_live = None
        # Drop held open plate immediately — hysteresis must not park open.png
        # after speech ends (full-cycle QA: idle still o≈0.9).
        self._plate_open_hyst = 0.0
        self._plate_openness_current = 0.0
        if hasattr(self, "_biomech"):
            gaze_x, gaze_y = 0.0, 0.0
            if str(getattr(self, "_zero_mood", "")) == ZERO_MOOD_WAITING:
                gaze_x, gaze_y = 0.18, 0.06
            self._biomech.eyes.look_at(gaze_x, gaze_y)
            if blink and was != PRESENCE_ZERO:
                self._biomech.eyes.request_blink()
        if was == PRESENCE_SPEAKING:
            self._telemetry.last_emotion = "NEUTRAL"
            self._held_speech_viseme = "REST"
        self._apply_zero_mood_overlay()

    def _zero_mood_drives(
        self,
    ) -> tuple[float, float, float, str, str]:
        """Return open, smile, surprise, phoneme, emotion for idle 0-state mood."""
        mood = str(getattr(self, "_zero_mood", ZERO_MOOD_NEUTRAL) or ZERO_MOOD_NEUTRAL)
        if mood == ZERO_MOOD_SMILE:
            return 0.0, 0.62, 0.0, "REST", "HAPPY"
        if mood == ZERO_MOOD_WAITING:
            return 0.0, 0.0, 0.14, "REST", "THINKING"
        # neutral / no impression
        return 0.0, 0.0, 0.0, "REST", "NEUTRAL"

    def _apply_zero_mood_overlay(self) -> None:
        """Keep TickFeed labels on the active idle mood while presence is zero."""
        if self._presence != PRESENCE_ZERO:
            return
        if self._tickfeed_live_mode() == "speech":
            return
        now = time.perf_counter() - self._clock0
        open_amt, smile_amt, surprise_amt, phoneme, emotion = self._zero_mood_drives()
        self._tickfeed_live = {
            "phoneme": phoneme,
            "open": open_amt,
            "smile": smile_amt,
            "surprise": surprise_amt,
            "emotion": emotion,
            "mode": "zero",
            "until": now + 1.5,
        }
        self._held_speech_viseme = phoneme
        self._telemetry.last_emotion = emotion

    def _set_zero_mood(self, mood: str, *, announce: bool = True) -> str:
        key = str(mood or "").strip().lower()
        aliases = {
            "neutral": ZERO_MOOD_NEUTRAL,
            "none": ZERO_MOOD_NEUTRAL,
            "no": ZERO_MOOD_NEUTRAL,
            "no_impression": ZERO_MOOD_NEUTRAL,
            "no-impression": ZERO_MOOD_NEUTRAL,
            "flat": ZERO_MOOD_NEUTRAL,
            "smile": ZERO_MOOD_SMILE,
            "happy": ZERO_MOOD_SMILE,
            "waiting": ZERO_MOOD_WAITING,
            "wait": ZERO_MOOD_WAITING,
            "attentive": ZERO_MOOD_WAITING,
        }
        resolved = aliases.get(key, key)
        if resolved not in ZERO_MOODS:
            raise RuntimeError(
                f"zero_mood must be one of {list(ZERO_MOODS)}, got {mood!r}"
            )
        self._zero_mood = resolved
        if self._presence == PRESENCE_ZERO:
            self._apply_zero_mood_overlay()
        if announce:
            labels = {
                ZERO_MOOD_NEUTRAL: "no impression (neutral)",
                ZERO_MOOD_SMILE: "smile",
                ZERO_MOOD_WAITING: "waiting",
            }
            print(f"0-state mood: {labels.get(resolved, resolved)}")
        return resolved

    def _cycle_zero_mood(self) -> str:
        cur = str(getattr(self, "_zero_mood", ZERO_MOOD_NEUTRAL))
        try:
            idx = ZERO_MOODS.index(cur)  # type: ignore[arg-type]
        except ValueError:
            idx = 0
        return self._set_zero_mood(ZERO_MOODS[(idx + 1) % len(ZERO_MOODS)])

    def _enter_hearing_state(self) -> None:
        """Waiting / listening look — lips stay closed, brows lift, soft gaze."""
        now = time.perf_counter() - self._clock0
        if self._tickfeed_live_mode() == "speech":
            # Never clobber an active speech overlay.
            return
        if self._presence != PRESENCE_HEARING and hasattr(self, "_biomech"):
            # Slightly toward the chat panel / attentive focus.
            self._biomech.eyes.look_at(0.22, 0.08)
        self._presence = PRESENCE_HEARING
        self._tickfeed_live = {
            "phoneme": "REST",
            "open": 0.0,
            "smile": 0.0,
            "surprise": 0.15,
            "emotion": "THINKING",
            "mode": "hearing",
            "until": now + 2.0,
        }
        self._held_speech_viseme = "REST"
        self._telemetry.last_emotion = "THINKING"

    def _update_presence(self) -> None:
        """0-state ↔ hearing (typing/pending) ↔ speaking (visemes/TTS)."""
        speech_active = bool(self._visemes) or self._tickfeed_live_mode() == "speech"
        if speech_active:
            self._presence = PRESENCE_SPEAKING
            return

        want_hearing = bool(self._chatbox.pending) or (
            self._chat_box_visible
            and bool(self._chatbox.focused)
            and bool(self._chatbox.text.strip())
        )
        if want_hearing:
            self._enter_hearing_state()
            return

        if self._presence == PRESENCE_SPEAKING:
            # Finishing a chat turn → back to 0-state with a blink.
            self._enter_zero_state(blink=True)
            return

        if self._presence == PRESENCE_HEARING:
            self._enter_zero_state(blink=False)
            return

        if self._presence != PRESENCE_ZERO:
            self._enter_zero_state(blink=False)

    def _layer_command_from_tickfeed(self) -> LayerCommand:
        """Build GPU layer directive from TickPackage labels (not MouthLayerTimeline)."""
        from aiface.tickfeed.schema import VISEME_TABLE

        labels = (
            self._tickfeed.last_labels
            if self._tickfeed is not None
            else None
        )
        open_amt = float(labels.open_amt) if labels is not None else 0.0
        smile_amt = float(labels.smile_amt) if labels is not None else 0.0
        phoneme = str(self._held_speech_viseme or "REST")
        if labels is not None and getattr(labels, "viseme_id", None) is not None:
            vid = int(labels.viseme_id)
            if 0 <= vid < len(VISEME_TABLE):
                phoneme = VISEME_TABLE[vid]
        src = "tickfeed-live" if self._tickfeed_live_active() else "tickfeed-labels"
        # FIELD (TickPackage ch0/1) owns lip separation. Jaw muscle warp only
        # pulls the lower lip and stacked on residual flow as "lip slides down".
        return LayerCommand(
            phoneme=phoneme,
            atlas_viseme=phoneme,
            open_amount=open_amt,
            smile_amount=smile_amt,
            jaw_target=0.0,
            plate_openness=open_amt,
            source=src,
            active_until=float(
                self._tickfeed_live.get("until", 0.0)
                if self._tickfeed_live is not None
                else 0.0
            ),
        )

    def _update_open_close_ml(self, frame_time: float) -> None:
        """LOOK: TickFeed labels when enabled; else MouthLayerTimeline."""
        del frame_time  # Layer visibility snaps; no openness ease clock.
        from aiface.biomechanics.intent import PHONEME_JAW_TARGET
        from aiface.plates import HARD_SNAP_THRESHOLD

        now = time.perf_counter() - self._clock0
        if self._tickfeed_look_authority and self._tickfeed is not None:
            # Expire live overlay
            if self._tickfeed_live is not None and not self._tickfeed_live_active():
                self._tickfeed_live = None
            cmd = self._layer_command_from_tickfeed()
            self._layer_command = cmd
            self._ml_openness = float(cmd.plate_openness)
            self._ml_smile = float(cmd.smile_amount)
            self._plate_openness_current = float(cmd.plate_openness)
            self._held_speech_viseme = cmd.atlas_viseme
            self._open_close_source = cmd.source
            self._ml_jaw = float(cmd.jaw_target)
            self._biomech.jaw.set_speech_target(float(cmd.jaw_target))
            phoneme = cmd.phoneme
            phoneme_jaw = float(
                self._condition_jaw.get(
                    phoneme, PHONEME_JAW_TARGET.get(phoneme, 0.1)
                )
            )
            controls = self._live_vector.resolve(
                phoneme=phoneme, phoneme_jaw=phoneme_jaw
            )
            self._ml_width = float(controls.width_n)
            self._ml_plate_gate = float(controls.plate_gate)
            self._refresh_frame_layers()
            return

        envelope = self._open_close_envelope
        start = self._open_close_start
        if envelope is not None and start is not None:
            t = now - float(start)
            if 0.0 <= t <= float(envelope.duration) + 0.05:
                self._live_vector.push_from_envelope(envelope, t)
                self._behavior.push_from_envelope(envelope, t)
            else:
                self._live_vector.push_rms(0.0)
                self._behavior.push_rms(0.0)
            self._open_close_peak = float(self._live_vector.peak_hint)
        elif not self._live_vector.has_history:
            self._live_vector.push_rms(0.0)
            if not self._behavior.has_history:
                self._behavior.push_rms(0.0)

        phoneme = self._mouth_timeline.phoneme
        phoneme_jaw = float(
            self._condition_jaw.get(
                phoneme, PHONEME_JAW_TARGET.get(phoneme, 0.1)
            )
        )
        controls = self._live_vector.resolve(
            phoneme=phoneme, phoneme_jaw=phoneme_jaw
        )
        recipe = self._display_recipe
        hard = float(recipe.plate_sharpness) >= HARD_SNAP_THRESHOLD
        upcoming_due, upcoming_phoneme = self._upcoming_viseme()
        cmd = self._mouth_timeline.tick(
            now,
            width_n=float(controls.width_n),
            jaw_table=self._condition_jaw,
            smile_width_start=float(recipe.smile_width_start),
            smile_width_span=float(recipe.smile_width_span),
            smile_happy_floor=float(recipe.smile_happy_floor),
            hard_snap=hard,
            upcoming_due_at=upcoming_due,
            upcoming_phoneme=upcoming_phoneme,
        )
        behavior = self._behavior.resolve(
            phoneme=cmd.phoneme,
            video_t=None,
            smile_amount=float(cmd.smile_amount),
            open_amount=float(cmd.plate_openness),
        )
        if behavior.source.startswith(
            ("ml_fill", "measured", "observed", "heuristic")
        ):
            from aiface.live_vector.schema import LiveControlVector

            controls = LiveControlVector(
                openness_n=max(float(controls.openness_n), float(behavior.openness_n)),
                jaw_n=max(float(controls.jaw_n), float(behavior.jaw_n)),
                width_n=max(float(controls.width_n), float(behavior.width_n)),
                plate_gate=float(controls.plate_gate),
                source=f"{controls.source}+{behavior.source}",
            )
        self._layer_command = cmd
        self._ml_openness = float(cmd.plate_openness)
        self._ml_jaw = float(cmd.jaw_target)
        self._ml_width = float(controls.width_n)
        self._ml_plate_gate = float(controls.plate_gate)
        self._ml_smile = float(cmd.smile_amount)
        self._plate_openness_current = float(cmd.plate_openness)
        self._held_speech_viseme = cmd.atlas_viseme
        self._open_close_source = cmd.source
        self._biomech.jaw.set_speech_target(float(cmd.jaw_target))
        self._refresh_frame_layers()

    def _refresh_frame_layers(self) -> FrameLayerState:
        """Resolve L00–L11 active flags once per tick (realtime skip map)."""
        from aiface.plates import HARD_SNAP_THRESHOLD
        from aiface.speech import canonical_viseme

        cmd = self._layer_command
        recipe = self._display_recipe
        phoneme = canonical_viseme(cmd.phoneme or "REST")
        # TickFeed owns FIELD; L03 cell_groups steps unused (0).
        speaking_plate = float(cmd.plate_openness) > 0.02 or bool(
            getattr(self._render_state, "speaking", False)
        )
        self._frame_layers = evaluate_frame_layers(
            phoneme=phoneme,
            plate_open_amount=float(cmd.plate_openness),
            smile_amount=float(cmd.smile_amount),
            atlas_strength=float(recipe.atlas_strength),
            cavity_strength=float(recipe.cavity_strength),
            field_gain=float(recipe.field_warp_gain),
            expr_blend=float(self._expr_plate_blend),
            brow_raise=float(self._expr_brow_raise),
            speaking_plate=speaking_plate,
            cell_plan_steps=0,
            hard_snap=float(recipe.plate_sharpness) >= HARD_SNAP_THRESHOLD,
            chat_visible=bool(self._chat_box_visible),
        )
        return self._frame_layers

    def _ease_plate_blend(self, frame_time: float) -> None:
        """Refresh plate texture pair from the eased ML openness."""
        if self._plate_atlas is None:
            return
        self._sync_plate_blend_from_phoneme()

    def _load_plate_rgba(self, path: Path) -> np.ndarray | None:
        if not path.is_file():
            return None
        try:
            from PIL import Image

            image = Image.open(path).convert("RGBA")
            # AMIN step 11: keep capture resolution — the shader samples by
            # UV, so plates no longer get crushed to the 256² cell grid.
            if max(image.size) > MAX_DISPLAY_TEXTURE:
                scale = MAX_DISPLAY_TEXTURE / max(image.size)
                image = image.resize(
                    (
                        max(round(image.size[0] * scale), 1),
                        max(round(image.size[1] * scale), 1),
                    ),
                    Image.Resampling.LANCZOS,
                )
            # Match photo_at orientation (world y up → flip for texture).
            rgba = np.flipud(np.asarray(image, dtype=np.float32) / 255.0)
            return np.ascontiguousarray(rgba, dtype=np.float32)
        except (OSError, ValueError) as exc:
            print(f"warning: could not load expression plate {path.name} ({exc})")
            return None

    def _upload_rgba_texture(
        self, rgba: np.ndarray, *, mipmaps: bool = False
    ) -> moderngl.Texture:
        # Plates are photographs — 8-bit normalized keeps hi-res display
        # textures inside integrated-GPU bandwidth (f4 was 4× the bytes).
        # Default: LINEAR without mipmaps so lip edges stay sharp (mips were
        # a residual soft-focus source on open/smile/atlas). Photo base keeps
        # its own mipmapped upload path.
        data = np.clip(rgba, 0.0, 1.0)
        data = np.ascontiguousarray(np.rint(data * 255.0), dtype=np.uint8)
        texture = self.ctx.texture(
            (rgba.shape[1], rgba.shape[0]),
            components=4,
            data=data.tobytes(),
            dtype="f1",
        )
        if mipmaps:
            texture.build_mipmaps()
            texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        else:
            texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        texture.repeat_x = False
        texture.repeat_y = False
        return texture

    def _apply_capture_priors(self) -> None:
        """Scale jaw travel from talk-segment priors baked by aiface-capture."""
        try:
            header, _grid = load_bds(self.world_path)
        except (OSError, BDSFormatError, ValueError):
            return
        seed = header.get("application_metadata", {}).get("avatar_seed", {})
        capture = seed.get("capture") if isinstance(seed, dict) else None
        if not isinstance(capture, dict):
            return
        priors = capture.get("priors")
        if not isinstance(priors, dict):
            return
        jaw_scale = float(priors.get("jaw_travel_scale", 1.0))
        width_scale = float(priors.get("lip_width_scale", 1.0))
        open_scale = float(priors.get("lip_open_scale", 1.0))
        self._capture_priors = {
            "jaw_travel_scale": jaw_scale,
            "lip_width_scale": width_scale,
            "lip_open_scale": open_scale,
        }
        self._biomech.apply_capture_priors(
            jaw_travel_scale=jaw_scale,
            lip_width_scale=width_scale,
            lip_open_scale=open_scale,
        )
        print(
            f"Capture priors: jaw×{jaw_scale:.2f} width×{width_scale:.2f} "
            f"open×{open_scale:.2f}"
        )

    def _face_box_rect(self) -> "FaceBox":
        from aiface.seed import FaceBox

        box = self._face_box
        return FaceBox(
            int(round(box["x"])),
            int(round(box["y"])),
            int(round(box["width"])),
            int(round(box["height"])),
        )

    def _adopt_tissue_face_box(self, meta_path: Path) -> None:
        """Trust the rectangle the maps were baked from over the world header.

        Muscle anchors are face-box UV. Packing them against a different box
        than the bake used would slide every muscle off its own tissue.
        """
        if not meta_path.is_file():
            return
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        box = payload.get("face_box")
        if not isinstance(box, dict):
            return
        self._face_box = {
            key: float(box.get(key, self._face_box[key]))
            for key in ("x", "y", "width", "height")
        }

    def _seed_landmark_meta(self) -> dict[str, object]:
        try:
            header, _grid = load_bds(self.world_path)
        except (OSError, BDSFormatError, ValueError):
            return {}
        seed = header.get("application_metadata", {}).get("avatar_seed", {})
        landmarks = seed.get("landmarks")
        return landmarks if isinstance(landmarks, dict) else {}

    def _landmarks_from_seed(self):
        """Rebuild landmark centres for tissue bake when maps are missing."""
        from aiface.landmarks import FaceLandmarks

        meta = self._seed_landmark_meta()
        if not meta:
            return None
        try:
            return FaceLandmarks(
                face=self._face_box_rect(),
                left_eye=(
                    float(meta["left_eye_image"]["x"]),  # type: ignore[index]
                    float(meta["left_eye_image"]["y"]),  # type: ignore[index]
                ),
                right_eye=(
                    float(meta["right_eye_image"]["x"]),  # type: ignore[index]
                    float(meta["right_eye_image"]["y"]),  # type: ignore[index]
                ),
                mouth=(
                    float(meta["mouth_center_image"]["x"]),  # type: ignore[index]
                    float(meta["mouth_center_image"]["y"]),  # type: ignore[index]
                ),
                left_brow=(
                    float(meta.get("left_brow_image", meta["left_eye_image"])["x"]),  # type: ignore[index]
                    float(meta.get("left_brow_image", meta["left_eye_image"])["y"]),  # type: ignore[index]
                ),
                right_brow=(
                    float(meta.get("right_brow_image", meta["right_eye_image"])["x"]),  # type: ignore[index]
                    float(meta.get("right_brow_image", meta["right_eye_image"])["y"]),  # type: ignore[index]
                ),
                method=str(meta.get("method", "seed")),
                quality=float(meta.get("quality", 0.0)),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def _seed_mouth_center_image(self) -> tuple[float, float] | None:
        meta = self._seed_landmark_meta()
        mouth = meta.get("mouth_center_image")
        if isinstance(mouth, dict) and "x" in mouth and "y" in mouth:
            return float(mouth["x"]), float(mouth["y"])
        return None

    def _seed_eye_centers_grid(
        self,
    ) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """Image-space seed eyes converted to grid y-up coordinates."""
        meta = self._seed_landmark_meta()
        left = meta.get("left_eye_image")
        right = meta.get("right_eye_image")
        if not isinstance(left, dict) or not isinstance(right, dict):
            return None
        try:
            left_xy = (float(left["x"]), float(left["y"]))
            right_xy = (float(right["x"]), float(right["y"]))
        except (KeyError, TypeError, ValueError):
            return None
        height = float(self.grid_height)

        def to_grid(point: tuple[float, float]) -> tuple[float, float]:
            return (
                float(np.clip(point[0], 1.0, self.grid_width - 1.0)),
                float(np.clip(height - point[1], 1.0, self.grid_height - 1.0)),
            )

        return to_grid(left_xy), to_grid(right_xy)

    def _derive_face_geometry(self) -> None:
        """Resolve the definition's UV landmarks into grid-space uniforms."""
        definition = self._biomech.definition
        box = self._face_box
        height = float(self.grid_height)

        def to_grid(uv: Sequence[float]) -> tuple[float, float]:
            return (
                box["x"] + float(uv[0]) * box["width"],
                height - (box["y"] + float(uv[1]) * box["height"]),
            )

        mouth_config = definition.get("mouth_line", {})
        mouth_uv = definition.get("mouth_center", [0.50, 0.78])
        line_x, line_y = to_grid(mouth_uv)
        self._mouth_line = (
            line_x,
            line_y,
            float(mouth_config.get("half_width", 0.20)) * box["width"],
            float(mouth_config.get("softness_cells", 1.6)),
        )

        self._jaw = jaw_pose_from_definition(definition, box, self.grid_height)

        eye_config = definition.get("eye_shape", {})
        eye_positions = definition.get("eye_positions", {})
        self._eye_shape = (
            float(eye_config.get("half_width", 0.10)) * box["width"],
            float(eye_config.get("half_height", 0.040)) * box["height"],
            float(eye_config.get("gaze_travel_cells", 2.6)),
            0.0,
        )
        # Prefer seed-measured eyes (Path 1). Part anchors are diagnostic only —
        # they used to fight the definition UV and jumble the lids.
        measured = self._seed_eye_centers_grid()
        if measured is not None:
            left, right = measured
        else:
            left = to_grid(eye_positions.get("left", [0.30, 0.472]))
            right = to_grid(eye_positions.get("right", [0.70, 0.472]))
        if left[0] > right[0]:
            left, right = right, left
        self._eye_centers = (
            float(left[0]),
            float(left[1]),
            float(right[0]),
            float(right[1]),
        )

        # Mouth line follows measured lip centre when the seed recorded one.
        measured_mouth = self._seed_mouth_center_image()
        if measured_mouth is not None:
            line_x = measured_mouth[0]
            line_y = float(self.grid_height) - measured_mouth[1]
            self._mouth_line = (
                line_x,
                line_y,
                float(mouth_config.get("half_width", 0.20)) * box["width"],
                float(mouth_config.get("softness_cells", 1.6)),
            )

    def _update_avatar_uniforms(self) -> None:
        pose = self._mouth_pose
        state = self._render_state
        program = self.render_program

        self._avatar_base_texture.use(location=2)
        program["avatar_base_color"].value = 2
        self._avatar_parts_texture.use(location=9)
        program["avatar_part_ids"].value = 9
        self._avatar_tissue_texture.use(location=3)
        program["avatar_tissue"].value = 3
        self._avatar_open_plate_texture.use(location=4)
        program["avatar_open_plate"].value = 4
        self._avatar_smile_plate_texture.use(location=5)
        program["avatar_smile_plate"].value = 5
        self._sync_plate_blend_from_phoneme()
        ia, ib = self._plate_pair
        plate_a = self._atlas_textures[ia] if self._atlas_textures else self._avatar_open_plate_texture
        plate_b = self._atlas_textures[ib] if self._atlas_textures else self._avatar_smile_plate_texture
        plate_a.use(location=6)
        program["avatar_plate_a"].value = 6
        plate_b.use(location=7)
        program["avatar_plate_b"].value = 7
        blend = self._plate_blend_current
        program["avatar_plate_blend"].value = (
            float(blend[0]),
            float(blend[1]),
            0.0,
            0.0,
        )
        program["avatar_plates_ready"].value = 1 if self._use_plates else 0
        self._avatar_expr_plate_texture.use(location=8)
        program["avatar_expr_plate"].value = 8
        # Blink owns the aperture — do not upload widen/brow fight during close.
        blink_close = float(
            1.0 - min(float(state.lid_left), float(state.lid_right))
        )
        widen_upload = float(self._expr_eye_widen)
        brow_upload = float(self._expr_brow_raise)
        if blink_close > 0.05:
            widen_upload *= max(0.0, 1.0 - blink_close * 1.15)
            brow_upload *= max(0.0, 1.0 - blink_close * 0.85)
        program["avatar_expr_state"].value = (
            widen_upload,
            brow_upload,
            float(self._expr_plate_blend),
            1.0 if self._expression_catalog is not None else 0.0,
        )

        # TickFeed owns face FIELD + plates; muscle warp on top made lips muddy
        # and fought KEY clears. Keep lid geometry only when TickFeed is live.
        if self._tickfeed_look_authority:
            uniforms = pack_muscle_uniforms(
                self._biomech.registry,
                {},
                face_box=self._face_box,
                grid_height=self.grid_height,
                capacity=MAX_ACTIVE_MUSCLES,
            )
        else:
            uniforms = pack_muscle_uniforms(
                self._biomech.registry,
                state.muscle_activations,
                face_box=self._face_box,
                grid_height=self.grid_height,
                capacity=MAX_ACTIVE_MUSCLES,
            )
        program["muscle_geometry"].write(uniforms.geometry.tobytes())
        program["muscle_drive"].write(uniforms.drive.tobytes())
        program["muscle_count"].value = uniforms.count
        self._active_muscle_count = uniforms.count

        program["avatar_mouth_line"].value = self._mouth_line
        jaw = replace(self._jaw, angle=float(state.jaw_angle))
        program["avatar_jaw"].value = jaw.uniform
        program["avatar_jaw_span"].value = jaw.span_uniform
        program["avatar_eye_shape"].value = self._eye_shape
        program["avatar_eye_centers"].value = self._eye_centers
        program["avatar_eye_state"].value = (
            float(state.gaze_x),
            float(state.gaze_y),
            float(state.pupil),
            float(1.0 - min(state.lid_left, state.lid_right)),
        )
        program["avatar_mouth_pose"].value = (
            pose.width,
            pose.openness,
            pose.roundness,
            pose.expression,
        )
        # Digested GPU display recipe knobs (same contract at train and play).
        from aiface.plates import HARD_SNAP_THRESHOLD

        knobs = list(self._display_recipe.shader_knobs)
        held = str(getattr(self, "_held_speech_viseme", "REST") or "REST")
        speaking_plate = (
            float(self._plate_blend_current[1]) > 0.45
            and held not in {"REST", "CLOSED", ""}
            and float(self._display_recipe.plate_sharpness) >= HARD_SNAP_THRESHOLD
        )
        if speaking_plate:
            # Atlas must fully own the mouth while a speech plate is active.
            knobs[2] = max(float(knobs[2]), 1.0)
        program["avatar_recipe"].value = (
            float(knobs[0]),
            float(knobs[1]),
            float(knobs[2]),
            float(knobs[3]),
        )
        # NWR field warp: validated ±4 impulses integrated on the GPU move
        # unlocked tissue directly (steps 1–3 close the loop at the pixels).
        # While a speech plate owns the mouth, keep field travel low so the
        # unlocked disc does not smear under the static atlas billboard.
        field_gain = float(self._display_recipe.field_warp_gain)
        cal_mode = str(getattr(self, "_calibrate_mode", "normal") or "normal")
        # Legacy atlas path crushed FIELD so plates wouldn't smear.
        if cal_mode == "plate_only":
            field_gain = 0.0
        elif cal_mode == "field_only":
            # Full recipe gain — prove NWR warp without LOOK plate stack.
            field_gain = float(self._display_recipe.field_warp_gain)
        elif speaking_plate and not self._tickfeed_look_authority:
            field_gain *= 0.20
        elif self._tickfeed_look_authority:
            # Velocity-aware FIELD gating (ChatGPT C): mute during fast
            # OPENING/CLOSING so plate+field don't smear mid-band; keep some
            # travel at steady OPEN / REST (avoid frozen mid-open regression).
            plate_o = max(
                float(self._plate_blend_current[1]),
                float(self._plate_openness_current),
                float(self._ml_openness),
            )
            mouth_state = str(
                getattr(self, "_mouth_transition", "REST") or "REST"
            )
            vel = abs(float(getattr(self, "_plate_open_vel", 0.0) or 0.0))
            # Single-owner mid-band: LOOK plate owns oral disk during motion;
            # keep a little FIELD at steady OPEN so motion does not freeze.
            if mouth_state in {"OPENING", "CLOSING"}:
                if vel > 0.6 or 0.10 <= plate_o <= 0.65:
                    field_gain *= 0.05
                else:
                    field_gain *= 0.12
            elif plate_o >= 0.45 or mouth_state == "OPEN":
                field_gain *= 0.12
            elif plate_o >= 0.12:
                field_gain *= max(0.30, 1.0 - 0.75 * plate_o)
            # else REST / near-closed: full recipe gain
        self._field_gain_eff = float(field_gain)
        program["avatar_field_gain"].value = field_gain
        # Step 12: plate snap — commit toward the nearest captured mouth shape.
        # Early atlas commitment during OPENING/CLOSING reduces mid-band ghosts.
        sharp = float(self._display_recipe.plate_sharpness)
        if (
            self._tickfeed_look_authority
            and str(getattr(self, "_mouth_transition", "REST"))
            in {"OPENING", "CLOSING", "OPEN"}
        ):
            sharp = max(sharp, 0.95)
        program["avatar_plate_sharpness"].value = sharp
        # Cosmetic prefs (scaffolding) — grade only, never replace identity photo.
        if self._cosmetics is not None:
            skin = self._cosmetics.skin_tint_rgb
            eye = self._cosmetics.eye_tint_rgb
            program["avatar_skin_tint"].value = (
                float(skin[0]),
                float(skin[1]),
                float(skin[2]),
            )
            program["avatar_eye_tint"].value = (
                float(eye[0]),
                float(eye[1]),
                float(eye[2]),
            )
            program["avatar_makeup_strength"].value = float(
                self._cosmetics.makeup_strength
            )
        else:
            program["avatar_skin_tint"].value = (1.0, 1.0, 1.0)
            program["avatar_eye_tint"].value = (1.0, 1.0, 1.0)
            program["avatar_makeup_strength"].value = 0.0
        program["avatar_lock_overlay"].value = 1.0 if self._show_locks else 0.0
        program["avatar_breath_phase"].value = float(state.breath_phase)
        program["avatar_debug_view"].value = int(self._debug_view)
        program["avatar_deform"].value = 1 if self._use_tissue else 0
        active = list(state.muscle_activations.values())
        program["avatar_muscle_heat"].value = (
            float(sum(active) / len(active)) if active else 0.0
        )

    def _ease_mouth_pose(self, frame_time: float) -> None:
        """Approach the active viseme without snapping between mouth shapes."""
        rate = float(getattr(self, "_mouth_ease_rate", 5.0))
        amount = 1.0 - math.exp(-max(frame_time, 0.0) * rate)
        current = self._mouth_pose
        target = self._target_mouth_pose
        self._mouth_pose = MouthPose(
            current.width + (target.width - current.width) * amount,
            current.openness + (target.openness - current.openness) * amount,
            current.roundness + (target.roundness - current.roundness) * amount,
            current.expression + (target.expression - current.expression) * amount,
        )

    # ----------------------------------------------------------------- chat

    def _start_chat_thread(self) -> None:
        def worker() -> None:
            print("chat> ", end="", flush=True)
            while not self._shutdown.is_set():
                try:
                    line = sys.stdin.readline()
                except (OSError, ValueError):
                    break
                if line == "":
                    break
                text = line.strip()
                if not text:
                    print("chat> ", end="", flush=True)
                    continue
                if text.lower() in {"quit", "exit", ":q"}:
                    self._chat_queue.put("__quit__")
                    break
                self._chat_queue.put(text)
                print("chat> ", end="", flush=True)

        self._chat_thread = threading.Thread(
            target=worker,
            name="aiface-chat",
            daemon=True,
        )
        self._chat_thread.start()

    # ------------------------------------------------------------ chat box

    def _relayout_frame(self) -> None:
        """Fit the square portrait above the chat panel for the current window."""
        width, height = self._scene_size()
        if not self._chat_box_visible:
            self._face_frame = (0.0, 0.0, 1.0, 1.0)
            self._panel_rect = (0, 0, 0, 0)
            return
        self._face_frame, self._panel_rect = frame_layout(width, height)
        self._overlay_dirty = True

    def _panel_font_pair(self) -> tuple[object, object] | None:
        if self._panel_fonts is not None:
            return self._panel_fonts
        try:
            from PIL import ImageFont
        except ImportError:
            return None
        try:
            body = ImageFont.truetype("consola.ttf", 17)
            title = ImageFont.truetype("arial.ttf", 14)
        except OSError:
            body = ImageFont.load_default()
            title = body
        self._panel_fonts = (body, title)
        return self._panel_fonts

    def _apply_mouth_speed_preset(
        self, preset: object, *, announce: bool = True
    ) -> None:
        """Push a Slow/Normal/Fast preset into the 60 Hz mouth clock."""
        from aiface.mouth_speed import MouthSpeedPreset

        assert isinstance(preset, MouthSpeedPreset)
        self._mouth_speed = preset
        self._mouth_hold_min = float(preset.hold_seconds)
        self._plate_ease_rate = float(preset.plate_ease)
        self._mouth_ease_rate = float(preset.mouth_ease)
        self._plate_release_scale = float(preset.release_scale)
        if hasattr(self, "_biomech"):
            self._biomech.jaw.mass = float(preset.jaw_mass)
            self._biomech.jaw.damping = float(preset.jaw_damping)
            self._biomech.jaw.elasticity = float(preset.jaw_elasticity)
        self._set_mouth_hold_scale(float(preset.hold_scale), announce=False)
        self._mouth_menu_open = False
        self._overlay_dirty = True
        if announce:
            dwell_ms = int(self._mouth_timeline.min_dwell_s * 1000)
            print(
                f"Mouth speed: {preset.label} (hold {dwell_ms}ms)",
                flush=True,
            )
            self._chatbox.add(
                SPEAKER_SYSTEM,
                f"Mouth speed: {preset.label} — plate hold {dwell_ms}ms",
            )

    def _set_mouth_hold_scale(self, scale: float, *, announce: bool = False) -> None:
        """Realtime scrollbar: how long open/teeth plates stay painted."""
        self._mouth_hold_scale = clamp_hold_scale(scale)
        dwell, bridge, muscle = hold_scale_to_params(self._mouth_hold_scale)
        self._mouth_timeline.set_hold_timing(
            min_dwell_s=dwell, max_bridge_s=bridge
        )
        self._mouth_hold_min = float(muscle)
        self._overlay_dirty = True
        if announce:
            print(
                f"Mouth hold: {int(dwell * 1000)}ms "
                f"(bridge {int(bridge * 1000)}ms)",
                flush=True,
            )

    def _mouth_hold_scale_from_x(self, overlay_x: int) -> float | None:
        """Map overlay X onto the painted track (hit rect includes a pad)."""
        track = self._ui_hits.get("mouth_hold_track")
        if track is None:
            return None
        tx, _ty, tw, _th = track
        # chatbox pads the hit box by 8px on each side — undo for value.
        pad = 8
        visual_x = float(tx + pad)
        visual_w = float(max(tw - pad * 2, 1))
        return clamp_hold_scale((float(overlay_x) - visual_x) / visual_w)

    def _overlay_mouse(self, x: int, y: int) -> list[tuple[int, int]]:
        """Map window mouse coords → overlay pixel candidates.

        moderngl_window (pyglet) already reports **top-down** window pixels
        (y=0 at the top). The HUD is painted in framebuffer pixels, which
        can differ under DPI scaling. Prefer top-down; also try a Y flip
        for backends that leave mouse Y bottom-up.
        """
        buf_w, buf_h = self._scene_size()
        win_w = max(int(getattr(self.wnd, "width", buf_w) or buf_w), 1)
        win_h = max(int(getattr(self.wnd, "height", buf_h) or buf_h), 1)
        ox = int(round(float(x) * float(buf_w) / float(win_w)))
        oy_top = int(round(float(y) * float(buf_h) / float(win_h)))
        oy_flip = int(buf_h) - oy_top - 1
        points = [(ox, oy_top), (ox, oy_flip)]
        out: list[tuple[int, int]] = []
        for point in points:
            if point not in out:
                out.append(point)
        return out

    def _hit_ui(self, x: int, y: int) -> tuple[str | None, int, int]:
        """Return (control_id, overlay_x, overlay_y) for a window mouse point."""
        for ox, oy in self._overlay_mouse(int(x), int(y)):
            # Slider first — its hit pad can overlap the Mouth: button edge.
            track = self._ui_hits.get("mouth_hold_track")
            if track is not None:
                tx, ty, tw, th = track
                if tx <= ox < tx + tw and ty <= oy < ty + th:
                    return "mouth_hold_track", ox, oy
            control = hit_test(self._ui_hits, ox, oy)
            if control is not None:
                return control, ox, oy
        ox, oy = self._overlay_mouse(int(x), int(y))[0]
        return None, ox, oy

    def _set_mouth_speed_key(self, key: str) -> None:
        self._apply_mouth_speed_preset(preset_by_key(key), announce=True)

    def _cycle_mouth_speed(self) -> None:
        self._set_mouth_speed_key(next_preset_key(self._mouth_speed.key))

    def _paint_overlay(self, draw: object, width: int, height: int) -> None:
        if not self._chat_box_visible:
            return
        fonts = self._panel_font_pair()
        if fonts is None:
            return
        body, title = fonts
        dwell_ms = int(float(self._mouth_timeline.min_dwell_s) * 1000.0)
        self._ui_hits = paint_panel(
            draw,
            self._chatbox,
            rect=self._panel_rect,
            font=body,
            title_font=title,
            now=time.perf_counter() - self._clock0,
            title="chat",
            mouth_speed_label=self._mouth_speed.label,
            mouth_menu_open=self._mouth_menu_open,
            mouth_options=tuple(preset.label for preset in MOUTH_SPEED_PRESETS),
            mouth_hold_scale=float(self._mouth_hold_scale),
            mouth_hold_ms=dwell_ms,
        )

    def _submit_chat_line(self) -> None:
        spoken = self._chatbox.submit()
        self._overlay_dirty = True
        if spoken is None:
            return
        if spoken.lower() in {"quit", "exit", ":q"}:
            self.wnd.close()
            return
        self._chatbox.add(SPEAKER_YOU, spoken)
        if self._kick_llm(spoken):
            self._chatbox.pending = True
        else:
            self._chatbox.pending = False
            self._chatbox.add(SPEAKER_SYSTEM, "Still answering the last one.")

    def _handle_chat_key(self, key: int, modifiers: object) -> bool:
        """Route an editing key into the panel. Returns whether it was consumed."""
        if not self._chat_box_visible:
            return False
        keys = self.wnd.keys
        box = self._chatbox

        if key == getattr(keys, "ESCAPE", None):
            box.focused = not box.focused
            self._overlay_dirty = True
            if not box.focused and not box.pending and not self._visemes:
                self._enter_zero_state(blink=False)
            return True
        if not box.focused:
            return False

        control = bool(getattr(modifiers, "ctrl", False))
        if key in {keys.ENTER, getattr(keys, "NUMPAD_ENTER", keys.ENTER)}:
            self._submit_chat_line()
            # Submitted — stay in hearing/waiting until speech starts.
            if box.pending:
                self._enter_hearing_state()
            return True
        if key == keys.BACKSPACE:
            changed = box.delete_word() if control else box.backspace()
            self._overlay_dirty = self._overlay_dirty or changed
            if changed and box.text.strip():
                self._enter_hearing_state()
            return True
        if key == getattr(keys, "DELETE", None):
            self._overlay_dirty = box.delete() or self._overlay_dirty
            return True
        if key == keys.LEFT:
            self._overlay_dirty = box.move_caret(-1) or self._overlay_dirty
            return True
        if key == keys.RIGHT:
            self._overlay_dirty = box.move_caret(1) or self._overlay_dirty
            return True
        if key == getattr(keys, "HOME", None):
            self._overlay_dirty = box.caret_to_start() or self._overlay_dirty
            return True
        if key == getattr(keys, "END", None):
            self._overlay_dirty = box.caret_to_end() or self._overlay_dirty
            return True
        if key == keys.UP:
            self._overlay_dirty = box.recall_previous() or self._overlay_dirty
            return True
        if key == keys.DOWN:
            self._overlay_dirty = box.recall_next() or self._overlay_dirty
            return True
        # Focused typing owns the letter hotkeys; only chrome keys pass through.
        return key not in self._passthrough_keys()

    def _passthrough_keys(self) -> set[int]:
        keys = self.wnd.keys
        return {
            keys.F1,
            keys.F2,
            keys.F3,
            keys.F4,
            keys.F5,
            keys.F6,
            keys.F7,
            keys.F8,
            keys.F9,
            keys.F10,
            keys.F11,
        }

    def on_unicode_char_entered(self, char: str) -> None:
        if not self._chat_box_visible or not self._chatbox.focused:
            return
        if self._chatbox.insert(char):
            self._overlay_dirty = True
            # First keystroke → hearing / waiting look (lips stay closed).
            self._enter_hearing_state()

    # ---------------------------------------------------------------- voice

    def _start_speech_engine(self) -> None:
        """Build the synthesiser and open the speaker, or stay on text timing."""
        if not bool(getattr(self.argv, "tts", False)):
            return
        try:
            synthesizer = build_synthesizer(
                backend=str(getattr(self.argv, "tts_backend", "auto")),
                api_key=str(getattr(self.argv, "llm_api_key", "")),
                base_url=str(getattr(self.argv, "llm_base_url", DEFAULT_BASE_URL)),
                model=str(getattr(self.argv, "tts_model", DEFAULT_SPEECH_MODEL)),
                voice=str(getattr(self.argv, "tts_voice", DEFAULT_SPEECH_VOICE)),
                speed=float(getattr(self.argv, "tts_speed", 1.0)),
                instructions=str(getattr(self.argv, "tts_instructions", "")),
                command=str(getattr(self.argv, "tts_command", "")),
                alignment=str(getattr(self.argv, "tts_align", ALIGN_ENERGY)),
                transcribe_model=str(
                    getattr(self.argv, "tts_transcribe_model", DEFAULT_TRANSCRIBE_MODEL)
                ),
                warp_strength=float(getattr(self.argv, "tts_warp", 0.65)),
            )
        except TTSError as exc:
            print(f"warning: voice disabled ({exc}); lips follow written timing")
            return

        try:
            sink = open_audio_sink(str(getattr(self.argv, "audio_backend", "auto")))
        except AudioError as exc:
            print(f"warning: voice disabled, no audio output ({exc})")
            return

        self._speech = synthesizer
        self._audio = sink
        print(f"Voice: {synthesizer.description} -> {sink.name}")

    def _synthesize(self, spoken: str) -> PreparedSpeech | None:
        """Render a reply to audio with measured viseme timing.

        Called from the reply thread: synthesis is a network or subprocess round
        trip and must never stall the render loop. A failure is not fatal — the
        face falls back to timing derived from the written words.
        """
        synthesizer = self._speech
        if synthesizer is None:
            return None
        try:
            return synthesizer.prepare(spoken)
        except TTSError as exc:
            print(f"warning: voice synthesis failed ({exc}); using written timing")
            return None

    def _speech_pace(self) -> float:
        return max(
            0.85, min(1.60, float(self._display_recipe.speech_pace) or 1.0)
        )

    def _viseme_min_hold(self) -> float:
        return max(0.0, float(self._display_recipe.viseme_min_hold) or 0.0)

    def _schedule_audio(
        self, speech: PreparedSpeech, emotion: str
    ) -> list[VisemeEvent]:
        """Start playback and hang the viseme schedule off the audio clock.

        Playback is started here, on the render thread, precisely so that the
        moment the clip begins is the moment the schedule is anchored to. Every
        span was measured from the waveform, so the only unknown left is how long
        the device takes to open, which each sink reports and ``--tts-latency``
        trims by hand.
        """
        from aiface.tts import apply_speech_pace

        pace = self._speech_pace()
        raw_dur = float(speech.duration)
        # TickFeed word-lock: pace may stretch audio+spans together, but never
        # cumulatively shift later spans via min_hold (that breaks absolute sync).
        min_hold = (
            0.0
            if getattr(self, "_tickfeed_look_authority", False)
            or (
                self._tickfeed is not None
                and getattr(self._tickfeed, "enabled", False)
            )
            else self._viseme_min_hold()
        )
        speech = apply_speech_pace(speech, pace, min_hold=min_hold)
        if abs(pace - 1.0) > 1e-3:
            print(
                f"Speech pace {pace:.3f}: audio "
                f"{raw_dur:.2f}s → {float(speech.duration):.2f}s"
            )
        sink = self._audio
        if sink is not None:
            sink.play(speech.clip)
        # Anchor at play return; prefer sink.media_time() over fixed latency.
        self._audio_anchor = time.perf_counter() - self._clock0
        has_media = (
            sink is not None and callable(getattr(sink, "media_time", None))
        )
        latency = 0.0 if has_media else (
            sink.startup_latency if sink is not None else 0.0
        )
        start_at = float(self._audio_anchor) + latency + self._speech_trim
        # Open/close ML tracks this clip's RMS envelope on the same clock.
        self._open_close_envelope = rms_envelope(speech.clip)
        self._open_close_start = start_at
        self._live_vector.noise_floor = float(
            self._open_close_envelope.noise_floor()
        )
        self._live_vector.peak_hint = float(self._open_close_envelope.peak)
        self._behavior.noise_floor = float(self._live_vector.noise_floor)
        self._behavior.peak_hint = float(self._live_vector.peak_hint)
        self._open_close_peak = self._live_vector.peak_hint
        self._live_vector.clear_history()
        self._behavior.clear_history()

        # The voice is authoritative. Anything still queued belongs to a line
        # that is no longer being spoken.
        self._visemes.clear()
        self._mouth_timeline.clear()
        # Keep span durations close to measured audio; overlay release clamps
        # to next_due anyway (see speech_overlay_until).
        events = schedule_spans(
            speech.span_tuples(),
            emotion,
            start_at=start_at,
            minimum_hold=0.028,
        )
        events.append(
            VisemeEvent(
                phoneme="REST",
                emotion=emotion,
                due_at=start_at + speech.duration,
                duration=0.08,
            )
        )
        self._telemetry.alignment = speech.alignment
        self._telemetry.clip_seconds = speech.duration
        return events

    def _schedule_text(self, phonemes: Sequence[str], emotion: str) -> list[VisemeEvent]:
        """Estimate timing from the written words when there is no audio."""
        self._visemes.clear()
        self._mouth_timeline.clear()
        # Base 0.09s/phoneme scaled by speech_pace so text-only matches paced TTS.
        events = schedule_visemes(
            phonemes,
            emotion,
            start_at=time.perf_counter() - self._clock0 + 0.05,
            seconds_per_phoneme=0.09 * self._speech_pace(),
        )
        if events:
            last = events[-1]
            events.append(
                VisemeEvent(
                    phoneme="REST",
                    emotion=emotion,
                    due_at=last.due_at + last.duration,
                    duration=0.12,
                )
            )
        self._telemetry.alignment = "text"
        self._telemetry.clip_seconds = 0.0
        return events

    # -------------------------------------------------------- the sync channel

    def _open_voice_channel(self) -> None:
        """Prepare to lock the face to audio somebody else is producing.

        Nothing is allocated until a caller actually speaks: the channel exists
        so that a voice which already has audio — a realtime model, a phone
        bridge, a file replay — can drive these lips without this process owning
        a synthesiser.
        """
        self._voice: VoiceStream | None = None
        self._voice_lock = threading.Lock()
        self._voice_inbox: list[VisemeEvent] = []
        self._voice_caption = ""
        self._voice_emotion = "NEUTRAL"
        self._voice_epoch: float | None = None
        self._voice_trim = float(getattr(self.argv, "voice_trim", 0.0))
        self._voice_chunks = 0
        self._voice_events = 0
        self._voice_lag = 0.0

    def _voice_config(self, sample_rate: int) -> StreamConfig:
        return StreamConfig(
            sample_rate=sample_rate,
            lookahead_seconds=float(getattr(self.argv, "voice_lookahead", 0.05)),
        )

    def _voice_samples(self, payload: bytes, layout: str) -> object:
        """Interpret a chunk body as the layout the caller declared."""
        if layout == "float32":
            if len(payload) % 4:
                raise ValueError("float32 audio must be a whole number of samples")
            return np.frombuffer(payload, dtype="<f4")
        return payload

    def _voice_request(self, kind: str, payload: dict[str, object]) -> dict[str, object]:
        """Serve one voice-channel call. Runs on a bridge request thread.

        Alignment is signal processing over numpy arrays and touches no GPU
        state, so it happens here rather than being deferred to a frame: audio
        arrives on the speaker's clock, not ours. The only thing handed to the
        render loop is a list of timed events.
        """
        default_rate = int(getattr(self.argv, "voice_rate", DEFAULT_STREAM_RATE))
        with self._voice_lock:
            if kind == "expect":
                rate = int(payload.get("sample_rate") or 0) or default_rate
                stream = self._ensure_voice(rate)
                text = str(payload.get("text", ""))
                emotion = str(payload.get("emotion", "")).strip().upper()
                if emotion in EMOTION_IMPULSES:
                    self._voice_emotion = emotion
                queued = stream.expect(text)
                self._voice_caption = strip_tags(text).strip()
                return {"expecting": queued, "pending": stream.pending_visemes}

            if kind == "timeline":
                # Preferred path when the LLM host already timed phonemes to
                # its own audio clock. No PCM alignment; we only schedule.
                from aiface.speech import parse_timeline_spans, schedule_spans

                emotion = str(payload.get("emotion", "")).strip().upper()
                if emotion in EMOTION_IMPULSES:
                    self._voice_emotion = emotion
                caption = str(payload.get("caption", "")).strip()
                if caption:
                    self._voice_caption = strip_tags(caption)
                spans = parse_timeline_spans(list(payload.get("spans") or []))
                if self._voice_epoch is None:
                    self._voice_epoch = (
                        time.perf_counter() - self._clock0 + self._voice_trim
                    )
                events = schedule_spans(
                    spans, self._voice_emotion, start_at=self._voice_epoch
                )
                self._voice_inbox.extend(events)
                self._voice_events += len(events)
                return {
                    "scheduled": len(events),
                    "pending": 0,
                    "mode": "timeline",
                }

            if kind == "pcm":
                rate = int(payload.get("sample_rate") or 0) or default_rate
                stream = self._ensure_voice(rate)
                chunk = self._voice_samples(
                    bytes(payload.get("audio") or b""),
                    str(payload.get("format", "pcm16")),
                )
                if self._voice_epoch is None:
                    # Stream time zero is the arrival of the first audio. The
                    # caller's own playback delay is what --voice-trim absorbs.
                    self._voice_epoch = (
                        time.perf_counter() - self._clock0 + self._voice_trim
                    )
                spans = stream.feed(chunk)
                self._voice_chunks += 1
                # Live PCM → RMS history for open/close ML when no full clip.
                try:
                    samples = np.asarray(chunk, dtype=np.float32).reshape(-1)
                    if samples.size:
                        pcm_rms = float(np.sqrt(np.mean(np.square(samples))))
                        self._live_vector.push_rms(pcm_rms)
                        self._behavior.push_rms(pcm_rms)
                        self._open_close_peak = max(
                            self._open_close_peak, pcm_rms
                        )
                        self._live_vector.peak_hint = self._open_close_peak
                        self._behavior.peak_hint = self._open_close_peak
                except Exception:
                    pass
                return self._voice_reply(stream, spans)

            if kind == "end":
                stream = self._voice
                if stream is None:
                    return {"scheduled": 0, "pending": 0}
                spans = stream.finish()
                reply = self._voice_reply(stream, spans)
                stream.reset()
                self._voice_epoch = None
                return reply

        raise ValueError(f"unknown voice request {kind!r}")

    def _ensure_voice(self, sample_rate: int) -> VoiceStream:
        """The channel for this sample rate, rebuilding it if the rate changed."""
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        stream = self._voice
        if stream is not None and stream.config.sample_rate != sample_rate:
            if stream.received_seconds > 0.0:
                raise ValueError(
                    f"audio arrived at {stream.config.sample_rate} Hz and is now "
                    f"{sample_rate} Hz; end the utterance before changing rate"
                )
            stream = None
        if stream is None:
            stream = VoiceStream(self._voice_config(sample_rate))
            self._voice = stream
        return stream

    def _voice_reply(
        self, stream: VoiceStream, spans: Sequence[StreamedSpan]
    ) -> dict[str, object]:
        """Schedule what the channel decided and describe it back to the caller."""
        epoch = self._voice_epoch or 0.0
        events = stream_visemes(spans, self._voice_emotion, start_at=epoch)
        self._voice_inbox.extend(events)
        self._voice_events += len(events)
        if spans:
            self._voice_lag = max(span.lag for span in spans)
        stats = stream.stats()
        return {
            "scheduled": len(events),
            "pending": stats.pending,
            "received_seconds": round(stats.received_seconds, 4),
            "speaking": stats.speaking,
            "lag_ms": round(stats.mean_lag * 1000.0, 2),
            "peak_lag_ms": round(stats.peak_lag * 1000.0, 2),
            "rate_ms": round(stats.rate * 1000.0, 2),
        }

    def _drain_voice_channel(self) -> None:
        """Adopt streamed visemes and captions on the render thread."""
        with self._voice_lock:
            events = self._voice_inbox
            caption = self._voice_caption
            self._voice_inbox = []
            self._voice_caption = ""
        if caption:
            self._chatbox.pending = False
            self._chatbox.add(SPEAKER_FACE, caption)
            self._overlay_dirty = True
            self._telemetry.alignment = "stream"
        if not events:
            return
        self._visemes.extend(events)
        while len(self._visemes) > MAX_PHONEME_QUEUE:
            self._visemes.popleft()

    def _queue_spoken(self, spoken: str, speech: PreparedSpeech | None = None) -> None:
        phonemes, emotion = extract_states(spoken)
        self._conversation.last_emotion = emotion
        self._chatbox.pending = False
        self._chatbox.add(SPEAKER_FACE, strip_tags(spoken).strip())
        self._overlay_dirty = True
        if speech is not None and speech.spans:
            events = self._schedule_audio(speech, emotion)
        else:
            events = self._schedule_text(phonemes, emotion)
        self._visemes.extend(events)
        while len(self._visemes) > MAX_PHONEME_QUEUE:
            self._visemes.popleft()
        print(f"avatar: {spoken}")
        detail = (
            f"clip={speech.duration:.2f}s align={speech.alignment} "
            f"voice={speech.voice}"
            if speech is not None and speech.spans
            else f"visemes={phonemes[:12]}{'...' if len(phonemes) > 12 else ''}"
        )
        print(
            f"  {detail} emotion={emotion} history={self._conversation.turn_count}"
        )

    def _speak_without_chat(self, spoken: str) -> None:
        """Say a canned line, synthesising off the render thread if needed."""
        if self._speech is None:
            self._queue_spoken(spoken)
            return

        def worker() -> None:
            self._reply_queue.put(SpokenReply(spoken, self._synthesize(spoken)))

        threading.Thread(target=worker, name="aiface-voice", daemon=True).start()

    def _kick_llm(self, user_text: str) -> bool:
        """Start a reply. Returns False when one is already in flight."""
        if self._llm_thread is not None and self._llm_thread.is_alive():
            print("(waiting for previous reply)")
            return False

        def worker() -> None:
            try:
                reply = llm_reply(
                    user_text,
                    base_url=str(getattr(self.argv, "llm_base_url", DEFAULT_BASE_URL)),
                    model=str(getattr(self.argv, "llm_model", DEFAULT_MODEL)),
                    api_key=str(getattr(self.argv, "llm_api_key", "")),
                    session=self._conversation,
                )
            except Exception as error:  # noqa: BLE001 - a dead backend must not mute the face
                reply = (
                    f"[EMOTION:SAD] Something went wrong reaching my thoughts: {error}"
                )
            # Synthesis stays on this thread so the render loop never waits for
            # a network round trip.
            self._reply_queue.put(SpokenReply(reply, self._synthesize(reply)))

        self._llm_thread = threading.Thread(
            target=worker,
            name="aiface-llm",
            daemon=True,
        )
        self._llm_thread.start()
        return True

    def _drain_chat(self) -> None:
        while True:
            try:
                message = self._chat_queue.get_nowait()
            except queue.Empty:
                break
            if message == "__quit__":
                self.wnd.close()
                return
            self._kick_llm(message)
        while True:
            try:
                reply = self._reply_queue.get_nowait()
            except queue.Empty:
                break
            self._queue_spoken(reply.text, reply.speech)

    def _start_face_bridge(self) -> None:
        token = str(getattr(self.argv, "bridge_token", "") or "").strip() or new_token()
        speak = (
            self._speak_without_chat
            if bool(getattr(self.argv, "bridge_direct_speak", False))
            else self._kick_llm
        )
        self._bridge = FaceBridge(
            status_provider=self._bridge_status,
            preview_provider=self._preview,
            screenshot_provider=self._screenshot,
            speak_handler=speak,
            voice_handler=self._voice_request,
            probe_provider=self._mouth_probe_snapshot,
            cells_provider=self._cells_snapshot,
            cells_drive_handler=self._cells_drive,
            calibrate_handler=self._calibrate_set,
            token=token,
            host=str(getattr(self.argv, "bridge_host", DEFAULT_HOST)),
            port=int(getattr(self.argv, "bridge_port", DEFAULT_PORT)),
            allow_remote_bind=bool(getattr(self.argv, "allow_remote_bind", False)),
        )
        self._bridge.start()
        print(f"Face bridge: {self._bridge.url}")
        print(f"  Authorization: Bearer {token}")
        print("  GET /health /status /probe /cells /preview /screenshot")
        print('  POST /speak  {"text":"..."}')
        print(
            '  POST /cells/drive  {"mode":"cell","x":128,"y":80,"dx":0,"dy":-1}'
            '  # or mode=cluster|neighbor|batch'
        )
        print(
            '  POST /calibrate  {"mode":"normal|plate_only|field_only",'
            ' "speech_pace":1.12, "viseme_min_hold":0.10,'
            ' "zero_mood":"neutral|smile|waiting"}'
        )
        print("  Keys: Z cycles 0-state mood (neutral / smile / waiting)")
        print('  POST /voice/expect  {"text":"..."}   then /voice/pcm, /voice/end')
        print(
            '  POST /voice/timeline {"spans":[{"phoneme":"OU","start":0,"end":0.12}],'
            ' "caption":"..."}   # host-timed, no PCM'
        )

    def _bridge_status(self) -> dict[str, object]:
        state = self._render_state
        profile = self._avatar_bundle.profile
        return {
            "chat_pending": bool(self._chatbox.pending),
            "tick": int(self.tick),
            "fps": float(self._fps),
            "avatar": {
                "id": profile.id,
                "ok": bool(self._avatar_bundle.ok),
                "mouth_cell_count": int(profile.geometry.mouth_cell_count),
                "root": str(self._avatar_bundle.root),
            },
            "behavior": self._behavior.snapshot(),
            "phoneme": state.last_phoneme,
            "emotion": self._telemetry.last_emotion,
            "jaw_angle": float(state.jaw_angle),
            "pending_visemes": len(self._visemes),
            "active_muscles": int(self._active_muscle_count),
            "conversation_turns": int(self._conversation.turn_count),
            "last_emotion": self._conversation.last_emotion,
            "group_activations": dict(state.group_activations),
            "speaking": bool(state.speaking),
            "voice": self._speech.voice.name if self._speech is not None else "",
            "audio_backend": self._audio.name if self._audio is not None else "",
            "viseme_timing": self._telemetry.alignment or "text",
            "clip_seconds": float(self._telemetry.clip_seconds),
            "dropped_commands": int(self.dropped_commands),
            "voice_channel": self._voice_status(),
            "mouth_owners": list(self._mouth_ownership.owners),
            "mouth_ownership": self._mouth_ownership.as_dict(),
            "mean_speed": float(self._telemetry.mean_speed),
            "peak_speed": float(self._telemetry.peak_speed),
            "active_cells": int(self._telemetry.active_cells),
            "tickfeed": {
                "look_authority": bool(self._tickfeed_look_authority),
                "open": float(getattr(self, "_ml_openness", 0.0) or 0.0),
                "smile": float(getattr(self, "_ml_smile", 0.0) or 0.0),
                "plate_open": float(getattr(self, "_plate_openness_current", 0.0) or 0.0),
                "viseme": str(self._held_speech_viseme or ""),
                "presence": str(getattr(self, "_presence", "")),
                "zero_mood": str(getattr(self, "_zero_mood", ZERO_MOOD_NEUTRAL)),
                "field_gain_eff": float(getattr(self, "_field_gain_eff", 0.0) or 0.0),
                "calibrate_mode": str(getattr(self, "_calibrate_mode", "normal")),
                "speech_pace": float(self._display_recipe.speech_pace),
                "viseme_min_hold": float(self._display_recipe.viseme_min_hold),
                "side_a_debug": (
                    self._side_a_debug.summary()
                    if getattr(self, "_side_a_debug", None) is not None
                    and self._side_a_debug.enabled
                    else {"enabled": False}
                ),
                "side_a_recent": (
                    self._side_a_debug.recent(12)
                    if getattr(self, "_side_a_debug", None) is not None
                    and self._side_a_debug.enabled
                    else []
                ),
            },
        }

    def _voice_status(self) -> dict[str, object]:
        """What the sync channel has been asked to do, and how late it was."""
        with self._voice_lock:
            stream = self._voice
            stats = stream.stats() if stream is not None else None
            return {
                "open": stream is not None,
                "sample_rate": stream.config.sample_rate if stream is not None else 0,
                "lookahead_ms": round(
                    float(getattr(self.argv, "voice_lookahead", 0.05)) * 1000.0, 1
                ),
                "trim_ms": round(self._voice_trim * 1000.0, 1),
                "chunks": self._voice_chunks,
                "scheduled": self._voice_events,
                "peak_lag_ms": round(self._voice_lag * 1000.0, 2),
                "received_seconds": round(stats.received_seconds, 3) if stats else 0.0,
                "speaking": bool(stats.speaking) if stats else False,
            }

    def _on_world_reloaded(self) -> None:
        """Drop everything that described the world we just replaced.

        A reloaded ``.bds`` is a rest pose. Visemes still queued belong to the
        old field, and replaying them would drive muscles from a pose the solver
        no longer holds.
        """
        self._visemes.clear()
        self._mouth_timeline.clear()
        self._biomech.reset()
        self._mouth_pose = mouth_pose("REST", "NEUTRAL")
        self._target_mouth_pose = self._mouth_pose
        self._telemetry.alignment = ""
        self._telemetry.clip_seconds = 0.0
        if self._audio is not None:
            self._audio.stop()
        with self._voice_lock:
            self._voice_inbox.clear()
            self._voice_epoch = None
            if self._voice is not None:
                self._voice.reset()

    def _on_pre_draw(self) -> None:
        """Refresh avatar uniforms after TickFeed/constraint ingest, before draw."""
        self._update_avatar_uniforms()

    # ------------------------------------------------------------- impulses

    def _fire_impulse(
        self,
        phoneme: str,
        emotion: str,
        *,
        duration: float = 0.14,
        next_due_at: float | None = None,
        due_at: float | None = None,
    ) -> None:
        """Speech does not write geometry. It injects muscle impulses."""
        started = time.perf_counter()
        from aiface.plates import OPEN_TOOTH_VISEMES, VISEME_OPENNESS
        from aiface.speech import canonical_viseme, speech_overlay_until

        now = self._speech_now()
        key = canonical_viseme(phoneme)

        if self._tickfeed_look_authority:
            # TickFeed path: viseme table amounts; closures always interrupt opens.
            # Overlay until = absolute audio span end (not now+vowel floor).
            open_amt = float(VISEME_OPENNESS.get(key, 0.0))
            smile_amt = 0.0
            if open_amt < 0.2 and (emotion or "").upper() in {"HAPPY", "JOY"}:
                smile_amt = 0.55
            closure_keys = {"REST", "CLOSED", "PP", "MM", "SIL"}
            is_closure = key in closure_keys or open_amt < 0.12
            if is_closure:
                open_amt = 0.0
                smile_amt = 0.0
                self._plate_open_hyst = 0.0
            until = speech_overlay_until(
                now=now,
                due_at=due_at,
                duration=float(duration),
                next_due_at=next_due_at,
            )
            self._presence = PRESENCE_SPEAKING
            self._tickfeed_live = {
                "phoneme": key,
                "open": open_amt,
                "smile": smile_amt,
                "surprise": 0.0,
                "emotion": (emotion or "NEUTRAL").strip().upper() or "NEUTRAL",
                "mode": "speech",
                "until": until,
                "due_at": float(due_at) if due_at is not None else now,
                "end_at": until,
            }
            self._held_speech_viseme = key
            self._mouth_timeline.clear()
            # Light jaw assist only — LOOK plates come from TickPackage labels.
            self._biomech.submit_phoneme(
                phoneme,
                tick=self.tick + 1,
                emotion_label=emotion,
                duration=max(float(duration), 0.05),
            )
        else:
            self._mouth_timeline.fire(
                phoneme,
                now=now,
                duration=float(duration),
                emotion=emotion,
                next_due_at=next_due_at,
            )
            hold_floor = float(getattr(self, "_mouth_hold_min", 0.36))
            if key in OPEN_TOOTH_VISEMES:
                hold_floor = max(hold_floor, hold_floor * 1.15)
            self._presence = PRESENCE_SPEAKING
            self._biomech.submit_phoneme(
                phoneme,
                tick=self.tick + 1,
                emotion_label=emotion,
                duration=max(float(duration), hold_floor),
            )
        self._telemetry.last_phoneme = phoneme
        self._telemetry.last_emotion = emotion
        self._telemetry.impulses_fired += 1
        self._telemetry.command_latency_ms = (time.perf_counter() - started) * 1000.0

    def _speech_now(self) -> float:
        """Clock for viseme fire — prefer audio media_time when playing."""
        anchor = getattr(self, "_audio_anchor", None)
        sink = self._audio
        media_fn = getattr(sink, "media_time", None) if sink is not None else None
        if anchor is not None and callable(media_fn):
            media = media_fn()
            if media is not None:
                return float(anchor) + float(media) + float(self._speech_trim)
        return time.perf_counter() - self._clock0

    def _advance_visemes(self) -> None:
        now = self._speech_now()
        while self._visemes and self._visemes[0].due_at <= now:
            event = self._visemes.popleft()
            next_due, _next_ph = self._upcoming_viseme()
            self._fire_impulse(
                event.phoneme,
                event.emotion,
                duration=event.duration,
                next_due_at=next_due,
                due_at=float(event.due_at),
            )

    def _upcoming_viseme(self) -> tuple[float | None, str | None]:
        """Peek the next *speech* event for timeline bridging (skip REST)."""
        from aiface.mouth_owner import CLOSED_VISEMES
        from aiface.speech import canonical_viseme

        for nxt in self._visemes:
            key = canonical_viseme(nxt.phoneme)
            if key == "REST":
                continue
            # Closed lips are real speech events — bridge to them so open
            # clears on schedule, but open-plate bridging uses speech≠REST.
            return float(nxt.due_at), key
        return None, None

    def _refresh_mouth_ownership(self) -> MouthOwnership:
        """Resolve Path-A ownership from eased openness + emotion + phoneme."""
        mood = self._active_emotion()
        phoneme = self._render_state.last_phoneme or "REST"
        speaking = bool(self._render_state.speaking) or bool(self._visemes)
        ownership = resolve_mouth_ownership(
            openness=float(self._plate_openness_current),
            emotion=mood,
            phoneme=phoneme,
            speaking=speaking,
            surprise_blend=float(self._expr_plate_blend),
        )
        self._mouth_ownership = ownership
        return ownership

    def _enqueue_field_specs(self, specs: Sequence[FieldImpulseSpec]) -> None:
        """No-op — legacy ±4 muscle field writes removed (TickFeed owns ch0/1)."""
        del specs

    def _flush_pending_cell_commands(self) -> None:
        """No-op — legacy pending ±4 queue cleared; TickFeed owns FIELD."""
        self._pending_cell_commands = []

    def _enqueue_mouth_cell_plan(self) -> None:
        """No-op — legacy ±4 MouthCellPlan removed; TickFeed owns FIELD."""
        return

    def _cells_snapshot(self) -> dict[str, object]:
        index = self._cell_clusters
        layers = self._frame_layers.snapshot()
        tickfeed = None
        if self._tickfeed is not None:
            face = self._tickfeed.face
            ml_loaded = self._tickfeed.ml is not None
            transport = self._tickfeed.transport
            labels = self._tickfeed.last_labels
            tickfeed = {
                "enabled": bool(self._tickfeed.enabled),
                "face_box": [face.x, face.y, face.w, face.h],
                "timeline_ticks": len(self._tickfeed.timeline),
                "speech_ticks": len(self._tickfeed.speech_by_tick),
                "look_ticks": len(self._tickfeed.look_by_tick),
                "ml_loaded": ml_loaded,
                "hello_ok": bool(self._tickfeed.hello_ack_ok),
                "transport": (
                    transport.mode if transport is not None else "none"
                ),
                "wire_loop": bool(self._tickfeed.wire_loop),
                "wire_loop_source": (
                    self._tickfeed.wire_loop_source
                    if self._tickfeed.wire_loop
                    else None
                ),
                "presence": self._presence,
                "wire": "KEY/DELTA full-face",
                "labels": (
                    {
                        "smile": float(labels.smile_amt),
                        "open": float(labels.open_amt),
                        "surprise": float(labels.surprise_amt),
                        "viseme_id": int(labels.viseme_id),
                        "word": labels.word,
                    }
                    if labels is not None
                    else None
                ),
                "cosmetics": (
                    self._tickfeed.cosmetics.as_dict()
                    if self._tickfeed.cosmetics is not None
                    else None
                ),
            }
        if index is None:
            return {
                "regions": [],
                "command_budget": 0,
                "neighbor_blend": True,
                "display_layers": layers,
                "tickfeed": tickfeed,
            }
        return {
            "avatar": self._avatar_bundle.profile.as_dict(),
            "regions": index.summary(),
            "grid": [int(index.width), int(index.height)],
            "command_budget": 0,
            "neighbor_blend": True,
            "modes": ["tickfeed"],
            "display_layers": layers,
            "tickfeed": tickfeed,
        }

    def _calibrate_set(self, payload: dict) -> dict[str, object]:
        """Isolate LOOK plate vs NWR FIELD; optional speech_pace live tweak."""
        from dataclasses import replace as _dc_replace

        mode = str(payload.get("mode") or getattr(self, "_calibrate_mode", "normal"))
        mode = mode.strip().lower()
        allowed = {"normal", "plate_only", "field_only"}
        if mode not in allowed:
            raise RuntimeError(
                f"calibrate mode must be one of {sorted(allowed)}, got {mode!r}"
            )
        self._calibrate_mode = mode
        if "speech_pace" in payload:
            pace = max(0.85, min(1.60, float(payload["speech_pace"])))
            self._display_recipe = _dc_replace(
                self._display_recipe, speech_pace=pace
            )
        if "viseme_min_hold" in payload:
            hold = max(0.0, min(0.35, float(payload["viseme_min_hold"])))
            self._display_recipe = _dc_replace(
                self._display_recipe, viseme_min_hold=hold
            )
        zero_mood = str(getattr(self, "_zero_mood", ZERO_MOOD_NEUTRAL))
        if "zero_mood" in payload:
            zero_mood = self._set_zero_mood(
                str(payload.get("zero_mood") or ""), announce=False
            )
        blinked = False
        if payload.get("blink"):
            if hasattr(self, "_biomech") and self._biomech is not None:
                from aiface.biomechanics.eyes import (
                    BLINK_HOLD_S,
                    BLINK_OPEN_S,
                    BLINK_STATE_CLOSED,
                )

                eyes = self._biomech.eyes
                eyes.request_blink()
                # Park at full close so /preview can inspect lids (blink is
                # shorter than a slow bridge round-trip at lab FPS).
                eyes.state.blink_phase = float(BLINK_OPEN_S + BLINK_HOLD_S)
                eyes.state.blink_state = BLINK_STATE_CLOSED
                eyes.state.lid_left = 0.0
                eyes.state.lid_right = 0.0
                eyes.state.lid_tension = 1.0
                eyes.state.blink_pause_s = float(payload.get("blink_hold", 1.25) or 1.25)
                blinked = True
        return {
            "ok": True,
            "mode": mode,
            "speech_pace": float(self._display_recipe.speech_pace),
            "viseme_min_hold": float(self._display_recipe.viseme_min_hold),
            "zero_mood": zero_mood,
            "zero_moods": list(ZERO_MOODS),
            "blink": blinked,
            "note": {
                "normal": "LOOK plates + FIELD gain mute as usual",
                "plate_only": "FIELD warp gain forced 0 — plates only",
                "field_only": "LOOK open plates forced closed — NWR FIELD only",
            }[mode],
        }

    def _cells_drive(self, payload: dict) -> dict[str, object]:
        """Legacy ±4 cell drive disabled — use TickFeed packages."""
        del payload
        raise RuntimeError(
            "per-cell ±4 drive removed; FIELD is TickFeed KEY/DELTA "
            "(see docs/TickFeedDesign.md)"
        )
        tick = int(self.tick) + 1
        impulses = parse_drive_request(payload, index=index, tick=tick)
        commands = to_commands(
            impulses,
            grid_width=self.grid_width,
            grid_height=self.grid_height,
        )
        self._pending_cell_commands.extend(commands)
        return {
            "queued": True,
            "impulses": len(commands),
            "tick": tick,
            "mode": mode,
        }

    def _step_biomechanics(self, frame_time: float) -> None:
        cpu_started = time.perf_counter()
        render, specs = self._biomech.step(
            max(frame_time, 0.0),
            tick=self.tick + 1,
            tickfeed_field=bool(self._tickfeed_look_authority),
        )
        self._render_state = render
        # expression.w drives smile.png in the shader. Use capture smile amount
        # (HAPPY or live width) — do not zero it for NEUTRAL speech.
        expression = max(
            0.0,
            min(1.0, max(float(render.expression), float(getattr(self, "_ml_smile", 0.0)))),
        )
        if self._tickfeed_look_authority:
            # Labels are already 0..1 LOOK amounts — do not apply old ×12 boost
            # (that left the mouth stuck open on small residuals).
            openness = float(self._plate_openness_current) * 8.0
            expression = float(self._ml_smile)
        else:
            # Openness units for shader (/14): boost from plate drive so open.png
            # has a strong signal even when jaw physics is still catching up.
            openness = max(
                float(render.mouth_openness),
                float(self._plate_openness_current)
                * float(self._display_recipe.openness_plate_boost),
            )
        self._target_mouth_pose = MouthPose(
            width=render.mouth_width,
            openness=openness,
            roundness=render.mouth_roundness,
            expression=expression,
        )
        del specs  # legacy ±4 muscle field impulses unused (TickFeed owns ch0/1)
        self._refresh_frame_layers()
        self._cpu_frame_ms = (time.perf_counter() - cpu_started) * 1000.0

    def _hysteresis_plate_open(self, open_amt: float) -> float:
        """Hold LOOK open amount so brief dips don't smear mid-blend frames.

        Open follows quickly; closes (especially bilabial zeros) win immediately
        so word locking is not fighting a sticky open hold.
        """
        target = max(0.0, min(1.0, float(open_amt)))
        prev = float(getattr(self, "_plate_open_hyst", 0.0) or 0.0)
        if target <= 0.02:
            self._plate_open_hyst = 0.0
            return 0.0
        if target >= prev:
            if target - prev >= 0.03 or target >= 0.95:
                self._plate_open_hyst = target
        elif prev - target >= 0.05 or target <= 0.08:
            self._plate_open_hyst = target
        elif target < 0.20 and prev > 0.25:
            self._plate_open_hyst = max(target, prev * 0.45)
        return float(self._plate_open_hyst)

    def _update_mouth_transition(self, plate_amt: float) -> str:
        """REST / OPENING / OPEN / CLOSING from openness velocity (single-owner)."""
        dt = 1.0 / 60.0
        prev = float(getattr(self, "_plate_open_prev", 0.0) or 0.0)
        raw_vel = (float(plate_amt) - prev) / max(dt, 1e-4)
        self._plate_open_vel = (
            0.65 * float(getattr(self, "_plate_open_vel", 0.0) or 0.0)
            + 0.35 * raw_vel
        )
        self._plate_open_prev = float(plate_amt)
        vel = float(self._plate_open_vel)
        if plate_amt < 0.08 and abs(vel) < 0.8:
            state = "REST"
        elif vel > 1.0 or (0.08 <= plate_amt < 0.55 and vel > 0.25):
            state = "OPENING"
        elif vel < -1.0 or (plate_amt > 0.12 and vel < -0.25):
            state = "CLOSING"
        elif plate_amt >= 0.45:
            state = "OPEN"
        else:
            state = "OPENING" if plate_amt >= 0.08 else "REST"
        self._mouth_transition = state
        return state

    def _apply_tickfeed_labels_to_look(self, pkg) -> None:
        """B4 — LOOK plates owned by TickPackage labels (sole authority)."""
        # Miss must not use producer last_labels — with ring lead those are
        # ahead of the master tick and open plates while FIELD is still empty.
        if pkg is None:
            return
        labels = getattr(pkg, "labels", None)
        if labels is None:
            return
        smile = float(labels.smile_amt)
        open_amt = float(labels.open_amt)
        surprise = float(labels.surprise_amt)
        brow = float(getattr(labels, "brow_amt", 0.0) or 0.0)
        # field_only: keep label openness for debug/status, but force plates closed
        # so NWR FIELD is the only visible mouth motion.
        plate_amt = self._hysteresis_plate_open(open_amt)
        if str(getattr(self, "_calibrate_mode", "normal")) == "field_only":
            plate_amt = 0.0
            self._plate_open_hyst = 0.0
            smile = 0.0
            surprise = 0.0
        # Single-owner transition state for FIELD/atlas policy this tick.
        state = self._update_mouth_transition(plate_amt)
        self._tickfeed_look_authority = True
        self._ml_smile = smile
        self._ml_openness = open_amt
        self._plate_openness_current = plate_amt
        self._expr_plate_blend = surprise
        # Side B emotion LOOK: brow from look_drive (ANGRY/SURPRISE) — full-face
        # FIELD already carries measured upper-face velocity in the timeline.
        self._expr_target_brow = max(float(self._expr_target_brow), brow)
        if surprise > 0.2 or brow > 0.35:
            self._expr_target_widen = max(float(self._expr_target_widen), brow * 0.55)
        if getattr(labels, "viseme_id", None) is not None:
            from aiface.tickfeed.schema import VISEME_TABLE

            vid = int(labels.viseme_id)
            if 0 <= vid < len(VISEME_TABLE):
                self._held_speech_viseme = VISEME_TABLE[vid]
        if self._plate_atlas is not None and self._atlas_textures:
            # Plate = viseme identity; weight = openness (B4 label intensity).
            phoneme = str(self._held_speech_viseme or "REST")
            ia, ib, mix = self._plate_atlas.pair_for_viseme(
                phoneme, hard_snap=True
            )
            ia = max(0, min(ia, len(self._atlas_textures) - 1))
            ib = max(0, min(ib, len(self._atlas_textures) - 1))
            # During OPENING/CLOSING, commit plate amount earlier so the oral
            # disk isn't half plate / half FIELD (ghost mid-band blur).
            amount = plate_amt
            if state in {"OPENING", "CLOSING"} and amount > 0.06:
                amount = max(amount, min(1.0, amount * 1.25 + 0.10))
            elif state == "OPEN" and amount > 0.35:
                amount = max(amount, min(1.0, amount * 1.08))
            self._plate_pair = (ia, ib)
            self._plate_blend = (float(mix), amount)
            self._plate_blend_current = (float(mix), amount)
            self._plate_openness_current = amount
        # Open speech owns LOOK — never stack smile.png (corner scars / double mouth).
        if plate_amt > 0.10:
            smile = 0.0
            self._ml_smile = 0.0
        else:
            self._ml_smile = smile

    def _simulate_tick(self) -> None:
        """Push ahead into ring → master pops current tick → GPU ingest (B1–B3)."""
        from aiface.tickfeed.schema import RING_DEPTH

        next_tick = self.tick + 1
        if self._tickfeed is not None and self._tickfeed.enabled:
            live = self._tickfeed_live_active()
            mode = self._tickfeed_live_mode()
            # Keep speech overlay while visemes remain even if until lapsed —
            # otherwise measured timeline steals mid-sentence LOOK/FIELD.
            speech_pending = bool(self._visemes) or self._presence == PRESENCE_SPEAKING
            if (live or speech_pending) and self._tickfeed_live is not None:
                if not live and speech_pending:
                    # Short decay bridge — do NOT sticky-refresh the prior open
                    # amount (that parked mid-open until the next vowel).
                    now = time.perf_counter() - self._clock0
                    prev_until = float(self._tickfeed_live.get("until") or 0.0)
                    age = max(0.0, now - prev_until)
                    open_amt = float(self._tickfeed_live.get("open", 0.0))
                    decay = max(0.0, 1.0 - age / 0.10)
                    open_amt = open_amt * decay if open_amt > 0.10 else 0.0
                    if open_amt < 0.06:
                        open_amt = 0.0
                        self._tickfeed_live["phoneme"] = "CLOSED"
                        self._held_speech_viseme = "CLOSED"
                        self._plate_open_hyst = 0.0
                    self._tickfeed_live["open"] = open_amt
                    self._tickfeed_live["smile"] = 0.0
                    self._tickfeed_live["until"] = now + 0.04
                    live = True
                    mode = str(self._tickfeed_live.get("mode") or "speech")
                open_amt = float(self._tickfeed_live.get("open", 0.0))
                smile_amt = float(self._tickfeed_live.get("smile", 0.0))
                surprise_amt = float(self._tickfeed_live.get("surprise", 0.0))
                phoneme = str(self._tickfeed_live.get("phoneme") or "REST")
                emotion = str(self._tickfeed_live.get("emotion") or "NEUTRAL")
            elif self._presence == PRESENCE_ZERO:
                # Switchable idle LOOK: neutral / smile / waiting.
                self._apply_zero_mood_overlay()
                open_amt, smile_amt, surprise_amt, phoneme, emotion = (
                    self._zero_mood_drives()
                )
                live = True
                mode = "zero"
            else:
                # Fallback closed REST — blink comes from EyeSystem.
                open_amt = 0.0
                smile_amt = 0.0
                surprise_amt = 0.0
                phoneme = "REST"
                emotion = "NEUTRAL"
            # Speech / hearing / zero-mood overlays own LOOK; keep FIELD still.
            live_overlay = live and mode in {"speech", "hearing", "zero"}
            # Local-ring: produce the tick we consume (same 16.7 ms). Wire-loop
            # keeps B3 producer lead so the transport can jitter.
            if self._tickfeed.wire_loop:
                produce_tick = int(next_tick) + int(RING_DEPTH)
            else:
                produce_tick = int(next_tick)
            self._tickfeed.push_drives(
                tick=produce_tick,
                open_amt=open_amt,
                smile_amt=smile_amt,
                surprise_amt=surprise_amt,
                phoneme=phoneme,
                emotion=emotion,
                live_speech=live_overlay,
            )
            # Wire-loop: produce only pushes transport; master ring is fed from wire.
            if self._tickfeed.wire_loop:
                self._tickfeed.ingest_from_wire(produce_tick)
            pkg = self._tickfeed.pop_for_master(next_tick)
            self._apply_tickfeed_labels_to_look(pkg)
            self._tickfeed_last_pkg = pkg
            self._tickfeed_last_live = bool(live_overlay)
            self._tickfeed_last_live_mode = str(mode or "")
            self.queue_tick_package(pkg)
        else:
            self._tickfeed_last_pkg = None
            self._tickfeed_last_live = False
            self._tickfeed_last_live_mode = ""
            self.queue_tick_package(None)
        super()._simulate_tick()

    def _sample_mouth_telemetry(self) -> None:
        # Hot path: read back only the mouth row band (~35 rows ≈ 1.1 MB), not
        # the whole 8 MB world — BDS/NWR state belongs on the GPU.
        cx, cy = int(self._mouth_center[0]), int(self._mouth_center[1])
        radius = int(MOUTH_RADIUS) + 2
        rows = self._read_world_rows(cy - radius, cy + radius + 1)
        patch = rows[
            :,
            max(cx - radius, 0) : min(cx + radius + 1, self.grid_width),
        ]
        speed = np.hypot(patch[..., 0], patch[..., 1])
        unlocked = patch[..., HUMAN_LOCK_CHANNEL] < 0.5
        self._telemetry.mean_speed = (
            float(speed[unlocked].mean()) if unlocked.any() else 0.0
        )
        self._telemetry.peak_speed = (
            float(speed[unlocked].max()) if unlocked.any() else 0.0
        )
        self._telemetry.active_cells = int((unlocked & (speed > 0.02)).sum())

    def _mouth_probe_snapshot(self) -> dict[str, object]:
        """Live GPU metrics for Path A: ownership + unlocked mouth disc."""
        grid = self._read_world()
        ownership = self._mouth_ownership
        cx, cy = int(self._mouth_center[0]), int(self._mouth_center[1])
        radius = int(MOUTH_RADIUS) + 2
        y0, y1 = max(cy - radius, 0), min(cy + radius + 1, self.grid_height)
        x0, x1 = max(cx - radius, 0), min(cx + radius + 1, self.grid_width)
        patch = grid[y0:y1, x0:x1]
        unlocked = patch[..., HUMAN_LOCK_CHANNEL] < 0.5
        speed = np.hypot(patch[..., 0], patch[..., 1])
        global_unlocked = float((grid[..., HUMAN_LOCK_CHANNEL] < 0.5).mean())
        disc_unlocked = float(unlocked.mean()) if unlocked.size else 0.0
        albedo = patch[..., 8:11]
        return {
            "tick": int(self.tick),
            "ownership": ownership.as_dict(),
            "mouth_center": [float(self._mouth_center[0]), float(self._mouth_center[1])],
            "mouth_radius": float(MOUTH_RADIUS),
            "disc": {
                "unlocked_frac": disc_unlocked,
                "mean_speed": float(speed[unlocked].mean()) if unlocked.any() else 0.0,
                "peak_speed": float(speed[unlocked].max()) if unlocked.any() else 0.0,
                "active_cells": int((unlocked & (speed > 0.02)).sum()),
                "albedo_mean": float(albedo.mean()) if albedo.size else 0.0,
            },
            "global_unlocked_frac": global_unlocked,
            "phoneme": self._render_state.last_phoneme,
            "emotion": self._telemetry.last_emotion,
            "display_layers": self._frame_layers.snapshot(),
            "plate_openness": float(self._plate_openness_current),
            "plate_blend": [
                float(self._plate_blend_current[0]),
                float(self._plate_blend_current[1]),
            ],
            "open_close_source": self._open_close_source,
            "ml_openness": float(self._ml_openness),
            "ml_jaw": float(self._ml_jaw),
            "ml_width": float(self._ml_width),
            "plate_gate": float(self._ml_plate_gate),
            "live_controls": {
                "openness_n": float(self._ml_openness),
                "jaw_n": float(self._ml_jaw),
                "width_n": float(self._ml_width),
                "plate_gate": float(self._ml_plate_gate),
                "source": self._open_close_source,
            },
            "invariants": {
                "dark_cavity": False,
                "muscle_warp_primary": True,
                "albedo_identity": True,
            },
        }

    # --------------------------------------------------------------- frames

    def on_render(self, time_value: float, frame_time: float) -> None:
        frame_started = time.perf_counter()
        if self._bridge is not None:
            self._bridge.service()
        self._drain_chat()
        self._drain_voice_channel()
        self._advance_visemes()
        self._update_presence()
        # ML open/close first so jaw bias lands in this tick's biomechanics step.
        self._update_open_close_ml(frame_time)
        # Gate field/smile before impulses land this tick.
        self._refresh_mouth_ownership()
        self._step_biomechanics(frame_time)
        self._ease_mouth_pose(frame_time)
        self._ease_plate_blend(frame_time)
        self._sync_expression_from_emotion()
        self._ease_expression_state(frame_time)
        # Re-resolve after openness ease so plate amount matches gates.
        self._refresh_mouth_ownership()
        self._sync_plate_blend_from_phoneme()
        # Avatar uniforms refresh in _on_pre_draw AFTER TickFeed ingest so
        # field_gain mute matches this tick's LOOK open (not the previous).
        super().on_render(time_value, frame_time)
        self._emit_gpu_tick_log()
        self._emit_side_a_debug()
        # GPU latency: wait for the submitted work to land, then sample.
        self.ctx.finish()
        self._gpu_times.append((time.perf_counter() - frame_started) * 1000.0)
        self._telemetry.gpu_frame_ms = sum(self._gpu_times) / max(
            len(self._gpu_times), 1
        )
        self._telemetry_age += 1
        if self._telemetry_age >= 4:
            self._telemetry_age = 0
            self._sample_mouth_telemetry()
        self._write_capture_frame()

    def _emit_side_a_debug(self) -> None:
        """Record the TickPackage just posted to Side A (GPU master)."""
        dbg = getattr(self, "_side_a_debug", None)
        if dbg is None or not dbg.enabled:
            return
        if self._tickfeed is None or not self._tickfeed.enabled:
            return
        tick = int(self.tick)
        if tick == getattr(self, "_side_a_debug_last_tick", -1):
            return
        self._side_a_debug_last_tick = tick
        dbg.record(
            master_tick=tick,
            package=self._tickfeed_last_pkg,
            live_speech=bool(self._tickfeed_last_live),
            live_mode=str(self._tickfeed_last_live_mode or ""),
            plate_open=float(self._plate_openness_current),
            smile=float(self._ml_smile),
            viseme=str(self._held_speech_viseme or ""),
            field_gain_recipe=float(self._display_recipe.field_warp_gain),
            field_gain_eff=float(self._field_gain_eff),
            muscles=int(self._active_muscle_count),
            gpu_peak=float(self._telemetry.peak_speed),
            gpu_mean=float(self._telemetry.mean_speed),
            gpu_cells=int(self._telemetry.active_cells),
            presence=str(self._presence),
        )

    def _emit_gpu_tick_log(self) -> None:
        """Log GPU recipe object drives once per 60 Hz simulation tick."""
        if not self._gpu_log:
            return
        tick = int(self.tick)
        if tick == self._gpu_log_last_tick:
            return
        self._gpu_log_last_tick = tick
        from aiface.speech import canonical_viseme

        cmd = self._layer_command
        phoneme = canonical_viseme(cmd.phoneme or "REST")
        plate_viseme = canonical_viseme(cmd.atlas_viseme or phoneme)
        ia, ib = self._plate_pair
        mix, amount = self._plate_blend_current
        jaw = float(getattr(self._render_state, "jaw_angle", 0.0))
        layers = "+".join(self._frame_layers.ordered_active())
        line = (
            f"gpu@60 t={tick} viseme={phoneme} plate={plate_viseme} "
            f"jaw={jaw:.3f} "
            f"open={float(cmd.plate_openness):.3f} "
            f"smile={float(cmd.smile_amount):.3f} "
            f"mouth_obj={int(self._mouth_object_cells)}"
            f"@({self._mouth_center[0]:.1f},{self._mouth_center[1]:.1f}) "
            f"plates=open:{amount:.2f} smile:{float(cmd.smile_amount):.2f} "
            f"atlas={ia}/{ib} mix={mix:.2f} "
            f"muscles={int(self._active_muscle_count)} "
            f"brow={float(self._expr_brow_raise):.2f} "
            f"widen={float(self._expr_eye_widen):.2f} "
            f"expr={self._expr_role_name}:{float(self._expr_plate_blend):.2f} "
            f"layers={layers} "
            f"recipe=on warp_gain={float(self._display_recipe.field_warp_gain):.2f} "
            f"sharpness={float(self._display_recipe.plate_sharpness):.2f} "
            f"src={cmd.source} until={cmd.active_until:.2f}"
        )
        if self._gpu_log_handle is not None:
            self._gpu_log_handle.write(line + "\n")
            if tick % 12 == 0:
                self._gpu_log_handle.flush()
        if self._gpu_log_verbose or tick % 12 == 0:
            print(line, flush=True)

    def _write_capture_frame(self) -> None:
        if self._capture_directory is None:
            return
        if self._captured >= self._capture_budget:
            self.wnd.close()
            return
        path = Path(self._capture_directory) / f"frame_{self._captured:04d}.png"
        path.write_bytes(self._preview())
        self._captured += 1

    def _voice_hud(self) -> str:
        """Show which clock the lips are following, and how late it is."""
        stream = self._voice
        if stream is not None and self._voice_chunks:
            stats = stream.stats()
            return (
                f"  sync {stats.received_seconds:.1f}s in  "
                f"lag {stats.mean_lag * 1000:.0f}ms"
            )
        if self._speech is None:
            return ""
        telemetry = self._telemetry
        clip = f" {telemetry.clip_seconds:.1f}s" if telemetry.clip_seconds else ""
        return f"  lip-lock {telemetry.alignment or 'ready'}{clip}"

    def _drop_hud(self) -> str:
        """Never let a dropped command packet be invisible."""
        if not self.dropped_commands:
            return ""
        return f"  DROPPED {self.dropped_commands}"

    def _hud_lines(self) -> list[str]:
        telemetry = self._telemetry
        state = self._render_state
        ranked = sorted(
            (
                (name, value)
                for name, value in state.group_activations.items()
                if value >= 0.08
            ),
            key=lambda item: (-item[1], item[0]),
        )[:3]
        muscles = (
            ",".join(f"{name[:4]}:{value:.2f}" for name, value in ranked)
            if ranked
            else "none"
        )
        return [
            (
                f"AIFACE BIOMECH  FPS {self._fps:.0f}  |  "
                f"GPU {telemetry.gpu_frame_ms:.2f} ms  |  "
                f"CPU {self._cpu_frame_ms:.2f} ms  |  "
                f"view {DEBUG_VIEW_NAMES.get(self._debug_view, '?')}"
            ),
            (
                f"Phoneme {state.last_phoneme}/{telemetry.last_emotion}  "
                f"pending {len(self._visemes)}  "
                f"impulses {state.active_impulse_count}  "
                f"dom {state.dominant_emotion}:{state.dominant_emotion_value:+.2f}"
                f"{self._voice_hud()}{self._drop_hud()}"
            ),
            (
                f"Jaw {state.jaw_angle:.3f} rad  blink {state.blink_timer:.2f}s  "
                f"breath {state.breath_phase:.2f}  "
                f"contracting {self._active_muscle_count}/"
                f"{len(self._biomech.registry)}  [{muscles}]"
            ),
            (
                f"Mouth |V| mean {telemetry.mean_speed:.3f} "
                f"peak {telemetry.peak_speed:.3f}  "
                f"cells {telemetry.active_cells}  |  "
                f"t={state.simulation_time:.1f}s  "
                f"tick {self.tick}@{TICK_RATE_HZ}Hz"
            ),
        ]

    def on_resize(self, width: int, height: int) -> None:
        super().on_resize(width, height)
        self._relayout_frame()

    def on_key_event(self, key: int, action: int, modifiers: object) -> None:
        keys = self.wnd.keys
        if action == keys.ACTION_PRESS and self._handle_chat_key(key, modifiers):
            return
        if action == keys.ACTION_PRESS:
            if key == keys.H:
                self._show_locks = not self._show_locks
                self._update_avatar_uniforms()
                print(f"Master-Lock overlay: {'ON' if self._show_locks else 'OFF'}")
                return
            if key == keys.M and not (
                self._chat_box_visible and self._chatbox.focused
            ):
                self._cycle_mouth_speed()
                return
            if key == keys.Z and not (
                self._chat_box_visible and self._chatbox.focused
            ):
                self._cycle_zero_mood()
                return
            function_keys = {
                keys.F1: 1,
                keys.F2: 2,
                keys.F3: 3,
                keys.F4: 4,
                keys.F5: 5,
                keys.F6: 6,
                keys.F7: 7,
                keys.F8: 8,
                keys.F9: 9,
                keys.F10: 10,
                keys.F11: 11,
            }
            if key in function_keys:
                requested = function_keys[key]
                self._debug_view = 0 if self._debug_view == requested else requested
                self._update_avatar_uniforms()
                print(
                    f"Debug view: {DEBUG_VIEW_NAMES.get(self._debug_view, 'portrait')}"
                )
                return
        super().on_key_event(key, action, modifiers)

    def on_mouse_press_event(self, x: int, y: int, button: int) -> None:
        """Mouth hold slider + speed dropdown in the chat panel."""
        if not self._chat_box_visible:
            return
        control, ox, _oy = self._hit_ui(int(x), int(y))
        if control is None:
            self._mouth_hold_dragging = False
            if self._mouth_menu_open:
                self._mouth_menu_open = False
                self._overlay_dirty = True
            return
        if control == "mouth_hold_track":
            self._mouth_hold_dragging = True
            scale = self._mouth_hold_scale_from_x(ox)
            if scale is not None:
                self._set_mouth_hold_scale(scale, announce=False)
            return
        if control == "mouth_button":
            self._mouth_menu_open = not self._mouth_menu_open
            self._overlay_dirty = True
            return
        if control.startswith("mouth_opt_"):
            label = control.removeprefix("mouth_opt_")
            for preset in MOUTH_SPEED_PRESETS:
                if preset.label == label:
                    self._apply_mouth_speed_preset(preset, announce=True)
                    break

    def on_mouse_drag_event(self, x: int, y: int, dx: int, dy: int) -> None:
        del y, dx, dy
        if not self._mouth_hold_dragging:
            return
        _control, ox, _oy = self._hit_ui(int(x), int(y))
        scale = self._mouth_hold_scale_from_x(ox)
        if scale is not None:
            self._set_mouth_hold_scale(scale, announce=False)

    def on_mouse_position_event(self, x: int, y: int, dx: int, dy: int) -> None:
        """Some backends skip drag events — keep sliding while pressed."""
        del dx, dy
        if not self._mouth_hold_dragging:
            return
        _control, ox, _oy = self._hit_ui(int(x), int(y))
        scale = self._mouth_hold_scale_from_x(ox)
        if scale is not None:
            self._set_mouth_hold_scale(scale, announce=False)

    def on_mouse_release_event(self, x: int, y: int, button: int) -> None:
        del x, y, button
        if self._mouth_hold_dragging:
            self._mouth_hold_dragging = False
            dwell_ms = int(self._mouth_timeline.min_dwell_s * 1000)
            print(f"Mouth hold: {dwell_ms}ms", flush=True)

    def on_close(self) -> None:
        self._shutdown.set()
        if self._gpu_log_handle is not None:
            with contextlib.suppress(OSError):
                self._gpu_log_handle.flush()
                self._gpu_log_handle.close()
            self._gpu_log_handle = None
        if self._bridge is not None:
            self._bridge.stop()
            self._bridge = None
        if self._audio is not None:
            self._audio.close()
            self._audio = None
        self._avatar_base_texture.release()
        self._avatar_parts_texture.release()
        self._avatar_tissue_texture.release()
        self._avatar_open_plate_texture.release()
        self._avatar_smile_plate_texture.release()
        self._avatar_expr_plate_texture.release()
        for texture in self._atlas_textures:
            texture.release()
        super().on_close()


__all__ = [
    "DEBUG_VIEW_NAMES",
    "MOUTH_RADIUS",
    "AvatarFaceApp",
    "SpokenReply",
    "VectorTelemetry",
]
