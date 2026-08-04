"""Console entry points: ``chorusface``, ``chorusface-seed``, and ``chorusface-sync``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from chorusface.paths import DEFAULT_AVATAR_FACE, ensure_output_tree


def _ensure_default_world(args: list[str]) -> list[str]:
    """Seed a deterministic synthetic face when the user has not built one.

    ``moderngl_window`` parses ``argv`` inside ``run_window_config``, so the
    world has to exist before the window opens.
    """
    if any(part == "--world" or part.startswith("--world=") for part in args):
        return args
    if not DEFAULT_AVATAR_FACE.exists():
        from chorusface.seed import build_avatar_seed, write_seed_bundle

        ensure_output_tree()
        written = write_seed_bundle(
            build_avatar_seed(synthetic=True), DEFAULT_AVATAR_FACE
        )
        print(f"Seeded a synthetic avatar at {written['world']}")
        print("Build your own portrait with: chorusface-seed --input portrait.jpg")
    return ["--world", str(DEFAULT_AVATAR_FACE), *args]


def main_app(argv: Sequence[str] | None = None) -> int:
    """Open the chat-driven avatar window."""
    import moderngl_window as mglw

    from chorusface.app import AvatarFaceApp

    args = _ensure_default_world(list(sys.argv[1:] if argv is None else argv))
    sys.argv = [sys.argv[0], *args]
    mglw.run_window_config(AvatarFaceApp)
    return 0


def _run_repo_script(script_name: str, argv: Sequence[str] | None = None) -> int:
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[2] / "scripts" / script_name
    if not script.is_file():
        print(f"Missing scripts/{script_name} — run from the ChorusFace repo root.")
        return 2
    saved = list(sys.argv)
    try:
        sys.argv = [str(script), *(list(argv) if argv is not None else list(sys.argv[1:]))]
        spec = importlib.util.spec_from_file_location(f"chorusface_{script.stem}", script)
        if spec is None or spec.loader is None:
            print(f"Could not load {script}")
            return 2
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return int(module.main())
    finally:
        sys.argv = saved


def main_beta(argv: Sequence[str] | None = None) -> int:
    """Product beta launcher — host owns LLM; FaceBridge /speak drives the face."""
    return _run_repo_script("run_chorusface_beta.py", argv)


def main_service(argv: Sequence[str] | None = None) -> int:
    """Headless/container face service — PrismAPI speak + MJPEG embed."""
    return _run_repo_script("run_chorusface_service.py", argv)


def main_seed(argv: Sequence[str] | None = None) -> int:
    """Convert a portrait (or a synthetic face) into a locked avatar seed."""
    from chorusface.seed import main as seed_main

    return seed_main(argv)


def main_capture(argv: Sequence[str] | None = None) -> int:
    """Digest a short face video or stills into seed + expression plates."""
    from chorusface.capture import main as capture_main

    return capture_main(argv)


def _sync_parser() -> argparse.ArgumentParser:
    from chorusface.stream import DEFAULT_CHUNK_SECONDS, DEFAULT_LOOKAHEAD
    from chorusface.sync import DEFAULT_BUDGET_MS

    parser = argparse.ArgumentParser(
        prog="chorusface-sync",
        description=(
            "Measure what streaming costs. The same utterance is aligned twice: "
            "once offline with the whole clip available, once through the live "
            "channel a chunk at a time. The difference in onset times is reported "
            "in milliseconds, and a budget turns it into a pass or a fail."
        ),
    )
    parser.add_argument(
        "--text",
        action="append",
        default=[],
        help="Utterance to measure; repeatable. Defaults to the fixture set",
    )
    parser.add_argument(
        "--wav",
        type=Path,
        default=None,
        help="Measure this recording instead of synthesising (needs one --text)",
    )
    parser.add_argument(
        "--chunk-ms",
        type=float,
        default=DEFAULT_CHUNK_SECONDS * 1000.0,
        help="Chunk size the audio is delivered in",
    )
    parser.add_argument(
        "--lookahead-ms",
        type=float,
        default=DEFAULT_LOOKAHEAD * 1000.0,
        help="Arrived audio the channel holds back before judging it",
    )
    parser.add_argument(
        "--budget-ms",
        type=float,
        default=DEFAULT_BUDGET_MS,
        help="Fail if the trim-compensated 95th percentile exceeds this",
    )
    parser.add_argument(
        "--tts-backend",
        choices=("auto", "openai", "sapi", "command"),
        default="auto",
        help="Voice used to produce the fixture clips",
    )
    parser.add_argument(
        "--tts-command",
        default="",
        help="Explicit synthesiser command line, for the command backend",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON only")
    parser.add_argument(
        "--detail", action="store_true", help="Print every viseme comparison"
    )
    return parser


def main_sync(argv: Sequence[str] | None = None) -> int:
    """Report the streaming-versus-offline alignment error, and gate on it."""
    from chorusface.audio import AudioError, decode_audio
    from chorusface.stream import StreamConfig
    from chorusface.sync import ORACLE_LINES, measure_sync, summarise
    from chorusface.tts import TTSError, build_synthesizer

    args = _sync_parser().parse_args(argv)
    lines = list(args.text) or list(ORACLE_LINES)

    clips = []
    if args.wav is not None:
        if len(lines) != 1:
            print("--wav needs exactly one --text: the words spoken in it")
            return 2
        try:
            clips.append((lines[0], decode_audio(args.wav.read_bytes())))
        except (OSError, AudioError) as exc:
            print(f"could not read {args.wav}: {exc}")
            return 2
        source = str(args.wav)
    else:
        try:
            voice = build_synthesizer(
                backend=args.tts_backend, command=args.tts_command
            ).voice
            for text in lines:
                clips.append((text, voice.synthesize(text)))
        except TTSError as exc:
            print(f"no fixture voice available: {exc}")
            print("Install espeak-ng, or pass --wav with a recording.")
            return 2
        source = voice.name

    reports = []
    for text, clip in clips:
        reports.append(
            measure_sync(
                text,
                clip,
                config=StreamConfig(
                    sample_rate=clip.sample_rate,
                    lookahead_seconds=args.lookahead_ms / 1000.0,
                ),
                chunk_seconds=args.chunk_ms / 1000.0,
            )
        )

    summary = summarise(reports)
    passed = all(report.within(args.budget_ms) for report in reports)
    if args.json:
        print(
            json.dumps(
                {
                    "source": source,
                    "budget_ms": args.budget_ms,
                    "passed": passed,
                    "summary": summary,
                    "utterances": [report.as_dict() for report in reports],
                },
                indent=2,
            )
        )
    else:
        print(f"clip source      {source}")
        for report in reports:
            print()
            print(report.table())
            if args.detail:
                print(report.rows())
        worst = summary["trimmed_p95_ms"]["worst"]  # type: ignore[index]
        mean = summary["trimmed_p95_ms"]["mean"]  # type: ignore[index]
        print()
        print(
            f"{summary['utterances']} utterances, {summary['coverage']:.0%} of "
            f"offline visemes matched"
        )
        print(f"after trim       mean p95 {mean:.1f} ms   worst p95 {worst:.1f} ms")
        print(
            f"budget {args.budget_ms:.0f} ms  ->  {'PASS' if passed else 'FAIL'}"
        )
    return 0 if passed else 1


__all__ = ["main_app", "main_capture", "main_seed", "main_sync"]
