"""Package a world into a bundle an external chat assistant can actually use.

ChatGPT and Gemini cannot run this runtime: there is no GPU behind a chat
window, so handing them a raw ``.bds`` gets you a binary blob and a guess. This
module produces the artifacts a chat model *can* consume — a contact sheet and
an animated loop of the world in motion, per-sample numeric summaries, the
command grammar, and a dependency-light reader for code interpreters — so the
model can watch the simulation, reason about it, and reply with commands you
apply through :mod:`ai_agent`.

The bundle is observation only. Nothing here writes GPU memory, and a model's
reply still passes the same validation and human-lock enforcement as any other
command source.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from ai_commands import Operation, schema_for_authority
from ai_world import export_ai_context, generate_ai_summary
from bds_format import (
    PRIORITY_LEVELS,
    TICK_RATE_HZ,
    VECTOR_DIMENSIONS,
    create_initial_grid,
    load_bds,
    save_bds,
)
from paths import DEFAULT_HANDOFF

BUNDLE_VERSION: Final = "nwr-handoff-1.0"
# A bundle reply is submitted through the bridge, which grants AI authority by
# default, so the grammar the bundle advertises is narrowed to that level.
BUNDLE_AUTHORITY: Final = PRIORITY_LEVELS["ai"]
DEFAULT_TICKS: Final = 600
DEFAULT_FRAMES: Final = 9
DEFAULT_TILE: Final = 384
GIF_FRAME_MS: Final = 400
LABEL_BAND: Final = 28

# Kept small on purpose: a chat model reasons better from a short, honest
# briefing plus structured JSON than from a wall of prose.
INSTRUCTIONS: Final = """# Neural World Runtime — assistant handoff

You are looking at a snapshot of a running physics substrate, not a video file
and not a game you can execute. Read this before answering.

## What is in this folder

| File | What it is |
| --- | --- |
| `filmstrip.png` | The world simulated forward, sampled into one labelled image. Read this first. |
| `world.gif` | The same samples as an animated loop. |
| `frames/` | The individual sampled frames at full resolution. |
| `timeline.json` | Numeric summary at each sampled tick. Trust these numbers over the pixels. |
| `context.json` | World metadata, materials, statistics, and the full command grammar. |
| `world.bds` | The binary world itself, if you want to compute on the raw field. |
| `read_bds.py` | A NumPy-only reader for `world.bds`, for use in a code interpreter. |

## What the world is

A {width} x {height} grid. Every cell is {channels} float32 values, grouped as
physics (0-7), material (8-15), intent (16-23), and rules (24-31). It advances
at {tick_rate} ticks per second through GPU compute passes. Channel 24 is
`hard_surface`, a barrier that deflects flow. Channel 31 is `human_lock`, which
only a human may set and which you can never overwrite.

{origin}

## What you can and cannot do

You cannot step the simulation, render a frame, or write cells directly. You
propose commands as JSON; a human applies them and sends you the next
filmstrip. Supported actions, their exact parameters, and the capabilities that
are deliberately unsupported are all in `context.json` — read
`command_schema`, `supported_commands`, and `unsupported_commands` there rather
than guessing. Anything you invent is rejected by the compiler, not silently
approximated.

## How to answer

Reply with one JSON object matching `command_schema` in `context.json`, and
nothing else around it. This example is taken verbatim from the live grammar
and is known to compile:

```json
{example}
```

State your reasoning before the JSON if you want, but end with the object
alone. The human runs:

```
python ai_agent.py --commands-file reply.json --dry-run
```

which validates it, and drops `--dry-run` to apply it to the live world.
"""

READER_SOURCE: Final = '''"""Minimal .bds reader: NumPy only, no runtime dependencies.

Layout: an 8-byte magic, a little-endian uint32 JSON length, the UTF-8 JSON
header, zero padding to 4096 bytes, then a C-contiguous little-endian float32
payload of shape (height, width, 32). This mirrors bds_format.load_bds and is
verified against it by the project's test suite.
"""

from __future__ import annotations

import json
import struct
import zlib

import numpy as np

MAGIC = b"BDS1\\0\\0\\0\\0"
HEADER_SIZE = 4096
VECTOR_DIMENSIONS = 32
_PREFIX = struct.Struct("<8sI")


def load_bds(path):
    """Return ``(header, grid)`` after checking the magic and payload CRC32."""
    with open(path, "rb") as handle:
        fixed = handle.read(HEADER_SIZE)
        magic, length = _PREFIX.unpack_from(fixed)
        if magic != MAGIC:
            raise ValueError("Not a .bds file")
        header = json.loads(
            fixed[_PREFIX.size : _PREFIX.size + length].decode("utf-8")
        )
        payload = handle.read(int(header["payload_bytes"]))

    if f"{zlib.crc32(payload) & 0xFFFFFFFF:08x}" != str(header["payload_crc32"]).lower():
        raise ValueError("Payload CRC32 mismatch")

    width, height, _depth = header["grid_dimensions"]
    grid = np.frombuffer(payload, dtype="<f4").reshape(
        (height, width, VECTOR_DIMENSIONS)
    )
    return header, np.array(grid, dtype="<f4", order="C", copy=True)


CHANNELS = [
    "velocity_x", "velocity_y", "velocity_z", "density",
    "pressure", "shear", "temperature", "energy",
    "albedo_r", "albedo_g", "albedo_b", "opacity",
    "roughness", "metallic", "emission", "refraction",
    "attraction", "alignment", "user_affinity", "growth",
    "decay", "lifespan", "reserved_22", "reserved_23",
    "hard_surface", "permeability", "thermal_threshold", "phase_trigger",
    "reserved_28", "reserved_29", "authority_priority", "human_lock",
]

if __name__ == "__main__":
    import sys

    header, grid = load_bds(sys.argv[1] if len(sys.argv) > 1 else "world.bds")
    print(f"grid {grid.shape[1]}x{grid.shape[0]}, {grid.shape[2]} channels")
    for index, name in enumerate(CHANNELS):
        channel = grid[..., index]
        if float(np.abs(channel).max()) > 0.0:
            print(f"  {index:2d} {name:20s} mean {channel.mean():+.4f} max {channel.max():+.4f}")
'''


class HandoffError(RuntimeError):
    """A user-facing failure while building a bundle."""


@dataclass(frozen=True, slots=True)
class Sample:
    """One rendered moment of the world, with the numbers behind it."""

    tick: int
    png: bytes
    summary: dict[str, Any]


@dataclass(frozen=True, slots=True)
class BundleReport:
    """What a build produced."""

    directory: Path
    ticks: int
    samples: int
    files: list[str]
    total_bytes: int

    def describe(self) -> str:
        listing = "\n".join(f"  {name}" for name in self.files)
        return (
            f"Wrote {len(self.files)} files ({self.total_bytes / 1e6:.1f} MB) "
            f"covering {self.ticks} ticks in {self.samples} samples\n"
            f"{self.directory}\n{listing}"
        )


def sample_ticks(ticks: int, frames: int) -> list[int]:
    """Evenly spaced sample points from tick 0 through ``ticks`` inclusive."""
    if ticks < 0:
        raise ValueError("ticks must not be negative")
    if frames < 1:
        raise ValueError("frames must be at least 1")
    if frames == 1:
        return [0]
    step = ticks / (frames - 1)
    return sorted({int(round(index * step)) for index in range(frames)})


def sample_world(
    grid: npt.NDArray[np.float32],
    *,
    ticks: int = DEFAULT_TICKS,
    frames: int = DEFAULT_FRAMES,
    resolution: int = DEFAULT_TILE,
    schedule: Mapping[int, list[Operation]] | None = None,
    use_neural_material: bool = False,
    weights_path: Path | None = None,
) -> list[Sample]:
    """Simulate ``ticks`` forward, capturing a render and summary at each sample."""
    from export_video import HeadlessRenderer

    targets = sample_ticks(ticks, frames)
    pending = dict(schedule or {})
    captured: list[Sample] = []

    with HeadlessRenderer(
        resolution=resolution,
        grid=grid,
        use_neural_material=use_neural_material,
        weights_path=weights_path,
    ) as renderer:
        remaining = list(targets)
        while remaining:
            if renderer.tick == remaining[0]:
                remaining.pop(0)
                captured.append(
                    Sample(
                        tick=renderer.tick,
                        png=renderer.render_frame(),
                        summary=generate_ai_summary(
                            renderer.read_world(),
                            tick=renderer.tick,
                        ),
                    )
                )
                continue
            renderer.step(pending.pop(renderer.tick + 1, None))
    return captured


def build_filmstrip(samples: Sequence[Sample], *, tile: int = DEFAULT_TILE) -> bytes:
    """Lay the samples out as one labelled contact sheet.

    A single image is the only view every chat model can reliably read, so this
    is the artifact that actually conveys motion.
    """
    if not samples:
        raise ValueError("At least one sample is required")
    from PIL import Image, ImageDraw

    columns = max(1, math.ceil(math.sqrt(len(samples))))
    rows = math.ceil(len(samples) / columns)
    cell_height = tile + LABEL_BAND
    sheet = Image.new("RGB", (columns * tile, rows * cell_height), (8, 10, 18))
    draw = ImageDraw.Draw(sheet)
    font = _label_font()

    for index, sample in enumerate(samples):
        column, row = index % columns, index // columns
        left, top = column * tile, row * cell_height
        frame = Image.open(io.BytesIO(sample.png)).convert("RGB")
        if frame.size != (tile, tile):
            frame = frame.resize((tile, tile), Image.LANCZOS)
        sheet.paste(frame, (left, top))
        seconds = sample.tick / TICK_RATE_HZ
        draw.text(
            (left + 8, top + tile + 6),
            f"tick {sample.tick}  ({seconds:.2f}s)",
            fill=(210, 220, 240),
            font=font,
        )

    buffer = io.BytesIO()
    sheet.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def build_gif(
    samples: Sequence[Sample],
    *,
    tile: int = DEFAULT_TILE,
    frame_ms: int = GIF_FRAME_MS,
) -> bytes:
    """Animate the samples into a looping GIF."""
    if not samples:
        raise ValueError("At least one sample is required")
    from PIL import Image

    images = []
    for sample in samples:
        frame = Image.open(io.BytesIO(sample.png)).convert("RGB")
        if frame.size != (tile, tile):
            frame = frame.resize((tile, tile), Image.LANCZOS)
        images.append(frame.convert("P", palette=Image.ADAPTIVE))

    buffer = io.BytesIO()
    images[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=frame_ms,
        loop=0,
    )
    return buffer.getvalue()


def build_timeline(samples: Sequence[Sample]) -> dict[str, Any]:
    """Numeric companion to the filmstrip, so motion can be read as data."""
    frames = [
        {
            "tick": sample.tick,
            "seconds": round(sample.tick / TICK_RATE_HZ, 4),
            **sample.summary,
        }
        for sample in samples
    ]
    return {
        "bundle_version": BUNDLE_VERSION,
        "tick_rate_hz": TICK_RATE_HZ,
        "sampled_ticks": [sample.tick for sample in samples],
        "frames": frames,
    }


def describe_origin(header: Mapping[str, Any]) -> str:
    """One paragraph telling the model where this world came from."""
    application = header.get("application_metadata") or {}
    source = application.get("source") or {}
    conversion = application.get("video2game") or {}
    if source.get("kind") != "video":
        return (
            "This world was authored in the runtime itself, so its structure is "
            "whatever a human or the simulation put there."
        )
    width, height = source.get("resolution", [0, 0])
    return (
        f"This world was lifted from video: `{source.get('filename')}` "
        f"({width}x{height}), frame {source.get('frame_index')} of "
        f"{source.get('frame_count')}, using the `{conversion.get('preset')}` "
        "preset. Only motion, colour, and contours survived that conversion — "
        "there are no objects, characters, depth, or story in this world. Do "
        "not describe it as if it were the video's scene."
    )


def render_instructions(
    grid: npt.NDArray[np.float32],
    header: Mapping[str, Any],
) -> str:
    """Fill the briefing for this specific world.

    The worked example comes from the live grammar rather than being written
    out here, so it cannot drift away from what the compiler accepts. It is
    narrowed to bridge authority because that is the level the human applying
    the reply will be submitting under.
    """
    return INSTRUCTIONS.format(
        width=grid.shape[1],
        height=grid.shape[0],
        channels=VECTOR_DIMENSIONS,
        tick_rate=TICK_RATE_HZ,
        origin=describe_origin(header),
        example=json.dumps(
            schema_for_authority(BUNDLE_AUTHORITY)["example"], indent=2
        ),
    )


def write_bundle(
    output_directory: str | Path,
    *,
    world: str | Path | None = None,
    log: str | Path | None = None,
    ticks: int = DEFAULT_TICKS,
    frames: int = DEFAULT_FRAMES,
    tile: int = DEFAULT_TILE,
    include_world: bool = True,
    include_gif: bool = True,
    use_neural_material: bool = False,
    weights_path: Path | None = None,
) -> BundleReport:
    """Build a complete assistant handoff bundle on disk."""
    directory = Path(output_directory)
    if world is None:
        header: dict[str, Any] = {}
        grid = create_initial_grid()
    else:
        try:
            header, grid = load_bds(world)
        except (OSError, ValueError) as exc:
            raise HandoffError(f"Could not read {world}: {exc}") from None

    schedule = None
    if log is not None:
        from export_video import schedule_from_log

        try:
            schedule = schedule_from_log(log)
        except (OSError, ValueError) as exc:
            raise HandoffError(f"Could not read {log}: {exc}") from None

    samples = sample_world(
        grid,
        ticks=ticks,
        frames=frames,
        resolution=tile,
        schedule=schedule,
        use_neural_material=use_neural_material,
        weights_path=weights_path,
    )

    directory.mkdir(parents=True, exist_ok=True)
    frames_directory = directory / "frames"
    frames_directory.mkdir(exist_ok=True)

    written: list[Path] = []
    for sample in samples:
        path = frames_directory / f"tick_{sample.tick:06d}.png"
        path.write_bytes(sample.png)
        written.append(path)

    written.append(_write_bytes(directory / "filmstrip.png", build_filmstrip(samples, tile=tile)))
    if include_gif:
        written.append(_write_bytes(directory / "world.gif", build_gif(samples, tile=tile)))

    ai_metadata = header.get("ai_metadata") or {}
    context = export_ai_context(
        grid,
        tick=0,
        world_name=str(ai_metadata.get("world_name", "Neural World")),
        description=str(ai_metadata.get("description", "Interactive semantic simulation")),
        authority=BUNDLE_AUTHORITY,
    )
    context["bundle_version"] = BUNDLE_VERSION
    context["source_header"] = {
        key: value
        for key, value in header.items()
        # The anchor table and channel schema are already inside `metadata`.
        if key not in {"anchors", "channel_schema", "ai_metadata"}
    }
    written.append(_write_json(directory / "context.json", context))
    written.append(_write_json(directory / "timeline.json", build_timeline(samples)))

    written.append(
        _write_text(
            directory / "READ_ME_FIRST.md",
            render_instructions(grid, header),
        )
    )
    written.append(_write_text(directory / "read_bds.py", READER_SOURCE))

    if include_world:
        destination = directory / "world.bds"
        if world is not None:
            shutil.copyfile(world, destination)
        else:
            save_bds(destination, grid)
        written.append(destination)

    files = sorted(str(path.relative_to(directory)).replace("\\", "/") for path in written)
    return BundleReport(
        directory=directory,
        ticks=ticks,
        samples=len(samples),
        files=files,
        total_bytes=sum(path.stat().st_size for path in written),
    )


def _label_font() -> Any:
    from PIL import ImageFont

    try:
        return ImageFont.truetype("arial.ttf", 16)
    except OSError:
        return ImageFont.load_default()


def _write_bytes(path: Path, payload: bytes) -> Path:
    path.write_bytes(payload)
    return path


def _write_text(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _write_json(path: Path, payload: Any) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Package a world into a bundle you can upload to ChatGPT or Gemini."
        ),
    )
    parser.add_argument("--world", type=Path, help="Source .bds (default: seed world)")
    parser.add_argument("--log", type=Path, help="Optional .bdl history to replay")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_HANDOFF,
        help=f"Bundle directory (default: {DEFAULT_HANDOFF})",
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=DEFAULT_TICKS,
        help=f"How far to simulate (default {DEFAULT_TICKS})",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=DEFAULT_FRAMES,
        help=f"How many moments to sample (default {DEFAULT_FRAMES})",
    )
    parser.add_argument(
        "--tile",
        type=int,
        default=DEFAULT_TILE,
        help=f"Pixel size of each sampled frame (default {DEFAULT_TILE})",
    )
    parser.add_argument(
        "--no-world",
        action="store_true",
        help="Omit the .bds copy to keep the bundle small",
    )
    parser.add_argument("--no-gif", action="store_true", help="Skip the animated loop")
    parser.add_argument("--neural-material", action="store_true")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path(__file__).resolve().parent / "material_weights.npy",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    try:
        report = write_bundle(
            arguments.output,
            world=arguments.world,
            log=arguments.log,
            ticks=arguments.ticks,
            frames=arguments.frames,
            tile=arguments.tile,
            include_world=not arguments.no_world,
            include_gif=not arguments.no_gif,
            use_neural_material=arguments.neural_material,
            weights_path=arguments.weights,
        )
    except (HandoffError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(report.describe())
    print(
        "\nUpload the whole folder, or at minimum READ_ME_FIRST.md, "
        "filmstrip.png, and context.json."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BUNDLE_VERSION",
    "BundleReport",
    "DEFAULT_FRAMES",
    "DEFAULT_TICKS",
    "DEFAULT_TILE",
    "HandoffError",
    "INSTRUCTIONS",
    "READER_SOURCE",
    "Sample",
    "build_filmstrip",
    "build_gif",
    "build_timeline",
    "describe_origin",
    "main",
    "render_instructions",
    "sample_ticks",
    "sample_world",
    "write_bundle",
]
