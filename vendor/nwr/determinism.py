"""Measure simulation determinism instead of asserting it.

Floating-point compute shaders are not bit-identical across vendors: drivers are
free to choose different fused-multiply-add contractions, different
transcendental approximations, and different work distribution. No amount of
application code changes that, so this module does not claim bit-exactness
across devices. It makes the claim falsifiable instead.

Three separate guarantees, each measured rather than asserted:

1. **Same device, repeated run** - must be bit-identical. Any divergence is a
   real bug (an unsynchronised buffer, a race, uninitialised memory).
2. **Same device, replayed command log** - must be bit-identical to the
   original run. This is what the `.bdl` format promises. The live run writes a
   real log file and the replay leg reads *that file* back; replaying the same
   in-memory schedule would compare a run against itself and could not fail
   however much the log format lost.
3. **Command-log fidelity** - the log must also come back field for field, not
   merely produce the same world. A world comparison can only catch a lost field
   that happens to matter to this particular run, so the operations read back are
   checked against the ones that went in. The probe schedule is chosen so that
   each plausible loss - writer identity, authority level, a whole operation
   kind - changes the world as well.
4. **Different device** - compared against a recorded report with an explicit
   numeric tolerance. The report carries the device fingerprint, so a
   divergence is attributable rather than mysterious.

Usage::

    python determinism.py --record output/reports/intel-uhd.json --ticks 240
    python determinism.py --verify output/reports/intel-uhd.json --ticks 240
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from itertools import zip_longest
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from ai_commands import (
    CATEGORY_IDS,
    WRITER_AI,
    WRITER_HUMAN,
    Control,
    Segment,
    TemperatureDelta,
    VelocityImpulse,
)
from bds_format import (
    DTYPE,
    FORMAT_VERSION,
    PRIORITY_LEVELS,
    TICK_RATE_HZ,
    VECTOR_DIMENSIONS,
    create_initial_grid,
    load_bds,
)
from bds_history import (
    HistoryLog,
    SessionRecorder,
    is_lossy_version,
    read_log,
    replay_operations,
)
from paths import REPORTS

# 1.0 reports carry a `replay_match` that compared a run against itself, which
# could not fail. A 1.1 report's replay leg round-trips an actual log file, so
# the two numbers are not comparable and old reports must be re-recorded.
REPORT_VERSION: Final = "nwr-determinism-1.1"
DEFAULT_TICKS: Final = 240
# Empirical headroom for cross-vendor float32 drift over a few hundred ticks of
# a clamped field. Tightening this is a hardware question, not a code question.
DEFAULT_TOLERANCE: Final = 1e-3


@dataclass(frozen=True, slots=True)
class WorldDivergence:
    """Quantified difference between two worlds of the same shape."""

    bit_identical: bool
    max_absolute_error: float
    mean_absolute_error: float
    divergent_cells: int
    total_cells: int
    worst_channel: int | None

    @property
    def divergent_fraction(self) -> float:
        return self.divergent_cells / self.total_cells if self.total_cells else 0.0

    def within(self, tolerance: float) -> bool:
        return self.max_absolute_error <= tolerance

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["divergent_fraction"] = round(self.divergent_fraction, 8)
        return payload

    def describe(self) -> str:
        if self.bit_identical:
            return "bit-identical"
        # ASCII only: this text reaches Windows consoles using legacy codepages.
        return (
            f"max abs error = {self.max_absolute_error:.3e}, "
            f"mean abs error = {self.mean_absolute_error:.3e}, "
            f"{self.divergent_cells}/{self.total_cells} cells differ "
            f"({self.divergent_fraction:.4%}), worst channel {self.worst_channel}"
        )


def world_digest(grid: npt.NDArray[np.float32]) -> str:
    """Stable content hash of the exact float32 payload."""
    array = np.ascontiguousarray(grid, dtype=DTYPE)
    digest = hashlib.blake2b(array.tobytes(order="C"), digest_size=16)
    digest.update(
        json.dumps(
            [int(dimension) for dimension in array.shape],
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return digest.hexdigest()


def compare_worlds(
    left: npt.NDArray[np.float32],
    right: npt.NDArray[np.float32],
    *,
    epsilon: float = 0.0,
) -> WorldDivergence:
    """Compare two worlds channel by channel without hiding the difference."""
    first = np.ascontiguousarray(left, dtype=DTYPE)
    second = np.ascontiguousarray(right, dtype=DTYPE)
    if first.shape != second.shape:
        raise ValueError(
            f"Worlds have different shapes: {first.shape} vs {second.shape}"
        )
    if first.ndim != 3:
        raise ValueError("Worlds must have shape (height, width, channels)")

    difference = np.abs(first.astype(np.float64) - second.astype(np.float64))
    total_cells = int(first.shape[0] * first.shape[1])
    maximum = float(difference.max(initial=0.0))
    per_cell = difference.max(axis=-1)
    divergent = int((per_cell > epsilon).sum())
    worst: int | None = None
    if maximum > 0.0:
        worst = int(np.unravel_index(int(difference.argmax()), difference.shape)[2])
    return WorldDivergence(
        bit_identical=bool(np.array_equal(first, second)),
        max_absolute_error=maximum,
        mean_absolute_error=float(difference.mean()) if difference.size else 0.0,
        divergent_cells=divergent,
        total_cells=total_cells,
        worst_channel=worst,
    )


def device_fingerprint(context: Any) -> dict[str, str]:
    """Identify the GPU and driver a report was produced on."""
    info = getattr(context, "info", {}) or {}
    return {
        "renderer": str(info.get("GL_RENDERER", "unknown")),
        "vendor": str(info.get("GL_VENDOR", "unknown")),
        "version": str(info.get("GL_VERSION", "unknown")),
        "platform": sys.platform,
    }


@dataclass(frozen=True, slots=True)
class DeterminismReport:
    """A reproducible fingerprint of one simulation run."""

    report_version: str
    format_version: str
    device: dict[str, str]
    grid: list[int]
    ticks: int
    tick_rate_hz: int
    seed_digest: str
    final_digest: str
    channel_means: list[float]
    same_device_repeat: dict[str, Any]
    replay_match: dict[str, Any] | None = None
    log_fidelity: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_json(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return destination

    @classmethod
    def read(cls, path: str | Path) -> DeterminismReport:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_mapping(document)

    @classmethod
    def from_mapping(cls, document: Mapping[str, Any]) -> DeterminismReport:
        try:
            version = str(document["report_version"])
            if version != REPORT_VERSION:
                raise ValueError(f"Unsupported report version: {version!r}")
            return cls(
                report_version=version,
                format_version=str(document["format_version"]),
                device=dict(document["device"]),
                grid=[int(value) for value in document["grid"]],
                ticks=int(document["ticks"]),
                tick_rate_hz=int(document["tick_rate_hz"]),
                seed_digest=str(document["seed_digest"]),
                final_digest=str(document["final_digest"]),
                channel_means=[float(value) for value in document["channel_means"]],
                same_device_repeat=dict(document["same_device_repeat"]),
                replay_match=(
                    dict(document["replay_match"])
                    if document.get("replay_match") is not None
                    else None
                ),
                log_fidelity=(
                    dict(document["log_fidelity"])
                    if document.get("log_fidelity") is not None
                    else None
                ),
                notes=[str(note) for note in document.get("notes", [])],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid determinism report: {exc}") from None

    def describe(self) -> str:
        repeat = self.same_device_repeat.get("bit_identical")
        lines = [
            f"Device: {self.device.get('renderer')} "
            f"({self.device.get('vendor')}, {self.device.get('version')})",
            f"World: {self.grid[0]}x{self.grid[1]} at {self.tick_rate_hz} Hz, "
            f"{self.ticks} ticks",
            f"Final digest: {self.final_digest}",
            f"Same-device repeat: {'bit-identical' if repeat else 'DIVERGED'}",
        ]
        if self.replay_match is not None:
            match = self.replay_match.get("bit_identical")
            lines.append(
                f"Command-log replay: {'bit-identical' if match else 'DIVERGED'}"
            )
        if self.log_fidelity is not None:
            fidelity = self.log_fidelity
            scheduled = fidelity.get("operations_scheduled")
            replayed = fidelity.get("records_replayed")
            verdict = "complete" if fidelity.get("complete") else "LOSSY"
            lines.append(
                f"Command-log fidelity: {verdict} "
                f"({replayed}/{scheduled} records, "
                f"{fidelity.get('log_version')})"
            )
            if fidelity.get("lossy_version"):
                lines.append(
                    "  note: this log format predates recorded writer identity, "
                    "so its replay is best-effort"
                )
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Outcome of checking a live run against a recorded report."""

    passed: bool
    same_device: bool
    digest_match: bool
    divergence: WorldDivergence
    tolerance: float
    reference_device: dict[str, str]
    observed_device: dict[str, str]

    def describe(self) -> str:
        scope = "same device" if self.same_device else "different device"
        verdict = "PASS" if self.passed else "FAIL"
        expectation = (
            "bit-identical required"
            if self.same_device
            else f"tolerance {self.tolerance:.1e}"
        )
        return (
            f"{verdict} ({scope}, {expectation})\n"
            f"  reference: {self.reference_device.get('renderer')}\n"
            f"  observed:  {self.observed_device.get('renderer')}\n"
            f"  {self.divergence.describe()}"
        )


def authority_probe_schedule() -> dict[int, list[Any]]:
    """A short schedule that makes log infidelity visible in the world.

    A replay check is only as good as the run it replays. A schedule of plain
    human paints cannot detect a lost writer identity or a lost authority level,
    because every plausible mistake reconstructs the same command. Each entry
    here is chosen so that one specific loss changes the final world:

    - The human barrier mints a lock at ``(100, 100)``.
    - The AI erase over it carries ``user`` authority but AI identity. If the
      writer is lost and re-guessed from authority, it replays as a human erase
      and destroys the lock. Authority and identity disagree on purpose.
    - The AI paint records its own authority in channel 30, so replaying it at
      the default level changes that channel.
    - The temperature delta and velocity impulse are dropped entirely by any
      replay path that filters to segments.
    """
    return {
        2: [
            Segment(
                start_x=100.5,
                start_y=100.5,
                end_x=100.5,
                end_y=100.5,
                radius=3.0,
                category=CATEGORY_IDS["human_barrier"],
                erase=False,
                priority=PRIORITY_LEVELS["user"],
                source=WRITER_HUMAN,
            )
        ],
        4: [
            Segment(
                start_x=100.5,
                start_y=100.5,
                end_x=100.5,
                end_y=100.5,
                radius=3.0,
                category=0,
                erase=True,
                priority=PRIORITY_LEVELS["user"],
                source=WRITER_AI,
            )
        ],
        6: [
            Segment(
                start_x=60.5,
                start_y=60.5,
                end_x=72.5,
                end_y=60.5,
                radius=4.0,
                category=CATEGORY_IDS["active_fluid"],
                erase=False,
                priority=PRIORITY_LEVELS["ai"],
                source=WRITER_AI,
            )
        ],
        8: [
            TemperatureDelta(
                start_x=60.5,
                start_y=60.5,
                end_x=68.5,
                end_y=60.5,
                radius=5.0,
                delta=0.35,
                priority=PRIORITY_LEVELS["ai"],
            )
        ],
        10: [
            VelocityImpulse(
                x=64.5,
                y=64.5,
                velocity_x=0.4,
                velocity_y=-0.55,
                radius=4.0,
                priority=PRIORITY_LEVELS["ai"],
                source=WRITER_AI,
            )
        ],
    }


def _flatten(schedule: Mapping[int, Sequence[Any]]) -> list[tuple[int, Any]]:
    """Tick-stamped operations in the order a log would store them."""
    return [
        (tick, operation)
        for tick in sorted(schedule)
        for operation in schedule[tick]
    ]


@dataclass(frozen=True, slots=True)
class RunOutcome:
    """A finished run, plus what the renderer actually did with the schedule."""

    world: npt.NDArray[np.float32]
    device: dict[str, str]
    applied_writes: int
    unspent_writes: int
    ignored_controls: list[str]


def _simulate(
    grid: npt.NDArray[np.float32],
    *,
    ticks: int,
    schedule: Mapping[int, list[Any]] | None = None,
    recorder: Any | None = None,
) -> RunOutcome:
    """Run the real pipeline headlessly and return the final world.

    When a `recorder` is supplied, every operation is written to a `.bdl` as it
    is scheduled — the same point at which the interactive runtime records — so
    the log is a product of the run rather than a restatement of the schedule.
    """
    from export_video import HeadlessRenderer

    pending = dict(schedule or {})
    with HeadlessRenderer(resolution=64, grid=grid) as renderer:
        device = device_fingerprint(renderer.context)
        for _ in range(ticks):
            tick = renderer.tick + 1
            operations = pending.pop(tick, None)
            if recorder is not None and operations:
                recorder.record(operations, tick=tick)
            renderer.step(operations)
        final = np.array(renderer.read_world(), dtype=DTYPE, copy=True)
        applied = RunOutcome(
            world=final,
            device=device,
            applied_writes=renderer.applied_writes,
            unspent_writes=renderer.pending_writes,
            ignored_controls=list(renderer.ignored_controls),
        )
    if pending:
        raise ValueError(
            f"{sum(len(batch) for batch in pending.values())} scheduled "
            f"operations were never reached in {ticks} ticks"
        )
    return applied


def run_probe(
    *,
    world: str | Path | None = None,
    log: str | Path | None = None,
    ticks: int = DEFAULT_TICKS,
) -> tuple[DeterminismReport, npt.NDArray[np.float32]]:
    """Run the probe twice, plus a log replay when one is supplied."""
    if ticks <= 0:
        raise ValueError("ticks must be positive")
    if world is None:
        seed = create_initial_grid()
    else:
        _header, seed = load_bds(world)
    if seed.ndim != 3 or seed.shape[2] != VECTOR_DIMENSIONS:
        raise ValueError(f"World must have {VECTOR_DIMENSIONS} channels")

    notes: list[str] = []
    if log is not None:
        from export_video import schedule_from_log

        requested = schedule_from_log(log)
        notes.append(f"Replaying the supplied log across {len(requested)} ticks")
    else:
        requested = authority_probe_schedule()
        notes.append(
            "Replaying a recorded authority probe: a human lock, an AI erase "
            "over it whose identity and authority disagree, an AI paint, a "
            "temperature delta, and a velocity impulse"
        )

    # A run shorter than the schedule is legitimate, but the part that falls
    # outside the window must be excluded rather than quietly never applied, or
    # the fidelity check would compare against operations no run ever saw.
    schedule = {
        tick: list(batch) for tick, batch in requested.items() if 1 <= tick <= ticks
    }
    beyond = sum(
        len(batch) for tick, batch in requested.items() if not 1 <= tick <= ticks
    )
    if beyond:
        notes.append(
            f"{beyond} scheduled operations fall beyond tick {ticks} and are "
            "outside this run"
        )

    with tempfile.TemporaryDirectory(prefix="nwr-determinism-") as scratch:
        recorder = SessionRecorder(
            scratch,
            grid_width=int(seed.shape[1]),
            grid_height=int(seed.shape[0]),
            name="probe",
            metadata={"purpose": "determinism replay leg"},
        )
        try:
            live = _simulate(
                seed,
                ticks=ticks,
                schedule=schedule,
                recorder=recorder,
            )
        finally:
            recorder.close()

        first = live.world
        device = live.device
        repeat = compare_worlds(first, _simulate(seed, ticks=ticks, schedule=schedule).world)

        # The replay leg reads back the log the live run just wrote. Replaying
        # the in-memory schedule instead would compare a run against itself and
        # could never fail, whatever the log format lost.
        replayed_log = read_log(recorder.log_path)
        replay = _simulate(
            seed,
            ticks=ticks,
            schedule=schedule_from_records(replayed_log),
        )
        replay_match = compare_worlds(first, replay.world).to_json()
        fidelity = _log_fidelity(
            schedule,
            replayed_log,
            live=live,
            replay=replay,
            beyond_window=beyond,
        )

    report = DeterminismReport(
        report_version=REPORT_VERSION,
        format_version=FORMAT_VERSION,
        device=device,
        grid=[int(seed.shape[1]), int(seed.shape[0])],
        ticks=int(ticks),
        tick_rate_hz=TICK_RATE_HZ,
        seed_digest=world_digest(seed),
        final_digest=world_digest(first),
        channel_means=[
            round(float(first[..., channel].mean()), 9)
            for channel in range(VECTOR_DIMENSIONS)
        ],
        same_device_repeat=repeat.to_json(),
        replay_match=replay_match,
        log_fidelity=fidelity,
        notes=notes,
    )
    return report, first


def schedule_from_records(log: HistoryLog) -> dict[int, list[Any]]:
    """Group a parsed log into the tick-indexed schedule the renderer takes."""
    schedule: dict[int, list[Any]] = {}
    for tick, operations in replay_operations(log):
        schedule.setdefault(tick, []).extend(operations)
    return schedule


def _log_fidelity(
    scheduled: Mapping[int, Sequence[Any]],
    log: HistoryLog,
    *,
    live: RunOutcome,
    replay: RunOutcome,
    beyond_window: int = 0,
) -> dict[str, Any]:
    """Check the log is complete and that every record actually reached the GPU.

    Three things have to hold, and a world comparison alone proves none of them:

    - The operations read back match the ones that went in, field for field. A
      lost field that happens not to matter to this run is still lost.
    - Both runs applied as many writes as were scheduled. This is the check the
      world comparison structurally cannot make: both legs go through the same
      renderer, so a write dropped there is dropped on both sides and cancels
      out of the comparison entirely.
    - Nothing was left queued or quietly ignored at the end of the run.
    """
    expected = _flatten(scheduled)
    actual = [(entry.tick, entry.to_operation()) for entry in log.operations]
    mismatches = [
        {
            "index": index,
            "expected": repr(want),
            "actual": repr(got) if got is not None else None,
        }
        for index, (want, got) in enumerate(
            zip_longest(expected, actual, fillvalue=None)
        )
        if not _operations_agree(want, got)
    ]
    writes = sum(1 for _tick, operation in expected if not isinstance(operation, Control))
    fully_applied = (
        live.applied_writes == writes
        and replay.applied_writes == writes
        and not live.unspent_writes
        and not replay.unspent_writes
    )
    return {
        "log_version": str(log.header.get("format_version")),
        "lossy_version": is_lossy_version(log.header),
        "operations_scheduled": len(expected),
        "operations_beyond_window": beyond_window,
        "records_replayed": len(actual),
        "writes_scheduled": writes,
        "writes_applied_live": live.applied_writes,
        "writes_applied_on_replay": replay.applied_writes,
        "writes_left_queued": live.unspent_writes + replay.unspent_writes,
        "controls_ignored_by_replay": sorted(set(replay.ignored_controls)),
        "records_match": not mismatches,
        "complete": not mismatches and fully_applied,
        # Bounded on purpose: a diverging log tends to diverge everywhere, and a
        # report is meant to be read.
        "mismatches": mismatches[:8],
        "truncated_bytes": log.truncated_bytes,
    }


def _operations_agree(
    want: tuple[int, Any] | None,
    got: tuple[int, Any] | None,
) -> bool:
    """Whether a scheduled operation and its replayed form are the same edit.

    Compared through float32, because that is the width the record stores and
    the GPU consumes. Holding the log to float64 equality would fail on a
    faithful round trip.
    """
    if want is None or got is None:
        return False
    if want[0] != got[0] or type(want[1]) is not type(got[1]):
        return False
    for name in want[1].__slots__:
        left = getattr(want[1], name)
        right = getattr(got[1], name)
        if isinstance(left, float) or isinstance(right, float):
            if np.float32(left) != np.float32(right):
                return False
        elif left != right:
            return False
    return True


def verify_against(
    reference: DeterminismReport,
    observed: DeterminismReport,
    final_world: npt.NDArray[np.float32],
    reference_world: npt.NDArray[np.float32] | None = None,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> VerificationResult:
    """Check a live run against a recorded report.

    On the same device the standard is bit-identity. On a different device the
    standard is the supplied tolerance, compared through the recorded per-channel
    means when the reference world itself is not available.
    """
    same_device = (
        reference.device.get("renderer") == observed.device.get("renderer")
        and reference.device.get("version") == observed.device.get("version")
    )
    digest_match = reference.final_digest == observed.final_digest

    if reference_world is not None:
        divergence = compare_worlds(reference_world, final_world)
    else:
        # Without the reference payload, compare the recorded channel means.
        expected = np.asarray(reference.channel_means, dtype=np.float64)
        actual = np.asarray(observed.channel_means, dtype=np.float64)
        if expected.shape != actual.shape:
            raise ValueError("Reports describe different channel counts")
        difference = np.abs(expected - actual)
        divergence = WorldDivergence(
            bit_identical=digest_match,
            max_absolute_error=float(difference.max(initial=0.0)),
            mean_absolute_error=float(difference.mean()) if difference.size else 0.0,
            divergent_cells=int((difference > 0.0).sum()),
            total_cells=int(difference.size),
            worst_channel=(
                int(difference.argmax()) if float(difference.max(initial=0.0)) > 0.0
                else None
            ),
        )

    passed = digest_match if same_device else divergence.within(tolerance)
    return VerificationResult(
        passed=passed,
        same_device=same_device,
        digest_match=digest_match,
        divergence=divergence,
        tolerance=tolerance,
        reference_device=dict(reference.device),
        observed_device=dict(observed.device),
    )


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record or verify a simulation determinism report.",
    )
    parser.add_argument("--world", type=Path, help="Seed .bds (default: seed world)")
    parser.add_argument("--log", type=Path, help="Optional .bdl log to replay")
    parser.add_argument("--ticks", type=int, default=DEFAULT_TICKS)
    parser.add_argument(
        "--record",
        type=Path,
        help=f"Write a report to this path (suggested: {REPORTS / 'probe.json'})",
    )
    parser.add_argument("--verify", type=Path, help="Check against this report")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    if arguments.record is None and arguments.verify is None:
        print("Nothing to do: pass --record and/or --verify")
        return 2
    try:
        report, final = run_probe(
            world=arguments.world,
            log=arguments.log,
            ticks=arguments.ticks,
        )
    except Exception as exc:
        print(f"determinism error: {exc}", file=sys.stderr)
        return 1

    print(report.describe())
    if not report.same_device_repeat.get("bit_identical"):
        print(
            "\nSame-device repeat diverged. This is a real bug, not hardware "
            "drift; the pipeline is not reproducible on its own GPU."
        )
        return 1

    exit_code = 0
    if arguments.record is not None:
        written = report.write(arguments.record)
        print(f"\nWrote {written}")
    if arguments.verify is not None:
        try:
            reference = DeterminismReport.read(arguments.verify)
        except (OSError, ValueError) as exc:
            print(f"determinism error: {exc}", file=sys.stderr)
            return 1
        result = verify_against(
            reference,
            report,
            final,
            tolerance=arguments.tolerance,
        )
        print("\n" + result.describe())
        if not result.passed:
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_TICKS",
    "DEFAULT_TOLERANCE",
    "DeterminismReport",
    "REPORT_VERSION",
    "VerificationResult",
    "WorldDivergence",
    "compare_worlds",
    "device_fingerprint",
    "main",
    "run_probe",
    "verify_against",
    "world_digest",
]
