"""ModernGL application for the Neural World Runtime video playground."""

from __future__ import annotations

import argparse
import io
import math
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import moderngl
import moderngl_window as mglw
import numpy as np

from ai_bridge import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    ControlBridge,
    encode_png,
    generate_token,
    summarize_world,
)
from ai_commands import (
    OPERATOR_CONTROL_ACTIONS,
    WRITER_AI,
    WRITER_HUMAN,
    Control,
    Operation,
    RemoveEntity,
    Segment,
    SpawnEntity,
    TemperatureDelta,
    VelocityImpulse,
)
from ai_command_compiler import compile_ai_json
from ai_world import (
    export_ai_context,
    generate_ai_summary,
    inspect_region,
)
from entities import EntityError, EntityRegistry
from bds_format import (
    BDSFormatError,
    DTYPE,
    GRID_HEIGHT,
    GRID_WIDTH,
    HUMAN_LOCK_CHANNEL,
    PRIORITY_LEVELS,
    TICK_RATE_HZ,
    VECTOR_DIMENSIONS,
    create_initial_grid,
    load_bds,
    save_bds,
)
from bds_chunks import DEFAULT_CHUNK_SIZE, ChunkedWorld
from bds_codec import encode_anchor_residual
from bds_history import (
    LoggedOperation,
    OperationKind,
    SessionRecorder,
    operation_to_logged,
)
from net_relay import DEFAULT_HOST as RELAY_HOST
from net_relay import DEFAULT_PORT as RELAY_PORT
from net_relay import OperationRelay
from material_network import load_material_weights
from shader_library import (
    COMPUTE_PASSES,
    WORKGROUP_SIZE,
    load_compute_passes,
    load_shader,
    normalize_priority,
)
from swarm_agent import AutonomousSwarmAgent
from paths import DEFAULT_WORLD, ROOT

WORLD_PATH: Final = DEFAULT_WORLD
MATERIAL_WEIGHTS_PATH: Final = ROOT / "material_weights.npy"
FIXED_STEP: Final = 1.0 / TICK_RATE_HZ
MAX_STEPS_PER_FRAME: Final = 5
MAX_COMMANDS_PER_TICK: Final = 64
MAX_QUEUED_PAINT_COMMANDS: Final = 65536
COMMAND_FLOATS: Final = 8
LEFT_BUTTON: Final = 1
RIGHT_BUTTON: Final = 2
SWARM_PERCEPTION_INTERVAL: Final = 3
HUD_REFRESH_INTERVAL: Final = 0.25
PREVIEW_RESOLUTION: Final = 1024


@dataclass(frozen=True, slots=True)
class PaintCommand:
    """A deterministic grid-space command scheduled for a simulation tick."""

    tick: int
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    radius: float
    category: int
    operation: float
    priority: int = PRIORITY_LEVELS["user"]
    source: int = 0  # 0 = human, 1 = AI
    temperature_delta: float | None = None
    # When set, segment.zw carries (V_x, V_y) and operation becomes ±4.
    velocity_impulse: tuple[float, float] | None = None

    def as_row(self) -> tuple[float, ...]:
        # ±1 = human paint/erase, ±2 = AI paint/erase,
        # ±3 = temperature delta, ±4 = velocity impulse.
        if self.velocity_impulse is not None:
            vx, vy = self.velocity_impulse
            signed = -4.0 if self.source == 1 else 4.0
            return (
                self.start_x,
                self.start_y,
                float(vx),
                float(vy),
                self.radius,
                0.0,
                signed,
                normalize_priority(self.priority),
            )
        if self.temperature_delta is not None:
            signed = 3.0 if self.temperature_delta >= 0.0 else -3.0
            return (
                self.start_x,
                self.start_y,
                self.end_x,
                self.end_y,
                self.radius,
                abs(float(self.temperature_delta)),
                signed,
                normalize_priority(self.priority),
            )
        magnitude = 2.0 if self.source == 1 else 1.0
        signed = -magnitude if self.operation < 0.0 else magnitude
        return (
            self.start_x,
            self.start_y,
            self.end_x,
            self.end_y,
            self.radius,
            float(self.category),
            signed,
            normalize_priority(self.priority),
        )


def _resolve_ai_authority(value: object) -> int:
    """Map a `--ai-authority` name onto a priority level."""
    name = str(value or "ai")
    if name not in PRIORITY_LEVELS:
        raise SystemExit(
            f"Unknown --ai-authority '{name}'; choose one of "
            f"{sorted(PRIORITY_LEVELS)}"
        )
    return PRIORITY_LEVELS[name]


class VideoPlayground(mglw.WindowConfig):
    """GPU-resident 32-channel field runtime; base class for every application."""

    gl_version = (4, 3)
    title = "Neural World Runtime — Field Sandbox"
    window_size = (1024, 1024)
    aspect_ratio = 1.0
    resizable = True
    vsync = True

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--ai-server",
            action="store_true",
            default=_env_flag("NWR_AI_SERVER"),
            help="Serve the AI control bridge on loopback HTTP",
        )
        parser.add_argument(
            "--ai-host",
            default=os.environ.get("NWR_AI_HOST", DEFAULT_HOST),
            help=f"Bridge bind address (default {DEFAULT_HOST})",
        )
        parser.add_argument(
            "--ai-port",
            type=int,
            default=int(os.environ.get("NWR_AI_PORT", DEFAULT_PORT)),
            help=f"Bridge port, or 0 to pick a free one (default {DEFAULT_PORT})",
        )
        parser.add_argument(
            "--ai-token",
            default=os.environ.get("NWR_AI_TOKEN", ""),
            help="Bridge access token; generated when omitted",
        )
        parser.add_argument(
            "--ai-authority",
            choices=sorted(PRIORITY_LEVELS),
            default=os.environ.get("NWR_AI_AUTHORITY", "ai"),
            help=(
                "Write authority granted to bridge and relay callers "
                "(default ai). 'user' also hands them the world file and "
                "lock minting, so give it only to a caller you trust as much "
                "as your own hands."
            ),
        )
        parser.add_argument(
            "--allow-remote-bind",
            action="store_true",
            default=_env_flag("NWR_ALLOW_REMOTE_BIND"),
            help=(
                "Permit binding the bridge and relay to a non-loopback address. "
                "Without this they refuse any address reachable from the network."
            ),
        )
        parser.add_argument(
            "--neural-material",
            action="store_true",
            default=_env_flag("NWR_NEURAL_MATERIAL"),
            help="Shade with the trained material network when weights exist",
        )
        parser.add_argument(
            "--record",
            type=Path,
            default=None,
            help="Record a replayable session into this directory",
        )
        parser.add_argument(
            "--world",
            type=Path,
            default=WORLD_PATH,
            help="World snapshot to load and save (default: output/worlds/playground/world.bds)",
        )
        parser.add_argument(
            "--world-width",
            type=int,
            default=int(os.environ.get("NWR_WORLD_WIDTH", GRID_WIDTH)),
            help="World width in cells",
        )
        parser.add_argument(
            "--world-height",
            type=int,
            default=int(os.environ.get("NWR_WORLD_HEIGHT", GRID_HEIGHT)),
            help="World height in cells",
        )
        parser.add_argument(
            "--chunk-size",
            type=int,
            default=DEFAULT_CHUNK_SIZE,
            help="Chunk edge length used for paging and storage reports",
        )
        parser.add_argument(
            "--relay",
            action="store_true",
            default=_env_flag("NWR_RELAY"),
            help="Host an operation relay so remote peers can share this world",
        )
        parser.add_argument(
            "--relay-host",
            default=os.environ.get("NWR_RELAY_HOST", RELAY_HOST),
            help="Relay bind address",
        )
        parser.add_argument(
            "--relay-port",
            type=int,
            default=int(os.environ.get("NWR_RELAY_PORT", RELAY_PORT)),
            help="Relay bind port",
        )
        parser.add_argument(
            "--relay-token",
            default=os.environ.get("NWR_RELAY_TOKEN", ""),
            help="Relay access token; generated when omitted",
        )

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)

        self.grid_width = int(getattr(self.argv, "world_width", GRID_WIDTH))
        self.grid_height = int(getattr(self.argv, "world_height", GRID_HEIGHT))
        if self.grid_width <= 0 or self.grid_height <= 0:
            raise ValueError("World dimensions must be positive")
        self.bounds = (self.grid_width, self.grid_height)
        self.world_path = Path(getattr(self.argv, "world", WORLD_PATH))

        self.compute_passes = {
            name: self.ctx.compute_shader(source)
            for name, source in load_compute_passes().items()
        }
        self.render_program = self.ctx.program(
            vertex_shader=load_shader("fullscreen.vert"),
            fragment_shader=load_shader("world.frag"),
        )
        self.screen_triangle = self.ctx.vertex_array(self.render_program, [])
        self._scene_texture: moderngl.Texture | None = None
        self._scene_framebuffer: moderngl.Framebuffer | None = None
        self._preview_texture: moderngl.Texture | None = None
        self._preview_framebuffer: moderngl.Framebuffer | None = None
        self._material_texture: moderngl.Texture | None = None
        self.use_neural_material = bool(
            getattr(self.argv, "neural_material", False)
        )

        self.command_buffer = self.ctx.buffer(
            reserve=MAX_COMMANDS_PER_TICK * COMMAND_FLOATS * np.dtype("f4").itemsize
        )
        self.command_buffer.bind_to_storage_buffer(2)
        self.world_buffers: list[moderngl.Buffer] = []
        self.current_buffer = 0
        startup_grid = self._load_startup_world()
        self._replace_world(startup_grid)

        self.tick = 0
        # Loading and resetting rewind the world tick; the session tick never
        # does, so the history log stays monotonically ordered.
        self.session_tick = 0
        self.accumulator = 0.0
        self.paused = False
        self.selected_category = 2
        self.brush_radius = 2.5
        self._commands: deque[PaintCommand] = deque()
        self.entities = EntityRegistry(bounds=self.bounds)
        self._active_button: int | None = None
        self._last_grid_position: tuple[float, float] | None = None
        self._cursor_grid_position = (
            self.grid_width / 2.0,
            self.grid_height / 2.0,
        )
        self._shift_down = False
        self.recorder: SessionRecorder | None = None
        self._relay_sequence = 0
        self.chunk_size = max(
            int(getattr(self.argv, "chunk_size", DEFAULT_CHUNK_SIZE)), 1
        )
        self._start_recording(startup_grid)

        for program in self.compute_passes.values():
            program["grid_size"].value = self.bounds
        self.render_program["grid_size"].value = self.bounds
        self._update_viewport_uniform()
        self._configure_material()

        self.swarm = AutonomousSwarmAgent()
        self.swarm.start()
        self._frames_since_perception = 0
        self._hud_lock_cells = 0
        self._fps = 0.0
        self._fps_accum = 0.0
        self._fps_frames = 0
        self._hud_age = HUD_REFRESH_INTERVAL
        self._hud_texture: moderngl.Texture | None = None
        self._hud_program = self.ctx.program(
            vertex_shader=load_shader("fullscreen.vert"),
            fragment_shader=load_shader("hud.frag"),
        )
        self._hud_vao = self.ctx.vertex_array(self._hud_program, [])
        self._ensure_hud_texture()
        # Seed HUD lock count and give the planner an initial view.
        self.swarm.publish_grid(startup_grid, tick=0)
        self._hud_lock_cells = int((startup_grid[..., 31] >= 0.5).sum())

        print(self._control_help())

        self._ai_authority = _resolve_ai_authority(
            getattr(self.argv, "ai_authority", "ai")
        )
        self.bridge: ControlBridge | None = None
        if getattr(self.argv, "ai_server", False):
            self._start_bridge()
        self.relay: OperationRelay | None = None
        if getattr(self.argv, "relay", False):
            self._start_relay()

    @staticmethod
    def _control_help() -> str:
        return (
            "Controls: left-drag paint, right-drag/Shift+left erase, "
            "1 locked barrier, 2 fluid, 3 solid wall, N material network, "
            "A swarm AI, D field vortex, C clear AI fluid, "
            "E spawn blob, X remove entities, "
            "Space pause, S save, L load, R reset"
        )

    def _start_bridge(self) -> None:
        token = str(getattr(self.argv, "ai_token", "") or "") or generate_token()
        bridge = ControlBridge(
            status_provider=self._runtime_status,
            state_provider=self._observation,
            screenshot_provider=self._screenshot,
            context_provider=self._ai_context,
            inspect_provider=self._inspect,
            preview_provider=self._preview,
            token=token,
            host=str(getattr(self.argv, "ai_host", DEFAULT_HOST)),
            port=int(getattr(self.argv, "ai_port", DEFAULT_PORT)),
            bounds=self.bounds,
            authority=self._ai_authority,
            allow_remote_bind=bool(getattr(self.argv, "allow_remote_bind", False)),
        )
        try:
            base_url = bridge.start()
        except OSError as exc:
            print(f"AI bridge could not start: {exc}")
            return
        self.bridge = bridge
        print(f"AI bridge listening on {base_url}")
        print(f"AI bridge token: {token}")

    def _start_relay(self) -> None:
        token = str(getattr(self.argv, "relay_token", "") or "") or generate_token()
        relay = OperationRelay(
            token=token,
            host=str(getattr(self.argv, "relay_host", RELAY_HOST)),
            port=int(getattr(self.argv, "relay_port", RELAY_PORT)),
            bounds=self.bounds,
            tick_provider=lambda: self.tick + 1,
            allow_remote_bind=bool(getattr(self.argv, "allow_remote_bind", False)),
        )
        try:
            host, port = relay.start()
        except OSError as exc:
            print(f"Operation relay could not start: {exc}")
            return
        self.relay = relay
        self._relay_sequence = 0
        print(f"Operation relay listening on {host}:{port}")
        print(f"Operation relay token: {token}")

    def _drain_relay(self) -> None:
        """Apply operations other peers published since the last frame."""
        if self.relay is None:
            return
        history = self.relay.history()
        if len(history) <= self._relay_sequence:
            return
        for batch in history[self._relay_sequence :]:
            self._apply_operations(list(batch.operations))
        self._relay_sequence = len(history)

    def _start_recording(self, grid: np.ndarray) -> None:
        directory = getattr(self.argv, "record", None)
        if directory is None:
            return
        buffer = io.BytesIO()
        snapshot_path = Path(directory) / "session.bds"
        try:
            save_bds(snapshot_path, grid, metadata={"simulation_tick": 0})
            buffer.write(snapshot_path.read_bytes())
            self.recorder = SessionRecorder(
                directory,
                grid_width=self.grid_width,
                grid_height=self.grid_height,
                snapshot_bytes=buffer.getvalue(),
                metadata={"window_size": list(self.window_size)},
            )
        except (OSError, ValueError) as exc:
            print(f"Recording disabled: {exc}")
            return
        print(f"Recording session to {self.recorder.log_path}")

    def _record(self, command: PaintCommand, tick: int) -> None:
        """Log a command exactly as the GPU will receive it.

        A `PaintCommand` is one of three shapes, and the log has to keep them
        apart: recording a temperature delta or a velocity impulse as a paint
        would replay it as a material rewrite, which is a different edit.
        """
        if self.recorder is None:
            return
        writer = WRITER_AI if command.source == 1 else WRITER_HUMAN
        if command.velocity_impulse is not None:
            impulse_x, impulse_y = command.velocity_impulse
            entry = LoggedOperation(
                tick=tick,
                kind=OperationKind.VELOCITY,
                priority=command.priority,
                source=writer,
                start_x=command.start_x,
                start_y=command.start_y,
                end_x=float(impulse_x),
                end_y=float(impulse_y),
                radius=command.radius,
            )
        elif command.temperature_delta is not None:
            entry = LoggedOperation(
                tick=tick,
                kind=OperationKind.TEMPERATURE,
                priority=command.priority,
                source=writer,
                start_x=command.start_x,
                start_y=command.start_y,
                end_x=command.end_x,
                end_y=command.end_y,
                radius=command.radius,
                delta=float(command.temperature_delta),
            )
        else:
            entry = LoggedOperation(
                tick=tick,
                kind=(
                    OperationKind.ERASE
                    if command.operation < 0.0
                    else OperationKind.PAINT
                ),
                category=command.category,
                priority=command.priority,
                source=writer,
                start_x=command.start_x,
                start_y=command.start_y,
                end_x=command.end_x,
                end_y=command.end_y,
                radius=command.radius,
            )
        self.recorder.writer.append([entry])

    def _record_control(self, action: str) -> None:
        if self.recorder is None:
            return
        self.recorder.writer.append(
            [operation_to_logged(Control(action=action), tick=self.session_tick + 1)]
        )

    def _runtime_status(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "tick": self.tick,
            "paused": self.paused,
            "tick_rate_hz": TICK_RATE_HZ,
            "grid": {"width": self.grid_width, "height": self.grid_height},
            "pending_paint_commands": len(self._commands),
            "entities": self.entities.describe(),
            "material": (
                "neural"
                if self.render_program["use_neural_material"].value
                else "procedural"
            ),
        }

    def _observation(self) -> dict[str, Any]:
        payload = self._runtime_status()
        grid = self._read_world()
        payload["world"] = summarize_world(grid)
        payload["summary"] = generate_ai_summary(grid, tick=self.tick)
        payload["context"] = export_ai_context(
            grid,
            tick=self.tick,
            world_name=self.world_path.stem,
            entities=self.entities.describe(),
            authority=self._ai_authority,
        )
        payload["storage"] = self._storage_report(grid)
        if self.relay is not None:
            payload["relay"] = {
                "address": "{}:{}".format(*self.relay.address),
                "clients": self.relay.client_count(),
                "sequence": self.relay.sequence,
            }
        return payload

    def _inspect(self, x: float, y: float, radius: float) -> dict[str, Any]:
        return inspect_region(self._read_world(), x, y, radius)

    def _ai_context(self) -> dict[str, Any]:
        return export_ai_context(
            self._read_world(),
            tick=self.tick,
            world_name=self.world_path.stem,
            entities=self.entities.describe(),
            authority=self._ai_authority,
        )

    def _preview(self) -> bytes:
        """Render the live world at exactly 1024x1024, whatever the window size."""
        framebuffer = self._ensure_preview_framebuffer()
        framebuffer.use()
        framebuffer.clear(0.002, 0.004, 0.012, 1.0)
        self.world_buffers[self.current_buffer].bind_to_storage_buffer(0)
        self.render_program["viewport_size"].value = (
            float(PREVIEW_RESOLUTION),
            float(PREVIEW_RESOLUTION),
        )
        self.screen_triangle.render(mode=moderngl.TRIANGLES, vertices=3)
        self.ctx.finish()
        payload = framebuffer.read(components=3)
        self._update_viewport_uniform()
        self.wnd.use()
        return encode_png(payload, PREVIEW_RESOLUTION, PREVIEW_RESOLUTION, 3)

    def _storage_report(self, grid: np.ndarray) -> dict[str, Any]:
        """Measure what this world costs to page out losslessly and to archive."""
        chunks = ChunkedWorld(
            self.grid_width,
            self.grid_height,
            chunk_size=self.chunk_size,
            grid=grid,
        )
        for chunk in list(chunks.chunks()):
            chunks.evict(*chunk.index)
        paged = chunks.memory_usage()
        _encoded, archive = encode_anchor_residual(grid)
        return {
            "chunks": {
                "total": chunks.chunk_count,
                "chunk_size": chunks.chunk_size,
                "columns": chunks.columns,
                "rows": chunks.rows,
            },
            "paged_lossless": {
                "compressed_bytes": paged["compressed_bytes"],
                "uncompressed_bytes": paged["uncompressed_equivalent"],
                "ratio": round(
                    paged["uncompressed_equivalent"]
                    / max(paged["compressed_bytes"], 1),
                    1,
                ),
            },
            "archive_lossy": {
                "encoded_bytes": archive.encoded_bytes,
                "ratio": round(archive.ratio, 1),
                "max_absolute_error": archive.max_absolute_error,
            },
        }

    def _screenshot(self) -> bytes:
        framebuffer = self._ensure_scene_framebuffer()
        width, height = framebuffer.size
        return encode_png(framebuffer.read(components=3), width, height, 3)

    def _read_world(self) -> np.ndarray:
        self.ctx.finish()
        payload = self.world_buffers[self.current_buffer].read()
        return np.frombuffer(payload, dtype="<f4").reshape(
            (self.grid_height, self.grid_width, VECTOR_DIMENSIONS)
        )

    def _apply_operations(self, operations: list[Operation | TemperatureDelta]) -> None:
        for operation in operations:
            if isinstance(operation, Segment):
                self._enqueue(
                    PaintCommand(
                        tick=self.tick + 1,
                        start_x=operation.start_x,
                        start_y=operation.start_y,
                        end_x=operation.end_x,
                        end_y=operation.end_y,
                        radius=operation.radius,
                        category=operation.category,
                        operation=-1.0 if operation.erase else 1.0,
                        priority=operation.priority,
                        source=1,
                    )
                )
            elif isinstance(operation, TemperatureDelta):
                self._enqueue(
                    PaintCommand(
                        tick=self.tick + 1,
                        start_x=operation.start_x,
                        start_y=operation.start_y,
                        end_x=operation.end_x,
                        end_y=operation.end_y,
                        radius=operation.radius,
                        category=0,
                        operation=1.0 if operation.delta >= 0.0 else -1.0,
                        priority=operation.priority,
                        source=1,
                        temperature_delta=operation.delta,
                    )
                )
            elif isinstance(operation, VelocityImpulse):
                self._enqueue(
                    PaintCommand(
                        tick=self.tick + 1,
                        start_x=operation.x,
                        start_y=operation.y,
                        end_x=operation.velocity_x,
                        end_y=operation.velocity_y,
                        radius=operation.radius,
                        category=0,
                        operation=1.0,
                        priority=operation.priority,
                        source=1,
                        velocity_impulse=(
                            operation.velocity_x,
                            operation.velocity_y,
                        ),
                    )
                )
            elif isinstance(operation, SpawnEntity):
                self._spawn_entity(operation)
            elif isinstance(operation, RemoveEntity):
                self._remove_entity(operation.entity_id)
            elif isinstance(operation, Control):
                # Everything arriving here came off the bridge or the relay, so
                # it carries the caller's authority, never the operator's.
                self._run_control(operation.action, authority=self._ai_authority)

    def _spawn_entity(self, request: SpawnEntity) -> None:
        try:
            entity, operations = self.entities.spawn(
                request.kind,
                (request.x, request.y),
                tick=self.tick + 1,
                radius=request.radius,
                priority=request.priority,
            )
        except EntityError as exc:
            print(f"Spawn rejected: {exc}")
            return
        self._apply_operations(operations)
        print(f"Spawned {entity.entity_id} at ({entity.x:.1f}, {entity.y:.1f})")

    def _remove_entity(self, entity_id: str) -> None:
        try:
            operations = self.entities.remove(entity_id)
        except EntityError as exc:
            print(f"Remove rejected: {exc}")
            return
        self._apply_operations(operations)
        print(f"Removed {entity_id}")

    def _advance_entities(self, grid: np.ndarray) -> None:
        """Step tracked entities against the snapshot we already read back."""
        if not len(self.entities):
            return
        try:
            operations = self.entities.advance(grid, tick=self.tick + 1)
        except EntityError as exc:
            print(f"Entity step failed: {exc}")
            return
        self._apply_operations(operations)

    def _drain_swarm(self) -> None:
        """Pull background-planned AI packets into the deterministic queue."""
        for command in self.swarm.drain_commands(limit=MAX_COMMANDS_PER_TICK):
            self._enqueue(
                PaintCommand(
                    tick=self.tick + 1,
                    start_x=command.start_x,
                    start_y=command.start_y,
                    end_x=command.end_x,
                    end_y=command.end_y,
                    radius=command.radius,
                    category=command.category,
                    operation=command.operation,
                    priority=command.priority,
                    source=1,
                )
            )

    def _publish_swarm_perception(self, frame_time: float) -> None:
        """Amortize one readback across the planner, entities, and the HUD."""
        self._frames_since_perception += 1
        needs = (
            self.swarm.active
            or len(self.entities)
            or self._frames_since_perception >= SWARM_PERCEPTION_INTERVAL * 4
        )
        if not needs or self._frames_since_perception < SWARM_PERCEPTION_INTERVAL:
            return
        self._frames_since_perception = 0
        grid = self._read_world()
        summary = self.swarm.publish_grid(
            grid,
            tick=self.tick,
            frame_time=frame_time,
        )
        self._hud_lock_cells = summary.human_lock_cells
        self._advance_entities(grid)

    def _run_control(
        self,
        action: str,
        *,
        authority: int = PRIORITY_LEVELS["user"],
    ) -> None:
        """Run a control action on behalf of a caller with `authority`.

        The command compiler already refuses file actions below human authority.
        This second check is here because the control plane is the one path that
        can replace a locked world without touching the write path, so it should
        not depend on a single validator staying correct.
        """
        below_human = authority < PRIORITY_LEVELS["user"]
        if below_human and action in OPERATOR_CONTROL_ACTIONS:
            print(f"Refused '{action}': reserved for human authority")
            return
        self._record_control(action)
        if action == "reset":
            self._reset_world(preserve_locks=below_human)
        elif action == "save":
            self._save_world()
        elif action == "load":
            self._load_world()
        elif action == "pause":
            self.paused = True
            print("Paused")
        elif action == "resume":
            self.paused = False
            print("Running")

    def _enqueue(self, command: PaintCommand) -> None:
        if len(self._commands) >= MAX_QUEUED_PAINT_COMMANDS:
            self._commands.popleft()
        self._commands.append(command)
        self._record(command, self.session_tick + 1)

    def _new_world(self) -> np.ndarray:
        if self.bounds == (GRID_WIDTH, GRID_HEIGHT):
            return create_initial_grid()
        return create_initial_grid(self.grid_width, self.grid_height)

    def _load_startup_world(self) -> np.ndarray:
        if not self.world_path.exists():
            return self._new_world()
        try:
            _header, grid = load_bds(self.world_path)
            self._require_runtime_shape(grid)
            print(f"Loaded {self.world_path}")
            return grid
        except (OSError, BDSFormatError, ValueError) as exc:
            print(f"Could not load {self.world_path}: {exc}; using a new world")
            return self._new_world()

    def _require_runtime_shape(self, grid: np.ndarray) -> None:
        expected = (self.grid_height, self.grid_width, VECTOR_DIMENSIONS)
        if grid.shape != expected:
            raise ValueError(f"Runtime requires grid shape {expected}, got {grid.shape}")

    def _replace_world(self, grid: np.ndarray) -> None:
        self._require_runtime_shape(grid)
        payload = np.ascontiguousarray(grid, dtype="<f4").tobytes(order="C")
        new_buffers = [self.ctx.buffer(payload), self.ctx.buffer(payload)]
        old_buffers = self.world_buffers
        self.world_buffers = new_buffers
        self.current_buffer = 0
        for buffer in old_buffers:
            buffer.release()

    def _scene_size(self) -> tuple[int, int]:
        return (max(int(self.wnd.buffer_width), 1), max(int(self.wnd.buffer_height), 1))

    def _ensure_scene_framebuffer(self) -> moderngl.Framebuffer:
        """Keep an offscreen colour target so frames survive buffer swaps."""
        size = self._scene_size()
        if self._scene_framebuffer is not None and self._scene_framebuffer.size == size:
            return self._scene_framebuffer
        if self._scene_framebuffer is not None:
            self._scene_framebuffer.release()
        if self._scene_texture is not None:
            self._scene_texture.release()
        self._scene_texture = self.ctx.texture(size, 4)
        self._scene_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._scene_framebuffer = self.ctx.framebuffer(
            color_attachments=[self._scene_texture]
        )
        return self._scene_framebuffer

    def _ensure_preview_framebuffer(self) -> moderngl.Framebuffer:
        """A fixed-size target so `/preview` is resolution-stable for callers."""
        if self._preview_framebuffer is not None:
            return self._preview_framebuffer
        self._preview_texture = self.ctx.texture(
            (PREVIEW_RESOLUTION, PREVIEW_RESOLUTION), 4
        )
        self._preview_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._preview_framebuffer = self.ctx.framebuffer(
            color_attachments=[self._preview_texture]
        )
        return self._preview_framebuffer

    def _configure_material(self) -> None:
        """Bind the trained material network when its weights are available."""
        rows = 0
        if MATERIAL_WEIGHTS_PATH.exists():
            try:
                weights = load_material_weights(MATERIAL_WEIGHTS_PATH)
            except (OSError, ValueError) as exc:
                print(f"Material network ignored: {exc}")
            else:
                rows = weights.shape[0]
                self._material_texture = self.ctx.texture(
                    (weights.shape[1], weights.shape[0]),
                    1,
                    weights.tobytes(order="C"),
                    dtype="f4",
                )
                self._material_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
                self._material_texture.use(location=0)
                self.render_program["material_weights"].value = 0
                print(f"Material network loaded from {MATERIAL_WEIGHTS_PATH.name}")
        self.render_program["weight_rows"].value = rows
        self.render_program["use_neural_material"].value = (
            1 if rows and self.use_neural_material else 0
        )

    def _update_viewport_uniform(self) -> None:
        width, height = self._scene_size()
        self.render_program["viewport_size"].value = (float(width), float(height))

    def _commands_for_tick(self, next_tick: int) -> list[PaintCommand]:
        selected: list[PaintCommand] = []
        while (
            self._commands
            and self._commands[0].tick <= next_tick
            and len(selected) < MAX_COMMANDS_PER_TICK
        ):
            selected.append(self._commands.popleft())
        # AI packets first, human packets last so human supremacy wins ties.
        selected.sort(key=lambda command: 0 if command.source == 1 else 1)
        return selected

    def _simulate_tick(self) -> None:
        next_tick = self.tick + 1
        commands = self._commands_for_tick(next_tick)
        if commands:
            command_array = np.asarray(
                [command.as_row() for command in commands],
                dtype="<f4",
            )
            self.command_buffer.write(command_array.tobytes(order="C"))

        groups = (
            math.ceil(self.grid_width / WORKGROUP_SIZE),
            math.ceil(self.grid_height / WORKGROUP_SIZE),
        )
        for name in COMPUTE_PASSES:
            program = self.compute_passes[name]
            destination_index = 1 - self.current_buffer
            self.world_buffers[self.current_buffer].bind_to_storage_buffer(0)
            self.world_buffers[destination_index].bind_to_storage_buffer(1)
            if name == "constraint":
                self.command_buffer.bind_to_storage_buffer(2)
                program["command_count"].value = len(commands)
            program.run(group_x=groups[0], group_y=groups[1], group_z=1)
            self.ctx.memory_barrier()
            self.current_buffer = destination_index
        self.tick = next_tick
        self.session_tick += 1

    def on_render(self, _time: float, frame_time: float) -> None:
        self._update_fps(frame_time)
        if self.bridge is not None:
            self._apply_operations(self.bridge.take_operations())
        self._drain_relay()
        self._drain_swarm()
        self._publish_swarm_perception(frame_time)

        self.accumulator = min(
            self.accumulator + max(frame_time, 0.0),
            FIXED_STEP * (MAX_STEPS_PER_FRAME + 1),
        )
        if not self.paused:
            steps = 0
            while self.accumulator >= FIXED_STEP and steps < MAX_STEPS_PER_FRAME:
                self._simulate_tick()
                self.accumulator -= FIXED_STEP
                steps += 1
            if self.accumulator >= FIXED_STEP:
                self.accumulator = math.fmod(self.accumulator, FIXED_STEP)

        scene = self._ensure_scene_framebuffer()
        scene.use()
        scene.clear(0.002, 0.004, 0.012, 1.0)
        self.world_buffers[self.current_buffer].bind_to_storage_buffer(0)
        self._update_viewport_uniform()
        self.screen_triangle.render(mode=moderngl.TRIANGLES, vertices=3)
        self._draw_hud(scene)
        self.ctx.copy_framebuffer(self.wnd.fbo, scene)
        self.wnd.use()
        self._refresh_hud_title()

        if self.bridge is not None:
            self.bridge.run_jobs()

    def _update_fps(self, frame_time: float) -> None:
        self._fps_accum += max(frame_time, 0.0)
        self._fps_frames += 1
        if self._fps_accum >= 0.25:
            self._fps = self._fps_frames / max(self._fps_accum, 1e-6)
            self._fps_accum = 0.0
            self._fps_frames = 0

    def _hud_lines(self) -> list[str]:
        return [
            f"FPS {self._fps:.0f}  |  Tick {self.tick} @ {TICK_RATE_HZ} Hz",
            f"AI State: {self.swarm.status_label()}",
            f"Human Lock Cells: {self._hud_lock_cells}"
            f"  |  Entities: {len(self.entities)}",
        ]

    def _refresh_hud_title(self) -> None:
        self.wnd.title = (
            "Neural World Runtime — "
            + "  ·  ".join(self._hud_lines())
        )

    def _ensure_hud_texture(self) -> moderngl.Texture:
        size = self._scene_size()
        if self._hud_texture is not None and self._hud_texture.size == size:
            return self._hud_texture
        if self._hud_texture is not None:
            self._hud_texture.release()
        self._hud_texture = self.ctx.texture(size, 4)
        self._hud_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._hud_program["hud_texture"].value = 1
        return self._hud_texture

    def _draw_hud(self, target: moderngl.Framebuffer) -> None:
        self._hud_age += 1.0 / max(self._fps, 1.0)
        if self._hud_age >= HUD_REFRESH_INTERVAL:
            self._hud_age = 0.0
            self._upload_hud_texture()
        if self._hud_texture is None:
            return
        previous = self.ctx.fbo
        target.use()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._hud_texture.use(location=1)
        self._hud_vao.render(mode=moderngl.TRIANGLES, vertices=3)
        self.ctx.disable(moderngl.BLEND)
        if previous is not None:
            previous.use()

    def _upload_hud_texture(self) -> None:
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            return
        texture = self._ensure_hud_texture()
        width, height = texture.size
        image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except OSError:
            font = ImageFont.load_default()
        band = min(128, height)
        draw.rectangle((0, 0, width, band), fill=(4, 10, 22, 150))
        y = 8
        for line in self._hud_lines():
            draw.text((12, y), line, fill=(180, 245, 255, 240), font=font)
            y += 26
        texture.write(image.tobytes())

    def _grid_position(self, x: float, y: float) -> tuple[float, float]:
        width = max(float(self.wnd.width), 1.0)
        height = max(float(self.wnd.height), 1.0)
        grid_x = float(
            np.clip(x / width * self.grid_width, 0.0, self.grid_width - 1e-4)
        )
        grid_y = float(
            np.clip(
                (height - y) / height * self.grid_height,
                0.0,
                self.grid_height - 1e-4,
            )
        )
        return grid_x, grid_y

    def _queue_segment(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        erase: bool,
    ) -> None:
        self._enqueue(
            PaintCommand(
                tick=self.tick + 1,
                start_x=start[0],
                start_y=start[1],
                end_x=end[0],
                end_y=end[1],
                radius=self.brush_radius,
                category=self.selected_category,
                operation=-1.0 if erase else 1.0,
                priority=PRIORITY_LEVELS["user"],
            )
        )

    def on_mouse_position_event(self, x: int, y: int, _dx: int, _dy: int) -> None:
        self._cursor_grid_position = self._grid_position(x, y)

    def on_mouse_press_event(self, x: int, y: int, button: int) -> None:
        if button not in (LEFT_BUTTON, RIGHT_BUTTON):
            return
        self._active_button = button
        position = self._grid_position(x, y)
        self._cursor_grid_position = position
        self._last_grid_position = position
        self._queue_segment(
            position,
            position,
            erase=button == RIGHT_BUTTON or self._shift_down,
        )

    def on_mouse_drag_event(
        self,
        x: int,
        y: int,
        _dx: int,
        _dy: int,
    ) -> None:
        if self._active_button is None:
            return
        position = self._grid_position(x, y)
        self._cursor_grid_position = position
        start = self._last_grid_position or position
        self._queue_segment(
            start,
            position,
            erase=self._active_button == RIGHT_BUTTON or self._shift_down,
        )
        self._last_grid_position = position

    def on_mouse_release_event(self, x: int, y: int, button: int) -> None:
        if button != self._active_button:
            return
        position = self._grid_position(x, y)
        start = self._last_grid_position or position
        self._queue_segment(
            start,
            position,
            erase=button == RIGHT_BUTTON or self._shift_down,
        )
        self._active_button = None
        self._last_grid_position = None

    def on_key_event(self, key: int, action: int, modifiers: object) -> None:
        keys = self.wnd.keys
        self._shift_down = bool(getattr(modifiers, "shift", False))
        if action != keys.ACTION_PRESS:
            return
        if key == keys.SPACE:
            self.paused = not self.paused
            print("Paused" if self.paused else "Running")
        elif key == keys.S:
            self._save_world()
        elif key == keys.L:
            self._load_world()
        elif key == keys.R:
            self._reset_world()
        elif key == keys.NUMBER_1:
            self.selected_category = 1
            print("Brush: Human Barrier")
        elif key == keys.NUMBER_2:
            self.selected_category = 2
            print("Brush: Active Fluid")
        elif key == keys.NUMBER_3:
            self.selected_category = 3
            print("Brush: Solid (wall, no human lock)")
        elif key == keys.E:
            position = self._cursor_grid_position
            self._spawn_entity(
                SpawnEntity(
                    kind="blob",
                    x=position[0],
                    y=position[1],
                    priority=PRIORITY_LEVELS["user"],
                )
            )
        elif key == keys.X:
            removed = len(self.entities)
            self._apply_operations(self.entities.clear())
            print(f"Removed {removed} entities")
        elif key == keys.N:
            self.use_neural_material = not self.use_neural_material
            self._configure_material()
            print(
                "Material: neural network"
                if self.render_program["use_neural_material"].value
                else "Material: procedural"
            )
        elif key == keys.A:
            mode = self.swarm.toggle_swarm()
            print(
                "AI State: ACTIVE (Swarm Mode)"
                if mode.name == "SWARM"
                else "AI State: INACTIVE"
            )
            self._frames_since_perception = SWARM_PERCEPTION_INTERVAL
        elif key == keys.D:
            self.swarm.trigger_vortex()
            self._frames_since_perception = SWARM_PERCEPTION_INTERVAL
            print("Field Director: vortex injected")
        elif key == keys.C:
            self.swarm.request_clear()
            self._commands = deque(
                command for command in self._commands if command.source != 1
            )
            self._frames_since_perception = SWARM_PERCEPTION_INTERVAL
            print("Cleared pending AI commands; erasing unlocked AI fluid")

    def _save_world(self) -> None:
        grid = self._read_world()
        metadata: dict[str, object] = {"simulation_tick": self.tick}
        try:
            header, _existing = load_bds(self.world_path)
            previous = header.get("application_metadata")
            if isinstance(previous, dict):
                metadata = {**previous, "simulation_tick": self.tick}
        except (OSError, BDSFormatError, ValueError, TypeError):
            pass
        try:
            save_bds(
                self.world_path,
                grid,
                metadata=metadata,
            )
        except (OSError, ValueError) as exc:
            print(f"Save failed: {exc}")
        else:
            print(f"Saved {self.world_path} at tick {self.tick}")

    def _load_world(self) -> None:
        try:
            header, grid = load_bds(self.world_path)
            self._require_runtime_shape(grid)
        except (OSError, BDSFormatError, ValueError) as exc:
            print(f"Load failed: {exc}")
            return
        self._replace_world(grid)
        metadata = header.get("application_metadata", {})
        stored_tick = metadata.get("simulation_tick", 0)
        self.tick = stored_tick if isinstance(stored_tick, int) and stored_tick >= 0 else 0
        self.accumulator = 0.0
        self._commands.clear()
        self.entities = EntityRegistry(bounds=self.bounds)
        print(f"Loaded {self.world_path} at tick {self.tick}")

    def _reset_world(self, *, preserve_locks: bool = False) -> None:
        """Replace the world with a fresh one.

        A human reset clears everything, which is what pressing ``R`` should do.
        An AI reset carries the human-locked cells across, so the control plane
        cannot be used to launder away a boundary that the AI is forbidden to
        overwrite one cell at a time.
        """
        fresh = self._new_world()
        if preserve_locks:
            current = self._read_world()
            locked = current[..., HUMAN_LOCK_CHANNEL] >= 0.5
            if bool(locked.any()):
                fresh = np.array(fresh, dtype=DTYPE, copy=True)
                fresh[locked] = current[locked]
                print(f"Reset preserved {int(locked.sum())} human-locked cells")
        self._replace_world(fresh)
        self.tick = 0
        self.accumulator = 0.0
        self._commands.clear()
        self.entities = EntityRegistry(bounds=self.bounds)
        print("World reset")

    def on_resize(self, _width: int, _height: int) -> None:
        self._update_viewport_uniform()

    def on_close(self) -> None:
        if self.bridge is not None:
            self.bridge.stop()
            self.bridge = None
        if self.relay is not None:
            self.relay.stop()
            self.relay = None
        self.swarm.stop()
        if self._scene_framebuffer is not None:
            self._scene_framebuffer.release()
            self._scene_framebuffer = None
        if self._scene_texture is not None:
            self._scene_texture.release()
            self._scene_texture = None
        if self._preview_framebuffer is not None:
            self._preview_framebuffer.release()
            self._preview_framebuffer = None
        if self._preview_texture is not None:
            self._preview_texture.release()
            self._preview_texture = None
        if self._material_texture is not None:
            self._material_texture.release()
            self._material_texture = None
        if self._hud_texture is not None:
            self._hud_texture.release()
            self._hud_texture = None
        if self.recorder is not None:
            self.recorder.close()
            self.recorder = None


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    mglw.run_window_config(VideoPlayground)
