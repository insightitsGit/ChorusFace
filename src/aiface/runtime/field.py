"""Minimal GPU field runtime for the avatar.

This is the small slice of a 32-channel field substrate that a talking face
actually needs: two world SSBOs, a bounded command buffer, a constraint-only
compute tick, a fullscreen render pass, and a text HUD.

Deliberately absent: physics and semantic advection, entities, swarm planning,
networking, and any HTTP control surface. The avatar renders an immutable
photograph, so a pass that moves matter between cells would smear identity.
"""

from __future__ import annotations

import argparse
import io
import math
from collections import deque
from pathlib import Path
from typing import Any, Final

import moderngl
import moderngl_window as mglw
import numpy as np

from aiface.paths import DEFAULT_AVATAR_FACE
from aiface.runtime.bds import (
    GRID_HEIGHT,
    GRID_WIDTH,
    PRIORITY_LEVELS,
    TICK_RATE_HZ,
    VECTOR_DIMENSIONS,
    VELOCITY_CHANNELS,
    BDSFormatError,
    load_bds,
    save_bds,
)
from aiface.runtime.commands import PaintCommand
from aiface.runtime.shaders import (
    COMPUTE_PASSES,
    WORKGROUP_SIZE,
    load_compute_passes,
    load_shader,
    load_tick_ingest_shader,
)

FIXED_STEP: Final = 1.0 / TICK_RATE_HZ
MAX_STEPS_PER_FRAME: Final = 5
MAX_COMMANDS_PER_TICK: Final = 256
MAX_QUEUED_COMMANDS: Final = 8192
COMMAND_FLOATS: Final = 8
HUD_REFRESH_INTERVAL: Final = 0.25
PREVIEW_RESOLUTION: Final = 1024


def encode_png(pixels: bytes, width: int, height: int, components: int = 3) -> bytes:
    """Encode a bottom-up OpenGL pixel buffer as a top-down PNG."""
    from PIL import Image

    mode = {3: "RGB", 4: "RGBA"}.get(components)
    if mode is None:
        raise ValueError(f"Unsupported component count: {components}")
    expected = width * height * components
    if len(pixels) != expected:
        raise ValueError(f"Expected {expected} pixel bytes, received {len(pixels)}")
    image = Image.frombytes(mode, (width, height), pixels).transpose(
        Image.Transpose.FLIP_TOP_BOTTOM
    )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


class FieldRuntime(mglw.WindowConfig):
    """GPU-resident 32-channel field window running constraint-only ticks."""

    gl_version = (4, 3)
    title = "AIFace"
    window_size = (1024, 1024)
    aspect_ratio = 1.0
    resizable = True
    vsync = True
    fragment_shader = "avatar.frag"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--world",
            type=Path,
            default=DEFAULT_AVATAR_FACE,
            help=f"Seed world to load and save (default {DEFAULT_AVATAR_FACE})",
        )
        parser.add_argument(
            "--world-width",
            type=int,
            default=GRID_WIDTH,
            help="World width in cells",
        )
        parser.add_argument(
            "--world-height",
            type=int,
            default=GRID_HEIGHT,
            help="World height in cells",
        )

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)

        self.grid_width = int(getattr(self.argv, "world_width", GRID_WIDTH))
        self.grid_height = int(getattr(self.argv, "world_height", GRID_HEIGHT))
        if self.grid_width <= 0 or self.grid_height <= 0:
            raise ValueError("World dimensions must be positive")
        self.bounds = (self.grid_width, self.grid_height)
        self.world_path = Path(getattr(self.argv, "world", DEFAULT_AVATAR_FACE))

        self.compute_passes = {
            name: self.ctx.compute_shader(source)
            for name, source in load_compute_passes().items()
        }
        # TickFeed B1: full-face KEY/DELTA ingest (in-place on current world).
        self._tick_ingest = self.ctx.compute_shader(load_tick_ingest_shader())
        self._tick_dense_buf = self.ctx.buffer(reserve=GRID_WIDTH * GRID_HEIGHT * 4)
        self._tick_sparse_idx_buf = self.ctx.buffer(reserve=GRID_WIDTH * GRID_HEIGHT * 4)
        self._tick_sparse_vel_buf = self.ctx.buffer(reserve=GRID_WIDTH * GRID_HEIGHT * 4)
        self._pending_tick_package = None  # set by AvatarFaceApp each sim tick
        self.render_program = self.ctx.program(
            vertex_shader=load_shader("fullscreen.vert"),
            fragment_shader=load_shader(self.fragment_shader),
        )
        self.screen_triangle = self.ctx.vertex_array(self.render_program, [])
        self._scene_texture: moderngl.Texture | None = None
        self._scene_framebuffer: moderngl.Framebuffer | None = None
        self._preview_texture: moderngl.Texture | None = None
        self._preview_framebuffer: moderngl.Framebuffer | None = None

        self.command_buffer = self.ctx.buffer(
            reserve=MAX_COMMANDS_PER_TICK * COMMAND_FLOATS * np.dtype("f4").itemsize
        )
        self.command_buffer.bind_to_storage_buffer(2)
        self.world_buffers: list[moderngl.Buffer] = []
        self.current_buffer = 0
        self._replace_world(self._load_startup_world())

        self.tick = 0
        self.accumulator = 0.0
        self.paused = False
        self._commands: deque[PaintCommand] = deque()
        self.dropped_commands = 0

        for compute in self.compute_passes.values():
            compute["grid_size"].value = self.bounds
        self._tick_ingest["grid_size"].value = self.bounds
        self.render_program["grid_size"].value = self.bounds
        # Portrait rectangle in UV. Subclasses shrink this to letterbox the
        # face; the default fills the window so plain runtimes are unaffected.
        self._face_frame: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
        self._update_viewport_uniform()
        self._update_frame_uniform()

        self._fps = 0.0
        self._fps_accum = 0.0
        self._fps_frames = 0
        self._hud_age = HUD_REFRESH_INTERVAL
        # Typing must not wait out the HUD's lazy refresh, so input marks the
        # overlay dirty and the next frame repaints it.
        self._overlay_dirty = True
        self._hud_texture: moderngl.Texture | None = None
        self._hud_program = self.ctx.program(
            vertex_shader=load_shader("fullscreen.vert"),
            fragment_shader=load_shader("hud.frag"),
        )
        self._hud_vao = self.ctx.vertex_array(self._hud_program, [])
        self._ensure_hud_texture()

    # ---------------------------------------------------------------- world

    def _load_startup_world(self) -> np.ndarray:
        if not self.world_path.exists():
            raise FileNotFoundError(
                f"World not found: {self.world_path}. Build one with `aiface-seed`."
            )
        _header, grid = load_bds(self.world_path)
        self._require_runtime_shape(grid)
        print(f"Loaded {self.world_path}")
        return grid

    def _require_runtime_shape(self, grid: np.ndarray) -> None:
        expected = (self.grid_height, self.grid_width, VECTOR_DIMENSIONS)
        if grid.shape != expected:
            raise ValueError(
                f"Runtime requires grid shape {expected}, got {grid.shape}"
            )

    def _replace_world(self, grid: np.ndarray) -> None:
        self._require_runtime_shape(grid)
        payload = np.ascontiguousarray(grid, dtype="<f4").tobytes(order="C")
        new_buffers = [self.ctx.buffer(payload), self.ctx.buffer(payload)]
        old_buffers = self.world_buffers
        self.world_buffers = new_buffers
        self.current_buffer = 0
        for buffer in old_buffers:
            buffer.release()

    def _read_world(self) -> np.ndarray:
        self.ctx.finish()
        payload = self.world_buffers[self.current_buffer].read()
        return np.frombuffer(payload, dtype="<f4").reshape(
            (self.grid_height, self.grid_width, VECTOR_DIMENSIONS)
        )

    def _read_world_rows(self, y0: int, y1: int) -> np.ndarray:
        """Read only rows ``[y0, y1)`` of the field back to the CPU.

        NWR keeps the world GPU-resident; periodic telemetry must not haul the
        whole buffer (8 MB at 256²×32) across the bus every few frames. Rows
        are contiguous in the SSBO, so a row band is a single ranged read.
        """
        y0 = max(int(y0), 0)
        y1 = min(int(y1), self.grid_height)
        if y1 <= y0:
            return np.empty((0, self.grid_width, VECTOR_DIMENSIONS), dtype="<f4")
        row_bytes = self.grid_width * VECTOR_DIMENSIONS * 4
        payload = self.world_buffers[self.current_buffer].read(
            size=(y1 - y0) * row_bytes, offset=y0 * row_bytes
        )
        return np.frombuffer(payload, dtype="<f4").reshape(
            (y1 - y0, self.grid_width, VECTOR_DIMENSIONS)
        )

    def _persistable_world(self) -> np.ndarray:
        """The live field as a rest pose, ready to become a seed again.

        A ``.bds`` is a pose the runtime can start from, so transient tissue
        motion is dropped: saving mid-word would otherwise reload a face frozen
        part-way through a push, and save/load would not be a fixed point.
        """
        grid = self._read_world().copy()
        grid[..., list(VELOCITY_CHANNELS)] = np.float32(0.0)
        return grid

    def _save_world(self) -> None:
        """Persist the field while preserving the seed's application metadata.

        The avatar seed records the face box and mouth centre; dropping them on
        save would leave the renderer unable to register lip pieces later.
        """
        grid = self._persistable_world()
        metadata: dict[str, Any] = {"simulation_tick": self.tick}
        try:
            header, _existing = load_bds(self.world_path)
            previous = header.get("application_metadata")
            if isinstance(previous, dict):
                metadata = {**previous, "simulation_tick": self.tick}
        except (OSError, BDSFormatError, ValueError, TypeError):
            pass
        try:
            save_bds(self.world_path, grid, metadata=metadata)
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
        self.tick = (
            stored_tick if isinstance(stored_tick, int) and stored_tick >= 0 else 0
        )
        self.accumulator = 0.0
        self._commands.clear()
        self._on_world_reloaded()
        print(f"Loaded {self.world_path} at tick {self.tick}")

    def _on_world_reloaded(self) -> None:
        """Hook for subclasses holding state that describes the old world."""

    # ------------------------------------------------------------- commands

    def _enqueue(self, command: PaintCommand) -> None:
        if len(self._commands) >= MAX_QUEUED_COMMANDS:
            self._commands.popleft()
            self.dropped_commands += 1
            if self.dropped_commands == 1:
                print(
                    f"Command queue full at {MAX_QUEUED_COMMANDS}; dropping the "
                    "oldest packets. Watch the HUD drop counter."
                )
        self._commands.append(command)

    def _commands_for_tick(self, next_tick: int) -> list[PaintCommand]:
        """Take this tick's packets, leaving any overflow queued for the next.

        Overflow is carried rather than truncated: a burst that loses its tail
        would make the same session render differently on a replay.
        """
        selected: list[PaintCommand] = []
        while (
            self._commands
            and self._commands[0].tick <= next_tick
            and len(selected) < MAX_COMMANDS_PER_TICK
        ):
            selected.append(self._commands.popleft())
        # AI packets first, human packets last so human supremacy wins ties.
        selected.sort(key=lambda command: 0 if command.is_ai else 1)
        return selected

    def queue_tick_package(self, package: object | None) -> None:
        """Stage a TickPackage (or None for miss-damp) for the next simulate."""
        self._pending_tick_package = package

    def _run_tick_ingest(self) -> None:
        """Apply pending TickPackage into world ch0/1 (bridge B1/B2/B3)."""
        from aiface.tickfeed.gpu_pack import (
            dense_uints_from_package,
            face_uniforms,
            ingest_encoding,
            sparse_buffers_from_package,
        )
        from aiface.tickfeed.package import TickPackage
        from aiface.tickfeed.schema import (
            DeltaEncoding,
            PackageKind,
            VELOCITY_MISS_DAMP,
        )

        package = self._pending_tick_package
        self._pending_tick_package = None
        compute = self._tick_ingest
        self.world_buffers[self.current_buffer].bind_to_storage_buffer(0)

        if package is None:
            # Miss: damp face ROI if subclass set face uniforms via last package
            # Use full-grid damp of velocity via encoding 4 with last face box
            face = getattr(self, "_tick_face_box", None)
            if face is None:
                return
            compute["face_offset"].value = (int(face.x), int(face.y))
            compute["face_size"].value = (int(face.w), int(face.h))
            compute["is_keyframe"].value = 0
            compute["encoding"].value = 4
            compute["sparse_count"].value = 0
            compute["miss_damp"].value = float(VELOCITY_MISS_DAMP)
            groups = (
                math.ceil(int(face.w) / WORKGROUP_SIZE),
                math.ceil(int(face.h) / WORKGROUP_SIZE),
            )
            compute.run(group_x=max(groups[0], 1), group_y=max(groups[1], 1), group_z=1)
            self.ctx.memory_barrier()
            return

        if not isinstance(package, TickPackage):
            return
        self._tick_face_box = package.face
        uniforms = face_uniforms(package.face)
        compute["face_offset"].value = uniforms["face_offset"]
        compute["face_size"].value = uniforms["face_size"]
        compute["is_keyframe"].value = 1 if package.kind == PackageKind.KEYFRAME else 0
        compute["miss_damp"].value = float(VELOCITY_MISS_DAMP)
        enc = ingest_encoding(package)
        compute["encoding"].value = int(enc)

        if enc == 3:
            return
        if enc == 2:
            idx, vel = sparse_buffers_from_package(package)
            raw_idx = np.ascontiguousarray(idx, dtype="<u4").tobytes()
            raw_vel = np.ascontiguousarray(vel, dtype="<u4").tobytes()
            if len(raw_idx) > self._tick_sparse_idx_buf.size:
                self._tick_sparse_idx_buf.orphan(len(raw_idx))
            if len(raw_vel) > self._tick_sparse_vel_buf.size:
                self._tick_sparse_vel_buf.orphan(len(raw_vel))
            self._tick_sparse_idx_buf.write(raw_idx)
            self._tick_sparse_vel_buf.write(raw_vel)
            self._tick_sparse_idx_buf.bind_to_storage_buffer(2)
            self._tick_sparse_vel_buf.bind_to_storage_buffer(3)
            compute["sparse_count"].value = int(idx.size)
            # 16x16 local size → 256 threads / group
            groups_1d = math.ceil(max(int(idx.size), 1) / 256)
            compute.run(group_x=max(groups_1d, 1), group_y=1, group_z=1)
            self.ctx.memory_barrier()
            return

        # Dense KEY or DENSE_DELTA
        if package.delta_encoding == DeltaEncoding.EMPTY:
            return
        packed = dense_uints_from_package(package)
        raw = np.ascontiguousarray(packed, dtype="<u4").tobytes()
        if len(raw) > self._tick_dense_buf.size:
            self._tick_dense_buf.orphan(len(raw))
        self._tick_dense_buf.write(raw)
        self._tick_dense_buf.bind_to_storage_buffer(1)
        compute["sparse_count"].value = 0
        fw, fh = int(package.face.w), int(package.face.h)
        groups = (
            math.ceil(fw / WORKGROUP_SIZE),
            math.ceil(fh / WORKGROUP_SIZE),
        )
        compute.run(group_x=max(groups[0], 1), group_y=max(groups[1], 1), group_z=1)
        self.ctx.memory_barrier()

    def _simulate_tick(self) -> None:
        next_tick = self.tick + 1
        # TickFeed first: write full-face velocity, then constraint damps/snaps.
        self._run_tick_ingest()
        # Legacy ±4 PaintCommands disabled for speech; keep queue for debug only.
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
            compute = self.compute_passes[name]
            destination_index = 1 - self.current_buffer
            self.world_buffers[self.current_buffer].bind_to_storage_buffer(0)
            self.world_buffers[destination_index].bind_to_storage_buffer(1)
            if name == "constraint":
                self.command_buffer.bind_to_storage_buffer(2)
                compute["command_count"].value = len(commands)
            compute.run(group_x=groups[0], group_y=groups[1], group_z=1)
            self.ctx.memory_barrier()
            self.current_buffer = destination_index
        self.tick = next_tick

    # --------------------------------------------------------------- render

    def on_render(self, _time: float, frame_time: float) -> None:
        self._update_fps(frame_time)
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
        self._update_frame_uniform()
        self.screen_triangle.render(mode=moderngl.TRIANGLES, vertices=3)
        self._draw_hud(scene)
        self.ctx.copy_framebuffer(self.wnd.fbo, scene)
        self.wnd.use()

    def _scene_size(self) -> tuple[int, int]:
        return (
            max(int(self.wnd.buffer_width), 1),
            max(int(self.wnd.buffer_height), 1),
        )

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
        """A fixed-size target so previews are resolution-stable for callers."""
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

    def _update_viewport_uniform(self) -> None:
        width, height = self._scene_size()
        self.render_program["viewport_size"].value = (float(width), float(height))

    def _update_frame_uniform(self) -> None:
        self.render_program["avatar_frame"].value = tuple(
            float(value) for value in self._face_frame
        )

    def _screenshot(self) -> bytes:
        framebuffer = self._ensure_scene_framebuffer()
        width, height = framebuffer.size
        return encode_png(framebuffer.read(components=3), width, height, 3)

    def _preview(self) -> bytes:
        """Render the live face at exactly 1024x1024, whatever the window size.

        The preview is the portrait alone: window chrome such as the chat frame
        is dropped so callers get a stable image no matter how the window is
        laid out.
        """
        framebuffer = self._ensure_preview_framebuffer()
        framebuffer.use()
        framebuffer.clear(0.002, 0.004, 0.012, 1.0)
        self.world_buffers[self.current_buffer].bind_to_storage_buffer(0)
        self.render_program["viewport_size"].value = (
            float(PREVIEW_RESOLUTION),
            float(PREVIEW_RESOLUTION),
        )
        self.render_program["avatar_frame"].value = (0.0, 0.0, 1.0, 1.0)
        self.screen_triangle.render(mode=moderngl.TRIANGLES, vertices=3)
        self.ctx.finish()
        payload = framebuffer.read(components=3)
        self._update_viewport_uniform()
        self._update_frame_uniform()
        self.wnd.use()
        return encode_png(payload, PREVIEW_RESOLUTION, PREVIEW_RESOLUTION, 3)

    # ------------------------------------------------------------------ hud

    def _update_fps(self, frame_time: float) -> None:
        self._fps_accum += max(frame_time, 0.0)
        self._fps_frames += 1
        if self._fps_accum >= 0.25:
            self._fps = self._fps_frames / max(self._fps_accum, 1e-6)
            self._fps_accum = 0.0
            self._fps_frames = 0

    def _hud_lines(self) -> list[str]:
        return [f"FPS {self._fps:.0f}  |  Tick {self.tick} @ {TICK_RATE_HZ} Hz"]

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
        if self._hud_age >= HUD_REFRESH_INTERVAL or self._overlay_dirty:
            self._hud_age = 0.0
            self._overlay_dirty = False
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
        self._paint_overlay(draw, width, height)
        texture.write(image.tobytes())

    def _paint_overlay(self, draw: Any, width: int, height: int) -> None:
        """Hook for extra overlay chrome drawn above the HUD band."""

    # ----------------------------------------------------------------- keys

    def on_key_event(self, key: int, action: int, modifiers: object) -> None:
        keys = self.wnd.keys
        if action != keys.ACTION_PRESS:
            return
        if key == keys.SPACE:
            self.paused = not self.paused
            print("Paused" if self.paused else "Running")
        elif key == keys.S:
            self._save_world()
        elif key == keys.L:
            self._load_world()

    def on_resize(self, _width: int, _height: int) -> None:
        self._update_viewport_uniform()

    def on_close(self) -> None:
        for resource in (
            self._scene_framebuffer,
            self._scene_texture,
            self._preview_framebuffer,
            self._preview_texture,
            self._hud_texture,
        ):
            if resource is not None:
                resource.release()
        self._scene_framebuffer = None
        self._scene_texture = None
        self._preview_framebuffer = None
        self._preview_texture = None
        self._hud_texture = None


__all__ = [
    "FIXED_STEP",
    "MAX_COMMANDS_PER_TICK",
    "PREVIEW_RESOLUTION",
    "PRIORITY_LEVELS",
    "FieldRuntime",
    "encode_png",
]
