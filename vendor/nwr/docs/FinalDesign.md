# Neural World Runtime — Substrate Specification

Status: implementation specification  
Format versions: `bds-1.0` world snapshots, `bdl-1.1` command logs (`bdl-1.0` readable)  
Platform: Windows or Linux with OpenGL 4.3+

## 1. Purpose

The Neural World Runtime is a deterministic, GPU-resident field simulation and renderer. It proves that:

1. a persistent 32-channel world state can live entirely on the GPU;
2. per-cell updates need no CPU involvement;
3. authority over cells can be resolved by rank rather than by a single override bit;
4. history, networking, and persistence can all be expressed as one operation stream;
5. rendering can be driven by a trained network rather than hand-written shading.

The CPU owns the window, event queue, dispatch scheduling, and explicit persistence. Simulation state and per-cell updates remain on the GPU. The simulation is a coupled field model; it does not claim physically accurate fluid dynamics.

## 2. Scope

### Implemented

- Worlds of arbitrary size, `256 × 256` by default, with the grid size supplied as a uniform to every stage.
- 32 little-endian `float32` values per cell.
- 60 Hz fixed simulation with three compute passes per tick: physics, semantic, constraint.
- Two SSBOs used as a ping-pong pair, swapped after each pass.
- A command SSBO carrying tick-stamped segments with an authority level.
- Four-level authority resolution: background, AI, constraint, user.
- Validated and checksummed `.bds` snapshots and append-only `.bdl` command logs with torn-tail recovery, recording every operation that reaches the GPU along with its authority and its writer.
- Lossless deflate paging and a lossy anchor/residual archive codec with a bounded error.
- A chunked world model with activity tracking and quiescent-chunk eviction.
- One fragment pass with two logical stages: material shading and edge overlay.
- A trained material MLP (`32 → 16 → 3`) evaluated in the fragment shader from a weight texture.
- An opt-in AI control layer: a validated command grammar, a loopback HTTP bridge with observation and frame capture, and a client for OpenAI-compatible models.
- An operation-stream relay so multiple clients converge on one world from a few dozen bytes per edit.
- A headless deterministic exporter producing PNG sequences and, when ffmpeg is present, video.

### Deferred

- GPU-resident chunk streaming: chunking currently governs storage and paging, not which chunks are resident in VRAM. The whole world stays resident.
- DirectStorage and `O_DIRECT`; neither has a Python binding, so both need a native extension.
- `.bdc` GPU caches and `.bda` asset packs.
- Branching and merging timelines; the log is linear.
- VR and stereo projections, which require an OpenXR runtime and hardware.

### Determinism guarantee

Identical command streams reproduce bit-identical worlds on the same GPU and driver, which the test suite verifies. Floating-point results are not guaranteed to match across different devices, so networked peers agree on commands rather than on bits.

## 3. Cell schema

Each cell is a contiguous array of 32 floats:

| Range | Group | Meaning |
| --- | --- | --- |
| 0–7 | Kinematics | velocity X/Y/Z, density, pressure, shear, temperature, energy |
| 8–15 | Material | albedo R/G/B, opacity, roughness, metallic, emission, refraction |
| 16–23 | Intent | attraction, alignment, user affinity, growth, decay, lifespan, reserved, reserved |
| 24–31 | Rules | hard surface, permeability, thermal threshold, phase trigger, reserved ×2, authority priority, human lock |

Channels 30 and 31 are authority metadata rather than material state, and only commands write them. They are excluded from diffusion, from clamping, and from anchor snapping. Excluding them from snapping is load-bearing: a cell whose rules group happened to snap to the barrier anchor would otherwise acquire a permanent lock that no user action had requested.

Channel 31 is the lock. Values at or above `0.5` mean the cell is immovable, and all three compute passes copy it through unchanged. Locking follows from the painted anchor rather than from the act of painting, so barriers lock and fluid does not: painted fluid is injected and then evolves.

Channel 30 holds the authority level that last wrote the cell, normalized to `level / 3`.

## 4. Simulation

### 4.1 Fixed-step scheduler

- Tick duration: `1 / 60` seconds.
- Rendering is independent of simulation frequency.
- At most five simulation steps run in one rendered frame.
- Excess accumulated time is discarded to prevent an unbounded catch-up spiral.
- Input commands are assigned to the next simulation tick.

### 4.2 Pass structure

Each tick runs three compute dispatches, ping-ponging the world buffers after every one so no pass ever reads a buffer it is writing:

1. **Physics** (`shaders/physics.comp`) updates kinematics, channels 0–7, and damps velocity.
2. **Semantic** (`shaders/semantic.comp`) updates material and intent, channels 8–23, and decays intent.
3. **Constraint** (`shaders/constraint.comp`) clamps, snaps to anchors, garbage-collects the authority of emptied cells, and applies commands.

Only the constraint pass binds the command buffer, which keeps authority resolution in exactly one place. Every pass copies the channels it does not own through unchanged.

The anchor codebook and channel layout are generated as GLSL constants from `bds_format.py` by `shader_library.py` and injected into each stage, so the CPU and GPU views of a cell cannot drift apart.

### 4.3 Neighborhood coupling

Every unlocked cell reads itself and its valid 8-neighbor Moore neighborhood from the read SSBO. A neighbor contributes according to a scale-aware semantic weight:

`weight = max(cosine(self[8:24], neighbor[8:24]), 0) × min(length(neighbor[8:24]) / 4, 1)`

The self cell always has a weight of `1`, making an isolated state stable. The weighted neighborhood mean is applied by group:

- velocity channels 0–2: blend factor `0.04`, followed by damping `0.998`;
- remaining kinematic channels 3–7: blend factor `0.02`;
- material channels 8–15: blend factor `0.025`, no decay;
- intent channels 16–23: blend factor `0.015`, followed by decay `0.9995`;
- rule channels 24–31: no diffusion.

This is a deterministic coupled field model, not Navier–Stokes. Material channels have no decay term, so a painted body of fluid spreads until it reaches equilibrium with its surroundings rather than fading away.

### 4.4 Stability

In the constraint pass, for unlocked cells:

1. channels 0–29 are clamped to `[-1.5, 1.5]`;
2. the kinematics, material, and intent groups are each snapped to the nearest category anchor group when their L2 norm exceeds `1.5`;
3. the rules group is clamped but never snapped, which keeps authority out of the snapping path;
4. a cell whose density and material norm both fall below `0.01` releases its authority back to `background`;
5. commands are applied last, subject to authority resolution.

Anchor group norms are constrained to at most `1.4` so snapping always contracts.

## 5. Authority resolution

Four levels replace the original single override bit:

| Level | Name | Typical writer |
| --- | --- | --- |
| 3 | `user` | mouse input |
| 2 | `constraint` | scripted invariants |
| 1 | `ai` | bridge and relay clients |
| 0 | `background` | simulation only |

A write succeeds where the command's level is greater than or equal to the target cell's stored level. A caller may request a level at or below its own and never above it, which the compiler enforces before anything reaches the GPU. Erase releases the cell to `background`, so clearing never leaves an unwritable hole behind.

The practical effect: AI edits cannot destroy user work, users can always overrule AI, and the simulation itself can overwrite nothing that was authored.

Channel 31 is governed by a rule the priority table does not express: **an AI write may never raise it.** The constraint pass copies every channel from the chosen anchor, and one anchor (`human_barrier`) carries a lock, so the pass captures the cell's own lock value and holds it down for AI writers. Without that, an agent could mint a boundary it is then permanently barred from crossing — an AI erase is `±2`, which the same pass refuses on locked cells — and locked cells are excluded from advection and diffusion, so the region would be permanently inert. Enforcement is in three layers, and the shader one is the only one that cannot be routed around:

| Layer | Location | Role |
| --- | --- | --- |
| Shader | `shaders/constraint.comp` | Authoritative. Holds channel 31 down for any `±2`/`±4` write, whatever reached the command buffer |
| Compiler | `ai_commands._compile_command` | Refuses `human_barrier` below `user` authority and names `solid` instead |
| Grammar | `schema_for_authority` | Does not advertise `human_barrier` to a caller that cannot paint it |

Three further consequences are easy to miss, and all are load-bearing:

- Because a *human* write may overrule a lock by design, an application whose premise is that some world must not change cannot rely on channel 31 alone. It has to withhold the human write path as well; `avatar_chat_driver.py` disables its inherited brush and world controls for exactly this reason.
- Because authority and writer class both decide the outcome, a session log has to record both. §7's `.bdl` record stores the writer alongside the priority, so a replayed AI erase is refused on locked cells exactly as it was live. Neither is ever inferred from the other, in either direction — except for `bdl-1.0` logs, which did not store the writer at all.
- Because the two are separate facts, top authority does not buy AI identity a lock. An AI write at `user` priority still cannot raise channel 31.

## 6. Input commands

A command contains eight float32 values:

`x0, y0, x1, y1, radius, category, operation, priority`

Coordinates and radius are in grid-cell units. The compute shader measures each cell-center's distance from the segment, producing continuous strokes even when mouse events are sparse.

- Left drag: paint the selected anchor.
- Right drag or Shift+left drag: erase to Vacuum.
- Among commands the caller is permitted to apply, the last one affecting a cell wins.
- Up to 64 commands are submitted per tick; overflow remains queued for later ticks. This applies to the headless exporter as well as the interactive runtime: a replay that discarded the overflow would not reproduce the session it came from.

## 7. History and replay

A session is a base `.bds` snapshot plus an append-only `.bdl` log of tick-stamped operations, so history costs 32 bytes per edit rather than a frame of state.

- The header is a fixed 1 KiB block: magic `BDL1`, a JSON length, then the JSON document.
- Records are fixed 32-byte structs, which makes the log seekable and its integrity checkable by length alone. A record carries the tick, the operation kind, the category, the authority level, the writer class, the geometry, and one delta.
- Reading tolerates a torn final record, the expected outcome when a process is killed mid-write, and reports how many bytes were discarded. Strict mode rejects it instead.
- Records reach the operating system on every append, so an abruptly killed process still leaves a replayable log.
- Log ticks come from a session counter that never rewinds. `reset` and `load` rewind the world tick, so logging against the world tick would produce a non-monotonic log.

**Every operation that reaches the GPU has a record.** Paint and erase, the five controls, temperature deltas (`±3`) and velocity impulses (`±4`). Anything else raises at record time rather than being skipped, because a log that quietly omits a write turns a replay guarantee into a guess. Two shapes share bytes with segments rather than growing the record: a velocity impulse reuses the two floats a segment spends on its end point, matching the shader, which reads the impulse from `segment.zw`; a temperature delta uses the sixth float, which `bdl-1.0` left as padding.

Entity spawn and removal are **not** records. They are intents — they name a kind and a place, and the entity registry resolves them into ordinary segments at the caller's authority before anything reaches the GPU. Those segments are logged as they are enqueued, so a session using entities already replays faithfully, by replaying the writes rather than re-deriving them. Logging the intent as well would double-apply on replay and would make replay depend on registry allocation order rather than on the log.

`bdl-1.0` is byte-compatible and still loads: same magic, same 32-byte record, same field offsets. It differs only in what two bytes mean. It wrote the writer byte as a literal zero and left the trailing four as padding, so its writer identity is genuinely unrecoverable and replaying one falls back to inferring the writer from authority. That is **best-effort, not faithful**, and `is_lossy_version` exists so a caller can say so rather than overclaim.

`export_video.py` replays a log against its snapshot and verifies the snapshot against the checksum recorded in the log header. §20 states what has actually been measured about that replay.

## 8. Storage and compression

Two codecs serve different needs, and the distinction is deliberate:

- **Lossless** (`pack_lossless`) deflates raw float32 and is used whenever data must survive a round trip unchanged, notably chunk paging. Paging must never alter a world, so eviction never uses the lossy codec.
- **Anchor/residual** (`encode_anchor_residual`) stores a per-cell anchor index plus a quantized residual, reaching roughly 380× on seeded worlds. The quantization step defaults to whatever value prevents any residual from clipping, which bounds the absolute error at half a step and reports it. A fixed step may be requested instead, trading that guarantee for a known precision.

A `ChunkedWorld` divides a world into chunks, tracks per-chunk activity by checksum, and evicts quiescent chunks to compressed memory. Restoring a chunk is bit-exact.

## 9. Networking

`net_relay.py` distributes operations, not tensors. The relay compiles and validates each request once and broadcasts the resulting geometry, so peers cannot diverge by recompiling a request independently. Clients authenticate with a shared token, receive a sequence number per batch, and apply batches in order.

## 10. AI control layer

AI participates as a command compiler, never as a direct writer of cell vectors or pixels. The layer has three parts.

### 10.1 Command grammar

A request is `{"commands": [...]}` with at most 64 ordered commands. Each command is either a cell write or a control action.

- `paint` requires `category` (`human_barrier`, `active_fluid`, or `solid`) and a `region`, and accepts an optional `priority`. Painting writes a full anchor; whether the result is locked follows from that anchor.
- `erase` requires a `region` and accepts an optional `priority`. It writes Vacuum and releases the cell's authority. Painting Vacuum is rejected so that clearing has exactly one spelling.
- `reset`, `save`, `load`, `pause`, and `resume` take no other fields.

Two of these are restricted by the caller's authority, because neither is reachable by the per-cell rules of §5:

- `save` and `load` read and write the world file, replacing a world without submitting a single cell write. A caller below `user` authority is refused. `reset` remains open to any caller, and the runtime carries human-locked cells across a reset requested below `user` authority, so §5's guarantee survives the control plane.
- `human_barrier` mints a lock on every cell it covers. Only a caller at `user` authority may paint it; below that the request is refused and directed to `solid`, which is a hard surface carrying no lock. This keeps §5's "only human input may claim channel 31" true of the command grammar and not just of the shader.

Regions are `point`, `line`, `polyline`, `circle`, `ring`, `rectangle`, `rectangle_outline`, `polygon`, and `polygon_outline`. Coordinates are grid cells with the origin at the bottom-left, matching the mouse mapping, and are validated against the running world's actual size.

### 10.2 Compilation and validation

`ai_commands.py` is the only path from a request to the GPU. It rejects unknown actions, unknown or unexpected fields, non-numeric or non-finite values, out-of-range coordinates, brush radii outside `[0.5, 64]`, inverted rectangles, priorities above the caller's own authority, and requests needing more than 4096 segments. Accepted regions are rasterized into the same capsule segment the mouse produces:

- filled discs become one segment whose radius is the circle radius;
- filled rectangles and polygons become per-row segments between cell centres with radius `0.5`, giving exact even-odd coverage;
- outlines become closed strokes with radius `thickness / 2`.

Because compilation targets the existing primitive, AI input inherits the same authority rules, tick stamping, and per-tick command limit as human input. Large fills therefore span several ticks and appear progressively.

### 10.3 Bridge and observation

`ai_bridge.py` serves loopback HTTP with bearer-token authentication and its own authority level, `ai` by default. OpenGL work is confined to the render thread: requests needing world data are queued as jobs, the render loop fulfils them through `run_jobs`, and queued operations are drained through `take_operations`. A request that arrives while the window is not rendering fails with a timeout rather than blocking indefinitely.

Observation is deliberately model-friendly: category counts by nearest anchor, density/energy/emission statistics, occupied and locked bounding boxes, a coarse occupancy map rendered as digit rows with the top of the grid first, and a storage report giving both the lossless paging footprint and the lossy archive footprint. Frames are captured from a dedicated offscreen colour target, because the window's back buffer holds undefined contents after a swap.

`ai_agent.py` is a client for any OpenAI-compatible chat completions endpoint. It fetches the grammar and optionally the current observation, requests a JSON object, revalidates the result locally, and only then submits it.

## 11. Category anchors

The codebook contains full 32-dimensional anchors:

- `Vacuum`: all zero.
- `Human Barrier`: bright magenta, opaque, emissive, hard-surface rule, locked.
- `Active Fluid`: blue/cyan material, density/pressure/energy, permeability, and mild behavioral intent.

`bds_format.py` is the single source of these definitions. They are written into `.bds` metadata and generated into GLSL by `shader_library.py`, so the two representations cannot disagree.

## 12. Rendering

The application draws a fullscreen triangle that reads the current simulation SSBO directly. It renders into an offscreen colour target and copies that target to the window, so a completed frame remains readable for screenshots after the buffer swap.

Stage A produces the material colour by one of two paths:

- **Procedural**: channels 8–10 give base colour, channel 14 and kinematic energy give emission, and the norm of channels 16–23 adds a secondary glow.
- **Neural**: a `32 → 16 → 3` MLP with ReLU hidden units, evaluated per fragment from a single-channel float weight texture. Row `i` holds unit `i`'s weights with its bias in the final column.

The network is trained by `material_network.py`, which distills the procedural response using Adam in numpy and exports the weights. Distillation makes the result measurable rather than decorative: validation MSE is about `8 × 10⁻⁵`, anchors reproduce to within `0.02`, and vacuum maps to under `0.01`. That last property matters more than the average error, because vacuum is the most common cell and any bias there lifts the rendered black level into grey. Training therefore includes a fraction of exact, noise-free anchor samples.

Stage B applies a 3×3 Sobel operator to density, hard-surface, and lock signals, compositing a cyan/magenta neon edge over the material layer. The default `1024 × 1024` window gives an integer four-screen-pixels-per-cell scale and avoids fractional grid shimmer.

## 13. `.bds` binary format

All integer and float values are little-endian.

| Offset | Size | Value |
| --- | --- | --- |
| `0x0000` | 8 | Magic bytes `BDS1\0\0\0\0` |
| `0x0008` | 4 | JSON byte length as little-endian `uint32` |
| `0x000C` | variable | UTF-8 JSON |
| after JSON | to `0x1000` | zero padding |
| `0x1000` | `width × height × 32 × 4` | contiguous float32 payload |

The JSON header includes format version, grid dimensions, vector dimensions, tick rate, dtype, byte order, payload length, CRC32, schema, anchors, and optional application metadata.

Readers must reject:

- wrong magic or unsupported version;
- malformed or oversized JSON;
- incompatible dimensions, dtype, or byte order;
- truncated or trailing payload bytes;
- CRC mismatch;
- non-finite payload values.

Writers use a temporary file, flush and `fsync`, then atomically replace the destination. Normal buffered I/O is used.

## 14. Export

`export_video.py` renders without a window. It creates a standalone OpenGL 4.3 context, runs the same three compute passes and fragment shader as the interactive application, and writes a PNG sequence at a fixed tick rate. Given a `.bds` snapshot and a `.bdl` log it reproduces a recorded session; given ffmpeg on `PATH` it also muxes the sequence into a video. When ffmpeg is absent it says so and keeps the frames rather than failing.

Because it shares the pipeline with the application, the exporter is also the determinism harness: rendering the same inputs twice yields byte-identical PNG files.

## 15. Runtime controls

- Left mouse drag: paint.
- Right mouse drag or Shift+left drag: erase.
- `1` and `2`: select the Human Barrier or Active Fluid brush.
- `N`: toggle the neural material network.
- Space: pause/resume.
- `S`: save.
- `L`: load.
- `R`: reset.
- Escape: close.

The default world path is `output/worlds/playground/world.bds`. The AI bridge is opt-in through `--ai-server` or `NWR_AI_SERVER` and the relay through `--relay` or `NWR_RELAY`; both require a bearer token that is generated and printed unless supplied, and both *refuse* to bind a non-loopback address. A routable bind is an explicit operator decision made with `--allow-remote-bind`, not something a stray `--ai-host` or environment variable can do by accident. `--ai-authority` sets the write authority granted to bridge and relay callers, `ai` by default. `--record DIR` captures a replayable session, and `--world-width`/`--world-height` set the world size.

## 16. Performance and compatibility

A `256 × 256` grid occupies `256 × 256 × 32 × 4 = 8 MiB`, so the two simulation SSBOs occupy 16 MiB and scale linearly with cell count. The command buffer is negligible. Three passes per tick means three dispatches and three barriers per tick rather than one.

OpenGL compute requires version 4.3. macOS OpenGL is limited to 4.1 and is unsupported. No unmeasured CPU-utilization percentage is claimed. The measurable guarantee is that no per-cell simulation or rendering loop runs on the CPU.

## 17. Remaining architecture work

The pipeline is now:

`World State → Chunk Storage → GPU Scheduler → Physics → Semantic → Constraint → Material → Edge Overlay`

The missing link is GPU-resident chunk streaming. Chunking today governs storage and paging while the whole world stays resident in VRAM, which is the right order to build it in but does not yet bound VRAM for very large worlds. That needs per-chunk activity computed on the GPU with atomic counters, a residency table, and re-upload on activation. Shared-memory stencil tiling and a structure-of-arrays layout are the natural follow-ups once chunk residency exists.

## 18. Optional video2game product

`video2game.py` is a separate ingestion product that converts a selected video
frame and local optical flow into a standard `.bds` snapshot. It uses a separate
OpenCV dependency set and never becomes part of the runtime's import graph.
The original playground remains usable with `python main.py`; converted worlds
are opened explicitly with `python main.py --world output/worlds/from-video/game_world.bds`.

The converter maps optical flow, luminance, RGB, frame difference, and Sobel
edges onto the existing channel schema, then applies one deterministic preset:
`FLUID_SANDBOX`, `GRAVITY_REVERSE`, or `SOLID_EDGES`. It always leaves the human
lock at zero. Hard-surface cells are stable coupling barriers, while only human
input may claim channel 31.

The subsystem's complete product boundary, mappings, metadata, and guarantees
are specified in [`Video2GameDesign.md`](Video2GameDesign.md).

## 19. AI interoperability surface

Assistants interact through observation APIs and validated commands only:

- `.bds` headers embed optional `ai_metadata` for drag-into-assistant discovery.
- `ai_world.generate_ai_summary`, `inspect_region`, and `export_ai_context`
  produce compact semantic JSON without dumping every cell.
- `ai_command_compiler` accepts named commands (`PaintMaterial`, `SetMaterial`,
  `Erase`, `IncreaseTemperature`, `DecreaseTemperature`) with CRC-sealed
  envelopes, and still accepts the native region grammar.
- Temperature deltas are a real GPU path (`operation = ±3`) that adjust
  channel 6 without rewriting material.
- `spawn_entity` and `remove_entity` are backed by the `entities.py` registry:
  named entities carry identity, drift with the field, and express themselves
  through the same segment and temperature-delta operations. Capabilities the
  runtime genuinely lacks are still rejected rather than approximated.
- Bridge routes `/context`, `/inspect`, and `/preview` complement `/state` and
  `/screenshot`. `/preview` renders at exactly 1024 × 1024 through a dedicated
  offscreen framebuffer, independent of window size.
- `ai_handoff.py` packages a world for assistants that cannot reach the bridge.
  Chat models have no GPU, so the bundle substitutes a sampled filmstrip and
  GIF for live rendering, `timeline.json` for live observation, and a
  NumPy-only reader for the binary payload. The briefing's worked example is
  generated from the live `COMMAND_SCHEMA` so it cannot drift, and the loop
  stays turn-based: the model proposes, a human applies through `ai_agent.py`,
  and the same validation and lock enforcement apply.

Operator documentation lives in [`AI_API.md`](AI_API.md).

## 20. Determinism verification

`determinism.py` turns reproducibility from a claim into a measurement. It
fingerprints the device, hashes the world, and compares a reference report
against an observed one within tolerance, so cross-GPU divergence is located
rather than assumed.

The claim it supports is **same-device bit-identical replay of a `bdl-1.1` log**.
Cross-device agreement is a measured tolerance with a device fingerprint, not a
bit-exactness claim, and never will be: drivers are free to choose different
FMA contractions and work distribution.

How the replay leg is built matters, because an earlier version of it could not
fail. It compared two simulations of the same in-memory schedule, which is a
third repetition of the same run — it re-measured same-device repeatability while
reporting a log-fidelity result. It said "bit-identical" the whole time replay was
silently promoting every AI write to user authority. What runs now:

1. A live run drives a probe schedule through the pipeline while a `SessionRecorder` writes a real `.bdl`.
2. The replay leg reads *that file back* and runs it from the same seed.
3. The two final worlds are compared, and separately the operations read back are compared to the ones that went in, field by field. A world comparison alone can only catch a lost field that happens to matter to this particular run.
4. Both runs report how many writes actually reached the GPU. This is the check a world comparison structurally cannot make: both legs go through the same renderer, so a write dropped there is dropped on both sides and cancels out.

The probe schedule is chosen so that each plausible loss changes the world. It
mints a human lock, then aims an AI erase at it whose authority (`user`) and
identity (AI) **disagree on purpose** — so a writer that is lost and re-guessed
from authority replays as a human erase and destroys the lock. It also records an
AI paint whose authority lands in channel 30, a temperature delta, and a velocity
impulse. `tests/test_determinism.py` reintroduces each bug and asserts the harness
notices.

## 21. Capability boundaries

The runtime's measured limits and the use cases it does and does not serve are
documented separately so the specification does not overstate its reach:

- [`Video2GameDesign.md`](Video2GameDesign.md) §6 — world-size cost curve,
  the shader-storage ceiling, and the semantic limits of video ingestion.
- [`UseCases.md`](UseCases.md) — supported applications, adjacent ones with
  their specific gaps, and what is out of scope.
