# Neural World Runtime

**A deterministic, GPU-resident field substrate.** Every cell of a dense 2D grid carries a 32-channel vector. Three OpenGL 4.3 compute passes advance the entire grid each tick over ping-pong SSBOs, and the renderer reads that GPU state directly. The CPU owns the window, the event queue, and explicit saves — it is never in the simulation loop.

That sentence is the whole idea, and everything below is a consequence of it. Because state never leaves the GPU, a 256 × 256 world costs 8 MiB and a tick costs three dispatches. Because every write — mouse, AI, entity, or game logic — funnels through one command buffer and one constraint shader, authority is resolved in exactly one place. And because that place is a shader with no branches on wall-clock time, the same inputs produce the same world, byte for byte, every run.

**This is not one application.** It is the layer underneath several. A talking face, a playable game, a video-to-world converter, and an AI benchmark environment all ship here, and they are the same engine answering two questions differently: *what writes the channels*, and *who reads the results*. They share one file format, one validated write path, and one set of authority rules.

## This repository is the heart of the product

Child products are built *on* the substrate, not beside it. They inherit the cell schema, the `.bds` format, GPU-resolved authority, and the Master Lock, and they are judged against the invariants defined here. When a child and this runtime disagree about a substrate rule, **this runtime is correct.**

The first one to graduate into its own repository is [**AIFace**](https://github.com/insightitsGit/AIFace) — the productised talking face. It has run well past what lives here: a continuous muscle displacement field instead of rigid pieces, ~39 muscles with groups and asymmetry, an in-window chat frame. It vendors a minimal copy of this substrate rather than importing the package, so it deploys alone while still honouring the schema and the lock. Build faces there; change substrate rules here.

## The two things that actually make it different

Plenty of engines render fields. Two properties are hard to get elsewhere, and they are the reason to choose this one.

**Determinism you can verify.** Not "we tried to avoid nondeterminism" — a harness that fingerprints your GPU, hashes the world tick by tick, and tells you *where* two machines diverged. Same device and same command log means bit-identical worlds. Across devices, floating point drifts, so peers agree on commands rather than on bits, and `determinism.py --verify` measures the gap instead of assuming it away.

**Authority you cannot argue with.** Channel 31 is a hard, GPU-level veto. An AI-sourced write aborts on a human-locked cell *before* priority is even considered, so a model claiming `user` authority still gets refused by the shader:

```glsl
// shaders/constraint.comp — the lock test precedes the priority test, deliberately
if (is_ai && state[HUMAN_LOCK_CHANNEL] >= 0.5) {
    continue;
}
float command_priority = clamp(command.settings.w, 0.0, 1.0);
if (command_priority + PRIORITY_TOLERANCE < state[PRIORITY_CHANNEL]) {
    continue;
}
```

That makes "the human's boundary is inviolable" an executable property rather than a paragraph in a policy document. Lock a region by hand, tell a connected model to destroy it, and watch the write get dropped by the GPU rather than by a prompt.

| Primitive | Where | What it gives you |
| --- | --- | --- |
| 32-channel field substrate | `bds_format.py`, `shaders/` | Dense state evolving on the GPU, no CPU in the loop |
| Deterministic snapshot + log | `bds_format.py`, `bds_history.py`, `determinism.py` | Bit-exact replay at 32 bytes per edit |
| GPU-enforced authority | `shaders/constraint.comp` | `user` > `constraint` > `ai` > `background`, resolved in the shader |
| Machine-readable control API | `ai_bridge.py`, `ai_commands.py`, `ai_world.py` | Observe, inspect, and command over HTTP with validated JSON |
| Ingestion adapters | `video2game.py`, `build_avatar_seed.py` | Any external signal becomes a starting world |

The binding specification is [`docs/FinalDesign.md`](docs/FinalDesign.md); it wins when documents disagree. The capability map, including honest gaps, is [`docs/UseCases.md`](docs/UseCases.md).

## Requirements

- Python 3.10+
- Windows or Linux
- An OpenGL 4.3-capable GPU and a current driver

macOS is unsupported: its OpenGL stops at 4.1 and provides no compute shaders.

## Install

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On Linux, activate with `source .venv/bin/activate`. The core runtime is four packages — `moderngl`, `moderngl-window`, `numpy`, `Pillow`. Image and video ingestion is an optional extra that the runtime never imports:

```powershell
python -m pip install -r requirements-video2game.txt   # adds opencv-python-headless
```

## First run

`output/` is generated and git-ignored, so a fresh clone contains no worlds. Nothing here needs one — the sandbox builds an empty world when the file is missing:

```powershell
python main.py
```

Paint with the left mouse button, erase with the right. Press `1` for a human-locked barrier, `2` for fluid, `3` for a solid wall, then `S` to save. You now have a world at `output/worlds/playground/world.bds` that every other tool will accept.

---

# What you can build on it

| Use case | Entry point | One line | Extra deps |
| --- | --- | --- | --- |
| [Chat-driven avatar face](#1-chat-driven-avatar-face) | `avatar_chat_driver.py` | An LLM reply moves a real face's jaw, not its locked eyes | OpenCV (seeding) |
| [Playable game](#2-playable-game) | `game.py` | Rising-tide game whose level is live simulation state | — |
| [Video → world ingestion](#3-video--world-ingestion) | `video2game.py` | A frame of footage becomes terrain and a motion field | OpenCV |
| [AI environment & benchmark](#4-ai-environment-and-benchmark) | `main.py --ai-server` | Gradeable spatial tasks over a validated HTTP API | — |
| [AI-safety demonstration](#5-ai-safety-demonstration) | `main.py --ai-server` | "The human's boundary is inviolable" as executable code | — |
| [Interactive field sandbox](#6-interactive-field-sandbox) | `main.py` | Paint fluid and walls, watch the physics resolve | — |
| [Headless deterministic render](#7-headless-deterministic-render) | `export_video.py` | Reproducible PNG/MP4 from a world plus a command log | ffmpeg (video) |
| [Shared multi-peer world](#8-shared-multi-peer-world) | `main.py --relay` | Peers exchange operations, not pixels | — |
| [GPU conformance testing](#9-gpu-conformance-testing) | `determinism.py` | Does this GPU agree with that one, within tolerance | — |

## 1. Chat-driven avatar face

The clearest demonstration of what the substrate is for: an AI gets real power over a live face, and provably bounded power.

> **Reference implementation, not the product.** The talking face graduated into [**AIFace**](https://github.com/insightitsGit/AIFace), which owns the ongoing work and is well ahead of the modules below. What stays here is the smallest honest proof that the authority rules hold on something legible.

```powershell
python build_avatar_seed.py --synthetic          # deterministic test face, no photo needed
python avatar_chat_driver.py --demo

python build_avatar_seed.py --input path/to/portrait.jpg     # or use a real one
python avatar_chat_driver.py
```

Seeding is required, not optional: the driver refuses to start without `output/worlds/avatar/avatar_face.bds`.

Seeding keeps the photograph's RGB albedo untouched and writes Sobel outlines into channel 24 for rules only. Channel 31 Master-Locks the eyes, nose, skull, cheeks, forehead, jaw, and chin, so identity cannot change; only the mouth cavity and lip interior stay unlocked. It also writes an anatomical part atlas beside the world — `face_parts.npy` plus a `.json` sidecar and a tinted `.png` preview — segmenting the portrait into nine labelled pieces (face, nose, brows, eyes, upper and lower lip, mouth cavity) with the part ID encoded in the alpha channel.

`avatar_chat_driver.py` turns chat replies into phonemes, then into **muscle impulses** through the biomechanical layer (`avatar_biomechanics.py`, `facial_muscles.py`, and the emotion / eye / jaw / breathing / idle / intent modules). Speech never writes geometry directly: jaw physics and muscle activation produce render uniforms plus velocity impulses (`±4`) on unlocked cells, and `avatar.frag` composites the lip, eye, and brow pieces from those uniforms. It builds the part atlas automatically if one is missing.

This app is the **one deliberate exception** to the three-pass tick: it runs the `constraint` pass alone. Advection and diffusion are exactly what you do not want near a face, because moving matter between cells smears a person into a smudge.

Press `H` for lock boundaries and `F1`–`F8` for debug views, `F8` being the colour-coded part atlas — the fastest way to check that the pieces actually landed on the right pixels. `R` reloads the seed.

The sandbox brush and its world controls are deliberately switched off here, and that is worth explaining rather than just listing. Mouse strokes and keyboard paints are *human*-authority writes, and the constraint shader only vetoes *AI* writes on locked cells — human supremacy is the intended rule everywhere else. In an app whose whole claim is "this face cannot change", that rule is a hole: one drag would repaint the eyes straight through the Master Lock, and the sandbox meaning of `R` would leave you talking to an empty grid. So the face is not a canvas, and `R` restores the seed instead of emptying the world. Speech is unaffected: it writes velocity impulses at *AI* authority, which is exactly what the lock is there to bound.

Character muscles live in [`face_definition.json`](face_definition.json). Details: [`docs/AvatarChat.md`](docs/AvatarChat.md).

## 2. Playable game

```powershell
python game.py
```

**Flow Runner.** The level is lifted out of a `.bds` world: every hard-surface cell becomes a wall, everything else is cleared. Steer a magenta avatar with WASD, touch all five rings, and stay ahead of a tide that rises from the floor and covers the world when the clock expires. Three seconds under water ends the run.

Right-drag digs through a wall. That is the move no other engine gives you — the level is live simulation state, not baked geometry, so the hole is real: the collision mask opens with the same cells the GPU cleared, and the tide finds the new tunnel and pours through it.

With no `--world`, it picks up a converted video world if you have one and otherwise falls back to your playground save, so give it something with structure in it:

```powershell
python video2game.py --input clip.mp4 --output output/worlds/from-video/clip_solid-edges_256.bds --preset SOLID_EDGES
python game.py --world output/worlds/from-video/clip_solid-edges_256.bds --targets 7 --run-seconds 140
```

Watch a pathfinding bot play it headlessly, which also verifies that a converted world is actually winnable:

```powershell
python tools/autoplay_demo.py --world output/worlds/from-video/clip_solid-edges_256.bds
```

Design notes: [`docs/FlowRunner.md`](docs/FlowRunner.md).

## 3. Video → world ingestion

```powershell
python video2game.py --input clip.mp4 --output output/worlds/from-video/clip_solid-edges_256.bds --preset SOLID_EDGES
python main.py --world output/worlds/from-video/clip_solid-edges_256.bds
```

Maps optical flow, luminance, RGB, frame activity, and Sobel boundaries onto the 32-channel schema, leaving every human lock clear — only a person may mint a lock. Presets: `FLUID_SANDBOX`, `GRAVITY_REVERSE`, `SOLID_EDGES`.

Source resolution and duration are irrelevant; only two frames are decoded and both are resized. The output grid is what costs:

```powershell
python video2game.py --input clip.mp4 --output output/worlds/from-video/clip_solid-edges_512.bds --width 512 --height 512 --edge-threshold 0.25
python main.py --world output/worlds/from-video/clip_solid-edges_512.bds --world-width 512 --world-height 512
```

Conversion lifts *motion and texture*, not story. There is no object identity, depth, or narrative in the result — it produces terrain, and something else has to make it mean anything. See [`docs/Video2GameDesign.md`](docs/Video2GameDesign.md).

## 4. AI environment and benchmark

Runs are deterministic and malformed commands are rejected rather than silently approximated, so tool-use success *and failure* are both measurable. That is the part most benchmark harnesses cannot give you.

```powershell
python main.py --ai-server
```

The window prints a bridge URL and an access token. Every route except `/` and `/health` requires `Authorization: Bearer <token>`, and the bridge refuses to bind anything but a loopback address — `--ai-host 0.0.0.0` is rejected outright unless you also pass `--allow-remote-bind`, because a reachable port here is world control for anyone who can open it. The relay works the same way.

| Route | Purpose |
| --- | --- |
| `GET /health` | Tick, pause state, queue depth, without touching the GPU |
| `GET /schema` | Command grammar to hand a model, narrowed to the caller's authority |
| `GET /state` | Category counts, field statistics, bounds, occupancy map, storage costs |
| `GET /context` | Bundled AI context JSON |
| `GET /inspect` | Region statistics (`x`, `y`, `radius`) |
| `GET /screenshot` | PNG of the latest composited frame |
| `GET /preview` | PNG rendered at exactly 1024 × 1024, independent of window size |
| `POST /commands` | Submit `{"commands": [...]}` or a named command object |

```powershell
$env:NWR_AI_TOKEN = "<token printed by main.py>"
python ai_agent.py --commands-file examples/arena.json

$env:NWR_LLM_API_KEY = "<your key>"
python ai_agent.py "build a walled reservoir in the lower half" --observe
```

Models never write cell vectors. They emit regions — `point`, `line`, `polyline`, `circle`, `ring`, `rectangle`, `rectangle_outline`, `polygon`, `polygon_outline` — and control actions (`reset`, `pause`, `resume`). `ai_commands.py` validates every field and rasterizes into the same GPU primitive the mouse produces, which is why AI input inherits the same authority rules and the same 64-commands-per-tick budget. Large fills therefore appear progressively over several ticks.

Two things a bridge caller cannot do at the default `ai` authority, because the per-cell write path is not the only way to damage a world:

- **`save` and `load`.** These swap the world through the file system, where no cell-level rule applies. They need `--ai-authority user`. `reset` stays available, and an AI reset carries every human-locked cell across into the fresh world — the boundary survives the thing that used to be the way around it.
- **Painting `human_barrier`.** That category mints a lock on every cell it covers, and minting a boundary its own writes can never cross is a human prerogative. Agents build walls with `solid`, which is a hard surface with no lock.

A refusal a model only discovers by being refused is a bad interface, so nothing above has to be memorised. `GET /schema` is narrowed to the authority the caller was granted: the withheld actions are absent, `human_barrier` is absent from the paintable categories, the worked example uses `solid`, and a `restrictions` block names what was held back and why. The same narrowing reaches `/state`, `/context`, the offline `ai_handoff.py` bundle, and the grammar `ai_agent.py` puts in front of a model — which also validates against the authority the bridge reported rather than assuming its own, so a rejected plan is caught before it is sent. Raise the level with `--ai-authority user` and the full grammar returns.

Useful options: `--dry-run` validates without submitting, `--state` prints the observation, `--screenshot frame.png` saves a frame, and `--base-url` with `--model` targets a local server such as Ollama or LM Studio. Ready-made command files live in [`examples/`](examples/): `arena.json`, `lake_with_walls.json`, and `tennis-racket.json` (generated by `tools/make_tennis_racket.py`).

**Handing a world to ChatGPT or Gemini.** Chat models have no GPU, so a bare `.bds` is an opaque blob. Package one into a labelled filmstrip and GIF, per-sample numbers, the command grammar, and a NumPy-only reader:

```powershell
python ai_handoff.py --world output/worlds/playground/world.bds --output output/handoffs/demo --ticks 600 --frames 9
```

Upload the folder, apply the reply with `python ai_agent.py --commands-file reply.json`, then rebuild for the next turn. The loop stays turn-based on purpose: the model proposes, a human applies, and the same validation and lock enforcement apply either way. See [`docs/AI_API.md`](docs/AI_API.md).

## 5. AI-safety demonstration

Same runtime, different point. Channel 31 is a hard veto enforced in `constraint.comp` and covered by a test that says exactly what it does:

```
test_ai_source_cannot_overwrite_human_locked_cells_even_with_user_priority
```

Lock a region by hand, tell a connected model to destroy it, and the write is dropped by the shader. The avatar face in §1 is this same guarantee applied to something a person can read at a glance: the model animates the jaw and *cannot* touch the eyes. Not because it was asked politely — because the GPU refuses.

The interesting part of a guarantee like this is not the path it covers but the paths around it, and there are four worth naming because each needed its own answer:

- **Minting, not just overwriting.** Blocking AI writes *onto* a locked cell says nothing about an AI *creating* one. `human_barrier` is the one anchor whose channel-31 value is `1.0`, and the constraint pass copies every channel from the anchor — so painting it used to hand an agent a lock. That matters more than a forged provenance record: an AI erase is `±2`, which the same pass refuses on locked cells, so an agent could wall off regions it was permanently barred from clearing, and locked cells are excluded from advection and diffusion. The pass now holds channel 31 down for every AI write, at any authority. An AI caller painting `human_barrier` gets the wall and not the lock.
- **The control plane.** `reset` and `load` replace a world wholesale without ever submitting a cell write, so no shader-level rule sees them. `save` and `load` require human authority, and an AI `reset` carries every locked cell into the new world.
- **Offline replay.** A `.bdl` log replayed by the exporter runs through the same shader, so it is only as safe as the writer identity in the log. Records carry the authority and the writer as separate fields, and neither is ever inferred from the other — except in a `bdl-1.0` log, which did not store a writer at all and is therefore read as best-effort rather than assumed human.
- **Human supremacy itself.** Human writes are *meant* to override locks; that is the point of the hierarchy. It stops being a feature in an app whose premise is that a particular world may not change, which is why the avatar disables the brush (§1) rather than relying on the shader to save it.

Each of these is covered by tests that fail if the hole reopens — and for the two that are enforced on the GPU, by tests that were confirmed to fail with the gate removed.

## 6. Interactive field sandbox

```powershell
python main.py
python main.py --world-width 512 --world-height 384
```

The reference application: a substrate with no objective, which makes it the right place to find out what the rules actually do. Paint fluid, solid walls, and human-locked barriers; watch advection, diffusion, reflection, and constraint snapping resolve at 60 Hz.

- Left drag paints the selected category; right drag or Shift+left erases.
- `1` Human Barrier · `2` Active Fluid · `3` Solid wall
- `N` toggle the neural material network
- `A` Autonomous Swarm AI · `D` Field Director vortex · `C` clear AI fluid
- `E` spawn a blob entity · `X` remove entities
- Space pause · `S` save · `L` load · `R` reset · Escape quit

Barriers are immovable once painted; fluid is injected and then evolves. Erasing returns cells to Vacuum and releases their authority. The swarm agent runs on a background thread and can never overwrite human-locked cells.

## 7. Headless deterministic render

```powershell
python main.py --record output/sessions/demo
python export_video.py --world output/sessions/demo/session.bds --log output/sessions/demo/session.bdl --frames 300 --output output/renders/demo
python export_video.py --frames 600 --video output/renders/out.mp4       # needs ffmpeg on PATH
```

Recording writes `session.bds` plus an append-only `session.bdl` command log. Same seed and same log means the same frames — which is what a render farm needs, and also why the exporter doubles as the determinism harness. When ffmpeg is absent it says so and keeps the frames rather than failing.

For that to hold, a replay has to be indistinguishable from the live run, and three separate things have to be true of the log:

- **It carries who wrote each edit, not just the geometry.** Each record stores the writer's authority *and* whether it was a human or the AI. The shader vetoes AI writes on human-locked cells, so an AI erase that came back out of the log as a human one would apply an edit the live run refused.
- **It holds every operation that reaches the GPU.** Paint, erase, controls, temperature deltas, and velocity impulses. Anything it cannot encode raises at record time instead of being skipped, and the exporter likewise refuses a kind it cannot replay — a log that quietly omits a write turns a replay guarantee into a guess. Entity spawns are the one exception, and deliberately: they are resolved into ordinary segments before anything reaches the GPU, and those segments are what gets logged.
- **It fits the same per-tick budget.** The exporter carries a burst's leftovers onto later ticks rather than dropping them, exactly as the interactive runtime does; if a run ends with writes still queued, the report says how many.

`determinism.py` proves this by writing a real log during a live run and replaying *that file*, then comparing both the final worlds and the records themselves. Its earlier version compared two simulations of the same in-memory schedule, which could not fail — and duly reported "bit-identical" throughout the period when replay was promoting every AI write to human authority. The probe schedule now aims an AI erase whose authority and identity disagree at a human-minted lock, so each way the log could lose a field shows up as a different world.

This path emits images only. Raw `(H, W, 32)` tensor-sequence export for surrogate-model training is **not** implemented ([`docs/UseCases.md`](docs/UseCases.md) §3.2).

## 8. Shared multi-peer world

```powershell
python main.py --relay
```

Peers exchange compiled operations, not pixels, so a shared session costs a few dozen bytes per edit:

```python
from net_relay import RelayClient

peer = RelayClient("127.0.0.1", 8770, token="<token printed by main.py>")
peer.submit({"commands": [
    {"action": "paint", "category": "active_fluid",
     "region": {"type": "circle", "center": [70, 190], "radius": 12}},
]})
```

The relay validates and compiles each request *once*, then broadcasts the result, so peers cannot drift apart by compiling the same request differently.

## 9. GPU conformance testing

```powershell
python determinism.py --ticks 120 --record output/reports/seed-120ticks.json
python determinism.py --ticks 120 --verify output/reports/seed-120ticks.json
```

Fingerprints the device, hashes worlds tick by tick, and checks one machine against another within tolerance. Entirely independent of subject matter: a general harness for "does this GPU agree with that one."

---

# Anatomy of a world

## The 32 channels

Four groups of eight. Four slots are reserved placeholders, which is why the arithmetic closes at 32 with only 28 named channels.

| Group | Channels | Contents |
| --- | --- | --- |
| Kinematics | 0–7 | velocity x/y/z, density, pressure, shear, temperature, energy |
| Material | 8–15 | albedo r/g/b, opacity, roughness, metallic, emission, refraction |
| Intent | 16–23 | attraction, alignment, user affinity, growth, decay, lifespan, *reserved ×2* |
| Rules | 24–31 | hard surface, permeability, thermal threshold, phase trigger, *reserved ×2*, **authority priority (30)**, **human lock (31)** |

Channels 30 and 31 are authority metadata, not material state. Only commands write them, and they are excluded from diffusion, clamping, and anchor snapping. That exclusion is load-bearing: a cell whose rules group happened to snap to the barrier anchor would otherwise acquire a permanent lock nobody asked for.

`bds_format.py` is the single source of truth. `shader_library.py` generates the layout and the anchor codebook into GLSL constants, injected at every `//#prelude` token, so the CPU and GPU views of a cell **cannot** drift apart.

## The anchor codebook

Four full 32-float prototype cells. Their order is the GPU-side anchor code.

| Anchor | Character |
| --- | --- |
| `vacuum` | All zeros; the erase target |
| `human_barrier` | Magenta, opaque, hard surface, **locked** |
| `active_fluid` | Blue, dense, permeable, unlocked — injected, then left to evolve |
| `solid` | Grey, hard surface, **no lock** — a wall that deflects flow without claiming human authority |

That last distinction is subtle and intentional. A solid wall stops fluid; it does not pretend a human drew it. Only human input mints a lock.

## Every write is one of four operations

| Op | Meaning |
| --- | --- |
| `±1` | Human paint / erase |
| `±2` | AI paint / erase |
| `±3` | Temperature delta |
| `±4` | Velocity impulse `(Vx, Vy)` |

A command is eight floats — `x0, y0, x1, y1, radius, category, operation, priority` — and the shader measures each cell centre's distance from that segment, so strokes stay continuous even when mouse events arrive sparsely. Up to 64 land per tick; the rest stay queued.

## The tick

Three compute dispatches at a fixed 60 Hz, ping-ponging two SSBOs so no pass reads a buffer it is writing. Workgroups are 16 × 16.

1. **`physics.comp`** — semi-Lagrangian advection of channels 0–23, velocity damping, wall reflection. Rules and authority never move.
2. **`semantic.comp`** — diffuses material and intent, decays intent. Skips locked cells and hard surfaces.
3. **`constraint.comp`** — clamps, snaps to anchors, releases the authority of emptied cells, then applies commands. **The only pass that sees the command buffer**, which is what keeps authority resolution in exactly one place.

At most five simulation steps run per rendered frame, and excess accumulated time is discarded rather than triggering a catch-up spiral. `avatar_chat_driver.py` is the sole exception to the three-pass rule (§1).

## Writing your own use case

The integration contract is one function call. Anything you can express as an `(H, W, 32)` float32 array becomes a world:

```python
import numpy as np
from bds_format import VECTOR_DIMENSIONS, save_bds

field = np.load("my_sensor_grid.npy")            # any (256, 256) float array

grid = np.zeros((256, 256, VECTOR_DIMENSIONS), dtype="<f4")
grid[..., 3] = field                             # channel 3  = density
grid[..., 24] = (field > 0.8).astype("<f4")      # channel 24 = hard surface
grid[..., 31] = 0.0                              # channel 31 = human lock

save_bds("output/worlds/mine.bds", grid, metadata={"source": "my_sensor"})
```

Thermal imagery, weather rasters, satellite bands, microscopy stacks, medical slices, sensor grids. A new adapter is a script, not a subsystem change — `video2game.py` and `build_avatar_seed.py` are both just this.

## Output layout

Every generated artifact goes under [`output/`](output/README.md), which is git-ignored. `paths.py` is the single source of these locations; import from it rather than hard-coding strings.

| Folder | Contents |
| --- | --- |
| `output/worlds/avatar/` | Face seeds and part atlases from `build_avatar_seed.py` |
| `output/worlds/from-video/` | `.bds` snapshots from `video2game.py` |
| `output/worlds/playground/` | Interactive saves from `main.py` |
| `output/previews/` | Smoke-test and demo PNG frames |
| `output/handoffs/` | ChatGPT / Gemini upload bundles |
| `output/renders/` | Frame sequences and video from `export_video.py` |
| `output/sessions/` | Recorded `.bds` + `.bdl` pairs |
| `output/reports/` | Determinism probes and measurement JSON |

## Material network

Ships with trained weights in `material_weights.npy`: a `32 → 16 → 3` MLP that shades each fragment from a weight texture. Enable with `--neural-material` or press `N`. To retrain:

```powershell
python material_network.py --epochs 700 --samples 120000
```

Training distills the procedural shading response and reports train/validation MSE plus worst-case error, which makes the result measurable rather than decorative. Vacuum accuracy matters more than average error — it is the most common cell, and any bias there lifts the rendered black level to grey — so training deliberately includes exact, noise-free anchor samples.

## Tests

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest
```

Use `python -m pytest`, not bare `pytest`: there is no `conftest.py` or packaging config, so `-m` is what puts the repository root on `sys.path`.

How many tests you see depends on what is installed, so `python -m pytest --collect-only -q` is the honest answer for your machine rather than a number printed here. A full environment collects a little under five hundred; without OpenCV the video and face modules skip at import and are not collected at all, which is why a partial environment reports a smaller total rather than a pile of failures.

| Needs | Skips without it |
| --- | --- |
| **Nothing** | Format, history, codec, chunking, paths, shader-library, command-compiler, relay, bind policy, entities, swarm, biomechanics, and material tests all run on CPU alone |
| **OpenGL 4.3** | `test_gpu_pipeline.py`, `test_app_runtime.py`, `test_game.py`, `test_export_video.py`, `test_determinism.py`, `test_ai_handoff.py`, `test_avatar_hardening.py` |
| **OpenCV** | `test_video2game.py`, `test_avatar_seed.py`, `test_face_parts.py` — install `requirements-video2game.txt` to run these |

## Layout

| Path | Role |
| --- | --- |
| `bds_format.py` | Cell schema, anchors, `.bds` read/write |
| `bds_history.py` | Append-only `.bdl` command log and replay |
| `bds_codec.py` | Lossless and anchor/residual codecs |
| `bds_chunks.py` | Chunked world with activity tracking and paging |
| `paths.py` | Canonical output locations; the source of the table above |
| `shaders/` | GLSL stages; `//#prelude` is expanded at load time |
| `shader_library.py` | Generates GLSL constants from the Python schema |
| `main.py` | Reference sandbox application and window runtime |
| `game.py` | Flow Runner: a playable game over a `.bds` world |
| `face_definition.json` | Character muscle map, anchors, phoneme→muscle table |
| `facial_muscles.py` | Muscle registry, impulse queue, spring-damper solver |
| `avatar_biomechanics.py` | Orchestrates emotion / eye / jaw / breath / idle / intent |
| `emotion_system.py` · `eye_system.py` · `jaw_system.py` | Continuous biomechanical subsystems |
| `breathing_system.py` · `idle_system.py` · `intent_system.py` | Ambient motion and the LLM intent bridge |
| `build_avatar_seed.py` | Face image → locked avatar `.bds` seed + part atlas |
| `face_parts.py` | Segments a portrait into nine labelled anatomical pieces |
| `avatar_chat_driver.py` | Chat window driving the biomechanical face |
| `video2game.py` | Optional video-to-`.bds` ingestion pipeline |
| `ai_commands.py` | Command grammar, validation, rasterization |
| `ai_command_compiler.py` | Named AI commands, CRC sealing, temperature deltas |
| `ai_world.py` | `generate_ai_summary`, `inspect_region`, `export_ai_context` |
| `ai_bridge.py` | Loopback HTTP control bridge and observation |
| `ai_agent.py` | Client for OpenAI-compatible models |
| `ai_handoff.py` | Bundles a world into artifacts a chat model can read |
| `swarm_agent.py` | Background Autonomous Swarm / Field Director agent |
| `entities.py` | Named, tracked entities layered over the field |
| `net_relay.py` | Operation-stream relay for shared worlds |
| `determinism.py` | Device fingerprinting and cross-GPU verification |
| `material_network.py` | Trains and exports the material MLP |
| `export_video.py` | Headless deterministic frame and video export |
| `tools/autoplay_demo.py` | Headless pathfinding bot that plays and captures a run |
| `tools/make_tennis_racket.py` | Generates the `examples/tennis-racket.json` command file |
| `examples/` | Ready-made AI command files for `ai_agent.py --commands-file` |

## Documentation

Start at [`docs/README.md`](docs/README.md) for the index. The ones that matter most:

- [`docs/FinalDesign.md`](docs/FinalDesign.md) — the binding specification. Authoritative when documents disagree.
- [`docs/UseCases.md`](docs/UseCases.md) — what works today, what is adjacent with named gaps, what is out of scope.
- [`docs/AI_API.md`](docs/AI_API.md) — how assistants inspect and command a world.
- [`docs/AvatarChat.md`](docs/AvatarChat.md) — the biomechanical face (reference implementation; see [AIFace](https://github.com/insightitsGit/AIFace)).
- [`docs/FlowRunner.md`](docs/FlowRunner.md) — the game layer.
- [`docs/Video2GameDesign.md`](docs/Video2GameDesign.md) — video ingestion, with measured size and semantic limits.

## What this is not

Stated plainly, so the boundary is not mistaken for a roadmap. The simulation is a **coupled field model, not Navier–Stokes** — plausible, not calibrated to physical constants. There is no 3D volumetric field (the substrate is 2D × 32 channels), no rigid-body dynamics with mass and torque, no scene understanding or object recognition, and no audio. Chunking governs storage and paging, not VRAM residency: the whole world stays resident, which caps practical size near the shader-storage ceiling.
