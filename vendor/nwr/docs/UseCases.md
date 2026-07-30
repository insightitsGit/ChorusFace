# Use Cases

Status: capability map  
Scope: what the current runtime supports, and what each further use case needs

## 1. What the product actually is

Video-to-game ingestion is one *producer* of worlds, not the product. Strip the
framing away and the runtime is:

> A deterministic, GPU-resident field simulator over a 32-channel dense grid,
> with a replayable command log and a machine-readable control and observation
> API.

Every use case below is a different answer to two questions: *what writes the
channels* and *who reads the results*. Nothing else changes.

The five reusable primitives:

| Primitive | Module | What it gives a use case |
| --- | --- | --- |
| 32-channel field substrate | `bds_format.py`, `shaders/` | Dense state evolving on the GPU with no CPU in the loop |
| Deterministic snapshot + log | `bds_format.py`, `bds_history.py`, `determinism.py` | Bit-exact reproduction and verifiable replay |
| Priority-resolved writes | `shaders/constraint.comp` | Human > constraint > AI > background, enforced on the GPU |
| Machine-readable control API | `ai_bridge.py`, `ai_commands.py`, `ai_world.py` | Observe, inspect, and command over HTTP with validated JSON |
| Ingestion adapters | `video2game.py`, `build_avatar_seed.py` | Any external signal becomes a starting world |

## 2. Supported today

These run on the shipped code with no new modules.

### 2.1 Interactive field sandbox (the reference application)

`python main.py`. Paint fluid, solid walls, and human-locked barriers; watch
advection, diffusion, reflection, and constraint snapping resolve at 60 Hz.

### 2.2 Reinforcement-learning and agent environment

The strongest non-video fit, because it needs exactly what the substrate
guarantees. `ai_bridge.py` exposes observation, action, and reset over HTTP;
`determinism.py` proves that identical seeds and command logs reproduce
identical worlds, so runs are comparable. Rewards come from
`world.inspect_region()`. Distinct from typical RL environments in that the
state is a dense continuous field rather than sprites or a physics scene graph.

### 2.3 LLM tool-use and spatial-reasoning benchmark

`docs/AI_API.md` plus `render_preview()` (exact 1024 × 1024) gives a model both
a JSON summary and an image of the same world. Tasks like "wall off the leak,"
"cool the hot region," or "route fluid to the corner" are gradeable from
`generate_ai_summary()`. Malformed commands are rejected rather than silently
approximated, so tool-use failures are measurable instead of hidden.

### 2.4 Human-authority and AI-safety demonstration

Channel 31 is a hard, GPU-level veto: an AI-sourced write aborts on a
human-locked cell regardless of the priority it claims (`constraint.comp`, and
`test_ai_source_cannot_overwrite_human_locked_cells_even_with_user_priority`).
This makes "the human's boundary is inviolable" an executable property, not a
policy document — useful for oversight and containment demos.

### 2.5 Emergent-behaviour and swarm research

`swarm_agent.py` reads a downsampled grid on a background thread and issues
pathing or vortex commands without touching the render loop. Swap the decision
function to study a different collective policy; the perception, command
validation, and lock enforcement are already in place.

### 2.6 Generative art and procedural motion design

`export_video.py` renders deterministic frame sequences headlessly, and
`material_network.py` shades cells through a trained network instead of fixed
anchor colours. Same seed, same output, which is what a render farm needs.

### 2.7 Reproducibility and GPU-conformance testing

`determinism.py` fingerprints the device, hashes worlds tick by tick, and
verifies one machine against another within tolerance. Independent of the
simulation's subject matter: it is a general harness for "does this GPU agree
with that one."

### 2.8 Shared-world networking

`net_relay.py` streams validated operations between peers, replaying them
through the same priority resolution. Multiple clients converge because the
substrate is deterministic and the log is ordered.

### 2.9 Ingestion of any 2D signal

`video2game.py` is one adapter, but the substrate accepts any source that can be
written into a `(H, W, 32)` array — thermal imagery, weather rasters, satellite
bands, microscopy stacks, medical slices, simulation dumps, sensor grids.
Writing a `.bds` with `save_bds()` is the entire integration contract, so a new
adapter is a script, not a subsystem change.

### 2.10 Playable games

`python game.py`. Flow Runner turns any `.bds` world into a level: the source
world's hard surfaces become walls, five rings are placed where a player-sized
disc can actually reach them, and a tide rises from the floor until it covers
everything. You steer an avatar with WASD, and you can dig through a wall
mid-run because the level is live simulation state rather than baked geometry.

This is the use case that was listed as a gap until the win and lose layer
existed. The substrate supplied the physics, the walls, the deterministic
timestep, and the validated write path; the game added objectives, a controlled
player, and an end condition. See [`FlowRunner.md`](FlowRunner.md).

Still missing for a *general* game engine: a camera, sprites, audio, and any
notion of a scene graph. Flow Runner is one game, not a framework for making
them.

### 2.11 Bounded-authority biomechanical character

**Graduated.** This use case became the first child product,
[AIFace](https://github.com/insightitsGit/AIFace), and that is where the work
continues. The child replaced rigid cut-out pieces with a continuous muscle
displacement field, grew to ~39 muscles with groups and asymmetry, and added an
in-window chat frame and its own control bridge. It vendors a minimal copy of
the substrate rather than importing this package, so it deploys alone while
still honouring the schema, the Master Lock, and the constraint-only tick
defined here.

What remains below is the reference implementation: the smallest honest proof
that GPU-enforced authority holds on something a person can read at a glance.
Treat it as a demonstration, not a roadmap.

`python build_avatar_seed.py --input face.png` then `python avatar_chat_driver.py`.
A face image becomes a world whose eyes, nose, skull, cheeks, forehead, jaw, and
chin carry the Master Lock, while only the mouth cavity and lip interior stay
unlocked. Chat text becomes phonemes, phonemes become **muscle impulses**, and a
deterministic biomechanical layer (`facial_muscles.py`, emotion/eye/jaw/
breathing/idle/intent systems) integrates those impulses into continuous state.
The renderer visualises that state; speech never writes pixels directly.

Character swap is data-only via `face_definition.json`. Debug views `F1`–`F8`
expose density, velocity, muscle activation, emotion, jaw, locks, impulse heat,
and influence wireframes. See [`AvatarChat.md`](AvatarChat.md).

Still missing for production avatars: audio synthesis/playback, landmark-accurate
face detection beyond Haar+fallback, and per-muscle GPU residency outside the
unlocked mouth disc (cheek/brow muscles currently drive render uniforms and
CPU state rather than locked field cells).

## 3. Adjacent, with known gaps

Honest assessment: each of these is reachable but needs specific work. None is
claimed as working today.

### 3.1 Fluid and diffusion teaching tool

Have: visible advection, diffusion, and reflection with direct manipulation.
Need: units, labelled axes, and a per-channel readout. The current rules are
plausible, not calibrated to real physical constants.

### 3.2 Predictive world modelling / neural surrogate training

Have: a determinism-verified generator of `(H, W, 32)` state sequences, which is
ideal training data.
Need: a dataset export path and a train/eval split. `export_video.py` writes
images; tensor sequence dumps are not implemented.

### 3.3 Robotics occupancy and path planning

Have: hard-surface barriers, `inspect_region()`, deterministic replay.
Need: a metric coordinate frame, sensor ingestion, and a planner. The grid is
currently unitless.

### 3.4 Large-world streaming

Have: `bds_chunks.py` paging and storage reporting.
Need: chunk-level GPU residency. The runtime uploads the whole world, which caps
practical size near the shader-storage limit (see
`docs/Video2GameDesign.md` §6.2).

### 3.5 Multi-frame / temporal video ingestion

Have: single-frame seeding with a two-frame motion estimate.
Need: a temporal driver that keeps injecting flow as the simulation runs. This
is the gap that would let a converted world follow the source footage instead of
departing from it after tick zero.

## 4. Explicitly out of scope

Stated so the boundary is not mistaken for a roadmap: 3D volumetric fields
(the substrate is 2D × 32 channels), rigid-body dynamics with mass and torque
(entities advect, they do not collide as bodies), scene understanding or object
recognition, and audio.

## 5. Choosing a direction

The substrate's differentiators are **determinism you can verify** and
**GPU-enforced authority ordering**. Use cases that lean on those — RL
environments, AI-safety demonstrations, LLM benchmarks, conformance testing —
are stronger fits than ones that mainly need rendering or gameplay, where
mature engines already win.
