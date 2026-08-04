# Architecture

Amin walkthrough (design + step implementation): [`AMIN_DESIGN.md`](AMIN_DESIGN.md) ·
Doc index: [`README.md`](README.md)

New contracts: [`DisplayLayers.md`](DisplayLayers.md) ·
[`AvatarAdoption.md`](AvatarAdoption.md) · [`AvatarBehavior.md`](AvatarBehavior.md) ·
[`MouthCellGroups.md`](MouthCellGroups.md)

ChorusFace is three layers with one hard rule between them: **nothing below the
speech layer is allowed to change who the face is.**

```
chat text                         someone else's audio + transcript
    │                                          │
    ▼                                          ▼
chorusface.speech          pure Python — visemes, mouth poses, LLM client
    │                                          │
    │                             chorusface.stream — locks visemes to arriving PCM
    │                                          │
    ▼◄─────────────────────────────────────────┘
chorusface.biomechanics    pure Python — muscles, jaw, eyes, breathing, emotion
    │
    ├── render state (jaw angle, lid closure, gaze, brow raise, mouth pose)
    ├── MouthCellPlan / mouth_groups (L03 cell→neighbor ±4)
    ├── BehaviorDriver (measured track → ML fill → table)
    └── field impulse specs (velocity, radius, priority)
    │
    ▼
chorusface.runtime         GPU — constraint tick, Master Lock, continuous deform
    │                  avatar.frag composites L00–L11 (display_layers)
    ▼
a face   (any world opened via avatar_profile.open_avatar)
```

Only the bottom layer needs a GPU. The two above it are ordinary, seeded,
side-effect-free Python, which is why the speech and simulation tests run
everywhere and the rendering tests skip cleanly without a driver.

Speech carries a rolling `ConversationSession` so emotion and topic survive
across turns. An optional loopback `FaceBridge` (`chorusface --bridge`) exposes
NWR-style `/speak`, `/preview`, `/status` jobs drained on the render thread.

## Three clocks, one mouth

Viseme times can come from three places, and it is worth being explicit about
which is which, because they carry very different claims:

| Source | Timing from | Status |
| --- | --- | --- |
| `chorusface.stream` | Audio arriving from another voice, chunk by chunk | The primary path |
| `chorusface.tts` | A clip this process synthesised and holds in full | Fixture, and the oracle's yardstick |
| `chorusface.speech` | Nominal phoneme durations — a letter count | Fallback when there is no audio at all |

The streaming path is the product; the batch path exists so the streaming path
can be measured against something. `chorusface.sync` runs an utterance through both
and reports the difference in milliseconds — see
[Voice Sync](VoiceSync.md).

The voice routes are the one deliberate exception to "all external work is a job
drained on the render thread". Audio arrives in real time, and making a 20 ms
chunk wait for a frame would tie the audio clock to the frame rate. Alignment
touches no GPU state, so those calls run on the request thread and hand the
render loop nothing but timed viseme events.

## The substrate

A world is a `256 × 256 × 32` float32 grid stored in a `.bds` file: a 4 KiB
JSON header (schema, CRC32, application metadata) followed by the raw payload.
The 32 channels are grouped in fours of eight — kinematics, material, intent,
rules — and defined in exactly one place, `chorusface/runtime/bds.py`.

Channels this project actually cares about:

| Channel | Name | Role |
| --- | --- | --- |
| 0, 1 | `velocity_x`, `velocity_y` | What a speech impulse adds |
| 3 | `density` | Soft tissue the impulse has to push |
| 8–10 | `albedo_r/g/b` | The photograph; never written after seeding |
| 24 | `hard_surface` | Structural contours from the Sobel pass |
| 30 | `authority_priority` | Who last claimed the cell |
| 31 | `human_lock` | **Master Lock** — identity, enforced on the GPU |

`chorusface/runtime/shaders.py` generates a GLSL prelude from that same Python
schema and injects it into every stage, so the CPU and GPU views of a cell
cannot drift. `tests/test_shader_contract.py` asserts the two stay equal.

## The tick

`FieldRuntime` steps at a fixed 60 Hz with a maximum of five catch-up steps per
frame, so a stalled frame cannot fast-forward the simulation. Each tick:

1. Pop up to 64 commands due for the next tick. AI packets sort first so a
   human write wins any tie. Overflow stays queued for the following tick rather
   than being truncated, and anything the queue does have to shed is counted and
   shown on the HUD.
2. Write them as eight-float rows into a bound SSBO.
3. Run the constraint compute pass across two ping-ponged world buffers.
4. Render the front buffer through `avatar.frag` into an offscreen target, draw
   the HUD, and blit to the window.

Speech energy has to leave the field as well as enter it. With no physics pass to
dissipate it, the constraint stage relaxes tissue velocity in every unlocked cell
by `VELOCITY_DAMPING` per tick and lands exactly on rest below `EMPTY_EPSILON`.
Without that, impulses only ever accumulate: the field pins itself at the clamp
and the velocity read-outs stop describing anything. Velocity is also excluded
from the anchor-quantisation trigger, because how hard a cell was just pushed must
not decide what the cell *is*.

Command rows encode their kind in the operation magnitude:

| Operation | Meaning |
| --- | --- |
| `±1` | Human paint / erase |
| `±2` | AI paint / erase |
| `±3` | Temperature delta |
| `±4` | Velocity impulse — `(V_x, V_y)` in a disc around `segment.xy` |

The avatar only ever emits `±4`. Per-cell / neighbor / cluster drives share the
same command budget (see `chorusface.cell_cluster`, bridge `/cells/drive`).

### Display stack, adoption, behavior

| Contract | Role |
| --- | --- |
| `display_layers` L00–L11 | Ordered field → look → presentation; skip idle work per tick |
| `avatar_profile` | Any qualifying world dir → same load path |
| `behavior` | Measured group transitions from upload; ML fills gaps; retrain on new video |
| `mouth_groups` + `mouth_cell_plan` | Word-timed L03 drives into unlocked mouth cells |

Existing live-vector + plate recipe remain authority when present; behavior /
cell plan overlay gaps and per-cell motion without inventing face RGB.

### What is deliberately missing

A general field runtime would also run physics and semantic advection passes.
ChorusFace ships neither. Advection moves matter between cells, and a face made of
moved matter is a smeared face. The tick is constraint-only, which is what makes
`test_the_photograph_channels_never_mutate` pass after any amount of speech.

Also absent, by design: entities, swarm planning, networking, HTTP control
surfaces, and video ingestion.

## The authority model

Two independent guarantees, both enforced below the level that speech can reach:

**The Master Lock.** `build_region_masks` sets channel 31 over the skull,
cheeks, forehead, eyes, and nose. In `constraint.comp`, any command with an AI
operation magnitude aborts on a cell where `human_lock >= 0.5`. AI paint may
still place barrier material on an unlocked cell, but it cannot *raise* channel
31 — that mint is a human act (NWR `f3ce4f5`). Only the mouth cavity and lip
interior are left unlocked. A malformed impulse aimed straight at an eye is a
no-op — see `TestLockAuthority` and `TestOnlyAHumanMintsALock` in
`tests/test_runtime.py`.

**The immutable photograph.** The renderer never samples the mutating field for
colour. It samples a texture uploaded once from `source_face.png` (or the part
atlas built from it). Visible lip, eye, and brow motion is inverse-mapped piece
compositing in the fragment shader, not field motion.

Priority is the third, softer guard: each command carries a normalised
`authority_priority`, and a lower-priority write cannot overwrite a cell claimed
at a higher one. Writer and authority travel together by construction —
`PaintCommand` refuses to build an AI-sourced row claiming human authority, since
a mislabelled row is how a lock stops meaning anything on paths that rebuild
commands rather than emit them live.

**The control surface.** `FaceBridge` binds loopback unless `--allow-remote-bind`
says otherwise, and owns no route that resets, saves, or loads a world. A token
holder can make the face speak; it cannot overwrite who the face is. A `.bds`
saved from a live session is written as a rest pose, so persistence cannot smuggle
mid-word state into an identity artifact either.

**Product beta.** Host products own the LLM and POST assistant text to
`/speak` (`--bridge-direct-speak`). See [`ProductBeta.md`](ProductBeta.md).

## Portrait identity (Path 1)

A seed is one colocated bundle: `.bds` + `source_face.png` + tissue maps +
(diagnostic) part atlas. `chorusface-seed --input` face-normalizes the photo,
measures eyes/mouth (`chorusface.landmarks`), and stores those centres in
`application_metadata.avatar_seed.landmarks`. Tissue bake and lid uniforms
prefer those measurements over definition defaults — that is what keeps blink
and gaze on a real portrait.

`--face-image` alone is not a face swap: it only changes the sampled photo.
Always reseed to replace identity.

## Speech motion (mouth)

Visemes inject **muscle impulses + a jaw target**. The fragment shader inverse-
warps the photograph through the summed displacement field and drops the jaw as
bone-like occlusion. A parted lip line may show a **soft photo shadow** in the
gap — never painted teeth. When `chorusface-capture` has written `open.png` /
`smile.png`, those **real plates** are composited inside `mouth_gap`. MouthPose
uniforms are telemetry / debug, not a second deform path.

See [AvatarCapture.md](AvatarCapture.md).

## The part atlas

A frontal face box is split into labelled anatomical pieces for **F8 debug and
legacy anchors**. Visible speech motion is the continuous displacement field in
`avatar.frag`, not rigid piece sliding. The atlas RGB still carries the photo;
alpha is `part_id / 10` and is sampled with `NEAREST` so IDs do not blend.

## The artifact contract

Core Path 1 files live in one directory; the renderer resolves companions by
name from the `.bds`:

```
avatar_face.bds     the locked field seed, with face box + mouth centre metadata
face_parts.npy      the RGBA part atlas
face_parts.json     part anchors (used to register lip and eye pieces) and counts
source_face.png     the immutable render portrait (rest identity)
smile.png           optional real smile plate (chorusface-capture)
open.png            optional real open-mouth plate (chorusface-capture)
capture_meta.json   optional capture digest / travel priors
```

`chorusface-seed` writes the core four from the same resized pixels.
`chorusface-capture` seeds from a rest frame and adds smile/open plates + priors.
`face_parts.png` is written alongside as a colourised preview for eyeballing
the split.

Saving from the running app merges the existing `application_metadata` forward.
That is not incidental: dropping `avatar_seed.mouth_center_image` would leave a
reloaded world unable to register the lip pieces, and the mouth would drift off
the photograph. `TestPersistence` guards it.

## Determinism

Every stochastic subsystem is a seeded LCG — eye microsaccades, blink intervals,
idle micro-behaviour. Two `BiomechanicalFace` instances built with the same seed
produce identical traces for as long as you care to run them, which is what
makes the simulation testable at all.
