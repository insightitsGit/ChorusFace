# BDS motion map — tools, ownership, and what signals change

Investigation artifact after reviewing NWR design docs
(`C:/code/NWR/docs/FinalDesign.md`, `AI_API.md`, README) and probing the
ChorusFace avatar with phoneme / emotion / intent signals
(`scripts/probe_bds_motion.py` → `output/worlds/avatar/bds_motion_probe.json`).

Related design: [`DisplayLayers.md`](DisplayLayers.md) (L01 field under plates) ·
[`MouthCellGroups.md`](MouthCellGroups.md) · [`AvatarBehavior.md`](AvatarBehavior.md)

## 1. Positioning (NWR parent vs ChorusFace child)

| Rule (from NWR) | ChorusFace stance |
| --- | --- |
| 32-channel `.bds`, Master Lock ch.31 | Inherited; do not redefine |
| AI proposes commands; GPU validates | Speech → muscle impulses → `±4` velocity on unlocked cells |
| Human authority wins; AI cannot mint locks | Same |
| Full NWR tick = physics + semantic + constraint | **Constraint-only** — no advection (would smear the face) |
| Rendering may use a trained material MLP | ChorusFace warps an **immutable photograph** instead |

NWR purpose quote (FinalDesign §1): prove a GPU-resident field with authority,
and that *rendering can be driven by a trained network*. ChorusFace uses the
authority substrate but **does not** regenerate face RGB from an MLP each frame.
Identity is the seeded albedo + Master Lock.

## 2. What each channel means for a face

| Channels | Name | Face role |
| --- | --- | --- |
| 0–2 | velocity | Speech may write; damped (~0.88/tick); **not** the main visible lip warp today |
| 3 | density | Soft tissue mass in unlocked mouth |
| 8–10 | albedo RGB | **Photograph identity — never rewritten by speech** |
| 11 | opacity | Presence |
| 16–23 | intent | Unused for mouth drive today |
| 24 | hard_surface | Contours / lip rim structure |
| 25 | permeability | Soft vs hard compliance |
| 30 | authority_priority | Who last wrote |
| 31 | human_lock | Master Lock (≥0.5 = immovable identity) |

Probe on current seed (`avatar_face.bds`): velocity at rest = 0; albedo and
lock fingerprints stable across all signal probes.

## 3. Two motion paths (the dual stack)

```text
chat / audio / bridge
        │
        ▼
  viseme + emotion + intent     (chorusface.speech / biomechanics)
        │
        ├──────────────────────────────┐
        ▼                              ▼
 muscle activations + jaw          FieldImpulseSpec (±4)
        │                              │
        ▼                              ▼
 shader inverse-warp               BDS unlocked velocity
 (VISIBLE lip shape)               (gated, damped; underused)
        │
        ▼
  photo sample (albedo unchanged)
```

**Colors (albedo):** change only if you reseed / swap `source_face.png`. Speech
does not recolor the face.

**Shapes (what you see speaking):** almost entirely **muscle displacement + jaw
profile** in `avatar.frag`, plus optional gated plates. BDS velocity is
telemetry / secondary energy today.

## 4. Probe results (signals → what moved)

Offline biomechanics on the current face definition (Normal path):

| Signal | Jaw | Open | Width | Expr | Eye widen | Brow | Field impulses |
| --- | --- | --- | --- | --- | --- | --- | --- |
| REST/NEUTRAL | 0 | ~1 | ~14 | 0 | 0 | 0 | 0 |
| PP/NEUTRAL | 0 | ↑ | narrower | 0 | 0 | 0 | 4 |
| OU/NEUTRAL | mid | mid | narrow/round | 0 | 0 | 0 | 7 |
| AA/AH/NEUTRAL | high | high | ~14 | ≤0 | 0 | 0 | 7 |
| EE/NEUTRAL | low | low | **wide** | ~0 | 0 | 0 | 7 |
| AH/HAPPY | high | high | wider | + | some | some | 7 |
| AH/SURPRISED | high | high | ~14 | ≤0 | **1.0** | **1.0** | 7 |
| PP/SAD | 0 | low | narrow | − | 0 | 0 | 4 |

Intent JSON `{emotion: surprise, speech: AH}` raised brow/widen immediately;
jaw still comes from the phoneme jaw target path.

**Invariant confirmed:** albedo + Master Lock map unchanged after every probe.

## 5. What we underuse (vs NWR design power)

1. **Velocity → visible warp** — improved: per-cell ±4 + velocity neighbor blend
2. **Spatial writers** — fixed: muscle anchors kept; speech spreads across nearby
   mouth cluster cells (legacy single-disc remap is opt-in only)
3. **Intent / density / hard_surface** as contact & seal, not just debug  
4. **NWR observation loop** — filmstrip / timeline / schema for an external model  
5. **Trained render MLP** (NWR) — intentionally *not* used for face identity here  

### Per-cell / neighbor control (runtime)

| Surface | Role |
| --- | --- |
| `chorusface.cell_cluster` | Full mouth cell index from `.bds`; cell / cluster / neighbor drives |
| `GET /cells` | List controllable regions + cell counts |
| `POST /cells/drive` | `{mode:cell\|cluster\|neighbor\|batch, ...}` → AI ±4 at radius 0.5 |
| `constraint.comp` | Unlocked soft cells exchange **velocity only** with Moore neighbors |
| Master Lock | Still rejects AI on ch31≥0.5; albedo never written by cell drive |

Command budget raised to **256 / tick** so a sweep can address many cells per frame.

## 6. Where ML fits (without breaking positioning)

Aligned with NWR’s “AI proposes, runtime validates” and ChorusFace identity rules:

```text
video take  →  labeled control trajectories
audio/text  →  ML model  →  control vector
                              │
                              ▼
              jaw, muscle drives, brow, widen, plate role
                              │
                              ▼
              existing BDS + muscle warp (+ gated plates)
```

| ML outputs | OK? |
| --- | --- |
| Control vector (muscles, jaw, emotion axes, catalog role) | **Yes** |
| New face RGB / latent morph each frame | **No** — breaks Master Lock story |
| Optical-flow terrain write into albedo | **No** — that is NWR video2game, not Path 1 face |

Training data = capture video + current probe metrics (viseme timing, jaw,
muscle activations). The model replaces hand-tuned `phoneme_muscles` tables;
it does not replace `.bds` authority.

## 7. How to re-run the probe

```bash
python scripts/probe_bds_motion.py
# writes output/worlds/avatar/bds_motion_probe.json
```

Live bridge (optional): `chorusface --demo --tts --bridge` then `POST /speak` with
bearer token — same speech path, GPU-visible.

## 8. Mouth ownership status (NWR-first — no Path A seals)

Single status module: [`chorusface/mouth_owner.py`](../src/chorusface/mouth_owner.py).
Per [`AMIN_DESIGN.md`](AMIN_DESIGN.md) this is **reporting only**: it never
blocks jaw, muscle warp, or field velocity. Master Lock (ch 31) on the GPU is
the only hard reject, and field writes always propose ±4 impulses.

| Status flag | True when |
| --- | --- |
| `muscle_warp` + `jaw` + `field_velocity` | Always (never sealed) |
| `plate_atlas` | Eased openness above the fade-in floor |
| `smile_plate` | Emotion `HAPPY` (live `width_n` can still paint smile.png) |
| `upper_expr_plate` | `SURPRISED` / surprise blend (only flag that gates a blend) |
| `dark_cavity` | **Never** |

### Live GPU probe

With the demo bridge on:

```bash
chorusface --demo --tts --bridge --world output/worlds/avatar/avatar_face.bds
# note the printed Bearer token
curl -s -H "Authorization: Bearer TOKEN" http://127.0.0.1:8766/probe
```

Or: `python scripts/probe_mouth_live.py` (offline ownership table; hits `/probe`
when `CHORUSFACE_BRIDGE_TOKEN` is set).

`GET /status` also includes `mouth_owners` / `mouth_ownership`.

## 9. Honest status: do we have full understanding?

| Area | Status |
| --- | --- |
| Schema + Master Lock (NWR docs) | **Yes** |
| ChorusFace constraint-only vs full physics | **Yes** |
| What speech is allowed to write | **Yes** (velocity on unlocked cells) |
| What actually moves pixels today | **Yes** (muscle warp + jaw + NWR field velocity warp; plates on top) |
| Who owns the mouth each frame | **Yes** — `mouth_owner` + `/probe` |
| Channel-by-channel mouth use in product | **Partial** — velocity (ch 0/1) now warps unlocked tissue at render (`field_warp_gain`); intent channels still sparse |
| ML training loop on video | **Designed above; not built** |

Next build choice, when you want it: **ML control predictor** trained on
capture video labels, emitting the control vector this probe already measures —
not a neural face renderer.
