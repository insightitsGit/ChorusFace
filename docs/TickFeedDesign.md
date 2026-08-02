# Tick Feed Design — full face @ 60 Hz (design session master)

**Status:** implementing on `tickfeedmaster` — codec, GPU ingest, app hot path,
Side B `face_cell_timeline` prepare. Legacy ±4 MouthCellPlan **disabled**.  
**Branch:** `tickfeedmaster`  
**Split from:** AminIntheLoop  


**Purpose:** Single from-scratch design doc capturing the Side A / Side B
conversation so you can research and decide before build.

**Related detail docs**

| Doc | Detail |
| --- | --- |
| [`TickPackageHandshake.md`](TickPackageHandshake.md) | Exact bytes, dtypes, KEY/DELTA |
| [`CellFeedBandwidth.md`](CellFeedBandwidth.md) | Rates, CHORUS, delta ladder |
| [`SideB_VideoCellCollection.md`](SideB_VideoCellCollection.md) | Collect from video + 8s script |
| [`MultiLayerTickML.md`](MultiLayerTickML.md) | L1–L5 + abstract packets |
| [`AvatarScaffolding.md`](AvatarScaffolding.md) | Lock scaffold / user cosmetics |
| [`DesignMissingParts.md`](DesignMissingParts.md) | Build backlog |
| [`NWRDataDesign.md`](NWRDataDesign.md) | Broader world dataset layers |
| [`DisplayLayers.md`](DisplayLayers.md) | L00–L11 LOOK vs FIELD |

---

## 1. Why this design exists

The demo path could show plates and sparse mouth drives, but it did **not**
deliver trustworthy **per-tick full-face cell values**. Gap-fill ML without
measured tick truth looked empty or wrong.

**New design stance**

- Greenfield tick pipeline (not required to keep old ≤256 ±4 as the core).
- May reuse `.bds`, plates, adoption, display layers where they still fit.
- Identity albedo and LOOK plates stay photographed — never generative face RGB
  as “who they are.”
- Accuracy target: every **≈ 16.67 ms** (60 Hz) the master has correct **full-face**
  cell channel values (phase-1: velocity), plus labels for speech/look/emotion.

---

## 2. Core insight

Cells do **not** leave the grid. At each tick they **change channel values**.

```text
tick t     →  values on face cells
tick t+1   →  new values (~16.7 ms later)
“motion”   →  how values change (phase-1 = velocity vx, vy)
```

Side B **prepares** those packages (often offline).  
Side A **pushes** them into the NWR master.  
Transport prefers **deltas** after one keyframe.

---

## 3. Numbers (locked arithmetic)

| Quantity | Value |
| --- | --- |
| Tick rate | **60 Hz** |
| Tick period | **≈ 16.67 ms** |
| World grid | **256 × 256 × 32** float32 |
| Full world rewrite / tick | **≈ 8 MB** |
| Full world @ 60 Hz | **≈ 480 MB/s** (ceiling, all cells × 32) |
| This avatar face box | ~158 × 199 ≈ **31,442** cells |
| Full face × 32 @ 60 Hz | **≈ 241 MB/s** (truth ceiling for ROI) |
| One cell × 32 @ 60 Hz | **1,920** floats/s (**7.5 KB/s**) |
| Face × vx,vy keyframe f32 | **≈ 0.25 MB** once |
| Face × 32 keyframe f32 | **≈ 3.8 MB** once |

**Locked transport mode**

```text
ROI        = FULL FACE (not mouth-only)
t = 0      = KEYFRAME (full package once)
t > 0      = DELTAS only
phase-1    = velocity vx, vy (NWR ch 0/1 family)
wire       = prefer f16; sparse deltas when possible
steady     ≈ 0.3–2 MB/s typical speech (phase-1)
optional   = compact code c_t later (~15–60 KB/s)
```

Raw **241 MB/s every tick** is the accuracy *budget*, not the product wire rate.  
Bidirectional send/receive of full face every second is rejected — **one-way push**
into the master only.

---

## 4. Big picture

```text
┌─────────────────────────────────────────────────────────────────┐
│  SCAFFOLDING (same for every avatar)                            │
│  8s calibration script · face box · TickPackage · L00–L11       │
│  User may change cosmetics; must pass script + lock geometry      │
└─────────────────────────────────────────────────────────────────┘
                │
                ▼
┌───────────────────┐     packets      ┌───────────────────┐
│  SIDE B           │ ───────────────► │  ML LAYERS        │
│  Collect/prepare  │   teachers       │  L1…L5            │
│  from video+words │ ◄── gap/live ─── │  abstract APIs    │
└───────────────────┘                  └───────────────────┘
                │                                │
                │     TickPackage KEY / Δ        │
                └────────────┬───────────────────┘
                             ▼
                    ┌─────────────────┐
                    │  SIDE A         │
                    │  CHORUS push    │
                    │  one-way @ 60Hz │
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │  NWR MASTER     │
                    │  S ← KEY        │
                    │  S ← S + Δ      │
                    │  LOOK plates    │
                    │  resident       │
                    └─────────────────┘
```

---

## 5. Side B — provide the data (collect / prepare)

### 5.1 Problem

Video pixels + sparse landmarks do **not** automatically equal per-tick face
cell packages. We must **define and collect** them.

### 5.2 Calibration script (~8s) — labels by contract

Do not guess when smile / “hi” / angry happen. Use ordered beats:

| Time | Beat | Collects |
| --- | --- | --- |
| 0–1s | REST | Rest / identity reference |
| 1–2s | SMILE | Smile look + face values |
| 2–3s | OPEN | Open / jaw |
| 3–4s | SAY_HI | Speech (“hi”) + visemes |
| 4–5s | SURPRISE | Upper face |
| 5–6s | ANGRY | Emotion section |
| 6–7.5s | TALK | Continuous speech |
| 7.5–8s | REST | Return |

- **Lab:** scripted sample (e.g. Gemini) to wire the pipeline.  
- **Production:** real user upload performs the **same script**.  
- Gemini is **not** identity albedo for the user’s world.

Artifact: `calibration_script.json` (beat windows + tick_rate=60).

### 5.3 Prepare pipeline (offline)

```text
Scripted video + audio + calibration_script
  → every frame registered to face_box (= .bds UV)
  → dense whole-face track (engine TBD: UV-flow / mesh / 3DMM+residual)
  → rasterize to face patch values
  → resample / interpolate to 60 Hz ticks
  → align words/visemes (SAY_HI / TALK windows)
  → attach beat / look / emotion labels
  → write FaceCellTimeline / TickPackages
```

**Live path does not re-run dense tracking inside 16.7 ms** — it reads prepared
packages (or ML that was trained on them).

### 5.4 What video can / cannot give

| Can | Cannot |
| --- | --- |
| Dense appearance per frame | Master Lock / authority from RGB |
| Dense 2D deformation → cell velocity | Invented identity RGB |
| Audio → words when aligned | Magic full 32-ch semantics |

Phase-1 maps measured deformation into **vx, vy** only; material/lock stay from digest.

---

## 6. Side A — feed the master (transport / apply)

### 6.1 Master clock

NWR (field runtime) is **master of time** at 60 Hz.  
Producer **pushes**; master does not echo full face state each tick.

### 6.2 CHORUS Fabric

- Binary **float32/f16 vector** transport (AI API ↔ AIFace/NWR).  
- Better than JSON/REST for vectors.  
- Cipher does **not** replace content compression (ROI + delta + optional `c_t`).

### 6.3 First load + deltas

```text
HELLO / ACK
KEYFRAME   →  S := values          (~0.25 MB vel f32 / ~126 KiB f16)
DELTA*     →  S := S + Δ           (sparse typical)
optional KEY refresh every ~2 s
```

Steady design target: **~0.3–2 MB/s** phase-1 full-face velocity deltas.

### 6.4 LOOK vs FIELD

| Plane | Examples | In TickPackage? |
| --- | --- | --- |
| FIELD values | vx, vy on face cells | **Yes** (body) |
| LOOK plates | smile.png, open, atlas | **No** — resident; **amounts** in labels |
| Identity | source_face, lock | **No** — resident |

`smile.png` is photographed look evidence, not the tick body.

### 6.5 Bridge stack (adopted)

Four mechanisms close Side A. They are **part of this design**, not optional
extras.

#### B1 — Direct staging ingest (GPU)

Push raw TickPackage body into a pre-allocated staging SSBO. One compute
dispatch unpacks the face patch into the master world (ch 0/1):

- KEY → `S.xy := vel`
- DELTA → `S.xy += vel` (dense or sparse unpack)
- EMPTY → no value write (labels may still update)
- Respect Master Lock: do not overwrite locked identity cells
- Prefer f16 packed `unpackHalf2x16` on dense path

#### B2 — ROI patch remapping

Wire carries contiguous full-face patch only (`face_w * face_h`).  
Header `face_x/y/w/h` maps patch index → world cell. GPU does the scatter.

#### B3 — Ring buffer + velocity decay

```text
Incoming packages → 3-tick lockstep ring (~50 ms)
  → if tick T present: apply KEY/Δ
  → if missing: v_xy *= γ  (γ ≈ 0.82–0.88), no invented coast integrate
Periodic KEY refresh clears drift
```

#### B4 — FIELD-driven LOOK

Master velocities warp sampling UVs; resident plates composite by **label
amounts** (smile/open/surprise/…). Full look stack, not smile-only.

```text
CHORUS → TickPackage → [3-tick ring] → GPU ingest (B1+B2)
       → damp if miss (B3) → field warp + plates by labels (B4)
```

---

## 7. TickPackage handshake (summary)

Full binary layout: [`TickPackageHandshake.md`](TickPackageHandshake.md).

| Part | Size | Contents |
| --- | --- | --- |
| Header | 64 B | magic TPK1, kind KEY/DELTA, tick, face_box, channel_mask, dtype, encoding |
| Labels | 48 B | beat, emotion, viseme, smile/open/surprise amt, word |
| KEY body | N×C×E | Dense face patch |
| DELTA body | sparse or dense | `SPARSE_DELTA` / `DENSE_DELTA` / `EMPTY` |

**Phase-1 locked**

```text
channel_mask   = 0x3          # vx, vy
meaning        = VELOCITY     # grid units / second (NWR ch 0/1 family)
value_dtype    = f16 preferred on wire
delta_encoding = SPARSE first; DENSE if very dense change
```

**Have today:** face_box geometry, conceptual channels.  
**Don’t have yet:** filled arrays, wire, master apply of full-face patches.

---

## 8. Multi-layer ML (connected by abstract packets)

Not one giant model. Layers retrain independently via versioned packets.

| Layer | Job | Teacher |
| --- | --- | --- |
| L1 SpeechClock | audio/text → viseme/word @ 60 Hz | force-align + script |
| L2 LookDrive | → smile/open/surprise amounts | catalog + curves |
| L3 FaceMotion | → full-face vx,vy (gaps + live) | Side B timeline |
| L4 TickCodec | patch ↔ compact `c_t` | autoencode patches |
| L5 GapPrior | inpaint low-confidence only | synthetic holes |

```text
SpeechClock → LookDrive → FaceMotion → TickCode → CHORUS → NWR
```

Measured tick wins when confidence is high; ML fills holes and live chat.

---

## 9. Scaffolding for future users

**Lock:** script, face box, TickPackage contract, lock geometry, display path.  
**Open:** eye/skin tint, style, voice — cosmetics that don’t break registration.  
**Upload:** any user video that satisfies the script + quality gates → same path.

See [`AvatarScaffolding.md`](AvatarScaffolding.md).

---

## 10. End-to-end authority

```text
calibration video (measured)
  → Side B TickPackages
  → ML only for gaps / live
  → KEY + Δ push (CHORUS)
  → NWR apply under Master Lock
  → LOOK plates by label amounts
```

Forbidden: generative face RGB as identity; inventing dense flow sold as measured
without a collect path; round-tripping 241 MB/s both ways.

---

## 11. What is locked vs still open

### Locked in this design

- 60 Hz / 16.7 ms master  
- Full-face ROI  
- Keyframe + deltas only  
- Phase-1 = **velocity** vx, vy  
- One-way CHORUS-style push  
- 8s calibration script  
- Multi-layer ML + abstract packets  
- New tick pipeline (not old sparse core)  
- TickPackage v1 field contract (detail doc)

### Open (research / build choices)

1. Dense tracker engine (UV-flow vs mesh vs 3DMM+residual)  
2. When L4 `c_t` becomes mandatory vs raw sparse Δ  
3. Apply/expand on AIFace CPU vs NWR GPU  
4. Lab Gemini sample first vs user self-take first  
5. Exact ε for sparse omit; KEY refresh interval tuning  

---

## 12. Completeness check (design only)

| Area | Components defined? | Connected? |
| --- | --- | --- |
| Side B | Yes | → TickPackage / teachers |
| Side A | Yes | ← TickPackage → NWR |
| Handshake | Yes | HELLO → KEY → Δ |
| ML | Yes | Via packets between sides |
| Scaffolding | Yes | Same contract for all users |

**Verdict:** Side A, Side B, and the connection are designed enough to research
and later implement. Remaining work is engine choice + implementation, not a
missing third architecture.

---

## 13. Bridge solutions — adopted (design verify)

**Status:** adopted into §6.5. Verified against this design only (not legacy code).

### 1) Direct staging buffer / compute ingest

| Design need | Does the proposal match? |
| --- | --- |
| Apply ~31k face velocities every 16.7 ms | **Yes** — parallel patch ingest, not per-cell CPU loops |
| `S := KEY` and `S := S + Δ` | **Yes** — `u_is_keyframe` write vs add |
| Phase-1 vx, vy | **Yes** — packed f16 pair |
| One-way push into master | **Yes** — buffer in, no echo |
| Sparse Δ + EMPTY (handshake) | **Partial** — sketch is dense-only; design also needs sparse unpack |
| Master Lock / identity not overwritten | **Must keep** — ingest only unlocked / allowed channels |

**Verdict:** **Aligned.** This *is* the Side A apply mechanism the design implies. Extend sketch for sparse/EMPTY and lock policy.

### 2) ROI patch remapping (face_box header)

| Design need | Match? |
| --- | --- |
| ROI = full face, not full 256² on wire | **Yes** |
| Contiguous patch ~31,442 | **Yes** |
| Header `face_x/y/w/h` → world map | **Yes** — same as TickPackage handshake |
| Labels separate from velocity body | **Yes** (amounts not in velocity buffer) |

**Verdict:** **Aligned.** Exact fit to locked transport + handshake.

### 3) Ring buffer + velocity decay on miss

| Design need | Match? |
| --- | --- |
| Master ticks at 60 Hz even if push jitters | **Yes** — 3-tick queue (~50 ms) |
| One-way CHORUS can drop/delay | **Yes** — starve policy |
| Phase-1 values = **velocity** | **Careful** — on miss, design wants **damp `v → 0`**, not a second physics that double-applies velocity as position |
| Periodic KEY refresh kills drift | **Yes** — pairs with decay |

**Verdict:** **Aligned with a rule tweak:**  
`missing Δ → v *= γ` (graceful stop). Optional short ring for smoothness. Do **not** require “coast by integrating v again” unless the design later switches meaning to displacement.

### 4) Field-driven LOOK deform (UV + plates)

| Design need | Match? |
| --- | --- |
| FIELD = cell velocities in TickPackage | **Yes** — sample field to warp |
| LOOK plates resident; only **amounts** in labels | **Yes** — `u_smile_amount` etc. |
| Identity photo under plates | **Yes** — base + smile mix |
| Full display stack (open/atlas/surprise/jaw…) | **Partial** — sketch shows smile only; design has more label-driven looks |

**Verdict:** **Aligned in principle.** FIELD warps; LOOK composites by label amounts. Design expects the full label set (smile/open/surprise/…), not smile alone.

### Design-only summary

| # | Needed by new design? | Proposal fills it? | Notes |
| --- | --- | --- | --- |
| 1 Compute ingest | **Yes** | **Yes** | Add sparse/EMPTY + lock |
| 2 Face ROI map | **Yes** | **Yes** | Handshake |
| 3 Ring + damp | **Yes** (push path) | **Yes** | Damp on miss; KEY resync |
| 4 LOOK via field warp | **Yes** | **Yes** | Generalize beyond smile |

```text
DESIGN BRIDGE (no legacy assumed)

Side B / ML → TickPackage (KEY|Δ, full-face, labels)
     → CHORUS one-way push
     → [3-tick ring]
     → GPU ingest: S:=KEY or S+=Δ  (vx,vy)
     → if miss: damp v
     → render: field warps UVs; plates mix by label amounts
```

**Overall:** all four solutions **belong in this design**. They close the Side A apply + jitter + FIELD/LOOK bridge the TickFeed doc already assumes. Only tighten: sparse Δ ingest, damp-not-double-coast, full label-driven looks.

---

## 14. Suggested research order

1. Dense monocular face trackers that output UV/grid-aligned deformation  
2. Sparse delta codecs / f16 face patches over gRPC (CHORUS Fabric)  
3. Force-alignment of short scripted speech to 60 Hz  
4. **Compute TickPackage ingest** (Gap 1–2) into Cell SSBO + lock  
5. Jitter ring + damp policy (Gap 3) on push path  
6. Wire label amounts into existing L04–L07 (Gap 4) — no stack rewrite  
7. Per-avatar L3 training size vs quality on 8s teachers  

---

## 15. Doc map (session artifacts)

```text
TickFeedDesign.md          ← this master (start here)
TickPackageHandshake.md    ← binary contract
CellFeedBandwidth.md       ← MB/s math + CHORUS
SideB_VideoCellCollection.md
MultiLayerTickML.md
AvatarScaffolding.md
DesignMissingParts.md      ← P0–P6 backlog when building
```

---

*End of master design. No code claimed implemented by this document.*
