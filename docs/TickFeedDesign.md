# Tick Feed Design — full face @ 60 Hz (design session master)

**Status (three planes — do not collapse):**

| Plane | State on `tickfeedmaster` |
| --- | --- |
| **Contract** (bytes, labels, KEY/Δ rules) | **Done** — see handshake |
| **Local Side A apply** (ring → GPU ingest → LOOK) | **Done** — codec + `tick_ingest.comp` |
| **Remote CHORUS transport of KEY/Δ** | **Done (lab)** — lane A `c_t` + lane B framed packages / TPK_REF; multi-host HELLO_ACK still operator |

Legacy ±4 **disabled**. TickFeed-native demo: labels sole LOOK authority (no
MouthLayerTimeline hard-snap). Measured timeline carries per-tick `source`
provenance (no synth sold as measured).  

**Branch:** `tickfeedmaster`  
**Split from:** AminIntheLoop  
**Operator step:** generate `calibration_take.mp4` from
[`AvatarCalibrationPrompt.md`](AvatarCalibrationPrompt.md) when replacing the
lab video, then `python scripts/build_tickfeed_demo.py --clean` (or
`train_tickfeed_ml.py --prepare`). Play: `python scripts/run_tickfeed_demo.py`.
Plate-only refresh: `python scripts/rebuild_tickfeed_plates.py [--timeline]`.

**Purpose:** Single from-scratch design doc capturing the Side A / Side B
conversation **and** the implementation map on this branch (see §16–§17).

**Doc layers (read in order):**

1. **§1–§12 — Initial TickFeed design** — contract, Side A/B, B1–B4 bridges,
   CHORUS, ML L1–L5, completeness of the architecture.
2. **§13 — Initial bridges adopted** — B1–B4 wired as designed.
3. **§14 — Post-initial design improvements** — mouth blur, word sync,
   idle moods, plates, aligner upgrades landed **after** the architecture
   above was already working. These do not change the TickPackage contract.
4. **§15–§17 — Remaining research + code map**

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
  → dense whole-face track (lab: Farneback optical flow on face crop)
  → resample / interpolate to 60 Hz ticks
  → align words/visemes (audio-energy force-align inside script windows)
  → attach beat / look / emotion labels (look floors may boost amounts;
     they must NOT be blended into stored FIELD velocities)
  → write FaceCellTimeline with per-tick source provenance
  → train L1–L5 / emit TickPackages at runtime
```

**Provenance (required):** each tick stores `source`:

| Code | Meaning | May win as “measured”? |
| --- | --- | --- |
| `0` | `measured_optical_flow` | **Yes** |
| `1` | `blend_legacy` (forbidden for new writes) | No |
| `2` | `synthetic_fallback` (no flow) | No — ML/gap only |

§10 forbids inventing dense flow sold as measured. Synth is allowed only as an
explicit fallback with `source=2` and lowered confidence.

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

**CHORUS ↔ TickPackage binding (two lanes)**

CHORUS Fabric carries **float32 vectors of fixed `CHORUS_DIM`** (TickFeed: 64).
A TickPackage is an **arbitrary byte blob** (64 B header + 48 B labels + body).
Those are not the same shape — the binding is explicit:

| Lane | Payload | When |
| --- | --- | --- |
| **A — `c_t`** | one `float32[64]` compact code | every tick (bandwidth path) |
| **B — TickPackage** | zlib bytes → framed into N×`float32[64]` chunks, or a **TPK_REF** ticket + shared spool when too large for inline | KEY / Δ / HELLO fidelity path |

Frame meta (first floats of each chunk) — **must be exact in float32**:

| Slot | CHUNK | REF |
| --- | --- | --- |
| 0 | `TPK_CHUNK_MAGIC` (= 64101) | `TPK_REF_MAGIC` (= 64102) |
| 1 | tick | tick |
| 2 | n_chunks | nbytes |
| 3 | chunk_i | crc32 lo uint16 |
| 4 | nbytes | crc32 hi uint16 |
| 5–6 | crc32 lo/hi uint16 | — |
| 7 | compressed_len | — |
| 8… | payload bytes as 0..255 | spool name bytes |

Magics and integer metas stay **≤ 2^24** so they survive IEEE-754 binary32.
CRC32 is **never** stuffed into one float32 (24-bit mantissa) — always two
uint16 halves. See `chorus_transport.py` + `reassemble_lane_b_chunks` tests.

Inline threshold (`AIFACE_CHORUS_TPK_INLINE_MAX`, default 4096 compressed bytes):
sparse Δ usually inlines; large KEY uses **TPK_REF + spool**.  
Lab HELLO: self-ACK in-process **and** both HELLO/ACK blobs pushed on lane B.
Remote multi-host ACK remains the production upgrade.

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
Producer clock and master clock are related but not the same step:

  produce_tick = master_tick + RING_DEPTH   # schedule ~50 ms ahead (depth=3)
  ring[produce_tick] ← package
  at master_tick T: pop ring[T] only
    → if present: apply KEY/Δ
    → if missing: v_xy *= γ  (γ ≈ 0.82–0.88), no invented coast integrate

Warm-up: first RING_DEPTH master ticks may damp until the lead fills.
**Lab local-ring** produces `tick == master` (push then pop same tick) so Side B
FIELD+labels land in the same 16.7 ms step — ring lead is for **wire-loop** /
remote jitter only. Never apply producer `last_labels` on a master miss (LOOK
would jump ahead of FIELD).
Periodic KEY refresh clears drift.
```

#### B4 — FIELD-driven LOOK

Master velocities warp sampling UVs; resident plates composite by **label
amounts** (smile/open/surprise/…). Full look stack, not smile-only.
Display quality (hybrid core+edge mattes, FIELD muted only inside the oral
disk when a plate owns it, plate-open hysteresis) stays inside this contract —
never generative face RGB, never globally killing ``open.png``.

**Label authority precedence (when TickFeed enabled):**

```text
1. TickPackage labels (smile/open/surprise/brow/viseme/emotion) ← sole LOOK amounts
2. Emotion-catalog easing                              ← only if TickFeed off
3. MouthLayerTimeline hard-snap                        ← disabled under TickFeed
```

**Emotion → face:** Side B `look_drive` carries `emotion_id` + `brow`; measured
`face_cell_timeline` velocity is full-face FIELD (brows/cheeks included). Labels
drive LOOK (plates/brow ease); FIELD warps identity. Lab default
`AIFACE_TICKFEED_ABSOLUTE=1` sends KEY every tick (`S:=vel`) so 16.7 ms frames
do not accumulate Δ residue (blink lids use overlay only — no blink muscle warp
on top of FIELD).

**FIELD semantics (phase-1):** ch0/1 are **rest-relative displacement** of the
face patch (Farneback from the first/rest frame → each frame), not frame-to-frame
optical flow. The avatar samples them as warp vectors. Frame-Δ flow + constraint
damping/neighbor blend left lower-lip residue; collect stores rest→frame, KEY
preserve skips damp/blend on ingest ticks, and TickFeed LOOK sets jaw assist to 0.

Catalog ease must not overwrite `_expr_plate_blend` / plate amounts while
TickFeed owns LOOK.

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

**Have today:** filled KEY/Δ arrays, codec (CRC = header[0..35]+body), GPU master
apply, labels→LOOK, lane A + lane B CHORUS push, producer-lead ring, measured
timeline provenance.  
**Operator-owned:** lab MFA; production multi-host HELLO_ACK / separate master pod.

---

## 8. Multi-layer ML (connected by abstract packets)

Not one giant model. Layers retrain independently via versioned packets.

| Layer | Job | Teacher | Phase-1 lab note |
| --- | --- | --- |
| L1 SpeechClock | audio/text → viseme/word @ 60 Hz | script windows + **audio-energy force-align** | MFA is the upgrade path |
| L2 LookDrive | → smile/open/surprise amounts | catalog + curves | |
| L3 FaceMotion | → full-face vx,vy (gaps + live) | Side B timeline (**measured** conf high) | |
| L4 TickCodec | patch ↔ compact `c_t` | **PCA** on measured patches (phase-1) | AE remains a future upgrade |
| L5 GapPrior | inpaint low-confidence only | synthetic holes | |

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
calibration video
  → Side B FaceCellTimeline (source=0 measured flow preferred)
  → ML only for gaps / live (source≠0 or low conf)
  → KEY + Δ + c_t push (CHORUS lanes A+B)
  → 3-tick producer-lead ring
  → NWR GPU apply under Master Lock
  → LOOK plates by TickPackage label amounts (sole authority)
```

Forbidden: generative face RGB as identity; inventing dense flow sold as measured
(no silent synth blend into FIELD); round-tripping full-face both ways;
MouthLayerTimeline hard-snap overriding TickFeed labels.

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

1. Better dense tracker than Farneback (UV-flow / mesh / 3DMM+residual)  
2. When L4 `c_t` becomes mandatory vs raw sparse Δ on the wire  
3. Multi-host remote HELLO_ACK (lab self-ACK today)  
4. MFA upgrade for L1 (energy force-align is the lab teacher)  
5. L4 autoencoder upgrade (PCA is phase-1)  
6. Exact ε / KEY refresh interval tuning  

---

## 12. Completeness check (design + branch)

| Area | Designed? | Implemented on `tickfeedmaster`? |
| --- | --- | --- |
| Side B collect + timeline | Yes | **Yes** — Farneback→60 Hz + `source` provenance |
| Side A codec KEY/Δ/HELLO | Yes | **Yes** — `package.py`, CRC scope correct |
| GPU ingest B1+B2 | Yes | **Yes** — `tick_ingest.comp` sparse/dense/EMPTY/lock |
| Ring B3 producer lead | Yes | **Yes** — `produce_tick = master + RING_DEPTH` |
| LOOK B4 label authority | Yes | **Yes** — TickFeed sole LOOK; catalog ease blocked |
| CHORUS lane A (`c_t`) | Yes | **Yes** — `push_code` |
| CHORUS lane B (packages) | Yes | **Yes** — framed inline / TPK_REF |
| ML L1–L5 | Yes | **Yes** — energy align + PCA L4 (lab notes in §8) |
| Scaffolding / cosmetics | Yes | **Yes** — prefs + GLSL grade uniforms |
| Legacy ±4 disabled | Yes | **Yes** |

**Verdict:** Side A, Side B, and the connection are **implemented** for the lab
single-host path (initial design). Mouth blur/sync/idle polish in **§14** is a
**post-initial** band on top of that — not a missing third architecture.
Remaining research is tracker/MFA/AE upgrades, a better capture take, and
multi-host ACK.

---

## 13. Initial design bridges — adopted and implemented

**Status:** B1–B4 from §6.5 are the **initial architecture** — adopted into
design **and** wired in runtime (see §17 code map). Everything in §14 is
layered **on top of** this, not a redesign of the contract.

| Bridge | Design | Runtime |
| --- | --- | --- |
| B1 Compute ingest | KEY write / Δ add / sparse / EMPTY / lock | `tick_ingest.comp` + `field._run_tick_ingest` |
| B2 ROI remap | `face_x/y/w/h` → world scatter | uniforms from package face box |
| B3 Ring + damp | producer lead + miss `v *= γ` | `app._simulate_tick` + `ring.py` (`γ=0.85`) |
| B4 LOOK by labels | smile/open/surprise/viseme sole amounts | `_apply_tickfeed_labels_to_look` + no catalog overwrite |

```text
INITIAL DESIGN BRIDGE (lab path — §1–§13)

Side B / ML → TickPackage (KEY|Δ, full-face, labels)
     → CHORUS lane A (c_t) + lane B (framed TPK / TPK_REF)
     → [3-tick producer-lead ring]
     → GPU ingest: S:=KEY or S+=Δ  (vx,vy)
     → if miss: damp v
     → render: field warps UVs; plates mix by label amounts
```

Demo ``--wire-loop`` is **opt-in** (`run_tickfeed_demo.py` defaults to
local-ring for FPS). Wire-loop proves the bandwidth path; local-ring is the
lab play default.

---

## 14. Post-initial design improvements (after B1–B4)

> **Scope:** These landed **after** the initial TickFeed design (§1–§13) was
> already working. They sharpen **readable speech** (transition blur +
> word/sentence sync + idle LOOK) without changing TickPackage bytes, KEY/Δ
> rules, or B4 label authority. Treat them as a second design band.

### 14.1 Problems addressed

| Issue | Symptom after initial design | Fix band |
| --- | --- | --- |
| Transition blur | Mid-open (`≈0.15–0.55`) smeared — FIELD + soft plate mix fought | Single-owner OPENING/OPEN/CLOSING + velocity FIELD mute |
| Word / sentence sync | Closures skipped / vowel hold floors drifted past consonants | Absolute overlay until + interruptible PP/MM/CLOSED |
| Always-smiling idle | Identity rest is soft-smile; 0-state looked “happy” | Switchable zero moods (`neutral` / `smile` / `waiting`) |
| Weak bilabials | Energy align buried PP inside vowels | Onset pin + RMS-valley snap; denser plate bank |

### 14.2 Scheduler / sync (live TTS → LOOK)

| Improvement | Behavior | Code |
| --- | --- | --- |
| Absolute overlay release | `until = due_at + duration`, capped by next event — **not** `now + vowel_hold_floor` | `speech.speech_overlay_until`, `app._fire_impulse` |
| No cumulative min_hold shift | TickFeed live path forces `min_hold=0` in `apply_speech_pace` | `app._schedule_audio` |
| Closures never skipped | PP/MM/CLOSED/REST interrupt open holds; clear plate hysteresis | `app._fire_impulse` |
| Playback clock | Viseme fire uses sink `media_time` when available | `audio.*Sink.media_time`, `app._speech_now` |
| Bilabial onset pin | Leading PP gets ~45 ms at word start | `tts._subdivide`, `bias_bilabial_onsets` |
| Energy valley snap | PP/MM/CLOSED pull toward nearby RMS trough | `tts.snap_bilabials_to_energy_valleys` |
| Whisper words (when keyed) | `--tts-align words` default if `OPENAI_API_KEY` / `AIFACE_LLM_API_KEY` set; else energy | `app` CLI, `run_tickfeed_demo.py` |

### 14.3 Renderer / transition ownership

| Improvement | Behavior | Code |
| --- | --- | --- |
| Transition state machine | `REST` / `OPENING` / `OPEN` / `CLOSING` from openness velocity | `app._update_mouth_transition` |
| Velocity-aware FIELD mute | Strong mute on OPENING/CLOSING mid-band; keep some travel at steady OPEN | `app._update_avatar_uniforms` |
| Early atlas commitment | Hard-snap `pair_for_viseme`; boost plate amount mid-transition | `_apply_tickfeed_labels_to_look`, `avatar.frag` |
| Stronger rest-align under plates | Higher `rest_mix` so FIELD does not smear under LOOK | `avatar.frag` |

### 14.4 Idle presence + demo ops

| Improvement | Behavior | Code |
| --- | --- | --- |
| Zero moods | `neutral` / `smile` / `waiting` inside presence `zero` | `app._zero_mood_*`, key **Z**, `POST /calibrate {"zero_mood":…}` |
| Hearing vs zero | Typing/pending → hearing look; chat end → zero mood | `app._update_presence` |
| Quiet demo defaults | `--gpu-log` / `--tickfeed-debug` opt-in (were tanking FPS) | `scripts/run_tickfeed_demo.py` |

Identity caveat (unchanged by moods): this take’s `source_face.png` is already
a soft smile — `neutral` removes smile **plate** / brow drives, but cannot
invent a flatter rest photo. New capture take required for true no-impression.

### 14.5 Plate bank + teacher FIELD (rebuild tools)

| Improvement | Behavior | Code / command |
| --- | --- | --- |
| Priority distinct plates | CLOSED/PP/FF/TH/AA prefer **different** frames when the take allows | `plates.PRIORITY_ATLAS_VISEMES`, `select_viseme_atlas_frames` |
| Closed openness = 0 | CLOSED/PP metadata forced sealed for hard-snap indexing | `capture._write_plate_atlas` |
| Plate rebuild script | Refresh LOOK atlas without full digest | `scripts/rebuild_tickfeed_plates.py` |
| Denser Farneback | More levels / larger window for rest→frame teacher FIELD | `tickfeed/collect._optical_flow_face_series` + `--timeline` |

Lab atlas after rebuild (illustrative): PP, CLOSED, FF, TH on distinct indices
when frames exist; `MM` aliases to PP.

### 14.6 QA helpers

| Script | Purpose |
| --- | --- |
| `scripts/_tmp_sync_blur_qa.py` | Phrase matrix: closure hits, mid-open FIELD gain, idle release |
| `scripts/_tmp_full_cycle.py` | Speak/preview capture for visual QA |
| `tests/test_speech_overlay_until.py` | Absolute until contract |
| `tests/test_bilabial_onset.py` | PP onset pin / borrow |
| `tests/test_eye_blink.py` | Blink envelope + state machine + dt cap |

### 14.7 Blink band (post-initial — same playbook as mouth)

> Same separation as §14.2–§14.3: **EyeSystem schedules**, L09 **owns** the
> aperture while blinking, FIELD/widen/brow must not fight the lids.

| Improvement | Behavior | Code |
| --- | --- | --- |
| Blink state machine | `OPEN` / `CLOSING` / `CLOSED` / `OPENING` | `biomechanics/eyes.py` |
| dt cap on phase | Huge FPS hitch cannot skim the closed hold | `BLINK_MAX_STEP_S` |
| Blink beats widen | Waiting/surprise widen muted during close | `app` expr upload + `avatar.frag` L09 |
| Eye-disk FIELD mute | Mute FIELD (+ soft muscle) under lids while blinking | `total_displacement` in `avatar.frag` |
| Harder lid commit | Earlier full shut; rest = photo (no soft globe bars) | L09 lid cover |
| L08/L10 bow-out | Surprise plate + procedural brow ease under blink | L08/L10 |

Non-goals (unchanged): no Orbicularis blink muscle under TickFeed FIELD; no
invented closed-eye RGB plates.

### 14.8 Explicitly deferred (still after §14)

These remain **future** — not part of the post-initial mouth/blink pass:

1. **Lab MFA** — full phoneme forced-align beyond Whisper words  
2. **New capture take** — true neutral rest + tongue-visible TH  
3. Multi-host HELLO_ACK + remote master (transport; zero local LOOK effect)  
4. L4 autoencoder (PCA is phase-1 codec)  
5. Per-avatar L3 size/quality research on more takes  

---

## 15. Suggested research order (remaining)

**Initial design (§1–§13) — done on branch:** compute ingest, ring lead, label
LOOK, CHORUS two-lane push, energy force-align teacher, measured provenance,
PCA L4.

**Post-initial mouth band (§14) — done on branch:** absolute overlay sync,
closure priority, transition FIELD ownership, zero moods, bilabial align,
Whisper-words default when keyed, denser plates + Farneback rebuild tools.

**Still useful next (in order):**

1. Lab MFA (or always-on Whisper words in environments with a key)  
2. New calibration take (neutral rest + TH tongue) → `build_tickfeed_demo --clean`  
3. Stronger dense tracker beyond Farneback (teacher FIELD quality)  
4. Multi-host HELLO_ACK + remote master consumer  
5. L4 AE if PCA quality is insufficient  
6. Per-avatar L3 size vs quality on more takes  

---

## 16. Doc map (session artifacts)

```text
TickFeedDesign.md          ← this master (start here)
  §1–§13  initial TickFeed design + B1–B4
  §14     post-initial mouth / sync / idle improvements
  §15–§17 remaining research + code map
TickPackageHandshake.md    ← binary contract + status table
CellFeedBandwidth.md       ← MB/s math + CHORUS
SideB_VideoCellCollection.md
MultiLayerTickML.md
AvatarScaffolding.md
DesignMissingParts.md      ← backlog (initial vs post-initial)
PhoneticFidelity.md        ← lip-reading inventory + post-initial sync notes
```

---

## 17. Implementation map (`tickfeedmaster`)

Honest code pointers. Prefer these over stale early-draft §7 text.

### Initial design (B1–B4 / transport)

| Concern | Module / entry |
| --- | --- |
| TickPackage encode/decode + CRC | `src/aiface/tickfeed/package.py` |
| CHORUS lanes A+B | `src/aiface/tickfeed/chorus_transport.py` |
| Driver push / HELLO / timeline loop | `src/aiface/tickfeed/driver.py` |
| Producer-lead ring | `src/aiface/tickfeed/ring.py`, `app._simulate_tick` |
| GPU KEY/Δ ingest | `src/aiface/shaders/tick_ingest.comp`, `runtime/field.py` |
| LOOK label authority (B4) | `app._apply_tickfeed_labels_to_look` |
| Measured collect + provenance | `src/aiface/tickfeed/collect.py`, `timeline_io.py` (`source`) |
| Teacher audio-energy force-align | `src/aiface/tickfeed/force_align.py` |
| L1–L5 train/load | `src/aiface/tickfeed/ml/` |
| Cosmetics GLSL | `cosmetics.py` + `avatar.frag` uniforms |
| Clean demo build | `scripts/build_tickfeed_demo.py` |
| Local CHORUS plane | `scripts/start_chorus_local.py` |
| Contract tests | `tests/test_tickfeed*.py` |

### Post-initial design (§14)

| Concern | Module / entry |
| --- | --- |
| Absolute overlay until | `speech.speech_overlay_until`, `app._fire_impulse` |
| Speech playback clock | `audio.*Sink.media_time`, `app._speech_now` |
| Bilabial onset + valley snap | `tts.bias_bilabial_onsets`, `snap_bilabials_to_energy_valleys` |
| Transition state + FIELD mute | `app._update_mouth_transition`, `_update_avatar_uniforms` |
| Zero moods | `app._set_zero_mood`, `_apply_zero_mood_overlay` |
| Priority plate select / rebuild | `plates.select_viseme_atlas_frames`, `scripts/rebuild_tickfeed_plates.py` |
| Denser Farneback | `tickfeed/collect._optical_flow_face_series` |
| Blink state + lid ownership | `biomechanics/eyes.py`, `avatar.frag` L09 / eye-disk FIELD mute |
| Demo play (quiet defaults) | `scripts/run_tickfeed_demo.py` |
| Sync/blur/blink unit tests | `tests/test_speech_overlay_until.py`, `test_bilabial_onset.py`, `test_eye_blink.py` |

Canonical world: `output/worlds/tickfeed/` (identity `source_face.png`, timeline,
`ml/`, `plate_atlas.json`). Do not treat old `output/worlds/avatar*` demos as
TickFeed truth.

---

*End of master design. **Initial** implementation status: header table + §12 /
§13. **Post-initial** mouth improvements: §14 / §17. Detail checklists:
`TickPackageHandshake.md` and `DesignMissingParts.md`.*
