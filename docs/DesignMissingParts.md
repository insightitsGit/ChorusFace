# Design session — missing parts (focus list)

**Status:** design only (not implemented).  
**Master narrative:** [`TickFeedDesign.md`](TickFeedDesign.md).  
Built on: Side A bandwidth · Side B collect · multi-layer ML · 8s calibration ·
scaffolding.

**Build stance:** this is a **new design** to resolve the accuracy / tick / feed
issues. We are not required to preserve the old sparse ±4 / group-track path as
the core. Existing code may be reused only where it still fits (e.g. `.bds`,
plates, adoption); the tick pipeline is greenfield.

**Locked transport:** ROI = **full face**; **keyframe once**, then **deltas only**
(~0.3–2 MB/s phase-1 target). Same quality; less transport (see Side A).

This is what we **do not have yet** vs the design target. Use it as the build
backlog when you green-light implementation.

---

## A. Per-tick package (core hole)

Each ~16.7 ms `TickPackage[t]` still lacks:

| Missing | Why it matters |
| --- | --- |
| Unified 60 Hz tick index for face+speech+look | One master clock |
| Whole-face dense cell motion (`vx`/`vy` …) | ~241 MB/s truth body |
| Per-cell / per-tick confidence | Know where to gap-fill |
| Beat label @ tick (REST/SMILE/SAY_HI/…) | Know *when* events are |
| Viseme/word @ tick from video teacher | Know *what* was said |
| Look amounts @ tick (teacher curve) | Smile/open/surprise drives |
| Emotion @ tick (e.g. ANGRY window) | Mood section of script |
| Compact `c_t` encode/decode | Side A push without 251 MB/s raw |

**Have today (not enough):** plates, catalog keyframes, ~12 fps × 8 group controls,
runtime visemes, rest `.bds`.

---

## B. Side B — video → cells

| Missing | Notes |
| --- | --- |
| `calibration_script.json` + 8s beat contract in code | Design only |
| Lab sample (Gemini/scripted) pipeline hook | Optional for QA |
| Dense face tracker (UV-flow / mesh / 3DMM) | **Engine not chosen** |
| Frame→face-box registration @ every frame | Beyond digest keyframes |
| Rasterize dense field → 256² face cells | |
| Resample to 60 Hz TickPackages | |
| `face_cell_timeline/` artifact writer | |
| Speech force-align into same ticks | Constrained by SAY_HI/TALK |
| QA: beat windows vs motion peaks | |

---

## C. Side A — feed into NWR

| Missing | Notes |
| --- | --- |
| CHORUS Fabric producer→NWR push path | Decided; not wired |
| One-way push @ 16.7 ms (no full echo) | Design only |
| Whole-face ROI apply (not only ≤256 ±4) | Budget / path gap |
| Delta + f16 + codec on wire | Balance vs ~241 MB/s still open |
| Master consumes `TickPackage` / `TickCode` | |

---

## D. Multi-layer ML

| Missing | Independent retrain via packets |
| --- | --- |
| Packet schemas v1 (`SpeechClock`, `LookDrive`, `FaceMotion`, `TickCode`, …) | |
| L1 SpeechClockML | |
| L2 LookDriveML | |
| L3 FaceMotionML (needs Side B teacher) | Hardest |
| L4 TickCodecML | |
| L5 GapPriorML | Optional |
| `retrain --layer lN` world layout | |

---

## E. Scaffolding / product

| Missing | Notes |
| --- | --- |
| Enforce 8s script (or capture kit+) on upload | |
| Accept user uploads under same lock | Adoption exists; script contract incomplete |
| Cosmetic prefs (eye/skin tint) API | Design only; must not break lock |
| Persist cosmetics across retrain | |

---

## F. Priority order (recommended)

```text
P0  TickPackage codec (encode/decode KEY|Δ)     ← in progress on tickfeedmaster
P0b GPU ingest compute (B1) + face_box map (B2)
P0c 3-tick ring + damp-on-miss (B3)
P0d Label amounts → LOOK drives (B4)
P1  Calibration script artifact + beat-labeled digest
P2  Dense track → face_cell_timeline (Side B teacher)
P3  L3 (+ L5) trained on timeline; L1/L2 teachers from beats/audio
P4  CHORUS push into ingest path
P5  L4 codec + wire compression balance
P6  User upload UX + cosmetics
```

---

## G. Still open decisions (block some missing parts)

1. Dense engine: UV-flow vs mesh vs 3DMM+residual  
2. How soon to require L4 `c_t` vs raw face deltas only  
3. Expander / apply on AIFace CPU vs NWR GPU  
4. Lab Gemini sample first vs user self-take first  

---

## Related docs

| Doc | Piece |
| --- | --- |
| [`SideB_VideoCellCollection.md`](SideB_VideoCellCollection.md) | Collect + 8s script |
| [`CellFeedBandwidth.md`](CellFeedBandwidth.md) | 480/241 MB/s + CHORUS |
| [`MultiLayerTickML.md`](MultiLayerTickML.md) | Layers + packets |
| [`AvatarScaffolding.md`](AvatarScaffolding.md) | Lock vs cosmetics |
