# Multi-layer ML for per-tick face data

**Status:** implemented (`aiface.tickfeed.ml` + `scripts/train_tickfeed_ml.py`).  
**Master:** [`TickFeedDesign.md`](TickFeedDesign.md).  
**Pair with:** [`SideB_VideoCellCollection.md`](SideB_VideoCellCollection.md) ·
[`CellFeedBandwidth.md`](CellFeedBandwidth.md)

Idea: **one ML layer per section** of the tick pipeline — not a single model
that invents the whole face.

---

## 1. Why multi-layer

| If one giant model… | If split layers… |
| --- | --- |
| Mixes look RGB + motion + words | Each layer has one job |
| Hard to retrain / debug | Retrain only the broken layer |
| Easy to violate identity | Identity stays out of ML |
| Gaps and live speech collide | Gap-fill ≠ live-drive can differ |

Authority stays:

```text
measured video/words  →  layer ML  →  next layer  →  TickPackage[t]  →  NWR
```

---

## 2. Layer stack (sections)

```text
                    words / chat / audio
                            │
                            ▼
L1  SpeechClockML      audio+text → viseme/word @ 60 Hz
                            │
                            ▼
L2  LookDriveML        emotion+speech → smile/open/surprise amounts
                            │
              video dense track (measured teacher)
                            │
                            ▼
L3  FaceMotionML       speech+look+prior → whole-face cell motion
                            │              (complete gaps / live ticks)
                            ▼
L4  TickCodecML        face patch ↔ compact c_t  (Side A bandwidth)
                            │
                            ▼
L5  GapPriorML         (optional) temporal inpaint only where conf low
                            │
                            ▼
                   TickPackage[t]  →  CHORUS push → NWR master
```

**Not an ML layer:** identity photo, Master Lock, `smile.png` / atlas pixels.  
Those are resident LOOK/FIELD assets; ML only outputs **drives / motion / codes**.

---

## 3. Each layer — I/O contract

### L1 — SpeechClockML

| | |
| --- | --- |
| **Section** | User words / chat / audio → time |
| **In** | audio features, text/chat tokens, optional lip proxy |
| **Out** | `viseme[t]`, `word[t]`, confidence @ 60 Hz |
| **Teacher** | script beat windows + **audio-energy force-align** (lab); MFA when available |
| **Must not** | move cells or paint RGB |

### L2 — LookDriveML

| | |
| --- | --- |
| **Section** | Which photographed look, how much |
| **In** | emotion tag, viseme, audio energy, catalog priors |
| **Out** | `smile_amt`, `open_amt`, `surprise_amt`, brow… |
| **Teacher** | expression_catalog peaks + measured width/open curves |
| **Must not** | invent new plate pixels |

### L3 — FaceMotionML (main tick body)

| | |
| --- | --- |
| **Section** | Whole-face cell motion every tick |
| **In** | L1+L2, previous motion, face ROI code / coarse landmarks |
| **Out** | face cell velocity/displacement patch (or per-cell channels) |
| **Teacher** | **Side B Face Cell Timeline** (dense measured) |
| **Role** | fill gaps + drive **live** chat not in the video |
| **Must not** | rewrite locked identity / albedo |

### L4 — TickCodecML

| | |
| --- | --- |
| **Section** | Bandwidth (Side A) |
| **In** | full face patch **or** compact `c_t` |
| **Out** | the other side (encode / decode) |
| **Teacher** | **PCA** on measured patches (phase-1 codec); AE is a future upgrade |
| **Role** | make push ≪ 251 MB/s while decoding to L3 quality |

### L5 — GapPriorML (optional)

| | |
| --- | --- |
| **Section** | Only low-confidence cells/ticks |
| **In** | neighbors in space/time, conf mask |
| **Out** | inpainted motion where tracker failed |
| **Teacher** | held-out measured ticks with synthetic holes |
| **Must not** | run on high-confidence measured cells |

---

## 4. How a tick is produced

```text
TRAIN / DIGEST
  Side B measures FaceCellTimeline
  Train L1…L5 on that avatar (replace on new upload)

LIVE TICK t (~16.7 ms)
  L1(speech) → L2(look) → L3(motion) → L4(encode c_t)
  if conf low: L5 patch holes
  CHORUS push → NWR expand/apply
```

When a measured package exists for `t` (replay mode):  
**prefer measured** → L3/L5 only where `conf` is low.

---

## 5. Abstract data layers (independent retrain)

Layers **never** call each other’s weights. They only pass **typed abstract
packets**. That is what makes independent retrain possible.

```text
L1 ──SpeechClock──► L2 ──LookDrive──► L3 ──FaceMotion──► L4 ──TickCode──► NWR
                         ▲                ▲
                         │                │
                    EmotionTag      MotionPrior / conf
```

### 5.1 Packet contracts (stable APIs)

| Packet | Schema (design) | Produced by | Consumed by |
| --- | --- | --- | --- |
| `SpeechClock` | `schema` + `tick, viseme, word, conf, audio_feat?` | L1 | L2, L3 |
| `LookDrive` | `schema` + `tick, smile, open, surprise, brow, conf` | L2 | L3, shader amounts |
| `FaceMotion` | `tick, face_box, vx/vy patch or cell list, conf` | L3 / Side B | L4, L5 |
| `TickCode` | `tick, c_t[], codec_id` | L4 encode | CHORUS / NWR |
| `GapMask` | `tick, conf_map` | Side B tracker | L5, L3 |
| `EmotionTag` | `label, strength` | chat / biomechanics | L2 |

Rules:

1. **Versioned schemas** — phase-1 dataclasses in `aiface.tickfeed.ml.packets`
   use `aiface.packet.SpeechClock.v1` / `LookDrive.v1` / `FaceMotionCode.v1`.
2. **No private tensors** across layers — only packets.  
3. A layer may be swapped if it still reads/writes the same packets.  
4. **Teacher data** for each layer is stored beside the world so that layer
   can retrain alone.

### 5.2 Independent retrain

| Retrain only… | Needs teachers | Frozen |
| --- | --- | --- |
| L1 | audio ↔ aligned words | L2–L5 |
| L2 | SpeechClock + catalog look curves | L1, L3–L5 |
| L3 | SpeechClock + LookDrive + FaceCellTimeline | L1, L2, L4 |
| L4 | FaceMotion patches ↔ codes | L1–L3, L5 |
| L5 | FaceMotion with holes ↔ full | L1–L4 |

Pipeline at tick time is always:

```text
packets in → layer → packets out
```

Not:

```text
L3.forward_into_L4_internal_state()   // forbidden
```

### 5.3 World dir layout (design)

```text
world/
  ml/
    l1_speech_clock.joblib|+meta   + teachers/
    l2_look_drive.…
    l3_face_motion.…
    l4_tick_codec.…
    l5_gap_prior.…   (optional)
  packets/   (optional recorded packet logs for replay/QA)
  face_cell_timeline/   (Side B measured — L3/L4/L5 teacher)
```

`retrain --layer l3` replaces only L3 weights; runtime loads whatever
implements the packet interfaces.

---

## 6. Retrain policy (when)

| Upload change | Retrain |
| --- | --- |
| New take, same face | L3 (+ L4, L5); L1 if voice/language shifts |
| New words only | L1 fine-tune; L3 live head |
| New face / new world | Full stack + Side B collect again |
| Codec/bandwidth only | L4 only |

---

## 7. Open choices

1. L3 outputs **dense patch** or **group+delta** then upsample?  
2. L4 shared with CHORUS payload format?  
3. Shared L1 across avatars vs per-world L1?  
4. Packet bus in-process only, or also logged to CHORUS?

**No implementation until green-lit.**
