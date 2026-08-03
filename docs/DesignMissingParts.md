# Design session — missing parts (focus list)

**Status:** Side A + Side B on `tickfeedmaster` — see three-plane status in
[`TickFeedDesign.md`](TickFeedDesign.md).  
**Master narrative:** [`TickFeedDesign.md`](TickFeedDesign.md)
(§1–§13 initial design · §14 post-initial mouth improvements).

---

## Initial design checklist (architecture)

| Design item | Status | Code |
| --- | --- | --- |
| TickPackage KEY/DELTA f16 sparse/dense/empty | **DONE** | `package.py` |
| CRC = header[0..35] + body | **DONE** | `package.encode/decode` |
| HELLO / HELLO_ACK negotiate | **DONE** (lab self-ACK) | `negotiate_hello` |
| GPU ingest KEY/Δ/sparse/EMPTY + lock | **DONE** | `tick_ingest.comp` |
| 3-tick ring with producer lead | **DONE** | `ring.py` + `app._simulate_tick` |
| Labels sole LOOK authority | **DONE** | TickFeed path in `app.py` |
| CHORUS lane A (`c_t`) | **DONE** | `push_code` |
| CHORUS lane B (TickPackage frames / TPK_REF) | **DONE** | `push_package_bytes` |
| Measured timeline provenance | **DONE** | `source` per tick in timeline |
| L1 energy force-align (lab teacher) | **DONE** | `force_align.py` |
| L4 PCA TickCodec (phase-1) | **DONE** | `ml/` (AE = future) |
| Legacy ±4 disabled | **DONE** | MouthCellPlan no-op |

---

## Post-initial design checklist (after B1–B4)

Landed **after** the architecture above was working. Does not change the
TickPackage contract. Full narrative: [`TickFeedDesign.md`](TickFeedDesign.md) §14.

| Item | Status | Code / notes |
| --- | --- | --- |
| Absolute LOOK overlay until (audio clock) | **DONE** | `speech_overlay_until` |
| Closures never skipped (PP/MM/CLOSED) | **DONE** | `app._fire_impulse` |
| Velocity-aware FIELD mute + transition owner | **DONE** | `_mouth_transition`, field gain |
| Playback `media_time` viseme clock | **DONE** | `audio.*Sink.media_time` |
| Bilabial onset pin + energy valley snap | **DONE** | `tts.bias_*`, `snap_bilabials_*` |
| Whisper words default when API key set | **DONE** | `--tts-align words` |
| Zero moods (neutral / smile / waiting) | **DONE** | key Z / `/calibrate` |
| Priority distinct plate bank + rebuild script | **DONE** | `rebuild_tickfeed_plates.py` |
| Denser Farneback for teacher timeline | **DONE** | `collect.py` + `--timeline` |
| Quiet demo logs (FPS) | **DONE** | `--gpu-log` / `--tickfeed-debug` opt-in |
| Blink state + eye-disk FIELD mute + hard lids | **DONE** | `eyes.py`, `avatar.frag` L09 (§14.7) |

---

## Operator-owned / future

| Item | Notes |
| --- | --- |
| Exact Gemini 8s MP4 / new take | `AvatarCalibrationPrompt.md` — need true neutral rest + tongue TH |
| Lab MFA | Energy + Whisper-words ship; MFA when WAV+dict available |
| Production multi-host HELLO_ACK | Lab self-ACK; remote ACK when master is a separate pod |
| L4 autoencoder upgrade | PCA is phase-1; design allows AE later |
| Dense tracker beyond Farneback | Farneback denser params landed; mesh/UV trackers still open |

---

## Related docs

| Doc | Piece |
| --- | --- |
| [`TickFeedDesign.md`](TickFeedDesign.md) | Master (§14 = post-initial) |
| [`TickPackageHandshake.md`](TickPackageHandshake.md) | Bytes + status table |
| [`MultiLayerTickML.md`](MultiLayerTickML.md) | L1–L5 |
| [`PhoneticFidelity.md`](PhoneticFidelity.md) | Lip-reading inventory + sync notes |
