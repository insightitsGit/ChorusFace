# TickFeed implementation handoff — gaps vs design

**Branch:** `tickfeedmaster`  
**Rule:** [`TickFeedDesign.md`](TickFeedDesign.md) is the authority. This doc
lists where **code does not yet match that design**. Do not “fix” by
weakening the design.

**Status snapshot:** Demo plays measured timeline before zero-mood; `source[]`
gates authority; HELLO/flags say `disp_vs_rest`; `lid_amt` on the wire; L1 uses
WAV RMS when available; mid-band plate blur hard-snapped; lip-sync uses absolute
audio-clock spans; local lane-B CHUNK reassemble is runtime-wired. Remaining
operator/research: multi-host CHORUS recv, MFA, stronger tracker, new capture take.

---

## Fix order (remaining)

| # | Severity | Item | Why |
| --- | --- | --- | --- |
| 1 | **P1** | Multi-host CHORUS fabric recv | Local lane-B reassemble is wired; remote pod still open |
| 2 | **P1** | Lab MFA | Energy force-align + Whisper-words when keyed |
| 3 | **P2** | New calibration take | True neutral rest + tongue TH |
| 4 | **P2** | Tracker beyond Farneback | Teacher FIELD quality ceiling |
| 5 | **P2** | L4 AE if PCA insufficient | Phase-1 PCA OK for bandwidth demos |

---

## Closed in fidelity pass (do not re-open without evidence)

| Item | Fix |
| --- | --- |
| A1 Demo skipped measured pass | `_tickfeed_calibration_active` → measured ticks then zero-mood |
| A2 `source[]` write-only | Driver loads + lowers conf / allows L5 on synth |
| A4 Velocity vs displacement | HELLO `disp_vs_rest` + `FLAG_VS_REST` on KEY/Δ |
| A5 Absolute KEY default | Default KEY→Δ; `AIFACE_TICKFEED_ABSOLUTE=1` QA-only |
| A6 Wire default `code` | Default `package` (app + run_tickfeed_demo) |
| A7 Miss LOOK from producer | `last_applied_labels` freeze |
| B1 `lid_amt` dead | Packed in TickLabels; look_drive + collect lid measure |
| L1 audio proxy as “audio” | `audio_feat.npz` from WAV RMS; meta `audio_feat_source` |
| L5 wrong hole task | Punch holes in patches → recover PCA codes |
| Train-only metrics | Holdout split in `tickfeed_ml.meta.json` |
| Live FIELD synth-first | L3 primary when motion present; synth fallback |
| Mid-band transition blur | Hard snap + shader `step` ownership |
| Lip-sync hold drift | Absolute `due_at` spans; TickFeed `minimum_hold=0` |
| Lane-B reassemble runtime | `pull_package_from_lane_b_frames` on transport |

---

## Still open (honest)

### CHORUS consume (P1 vs status)

**Design:** one-way fabric push; lane A/B.  
**Impl:** push + memory/`_latest_*` / spool + **local** CHUNK reassemble
(`pull_package_from_lane_b_frames`). Multi-host fabric recv + HELLO_ACK
remains operator.

**Status wording:** Push Done (lab); local reassemble Done; remote consume open.

### Operator-owned

| Item | Notes |
| --- | --- |
| New calibration take | `AvatarCalibrationPrompt.md` |
| Lab MFA | Beyond Whisper-words / energy force-align |
| Multi-host HELLO_ACK + remote master | Separate pod |
| Tracker beyond Farneback | Mesh / UV / 3DMM+residual |
| L4 autoencoder | When PCA quality insufficient |

---

## Truly done (keep)

- TickPackage KEY/DELTA/EMPTY, f16, sparse/dense, CRC = header[0..35]+body  
- `FLAG_VS_REST` + HELLO `disp_vs_rest`  
- GPU ingest B1+B2 + Master Lock  
- Legacy ±4 disabled  
- Collect Farneback → 60 Hz + `source` provenance + optional `lid`  
- Measured calibration pass then zero-mood  
- Label-driven LOOK; miss freezes applied labels  
- Mouth §14.2–§14.3 sync/transition tools  
- L1–L5 train/load with holdout metrics; L4 PCA; L5 patch-hole recover  

---

## Related docs

| Doc | Role |
| --- | --- |
| [`TickFeedDesign.md`](TickFeedDesign.md) | Design authority |
| [`TickPackageHandshake.md`](TickPackageHandshake.md) | Bytes + status |
| [`MultiLayerTickML.md`](MultiLayerTickML.md) | L1–L5 intent |

*End of handoff.*
