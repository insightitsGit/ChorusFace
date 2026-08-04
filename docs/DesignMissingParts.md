# TickFeed implementation handoff — gaps vs design

**Branch:** `tickfeedmaster`  
**Rule:** [`TickFeedDesign.md`](TickFeedDesign.md) is the authority. This doc
lists where **code does not yet match that design**. Do not “fix” by
weakening the design.

**Status snapshot:** Demo plays measured timeline before zero-mood; `source[]`
gates authority; HELLO/flags say `disp_vs_rest`; `lid_amt` on the wire; L1 uses
WAV RMS when available; mid-band plate blur hard-snapped; lip-sync uses absolute
audio-clock spans; local lane-B CHUNK reassemble is runtime-wired; multi-host
recv spool + master target landed; Whisper-words teacher when keyed; DIS dense
tracker default; L4 AE upgrades when PCA holdout MAE is insufficient.

Operator-owned remaining: **drop dense-kit MP4 v3** (script already in code:
true neutral REST + wide OPEN + `TONGUE_TH` “think” + deliberate `BLINK`).

---

## Fix order (remaining)

| # | Severity | Item | Why |
| --- | --- | --- | --- |
| 1 | **P2** | Drop dense+blink calibration MP4 | Script/prompt v3 ready — regenerate with BLINK then `--clean` rebuild |

---

## Closed (do not re-open without evidence)

| Item | Fix |
| --- | --- |
| A1 Demo skipped measured pass | `_tickfeed_calibration_active` → measured ticks then zero-mood |
| A2 `source[]` write-only | Driver loads + lowers conf / allows L5 on synth |
| A4 Velocity vs displacement | HELLO `disp_vs_rest` + `FLAG_VS_REST` on KEY/Δ |
| A5 Absolute KEY default | Default KEY→Δ; `CHORUSFACE_TICKFEED_ABSOLUTE=1` QA-only |
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
| Multi-host CHORUS recv | `chorus_master` TargetPod → recv spool; `pull_recv_*` |
| Whisper Side B teacher | `force_align` → `whisper_words_force_align` when keyed |
| Tracker beyond Farneback | DIS default (`CHORUSFACE_TICKFEED_FLOW=dis`) |
| L4 AE when PCA weak | `fit_l4_codec` upgrades; force `CHORUSFACE_TICKFEED_L4_AE=1` |

---

## Still open (honest)

### Operator-owned

| Item | Notes |
| --- | --- |
| Dense take v3 | Landed (`second_avatar_calibration.mp4`); atlas α still weak vs `open.png` |
| Full lab MFA phoneme align | Whisper words is the keyed teacher; Montreal MFA optional later |
| Multi-host HELLO_ACK ceremony | Recv path works; separate ACK ceremony still optional ops |

---

## Truly done (keep)

- TickPackage KEY/DELTA/EMPTY, f16, sparse/dense, CRC = header[0..35]+body  
- `FLAG_VS_REST` + HELLO `disp_vs_rest`  
- GPU ingest B1+B2 + Master Lock  
- Legacy ±4 disabled  
- Collect DIS/Farneback → 60 Hz + `source` provenance + optional `lid`  
- Measured calibration pass then zero-mood  
- Label-driven LOOK; miss freezes applied labels  
- Mouth §14.2–§14.3 sync/transition tools  
- L1–L5 train/load with holdout metrics; L4 PCA/AE; L5 patch-hole recover  
- CHORUS push + local reassemble + master recv spool  

---

## Related docs

| Doc | Role |
| --- | --- |
| [`TickFeedDesign.md`](TickFeedDesign.md) | Design authority |
| [`TickPackageHandshake.md`](TickPackageHandshake.md) | Bytes + status |
| [`MultiLayerTickML.md`](MultiLayerTickML.md) | L1–L5 intent |

*End of handoff.*
