# Design session — missing parts (focus list)

**Status:** implemented on `tickfeedmaster` (see checklist below).  
**Master narrative:** [`TickFeedDesign.md`](TickFeedDesign.md).

**Build stance:** TickFeed is the sole FIELD path. Legacy ±4 MouthCellPlan is
disabled. Identity LOOK plates stay photographed.

**Locked transport:** ROI = **full face**; **keyframe once**, then **deltas only**
(~0.3–2 MB/s phase-1 target).

---

## Implementation status

| Area | Status | Where |
| --- | --- | --- |
| TickPackage KEY/DELTA f16 sparse/dense/empty | **DONE** | `aiface.tickfeed.package` |
| GPU ingest + face_box map | **DONE** | `tick_ingest.comp` + `FieldRuntime` |
| Ring + damp-on-miss | **DONE** | CPU ring + GPU enc=4 |
| Labels → LOOK amounts | **DONE** | package labels → app uniforms |
| `calibration_script.json` + 8s beats | **DONE** | `aiface.tickfeed.calibration` |
| Gemini/user calibration prompt | **DONE** | `docs/AvatarCalibrationPrompt.md` |
| Dense UV-flow teacher | **DONE** | Farneback in `collect.py` |
| `face_cell_timeline.npz` @ 60 Hz + conf | **DONE** | Side B prepare |
| Beat QA (motion vs windows) | **DONE** | `aiface.tickfeed.qa` |
| Upload scaffolding validate | **DONE** | `validate_calibration_take` |
| L1–L5 packets + train/runtime | **DONE** | `aiface.tickfeed.ml` |
| Per-layer retrain `--layer` | **DONE** | `scripts/train_tickfeed_ml.py` |
| L4 `c_t` encode/decode | **DONE** | PCA codec in L4 |
| CHORUS one-way push + spool fallback | **DONE** | `chorus_transport.py` |
| Cosmetic prefs beside world | **DONE** | `aiface.tickfeed.cosmetics` |
| App hot path TickFeed only | **DONE** | `app.py` |

---

## Still external / operator-owned

| Item | Notes |
| --- | --- |
| Real Gemini-generated **exact** 8s MP4 | Prompt is ready; generate/save as `calibration_take.mp4` then `--prepare` |
| Live CHORUS control plane | Uses `chorus-fabric` when `localhost:50051` (or configured) is up; else spool |
| Force-aligned WAV transcript | L1 currently uses audio-proxy features from open/smile when WAV align unavailable |
| Shader wiring of cosmetic tint uniforms | Prefs API + JSON persist; bind to GLSL when LOOK grade path is extended |

---

## Priority order (historical → now)

```text
P0  TickPackage codec                         DONE
P0b GPU ingest + face_box map                 DONE
P0c Ring + damp-on-miss                       DONE
P0d Labels → LOOK amounts                     DONE
P1  Calibration script artifact               DONE
P2  Dense UV-flow tracker                     DONE (Farneback; mesh/3DMM optional later)
P3  Multi-layer ML L1–L5                      DONE (trained into world/ml/)
P4  CHORUS Fabric wire                        DONE (live or spool)
P5  L4 c_t codec                              DONE
P6  User cosmetics prefs                      DONE (JSON + uniforms dict)
```

---

## Related docs

| Doc | Piece |
| --- | --- |
| [`SideB_VideoCellCollection.md`](SideB_VideoCellCollection.md) | Collect + 8s script |
| [`CellFeedBandwidth.md`](CellFeedBandwidth.md) | 480/241 MB/s + CHORUS |
| [`MultiLayerTickML.md`](MultiLayerTickML.md) | Layers + packets |
| [`AvatarScaffolding.md`](AvatarScaffolding.md) | Lock vs cosmetics |
| [`AvatarCalibrationPrompt.md`](AvatarCalibrationPrompt.md) | Paste-into-Gemini prompt |
