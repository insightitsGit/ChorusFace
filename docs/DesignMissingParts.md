# Design session — missing parts (focus list)

**Status:** Side A + Side B design implemented on `tickfeedmaster`.  
**Master narrative:** [`TickFeedDesign.md`](TickFeedDesign.md).

---

## Implementation checklist (word-level)

| Design item | Status | Code |
| --- | --- | --- |
| TickPackage KEY/DELTA f16 sparse/dense/empty | **DONE** | `package.py` |
| HELLO / HELLO_ACK negotiate | **DONE** | `build_hello` / `negotiate_hello` |
| Sparse Δ + conf on wire | **DONE** | `build_delta` + encode |
| GPU ingest KEY/Δ/sparse/EMPTY + lock + f16 | **DONE** | `tick_ingest.comp` |
| 3-tick ring → master pop → GPU (miss damp) | **DONE** | `driver.pop_for_master` + `app._simulate_tick` |
| Labels → LOOK amounts (B4 sole authority) | **DONE** | `app._apply_tickfeed_labels_to_look` + plate sync |
| FIELD warp from velocities | **DONE** | `avatar.frag` |
| CHORUS one-way + spool + `c_t` pull/expand | **DONE** | `chorus_transport` + `expand_code_to_package` |
| Local CHORUS plane launcher | **DONE** | `scripts/start_chorus_local.py` + `run_tickfeed_demo.py` |
| L4 `c_t` encode/decode | **DONE** | `ml/runtime.py` |
| L1–L5 train + live | **DONE** | `ml/` + train script |
| 8s `calibration_script.json` | **DONE** | `calibration.py` |
| Every-frame dense flow → 60 Hz interpolate | **DONE** | `collect.py` |
| `face_cell_timeline/` layout (§4.5) | **DONE** | `timeline_io.py` |
| `speech_align.json` (audio energy force-align) | **DONE** | `force_align.py` (ffmpeg→WAV→RMS@60Hz) |
| `look_drive.json` | **DONE** | `speech_align.py` |
| `qa_report.json` | **DONE** | written by `timeline_io` |
| Per-cell confidence | **DONE** | timeline + KEY/Δ |
| Cosmetics prefs + GLSL grade | **DONE** | `cosmetics.py` + `avatar.frag` uniforms |
| Legacy ±4 disabled | **DONE** | MouthCellPlan no-op |
| Face-box ROI registration | **DONE** | crop-scale UV contract in `collect.py` |

---

## Operator-owned (cannot be faked in-repo)

| Item | Notes |
| --- | --- |
| Exact Gemini 8s MP4 | Prompt in `AvatarCalibrationPrompt.md`; dual takes under `assets/avatar_video_inputs/calibration_takes/` |
| Lab MFA (Montreal Forced Aligner) | Audio-energy force-align ships; swap MFA when a full lab WAV+dict pipeline is available |
| Production CHORUS cluster | Local plane via `start_chorus_local.py`; fabric when up, else spool |

---

## Related docs

| Doc | Piece |
| --- | --- |
| [`TickFeedDesign.md`](TickFeedDesign.md) | Master |
| [`TickPackageHandshake.md`](TickPackageHandshake.md) | Bytes |
| [`SideB_VideoCellCollection.md`](SideB_VideoCellCollection.md) | Collect |
| [`MultiLayerTickML.md`](MultiLayerTickML.md) | L1–L5 |
| [`AvatarCalibrationPrompt.md`](AvatarCalibrationPrompt.md) | Gemini prompt |
