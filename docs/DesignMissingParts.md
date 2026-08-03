# Design session — missing parts (focus list)

**Status:** Side A + Side B on `tickfeedmaster` — see three-plane status in
[`TickFeedDesign.md`](TickFeedDesign.md).  
**Master narrative:** [`TickFeedDesign.md`](TickFeedDesign.md).

---

## Implementation checklist

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
| L1 energy force-align (lab) | **DONE** | `force_align.py` |
| L4 PCA TickCodec (phase-1) | **DONE** | `ml/` (AE = future) |
| Legacy ±4 disabled | **DONE** | MouthCellPlan no-op |

---

## Operator-owned / future

| Item | Notes |
| --- | --- |
| Exact Gemini 8s MP4 | `AvatarCalibrationPrompt.md` |
| Lab MFA | Energy force-align ships; MFA when WAV+dict available |
| Production multi-host HELLO_ACK | Lab self-ACK; remote ACK when master is a separate pod |
| L4 autoencoder upgrade | PCA is phase-1; design allows AE later |
| Dense tracker beyond Farneback | §11 open research |

---

## Related docs

| Doc | Piece |
| --- | --- |
| [`TickFeedDesign.md`](TickFeedDesign.md) | Master |
| [`TickPackageHandshake.md`](TickPackageHandshake.md) | Bytes + status table |
| [`MultiLayerTickML.md`](MultiLayerTickML.md) | L1–L5 |
