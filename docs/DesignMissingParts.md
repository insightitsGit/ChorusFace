# TickFeed implementation handoff — gaps vs design

**Branch:** `tickfeedmaster`  
**Rule:** [`TickFeedDesign.md`](TickFeedDesign.md) is the authority. This doc
lists where **code does not yet match that design**. Do not “fix” by
weakening the design.

**Status snapshot:** Architecture pieces exist (codec, GPU ingest, ring,
collect, ML joblibs, LOOK path). Several **DONE** checklist rows were
overclaimed: measured timeline is bypassed at demo boot, provenance is
write-only, CHORUS consume is in-process self-pull, blink `lid_amt` is
unwired.

---

## Fix order (do these first)

| # | Severity | Item | Why |
| --- | --- | --- | --- |
| 1 | **P0** | Demo plays measured 8s timeline before zero-mood | Boot enters `PRESENCE_ZERO` → `live_speech=True` every tick; Side B teachers never own LOOK/FIELD |
| 2 | **P0** | Honor per-tick `source[]` at runtime | Provenance written on disk; driver never loads/gates on it |
| 3 | **P0** | Align displacement vs velocity meaning | Collect stores rest→frame displacement; HELLO claims `velocity_write` |
| 4 | **P0/P1** | Real CHORUS consume **or** demote status tables | Push + `_latest_*` memory loop ≠ fabric receive; `reassemble_lane_b_chunks` tests-only |
| 5 | **P0/P1** | Wire `lid_amt` end-to-end **or** un-DONE blink | App reads `labels.lid_amt` but labels have no field; `lid_measure.py` not in prepare |
| 6 | **P1** | Wire-loop fidelity default = `package` | Default `code` expands lossy PCA and discards lane-B KEY/Δ body |
| 7 | **P1** | Miss-path LOOK freeze | Plates skip on `pkg is None`; LayerCommand still rebuilds from producer `last_labels` |
| 8 | **P2** | Spool / absolute-KEY lab defaults | Absolute KEY every tick + spool I/O; OK as lab QA, wrong as “Δ steady Done” |

---

## A. Core contract / Side A–B fidelity

### A1 — Demo never consumes measured timeline (P0)

**Design:** §5.3 / §10 — measured `FaceCellTimeline` is authority; one
calibration pass then REST. B4 — TickPackage labels sole LOOK.

**Impl:** `--demo` calls `_enter_zero_state` immediately (`app.py` ~739).
While `PRESENCE_ZERO`, `_apply_zero_mood_overlay` sets `_tickfeed_live`
mode `"zero"` → every tick `push_drives(..., live_speech=True)`
(`app.py` ~3690). With `live_speech=True`, teacher `look_by_tick` /
`speech_by_tick` are skipped (`driver.py` ~207–225); closed REST zeros
FIELD; `past_end` never runs (requires `not live_speech`).

**Boot message lies:** prints “one 8s calibration pass, then 0-state”
then skips the pass.

**Fix direction:** For ticks `[0, timeline_length)` push with
`live_speech=False` from measured packages. Enter zero-mood **only after**
the pass (or explicit idle). Chat/TTS may overlay with `live_speech=True`
during its window only.

---

### A2 — `source` provenance write-only (P0)

**Design:** §5.3 / §10 — per-tick `source` (0=measured, 2=synth); never
sell synth as measured.

**Impl:** Collect + `timeline_io` write `source`. Driver loads velocity /
conf only — never `bundle["source"]` (`driver.py` ~128–141). No runtime
gate.

**Fix direction:** Load `source[]`. Refuse measured-authority / high-conf
when `source ≠ 0`. L5 / ML fill only on low-conf or synth ticks.

---

### A3 — CHORUS consume is in-process self-loop (P0 vs status)

**Design:** §6.2 — one-way fabric push; lane A `c_t`, lane B framed TPK /
TPK_REF. Header claims “Remote CHORUS transport … Done (lab)”.

**Impl:** `push_*` sets `_latest_code` / `_latest_package` then optional
`send_direct`. `pull_latest_*` reads that memory (or spool file).
`ingest_from_wire` uses those pulls (`driver.py` ~563–582).
`reassemble_lane_b_chunks` is **tests-only** — no fabric receive path.

**Fix direction:** Either (a) implement a real consumer that recv’s
vectors → reassemble / TPK_REF spool → ring, or (b) change status to
**Push Done / Consume Partial (memory loop)** and keep multi-host as open.

---

### A4 — Phase-1 meaning: displacement stored, velocity claimed (P0)

**Design tension:** packages / HELLO say `velocity_write`; collect Farneback
is rest→frame **displacement** (`collect.py`). Absolute KEY every tick
masks Δ-of-displacement bugs.

**Fix direction:** Pick one locked meaning. If displacement: set HELLO
`apply_mode`, docs, and optionally `FLAG_VS_REST`. If velocity: convert
collect to tick-to-tick Δ before packaging.

---

### A5 — Lab absolute KEY bypasses KEY→Δ steady (P1)

**Design:** §6.3 — KEY then Δ; refresh ~2s.

**Impl:** `AIFACE_TICKFEED_ABSOLUTE` defaults **on** → KEY every tick
(`driver.py` ~415–440). Bandwidth Δ is opt-out.

**Fix direction:** Default Δ for fidelity claims; absolute only as
explicit QA env.

---

### A6 — Wire-loop defaults to lossy `c_t` (P1)

**Design:** Lane B = KEY/Δ fidelity; lane A = compact bandwidth.

**Impl:** `--wire-loop-source` default `"code"` → PCA expand; lane-B body
also pushed but ignored for apply.

**Fix direction:** Fidelity demo default `package`; label `code` as
bandwidth-only.

---

### A7 — Miss path LOOK can use producer labels (P1)

**Design:** §6.5 — on miss, damp FIELD; do not invent LOOK from producer
sidecar.

**Impl:** `_apply_tickfeed_labels_to_look` returns if `pkg is None`.
`_layer_command_from_tickfeed` still reads `last_labels` (producer).

**Fix direction:** Freeze LOOK from last **applied** package only.

---

### A8 — HELLO is lab self-ACK (P2 if labeled honestly)

**Design:** Remote ACK open; lab self-ACK allowed.

**Impl:** `negotiate_hello(hello)` in-process; blobs pushed as artifacts.

**Fix direction:** Keep “lab self-ACK”; do not mark remote ACK plane Done.

---

### A9 — Ring producer-lead only in wire-loop (P2 doc)

**Design / §12:** implies `produce_tick = master + RING_DEPTH`.

**Impl:** Lead only when `wire_loop`; local default same-tick
(`app.py` ~3692–3697).

**Fix direction:** Split §12 / checklist into local-ring vs wire-loop rows.

---

## B. Post-initial §14 — claimed vs incomplete

| Item | Design | Impl reality | Sev |
| --- | --- | --- | --- |
| Absolute overlay / closures / media_time | §14.2 | Present (`speech_overlay_until`, `min_hold=0`, bilabial helpers) | OK |
| Mouth transition + FIELD mute | §14.3 | Present (`_update_mouth_transition`, oral-disk mute in frag) | OK |
| Zero moods | §14.4 | Exist but **steal** Side B at boot (see A1) | **P0** order |
| Plate rebuild / denser Farneback | §14.5 | Scripts/tools present | OK (teacher quality still Farneback-limited) |
| Blink SM + eye-disk FIELD mute | §14.7 | SM + mute largely real | OK-ish |
| Blink `lid_amt` teacher | §14.7 | **Dead** — no label field; `lid_measure` not in prepare | **P0/P1** |
| Eyes-closed plate | §14.7 | Uncommitted WIP (`bake_eyes_closed_plate.py`, frag sampler); without plate holds open photo | **P1** |

### B1 — `lid_amt` teacher dead (P0/P1)

App reads `getattr(labels, "lid_amt", 1.0)` → always default open →
`_tickfeed_lid_teacher` never true. `TickLabels` has `brow_amt` + reserved
bytes, not `lid_amt`. `lid_measure.py` exists but is not called from
collect / look_drive / timeline_io.

**Fix direction:** Pack lid into labels (or look_drive → labels), measure
in prepare, drive L09 from package when conf high.

---

## C. Demo / ML fidelity notes

| Issue | Sev | Notes |
| --- | --- | --- |
| Live speech FIELD = synth (+ optional ML) | P1 | Honest if labeled live; must not stamp measured conf |
| L1 runtime uses open/smile proxy, not WAV | P1 | Offline energy force-align teacher yes; live SpeechClock overclaimed |
| L4 PCA (not AE) | P2 | Design allows phase-1 PCA |
| Spool KEY spam / glob on cold pull | P2 | Lab ops; don’t rely on NTFS glob per tick for fidelity |
| Jaw still uploaded under TickFeed | P2 | LayerCommand jaw=0; still uploads `state.jaw_angle` |

---

## D. Doc status corrections needed

Update these when code catches up (or demote now):

| Claim | Where | Should read |
| --- | --- | --- |
| Remote CHORUS transport Done (lab) | TickFeedDesign header | Push Done; consume = memory/spool loop until fabric recv |
| §12 all Yes / connection implemented | TickFeedDesign §12 | Qualify: demo path bypasses measured timeline |
| Provenance Done | this file (old) | Written; **unused at authority** until A2 |
| CHORUS lane B Done | checklists | Encode/push yes; consume no |
| Blink band Done | this file (old) | SM + mute yes; `lid_amt` / plate incomplete |
| Ring lead Done as always `master+depth` | §12 | Wire-loop only |

Handshake §7 (“KEY/Δ on remote CHORUS Partial”) is **more honest** than the
design header — prefer that wording until consume exists.

---

## Truly done (keep; do not re-litigate)

- TickPackage KEY/DELTA/EMPTY, f16, sparse/dense, CRC = header[0..35]+body  
- GPU ingest B1+B2 (`tick_ingest.comp`) sparse/dense/EMPTY + Master Lock  
- Legacy ±4 disabled in app speech path  
- Collect Farneback → 60 Hz + disk `source` provenance  
- Label-driven LOOK path exists (when not overridden by zero-mood live)  
- Mouth §14.2–§14.3 sync/transition tools largely real  
- L2–L5 sklearn artifacts train/load; L4 PCA  

---

## Operator-owned (not in-repo fakes)

| Item | Notes |
| --- | --- |
| New calibration take | True neutral rest + tongue-visible TH — `AvatarCalibrationPrompt.md` |
| Lab MFA | Beyond Whisper-words / energy force-align |
| Multi-host HELLO_ACK + remote master | Separate pod; not self-ACK |
| Tracker beyond Farneback | Mesh / UV / 3DMM+residual research |
| L4 autoencoder | When PCA quality insufficient |

---

## Uncommitted WIP (at handoff time)

| Piece | Note |
| --- | --- |
| `bake_eyes_closed_plate.py` + gate test | Staged; refuse bake without blink evidence |
| `lid_measure.py` | Staged; **not wired** into prepare |
| `app.py` / `avatar.frag` eyes-closed path | Plate owns lids only if ready; `lid_amt` still missing |
| `io_limits.py` / tests | Related ops limits; verify before merge |

---

## Related docs

| Doc | Role |
| --- | --- |
| [`TickFeedDesign.md`](TickFeedDesign.md) | Design authority (§1–§13 core, §14 polish) |
| [`TickPackageHandshake.md`](TickPackageHandshake.md) | Bytes + (more honest) status table |
| [`MultiLayerTickML.md`](MultiLayerTickML.md) | L1–L5 intent |
| [`PhoneticFidelity.md`](PhoneticFidelity.md) | Lip-reading / sync inventory |

*End of handoff. Implementation work should close A1→A4 before claiming
demo fidelity again.*
