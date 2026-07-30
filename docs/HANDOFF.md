# AIFace agent handoff

**Read this first** when opening a new Cursor window on `C:/code/AIFace`.
Prior chat lived under the NWR workspace (`C:/code/NWR`) while building this
repo; the product code is here now.

| | |
| --- | --- |
| **Repo** | `C:/code/AIFace` → private GitHub `insightitsGit/AIFace` |
| **Origin** | configured; **no commits yet**, **do not push** without explicit permission |
| **Tests** | `330 passed` (as of handoff) — `pip install -e ".[dev]"` then `pytest -q` |
| **Parent project** | `C:/code/NWR` @ `f3ce4f5` (synced) — leave untouched unless the user asks |

---

## Story of how we got here

### Chapter 1 — Avatar inside NWR

Work started in Neural World Runtime: a chat-driven face on a locked 32-channel
GPU field. The face spoke via phonemes → muscle impulses → jaw/eye/emotion
subsystems. Identity was enforced with Master Lock (channel 31). Early rendering
moved **rigid anatomical pieces** (lips, eyes, brows) cut from a photo atlas —
good prototype, visible seams, limited naturalness.

### Chapter 2 — Standalone extraction

The avatar was strong enough to be its own product. Per plan
`extract_standalone_aiface_261d9b62`, everything was extracted to
`C:/code/AIFace`:

- Package `aiface` with CLI `aiface` / `aiface-seed` / `aiface-capture` / `aiface-sync`
- Vendored minimal runtime (BDS, constraint tick, FieldRuntime) — **not** the
  full NWR sandbox
- Synthetic seed path (no real portrait committed)
- Git + private GitHub remote; **commit/push deferred** until the user asks

Extraction plan is done. NWR was left alone.

### Chapter 3 — “More muscles like a real face?”

User asked whether to split into many more pieces (human-like muscle count /
“millions connected to the brain”) or keep the early rigid atlas.

**Decision (user confirmed):** *continuous displacement field + a few occlusion
layers + an expanded ~39-muscle set* — **not** a million cut-out objects, and
**not** staying on rigid pieces.

Rationale:

- Real expression uses ~40 named muscles pulling **one continuous skin sheet**
- More rigid pieces → more seams; naturalness comes from shared influence
- True discontinuities (mouth open, lids, jaw) stay as explicit occlusion layers

### Chapter 4 — Displacement-field implementation (current HEAD of work)

Implemented end-to-end:

1. **Muscle groups** — systems address `Frontalis`, `OrbicularisOris`, etc.;
   registry expands to left/right members with asymmetry `bias`
2. **`face_definition.json` v2.0** — 39 muscles, `travel`, `gate`, jaw landmarks
3. **`aiface.skinning`** — tissue bake (mobility / mouth side / slit / eye
   aperture), Wendland RBF packer, CPU `displacement_field` mirror of the shader
4. **`avatar.frag`** — inverse-warp continuous deform + mouth cavity + lids
5. **App uniforms** — tissue texture, muscle arrays, jaw profile + lateral span
6. **Biomechanics** — render state from group activations; phoneme/emotion maps
   updated (`JawOpener` vs closers; `ZygomaticusMinor`; negative emotion map)
7. **Tests** — skinning properties, shader contract, runtime; suite green

Tuning that mattered:

- Orphan mobile tissue reduced (~9k → ~72 cells) via soft mobility zones +
  wider muscle radii + ZygomaticusMinor
- Jaw folding fixed: mandible travel uses a **chin rise + neck fade** profile and
  a **lateral feather**, **not** mobility rim scaling (that crushed travel into
  a steep gradient and folded the chin)
- Full all-muscles-at-1.0 co-contraction can still stress the Jacobian; tests
  assert **per-muscle travel ≤ 0.21 × radius** and **solver-reachable** envelopes

### Chapter 5 — TTS demoted, the sync channel promoted

User's call, and it reframed the product: *“any future LLM already has the
voice — we just need to open the channel to sync with audio. Don't delete the
TTS, demote it. Keep it as a test fixture: it's the only way to produce a clip
with known ground-truth alignment, which is how you measure streaming sync
error.”*

So the primary path is now audio this process did **not** produce:

- **`aiface.stream`** — push-based aligner. PCM chunks in, timed viseme decisions
  out, each one recording when it was decided so the latency is measured rather
  than claimed. Energy spends against a per-viseme budget; full stops may only be
  spent by silence; every phrase boundary re-measures the speaker's rate so error
  cannot accumulate past it.
- **`aiface.sync`** — the oracle. Same utterance through the batch path (full
  lookahead, best possible) and the streaming path; reports bias, jitter,
  trimmed p95, decision lag, coverage. Clock-free, so it runs in milliseconds.
- **`aiface-sync`** — CLI wrapper with a pass/fail budget for CI.
- **Bridge routes** `/voice/expect`, `/voice/pcm`, `/voice/end`, answered on the
  request thread because a 20 ms chunk must not wait for a frame.
- **`--tts` now defaults off.** It is the fixture that feeds the oracle.

Current measurement (SAPI, five fixture lines, 20 ms chunks, 50 ms lookahead):
**mean 119 ms / worst 195 ms trimmed p95, 100% coverage**, budget 250 ms. Full
detail in `docs/VoiceSync.md`.

Three findings came out of building it, all fixed:

- The rate estimator could enter a feedback loop: a phrase where the script ran
  ahead measured a rate that made the next phrase run further ahead. A phrase is
  now disqualified from measuring the voice if the voice talked straight through
  the punctuation that ended it.
- A silence with script still owed was read as a hesitation forever, so the last
  word of an utterance landed a second late (worst line: 976 ms p95). Quiet that
  outlasts `long_silence` now ends the phrase and flushes at the moment the voice
  stopped — but only revisits a hesitation it already tolerated, never the next
  sentence. That took the line to 270 ms.
- Defaults were re-swept after those changes (`energy_blend` 0.5 → 0.28,
  `level_halflife` 0.5 → 0.85, `rate_trust` 0.8 → 0.6), scored on worst case as
  well as mean, and checked on held-out lines.

Also fixed while testing the routes: an error reply sent while the request body
was still undelivered dropped the connection instead of returning the error.

---

## Where we stand (architecture)

```
chat text  (in-window chat box, terminal chat>, or POST /speak)
    → aiface.chatbox         panel state, transcript, portrait/panel framing
    → aiface.speech          visemes / LLM / offline fallback
    → aiface.stream          PRIMARY: locks visemes to arriving PCM (/voice/pcm)
    → aiface.tts / audio     fixture: local synthesis + batch alignment (--tts)
    → aiface.biomechanics    39 muscles, groups, jaw/eyes/emotion/idle/breath
    → aiface.skinning        tissue maps + muscle uniform packing
    → aiface.runtime         BDS + constraint tick + Master Lock
    → avatar.frag            continuous displacement warp + occlusion layers
```

**Render path (important):** skin motion is **shader inverse-warp**, not field
advection. The photo (albedo) stays immutable. Master Lock still kills writes on
locked cells. Constraint-only tick (no advection) remains intentional.

**Framing:** the portrait is letterboxed into the `avatar_frame` uniform
(`vec4` xy origin + zw size, in UV) so the chat panel can own the bottom band
without stretching the face. `frame_layout()` keeps the portrait square.
`_preview()` deliberately forces the frame back to full, so bridge clients
always get the portrait alone regardless of window layout.

**Debug views:** F1–F11 (displacement / mobility / tissue gates on later keys).
See HUD in `app.py`.

**Diagnostic script:** `python scripts/probe_field.py` — coverage map, peak
displacement, Jacobian split (muscles vs jaw).

---

## Key files

| Path | Role |
| --- | --- |
| `src/aiface/app.py` | Avatar window; uniforms; tissue upload; chat wiring |
| `src/aiface/chatbox.py` | In-window chat panel: editing, transcript, `frame_layout` |
| `src/aiface/skinning.py` | Tissue bake, packer, jaw pose, CPU field |
| `src/aiface/shaders/avatar.frag` | Displacement + occlusion |
| `src/aiface/speech.py` | Text → visemes, chat backend, conversation memory |
| `src/aiface/stream.py` | **Live sync channel**: arriving PCM → timed visemes |
| `src/aiface/sync.py` | Streaming-vs-batch oracle; the millisecond number |
| `src/aiface/service/bridge.py` | Loopback HTTP surface incl. the `/voice/*` routes |
| `src/aiface/tts.py` | Fixture voice + batch (full-lookahead) alignment |
| `src/aiface/audio.py` | WAVE decode, RMS envelope, voiced intervals, playback |
| `src/aiface/biomechanics/data/face_definition.json` | Character / muscle authoring |
| `src/aiface/biomechanics/muscles.py` | Registry, groups, gates, solver |
| `src/aiface/biomechanics/face.py` | Orchestrator; group → mouth pose |
| `src/aiface/seed.py` | Seed world + tissue + parts |
| `scripts/probe_field.py` | Coverage / Jacobian probe |
| `scripts/tune_voice_sync.py` | Grid-search the streaming defaults against a real voice |
| `scripts/preview_chatbox.py` | Renders the framed window to PNG for layout QA |
| `tests/test_skinning.py` | Field properties |
| `tests/test_shader_contract.py` | Uniform / MAX_MUSCLES contract |
| `tests/test_stream.py` | Chunk invariance, silence handling, rate bounds |
| `tests/test_sync.py` | Oracle statistics + the 250 ms budget on a real voice |
| `docs/VoiceSync.md` | The channel, the oracle, and the measured numbers |

---

## What is done

- [x] Standalone package, CI, README, synthetic demo path
- [x] Minimal FieldRuntime (no NWR sandbox baggage)
- [x] Continuous displacement architecture (chosen approach)
- [x] ~39-muscle definition with groups, gates, bias, travel
- [x] Tissue maps + jaw profile that does not fold under realistic load
- [x] In-window chat frame: portrait letterboxed above a live chat panel
- [x] TTS-driven viseme timing (`--tts`): energy / words / linear alignment
- [x] Windows SAPI offline voice, kept as the oracle's clip fixture
- [x] **Live sync channel** (`aiface.stream`) + `/voice/expect|pcm|end`
- [x] **Sync oracle** (`aiface.sync`, `aiface-sync`) with a CI latency budget:
      119 ms mean / 195 ms worst trimmed p95 on SAPI, 100% coverage
- [x] TTS demoted to opt-in (`--tts`); the channel is the default path
- [x] Natural offline replies (no canned phoneme filler)
- [x] Polished stylized synthetic portrait aligned to muscle UVs
- [x] Test suite green
- [x] GitHub private repo + `origin` remote
- [x] NWR review cross-audit: authority coupling, bind guard, rest-pose saves,
      tissue relaxation (see the cross-audit section below)

---

## NWR power — what we keep, what we leave

**Standing position: NWR is the heart of the product, AIFace is a child of it.**
NWR owns the substrate contract — cell schema, `.bds`, authority ordering,
Master Lock — and AIFace does not redefine any of it. If the two disagree on a
substrate rule, NWR wins and this repository is the bug. AIFace owns the face:
muscles, displacement field, speech, chat, character data. In that domain it
runs deliberately ahead of NWR's reference avatar.

It does **not** re-import the NWR playground package (that would pull
physics/swarm and the older rigid cut-out face). Instead it vendors the minimal
substrate and brings NWR's control-surface pattern into the chat path:

| NWR capability | In AIFace |
| --- | --- |
| 32-ch BDS, Master Lock, constraint tick, 60 Hz, SSBO commands | Yes — `aiface.runtime` |
| Continuous muscle displacement (Wendland + occlusion) | Yes — ahead of NWR's old avatar |
| Conversation memory across turns | Yes — `ConversationSession` |
| Coarticulated viseme timing into muscle holds | Yes — `schedule_visemes` + event duration |
| TTS-locked audio visemes | Yes — as a fixture: `aiface.tts` / `aiface.audio`, `--tts` |
| Realtime lip-sync to an external voice | Yes — `aiface.stream`, `/voice/pcm`; ahead of NWR |
| A measured latency figure, not a claim | Yes — `aiface.sync`, `aiface-sync` |
| Loopback HTTP observe/command bridge | Yes — `aiface --bridge` (`FaceBridge`) |
| Chat and face in one surface | Yes — in-window chat box, `--no-chat-box` to opt out |
| Physics / semantic advection, swarm, entities | No — would smear identity |

**Do not** re-depend on `C:/code/NWR` as a package. Port patterns; keep the
face stack here. If a substrate rule genuinely needs to change, change it in
NWR first and re-vendor — never fork the contract locally.

---

## NWR review cross-audit (2026-07-30)

Synced against NWR `main @ f3ce4f5` ("Close authority mint holes and make .bdl
replay faithful."). Earlier audit was against `7d9740b`; the only substrate
delta still needed here was the GPU AI lock-mint refusal in `constraint.comp`
(AI may place barrier material, may not raise channel 31). `.bdl` replay,
command-compiler lock refusal, and AI control-plane schema narrowing are N/A —
AIFace has no paint/save/load bridge and no history log.

A senior QA review of NWR at `main @ 7d9740b` raised twelve findings
(`CR-001`…`CR-012`). Every one was checked against this repository. The review's
theme — *advertised guarantees fail outside the happy path* — found real
equivalents here, so the useful lesson generalised even where the code did not.

| NWR finding | Here |
| --- | --- |
| CR-002 replay drops writer priority | **Was latent.** `PaintCommand.priority` defaulted to `user` even for `source=1`, so an AI row could carry human authority. Now rejected at construction; `tests/test_commands.py` pins it. Two of our own tests were building exactly that row. |
| CR-005 bind not loopback-only | **Was present.** `--bridge-host` / `AIFACE_BRIDGE_HOST` accepted `0.0.0.0`. `FaceBridge` now refuses non-loopback binds unless `--allow-remote-bind`, using NWR's `net_guard` policy (resolve the name, require *every* answer to be loopback, fail closed) rather than a local variation. |
| CR-006 avatar destructive inheritance | **Partly present.** No `R` and no mouse paint here, but `S` persisted live speech state into the seed. Saves now write a rest pose, so save/load is a fixed point. |
| CR-007 stale state after load | **Was present.** `L` left visemes and muscle state describing the replaced world. `_on_world_reloaded()` clears them. |
| CR-008 unbounded relay input | **Analogue present.** Malformed `Content-Length` raised inside the handler, and the job list had no ceiling. Now a 400 and a 503. |
| CR-003 AI control plane, CR-010 AI mints locks | **Absent by construction** on the product path — the bridge has no reset/load/save route and speech emits nothing but lock-gated `±4` velocity. GPU gate also ported from `f3ce4f5`: AI paint of `human_barrier` cannot raise channel 31 (`TestOnlyAHumanMintsALock`). |
| CR-001, CR-004, CR-009, CR-011, CR-012 | Not applicable: no game, no exporter, no swarm agent. |

Auditing the field with that lens turned up two bugs of our own, both in
`constraint.comp`:

- The quantisation trigger counted velocity, so one loud phoneme re-anchored soft
  mouth tissue to `vacuum` and erased the density the seed gives it to push
  against. The trigger now reads material state only.
- Relaxation and clamping were gated on the hard-surface flag, and nothing
  dissipated velocity at all. Impulses accumulated indefinitely — the lip outline
  climbed past `CLAMP_LIMIT` and stayed there, which made the velocity debug view
  and the HUD's mouth-speed read-outs meaningless a second into any reply.
  Unlocked cells now relax toward rest with `VELOCITY_DAMPING`.

---

## What remains / good next work

Ordered by likely value:

1. **Path 1 Phase B** — in-app / bridge identity hot-swap on top of the MVP
   already shipped (`aiface.landmarks`, unified eye UV, `--preview` QA). Plan:
   [`docs/Path1Portrait.md`](Path1Portrait.md).

2. **Drive it from a real realtime API** end to end (OpenAI realtime, or any
   model that streams PCM + transcript deltas). The channel is built and measured
   against synthesised audio; a live session is the remaining unknown — expect
   the work to be in transport and back-pressure, not alignment.

3. **Mid-phrase anchoring** — the residual error is concentrated in long stretches
   of fluent speech whose transcript implies more syllables than the voice
   articulates, because only a boundary can re-anchor. A syllable-nucleus detector
   was tried and **made it worse** (292 ms vs 148 ms p95): RMS peaks under-detect
   in continuous voicing, and grapheme vowels do not line up with acoustic
   syllables. If revisited, it needs a real onset detector, and it needs to beat
   `aiface-sync` before it lands.

4. **Richer motor programs** — prosody curves, asymmetric micro-expression.
   Deepen drives on the existing field; do **not** add rigid cut-out pieces. The
   channel already supplies rhythm; motor programs should react to it (emphasis
   on loud frames, softer holds on quiet ones).

5. **Filmstrip / GIF sampling** — optional port of NWR `ai_handoff` sampling
   for offline QA of idle life and speech.

6. **`parts.py` atlas** — still useful for seed/diagnostics; renderer uses
   tissue displacement. Retire gradually if unused.

7. **NWR cleanup (optional, separate ask)** — stale avatar modules under NWR.

---

## Hard rules for the next agent

- **Do not push** to git without explicit user permission.
- **Do not commit** unless the user asks.
- Prefer **continuous field + occlusion** over new cut-out pieces.
- Keep shader and `skinning.py` CPU mirror in lockstep; `test_shader_contract`
  and `test_skinning` are the contract.
- Muscle authoring rule (in definition notes):  
  `travel ≤ 0.21 × influence_radius` (Wendland slope bound).
- Jaw: use `jaw_pose_from_definition` / `avatar_jaw` + `avatar_jaw_span`; do not
  scale mandible travel by the mobility rim.
- Install: `pip install -e ".[dev]"` from `C:/code/AIFace`.

---

## Quick commands

```bash
cd C:/code/AIFace
pip install -e ".[dev]"
pytest -q
python scripts/probe_field.py
aiface-seed --synthetic
aiface --demo
aiface-sync                         # streaming-vs-batch delta, pass/fail
aiface --bridge --no-chat-box       # then POST /voice/pcm from your own voice
aiface --tts --audio-backend null   # fixture voice, timing without speakers
```

---

## Prior chat transcript

Full history (NWR workspace Cursor session that built this):  
`C:/Users/parva/.cursor/projects/c-code-NWR/agent-transcripts/e76e265f-a4bf-4cdc-8846-bb407c86d4d8/`

Extraction plan (do not re-execute unless asked):  
`C:/Users/parva/.cursor/plans/extract_standalone_aiface_261d9b62.plan.md`

---

## One-line status for the next window

**AIFace is the flagship NWR use case: a continuous-muscle face that locks its
lips to audio somebody else is streaming (`/voice/pcm`), with the cost of doing
that live measured rather than claimed — 119 ms mean / 195 ms worst trimmed p95
against a 250 ms CI budget (`aiface-sync`). Local TTS is now just the fixture
that produces the ground-truth clip. Next is a live realtime-API session and a
real portrait seed — not more rigid pieces and not re-importing NWR.**
