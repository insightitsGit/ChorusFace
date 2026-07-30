# Path 1 — Replaceable portrait, same face motion

**Decision:** Path 1 is the product. Improve synthetic only as a fallback. Do
**not** build a parallel converter, and do **not** wait on NWR’s video driver
for still-image identity.

**Status (implemented):** Phase A MVP is in the tree. Eye UV is unified at
``v=0.472``, `aiface-seed --input` face-normalizes and measures landmarks
(`aiface.landmarks`), tissue/app consume seed metadata, and `--preview` writes
`seed_qa.png`. Phase B (in-app hot-swap) and Phase C remain future work.

NWR authority bugs are closed in AIFace as of `f3ce4f5` sync (see scorecard
below).

---

## NWR bug parity (gate before Path 1)

| Finding | Status in AIFace |
| --- | --- |
| AI command claims human priority | Fixed — `PaintCommand` + tests |
| Bridge binds off-loopback | Fixed — `is_loopback_host` |
| Save bakes mid-speech state | Fixed — rest-pose save |
| Load leaves stale speech | Fixed — `_on_world_reloaded` |
| Bad Content-Length / job flood | Fixed — 400 / 503 |
| Remote reset/load/save/paint | Fixed — routes absent + asserted |
| AI mints Master Lock on GPU | Fixed — `prior_lock` / `min` (`f3ce4f5`) |
| AI mints lock via HTTP/compiler | N/A — no paint control plane |
| Velocity never relaxes | Fixed — `VELOCITY_DAMPING` |
| Quantise keyed on velocity | Fixed — material-only trigger |
| Error reply with unread body | Fixed — `_discard_body` |
| Tokenless bridge | Fixed — bearer required + constant-time compare |

**Must-fix before Path 1:** none on the authority surface.

---

## As-is pipeline

```
photo | --synthetic
  → resize to 256×256 (full-frame stretch)
  → Haar face box (or synthetic fallback box)
  → fixed UV ellipses → locks + tissue + parts
  → write colocated bundle (bds + source_face + parts + tissue)
  → aiface warps source_face / atlas RGB with muscle displacement
```

CLI today:

```bash
aiface-seed --input portrait.jpg --preview
aiface --tts
```

Gemini / Imagen (or any generator) is the same pipeline: save a frontal
neutral PNG, then seed it. A demo portrait lives at
`assets/gemini_style_avatar.png` when no API key is available in this
environment:

```bash
aiface-seed --input assets/gemini_style_avatar.png --preview
aiface --tts
```

`--face-image` alone is **not** a face swap: it can desync the photo from
locks/tissue. Always reseed.

---

## Why eyes look jumbled (smoking gun)

| Subsystem | Eye V in face-box |
| --- | --- |
| Synthetic paint + seed lock masks | **0.365** |
| `parts.py`, tissue aperture, `face_definition.json`, lid uniforms | **0.472** |

Pupils are drawn in one place; blink/gaze/aperture run ~27 cells lower. That
alone produces jumbled eyes on the current demo. Real photos add a second
failure: Haar + fixed fractions never land on true pupils.

NWR video / `export_video` / optical-flow tools do **not** fix this. They are
for filmstrip QA or terrain later, not identity.

---

## Architecture choice

| Option | Decision |
| --- | --- |
| Harden `aiface-seed` | **Yes** — already emits the runtime contract |
| New image converter package | **No** — forks the artifact contract |
| Port NWR video driver for Path 1 | **No** — wrong problem |

Target quality bar for a shippable seed:

1. Face fills the same UV box the muscle definition was authored against.
2. Measured eyes: lid aperture covers real sclera/pupil.
3. Measured mouth: lip seam within a few cells of truth.
4. One pixel buffer → `source_face`, atlas RGB, tissue, BDS albedo.
5. Blink + jaw smoke: iris stays iris; cavity only in the gap.
6. Fail loud on profile / no-face / multi-face — never silent wrong fallback.

---

## Phased plan

### Phase A — MVP (done)

**Goal:** Drop a frontal photo (or Gemini export) → eyes/mouth register → speak.

1. **Unify eye UV** — synthetic + seed masks at **0.472** (`seed.py`).
2. **Face-normalize ingest** — `normalize_face_image` in `landmarks.py`.
3. **Landmarks** — MediaPipe when installed, else OpenCV eyes, else canonical.
4. **Bake measured centers** into `avatar_seed.landmarks`; tissue + app prefer them.
5. **QA gate** — `--preview` / `--require-qa`; `seed_qa.png` overlay.
6. **CLI** — prints `landmarks=… qa=… eyes@… mouth@…`.

Primary files: `landmarks.py`, `seed.py`, `skinning.py`, `app.py`,
`tests/test_landmarks.py`.

### Phase B — Product UX

1. In-app or bridge identity swap (authenticated; rest-pose only; no lock mint).
2. Hot-reload bundle after reseed (or clear restart — decide below).
3. Demote parts atlas to diagnostics; lids driven only by seed landmarks.
4. Consent/rights copy; keep portraits out of git (already ignored).
5. Optional studio matte so letterboxing doesn’t invent neck.

### Phase C — Stretch

1. Per-face muscle UV calibration from a denser landmark set.
2. Lighting / expression normalize for AI-generated junk.
3. Multi-identity slots; filmstrip QA (NWR export pattern, vendored later).
4. Webcam puppeteering as a **separate** driver — not Path 1 identity.
5. **Open-mouth / smile plates (only honest path to teeth):** use
   `aiface-capture` on a short HQ take (or stills). Real `open.png` /
   `smile.png` are composited inside `mouth_gap`. Do **not** invent enamel on a
   closed-mouth portrait. Speech motion stays muscles + jaw on
   `source_face.png`; judge lip silhouette first. See
   [AvatarCapture.md](AvatarCapture.md).

---

## Explicit non-goals

- Rebuilding motion as video morph / DeepFake / live mesh tracking for MVP.
- Re-importing NWR as a package.
- Making synthetic “good enough” instead of Path 1.
- Letting `--face-image` replace identity without a full reseed.

---

## Open decisions (need your call)

1. **MediaPipe** (best landmarks) vs **OpenCV-only** (smaller deps)?
2. **MVP delivery:** CLI + QA preview first, or in-app file picker immediately?
3. **Hot-swap** mid-session vs restart after `aiface-seed`?
4. **Gemini:** offline generate → seed, or in-app generate API later?
5. Keep photo backdrop vs soft studio matte?

Recommended defaults if you want speed: MediaPipe, CLI+preview first, restart
after reseed, offline Gemini images, keep backdrop until QA is green.

---

## First implementation slice (when you say go)

1. Fix 0.365 → 0.472 UV split (synthetic + seed masks) — unjumbles current demo.
2. Face-normalize + landmark mouth/eyes into metadata.
3. Seed QA overlay + tests on a fixture portrait.
4. Doc the one command path: `aiface-seed --input photo.jpg && aiface --tts`.

That is Path 1 MVP. Everything else builds on it.
