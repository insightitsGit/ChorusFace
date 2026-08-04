# Phonetic fidelity (lip-reading assist)

ChorusFace does **not** own playback. The LLM host speaks; we expose a **realtime
mouth timeline** the host can drive so the face lands on shapes a lip reader
can use.

## Research takeaway

| Approach | Role for ChorusFace |
| --- | --- |
| Grapheme → viseme (current fallback) | Works without a phoneme model; good enough for demos |
| **Oculus / MPEG-4 15-viseme set** | Standard visual inventory (sil, PP, FF, TH, DD, kk, CH, SS, nn, RR, aa, E, ih, oh, ou) |
| Host phoneme / IPA timestamps | Best fidelity — host already aligned audio |
| Energy alignment of text + PCM | When host has transcript + PCM but no phoneme clock |
| Acoustic phoneme ASR inside ChorusFace | Out of scope — worse than reading the host’s text |

Speed/latency is already in live-chat range (~100–200 ms trimmed p95 vs offline).
The remaining gap for lip-reading is **shape inventory + closures**, not more
jitter reduction.

## TickFeed post-initial sync (after B1–B4)

The TickFeed **initial design** (labels sole LOOK, KEY/Δ FIELD) is documented in
[`TickFeedDesign.md`](TickFeedDesign.md) §1–§13. A **second band** of work (§14)
landed afterward specifically for lip-readable motion:

| Band | What |
| --- | --- |
| Absolute overlay until | Release LOOK at `due_at + duration` (audio clock), not vowel hold floors |
| Closure priority | Never skip PP/MM/CLOSED while an open hold is active |
| Bilabial onset | Pin ~45 ms PP at word start; snap to energy valleys when using energy align |
| Whisper words | Default `--tts-align words` when an API key is present |
| Distinct plates | Priority atlas keeps CLOSED/PP/FF/TH on different frames when the take allows |

Still open for true speechreading: **lab MFA**, a **new capture take** with a
flat rest + tongue-visible TH, and host phoneme timelines (`/voice/timeline`).

## What we ship

1. **Canonical visemes** in `chorusface.speech` (`canonical_viseme`, Oculus aliases).
2. **Articulation = muscles + jaw** warping the immutable portrait. No shader
   MouthPose lip bias; no painted teeth / oral stamp.
3. **Lip silhouette** differs by viseme (AH open jaw, PP closed oris, OU round,
   EE wide). Judge quality on outline, not interior fill.
4. **Gap fill** is a tight photo-derived soft shadow by default. Real teeth
   come from **`chorusface-capture`** plates (`open.png` / `smile.png`) composited
   inside `mouth_gap` — not invented geometry (see
   [AvatarCapture.md](AvatarCapture.md)).
5. **Two sync modes:**
   - **Host timeline (preferred):** `POST /voice/timeline`
   - **Text + PCM aligner:** `POST /voice/expect` → `/voice/pcm` → `/voice/end`

## Host timeline (preferred)

Host plays audio on its clock and posts absolute viseme spans in the same clock:

```http
POST /voice/timeline
Authorization: Bearer <token>
Content-Type: application/json

{
  "caption": "Hello there.",
  "emotion": "happy",
  "spans": [
    {"phoneme": "EH", "start": 0.00, "end": 0.08},
    {"phoneme": "OU", "start": 0.08, "end": 0.16},
    {"viseme": "sil", "start": 0.16, "end": 0.22}
  ]
}
```

- Times are seconds from utterance start.
- Names may be Oculus lowercase (`sil`, `ou`) or ChorusFace uppercase (`REST`, `OU`).
- Unknown names become `REST` (closed), never a invented shape.
- Use `--voice-trim` if the host’s first audible sample is delayed vs POST time.

## Text + PCM (when the host has no phoneme clock)

```http
POST /voice/expect  {"text":"Hello there.","emotion":"happy","sample_rate":24000}
POST /voice/pcm?format=pcm16&rate=24000   <raw bytes>
POST /voice/end
```

Viseme *order* comes from the transcript; *when* comes from energy. See
[VoiceSync.md](VoiceSync.md).

## Portrait upload vs Gemini avatar

| Path | How |
| --- | --- |
| **User photo (product path)** | `chorusface-seed --input portrait.jpg --preview` then `chorusface` |
| **Gemini / Imagen avatar** | Generate a frontal neutral portrait (API or Studio) → save PNG → same `chorusface-seed --input …` |
| **Bundled demo portrait** | `assets/gemini_style_avatar.png` (generated stand-in when no Gemini key is present) |

Never use `--face-image` alone as a face swap — locks/tissue must be reseeded.

```bash
chorusface-seed --input assets/gemini_style_avatar.png --preview --require-qa 0.35
chorusface --tts   # local fixture voice only; production uses /voice/*
```

Phase B (in-app hot-swap over the bridge) is still open; today swap = reseed + restart
or reload the colocated world bundle.

## Viseme cheat sheet

| Canonical | Oculus-ish | Examples |
| --- | --- | --- |
| REST | sil | silence / neutral |
| PP | PP | p, b, m |
| FF | FF | f, v |
| TH | TH | think |
| DD | DD | t, d |
| KK | kk | k, g |
| CH | CH | chair, she |
| SS | SS | s, z |
| NN | nn | n, l |
| RR | RR | red |
| AA / AH | aa | car / open |
| EH | E | bed |
| IH | ih | tip |
| EE | (iy family) | see |
| OH | oh | toe |
| OU | ou | book / moon |
| CLOSED | — | sentence stop |

## Measuring

```bash
chorusface-sync --json --budget-ms 250
```

That gates **timing** fidelity. Shape fidelity is judged visually (and later by
host-provided phoneme timelines, which bypass grapheme approximation entirely).
