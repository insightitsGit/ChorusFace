# Avatar Chat — the biomechanical pipeline

**Product beta (host owns the LLM):** see [`ProductBeta.md`](ProductBeta.md) —
hosts POST assistant text to FaceBridge `/speak`; ChorusFace does not call the chat
model. Window is resizable with locked aspect; avatar identity is fixed for this
beta.

The face is not a viseme clip player. Speech, emotion, blinking, breathing,
idle micro-motion, and AI intent all inject **muscle impulses** into one queue,
a solver integrates them, and the renderer only visualises the result.

```
chat text / LLM reply                  transcript + PCM from another voice
    │                                              │
    ▼                                              ▼
extract_states → phoneme stream + emotion      chorusface.speech
    │                                              │
    ├─ (fixture) TTS synthesize + align          VoiceStream.feed
    │       chorusface.tts / chorusface.audio              chorusface.stream
    │       → measured PhonemeSpan timeline        → StreamedSpan, as decided
    │                                              │
    ▼◄─────────────────────────────────────────────┘
schedule_visemes / schedule_spans              chorusface.speech
    │
    ▼
IntentSystem.speech_impulses                   chorusface.biomechanics.intent
    │
    ▼
MuscleImpulseQueue  ◄── Emotion, Eyes, Breathing, Idle
    │
    ▼
MuscleSolver (spring-damper, 60 Hz)
    │
    ├── FaceRenderState  → shader uniforms
    └── FieldImpulseSpec → ±4 velocity commands on unlocked cells
```

## Modules

| Module | Role |
| --- | --- |
| `biomechanics/data/face_definition.json` | Character config: muscles, anchors, forces, phoneme map, jaw constraints |
| `biomechanics/muscles.py` | Muscle registry, `MuscleImpulse`, deterministic blend queue, spring-damper solver |
| `biomechanics/emotion.py` | Nine continuous emotion axes → muscle impulses |
| `biomechanics/eyes.py` | Gaze, microsaccades, asymmetric blink, pupil |
| `biomechanics/jaw.py` | Mass/damping jaw physics; speech only sets a target |
| `biomechanics/breathing.py` | Slow neck/cheek/jaw oscillation, silent or speaking |
| `biomechanics/idle.py` | Deterministic, non-looping micro-behaviour |
| `biomechanics/intent.py` | LLM JSON → emotion patches and speech impulses |
| `biomechanics/face.py` | Orchestrator and single source of truth |
| `speech.py` | Text → visemes → mouth poses, and the chat client |
| `stream.py` | Arriving PCM → viseme decisions, with the delay measured per decision |
| `sync.py` | Streaming vs. batch alignment, in milliseconds |
| `tts.py` | Fixture voice: synthesis + full-lookahead viseme alignment |
| `audio.py` | WAVE decode, RMS envelope, voiced intervals, playback sinks |
| `seed.py` | Photograph → locked `.bds` seed + part atlas + portrait |
| `parts.py` | Anatomical part atlas construction and I/O |
| `app.py` | Window, chat threads, GPU impulse enqueue, uniforms, HUD |

## Impulse blending

Overlapping impulses on one muscle blend by priority and envelope rather than
fighting. Each impulse contributes `(1 + priority) × (1 - progress)^falloff` as
its weight, and the queue returns the weighted mean. A high-priority impulse
with zero strength therefore *suppresses* a lower-priority drive instead of
being averaged away — which is how an emotion can hold a mouth closed against a
speech impulse.

The solver is a critically damped spring per muscle, with light coupling to
declared neighbours so a unilateral drive does not produce a face that tears.

## Jaw physics

`submit_phoneme` sets a target opening from `PHONEME_JAW_TARGET`; it never sets
an angle. The jaw then chases that target under mass, elasticity, and damping,
with a small bounce at the limits. The consequence is that fast speech produces
partial openings and overlap, exactly as a real jaw does, and
`test_jaw_physics_never_teleports` asserts the angle can never jump.

## Speech to field impulses

The muscles marked `writes_field` in the character definition become GPU
velocity commands. `OrbicularisOris` and `Masseter` are remapped from their
cheek anchors onto the mouth centre — a jaw impulse belongs where the mouth is,
not where the muscle attaches.

The app then clamps every impulse into the unlocked mouth disc before it is
enqueued, so a bad anchor cannot aim a write at the eyes. The Master Lock would
reject it anyway; this is belt and suspenders.

Impulse count per tick is capped by `constraints.max_field_impulses_per_tick`
(default 12) so the 64-command tick budget can never be exhausted by speech.

## Voice and lip-lock

Timing can come from three clocks, and they make different claims. The default
is **somebody else's audio**: `POST /voice/pcm` feeds `VoiceStream`, which emits
each viseme the moment the arriving energy justifies it and records how late that
decision was. See [Voice Sync](VoiceSync.md) for the algorithm and the measured
error.

Local synthesis (`--tts`, off by default) is the fixture that gives the oracle a
clip with knowable ground truth. The reply thread synthesises audio; alignment
then measures when each viseme is due, with the whole clip available:

* **energy** (default) — short-time RMS. Phrases pin to voiced stretches when
  the punctuation count matches the number of long silences; inside each
  stretch, cumulative energy decides the rhythm.
* **words** — word timestamps from an OpenAI-compatible transcription of the
  synthesised clip, matched back onto the script with a monotone diff.
* **linear** — uniform stretch, used when the clip has no measurable energy.

Playback starts on the render thread at the same instant the schedule is
anchored. Each sink reports a startup latency, and `--tts-latency` trims it by
hand, so the lips and the speaker stay in phase.

With no audio at all — no channel and no `--tts` — the viseme schedule is derived
from the written reply via `text_to_visemes`, which is a letter count and is
honest about being one. Offline `local_reply` returns plain spoken sentences —
never a canned `aa ee oo` filler — so the mouth articulates the same words a
voice would say.

## AI intent

Any model that can emit JSON can drive the face directly, bypassing text:

```json
{
  "emotion": {"valence": 0.7, "confidence": 0.9, "arousal": 0.3},
  "speech": {"phonemes": ["AH", "EE", "OO"]},
  "intent": {"thinking": 0.6, "emphasis": 0.8}
}
```

```python
from chorusface.biomechanics import BiomechanicalFace

face = BiomechanicalFace.from_file(seed=5)
face.submit_intent(payload, tick=1)
render_state, field_impulses = face.step(1 / 60, tick=1)
```

## Tagged replies

For text-driven chat the model is asked to prefix its reply with an emotion tag
and optionally sprinkle phoneme tags:

```
[EMOTION:HAPPY] [PHONEME:AH] [PHONEME:OO] Good to see you!
```

Explicit `[PHONEME:...]` tags win when present. Otherwise the phoneme stream is
derived from the letters, including digraphs (`TH`, `SH`, `OO`, `EA`). If the
model forgets the tag grammar entirely, a keyword scan recovers the emotion.
Nothing here can fail closed: an unparseable reply still produces a `REST` face.

Without an API key, `local_reply` returns a deterministic tagged sentence, so
the whole pipeline runs offline and in tests.

## Building a seed

```bash
chorusface-seed --synthetic                        # deterministic drawn face
chorusface-seed --input portrait.jpg               # your own front-facing photo
chorusface-seed --input portrait.jpg --edge-threshold 0.35
```

The seeder resizes the image, locates a frontal face (Haar cascade, with a
centred fallback), partitions the face box into locked and unlocked regions,
writes photographic albedo into channels 8–10 and Sobel contours into channel
24, and saves the world together with its part atlas and portrait.

Rebuild the seed whenever you change the photograph — the atlas and the lock
mask are both derived from it.

## Character swapping

Everything specific to *this* performer lives in `face_definition.json`:
muscle names, UV anchors, influence radii, stiffness and damping, force
directions, the phoneme→muscle map, the rest pose, and the jaw limits. Point
the app at a different file and the same engine drives a different face:

```bash
chorusface --face-definition creature.json
```

## Reading the HUD

```
CHORUSFACE BIOMECH  FPS 87  |  GPU 2.14 ms  |  CPU 0.31 ms  |  view portrait
Phoneme AH/HAPPY  pending 6  impulses 14  dom valence:+0.62
Jaw 0.238 rad  blink 1.42s  breath 0.71  muscles [Mas:0.41,Orb:0.22,Zyg:0.18]
Mouth |V| mean 0.031 peak 0.480  cells 96  |  t=4.2s  tick 254@60Hz
```

`Mouth |V|` is sampled from the unlocked disc only. If it stays at zero while
phonemes are pending, impulses are being rejected — check that the world's
mouth centre metadata survived the last save.
