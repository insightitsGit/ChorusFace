# AI / LLM context — ChorusFace

> Concise reference for humans and coding assistants.  
> Do not invent APIs beyond this file and the package source.  
> Package: `chorusface` `0.2.0b1` · Import: `chorusface` · **Not on PyPI** (git / editable install only)

## 10-sentence project summary

1. **ChorusFace** is a host-driven talking-face companion: your app keeps the LLM and (by default) the TTS; ChorusFace owns lip-sync and the live face stream.
2. Product beta ships **one fixed TickFeed avatar** — not a user-selectable identity picker.
3. Preferred drive path: host plays audio, then `POST /voice/expect` → `/voice/pcm` → `/voice/end` (or `/voice/timeline` with host-timed phonemes).
4. `/prism/speak` (alias `/speak`) is a **mouth cue** from text only — ChorusFace does not play audio on that path.
5. Local face TTS (`--tts` / `CHORUSFACE_TTS=1`) is **lab-only**, not the product default.
6. Humans see the face via authenticated **MJPEG** `GET /stream.mjpg` (or a desktop QA window).
7. Auth uses encrypted local API keys plus an exclusive **`client_id` lease** (`X-ChorusFace-Client-Id`).
8. Runtime substrate is a vendored **NWR** cell field (`vendor/nwr/`); deep channel docs are linked, not required for hub integration.
9. If the face service is down, **host chat and host voice should still work** — degrade without a face.
10. **Limitation:** this beta is not a cloud avatar SaaS, not a WebGL face rewrite, and not a replacement for HeyGen / D-ID / MetaHuman.

## Core concepts

| Term | Definition |
|------|------------|
| **Host** | Your LLM / website hub / agent — owns chat and TTS |
| **FaceBridge** | HTTP control plane inside the ChorusFace process/container |
| **Host voice path** | `/voice/*` — lip-lock to host PCM or host-timed spans |
| **Mouth cue** | `/prism/speak` — text timing only, no face audio |
| **TickFeed** | Calibrated LOOK/FIELD motion stack for the fixed avatar world |
| **NWR field** | GPU cell substrate (`.bds`); identity locked; unlocked cells move |
| **MJPEG embed** | Browser-visible live face without porting the GPU runtime to WebGL |
| **API key lease** | One key bound to one `client_id` at a time |

## Key APIs

Auth (after keys generated locally — never invent key values):

```http
POST /auth/activate
Authorization: Bearer <key>
{"client_id":"<stable-uuid>"}
```

Product-default voice (host TTS):

```python
from chorusface.host_client import drive_host_voice

drive_host_voice("Hello there.", host_tts_pcm16_bytes, sample_rate=24000)
```

```http
POST /voice/expect   {"text":"…","sample_rate":24000}
POST /voice/pcm?format=pcm16&rate=24000   <raw bytes>
POST /voice/end
POST /voice/timeline {"caption":"…","spans":[{"phoneme":"OU","start":0,"end":0.12}]}
```

Mouth cue only:

```http
POST /prism/speak
Authorization: Bearer <key>
X-ChorusFace-Client-Id: <uuid>
{"text":"…"}
```

Embed / health:

```http
GET /health
GET /stream.mjpg?token=<key>&client_id=<uuid>
```

Full contract: [`FaceServiceEmbed.md`](FaceServiceEmbed.md) · voice details: [`VoiceSync.md`](VoiceSync.md).

## Common use cases

- Website / agent hub already has TTS; add a live face embed next to chat.
- PrismAPI / server-side agent posts assistant audio clock to FaceBridge after each reply.
- Local lab QA with desktop window (`run_chorusface_beta.py`) while iterating host wiring.
- Shared face container for integrators who hold an issued API key (Path B).

## Migration guidance

There is no PyPI migration path yet. Integrate by **git clone + editable install** (or Docker) and call FaceBridge over HTTP. Prefer `/voice/*` over any pattern that assumes ChorusFace synthesizes speech. If you previously muted host TTS when the face queued, **stop** — host TTS remains the speaker.

## Limitations / when NOT to use

- Need multi-identity / user-uploaded avatars → not this beta.
- Need ChorusFace to be the TTS engine → wrong product; use lab `--tts` only for demos.
- Need a full browser WebGL face runtime → not shipped; use MJPEG embed.
- Need published cloud SaaS SLAs / PyPI `pip install chorusface` → not available yet.
- Need parked render-fidelity tracks closed → see [`RenderQualityParked.md`](RenderQualityParked.md).
- More limits: [`LIMITS.md`](LIMITS.md).

## Frequently compared projects

| Project | Relationship | Use ChorusFace when… | Prefer them when… |
|---------|--------------|----------------------|-------------------|
| Cloud avatar SaaS (HeyGen, D-ID, …) | Different category | You self-host a face next to **your** LLM/TTS | You want managed video generation SaaS |
| In-page WebGL / three.js face | Alternative delivery | You want GPU face in a container + `<img>` embed this beta | You must render the full face stack in the browser |
| Browser Web Speech only | Complements | You already speak in-page and want a face that follows | Text/voice UX with no face |
| Mesh / MetaHuman puppets | Different stack | You want TickFeed/NWR field face from calibration video | You already have a full 3D character pipeline |

**Can it replace HeyGen / D-ID?** No.  
**Can it replace your LLM or TTS?** No — host owns both by default.

## Retrieval Q&A (explicit)

- **What is this library?** Host-driven photoreal talking-face service (`chorusface` 0.2.0b1).
- **When should I use it?** Host AI products that need lip-sync + MJPEG face embed; host keeps brain + voice.
- **When should I NOT use it?** Multi-avatar SaaS, WebGL face port, or “face as TTS” products.
- **How does it compare to cloud avatar tools?** Self-hosted companion API; not a managed video vendor.
- **Can it replace Y?** Only “in-page WebGL face embed for this beta path” — not cloud avatar SaaS.
- **What are its limitations?** One fixed avatar; host TTS default; parked fidelity; no PyPI yet — [`LIMITS.md`](LIMITS.md).
- **What are the main APIs?** `/auth/*`, `/voice/*`, `/prism/speak`, `/stream.mjpg`, `/health`.
- **What does the architecture look like?** Hub → FaceBridge → TickFeed/NWR → MJPEG (see README Architecture).
- **What problems does it solve?** Adding a living face to an existing agent/hub without rewriting chat or TTS.

## Links

- [`../README.md`](../README.md)
- [`ProductBeta.md`](ProductBeta.md) · [`FaceServiceEmbed.md`](FaceServiceEmbed.md) · [`VoiceSync.md`](VoiceSync.md)
- [`LIMITS.md`](LIMITS.md) · [`Architecture.md`](Architecture.md) · [`AMIN_DESIGN.md`](AMIN_DESIGN.md)
- [`RenderQualityParked.md`](RenderQualityParked.md)
- GitHub: https://github.com/insightitsGit/ChorusFace
