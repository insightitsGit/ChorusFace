# ChorusFace face service — container API + PrismAPI + web embed

**Faster web path:** stream embed (not a WebGL rewrite). One shared TickFeed
avatar instance. Avatar is **not user-changeable** in this beta.

```
Host agent (PrismAPI / Website Hub)  owns LLM + TTS
    → POST /voice/expect|/pcm|/end   (default — host audio clock)
    → or POST /prism/speak           (mouth cue only, no face audio)
ChorusFace container (headless TickFeed)
    → GET /stream.mjpg               (human-visible embed)
Website page <img> shows the live face
```

## Why this design

| Layer | Role |
| --- | --- |
| **Host TTS** | Product-default audio — ChorusFace does not synthesize speech |
| **`/voice/*`** | Lip-lock to host PCM or host-timed phoneme spans |
| **`/prism/speak`** | AI↔AI mouth cue from text when PCM is not available |
| **FaceBridge** | Internal GPU control plane inside the container |
| **MJPEG `/stream.mjpg`** | Browser embed (no WebGL port required) |

Integrators should drive **host voice → `/voice/*`**, not rely on ChorusFace
local TTS. Lab-only: `run_chorusface_service.py --tts` / `CHORUSFACE_TTS=1`.
See [`VoiceSync.md`](VoiceSync.md) and [`ProductBeta.md`](ProductBeta.md).

## Run (Docker)

```powershell
# From ChorusFace repo (world must exist under output/worlds/tickfeed)
docker compose up --build
```

Env:

| Variable | Default |
| --- | --- |
| `CHORUSFACE_BRIDGE_TOKEN` | `chorusface-beta` |
| `CHORUSFACE_BRIDGE_PORT` | `8766` |
| `CHORUSFACE_BRIDGE_CORS` | `*` |
| `CHORUSFACE_STREAM_FPS` | `12` |

Local without Docker (debug):

```powershell
python scripts/run_chorusface_service.py
# visible window: python scripts/run_chorusface_service.py --visible
```

## API keys (required)

Generate **10 local encrypted sample keys** (never commit):

```powershell
python scripts/generate_api_keys.py
```

Keys live under `secrets/` (gitignored). Hand one key from
`secrets/api_keys.handoff.local.txt` to each integrator. Both **Prism speak**
and **ChorusFace FaceBridge** require that key.

**Security model (recommended):** one API key ↔ one AI/system via exclusive
`client_id` lease (not multi-IP sharing). Activate once, then send
`X-ChorusFace-Client-Id` on every request. Optional sticky IP via
`CHORUSFACE_KEY_BIND_IP=1`.

See [`secrets/README.md`](../secrets/README.md).

## API

### Health
`GET /health` — no auth.

### AI-to-AI speak (Prism)

```http
POST /auth/activate
Authorization: Bearer <api-key-from-handoff>
Content-Type: application/json

{"client_id":"11111111-2222-3333-4444-555555555555"}
```

### Host voice (product default)

```http
POST /voice/expect
Authorization: Bearer <api-key-from-handoff>
X-ChorusFace-Client-Id: 11111111-2222-3333-4444-555555555555
Content-Type: application/json

{"text":"Hello there. How are you today?","sample_rate":24000}
```

Then `POST /voice/pcm?format=pcm16&rate=24000` with raw host-TTS bytes, then
`POST /voice/end`. Or use `POST /voice/timeline` with host-timed spans.

Python:

```python
from chorusface.host_client import drive_host_voice

drive_host_voice("Hello there.", pcm16_bytes, sample_rate=24000)
```

### Mouth cue only (no ChorusFace audio)

```http
POST /prism/speak
Authorization: Bearer <api-key-from-handoff>
X-ChorusFace-Client-Id: 11111111-2222-3333-4444-555555555555
Content-Type: application/json

{"text":"Hello there. How are you today?"}
```

Also accepts `speech` | `message` | `response`. Alias: `POST /speak`.
Release with `POST /auth/release` when the AI shuts down.

```python
from chorusface.prism_adapter import forward_speak, SpeakIntent

forward_speak({"text": "Hello there"})           # mouth cue only
forward_speak(SpeakIntent(text="Hello there"))
```

If `prismlib-plus` is installed:

```python
from chorusface.prism_adapter import try_register_prism_provider
try_register_prism_provider()  # exposes chorusface_speak via prism.api
```

CLI:

```powershell
python -m chorusface.prism_adapter "Hello there"
```

### Web embed stream

```html
<img
  src="http://FACE_HOST:8766/stream.mjpg?token=chorusface_sk_…"
  alt="ChorusFace"
  width="320"
  height="320"
/>
```

Auth for `<img>` uses **`?token=`** (browsers cannot set `Authorization` on image URLs).
Single frame: `GET /preview.jpg?token=…` or `GET /preview` (PNG + Bearer).
Use a key from `secrets/api_keys.handoff.local.txt`.

## Insightits website

1. Run face service (compose or `run_chorusface_service.py`).
2. Flask env:
   - `CHORUSFACE_BRIDGE_ENABLED=1`
   - `CHORUSFACE_BRIDGE_URL=http://127.0.0.1:8766` (or container host)
   - `CHORUSFACE_BRIDGE_TOKEN=chorusface-beta`
   - `CHORUSFACE_EMBED_URL=http://127.0.0.1:8766/stream.mjpg?token=chorusface-beta`
3. Homepage chat widget shows the face via `connectors/web/chorusface-embed.js`
   (or Insightits `public/js/chorusface-embed.js`).
4. Hub keeps **host TTS** playing; face is driven via `/voice/*` or mouth-cue
   `/prism/speak` (see [`ProductBeta.md`](ProductBeta.md)).

## GPU notes

- **CPU/OSMesa/EGL in Docker** works for lab embed; fidelity depends on GL.
- **NVIDIA GPU**: `docker run --gpus all …` with `MODERNGL_BACKEND=egl`.
- Desktop `run_chorusface_beta.py` remains valid for local QA without containers
  (local TTS still off unless `--tts`).

## Acceptance

1. `docker compose up` → `GET /health` ok
2. Host PCM via `/voice/*` moves the mouth
3. `/prism/speak` mouth-cues without face audio
4. Browser shows live face at `/stream.mjpg?token=…`
5. Website chat + host TTS still work if the face container is down

## Related

- [`ProductBeta.md`](ProductBeta.md) — host contract overview
- [`AvatarChat.md`](AvatarChat.md) — speech pipeline
- [`RenderQualityParked.md`](RenderQualityParked.md) — parked fidelity tracks
