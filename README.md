# ChorusFace

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.2.0b1-informational)](https://github.com/insightitsGit/ChorusFace)
[![GitHub](https://img.shields.io/badge/github-insightitsGit%2FChorusFace-181717?logo=github)](https://github.com/insightitsGit/ChorusFace)

**Host-driven photoreal talking face for AI products — lip-sync + MJPEG embed; your app keeps the brain (and voice).**

Package: `chorusface` · **Not on PyPI yet** — install from git / editable only.

```bash
git clone https://github.com/insightitsGit/ChorusFace.git
cd ChorusFace
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
source .venv/bin/activate   # macOS / Linux
pip install -U pip
pip install -e ".[ml,voice]"
```

AI / coding-assistant context: [`docs/ai-overview.md`](docs/ai-overview.md) · [`docs/llm-context.md`](docs/llm-context.md)

> **Face down ≠ product down.** If ChorusFace is offline, host chat and host TTS should still work.  
> **Beta = one fixed TickFeed avatar** (not user-selectable).  
> Local face TTS is **lab-only** (`--tts`). Product default: host owns TTS → `/voice/*`.  
> Limits: [`docs/LIMITS.md`](docs/LIMITS.md) · parked fidelity: [`docs/RenderQualityParked.md`](docs/RenderQualityParked.md).

---

## What is ChorusFace?

ChorusFace is a **face companion** for host AI products (website hubs, PrismAPI agents, custom backends). After your assistant reply, you play **your** TTS and drive the face over HTTP FaceBridge; browsers show the live face via authenticated MJPEG.

| | |
|--|--|
| **Replaces** | In-page WebGL port of this face runtime for the beta embed path |
| **Complements** | Your LLM / agent / website-hub TTS · PrismAPI hosts · Insightits-style hubs |
| **Integrates with** | HTTP FaceBridge · MJPEG `<img>` / embed JS · Docker · vendored NWR substrate |

**ChorusFace is not a chatbot** (host owns the LLM). **It is not the default TTS engine** (host owns voice; lab `--tts` only). **It is not a required WebGL rewrite** for this beta — use `/stream.mjpg`.

### When NOT to use ChorusFace

- You need multi-identity / user-uploaded avatars → not this beta.
- You need ChorusFace to speak for you → wrong default; keep host TTS.
- You need a full browser WebGL face stack → not shipped; use MJPEG embed.
- You need managed cloud avatar SaaS (HeyGen / D-ID / …) → different category.
- You need PyPI `pip install chorusface` → not published yet.

---

## Why ChorusFace?

| Pain | ChorusFace answer |
|------|-------------------|
| Agent already has TTS; no face | Host PCM / timeline → `/voice/*` lip-lock |
| WebGL port of the face is too heavy | Container + `GET /stream.mjpg` embed |
| Face service must not own the chat model | Host POSTs after its own LLM reply |
| Face crash must not kill chat | Best-effort FaceBridge; degrade without face |
| Keys shared across agents | Exclusive API key ↔ `client_id` lease |

---

## Quick start (30 seconds)

World under `output/` is **not** in git. Build once, then keys + compose:

```bash
python scripts/build_tickfeed_demo.py
python scripts/generate_api_keys.py          # writes gitignored secrets/
docker compose up --build
curl http://localhost:8766/health
```

`/health` reports `local_tts_default: false` and `host_voice: /voice/expect|/voice/pcm|/voice/end`.

### Activate + host voice (product default)

Use **one** key from `secrets/api_keys.handoff.local.txt` (**path only — never commit keys**):

```bash
# PowerShell-friendly example
curl -X POST http://localhost:8766/auth/activate \
  -H "Authorization: Bearer <KEY>" \
  -H "Content-Type: application/json" \
  -d "{\"client_id\":\"11111111-2222-3333-4444-555555555555\"}"

# Prefer host TTS PCM:
#   POST /voice/expect  {"text":"Hello there.","sample_rate":24000}
#   POST /voice/pcm?format=pcm16&rate=24000  <raw bytes>
#   POST /voice/end
#
# Python:
#   from chorusface.host_client import drive_host_voice
#   drive_host_voice("Hello there.", pcm16_bytes, sample_rate=24000)
```

Mouth cue only (no ChorusFace audio):

```bash
curl -X POST http://localhost:8766/prism/speak \
  -H "Authorization: Bearer <KEY>" \
  -H "X-ChorusFace-Client-Id: 11111111-2222-3333-4444-555555555555" \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"Hello there.\"}"
```

Embed:

```text
http://localhost:8766/stream.mjpg?token=<KEY>&client_id=<UUID>
```

### Drop-in (website hub)

Server-side only — **no keys in public JS**.

| Env | Example |
|-----|---------|
| `CHORUSFACE_BRIDGE_ENABLED` | `1` |
| `CHORUSFACE_BRIDGE_URL` | `http://127.0.0.1:8766` |
| `CHORUSFACE_BRIDGE_TOKEN` | key from local handoff |
| `CHORUSFACE_CLIENT_ID` | stable UUID for that host |
| `CHORUSFACE_EMBED_URL` | `{BASE}/stream.mjpg?token=…&client_id=…` |

Embed helper: [`connectors/web/chorusface-embed.js`](connectors/web/chorusface-embed.js) · bridge helper: [`connectors/web/chorusface-bridge.js`](connectors/web/chorusface-bridge.js).

HTTPS pages cannot call `http://127.0.0.1` (mixed content). Prefer server-side `/voice/*` or a TLS face URL.

### Full API surface

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| `GET` | `/health` | none | Liveness + contract hints |
| `POST` | `/auth/activate` | Bearer | Lease key ↔ `client_id` |
| `POST` | `/auth/heartbeat` | Bearer + client id | Keep lease |
| `POST` | `/auth/release` | Bearer + client id | Free key |
| `POST` | `/voice/expect` | Bearer + client id | **Default** — host transcript |
| `POST` | `/voice/pcm` | Bearer + client id | Host TTS PCM |
| `POST` | `/voice/end` | Bearer + client id | End utterance |
| `POST` | `/voice/timeline` | Bearer + client id | Host-timed phonemes |
| `POST` | `/prism/speak` | Bearer + client id | Mouth cue only |
| `POST` | `/speak` | same | Alias of `/prism/speak` |
| `GET` | `/stream.mjpg` | `?token=&client_id=` | Live MJPEG |

Source of truth: [`docs/FaceServiceEmbed.md`](docs/FaceServiceEmbed.md) · voice: [`docs/VoiceSync.md`](docs/VoiceSync.md).

---

## How to implement (choose your path)

### 1) Self-host Docker (Path A)

```bash
python scripts/build_tickfeed_demo.py
python scripts/generate_api_keys.py
docker compose up --build
```

Mounts: `./output/worlds/tickfeed`, `./secrets` (read-only). Image: `chorusface-face:beta`.

Headless without Docker:

```bash
python scripts/run_chorusface_service.py
# lab local TTS only if needed:
python scripts/run_chorusface_service.py --tts
```

### 2) Shared container + issued key (Path B)

You need base URL + one issued key + stable `client_id`. Activate, then `/voice/*` (preferred) or `/prism/speak` mouth cue. Same lease rules: one key → one `client_id`.

### 3) Desktop QA window

```bash
python scripts/run_chorusface_beta.py
# or: chorusface-beta
# lab: python scripts/run_chorusface_beta.py --tts
```

Window is resizable with locked aspect ratio. Details: [`docs/ProductBeta.md`](docs/ProductBeta.md).

---

## Features

| Feature | Description |
|---------|-------------|
| **Host-voice lip-lock** | `/voice/expect` + `/pcm` + `/end` or `/voice/timeline` |
| **Mouth cue** | `/prism/speak` — text timing, no face audio |
| **MJPEG embed** | `/stream.mjpg` for existing web UIs |
| **Fixed TickFeed avatar** | One calibrated world for this beta |
| **API key vault** | Local Fernet vault under `secrets/` (gitignored) |
| **Exclusive lease** | One key ↔ one `client_id` |
| **Docker service** | `Dockerfile` + `docker-compose.yml` |
| **Lab `--tts`** | Optional local synthesis — not product default |
| **Degrade without face** | Host chat/voice continue if FaceBridge is down |

---

## Architecture

```
Host hub (LLM + TTS)
    │  POST /auth/activate
    │  play host audio ──────────────────┐
    │  POST /voice/expect|/pcm|/end  ◄───┘
    │  (or POST /prism/speak mouth cue)
    ▼
ChorusFace (Docker / local)
    │  TickFeed LOOK/FIELD → NWR field commands
    ├─► GET /stream.mjpg   (human embed)
    └─► desktop window     (QA)
```

Deep design (linked, not required for hub install):  
[`docs/Architecture.md`](docs/Architecture.md) · [`docs/AMIN_DESIGN.md`](docs/AMIN_DESIGN.md) · [`docs/NWRDataDesign.md`](docs/NWRDataDesign.md) · [`docs/BDSMotionMap.md`](docs/BDSMotionMap.md).

Pinned substrate: `vendor/nwr/` (`vendor/nwr/NWR_REVISION.txt`).

---

## Install & extras

```bash
pip install -e ".[ml,voice]"     # TickFeed ML + local audio sinks (lab)
pip install -e ".[dev]"          # pytest + ML + voice
pip install -e ".[seed]"         # portrait / MediaPipe seed helpers
```

Requires Python **3.11+**. GPU optional (better GL on Linux + NVIDIA Container Toolkit).

### Env (common)

| Variable | Default | Role |
|----------|---------|------|
| `CHORUSFACE_WORLD` | `output/worlds/tickfeed` | World dir |
| `CHORUSFACE_BRIDGE_HOST` | `127.0.0.1` / `0.0.0.0` in service | Bind |
| `CHORUSFACE_BRIDGE_PORT` | `8766` | Port |
| `CHORUSFACE_BRIDGE_TOKEN` | fallback token if no vault | Bearer |
| `CHORUSFACE_BRIDGE_CORS` | `*` | CORS |
| `CHORUSFACE_STREAM_FPS` | `12` | MJPEG FPS |
| `CHORUSFACE_TTS` | off | Lab local TTS |
| `CHORUSFACE_KEY_LEASE` | `1` | Exclusive lease |
| `CHORUSFACE_KEY_LEASE_TTL_S` | `900` | Idle lease TTL |

### Acceptance / production checklist

1. `GET /health` → ok, `local_tts_default: false`
2. Activate with local/issued key → 200
3. Host PCM via `/voice/*` moves the mouth
4. `/prism/speak` mouth-cues without face audio
5. Browser shows `/stream.mjpg?token=…&client_id=…`
6. Second process, same key, different `client_id` → 403
7. Host chat + host TTS still work if the face is down
8. No keys in public front-end; HTTPS uses TLS face URL or server-side drive

---

## CLI / helpers

| Entry | Role |
|-------|------|
| `chorusface` / `python -m chorusface` | Main app |
| `chorusface-beta` / `scripts/run_chorusface_beta.py` | Product beta window |
| `chorusface-service` / `scripts/run_chorusface_service.py` | Headless service |
| `chorusface-host` / `python -m chorusface.host_client` | Host client (`--voice` + `--pcm-file`) |
| `chorusface-prism` | Prism speak helper |
| `amin-train` | Amin digest train |

```bash
python -m chorusface.host_client --voice --pcm-file reply.pcm "Hello there."
python -m chorusface.host_client "Hello there."   # mouth cue only
```

---

## Documentation

| Doc | Description |
|-----|-------------|
| [`docs/ai-overview.md`](docs/ai-overview.md) | Canonical LLM / human one-pager |
| [`docs/llm-context.md`](docs/llm-context.md) | Alias → ai-overview |
| [`docs/LIMITS.md`](docs/LIMITS.md) | Beta limits, lease, mixed content |
| [`docs/ProductBeta.md`](docs/ProductBeta.md) | Host contract + window |
| [`docs/FaceServiceEmbed.md`](docs/FaceServiceEmbed.md) | Container API + embed |
| [`docs/VoiceSync.md`](docs/VoiceSync.md) | Host PCM / timeline channel |
| [`docs/Architecture.md`](docs/Architecture.md) | Stack diagram |
| [`docs/AMIN_DESIGN.md`](docs/AMIN_DESIGN.md) | Amin walkthrough → code |
| [`docs/NWRDataDesign.md`](docs/NWRDataDesign.md) | NWR dataset contract |
| [`docs/RenderQualityParked.md`](docs/RenderQualityParked.md) | Parked fidelity tracks |
| [`docs/README.md`](docs/README.md) | Full doc index |
| [`secrets/README.md`](secrets/README.md) | Local key vault (no secrets in git) |

---

## Examples

| Path | What it shows |
|------|----------------|
| [`connectors/web/chorusface-embed.js`](connectors/web/chorusface-embed.js) | MJPEG mount helper |
| [`connectors/web/chorusface-bridge.js`](connectors/web/chorusface-bridge.js) | Browser speak fire-and-forget (prefer server-side) |
| [`scripts/run_chorusface_service.py`](scripts/run_chorusface_service.py) | Headless FaceBridge |
| [`scripts/generate_api_keys.py`](scripts/generate_api_keys.py) | Local encrypted keys |
| [`src/chorusface/host_client.py`](src/chorusface/host_client.py) | `drive_host_voice` / speak helpers |

---

## Development

```bash
git clone https://github.com/insightitsGit/ChorusFace.git
cd ChorusFace
pip install -e ".[dev]"
pytest
# focused host-voice contract:
pytest tests/test_host_voice_default.py tests/test_host_client.py -q
```

---

## Status

**0.2.0b1** — product beta: fixed TickFeed avatar, FaceBridge, host-voice default, MJPEG embed, exclusive API-key lease, Docker packaging.

- Render-quality tracks (blink BJ2, occlusion, …) remain **parked** — [`docs/RenderQualityParked.md`](docs/RenderQualityParked.md).
- Not on PyPI; no LICENSE file in-repo yet (omit license badge until added).
- Public marketing still gated on render unpark (repo docs only).

Author: Insight IT Solutions · GitHub: [insightitsGit/ChorusFace](https://github.com/insightitsGit/ChorusFace)
