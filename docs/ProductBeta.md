# ChorusFace product beta — host AI chat → face

**Version:** `0.2.0b1` (TickFeed LOOK/FIELD fidelity beta)

ChorusFace is the **face companion**. The host product (e.g. Insightits website chat)
owns the LLM. After each assistant reply, the host POSTs the spoken text to
FaceBridge (or PrismAPI `/prism/speak`); ChorusFace TTS + TickFeed drive the mouth.

**Web product path (preferred):** containerized face service + MJPEG embed — see
[`FaceServiceEmbed.md`](FaceServiceEmbed.md).

```
Host chat (LLM)  →  POST /prism/speak  →  ChorusFace TTS + TickFeed LOOK
Website page     ←  GET /stream.mjpg   (embed)
```

---

## What this beta is (and is not)

| Included | Not in this beta |
| --- | --- |
| Fixed calibrated TickFeed avatar (blonde woman take) | **Avatar is not changeable** — no user photo swap, no multi-identity picker |
| Host-driven `/speak` (LLM outside ChorusFace) | ChorusFace-hosted chat model |
| Local TTS + lip sync from assistant text | Host PCM `/voice/*` lip-lock (documented upgrade) |
| Resizable window with **locked aspect ratio** | Free-stretch window that warps the face |
| FaceBridge CORS + host connectors | Cloud multi-tenant face relay |
| Fidelity ladder through RF6 / P3 / P12 HUD | Parked blink BJ2, kill `open.png`, occlusion (see [`RenderQualityParked.md`](RenderQualityParked.md)) |

**Fixed avatar:** product beta ships one world — `output/worlds/tickfeed` built from
the dense calibration take. Operators rebuild that world with
`scripts/build_tickfeed_demo.py`; end users do not pick or upload a face in this
release. Hot-swap identity is out of scope until a later product cut.

---

## Window size and aspect ratio

Default composition: **1024×1320** (square portrait + bottom chat band).

- Window is **resizable**.
- Resize **keeps the aspect ratio** so the face never stretches.
- Drag width or height; the other side snaps to match `1024/1320 ≈ 0.7758`.
- Size clamps between 480px and 2400px on the longer side.
- Face-only layout (`--no-chat-box`): square **1024×1024**, aspect `1.0`.

```powershell
python scripts/run_chorusface_beta.py
# optional initial width (height derived):
python -m chorusface --product-beta --bridge --bridge-direct-speak --tts --window-width 900 ...
```

Internals: [`src/chorusface/window_layout.py`](../src/chorusface/window_layout.py) +
`AvatarFaceApp.on_resize` snap. Viewport `fixed_aspect_ratio` matches.

---

## Launch

```powershell
# From ChorusFace repo (calibrated world required)
python scripts/run_chorusface_beta.py
# or: chorusface-beta
```

Optional flags on the launcher:

| Flag | Meaning |
| --- | --- |
| `--fidelity-hud` | Show FIDELITY overlay |
| `--allow-remote-bind` | LAN kiosk (with `CHORUSFACE_BRIDGE_HOST=0.0.0.0`) |
| `--no-tts` | Text-only visemes (not recommended for QA) |

Env:

| Variable | Default | Role |
| --- | --- | --- |
| `CHORUSFACE_WORLD` | `output/worlds/tickfeed` | Calibrated world directory (**fixed avatar**) |
| `CHORUSFACE_BRIDGE_HOST` | `127.0.0.1` | Bind host |
| `CHORUSFACE_BRIDGE_PORT` | `8766` | Bind port |
| `CHORUSFACE_BRIDGE_TOKEN` | `chorusface-beta` | Bearer token |
| `CHORUSFACE_BRIDGE_CORS` | `*` | CORS origins (`*` or comma list) |
| `CHORUSFACE_PRODUCT_BETA` | set by launcher | Banner / product mode |

**LAN kiosk** (Flask server → face machine):

```powershell
$env:CHORUSFACE_BRIDGE_HOST="0.0.0.0"
$env:CHORUSFACE_BRIDGE_TOKEN="replace-me"
python scripts/run_chorusface_beta.py --allow-remote-bind
```

Build / rebuild the fixed world (operator only):

```powershell
python scripts/build_tickfeed_demo.py --clean
```

---

## Host contract

### Speak (beta path)

```http
POST /speak
Authorization: Bearer <token>
Content-Type: application/json

{"text":"Hello there. How are you today?"}
```

Accepted text keys: `text` | `speech` | `message` | `response`.

Response: `{ "queued": true, "text": "..." }`.

Liveness (no auth): `GET /health` → `{ "ok": true, "service": "chorusface", "product": "beta" }`.

Other routes (auth required): `GET /status|/probe|/cells|/preview|/screenshot`,
`POST /voice/expect|pcm|end|timeline`, `POST /cells/drive`, `POST /calibrate`.

### Smoke

```powershell
python -m chorusface.host_client "Hello there"
# or: chorusface-host "Hello there"
```

### Python API

```python
from chorusface.host_client import speak, speak_async

speak_async(assistant_text)          # fire-and-forget
result = speak(assistant_text)       # never raises by default
```

### Browser (local/http demos only)

**HTTPS pages cannot call `http://127.0.0.1`** (mixed content). Prefer
server-side speak for production hosts.

```html
<script src="/path/to/connectors/web/chorusface-bridge.js"></script>
<script>
  window.CHORUSFACE_BRIDGE = { url: 'http://127.0.0.1:8766', token: 'chorusface-beta' };
  ChorusFaceBridge.speakFireAndForget(assistantText);
</script>
```

---

## Insightits website chat

Server-side hook (recommended for HTTPS):

1. On Flask: `CHORUSFACE_BRIDGE_ENABLED=1`, `CHORUSFACE_BRIDGE_URL=http://127.0.0.1:8766`, `CHORUSFACE_BRIDGE_TOKEN=chorusface-beta`
2. After a full assistant reply in `handle_website_agent_chat`, fire-and-forget `POST /speak` via `services/chorusface_bridge.py`
3. Response may include `chorusfaceSpeakQueued: true` so the browser skips Web Speech TTS (avoid double audio)

See `.env.local.example` in the Insightits repo.

Upgrade path (later): host server TTS → `POST /voice/timeline` or `/voice/pcm` for
lip-lock to the host voice clock ([`VoiceSync.md`](VoiceSync.md)).

---

## Operator checklist

1. World exists at `output/worlds/tickfeed` (plates, timeline, ML joblibs).
2. `python scripts/run_chorusface_beta.py` — window opens, ratio locked, bridge printed.
3. Resize the window — sides stay proportional; face stays square in the portrait frame.
4. `python -m chorusface.host_client "Hello there"` — TTS + mouth motion.
5. Enable Insightits bridge env; website chat reply moves the face.
6. Stop the face process — website chat still works (speak is best-effort).

---

## Acceptance

1. `python scripts/run_chorusface_beta.py` opens the face + bridge
2. Window resize keeps aspect; face does not stretch
3. `python -m chorusface.host_client "Hello there"` moves the mouth with TTS
4. Website chat reply triggers `/speak`; chat still works if the face is down
5. Avatar identity is the fixed TickFeed world for this beta

## Related docs

- [`AvatarChat.md`](AvatarChat.md) — chat / biomechanics pipeline
- [`Architecture.md`](Architecture.md) — FaceBridge control surface
- [`TickFeedDesign.md`](TickFeedDesign.md) — LOOK/FIELD packages
- [`RenderFidelityLadder.md`](RenderFidelityLadder.md) — fidelity steps kept in beta
- [`RenderQualityParked.md`](RenderQualityParked.md) — parked tracks
- [`AvatarCalibrationPrompt.md`](AvatarCalibrationPrompt.md) — how the fixed take was directed
