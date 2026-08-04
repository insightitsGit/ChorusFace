# ChorusFace product beta — host AI chat → face

**Version:** `0.2.0b1` (TickFeed LOOK/FIELD fidelity beta)

ChorusFace is the **face companion**. The host product (e.g. Insightits website
chat) owns the **LLM and the TTS**. After each assistant reply, the host plays
its own voice and drives the face over FaceBridge `/voice/*` (or cues the mouth
with `/prism/speak` when PCM is not available yet).

**Web product path (preferred):** containerized face service + MJPEG embed — see
[`FaceServiceEmbed.md`](FaceServiceEmbed.md). Host-voice lip-lock details:
[`VoiceSync.md`](VoiceSync.md).

```
Host chat (LLM + TTS)
    → play host audio
    → POST /voice/expect|/pcm|/end   (or /voice/timeline)
ChorusFace TickFeed LOOK
Website page ← GET /stream.mjpg   (embed)
```

---

## What this beta is (and is not)

| Included | Not in this beta |
| --- | --- |
| Fixed calibrated TickFeed avatar (blonde woman take) | **Avatar is not changeable** — no user photo swap, no multi-identity picker |
| Host-owned TTS → `/voice/*` lip-lock (**product default**) | ChorusFace as the default TTS engine |
| Mouth-cue `/speak` / `/prism/speak` (text timing, no face audio) | ChorusFace-hosted chat model |
| Optional lab `--tts` / `CHORUSFACE_TTS=1` | Required local synthesis for production hosts |
| Resizable window with **locked aspect ratio** | Free-stretch window that warps the face |
| FaceBridge CORS + host connectors | Cloud multi-tenant face relay |
| Fidelity ladder through RF6 / P3 / P12 HUD | Parked blink BJ2, kill `open.png`, occlusion (see [`RenderQualityParked.md`](RenderQualityParked.md)) |

**Fixed avatar:** product beta ships one world — `output/worlds/tickfeed` built from
the dense calibration take. Operators rebuild that world with
`scripts/build_tickfeed_demo.py`; end users do not pick or upload a face in this
release. Hot-swap identity is out of scope until a later product cut.

---

## Voice contract (read this first)

| Path | Who speaks | When to use |
| --- | --- | --- |
| **`/voice/expect` + `/voice/pcm` + `/voice/end`** | **Host TTS** | **Default.** Host has transcript + PCM (or realtime chunks). |
| **`/voice/timeline`** | **Host TTS** | Host already timed phonemes to its audio clock. |
| **`/speak` / `/prism/speak`** | Host (or silent) | Mouth cue from text only — ChorusFace does **not** play audio. |
| **`--tts` on ChorusFace** | ChorusFace | **Lab only** — demos without a host voice stack. |

Python helper (product default):

```python
from chorusface.host_client import drive_host_voice

drive_host_voice("Hello there.", host_tts_pcm16_bytes, sample_rate=24000)
```

CLI:

```powershell
python -m chorusface.host_client --voice --pcm-file reply.pcm "Hello there."
# text-only mouth cue (no face audio):
python -m chorusface.host_client "Hello there."
```

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
# lab local TTS (optional):
python scripts/run_chorusface_beta.py --tts
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
| `--tts` | Lab only: local ChorusFace TTS (`CHORUSFACE_TTS=1` also works) |

Env:

| Variable | Default | Role |
| --- | --- | --- |
| `CHORUSFACE_WORLD` | `output/worlds/tickfeed` | Calibrated world directory (**fixed avatar**) |
| `CHORUSFACE_BRIDGE_HOST` | `127.0.0.1` | Bind host |
| `CHORUSFACE_BRIDGE_PORT` | `8766` | Bind port |
| `CHORUSFACE_BRIDGE_TOKEN` | `chorusface-beta` | Bearer token (or vault key) |
| `CHORUSFACE_BRIDGE_CORS` | `*` | CORS origins (`*` or comma list) |
| `CHORUSFACE_PRODUCT_BETA` | set by launcher | Banner / product mode |
| `CHORUSFACE_TTS` | unset / off | Lab local TTS |

```powershell
$env:CHORUSFACE_BRIDGE_HOST="0.0.0.0"
$env:CHORUSFACE_BRIDGE_TOKEN="replace-me"
python scripts/run_chorusface_beta.py --allow-remote-bind
```

---

## FaceBridge surface

Liveness (no auth): `GET /health` → `{ "ok": true, "service": "chorusface", "product": "beta" }`.

Auth: activate + `X-ChorusFace-Client-Id` — see [`secrets/README.md`](../secrets/README.md).

Also available: `GET /status`, `/probe`, `/cells`, `/preview`, `/preview.jpg`,
`/stream.mjpg`, `POST /cells/drive`, `POST /calibrate`.

---

## Host connectors

**HTTPS pages cannot call `http://127.0.0.1`** (mixed content). Prefer
server-side voice/speak for production hosts.

```html
<script src="/path/to/connectors/web/chorusface-bridge.js"></script>
<script>
  window.CHORUSFACE_BRIDGE = { url: 'http://127.0.0.1:8766', token: 'chorusface-beta' };
  // Mouth cue only — host still plays its own TTS:
  ChorusFaceBridge.speakFireAndForget(assistantText);
</script>
```

---

## Insightits website chat

Server-side hook (recommended for HTTPS):

1. On Flask: `CHORUSFACE_BRIDGE_ENABLED=1`, `CHORUSFACE_BRIDGE_URL=http://127.0.0.1:8766`, `CHORUSFACE_BRIDGE_TOKEN=…`
2. After a full assistant reply, fire-and-forget face drive via `services/chorusface_bridge.py`
3. **Host TTS stays on** (browser Web Speech or server TTS). Do **not** mute host
   audio because the face queued — ChorusFace is not the speaker.
4. Preferred upgrade: host server TTS PCM → `drive_host_voice` / `/voice/*`
5. Until PCM is available: `/prism/speak` cues the mouth on text timing while the
   host voice plays independently

See `.env.local.example` in the Insightits repo and [`VoiceSync.md`](VoiceSync.md).

---

## Operator checklist

1. World exists at `output/worlds/tickfeed` (plates, timeline, ML joblibs).
2. `python scripts/run_chorusface_beta.py` — window opens, ratio locked, bridge printed, **local TTS OFF**.
3. Resize the window — sides stay proportional; face stays square in the portrait frame.
4. Host voice smoke: `python -m chorusface.host_client --voice --pcm-file reply.pcm "Hello there"`
5. Text mouth-cue smoke: `python -m chorusface.host_client "Hello there"` (no face audio).
6. Enable Insightits bridge env; website chat reply moves the face while host TTS still speaks.
7. Stop the face process — website chat + host TTS still work (face drive is best-effort).

---

## Acceptance

1. `python scripts/run_chorusface_beta.py` opens the face + bridge with local TTS off
2. Window resize keeps aspect; face does not stretch
3. Host PCM via `/voice/*` moves the mouth on the host audio clock
4. `/prism/speak` moves the mouth without ChorusFace playing audio
5. Website chat reply can drive the face; chat still works if the face is down
6. Avatar identity is the fixed TickFeed world for this beta

## Related docs

- [`VoiceSync.md`](VoiceSync.md) — host PCM / timeline channel
- [`FaceServiceEmbed.md`](FaceServiceEmbed.md) — container + MJPEG
- [`AvatarChat.md`](AvatarChat.md) — chat / biomechanics pipeline
- [`RenderQualityParked.md`](RenderQualityParked.md) — parked fidelity tracks
