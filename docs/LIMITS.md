# ChorusFace — limits (beta `0.2.0b1`)

Honest boundaries. Do not invent capabilities past this file and the package source.

## Product shape

| Limit | Meaning |
|-------|---------|
| **Face companion, not a chatbot** | Host owns the LLM. ChorusFace does not host chat. |
| **Host owns TTS (default)** | ChorusFace does not synthesize audio unless lab `--tts` / `CHORUSFACE_TTS=1`. |
| **One fixed avatar** | TickFeed world from the calibrated take — no user photo swap / multi-identity picker. |
| **Face down ≠ product down** | Host chat + host voice should keep working if FaceBridge is offline. |

## Auth & keys

- API keys live under `secrets/` (gitignored). Never commit vault, master key, or handoff plaintext.
- Exclusive lease: one key ↔ one `client_id` at a time (`X-ChorusFace-Client-Id`).
- Do not put keys in public front-end JS.

## Web / embed

- HTTPS pages cannot call `http://127.0.0.1` (mixed content). Prefer server-side `/voice/*` or `/prism/speak`, or a TLS-terminated face URL.
- MJPEG `<img>` auth uses `?token=` + `?client_id=` (browsers cannot set `Authorization` on image URLs).

## Render / identity

- Parked fidelity tracks (blink BJ2, kill `open.png`, occlusion, etc.): [`RenderQualityParked.md`](RenderQualityParked.md).
- Do not claim bit-identical replay across GPUs or “Master Lock marketing” authority claims until parked work is closed.
- Identity invariant: photo + Master Lock — no invented face RGB / teeth (design docs).

## Not in this beta

- Multi-tenant cloud face SaaS
- Full WebGL port of the face runtime in the browser
- Swappable / user-uploaded identities
- ChorusFace as the default TTS engine
- Published PyPI wheel (`pip install chorusface` from PyPI) — git / editable install only

## Related

- [`ProductBeta.md`](ProductBeta.md) · [`VoiceSync.md`](VoiceSync.md) · [`FaceServiceEmbed.md`](FaceServiceEmbed.md) · [`ai-overview.md`](ai-overview.md)
