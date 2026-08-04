# ChorusFace

**ChorusFace** is a talking-face companion for AI products. Your app (or agent)
keeps the brain — LLM, tools, website chat, orchestration. ChorusFace is the
**face**: it hears what the assistant is about to say, speaks it with TTS, and
moves a photoreal avatar in sync so users see a living face instead of text alone.

This repository is the **product beta** (`0.2.0b1`): one calibrated TickFeed
avatar, a containerized API, and a browser-embeddable live stream.

---

## Why it exists

Most AI products stop at text or browser TTS. Putting a real face on an agent
usually means either a heavy 3D stack, a cloud lip-sync vendor, or rebuilding
your whole UI in WebGL.

ChorusFace takes a different cut:

1. **Host owns the AI** — Insightits, PrismAPI agents, website hubs, or any
   backend that can HTTP POST after an assistant reply.
2. **ChorusFace owns the face** — one shared GPU world, lip sync, and stream.
3. **Integrators send speech intent** — not vertices, not meshes, not a second
   chat model. `POST /prism/speak` with the text; we handle voice + motion.
4. **Humans see the face over MJPEG** — drop an `<img>` (or our embed script)
   into an existing page. No WebGL port of the face runtime required for this beta.

The face itself is an **NWR cell field** (see below): a locked photo identity
with unlocked mouth/eye tissue driven by TickFeed LOOK/FIELD — not a puppet
mesh and not invented face RGB.

---

## What you get in this beta

| You get | You do not get (yet) |
| --- | --- |
| One fixed calibrated avatar (TickFeed world) | User-uploaded / swappable identities |
| AI↔AI speak API (`/prism/speak` + FaceBridge) | ChorusFace-hosted LLM or chat UI as the product |
| Local TTS + lip sync from assistant text | Multi-tenant cloud face SaaS |
| Docker service + `GET /stream.mjpg` embed | Full WebGL face port in the browser |
| Exclusive API keys (one key ↔ one `client_id`) | Anonymous public stream without auth |
| Desktop window for local QA | Free-stretch window that warps the face |

**Product shape:** face companion, not a chatbot. If the face container is down,
the host chat should still work — degraded, without a face.

---

## How it fits together

```
Your agent / website hub          (LLM, tools, session)
        │
        │  POST /auth/activate     (API key + stable client_id)
        │  POST /prism/speak       { "text": "assistant reply…" }
        ▼
ChorusFace service                (Docker or local GPU/CPU)
        │  TTS → TickFeed LOOK/FIELD → NWR field commands
        │
        ├─► humans:  GET /stream.mjpg?token=…&client_id=…
        └─► ops QA:  desktop window via run_chorusface_beta.py
```

Typical host loop:

1. User talks to **your** product.
2. Your LLM produces the assistant reply.
3. Your backend fire-and-forgets that text to ChorusFace `/prism/speak`.
4. Your page shows the live face via `/stream.mjpg` (mute browser TTS when the
   face is speaking to avoid double audio).

---

## NWR — technical brief

**NWR (Neural World Runtime)** is the GPU field substrate ChorusFace runs on.
Pinned copy: [`vendor/nwr/`](vendor/nwr/) (`vendor/nwr/NWR_REVISION.txt`).

### What NWR is

NWR is **not** a face renderer and **not** a second neural net in the hot path.
It is a deterministic, GPU-resident **cell field**:

| Idea | Detail |
| --- | --- |
| World | Dense 2D grid (face worlds use **256×256**) |
| Cell | **32 float channels** — kinematics, material, intent, rules |
| Authority | Channel **31 Master Lock** (`human_lock`) — GPU rejects AI writes on locked cells |
| Tick | OpenGL compute advances the field; CPU is not in the simulation loop |
| Snapshot | `.bds` binary world (~8.4 MB for 256×256×32×f32) |
| Contract | AI / host **proposes** commands → runtime **validates** → GPU **executes** |

Channel groups (simplified):

| Group | Channels | Face use |
| --- | --- | --- |
| Kinematics | 0–7 | Soft-tissue velocity / displacement (±4 impulses) |
| Material | 8–15 | Photo albedo / opacity (identity RGB) |
| Intent | 16–23 | Sparse / future control |
| Rules | 24–31 | Priority + **Master Lock** |

### How ChorusFace uses NWR

```text
Calibration video
  → digest / TickFeed train
  → avatar_face.bds + LOOK plates + GPU display recipe (L00–L11)
Host agent
  → POST /prism/speak
  → TTS + TickFeed LOOK / FIELD vectors
  → validated ±4 (and related) writes into the NWR field
GPU
  → same recipe paints the locked photo identity; mouth / lids move on unlocked cells
  → MJPEG / window shows the live field
```

Concrete mapping in this repo:

| NWR concept | ChorusFace |
| --- | --- |
| Substrate libs | `vendor/nwr/` |
| Field runtime / `.bds` / shaders | `src/chorusface/runtime/`, `shaders/` |
| Digest + cell learning | `src/amin_loop/` |
| Live motion from video / speech | TickFeed (`src/chorusface/tickfeed/`) → LOOK/FIELD |
| Host speak path | FaceBridge / Prism → speech → field commands |
| Identity invariant | Photo + Master Lock — **no invented face RGB / teeth** |

**Why this shape:** the face is one shared NWR world, not a mesh swap or per-user 3D model. Hosts only send speech intent; NWR ownership and the GPU recipe keep identity stable while unlocked mouth/eye cells move.

Deeper docs: [`docs/AMIN_DESIGN.md`](docs/AMIN_DESIGN.md), [`docs/NWRDataDesign.md`](docs/NWRDataDesign.md), [`docs/BDSMotionMap.md`](docs/BDSMotionMap.md).

---

## How to test (two paths)

| Path | When to use | What you need |
| --- | --- | --- |
| **A — Run your own container** | You cloned the repo and want a local face | Docker + Python 3.11+, GPU optional |
| **B — Use our container** | We already run a shared face service | API key + `client_id` we issued you |

**API keys are never in git.** Self-hosts generate their own vault. Shared-service testers get one key offline from the project team.

---

## Path A — Clone and run your own container

### 1. Prerequisites

- Git, Python **3.11+**, Docker Desktop (or Docker Engine + Compose)
- Windows, macOS, or Linux
- Optional: NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) (Linux) for better GL

### 2. Clone and install

```powershell
git clone https://github.com/insightitsGit/ChorusFace.git
cd ChorusFace
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows
# source .venv/bin/activate    # macOS / Linux
pip install -U pip
pip install -e ".[ml,voice]"
```

### 3. Build the TickFeed world (required once)

Calibrated worlds under `output/` are **not** committed. Build from the in-repo calibration take:

```powershell
python scripts/build_tickfeed_demo.py
# optional: --clean  to wipe output/worlds/tickfeed first
```

Expect `output/worlds/tickfeed/avatar_face.bds` (and companion plates). Without this step, `docker compose` / image build will fail.

### 4. Generate local API keys (required once)

```powershell
python scripts/generate_api_keys.py
```

Creates gitignored files under `secrets/`:

| File | Purpose |
| --- | --- |
| `.master_key` | Fernet master — **never share / never commit** |
| `api_keys.vault.enc` | Encrypted vault (mounted into the container) |
| `api_keys.handoff.local.txt` | Plaintext keys to give integrators |
| `api_keys.hashes.json` | SHA-256 digests only |

Hand **one** key from `api_keys.handoff.local.txt` to each AI/system. Details: [`secrets/README.md`](secrets/README.md).

### 5. Start the face container

```powershell
docker compose up --build
```

Service listens on **`http://localhost:8766`**.

Quick checks:

```powershell
curl http://localhost:8766/health
```

Headless service without Docker (debug):

```powershell
python scripts/run_chorusface_service.py
# optional window: python scripts/run_chorusface_service.py --visible
```

Desktop product beta (window + TTS):

```powershell
python scripts/run_chorusface_beta.py
```

### 6. Activate a key, speak, embed

Pick a stable UUID for your process (`client_id`). One API key ↔ one active `client_id` (exclusive lease).

```powershell
# Activate
curl -X POST http://localhost:8766/auth/activate `
  -H "Authorization: Bearer <KEY_FROM_HANDOFF>" `
  -H "Content-Type: application/json" `
  -d "{\"client_id\":\"11111111-2222-3333-4444-555555555555\"}"

# Speak (AI ↔ AI)
curl -X POST http://localhost:8766/prism/speak `
  -H "Authorization: Bearer <KEY_FROM_HANDOFF>" `
  -H "X-ChorusFace-Client-Id: 11111111-2222-3333-4444-555555555555" `
  -H "Content-Type: application/json" `
  -d "{\"text\":\"Hello there. How are you today?\"}"
```

Browser embed (token + client_id in query — `<img>` cannot send `Authorization`):

```html
<img
  src="http://localhost:8766/stream.mjpg?token=KEY_FROM_HANDOFF&client_id=11111111-2222-3333-4444-555555555555"
  alt="ChorusFace"
  width="320"
  height="320"
/>
```

Or use [`connectors/web/chorusface-embed.js`](connectors/web/chorusface-embed.js).

Release when your AI shuts down:

```powershell
curl -X POST http://localhost:8766/auth/release `
  -H "Authorization: Bearer <KEY_FROM_HANDOFF>" `
  -H "X-ChorusFace-Client-Id: 11111111-2222-3333-4444-555555555555"
```

### Compose / env (self-host)

| Variable | Default | Meaning |
| --- | --- | --- |
| `CHORUSFACE_BRIDGE_PORT` | `8766` | Host port mapped to the container |
| `CHORUSFACE_BRIDGE_CORS` | `*` | CORS for browser clients |
| `CHORUSFACE_STREAM_FPS` | `12` | MJPEG encode rate |
| `CHORUSFACE_KEY_LEASE` | `1` | Exclusive key ↔ `client_id` |
| `CHORUSFACE_KEY_BIND_IP` | `0` | Also sticky-bind activation IP |
| `CHORUSFACE_KEY_LEASE_TTL_S` | `900` | Idle lease expiry (seconds) |
| `MODERNGL_BACKEND` | `egl` | GL backend inside the container |

`docker-compose.yml` mounts:

- `./output/worlds/tickfeed` → world (read-only)
- `./secrets` → encrypted vault (read-only; **never bake keys into the image**)

GPU example (Linux + NVIDIA):

```bash
docker run --gpus all -p 8766:8766 \
  -v "$PWD/output/worlds/tickfeed:/app/output/worlds/tickfeed:ro" \
  -v "$PWD/secrets:/app/secrets:ro" \
  -e MODERNGL_BACKEND=egl \
  chorusface-face:beta
```

---

## Path B — Use our shared container (API key required)

If the project team runs a shared face service, you do **not** need to build the world or Docker image. You need:

1. **Base URL** of the shared service (example: `https://face.example.com` — ask the team for the current URL)
2. **One API key** from the team handoff (same shape as `chorusface_sk_…`)
3. A **stable `client_id` UUID** for your AI/process

Then:

```powershell
$BASE = "https://face.example.com"   # replace with the URL you were given
$KEY  = "chorusface_sk_…"                # your issued key
$CID  = "11111111-2222-3333-4444-555555555555"

curl -X POST "$BASE/auth/activate" `
  -H "Authorization: Bearer $KEY" `
  -H "Content-Type: application/json" `
  -d "{\"client_id\":\"$CID\"}"

curl -X POST "$BASE/prism/speak" `
  -H "Authorization: Bearer $KEY" `
  -H "X-ChorusFace-Client-Id: $CID" `
  -H "Content-Type: application/json" `
  -d "{\"text\":\"Hello from our agent.\"}"
```

Embed:

```text
{BASE}/stream.mjpg?token={KEY}&client_id={CID}
```

Health (no auth): `GET {BASE}/health`

**Rules for shared service**

- One key is leased to **one** `client_id` at a time. A second system using the same key gets `403` (“API key in use by another system”).
- Heartbeat or keep speaking so the lease does not idle-expire (`POST /auth/heartbeat`).
- Call `POST /auth/release` when done so others (or your next process) can activate.
- Do not put keys in front-end source that ships to the public internet; prefer **server-side** speak from your hub. HTTPS pages cannot call plain `http://127.0.0.1` from the browser.

---

## API surface (both paths)

| Method | Path | Auth | Purpose |
| --- | --- | --- | --- |
| `GET` | `/health` | none | Liveness |
| `POST` | `/auth/activate` | Bearer key | Lease key ↔ `client_id` |
| `POST` | `/auth/heartbeat` | Bearer + `X-ChorusFace-Client-Id` | Keep lease alive |
| `POST` | `/auth/release` | Bearer + `X-ChorusFace-Client-Id` | Free the key |
| `GET` | `/auth/status` | Bearer + client id | Lease status |
| `POST` | `/prism/speak` | Bearer + client id | AI↔AI speak (`text` / `speech` / `message` / `response`) |
| `POST` | `/speak` | same | Alias of `/prism/speak` |
| `GET` | `/stream.mjpg` | `?token=&client_id=` | Live MJPEG embed |
| `GET` | `/preview.jpg` | `?token=&client_id=` | Single JPEG frame |
| `GET` | `/preview` | Bearer | Single PNG frame |

Python helper (from this repo or installed package):

```python
from chorusface.prism_adapter import forward_speak, SpeakIntent

forward_speak({"text": "Hello there"})
forward_speak(SpeakIntent(text="Hello there"))
```

CLI:

```powershell
python -m chorusface.prism_adapter "Hello there"
```

Full contract: [`docs/FaceServiceEmbed.md`](docs/FaceServiceEmbed.md).

---

## Host website / Insightits-style hub

Server-side (recommended): after the assistant reply, fire-and-forget `POST /prism/speak` with the API key + `client_id`. Mute browser TTS when the face is speaking.

Typical env on the host:

| Env | Example |
| --- | --- |
| `CHORUSFACE_BRIDGE_ENABLED` | `1` |
| `CHORUSFACE_BRIDGE_URL` | `http://127.0.0.1:8766` or shared URL |
| `CHORUSFACE_BRIDGE_TOKEN` | your API key |
| `CHORUSFACE_CLIENT_ID` | stable UUID for that host |
| `CHORUSFACE_EMBED_URL` | `{BASE}/stream.mjpg?token=…&client_id=…` |

Embed snippet: [`connectors/web/chorusface-embed.js`](connectors/web/chorusface-embed.js). Product overview: [`docs/ProductBeta.md`](docs/ProductBeta.md).

---

## Acceptance checklist

1. `GET /health` → ok  
2. Activate with issued/local key → `200`  
3. `POST /prism/speak` moves the mouth  
4. Browser shows live face at `/stream.mjpg?token=…&client_id=…`  
5. Second process with the **same** key and a **different** `client_id` → `403`  
6. Host chat still works if the face service is down (degraded, no face)

---

## Design docs (Amin / NWR substrate)

| Doc | What |
| --- | --- |
| [`docs/AMIN_DESIGN.md`](docs/AMIN_DESIGN.md) | Master design + steps → code |
| [`docs/README.md`](docs/README.md) | Full doc index |
| [`docs/FaceServiceEmbed.md`](docs/FaceServiceEmbed.md) | Container API + Prism + embed |
| [`docs/ProductBeta.md`](docs/ProductBeta.md) | Host contract |
| [`docs/RenderQualityParked.md`](docs/RenderQualityParked.md) | Parked fidelity tracks |
| [`secrets/README.md`](secrets/README.md) | Local key vault |

### Amin walkthrough → code

| Step | What | Where |
| --- | --- | --- |
| 1 | What is NWR | `vendor/nwr` |
| 2 | 32 floats / cell | `amin_loop.cells` |
| 3 | Control + neighbors | `amin_loop.control` |
| 4 | Digest → regions | `amin_loop.digest` |
| 5–7 | Regions + props | `amin_loop.regions` |
| 6 | Word/sound/emotion maps | `amin_loop.mapping` |
| 8 | GPU display recipe | `amin_loop.gpu_recipe` + `avatar.frag` |
| 9 | Live vectors | `chorusface.live_vector` |
| 10 | Train + play | `scripts/amin_train.py` / `scripts/build_tickfeed_demo.py` |

### Rules we keep

1. World = GPU field of 32-float cells (`.bds`)  
2. AI proposes → runtime validates → GPU executes  
3. Channel 31 Master Lock = identity boundary  
4. No invented face RGB / teeth  
5. No Path A ownership seals  
6. Capture open/smile plates must **paint** on the mouth (same GPU recipe)

### Layout

```text
vendor/nwr/           NWR libs
src/amin_loop/        Walkthrough implementation
src/chorusface/           GPU runtime, FaceBridge, Prism adapter, key lease
scripts/              Train, TickFeed build, service, key generation
connectors/web/       Embed + bridge JS
secrets/              Local vault only (gitignored except README)
docs/                 Design + service docs
Dockerfile            Face service image
docker-compose.yml    Local / lab compose
```

---

## Troubleshooting

| Symptom | Likely fix |
| --- | --- |
| Docker build fails on `COPY output/worlds/tickfeed` | Run `python scripts/build_tickfeed_demo.py` first |
| `401` on speak/stream | Wrong key, or vault not mounted at `/app/secrets` |
| `403` API key in use | Another `client_id` holds the lease — release it or wait for TTL |
| Stream opens but blank / GL errors | Try host `run_chorusface_service.py --visible`; on Linux GPU use `--gpus all` |
| HTTPS site cannot load `http://127.0.0.1` stream | Mixed content — use a TLS-terminated shared URL, or proxy the stream through your hub |
| Mouth never moves | Confirm activate succeeded, then `POST /prism/speak` with both Bearer and `X-ChorusFace-Client-Id` |

Rotate keys (self-host):

```powershell
python scripts/generate_api_keys.py --force
# restart container so it reloads the vault
docker compose up -d --force-recreate
```
