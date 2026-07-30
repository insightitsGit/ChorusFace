# AI API

How an external assistant inspects and commands a Neural World Runtime world
without owning GPU memory or the world tensor.

## Architecture rule

```text
User → Natural Language → AI Assistant → AI Command Compiler
        → Validated Deterministic Commands → Scheduler → GPU → .bds
```

The runtime owns the world. The AI proposes deterministic operations. The
runtime validates them. The GPU executes them.

Human authority always wins. AI packets use source magnitude `±2` (paint/erase),
`±3` (temperature), or `±4` (velocity impulse). The constraint pass enforces two
things about the human boundary, on the GPU, whatever reached the command buffer:

- An AI write onto a cell with `channel 31 >= 0.5` is aborted.
- An AI write can never *raise* channel 31. Painting `human_barrier` — the one
  anchor carrying a lock — gives an AI caller the wall without the lock. So an
  agent cannot mint a boundary that it is then permanently barred from clearing.

Neither depends on the authority level. An AI packet at `user` priority is still
an AI packet.

## Handing a world to ChatGPT or Gemini

A chat model cannot run this runtime — there is no GPU behind a chat window — so
a bare `.bds` gives it a binary blob and a guess. `ai_handoff.py` packages the
artifacts a chat model can actually consume:

```powershell
python ai_handoff.py --world output/worlds/from-video/chorus-music_fluid-sandbox_256.bds --output output/handoffs/chorus-music_fluid-sandbox --ticks 600 --frames 9
```

| File | Purpose |
| --- | --- |
| `filmstrip.png` | The run sampled into one labelled contact sheet. A single image is the only view every model reads reliably, so this is what conveys motion. |
| `world.gif` | The same samples as an animated loop. |
| `frames/` | Each sampled frame at full resolution. |
| `timeline.json` | The numbers behind each sample, so the model reasons instead of guessing from pixels. |
| `context.json` | Metadata, statistics, and the full command grammar. |
| `world.bds` | The binary world, for a code interpreter. |
| `read_bds.py` | A NumPy-only reader, verified against `bds_format.load_bds`. |
| `READ_ME_FIRST.md` | The briefing: what the world is, what the model may not do, and how to reply. |

Upload the folder (or at minimum the briefing, filmstrip, and context), then
apply the model's reply:

```powershell
python ai_agent.py --commands-file reply.json --dry-run   # validate
python ai_agent.py --commands-file reply.json             # apply
```

Rebuild the bundle after applying to give the model its next observation. That
loop is turn-based by design: the model never touches GPU memory, and its
commands pass the same validation and human-lock enforcement as any other
source.

Two details keep the bundle honest. The worked example in the briefing is taken
from the live `COMMAND_SCHEMA`, so it cannot drift from what the compiler
accepts, and worlds converted from video carry an explicit warning that only
motion, colour, and contours survived ingestion — there are no objects or story
for the model to describe.

## Inspecting a world

### Drag a `.bds` into an assistant

Every saved `.bds` header now includes an optional `ai_metadata` block:

- world name / description
- cell schema groups
- materials
- supported commands
- unsupported commands, each with the reason it is unavailable

Older readers ignore unknown header fields.

### Python helpers

```python
from bds_format import load_bds
from ai_world import (
    generate_ai_summary,
    inspect_region,
    export_ai_context,
    render_preview,
)

header, grid = load_bds("world.bds")
print(header["ai_metadata"])
print(generate_ai_summary(grid, tick=0))
print(inspect_region(grid, x=128, y=128, radius=16))
context = export_ai_context(grid, tick=0, world_name="world")
# Optional pixels; prefer summaries for reasoning:
# png = render_preview(grid, resolution=1024)
```

### HTTP bridge

```powershell
python main.py --ai-server
```

The bridge binds loopback and will not do otherwise: a non-loopback `--ai-host`
is rejected at construction unless you also pass `--allow-remote-bind`. Every
route but `/` and `/health` needs `Authorization: Bearer <token>`.

| Route | Purpose |
| --- | --- |
| `GET /schema` | Grammar + named commands, narrowed to your authority |
| `GET /state` | Observation, summary, and context |
| `GET /context` | Bundled AI context JSON |
| `GET /inspect?x=&y=&radius=` | Region statistics |
| `POST /inspect` | Same with JSON body |
| `GET /screenshot` / `GET /preview` | Optional PNG |
| `POST /commands` | Native or named commands |

## Generating commands

### Native grammar (preferred for complex regions)

```json
{
  "commands": [
    {
      "action": "paint",
      "category": "solid",
      "region": {"type": "rectangle_outline", "min": [40, 40], "max": [216, 216], "thickness": 3}
    }
  ]
}
```

### What a bridge caller may not ask for

The bridge runs at `ai` authority by default, and two requests are refused there
because neither goes through the per-cell write path that channel 31 governs:

| Request | Why | Use instead |
| --- | --- | --- |
| `paint` with `category: "human_barrier"` | Mints a human lock on every cell it covers | `category: "solid"` — a hard surface with no lock |
| `save` / `load` | Reads or writes the world file, replacing a world wholesale | Ask the operator, or start the runtime with `--ai-authority user` |

`reset` is still available. A reset requested below `user` authority carries every
human-locked cell into the fresh world, so it cannot be used to clear work that a
cell write would have been refused on.

You do not have to memorise that table. `GET /schema` is narrowed to the authority
the bridge grants you: below `user`, the withheld actions are absent from
`actions`, `human_barrier` is absent from the paintable categories, the worked
example uses `solid`, and a `restrictions` block names what was withheld and why.
The same narrowing applies to `command_schema` and `paintable_categories` inside
`/state` and `/context`, and to the grammar `ai_agent.py` puts in front of a model
— so a model that plans from what it was handed cannot plan around a refusal.
Start the runtime with `--ai-authority user` and the full grammar comes back.

### Named AI commands

```json
{"command": "SetMaterial", "material": "Active Fluid", "center": [100, 120], "radius": 18}
{"command": "Erase", "center": [80, 80], "radius": 10}
{"command": "IncreaseTemperature", "center": [128, 128], "radius": 12, "amount": 0.1}
```

Python:

```python
from ai_command_compiler import (
    PaintMaterialCommand,
    EraseCommand,
    IncreaseTemperatureCommand,
    compile_ai_json,
    compile_sealed,
)

ops = PaintMaterialCommand("active_fluid", (100, 120), 18, tick=12).to_operations()
sealed = EraseCommand((80, 80), 10, tick=12).seal()
ops = compile_sealed(sealed)
```

Each sealed command includes a CRC over `{name, tick, payload}`. Tampered
envelopes are rejected.

### Entities

`spawn_entity` and `remove_entity` are supported. They resolve against the
CPU-side registry in `entities.py`, which gives each entity a deterministic id,
a position, and a behaviour that it expresses through ordinary segment and
temperature-delta writes — so entities obey the same authority ordering and
human locks as everything else.

```python
from ai_command_compiler import RemoveEntityCommand, SpawnEntityCommand

ops = SpawnEntityCommand("emitter", (64, 200), radius=4.0, tick=30).to_operations()
ops = RemoveEntityCommand("emitter-0001", tick=90).to_operations()
```

Kinds are `emitter`, `blob`, `obstacle`, `heater`, and `chiller`; read the live
catalogue from `export_ai_context()["entity_kinds"]` rather than hard-coding it.
Unknown kinds, out-of-range radii, unknown ids, and populations beyond
`MAX_ENTITIES` are rejected.

### Unsupported on purpose

`export_ai_context()["unsupported_commands"]` maps each unavailable capability to
the reason it is unavailable. Treat that list as authoritative and do not invent
behaviour for anything in it.

## Deterministic replay

1. Save a base `.bds` snapshot.
2. Record tick-stamped operations into a `.bdl` log (`python main.py --record DIR`).
3. Replay with `export_video.py --world DIR/session.bds --log DIR/session.bdl`.

Identical command streams reproduce bit-identical worlds on the same GPU and
driver. To check agreement across machines, record a reference report on one and
verify the other against it:

```powershell
python determinism.py --world session.bds --log session.bdl --ticks 240 --record output/reports/reference.json
python determinism.py --world session.bds --log session.bdl --ticks 240 --verify output/reports/reference.json
```

The report carries a device fingerprint and per-tick world hashes, so a
divergence is reported with the tick and channel where it first appears.

## Human priority

| Level | Writer |
| --- | --- |
| `user` | mouse |
| `constraint` | scripted invariants |
| `ai` | bridge / swarm / named commands |
| `background` | simulation only |

A write succeeds only when its authority is ≥ the cell's stored authority.
AI callers cannot request a priority above `ai`. Locked cells (`channel 31`)
reject AI source packets entirely, and no AI packet can raise that channel at any
authority — authority and writer identity are separate facts and the shader
checks both.

## Assistant integration examples

### ChatGPT / Claude / Gemini

1. Upload or paste `export_ai_context()` JSON, or the `.bds` header's `ai_metadata`.
   Pass `authority=` so the grammar in it matches what the bridge will grant.
2. Ask for a JSON command object only.
3. Validate locally with `compile_ai_json` / `ai_agent.py --dry-run`. A dry run does
   not contact the bridge, so it validates at the local default and says so; drop
   `--dry-run` to have the bridge's own authority applied before anything is sent.
4. Submit through the bridge.

### Cursor

1. Open `docs/AI_API.md` and `GET /schema`.
2. Generate commands against the live `/state` observation.
3. Keep GPU writes inside the compiler → bridge path.

### Local models

```powershell
python ai_agent.py "build a walled reservoir" --observe --base-url http://127.0.0.1:11434/v1 --model llama3.2
```

## Related products

- Original playground: `python main.py`
- Autonomous swarm: keys `A` / `D` / `C` inside the window
- Video ingestion: `python video2game.py --input clip.mp4 --output output/worlds/from-video/clip_solid-edges_256.bds` then `python main.py --world output/worlds/from-video/clip_solid-edges_256.bds`
