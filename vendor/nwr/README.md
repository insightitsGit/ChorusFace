# Vendored NWR core (AminIntheLoop)

Copied from `C:/code/NWR` at the revision in `NWR_REVISION.txt`.

## Included

| Area | Files |
| --- | --- |
| World format | `bds_format.py`, `bds_codec.py`, `bds_chunks.py`, `bds_history.py` |
| GPU / shaders | `shader_library.py`, `shaders/` |
| AI command path | `ai_commands.py`, `ai_command_compiler.py`, `ai_bridge.py`, `ai_world.py`, `ai_handoff.py`, `ai_agent.py` |
| Authority / net | `net_guard.py`, `determinism.py`, `paths.py` |
| Material MLP | `material_network.py` |
| Sandbox entry | `main.py` |
| Example tool | `tools/make_tennis_racket.py` |
| Specs | `docs/FinalDesign.md`, `AI_API.md`, `UseCases.md` |

## Not included (on purpose)

Avatar chat drivers, rigid face parts, video2game, swarm/game demos — add later
only when Amin asks, still under NWR rules.
