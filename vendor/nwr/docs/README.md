# Documentation index

- [`FinalDesign.md`](FinalDesign.md) — binding runtime implementation specification.
- [`AI_API.md`](AI_API.md) — how assistants inspect and command the world.
- [`AvatarChat.md`](AvatarChat.md) — biomechanical face: muscles, emotion, jaw, intent, debug views. Reference implementation; the productised descendant is the [AIFace](https://github.com/insightitsGit/AIFace) child repository.
- [`FlowRunner.md`](FlowRunner.md) — playable rising-tide game over a `.bds` world.
- [`Video2GameDesign.md`](Video2GameDesign.md) — optional video-ingestion subsystem, including measured size and semantic limits.
- [`UseCases.md`](UseCases.md) — what the substrate supports today, and the gaps in adjacent use cases.
- [`../output/README.md`](../output/README.md) — where generated artifacts land and how they are named.
- [`DesignGemini.md`](DesignGemini.md) — archived original prototype proposal.
- [`DesignChatGpt.md`](DesignChatGpt.md) — archived production-runtime evolution notes.

When documents disagree, `FinalDesign.md` is authoritative for the current implementation.

Neural World Runtime is the heart of the product; child products such as
[AIFace](https://github.com/insightitsGit/AIFace) inherit the substrate
contract defined here. A child may run ahead of this repository in its own
domain, but it does not get to redefine the cell schema, the authority
ordering, or the Master Lock — those are settled in `FinalDesign.md`.
