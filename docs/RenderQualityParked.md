# Render quality — parked vs implement-together

## Two parked tracks (do not re-land)

| # | Parked | Why |
| --- | --- | --- |
| **P-A** | Kill `open.png` / lower `rest_mix` / occlusion teeth-tongue masks / muscle **jaw residual** / `mouth_muscles` warp under TickFeed | Bundled fidelity pass — visual mess |
| **P-B** | Blink BJ2 (live `lid_amt`) + lash-matte L09 expand | Blink look worse; parked by operator |

## Implement-together (this wave) — not the parked set

| Item | Scope |
| --- | --- |
| **P3** Temporal phases | `MouthMotionState` for transition / FIELD gate only — **keeps** `open.png` dual-owner |
| **P12** Fuller fidelity HUD | phase, provenance, plate, jaw, occlusion stubs (no fake teeth) |
| **RF6 / P9 prep** Atlas open pick + vowel matte | Beat time-hints + thicker oral α — still dual-owner with `open.png` |

Explicitly **out** of this wave: P-A and P-B above.

## Status

Together-wave landed in tree (runtime + plate rebuild required for RF6).
Park count remains **two**: P-A and P-B.
