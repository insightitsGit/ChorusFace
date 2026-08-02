# Avatar scaffolding — lock what we need, free what users like

**Status:** design only (not implemented).  
**Pair with:** [`SideB_VideoCellCollection.md`](SideB_VideoCellCollection.md) ·
[`AvatarAdoption.md`](AvatarAdoption.md) · [`AvatarCapture.md`](AvatarCapture.md)

---

## 1. Idea

Keep one **scaffolding** (calibration script + digest + Side B timeline +
multi-layer ML + NWR path) for every avatar.

**Lock** the contract that makes ticks accurate.  
**Open** cosmetic choices users care about.  
**Accept** user-uploaded avatars that satisfy the lock.

```text
SCAFFOLDING (same for everyone)
  8s calibration beats · face-box registration · TickPackage @ 60 Hz
  LOOK plates · .bds lock · CHORUS push · L00–L11

USER freestyle (on top)
  face / skin tint · eye color · optional makeup / style
  their own uploaded video or stills (same script)

IDENTITY rule
  User's photographed (or uploaded) face stays authority.
  Cosmetics are drives / overlays — not invented replacement RGB for who they are
  unless product explicitly allows a styled plate bank derived from their upload.
```

---

## 2. What we lock (required)

| Locked | Why |
| --- | --- |
| Calibration script beats (REST→…→REST) | Know when smile / hi / angry happen |
| Frontal framing + face box | Same UV as `.bds` |
| Rest identity + Master Lock | NWR authority |
| Side B → FaceCellTimeline @ 60 Hz | Per-tick truth |
| Packet schemas (SpeechClock, LookDrive, FaceMotion…) | Independent ML retrain |
| Display path L00–L11 | Stable realtime composite |

User cannot skip these and still get full-accuracy ticks.

---

## 3. What users may change (cosmetic / preference)

| Open | How (design) |
| --- | --- |
| Skin / face color tint | Shader / material grade on unlocked or graded regions |
| Eye color | Iris tint uniform / small LOOK overlay |
| Makeup / style accents | Optional style plates or grade LUTs |
| Voice / language | L1 speech teachers; same face scaffold |
| Own avatar upload | Same 8s (or capture kit) script with **their** face |

Cosmetics must not destroy mouth plate registration or lock geometry.

---

## 4. User-uploaded avatars

```text
Upload video or stills
  → must follow scaffolding script (or capture kit minimum)
  → digest + Side B collect + train layers for THAT world dir
  → adopt via avatar_profile (same GPU path)
  → optional cosmetic prefs stored beside world
```

| Accept if | Reject / retake if |
| --- | --- |
| Beats present and distinct | REST≈SMILE, no OPEN, no face |
| Single frontal face | multi-face, profile, heavy blur |
| Mouth unlocked + plates writable | cannot seed `.bds` / plates |
| Script labels usable | no audio when SAY_HI/TALK required |

Lab Gemini sample = **scaffold QA only**.  
Production identity = **user upload** (or their recorded script take).

---

## 5. Future UX (design)

1. “Record / upload your 8s calibration”  
2. System locks scaffold + builds world  
3. Optional panel: eye color, skin tone, style  
4. Chat/TTS uses same NWR path  

Retrain: new upload replaces Side B + L3 (etc.); cosmetics can persist as prefs.
