# Avatar observations — the data gaps are filled **from**

The behavior ML does **not** invent a smile. Gap fill needs measured avatar
truth first. That truth is `avatar_observations.json`.

Related: [`AvatarBehavior.md`](AvatarBehavior.md) · [`NWRDataDesign.md`](NWRDataDesign.md)

## The hole you saw

| What we had | What was missing |
| --- | --- |
| `smile.png` pixels | A **smile vector** for gap fill |
| `expression_catalog` smile_width | Mapping to **GPU uniforms** |
| Behavior ML (audio → controls) | Anchor to **this avatar's** smile/open looks |
| Cell clusters on `.bds` | Stored with the look package |

Demo gaps looked empty because ML was interpolating landmark proxies without
the measured look anchors (how smile/open show on the GPU path).

## What we store now

```text
avatar_observations.json / .npz
  looks:
    rest / smile / open / surprise
      landmarks     ← from expression_catalog (avatar capture)
      gpu           ← uniforms avatar.frag actually reads
      plate         ← mouth-ROI pixel delta vs rest
      controls      ← group control vector
      delta_from_rest ← smile_vector / open_vector
    talk[]          ← talk_series landmark samples over time
  cells: mouth_unlocked counts + group membership from .bds
  smile_vector / open_vector
```

### Smile GPU vector (real shader contract)

| Uniform | Smile look | Open look |
| --- | --- | --- |
| `avatar_mouth_pose.w` | ~1.0 (drives `smile.png`) | 0 |
| `avatar_mouth_pose.y` / `plate_blend.y` | 0 | ~1.0 (drives `open.png` / atlas) |
| `avatar_jaw.z` | 0 | high |
| `avatar_smile_plate` | `smile.png` | — |
| `avatar_open_plate` | — | `open.png` |

Smile is a **look plate + drive**, not a different `.bds` field state.
Cell motion for smile = group flow toward measured `corner_dx` / width delta.

## Authority

```text
observed smile/open vectors  →  measured track  →  ML fill  →  table
```

ML only fills **between** observations. Retrain writes observations first:

```powershell
python scripts/retrain_behavior.py
```
