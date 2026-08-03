"""Quick plate alpha + beat assignment audit."""
from __future__ import annotations

import json
from pathlib import Path

import cv2

w = Path("output/worlds/tickfeed")
rep = json.loads((w / "plate_rebuild_report.json").read_text(encoding="utf-8"))
print("open.png @", rep.get("open_time"), "s")
print("--- plate alpha coverage ---")
names = ["open.png", "smile.png"] + [f"plates/plate_{i:02d}.png" for i in range(10)]
for name in names:
    p = w / name
    if not p.exists():
        print(name, "MISSING")
        continue
    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    a = im[..., 3].astype(float) / 255
    print(
        f"{name:22} a>0.3={(a > 0.3).mean():.4f} "
        f"a>0.5={(a > 0.5).mean():.4f} mean={a.mean():.4f}"
    )
print("--- beat assignment ---")
for b in rep["plate_beats"]:
    print(
        f"{b['viseme']:6} t={b['t']:.2f} "
        f"open={b['openness']:.3f} beat={b['beat']}"
    )
