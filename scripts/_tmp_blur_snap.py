"""Speak + capture mouth crops; score sharpness vs open."""
from __future__ import annotations

import io
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

BASE = "http://127.0.0.1:8766"
TOKEN = "tickfeed-lab"
OUT = Path("output/previews/blur_still")
OUT.mkdir(parents=True, exist_ok=True)


def req(method: str, path: str, body: dict | None = None) -> bytes:
    data = None if body is None else json.dumps(body).encode()
    r = urllib.request.Request(
        BASE + path,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(r, timeout=30) as resp:
        return resp.read()


def mouth_roi(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    return arr[int(h * 0.42) : int(h * 0.72), int(w * 0.30) : int(w * 0.70)]


def sharp(gray: np.ndarray) -> float:
    g = gray.astype(np.float32)
    return float(np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean())


for _ in range(40):
    st = json.loads(req("GET", "/status").decode())
    if int(st.get("tick") or 0) > 10:
        break
    time.sleep(0.3)

req("POST", "/speak", {"text": "Ah oh oo ee. Open wide now. Say ah ah ah."})
deadline = time.time() + 8
while time.time() < deadline:
    if float((json.loads(req("GET", "/status").decode()).get("tickfeed") or {}).get("open") or 0) > 0.4:
        break
    time.sleep(0.05)

rows = []
for i in range(12):
    st = json.loads(req("GET", "/status").decode())
    tf = st.get("tickfeed") or {}
    png = req("GET", "/preview")
    img = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
    roi = mouth_roi(img)
    gray = np.asarray(Image.fromarray(roi).convert("L"))
    path = OUT / f"m{i:02d}_o{float(tf.get('open') or 0):.2f}_g{float(tf.get('field_gain_eff') or 0):.2f}.png"
    Image.fromarray(roi).save(path)
    (OUT / f"full_{i:02d}.png").write_bytes(png)
    rows.append(
        {
            "i": i,
            "open": float(tf.get("open") or 0),
            "gain": float(tf.get("field_gain_eff") or 0),
            "sharp": sharp(gray),
            "path": str(path),
        }
    )
    time.sleep(0.12)

# Also score source + open plate mouth ROI for baseline sharpness
for name in ("source_face.png", "open.png"):
    p = Path("output/worlds/tickfeed") / name
    if not p.is_file():
        continue
    im = np.asarray(Image.open(p).convert("RGB"))
    roi = mouth_roi(im)
    Image.fromarray(roi).save(OUT / f"asset_{name}")
    rows.append(
        {
            "i": name,
            "open": None,
            "gain": None,
            "sharp": sharp(np.asarray(Image.fromarray(roi).convert("L"))),
            "path": str(OUT / f"asset_{name}"),
        }
    )

(OUT / "scores.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(json.dumps(rows, indent=2))
