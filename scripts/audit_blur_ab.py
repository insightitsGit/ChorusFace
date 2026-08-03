#!/usr/bin/env python3
"""Full blur audit: Side B teacher vs Side A ingest + timed screenshots.

Requires demo: ``--bridge --bridge-direct-speak --tickfeed-debug``.

Produces:
  output/previews/blur_audit/
    report.json          — A/B comparison + per-frame scores + verdict
    frames/*.png         — screenshots tagged with open/travel/gain
    isolation_* /        — plate_only / field_only / normal bursts
"""
from __future__ import annotations

import io
import json
import time
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
WORLD = ROOT / "output" / "worlds" / "tickfeed"
LOG = ROOT / "output" / "previews" / "tickfeed_side_a.jsonl"
OUT = ROOT / "output" / "previews" / "blur_audit"
BASE = "http://127.0.0.1:8766"
TOKEN = "tickfeed-lab"
SPEECH = "Ah oh oo ee. Open wide now. Say ah ah ah."


def _req(method: str, path: str, body: dict | None = None) -> bytes:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        BASE + path,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=45) as resp:
        return resp.read()


def _status() -> dict:
    return json.loads(_req("GET", "/status").decode("utf-8"))


def _wait_ready() -> None:
    for _ in range(90):
        try:
            if int(_status().get("tick") or 0) > 10:
                return
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.4)
    raise RuntimeError("bridge not ready")


def _wait_idle(timeout: float = 25.0) -> None:
    deadline = time.time() + timeout
    quiet = 0
    while time.time() < deadline:
        st = _status()
        tf = st.get("tickfeed") or {}
        speaking = bool(st.get("speaking")) or int(st.get("pending_visemes") or 0) > 0
        if not speaking and float(tf.get("open") or 0) < 0.1:
            quiet += 1
            if quiet >= 5:
                return
        else:
            quiet = 0
        time.sleep(0.12)


def mouth_roi(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    return arr[int(h * 0.40) : int(h * 0.72), int(w * 0.28) : int(w * 0.72)]


def sharpness(gray: np.ndarray) -> float:
    if gray.size < 16:
        return 0.0
    from numpy.lib.stride_tricks import sliding_window_view

    g = gray.astype(np.float32)
    kern = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    win = sliding_window_view(g, (3, 3))
    return float((win * kern).sum(axis=(-1, -2)).var())


def edge_energy(gray: np.ndarray) -> float:
    g = gray.astype(np.float32)
    gx = np.abs(np.diff(g, axis=1)).mean()
    gy = np.abs(np.diff(g, axis=0)).mean()
    return float(gx + gy)


def soft_veil_score(rgb: np.ndarray) -> float:
    """High when mouth ROI is low-contrast gray wash (classic soft-matte blur)."""
    g = rgb.astype(np.float32)
    if g.ndim == 3:
        lum = 0.299 * g[..., 0] + 0.587 * g[..., 1] + 0.114 * g[..., 2]
    else:
        lum = g
    contrast = float(lum.std())
    # Mid-gray mush: low contrast + mid mean
    mean = float(lum.mean())
    mush = max(0.0, 1.0 - contrast / 28.0) * (1.0 - abs(mean - 110.0) / 110.0)
    return float(max(0.0, mush))


def audit_teacher_b() -> dict:
    npz = WORLD / "face_cell_timeline.npz"
    look_path = WORLD / "face_cell_timeline" / "look_drive.json"
    if not npz.is_file():
        return {"ok": False, "error": "missing timeline"}
    vel = np.load(npz)["velocity"].astype(np.float32)
    look = []
    if look_path.is_file():
        look = json.loads(look_path.read_text(encoding="utf-8")).get("ticks") or []
    open_idx = [
        i
        for i, row in enumerate(look)
        if i < len(vel) and str(row.get("beat") or "") == "OPEN"
    ]
    seps, maxes = [], []
    for t in open_idx:
        p = vel[t]
        maxes.append(float(np.linalg.norm(p, axis=-1).max()))
        h = p.shape[0]
        my = int(h * 0.55)
        m = p[my - 5 : my + 25]
        mid = m.shape[0] // 2
        seps.append(float(m[mid:, :, 1].mean() - m[:mid, :, 1].mean()))
    # Plate asset check
    plates = {}
    for name in ("open.png", "smile.png"):
        p = WORLD / name
        if not p.is_file():
            plates[name] = {"exists": False}
            continue
        im = np.asarray(Image.open(p).convert("RGBA"), dtype=np.float32)
        a = im[..., 3] / 255.0
        plates[name] = {
            "exists": True,
            "alpha_mean": float(a.mean()),
            "alpha_gt_0_2_frac": float((a > 0.2).mean()),
            "alpha_gt_0_5_frac": float((a > 0.5).mean()),
            "soft_matte": float(a.mean()) > 0.08 and float((a > 0.5).mean()) < 0.04,
        }
    return {
        "ok": True,
        "n_open": len(open_idx),
        "field_max_median": float(np.median(maxes)) if maxes else 0.0,
        "sep_median": float(np.median(seps)) if seps else 0.0,
        "rest_max": float(np.linalg.norm(vel[0], axis=-1).max()) if len(vel) else 0.0,
        "plates": plates,
    }


def load_side_a_after(t_mark: float, tick0: int) -> list[dict]:
    if not LOG.is_file():
        return []
    rows = []
    for line in LOG.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("type") != "ingest":
            continue
        if float(row.get("t") or 0) + 1e-6 < t_mark:
            continue
        if int(row.get("master_tick") or 0) < tick0:
            continue
        rows.append(row)
    return rows


def capture_speech(folder: Path, *, seconds: float = 4.0, hz: float = 10.0) -> list[dict]:
    folder.mkdir(parents=True, exist_ok=True)
    frames: list[dict] = []
    n = max(1, int(seconds * hz))
    for i in range(n):
        st = _status()
        tf = st.get("tickfeed") or {}
        png = _req("GET", "/preview")
        img = Image.open(io.BytesIO(png)).convert("RGB")
        arr = np.asarray(img)
        roi = mouth_roi(arr)
        gray = np.asarray(Image.fromarray(roi).convert("L"))
        sharp = sharpness(gray)
        edge = edge_energy(gray)
        veil = soft_veil_score(roi)
        # blur score: high veil + low sharpness
        blur = veil * 0.65 + max(0.0, 1.0 - sharp / 12.0) * 0.35
        name = f"f{i:03d}_o{float(tf.get('open') or 0):.2f}_g{float(tf.get('field_gain_eff') or 0):.2f}.png"
        path = folder / name
        path.write_bytes(png)
        # Crop mouth for inspection
        Image.fromarray(roi).save(folder / f"mouth_{i:03d}.png")
        recent = tf.get("side_a_recent") or []
        last = recent[-1] if recent else {}
        frames.append(
            {
                "i": i,
                "tick": st.get("tick"),
                "path": str(path.relative_to(ROOT)),
                "mouth_path": str((folder / f"mouth_{i:03d}.png").relative_to(ROOT)),
                "open": float(tf.get("open") or 0),
                "plate_open": float(tf.get("plate_open") or 0),
                "viseme": tf.get("viseme"),
                "gain_eff": float(tf.get("field_gain_eff") or 0),
                "speech_pace": float(tf.get("speech_pace") or 0),
                "peak_speed": float(st.get("peak_speed") or 0),
                "muscles": int(st.get("active_muscles") or 0),
                "sharpness": sharp,
                "edge": edge,
                "veil": veil,
                "blur_score": blur,
                "side_a_last": {
                    "kind": last.get("kind"),
                    "field_max": (last.get("field") or {}).get("max"),
                    "travel": (last.get("gain") or {}).get("travel"),
                    "flags": last.get("blur_flags"),
                },
            }
        )
        time.sleep(1.0 / hz)
    return frames


def summarize_a(rows: list[dict]) -> dict:
    flags: Counter[str] = Counter()
    opens, field, travel, gain, sep = [], [], [], [], []
    for r in rows:
        for f in r.get("blur_flags") or []:
            flags[f] += 1
        lab = r.get("labels") or {}
        fld = r.get("field") or {}
        g = r.get("gain") or {}
        o = float(lab.get("plate_open") or 0)
        if o < 0.1 and float(fld.get("max") or 0) < 0.1:
            continue
        opens.append(o)
        field.append(float(fld.get("max") or 0))
        travel.append(float(g.get("travel") or 0))
        gain.append(float(g.get("effective") or 0))
        sep.append(float(fld.get("sep_l_minus_u") or 0))
    return {
        "n": len(rows),
        "n_hot": len(opens),
        "blur_flags": dict(flags),
        "open_max": max(opens) if opens else 0.0,
        "field_max_median": float(np.median(field)) if field else 0.0,
        "sep_median": float(np.median(sep)) if sep else 0.0,
        "travel_median": float(np.median(travel)) if travel else 0.0,
        "gain_median": float(np.median(gain)) if gain else 0.0,
    }


def correlate_blur(frames: list[dict]) -> dict:
    """Which signals track high blur_score?"""
    if len(frames) < 4:
        return {}
    blur = np.array([f["blur_score"] for f in frames], dtype=np.float64)
    signals = {
        "open": np.array([f["open"] for f in frames]),
        "gain_eff": np.array([f["gain_eff"] for f in frames]),
        "peak_speed": np.array([f["peak_speed"] for f in frames]),
        "travel": np.array(
            [float((f.get("side_a_last") or {}).get("travel") or 0) for f in frames]
        ),
        "veil": np.array([f["veil"] for f in frames]),
    }
    corr = {}
    for name, x in signals.items():
        if x.std() < 1e-9 or blur.std() < 1e-9:
            corr[name] = 0.0
        else:
            corr[name] = float(np.corrcoef(blur, x)[0, 1])
    # Mid-open blur: mean blur when 0.15<=open<=0.55 vs open>0.7 or open<0.1
    mid = [f["blur_score"] for f in frames if 0.15 <= f["open"] <= 0.55]
    high = [f["blur_score"] for f in frames if f["open"] >= 0.7]
    low = [f["blur_score"] for f in frames if f["open"] < 0.1]
    return {
        "corr_with_blur": corr,
        "blur_mean_mid_open": float(np.mean(mid)) if mid else None,
        "blur_mean_full_open": float(np.mean(high)) if high else None,
        "blur_mean_closed": float(np.mean(low)) if low else None,
        "worst_frames": sorted(frames, key=lambda f: f["blur_score"], reverse=True)[:8],
        "best_frames": sorted(frames, key=lambda f: f["blur_score"])[:5],
    }


def verdict(teacher: dict, side_a: dict, corr: dict, iso: dict) -> dict:
    reasons: list[str] = []
    layer = "UNKNOWN"

    plates = (teacher.get("plates") or {}).get("open.png") or {}
    if plates.get("soft_matte"):
        reasons.append(
            f"open.png is a wide soft matte "
            f"(alpha_mean={plates.get('alpha_mean'):.3f}, "
            f"alpha>0.5 only {100*float(plates.get('alpha_gt_0_5_frac') or 0):.1f}% of pixels) "
            "— stacks as a washed veil / fake motion blur in the fragment shader"
        )
        layer = "LOOK_PLATE_MATTE"

    if teacher.get("ok") and float(teacher.get("field_max_median") or 0) >= 0.5:
        reasons.append(
            f"Side B teacher FIELD OK (open max≈{teacher['field_max_median']:.2f}, "
            f"sep≈{teacher['sep_median']:.3f})"
        )
    else:
        reasons.append("Side B teacher FIELD weak")
        layer = "FEED_B"

    a_field = float(side_a.get("field_max_median") or 0)
    a_travel = float(side_a.get("travel_median") or 0)
    if a_field >= 0.3:
        reasons.append(
            f"Side A live FIELD OK (field_max_med≈{a_field:.2f}, travel_med≈{a_travel:.3f})"
        )
    else:
        reasons.append("Side A live FIELD weak vs teacher")
        layer = "FEED_A" if layer == "UNKNOWN" else "MIXED"

    flags = side_a.get("blur_flags") or {}
    if flags.get("plate+field_stack") or flags.get("transition_plate+field"):
        reasons.append(f"Side A stack flags still present: {flags}")
        layer = "STACK" if layer in {"UNKNOWN", "LOOK_PLATE_MATTE"} else "MIXED"

    mid = corr.get("blur_mean_mid_open")
    full = corr.get("blur_mean_full_open")
    closed = corr.get("blur_mean_closed")
    if mid is not None and full is not None and mid > full + 0.05:
        reasons.append(
            f"Screenshots blur WORSE at mid-open ({mid:.2f}) than full-open ({full:.2f}) "
            "— classic soft plate crossfade / open.png veil, not missing FIELD"
        )
        if layer == "UNKNOWN":
            layer = "LOOK_TRANSITION"
        elif layer == "LOOK_PLATE_MATTE":
            layer = "LOOK_PLATE_MATTE"

    c = corr.get("corr_with_blur") or {}
    if float(c.get("veil") or 0) > 0.4:
        reasons.append(
            f"blur_score correlates with soft-veil metric (r={c['veil']:.2f})"
        )
    if float(c.get("travel") or 0) > 0.35 and a_travel > 0.08:
        reasons.append(
            f"blur also tracks FIELD travel (r={c.get('travel'):.2f}) — residual warp smear"
        )
        layer = "MIXED" if layer.startswith("LOOK") else "NWR_WARP"

    # Isolation: if plate_only still soft and field_only sharper → LOOK
    p_blur = None
    f_blur = None
    if iso.get("plate_only"):
        p_blur = float(np.mean([x["blur_score"] for x in iso["plate_only"]]))
    if iso.get("field_only"):
        f_blur = float(np.mean([x["blur_score"] for x in iso["field_only"]]))
    if p_blur is not None and f_blur is not None:
        reasons.append(
            f"isolation mean blur: plate_only={p_blur:.2f} field_only={f_blur:.2f}"
        )
        if p_blur > f_blur + 0.08:
            reasons.append(
                "plate_only softer than field_only → LOOK composite/matte is the blur"
            )
            layer = "LOOK_PLATE_MATTE"
        elif f_blur > p_blur + 0.08:
            reasons.append(
                "field_only softer than plate_only → NWR warp path is the blur"
            )
            layer = "NWR_WARP"

    if layer == "UNKNOWN":
        layer = "LOOK_OR_COMPOSITE"
        reasons.append(
            "No FEED break; inspect worst mouth_*.png crops — likely open.png/atlas matte"
        )

    return {
        "primary_layer": layer,
        "reasons": reasons,
        "next_fix": _next_fix(layer),
    }


def _next_fix(layer: str) -> str:
    if layer in {"LOOK_PLATE_MATTE", "LOOK_TRANSITION", "LOOK_OR_COMPOSITE"}:
        return (
            "Harden/crop open.png alpha to oral interior only; under TickFeed "
            "mute open.png entirely when atlas viseme plate is active "
            "(capture_mute=1); keep FIELD muted during plate_open>0.15"
        )
    if layer == "NWR_WARP":
        return "Raise field mute earlier; reduce field_warp_gain during speech"
    if layer.startswith("FEED"):
        return "Rebuild face_cell_timeline / fix live synthesize_velocity"
    if layer == "STACK":
        return "Ensure gain mute after ingest; zero FIELD values when plate owns mouth"
    return "Compare worst vs best mouth crops in blur_audit/"


def run_isolation(mode: str) -> list[dict]:
    _wait_idle()
    _req("POST", "/calibrate", {"mode": mode})
    _req("POST", "/speak", {"text": "Ah ah ah. Open."})
    deadline = time.time() + 6
    while time.time() < deadline:
        if float((_status().get("tickfeed") or {}).get("open") or 0) > 0.25:
            break
        time.sleep(0.05)
    frames = capture_speech(OUT / f"isolation_{mode}", seconds=2.2, hz=8.0)
    _wait_idle()
    return frames


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== Side B teacher ===")
    teacher = audit_teacher_b()
    print(json.dumps({k: teacher[k] for k in teacher if k != "plates"}, indent=2))
    print("plates:", json.dumps(teacher.get("plates"), indent=2))

    print("\n=== Live bridge ===")
    _wait_ready()
    _req("POST", "/calibrate", {"mode": "normal"})
    st0 = _status()
    tick0 = int(st0.get("tick") or 0)
    t_mark = 0.0
    if LOG.is_file():
        for line in LOG.read_text(encoding="utf-8").splitlines()[::-1]:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("type") == "ingest":
                t_mark = float(row.get("t") or 0)
                break

    print("SPEAK:", SPEECH)
    _wait_idle()
    _req("POST", "/speak", {"text": SPEECH})
    deadline = time.time() + 8
    while time.time() < deadline:
        if float((_status().get("tickfeed") or {}).get("open") or 0) > 0.25:
            break
        time.sleep(0.05)

    print("Capturing screenshots @10Hz …")
    frames = capture_speech(OUT / "frames", seconds=4.0, hz=10.0)
    _wait_idle()
    side_a_rows = load_side_a_after(t_mark, tick0)
    side_a = summarize_a(side_a_rows)
    corr = correlate_blur(frames)

    print("\n=== Isolation modes ===")
    iso = {}
    for mode in ("plate_only", "field_only", "normal"):
        print(f"  {mode}…")
        iso[mode] = run_isolation(mode)
    _req("POST", "/calibrate", {"mode": "normal"})

    v = verdict(teacher, side_a, corr, iso)
    report = {
        "speech": SPEECH,
        "teacher_side_b": teacher,
        "live_side_a": side_a,
        "screenshot_correlation": {
            "corr_with_blur": corr.get("corr_with_blur"),
            "blur_mean_mid_open": corr.get("blur_mean_mid_open"),
            "blur_mean_full_open": corr.get("blur_mean_full_open"),
            "blur_mean_closed": corr.get("blur_mean_closed"),
            "worst": [
                {
                    "blur": w["blur_score"],
                    "open": w["open"],
                    "gain": w["gain_eff"],
                    "veil": w["veil"],
                    "sharp": w["sharpness"],
                    "path": w["mouth_path"],
                    "viseme": w["viseme"],
                }
                for w in (corr.get("worst_frames") or [])
            ],
            "best": [
                {
                    "blur": w["blur_score"],
                    "open": w["open"],
                    "path": w["mouth_path"],
                }
                for w in (corr.get("best_frames") or [])
            ],
        },
        "isolation_blur_means": {
            m: float(np.mean([f["blur_score"] for f in fs])) if fs else None
            for m, fs in iso.items()
        },
        "verdict": v,
    }
    out = OUT / "report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n=== VERDICT ===")
    print(json.dumps(v, indent=2))
    print(f"\nWrote {out}")
    print(f"Frames: {OUT / 'frames'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
