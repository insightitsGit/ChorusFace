#!/usr/bin/env python3
"""Isolate whether mouth blur is FEED data or NWR apply/render.

Requires demo with ``--bridge --bridge-direct-speak --tickfeed-debug``.

Phases
------
1. Offline teacher timeline audit (measured Side-B collect quality).
2. Live isolation speaks in three modes via ``POST /calibrate``:
   - ``plate_only``  — FIELD gain 0 (LOOK plates alone)
   - ``field_only``  — plates forced closed (NWR FIELD warp alone)
   - ``normal``      — production stack
3. Compare Side A package stats vs GPU peak + preview sharpness.

Verdicts: FEED | NWR | PLATE | STACK | MIXED
"""
from __future__ import annotations

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
OUT = ROOT / "output" / "previews" / "feed_vs_nwr"
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
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def _status() -> dict:
    return json.loads(_req("GET", "/status").decode("utf-8"))


def _wait_ready() -> None:
    for _ in range(80):
        try:
            if int(_status().get("tick") or 0) > 10:
                return
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.4)
    raise RuntimeError("bridge not ready — start scripts/run_tickfeed_demo.py")


def _wait_idle(timeout: float = 20.0) -> None:
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


def audit_teacher() -> dict:
    """Measured timeline quality — independent of live GPU/NWR."""
    npz = WORLD / "face_cell_timeline.npz"
    look_path = WORLD / "face_cell_timeline" / "look_drive.json"
    if not npz.is_file():
        return {"ok": False, "error": f"missing {npz}"}
    vel = np.load(npz)["velocity"].astype(np.float32)
    look = []
    if look_path.is_file():
        look = json.loads(look_path.read_text(encoding="utf-8")).get("ticks") or []
    open_idx = [
        i
        for i, row in enumerate(look)
        if i < len(vel) and str(row.get("beat") or "") == "OPEN"
    ]
    if not open_idx:
        # Fallback: top-quartile magnitude frames
        mags = np.linalg.norm(vel.reshape(len(vel), -1, 2), axis=-1).max(axis=1)
        thr = float(np.percentile(mags, 75))
        open_idx = [i for i, m in enumerate(mags) if m >= thr]

    seps: list[float] = []
    maxes: list[float] = []
    rigid: list[float] = []
    for t in open_idx:
        p = vel[t]
        mag = float(np.linalg.norm(p, axis=-1).max())
        maxes.append(mag)
        mean = p.reshape(-1, 2).mean(0)
        e = float(np.square(p).sum())
        r = float(np.square(p - mean).sum())
        rigid.append(1.0 - r / max(e, 1e-9))
        h = p.shape[0]
        my = int(h * 0.55)
        m = p[my - 5 : my + 25]
        mid = m.shape[0] // 2
        um = float(m[:mid, :, 1].mean())
        lm = float(m[mid:, :, 1].mean())
        seps.append(lm - um)

    report = {
        "ok": True,
        "n_ticks": int(len(vel)),
        "n_open": int(len(open_idx)),
        "open_field_max_median": float(np.median(maxes)) if maxes else 0.0,
        "open_field_max_p95": float(np.percentile(maxes, 95)) if maxes else 0.0,
        "open_sep_median": float(np.median(seps)) if seps else 0.0,
        "open_sep_p95": float(np.percentile(seps, 95)) if seps else 0.0,
        "open_rigid_median": float(np.median(rigid)) if rigid else 0.0,
        "rest_field_max": float(np.linalg.norm(vel[0], axis=-1).max())
        if len(vel)
        else 0.0,
    }
    flags: list[str] = []
    if report["open_field_max_median"] < 0.15:
        flags.append("teacher_field_weak")
    if report["open_sep_median"] < 0.04:
        flags.append("teacher_lip_sep_weak")
    if report["open_rigid_median"] > 0.85:
        flags.append("teacher_mostly_rigid")
    if report["rest_field_max"] > 0.08:
        flags.append("teacher_rest_not_zero")
    report["flags"] = flags
    report["verdict"] = (
        "FEED_TIMELINE"
        if flags
        else "TEACHER_OK"
    )
    return report


def _mouth_roi(png: bytes) -> np.ndarray:
    img = Image.open(__import__("io").BytesIO(png)).convert("L")
    arr = np.asarray(img, dtype=np.float32)
    h, w = arr.shape
    y0, y1 = int(h * 0.42), int(h * 0.72)
    x0, x1 = int(w * 0.30), int(w * 0.70)
    return arr[y0:y1, x0:x1]


def _mouth_sharpness(png: bytes) -> float:
    """Laplacian variance on lower-center ROI (higher = sharper)."""
    roi = _mouth_roi(png)
    if roi.size < 16:
        return 0.0
    kern = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    from numpy.lib.stride_tricks import sliding_window_view

    win = sliding_window_view(roi, (3, 3))
    lap = (win * kern).sum(axis=(-1, -2))
    return float(lap.var())


def _motion_energy(rois: list[np.ndarray]) -> float:
    """Mean absolute frame-to-frame change in mouth ROI."""
    if len(rois) < 2:
        return 0.0
    diffs = [float(np.mean(np.abs(rois[i] - rois[i - 1]))) for i in range(1, len(rois))]
    return float(np.median(diffs)) if diffs else 0.0


def _load_ingest_after(t_mark: float, tick0: int) -> list[dict]:
    if not LOG.is_file():
        return []
    rows: list[dict] = []
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


def _summarize_feed(rows: list[dict]) -> dict:
    opens = []
    field_max = []
    seps = []
    travel = []
    gain = []
    gpu = []
    flags: Counter[str] = Counter()
    kinds: Counter[str] = Counter()
    for r in rows:
        kinds[str(r.get("kind"))] += 1
        for f in r.get("blur_flags") or []:
            flags[f] += 1
        lab = r.get("labels") or {}
        field = r.get("field") or {}
        g = r.get("gain") or {}
        o = float(lab.get("plate_open") or 0)
        if o < 0.25 and float(field.get("max") or 0) < 0.15:
            continue
        opens.append(o)
        field_max.append(float(field.get("max") or 0))
        seps.append(float(field.get("sep_l_minus_u") or 0))
        travel.append(float(g.get("travel") or 0))
        gain.append(float(g.get("effective") or 0))
        gpu.append(float((r.get("gpu") or {}).get("peak") or 0))
    return {
        "n_hot": len(opens),
        "kinds": dict(kinds),
        "blur_flags": dict(flags),
        "open_max": max(opens) if opens else 0.0,
        "field_max_median": float(np.median(field_max)) if field_max else 0.0,
        "sep_median": float(np.median(seps)) if seps else 0.0,
        "gain_eff_median": float(np.median(gain)) if gain else 0.0,
        "travel_median": float(np.median(travel)) if travel else 0.0,
        "gpu_peak_median": float(np.median(gpu)) if gpu else 0.0,
        "gpu_over_field": (
            float(np.median(gpu) / max(np.median(field_max), 1e-6))
            if field_max and gpu
            else 0.0
        ),
    }


def run_mode(mode: str) -> dict:
    folder = OUT / mode
    folder.mkdir(parents=True, exist_ok=True)
    _wait_idle()
    cal = json.loads(_req("POST", "/calibrate", {"mode": mode}).decode("utf-8"))
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

    _req("POST", "/speak", {"text": SPEECH})
    # Wait for open activity
    deadline = time.time() + 8.0
    while time.time() < deadline:
        tf = _status().get("tickfeed") or {}
        if float(tf.get("open") or 0) > 0.25:
            break
        time.sleep(0.05)

    sharp: list[float] = []
    rois: list[np.ndarray] = []
    for i in range(12):
        png = _req("GET", "/preview")
        path = folder / f"frame_{i:03d}.png"
        path.write_bytes(png)
        sharp.append(_mouth_sharpness(png))
        rois.append(_mouth_roi(png))
        time.sleep(0.16)

    _wait_idle(timeout=12.0)
    rows = _load_ingest_after(t_mark, tick0)
    feed = _summarize_feed(rows)
    return {
        "mode": mode,
        "calibrate": cal,
        "sharpness_median": float(np.median(sharp)) if sharp else 0.0,
        "sharpness_max": float(max(sharp)) if sharp else 0.0,
        "motion_energy": _motion_energy(rois),
        "feed": feed,
        "preview_dir": str(folder.relative_to(ROOT)),
    }


def verdict(teacher: dict, modes: dict[str, dict]) -> dict:
    """Decide FEED vs NWR vs PLATE from isolation evidence."""
    reasons: list[str] = []
    layer = "UNKNOWN"

    t_flags = list(teacher.get("flags") or [])
    if t_flags:
        reasons.append(f"Teacher timeline flags: {t_flags}")

    normal = modes.get("normal", {}).get("feed") or {}
    plate = modes.get("plate_only", {})
    field = modes.get("field_only", {})
    n_feed = modes.get("normal", {})

    live_field = float(normal.get("field_max_median") or 0)
    live_sep = float(normal.get("sep_median") or 0)
    teacher_sep = float(teacher.get("open_sep_median") or 0)
    teacher_max = float(teacher.get("open_field_max_median") or 0)

    if live_field < 0.12 and float(normal.get("open_max") or 0) >= 0.5:
        reasons.append(
            "Live Side A FIELD weak while LOOK open high — FEED/synth issue"
        )
        layer = "FEED"
    elif live_sep < 0.03 and teacher_sep >= 0.06:
        reasons.append(
            "Live lip separation much weaker than teacher — live FEED synth"
        )
        layer = "FEED"
    elif teacher_max < 0.15 or "teacher_field_weak" in t_flags:
        reasons.append("Measured teacher FIELD itself is weak — FEED collect")
        layer = "FEED"

    plate_sharp = float(plate.get("sharpness_median") or 0)
    field_sharp = float(field.get("sharpness_median") or 0)
    normal_sharp = float(n_feed.get("sharpness_median") or 0)
    plate_motion = float(plate.get("motion_energy") or 0)
    field_motion = float(field.get("motion_energy") or 0)
    normal_motion = float(n_feed.get("motion_energy") or 0)
    field_feed = field.get("feed") or {}
    plate_feed = plate.get("feed") or {}

    plate_gain = plate_feed.get("gain_eff_median")
    if plate_gain is None or float(plate_gain) > 0.05:
        reasons.append(
            f"plate_only gain_eff_median={plate_gain!r} (want ≈0) — calibrate path broken"
        )
        layer = "MIXED"

    # NWR apply: in field_only, package field should be hot and gain full;
    # GPU peak should track package field_max (laggy telemetry OK if ratio >0.3).
    if layer not in {"FEED"}:
        f_gain = float(field_feed.get("gain_eff_median") or 0)
        f_max = float(field_feed.get("field_max_median") or 0)
        f_gpu_ratio = float(field_feed.get("gpu_over_field") or 0)
        if f_max >= 0.25 and f_gain >= 0.5 and f_gpu_ratio < 0.15:
            reasons.append(
                "field_only: package FIELD hot but GPU peak << package — NWR ingest/apply"
            )
            layer = "NWR"
        elif f_max >= 0.25 and f_gpu_ratio >= 0.3:
            reasons.append(
                f"field_only: GPU tracks package (gpu/field={f_gpu_ratio:.2f}) — NWR ingest OK"
            )
            if layer == "UNKNOWN":
                layer = "NWR_APPLY_OK"
        if field_motion < 0.4 and f_max >= 0.25 and f_gain >= 0.5:
            reasons.append(
                "field_only: hot FIELD+gain but little preview mouth motion — "
                "NWR warp/gain not visible in pixels"
            )
            layer = "NWR"
        elif field_motion >= 1.2 * max(plate_motion, 0.2) and f_gain >= 0.5:
            reasons.append(
                "field_only moves mouth more than plate_only — FIELD path reaches pixels"
            )

    if layer in {"UNKNOWN", "MIXED", "NWR_APPLY_OK"}:
        if plate_sharp < field_sharp * 0.7 and field_sharp > 0:
            reasons.append(
                "plate_only softer than field_only — LOOK open.png/atlas composite"
            )
            layer = "PLATE" if layer in {"UNKNOWN", "NWR_APPLY_OK"} else "MIXED"
        stack_flags = int((normal.get("blur_flags") or {}).get("plate+field_stack", 0))
        if stack_flags > 0:
            reasons.append(f"normal mode still has plate+field_stack x{stack_flags}")
            layer = "STACK"
        elif (
            normal_sharp + 1e-6 < min(plate_sharp, field_sharp) * 0.75
            and min(plate_sharp, field_sharp) > 0
        ):
            reasons.append(
                "normal softer than both isolations — residual plate+FIELD stack"
            )
            layer = "STACK"

    if layer in {"UNKNOWN", "NWR_APPLY_OK"}:
        # Live FEED strong + NWR apply OK → remaining soft look is plate/shader or muted travel.
        if live_field >= 0.3 and teacher_max >= 0.5:
            if float(normal.get("travel_median") or 0) < 0.05:
                reasons.append(
                    "FEED strong, NWR ingest tracks package, travel muted under open "
                    "plate — residual blur is LOOK plate composite / soft warp, not missing feed"
                )
                layer = "PLATE"
            else:
                reasons.append(
                    "FEED + NWR apply look healthy; inspect preview folders for soft warp"
                )
                layer = "NWR_OR_PLATE"
        elif not reasons:
            reasons.append(
                "No strong break signal; compare preview folders visually "
                f"(plate_only sharp={plate_sharp:.1f} motion={plate_motion:.2f}, "
                f"field_only sharp={field_sharp:.1f} motion={field_motion:.2f}, "
                f"normal sharp={normal_sharp:.1f} motion={normal_motion:.2f})"
            )

    return {
        "layer": layer,
        "reasons": reasons,
        "scores": {
            "teacher_sep": teacher_sep,
            "teacher_field_max": teacher_max,
            "live_sep": live_sep,
            "live_field_max": live_field,
            "plate_only_sharpness": plate_sharp,
            "field_only_sharpness": field_sharp,
            "normal_sharpness": normal_sharp,
            "plate_only_motion": plate_motion,
            "field_only_motion": field_motion,
            "normal_motion": normal_motion,
        },
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== 1) Teacher timeline (FEED collect) ===")
    teacher = audit_teacher()
    print(json.dumps(teacher, indent=2))

    print("\n=== 2) Live isolation (needs demo bridge) ===")
    _wait_ready()
    modes: dict[str, dict] = {}
    for mode in ("plate_only", "field_only", "normal"):
        print(f"\n-- mode {mode} --")
        result = run_mode(mode)
        modes[mode] = result
        print(
            f"  sharp_med={result['sharpness_median']:.1f} "
            f"motion={result['motion_energy']:.2f} "
            f"field_max_med={result['feed'].get('field_max_median', 0):.3f} "
            f"sep_med={result['feed'].get('sep_median', 0):.3f} "
            f"gain_med={result['feed'].get('gain_eff_median', 0):.3f} "
            f"gpu/field={result['feed'].get('gpu_over_field', 0):.2f}"
        )

    # Restore production mode
    _req("POST", "/calibrate", {"mode": "normal"})

    print("\n=== 3) Verdict ===")
    v = verdict(teacher, modes)
    report = {
        "speech": SPEECH,
        "teacher": teacher,
        "modes": {
            k: {
                "sharpness_median": m["sharpness_median"],
                "sharpness_max": m["sharpness_max"],
                "motion_energy": m["motion_energy"],
                "feed": m["feed"],
                "preview_dir": m["preview_dir"],
            }
            for k, m in modes.items()
        },
        "verdict": v,
    }
    out = OUT / "report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(v, indent=2))
    print(f"\nWrote {out}")
    print(f"Previews: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
