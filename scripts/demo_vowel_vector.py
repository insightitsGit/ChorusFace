"""VowelDesign visual demo — compose utterances and open an HTML lip-silhouette player.

Usage:
  python scripts/demo_vowel_vector.py
  python scripts/demo_vowel_vector.py --no-browser
"""

from __future__ import annotations

import argparse
import json
import webbrowser
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = ROOT / "output" / "worlds" / "tickfeed" / "vowel"
DEFAULT_OUT = ROOT / "output" / "teacher" / "vowel_vector_demo.html"

DEMOS = [
    {
        "id": "vowels",
        "label": "HAPPY — EE / OU / AA contrast",
        "payload": {
            "utterance_id": "demo_vowels",
            "text": "see you ah",
            "emotion_track": [{"emotion": "HAPPY", "start_s": 0.0, "end_s": 2.0}],
            "spans": [
                {"tag": "EE", "start_s": 0.05, "end_s": 0.45},
                {"tag": "OU", "start_s": 0.55, "end_s": 0.95},
                {"tag": "AA", "start_s": 1.10, "end_s": 1.60},
            ],
        },
    },
    {
        "id": "angry_vowels",
        "label": "ANGRY — EE / OU / AA (very angry take)",
        "payload": {
            "utterance_id": "demo_angry_vowels",
            "text": "see you ah",
            "emotion_track": [{"emotion": "ANGRY", "start_s": 0.0, "end_s": 2.0}],
            "spans": [
                {"tag": "EE", "start_s": 0.05, "end_s": 0.45},
                {"tag": "OU", "start_s": 0.55, "end_s": 0.95},
                {"tag": "AA", "start_s": 1.10, "end_s": 1.60},
            ],
        },
    },
    {
        "id": "sad",
        "label": "SAD — I miss you so much",
        "payload": {
            "utterance_id": "demo_sad",
            "text": "I miss you so much",
            "emotion_track": [{"emotion": "SAD", "start_s": 0.0, "end_s": 2.8}],
        },
    },
    {
        "id": "happy",
        "label": "HAPPY — See you tomorrow",
        "payload": {
            "utterance_id": "demo_happy",
            "text": "See you tomorrow",
            "emotion_track": [{"emotion": "HAPPY", "start_s": 0.0, "end_s": 2.5}],
        },
    },
    {
        "id": "thinking",
        "label": "THINKING — I can help with that",
        "payload": {
            "utterance_id": "demo_thinking",
            "text": "I can help with that",
            "emotion_track": [{"emotion": "THINKING", "start_s": 0.0, "end_s": 2.8}],
        },
    },
    {
        "id": "surprised",
        "label": "SURPRISED — What great news",
        "payload": {
            "utterance_id": "demo_surprised",
            "text": "What great news",
            "emotion_track": [{"emotion": "SURPRISED", "start_s": 0.0, "end_s": 2.2}],
        },
    },
    {
        "id": "neutral_blinks",
        "label": "NEUTRAL — state 0 + blinks",
        "payload": {
            "utterance_id": "demo_neutral_blinks",
            "text": "rest rest rest rest rest",
            "blinks": True,
            "blink_interval_s": 2.0,
            "blink_seed": 1,
            "emotion_track": [{"emotion": "NEUTRAL", "start_s": 0.0, "end_s": 5.0}],
            "spans": [
                {"tag": "AX", "start_s": 0.3, "end_s": 0.55},
                {"tag": "AX", "start_s": 1.3, "end_s": 1.55},
                {"tag": "AX", "start_s": 2.3, "end_s": 2.55},
                {"tag": "AX", "start_s": 3.3, "end_s": 3.55},
                {"tag": "AX", "start_s": 4.3, "end_s": 4.55},
            ],
        },
    },
    {
        "id": "blinks",
        "label": "HAPPY — blinks (~2s interval)",
        "payload": {
            "utterance_id": "demo_blinks",
            "text": "Hello how are you today please and thank you",
            "blinks": True,
            "blink_interval_s": 2.0,
            "blink_seed": 2,
            "emotion_track": [{"emotion": "HAPPY", "start_s": 0.0, "end_s": 5.5}],
        },
    },
]


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>VowelDesign Vector Demo</title>
<style>
  :root {
    --bg0: #1a1410;
    --bg1: #2a211c;
    --ink: #f3e6d8;
    --muted: #b9a894;
    --accent: #e07a3d;
    --line: #6e5848;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    min-height: 100vh;
    font-family: "Segoe UI", Georgia, serif;
    color: var(--ink);
    background:
      radial-gradient(1200px 600px at 20% -10%, #4a3428 0%, transparent 55%),
      radial-gradient(900px 500px at 100% 20%, #3a2a22 0%, transparent 50%),
      linear-gradient(180deg, var(--bg0), #120e0c 80%);
  }
  main {
    max-width: 980px;
    margin: 0 auto;
    padding: 28px 20px 48px;
  }
  h1 {
    font-weight: 600;
    letter-spacing: 0.02em;
    font-size: clamp(1.6rem, 3vw, 2.2rem);
    margin: 0 0 6px;
  }
  .sub { color: var(--muted); margin: 0 0 22px; max-width: 40rem; line-height: 1.45; }
  .row { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 16px; }
  button {
    appearance: none;
    border: 1px solid var(--line);
    background: rgba(255,255,255,0.04);
    color: var(--ink);
    border-radius: 999px;
    padding: 8px 14px;
    cursor: pointer;
    font: inherit;
  }
  button:hover { border-color: var(--accent); color: #fff; }
  button.active { background: var(--accent); border-color: var(--accent); color: #1a1008; }
  .stage {
    display: grid;
    grid-template-columns: 1.1fr 0.9fr;
    gap: 18px;
  }
  @media (max-width: 820px) { .stage { grid-template-columns: 1fr; } }
  .panel {
    background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 14px;
  }
  canvas { width: 100%; height: auto; display: block; border-radius: 12px; background: #120d0a; }
  .meta { display: flex; justify-content: space-between; color: var(--muted); font-size: 0.92rem; margin-top: 10px; }
  .bars { display: grid; gap: 8px; }
  .bar-row { display: grid; grid-template-columns: 110px 1fr 42px; gap: 8px; align-items: center; font-size: 0.85rem; }
  .track { height: 8px; background: rgba(255,255,255,0.08); border-radius: 99px; overflow: hidden; }
  .fill { height: 100%; background: linear-gradient(90deg, #c46b3a, #f0b27a); width: 0%; }
  .tags { margin-top: 12px; color: var(--muted); min-height: 1.4em; }
  .blink-badge {
    display: inline-block; margin-left: 8px; padding: 2px 8px;
    border-radius: 999px; background: rgba(224,122,61,0.25);
    color: #f0c3a0; font-size: 0.8rem;
  }
  .playbar { display: flex; gap: 10px; align-items: center; margin-top: 14px; }
  input[type=range] { width: 100%; }
</style>
</head>
<body>
<main>
  <h1>VowelDesign vector demo</h1>
  <p class="sub">Live 9D group controls from Model A/B → PulseChunk, drawn as lip silhouettes.
  Built from your HAPPY / SURPRISED / THINKING teacher clips.</p>
  <div class="row" id="clips"></div>
  <div class="stage">
    <div class="panel">
      <canvas id="face" width="640" height="480"></canvas>
      <div class="meta"><span id="label">—</span><span id="clock">0.00s</span></div>
      <div class="playbar">
        <button id="play">Play</button>
        <button id="pause">Pause</button>
        <input id="scrub" type="range" min="0" max="100" value="0"/>
      </div>
      <div class="tags" id="tags"></div>
    </div>
    <div class="panel">
      <div class="bars" id="bars"></div>
    </div>
  </div>
</main>
<script>
const DATA = __DATA__;
const CHANNELS = [
  "eye_aperture","eye_gaze","brow_raise","brow_knit",
  "mouth_gap","lip_spread","lip_round","teeth","jaw_drop"
];

const clipsEl = document.getElementById('clips');
const canvas = document.getElementById('face');
const ctx = canvas.getContext('2d');
const barsEl = document.getElementById('bars');
const labelEl = document.getElementById('label');
const clockEl = document.getElementById('clock');
const tagsEl = document.getElementById('tags');
const scrub = document.getElementById('scrub');

let current = DATA.demos[0];
let tick = 0;
let playing = true;
let raf = 0;
let last = 0;

CHANNELS.forEach((name, i) => {
  const row = document.createElement('div');
  row.className = 'bar-row';
  row.innerHTML = `<span>${name}</span><div class="track"><div class="fill" id="f${i}"></div></div><span id="v${i}">0</span>`;
  barsEl.appendChild(row);
});

DATA.demos.forEach((d, idx) => {
  const b = document.createElement('button');
  b.textContent = d.label;
  b.onclick = () => select(idx);
  if (idx === 0) b.classList.add('active');
  clipsEl.appendChild(b);
});

function select(idx) {
  current = DATA.demos[idx];
  tick = 0;
  [...clipsEl.children].forEach((b,i)=>b.classList.toggle('active', i===idx));
  labelEl.textContent = current.label;
  scrub.max = Math.max(0, current.controls.length - 1);
  draw();
}

function controlAt(t) {
  const c = current.controls[Math.min(t, current.controls.length-1)];
  return c;
}

function drawMouth(c) {
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0,0,w,h);
  // soft face plate
  const g = ctx.createRadialGradient(w*0.5, h*0.42, 40, w*0.5, h*0.45, 260);
  g.addColorStop(0, '#5a4034');
  g.addColorStop(1, '#241912');
  ctx.fillStyle = g;
  ctx.fillRect(0,0,w,h);

  const jaw = c[8], mouth = c[4], spread = c[5], round = c[6], teeth = c[7];
  const cx = w*0.5, cy = h*0.58;
  const open = 18 + 90 * Math.max(jaw, mouth);
  const halfW = 55 + 70 * Math.max(0, spread) + 20 * mouth - 35 * round;
  const pinch = 1 - 0.35 * round;

  // upper face emotion hints
  ctx.strokeStyle = 'rgba(243,230,216,0.55)';
  ctx.lineWidth = 3;
  const browY = h*0.28 - 18*c[2] + 12*c[3];
  ctx.beginPath();
  ctx.moveTo(w*0.28, browY);
  ctx.quadraticCurveTo(w*0.38, browY - 10*c[2], w*0.45, browY + 8*c[3]);
  ctx.moveTo(w*0.72, browY);
  ctx.quadraticCurveTo(w*0.62, browY - 10*c[2], w*0.55, browY + 8*c[3]);
  ctx.stroke();

  // eyes — C[0] eye_aperture: 0 open … 1 fully closed (blink)
  const close = Math.max(0, Math.min(1, c[0]));
  const eyeOpen = Math.max(1.2, (12 + 8*(1-close)) * (1 - 0.92*close));
  const eyeY = h*0.38;
  ctx.fillStyle = 'rgba(243,230,216,0.85)';
  ellipse(w*0.37, eyeY, 18, eyeOpen);
  ellipse(w*0.63, eyeY, 18, eyeOpen);
  if (close < 0.85) {
    ctx.fillStyle = '#1a120e';
    const pupil = Math.max(2, eyeOpen*0.45);
    ellipse(w*0.37, eyeY, 6, pupil);
    ellipse(w*0.63, eyeY, 6, pupil);
  } else {
    // sealed lids during blink
    ctx.strokeStyle = 'rgba(231,179,154,0.95)';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(w*0.37-16, eyeY); ctx.lineTo(w*0.37+16, eyeY);
    ctx.moveTo(w*0.63-16, eyeY); ctx.lineTo(w*0.63+16, eyeY);
    ctx.stroke();
  }
  // mouth cavity
  ctx.fillStyle = '#2a1210';
  ctx.beginPath();
  ctx.moveTo(cx - halfW, cy);
  ctx.bezierCurveTo(cx - halfW*0.6, cy - open*pinch, cx + halfW*0.6, cy - open*pinch, cx + halfW, cy);
  ctx.bezierCurveTo(cx + halfW*0.55, cy + open, cx - halfW*0.55, cy + open, cx - halfW, cy);
  ctx.closePath();
  ctx.fill();

  if (teeth > 0.15 && open > 28) {
    ctx.fillStyle = `rgba(240,230,220,${0.35 + 0.5*teeth})`;
    ctx.fillRect(cx - halfW*0.55, cy - open*0.15, halfW*1.1, 10 + 8*teeth);
  }

  // lip outline
  ctx.strokeStyle = '#e7b39a';
  ctx.lineWidth = 5;
  ctx.beginPath();
  ctx.moveTo(cx - halfW, cy);
  ctx.bezierCurveTo(cx - halfW*0.6, cy - open*pinch, cx + halfW*0.6, cy - open*pinch, cx + halfW, cy);
  ctx.bezierCurveTo(cx + halfW*0.55, cy + open, cx - halfW*0.55, cy + open, cx - halfW, cy);
  ctx.stroke();

  // vowel tag chip
  const tag = current.tags[tick] || '';
  ctx.fillStyle = 'rgba(0,0,0,0.35)';
  ctx.fillRect(16, h-48, 160, 28);
  ctx.fillStyle = '#f3e6d8';
  ctx.font = '16px Segoe UI, sans-serif';
  ctx.fillText(tag ? ('vowel ' + tag) : 'REST', 28, h-28);
}

function ellipse(x,y,rx,ry) {
  ctx.beginPath();
  ctx.ellipse(x,y,rx,ry,0,0,Math.PI*2);
  ctx.fill();
}

function draw() {
  if (!current) return;
  const c = controlAt(tick);
  drawMouth(c);
  CHANNELS.forEach((_, i) => {
    let v = c[i];
    let pct;
    if (i === 1 || i === 5) { // signed
      pct = ((v + 1) / 2) * 100;
    } else {
      pct = Math.max(0, Math.min(1, v)) * 100;
    }
    document.getElementById('f'+i).style.width = pct + '%';
    document.getElementById('v'+i).textContent = v.toFixed(2);
  });
  clockEl.textContent = (tick / DATA.tick_hz).toFixed(2) + 's / ' + (current.controls.length / DATA.tick_hz).toFixed(2) + 's';
  const blinking = c[0] > 0.55;
  tagsEl.innerHTML = (current.span_text || '') + '  ·  tick ' + tick
    + (blinking ? ' <span class="blink-badge">BLINK</span>' : '');
  scrub.value = String(tick);
}

function loop(ts) {
  if (playing) {
    if (!last) last = ts;
    const dt = ts - last;
    if (dt > (1000 / DATA.tick_hz)) {
      last = ts;
      tick += 1;
      if (tick >= current.controls.length) tick = 0;
      draw();
    }
  }
  raf = requestAnimationFrame(loop);
}

document.getElementById('play').onclick = () => { playing = true; last = 0; };
document.getElementById('pause').onclick = () => { playing = false; };
scrub.oninput = () => { tick = Number(scrub.value)|0; draw(); };

select(0);
raf = requestAnimationFrame(loop);
</script>
</body>
</html>
"""


def _tag_series(result) -> list[str]:
    tags = ["REST"] * result.chunk.n_ticks
    for span in result.payload.spans:
        for t in range(span.start_tick, min(span.end_tick, result.chunk.n_ticks)):
            tags[t] = span.tag
    return tags


def build(models: Path, out: Path) -> dict:
    from chorusface.vowel.pipeline import compose_utterance

    demos = []
    for spec in DEMOS:
        result = compose_utterance(spec["payload"], model_dir=models)
        demos.append(
            {
                "id": spec["id"],
                "label": spec["label"],
                "controls": np.asarray(result.controls, dtype=float).round(4).tolist(),
                "tags": _tag_series(result),
                "span_text": " → ".join(s.tag for s in result.payload.spans) or spec["payload"]["text"],
                "n_ticks": int(result.chunk.n_ticks),
            }
        )
    payload = {"tick_hz": 60, "demos": demos, "models": str(models)}
    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(payload))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    meta = out.with_suffix(".json")
    meta.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"html": str(out), "json": str(meta), "demos": [d["id"] for d in demos]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    info = build(args.models, args.out)
    print(json.dumps(info, indent=2))
    if not args.no_browser:
        webbrowser.open(args.out.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
