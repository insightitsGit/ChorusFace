"""Path A probe: ownership table offline + optional live GET /probe.

Offline (always)::

    python scripts/probe_mouth_live.py

Live (demo must be running with --bridge)::

    set CHORUSFACE_BRIDGE_TOKEN=...
    python scripts/probe_mouth_live.py
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from chorusface.mouth_owner import resolve_mouth_ownership


CASES = (
    (0.0, "NEUTRAL", "REST", False),
    (0.02, "NEUTRAL", "PP", True),
    (0.55, "NEUTRAL", "AH", True),
    (0.55, "NEUTRAL", "EE", True),
    (0.2, "HAPPY", "EH", True),
    (0.3, "SURPRISED", "AH", True),
)


def _offline() -> None:
    print("=== Mouth ownership (offline) ===")
    for openness, emotion, phoneme, speaking in CASES:
        own = resolve_mouth_ownership(
            openness=openness,
            emotion=emotion,
            phoneme=phoneme,
            speaking=speaking,
        )
        print(
            f"{phoneme}/{emotion} open={openness:.2f} speaking={speaking} "
            f"-> {list(own.owners)} plate={own.plate_amount:.2f}"
        )


def _live() -> None:
    token = os.environ.get("CHORUSFACE_BRIDGE_TOKEN", "").strip()
    host = os.environ.get("CHORUSFACE_BRIDGE_URL", "http://127.0.0.1:8766").rstrip("/")
    if not token:
        print("\n=== Live /probe skipped (set CHORUSFACE_BRIDGE_TOKEN) ===")
        return
    req = urllib.request.Request(
        f"{host}/probe",
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"\n=== Live /probe failed: {exc} ===")
        return
    print("\n=== Live GET /probe ===")
    print(json.dumps(payload, indent=2))


def main() -> None:
    _offline()
    _live()


if __name__ == "__main__":
    main()
