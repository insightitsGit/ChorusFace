"""Resolve the live face-bridge Bearer for TickFeed QA scripts.

Order: ``CHORUSFACE_BRIDGE_TOKEN`` env → ``output/worlds/tickfeed/.bridge_token``.
No hardcoded lab string (QA CR-001).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from chorusface.io_limits import resolve_bridge_token  # noqa: E402

DEFAULT_TOKEN_FILE = ROOT / "output" / "worlds" / "tickfeed" / ".bridge_token"


def bridge_token(*, token_file: Path | None = None) -> str:
    try:
        return resolve_bridge_token(token_file=token_file or DEFAULT_TOKEN_FILE)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc


def auth_header(**kwargs: object) -> dict[str, str]:
    return {"Authorization": f"Bearer {bridge_token(**kwargs)}"}  # type: ignore[arg-type]


__all__ = ["DEFAULT_TOKEN_FILE", "auth_header", "bridge_token"]
