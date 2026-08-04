"""Hard caps for untrusted / authenticated ingress (QA CR-001…007).

Lab demos stay usable; remote bind and wire-loop decode must not be able to
OOM or pickle-exec the process without an explicit, reviewed escape hatch.
"""

from __future__ import annotations

import hashlib
import json
import os
import zlib
from pathlib import Path
from typing import BinaryIO, Final

# Face ROI on the 256² field — never allocate larger from a wire header.
MAX_FACE_SIDE: Final = 256
MAX_FACE_CELLS: Final = MAX_FACE_SIDE * MAX_FACE_SIDE
MAX_SPARSE_COUNT: Final = MAX_FACE_CELLS

# Voice / chat backlog (authenticated DoS).
MAX_VOICE_BUFFER_SECONDS: Final = 30.0
MAX_VOICE_INBOX_EVENTS: Final = 2_048
MAX_CHAT_QUEUE: Final = 32
MAX_REPLY_QUEUE: Final = 32

# HTTP client bodies (TTS audio / LLM JSON).
MAX_TTS_RESPONSE_BYTES: Final = 8 * 1024 * 1024
MAX_LLM_RESPONSE_BYTES: Final = 512 * 1024

# Lane-B TickPackage after zlib (HELLO max_payload is 512 KiB; leave headroom).
MAX_ZLIB_OUTPUT_BYTES: Final = 2 * 1024 * 1024

# Published lab string — never accept for remote bind.
WEAK_BRIDGE_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "tickfeed-lab",
        "changeme",
        "secret",
        "password",
        "token",
        "test",
        "chorusface",
    }
)


def token_fingerprint(token: str) -> str:
    """Short non-reversible id for logs (full bearer stays off stdout)."""
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return digest[:12]


def resolve_bridge_token(
    *,
    env_var: str = "CHORUSFACE_BRIDGE_TOKEN",
    token_file: Path | str | None = None,
) -> str:
    """Env token, else world ``.bridge_token`` file written by the demo launcher."""
    env = str(os.environ.get(env_var, "") or "").strip()
    if env:
        return env
    path = Path(token_file) if token_file is not None else None
    if path is None:
        # Default TickFeed lab world — matches scripts/run_tickfeed_demo.py.
        here = Path(__file__).resolve()
        path = here.parents[2] / "output" / "worlds" / "tickfeed" / ".bridge_token"
    if path.is_file():
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        if lines and lines[0].strip():
            return lines[0].strip()
    raise RuntimeError(
        f"No bridge token in ${env_var} or {path}. "
        "Run scripts/run_tickfeed_demo.py or export CHORUSFACE_BRIDGE_TOKEN."
    )


def is_weak_bridge_token(token: str) -> bool:
    return str(token or "").strip().lower() in WEAK_BRIDGE_TOKENS


def read_capped(stream: BinaryIO, max_bytes: int) -> bytes:
    """Read at most ``max_bytes``; raise ``ValueError`` if the peer sent more."""
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    chunks: list[bytes] = []
    total = 0
    while True:
        block = stream.read(min(65536, max_bytes - total + 1))
        if not block:
            break
        total += len(block)
        if total > max_bytes:
            raise ValueError(f"response exceeds {max_bytes} bytes")
        chunks.append(block)
    return b"".join(chunks)


def zlib_decompress_capped(
    compressed: bytes, *, max_output: int, expect_nbytes: int | None = None
) -> bytes:
    """Decompress with a hard output ceiling (zip-bomb guard)."""
    ceiling = int(max_output)
    if expect_nbytes is not None:
        if expect_nbytes < 0 or expect_nbytes > ceiling:
            raise ValueError(
                f"declared nbytes {expect_nbytes} outside 0..{ceiling}"
            )
        ceiling = int(expect_nbytes)
    decoder = zlib.decompressobj()
    out = decoder.decompress(compressed, max_length=ceiling)
    if decoder.unconsumed_tail or not decoder.eof:
        # More output available than the cap allowed.
        raise ValueError(f"zlib output exceeds {ceiling} bytes")
    unused = decoder.unused_data
    if unused:
        raise ValueError("zlib trailing garbage after stream end")
    return out


def _sidecar_digest_path(model_path: Path) -> Path:
    return model_path.with_suffix(model_path.suffix + ".sha256")


def _manifest_digests(model_path: Path) -> dict[str, str]:
    manifest = model_path.parent / "checksums.json"
    if not manifest.is_file():
        return {}
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    digests = payload.get("sha256") if isinstance(payload, dict) else None
    return digests if isinstance(digests, dict) else {}


def expected_model_digest(model_path: Path) -> str | None:
    """Return pinned sha256 hex for ``model_path``, or None if unpinned."""
    sidecar = _sidecar_digest_path(model_path)
    if sidecar.is_file():
        text = sidecar.read_text(encoding="utf-8").strip().split()[0]
        return text.lower() or None
    digests = _manifest_digests(model_path)
    key = model_path.name
    value = digests.get(key)
    return str(value).lower() if value else None


def write_model_digest(model_path: Path) -> str:
    """Write ``*.joblib.sha256`` sidecar; return hex digest."""
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    _sidecar_digest_path(model_path).write_text(digest + "\n", encoding="utf-8")
    return digest


def dump_joblib(obj: object, model_path: Path | str) -> Path:
    """``joblib.dump`` + sha256 sidecar so loads can verify integrity."""
    import joblib

    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    write_model_digest(path)
    return path


def require_unpinned_models_allowed() -> bool:
    """Lab default allows unpinned models; remote/prod can require pins."""
    flag = os.environ.get("CHORUSFACE_REQUIRE_MODEL_DIGEST", "").strip().lower()
    if flag in {"1", "true", "yes", "on"}:
        return False
    allow = os.environ.get("CHORUSFACE_ALLOW_UNPINNED_MODELS", "1").strip().lower()
    return allow in {"1", "true", "yes", "on", ""}


def safe_joblib_load(model_path: Path | str, *, world_root: Path | str | None = None):
    """Load a joblib file with path containment + optional sha256 pin."""
    import joblib

    path = Path(model_path).resolve()
    if world_root is not None:
        root = Path(world_root).resolve()
        root = root if root.is_dir() else root.parent
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"refusing joblib outside world root: {path} not under {root}"
            ) from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = expected_model_digest(path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected is not None:
        if actual != expected:
            raise ValueError(
                f"model digest mismatch for {path.name}: "
                f"got {actual[:12]}… want {expected[:12]}…"
            )
    elif not require_unpinned_models_allowed():
        raise ValueError(
            f"unpinned model {path.name}: write {path.name}.sha256 or "
            "checksums.json, or set CHORUSFACE_ALLOW_UNPINNED_MODELS=1 for lab"
        )
    return joblib.load(path)


def validate_face_box(w: int, h: int, *, x: int = 0, y: int = 0) -> None:
    """Reject wire face dims that would OOM or escape the field grid."""
    if int(w) <= 0 or int(h) <= 0:
        raise ValueError(f"invalid face size {w}x{h}")
    if int(w) > MAX_FACE_SIDE or int(h) > MAX_FACE_SIDE:
        raise ValueError(
            f"face {w}x{h} exceeds max side {MAX_FACE_SIDE}"
        )
    if int(w) * int(h) > MAX_FACE_CELLS:
        raise ValueError(
            f"face cells {int(w) * int(h)} exceed max {MAX_FACE_CELLS}"
        )
    if int(x) < 0 or int(y) < 0:
        raise ValueError(f"negative face origin {(x, y)}")


__all__ = [
    "MAX_CHAT_QUEUE",
    "MAX_FACE_CELLS",
    "MAX_FACE_SIDE",
    "MAX_LLM_RESPONSE_BYTES",
    "MAX_REPLY_QUEUE",
    "MAX_SPARSE_COUNT",
    "MAX_TTS_RESPONSE_BYTES",
    "MAX_VOICE_BUFFER_SECONDS",
    "MAX_VOICE_INBOX_EVENTS",
    "MAX_ZLIB_OUTPUT_BYTES",
    "WEAK_BRIDGE_TOKENS",
    "expected_model_digest",
    "is_weak_bridge_token",
    "read_capped",
    "require_unpinned_models_allowed",
    "resolve_bridge_token",
    "safe_joblib_load",
    "token_fingerprint",
    "validate_face_box",
    "dump_joblib",
    "write_model_digest",
    "zlib_decompress_capped",
]
