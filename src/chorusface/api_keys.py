"""Local encrypted API-key vault for ChorusFace / Prism speak auth.

Ten sample keys live under ``secrets/`` (gitignored). The vault is Fernet-
encrypted with a local master key. Integrators must present one of these keys
as ``Authorization: Bearer …`` or ``?token=`` to use FaceBridge / Prism routes.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
from dataclasses import dataclass
from hmac import compare_digest
from pathlib import Path
from typing import Any, Final, Iterable, Sequence

DEFAULT_KEY_COUNT: Final = 10
KEY_PREFIX: Final = "chorusface_sk_"
VAULT_SCHEMA: Final = "chorusface.api_keys.vault.v1"
DEFAULT_SECRETS_DIR: Final = Path(__file__).resolve().parents[2] / "secrets"
MASTER_KEY_NAME: Final = ".master_key"
VAULT_NAME: Final = "api_keys.vault.enc"
HANDOFF_NAME: Final = "api_keys.handoff.local.txt"
HASHES_NAME: Final = "api_keys.hashes.json"


@dataclass(frozen=True, slots=True)
class ApiKeyRecord:
    id: str
    key: str
    label: str
    created_unix: float


class ApiKeyStore:
    """In-memory accept-list of API keys (plaintext loaded from decrypted vault)."""

    def __init__(self, keys: Sequence[str]) -> None:
        cleaned = tuple(str(k).strip() for k in keys if str(k).strip())
        if not cleaned:
            raise ValueError("ApiKeyStore requires at least one key")
        self._keys = cleaned
        # Precompute digests for O(1)-ish membership without leaking lengths unevenly.
        self._digests = {hashlib.sha256(k.encode("utf-8")).digest(): k for k in cleaned}

    @property
    def count(self) -> int:
        return len(self._keys)

    @property
    def primary(self) -> str:
        return self._keys[0]

    def __iter__(self):
        return iter(self._keys)

    def accepts(self, presented: str) -> bool:
        token = str(presented or "").strip()
        if not token:
            return False
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        # Constant-time path: compare against each stored digest.
        matched = False
        for stored in self._digests:
            if compare_digest(digest, stored):
                matched = True
        return matched


def secrets_dir(path: Path | str | None = None) -> Path:
    env = os.environ.get("CHORUSFACE_SECRETS_DIR", "").strip()
    if path is not None:
        return Path(path)
    if env:
        return Path(env)
    return DEFAULT_SECRETS_DIR


def _require_fernet():
    try:
        from cryptography.fernet import Fernet
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "cryptography is required for the API key vault. "
            'Install with: pip install "cryptography>=42"'
        ) from exc
    return Fernet


def generate_master_key() -> bytes:
    Fernet = _require_fernet()
    return Fernet.generate_key()


def generate_api_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(32)


def build_records(count: int = DEFAULT_KEY_COUNT) -> list[ApiKeyRecord]:
    now = time.time()
    records: list[ApiKeyRecord] = []
    for index in range(1, int(count) + 1):
        records.append(
            ApiKeyRecord(
                id=f"k{index:02d}",
                key=generate_api_key(),
                label=f"sample-{index:02d}",
                created_unix=now,
            )
        )
    return records


def vault_payload(records: Sequence[ApiKeyRecord]) -> dict[str, Any]:
    return {
        "schema": VAULT_SCHEMA,
        "created_unix": time.time(),
        "count": len(records),
        "keys": [
            {
                "id": r.id,
                "key": r.key,
                "label": r.label,
                "created_unix": r.created_unix,
                "sha256": hashlib.sha256(r.key.encode("utf-8")).hexdigest(),
            }
            for r in records
        ],
    }


def encrypt_vault(payload: dict[str, Any], master_key: bytes) -> bytes:
    Fernet = _require_fernet()
    token = Fernet(master_key).encrypt(
        json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    )
    return token


def decrypt_vault(blob: bytes, master_key: bytes) -> dict[str, Any]:
    Fernet = _require_fernet()
    raw = Fernet(master_key).decrypt(blob)
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("vault payload must be a JSON object")
    return payload


def write_vault(
    records: Sequence[ApiKeyRecord],
    *,
    directory: Path | str | None = None,
) -> dict[str, Path]:
    """Create master key + encrypted vault + local handoff file (all gitignored)."""
    root = secrets_dir(directory)
    root.mkdir(parents=True, exist_ok=True)
    # Restrictive perms on POSIX; Windows ignores mode bits mostly.
    master_path = root / MASTER_KEY_NAME
    vault_path = root / VAULT_NAME
    handoff_path = root / HANDOFF_NAME
    hashes_path = root / HASHES_NAME

    master_key = generate_master_key()
    master_path.write_bytes(master_key + b"\n")
    try:
        master_path.chmod(0o600)
    except OSError:
        pass

    payload = vault_payload(records)
    vault_path.write_bytes(encrypt_vault(payload, master_key))
    try:
        vault_path.chmod(0o600)
    except OSError:
        pass

    lines = [
        "# ChorusFace / PrismAPI sample API keys - LOCAL ONLY. Do not commit.",
        f"# schema={VAULT_SCHEMA} count={len(records)}",
        "# Use as: Authorization: Bearer <key>  or  /stream.mjpg?token=<key>",
        "",
    ]
    hash_rows: list[dict[str, str]] = []
    for record in records:
        lines.append(f"{record.id}\t{record.label}\t{record.key}")
        hash_rows.append(
            {
                "id": record.id,
                "label": record.label,
                "sha256": hashlib.sha256(record.key.encode("utf-8")).hexdigest(),
            }
        )
    handoff_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        handoff_path.chmod(0o600)
    except OSError:
        pass

    hashes_path.write_text(
        json.dumps({"schema": "chorusface.api_keys.hashes.v1", "keys": hash_rows}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return {
        "master_key": master_path,
        "vault": vault_path,
        "handoff": handoff_path,
        "hashes": hashes_path,
    }


def load_master_key(directory: Path | str | None = None) -> bytes:
    root = secrets_dir(directory)
    env_key = os.environ.get("CHORUSFACE_MASTER_KEY", "").strip()
    if env_key:
        return env_key.encode("utf-8")
    path = Path(os.environ.get("CHORUSFACE_MASTER_KEY_FILE", "") or (root / MASTER_KEY_NAME))
    if not path.is_file():
        raise FileNotFoundError(f"master key not found: {path}")
    return path.read_bytes().strip()


def load_vault_keys(directory: Path | str | None = None) -> ApiKeyStore:
    """Decrypt local vault → ApiKeyStore."""
    root = secrets_dir(directory)
    vault_path = Path(os.environ.get("CHORUSFACE_API_KEYS_VAULT", "") or (root / VAULT_NAME))
    if not vault_path.is_file():
        raise FileNotFoundError(f"API key vault not found: {vault_path}")
    master = load_master_key(root)
    payload = decrypt_vault(vault_path.read_bytes(), master)
    keys = [
        str(item.get("key", "")).strip()
        for item in (payload.get("keys") or [])
        if isinstance(item, dict)
    ]
    return ApiKeyStore(keys)


def resolve_api_key_store(
    *,
    bridge_token: str | None = None,
    directory: Path | str | None = None,
) -> ApiKeyStore:
    """Prefer encrypted vault; fall back to single bridge token / env list."""
    try:
        return load_vault_keys(directory)
    except (FileNotFoundError, RuntimeError, ValueError):
        pass

    multi = os.environ.get("CHORUSFACE_API_KEYS", "").strip()
    if multi:
        parts = [p.strip() for p in multi.replace(";", ",").split(",") if p.strip()]
        if parts:
            return ApiKeyStore(parts)

    single = (
        str(bridge_token or "").strip()
        or os.environ.get("CHORUSFACE_BRIDGE_TOKEN", "").strip()
    )
    if single:
        return ApiKeyStore([single])
    raise FileNotFoundError(
        "No API keys: run python scripts/generate_api_keys.py "
        "or set CHORUSFACE_BRIDGE_TOKEN / CHORUSFACE_API_KEYS"
    )


def token_accepted(store: ApiKeyStore | None, presented: str, *extra: str) -> bool:
    """Return True if presented matches the store or any extra legacy tokens."""
    token = str(presented or "").strip()
    if not token:
        return False
    if store is not None and store.accepts(token):
        return True
    for candidate in extra:
        other = str(candidate or "").strip()
        if other and compare_digest(token, other):
            return True
    return False


__all__ = [
    "DEFAULT_KEY_COUNT",
    "DEFAULT_SECRETS_DIR",
    "HANDOFF_NAME",
    "HASHES_NAME",
    "KEY_PREFIX",
    "MASTER_KEY_NAME",
    "VAULT_NAME",
    "VAULT_SCHEMA",
    "ApiKeyRecord",
    "ApiKeyStore",
    "build_records",
    "decrypt_vault",
    "encrypt_vault",
    "generate_api_key",
    "generate_master_key",
    "load_master_key",
    "load_vault_keys",
    "resolve_api_key_store",
    "secrets_dir",
    "token_accepted",
    "vault_payload",
    "write_vault",
]
