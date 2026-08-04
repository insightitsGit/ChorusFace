"""Exclusive API-key leases — one key ↔ one AI/system (client_id).

IP-only binding is brittle (NAT, laptops). Best practical approach for this
product:

1. Integrator generates a stable ``client_id`` (UUID) for their AI process
2. ``POST /auth/activate`` binds the API key to that client_id (exclusive)
3. Every later call must send the same key + ``X-ChorusFace-Client-Id``
4. A second system using the same key gets ``403 key in use``
5. Optional sticky IP (``CHORUSFACE_KEY_BIND_IP=1``) also rejects IP changes
6. Lease TTL + heartbeat; expired leases can be reclaimed

Bindings persist under ``secrets/api_key_bindings.json`` (gitignored).
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Final

from chorusface.api_keys import secrets_dir

BINDINGS_NAME: Final = "api_key_bindings.json"
DEFAULT_LEASE_TTL_S: Final = 900.0  # 15 minutes
HEADER_CLIENT_ID: Final = "X-ChorusFace-Client-Id"


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def lease_enabled() -> bool:
    """Exclusive leases on by default; set CHORUSFACE_KEY_LEASE=0 to disable."""
    return _env_flag("CHORUSFACE_KEY_LEASE", True)


def bind_ip_enabled() -> bool:
    return _env_flag("CHORUSFACE_KEY_BIND_IP", False)


def lease_ttl_s() -> float:
    try:
        return max(60.0, float(os.environ.get("CHORUSFACE_KEY_LEASE_TTL_S", DEFAULT_LEASE_TTL_S)))
    except ValueError:
        return DEFAULT_LEASE_TTL_S


def key_fingerprint(api_key: str) -> str:
    return sha256(str(api_key).encode("utf-8")).hexdigest()


def new_client_id() -> str:
    return str(uuid.uuid4())


@dataclass(slots=True)
class KeyBinding:
    key_fp: str
    client_id: str
    bound_ip: str
    activated_unix: float
    last_seen_unix: float
    label: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> KeyBinding:
        return cls(
            key_fp=str(data.get("key_fp") or ""),
            client_id=str(data.get("client_id") or ""),
            bound_ip=str(data.get("bound_ip") or ""),
            activated_unix=float(data.get("activated_unix") or 0.0),
            last_seen_unix=float(data.get("last_seen_unix") or 0.0),
            label=str(data.get("label") or ""),
        )


class KeyLeaseManager:
    """Process-wide exclusive leases for API keys."""

    def __init__(
        self,
        *,
        directory: Path | str | None = None,
        ttl_s: float | None = None,
        bind_ip: bool | None = None,
        enabled: bool | None = None,
    ) -> None:
        self._root = secrets_dir(directory)
        self._path = self._root / BINDINGS_NAME
        self._ttl_s = float(ttl_s if ttl_s is not None else lease_ttl_s())
        self._bind_ip = bool(bind_ip_enabled() if bind_ip is None else bind_ip)
        self._enabled = bool(lease_enabled() if enabled is None else enabled)
        self._lock = threading.RLock()
        self._bindings: dict[str, KeyBinding] = {}
        self._load()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def bind_ip(self) -> bool:
        return self._bind_ip

    @property
    def ttl_s(self) -> float:
        return self._ttl_s

    def _load(self) -> None:
        if not self._path.is_file():
            return
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        items = payload.get("bindings") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            return
        now = time.time()
        for raw in items:
            if not isinstance(raw, dict):
                continue
            binding = KeyBinding.from_json(raw)
            if not binding.key_fp or not binding.client_id:
                continue
            if now - binding.last_seen_unix > self._ttl_s:
                continue
            self._bindings[binding.key_fp] = binding

    def _save(self) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "chorusface.api_key_bindings.v1",
            "updated_unix": time.time(),
            "ttl_s": self._ttl_s,
            "bind_ip": self._bind_ip,
            "bindings": [b.to_json() for b in self._bindings.values()],
        }
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self._path)
        try:
            self._path.chmod(0o600)
        except OSError:
            pass

    def _expired(self, binding: KeyBinding, *, now: float | None = None) -> bool:
        t = time.time() if now is None else now
        return (t - float(binding.last_seen_unix)) > self._ttl_s

    def activate(
        self,
        api_key: str,
        client_id: str,
        *,
        peer_ip: str = "",
        label: str = "",
    ) -> dict[str, Any]:
        """Bind key → client_id. Fails if another live client holds the key."""
        if not self._enabled:
            return {
                "ok": True,
                "leased": False,
                "client_id": client_id,
                "detail": "leases disabled",
            }
        cid = str(client_id or "").strip()
        if not cid or len(cid) < 8:
            raise ValueError("client_id required (min 8 chars; use a UUID)")
        fp = key_fingerprint(api_key)
        now = time.time()
        with self._lock:
            current = self._bindings.get(fp)
            if current is not None and not self._expired(current, now=now):
                if current.client_id != cid:
                    raise PermissionError(
                        "API key already bound to another system "
                        f"(client_id={current.client_id[:8]}…)"
                    )
                if self._bind_ip and current.bound_ip and peer_ip and current.bound_ip != peer_ip:
                    raise PermissionError(
                        "API key bound to a different IP address"
                    )
                current.last_seen_unix = now
                if peer_ip and not current.bound_ip:
                    current.bound_ip = peer_ip
                self._save()
                return {
                    "ok": True,
                    "leased": True,
                    "client_id": cid,
                    "bound_ip": current.bound_ip,
                    "expires_in_s": self._ttl_s,
                    "reused": True,
                }
            binding = KeyBinding(
                key_fp=fp,
                client_id=cid,
                bound_ip=str(peer_ip or ""),
                activated_unix=now,
                last_seen_unix=now,
                label=str(label or ""),
            )
            self._bindings[fp] = binding
            self._save()
            return {
                "ok": True,
                "leased": True,
                "client_id": cid,
                "bound_ip": binding.bound_ip,
                "expires_in_s": self._ttl_s,
                "reused": False,
            }

    def release(self, api_key: str, client_id: str) -> dict[str, Any]:
        if not self._enabled:
            return {"ok": True, "released": False, "detail": "leases disabled"}
        fp = key_fingerprint(api_key)
        cid = str(client_id or "").strip()
        with self._lock:
            current = self._bindings.get(fp)
            if current is None:
                return {"ok": True, "released": False, "detail": "not bound"}
            if current.client_id != cid:
                raise PermissionError("client_id does not own this API key lease")
            del self._bindings[fp]
            self._save()
            return {"ok": True, "released": True}

    def heartbeat(self, api_key: str, client_id: str, *, peer_ip: str = "") -> dict[str, Any]:
        self.authorize(api_key, client_id, peer_ip=peer_ip, touch=True)
        return {"ok": True, "expires_in_s": self._ttl_s}

    def authorize(
        self,
        api_key: str,
        client_id: str,
        *,
        peer_ip: str = "",
        touch: bool = True,
    ) -> None:
        """Raise PermissionError if key/client/ip lease check fails."""
        if not self._enabled:
            return
        cid = str(client_id or "").strip()
        if not cid:
            raise PermissionError(
                f"missing client_id (send {HEADER_CLIENT_ID} header or client_id= query)"
            )
        fp = key_fingerprint(api_key)
        now = time.time()
        with self._lock:
            current = self._bindings.get(fp)
            if current is None or self._expired(current, now=now):
                if current is not None and self._expired(current, now=now):
                    del self._bindings[fp]
                    self._save()
                raise PermissionError(
                    "API key is not activated for this system — POST /auth/activate first"
                )
            if current.client_id != cid:
                raise PermissionError(
                    "API key in use by another system (one key → one AI)"
                )
            if self._bind_ip and current.bound_ip and peer_ip and current.bound_ip != peer_ip:
                raise PermissionError("API key bound to a different IP address")
            if touch:
                current.last_seen_unix = now
                self._save()

    def status(self) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            live = [
                {
                    "client_id_prefix": b.client_id[:8],
                    "bound_ip": b.bound_ip if self._bind_ip else "",
                    "age_s": round(now - b.activated_unix, 1),
                    "idle_s": round(now - b.last_seen_unix, 1),
                    "label": b.label,
                }
                for b in self._bindings.values()
                if not self._expired(b, now=now)
            ]
        return {
            "enabled": self._enabled,
            "bind_ip": self._bind_ip,
            "ttl_s": self._ttl_s,
            "active_leases": len(live),
            "leases": live,
        }


__all__ = [
    "BINDINGS_NAME",
    "DEFAULT_LEASE_TTL_S",
    "HEADER_CLIENT_ID",
    "KeyBinding",
    "KeyLeaseManager",
    "bind_ip_enabled",
    "key_fingerprint",
    "lease_enabled",
    "lease_ttl_s",
    "new_client_id",
]
