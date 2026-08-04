"""Exclusive API key leases — one key → one AI/system."""

from __future__ import annotations

from pathlib import Path

import pytest

from chorusface.key_lease import KeyLeaseManager, new_client_id


def test_one_key_one_client(tmp_path: Path) -> None:
    mgr = KeyLeaseManager(directory=tmp_path, ttl_s=600, bind_ip=False, enabled=True)
    key = "chorusface_sk_test_key_one"
    a = new_client_id()
    b = new_client_id()
    out = mgr.activate(key, a, peer_ip="1.1.1.1")
    assert out["ok"] is True
    assert out["client_id"] == a
    mgr.authorize(key, a, peer_ip="1.1.1.1")
    with pytest.raises(PermissionError, match="another system"):
        mgr.activate(key, b, peer_ip="2.2.2.2")
    with pytest.raises(PermissionError, match="another system"):
        mgr.authorize(key, b, peer_ip="2.2.2.2")


def test_release_allows_rebind(tmp_path: Path) -> None:
    mgr = KeyLeaseManager(directory=tmp_path, ttl_s=600, bind_ip=False, enabled=True)
    key = "chorusface_sk_test_key_two"
    a = new_client_id()
    b = new_client_id()
    mgr.activate(key, a)
    mgr.release(key, a)
    out = mgr.activate(key, b)
    assert out["client_id"] == b


def test_bind_ip_rejects_other_address(tmp_path: Path) -> None:
    mgr = KeyLeaseManager(directory=tmp_path, ttl_s=600, bind_ip=True, enabled=True)
    key = "chorusface_sk_test_key_ip"
    cid = new_client_id()
    mgr.activate(key, cid, peer_ip="10.0.0.1")
    mgr.authorize(key, cid, peer_ip="10.0.0.1")
    with pytest.raises(PermissionError, match="different IP"):
        mgr.authorize(key, cid, peer_ip="10.0.0.2")


def test_expired_lease_can_be_stolen(tmp_path: Path) -> None:
    mgr = KeyLeaseManager(directory=tmp_path, ttl_s=1, bind_ip=False, enabled=True)
    key = "chorusface_sk_test_key_ttl"
    a = new_client_id()
    b = new_client_id()
    mgr.activate(key, a)
    # Force expiry.
    binding = mgr._bindings[list(mgr._bindings)[0]]
    binding.last_seen_unix -= 10
    out = mgr.activate(key, b)
    assert out["client_id"] == b
