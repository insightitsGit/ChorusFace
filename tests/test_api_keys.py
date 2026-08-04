"""Encrypted local API key vault."""

from __future__ import annotations

from pathlib import Path

from chorusface.api_keys import (
    ApiKeyStore,
    build_records,
    decrypt_vault,
    load_vault_keys,
    token_accepted,
    write_vault,
)


def test_store_accepts_only_known_keys() -> None:
    store = ApiKeyStore(["alpha-key", "beta-key"])
    assert store.accepts("alpha-key")
    assert store.accepts("beta-key")
    assert not store.accepts("gamma-key")
    assert not store.accepts("")


def test_write_and_load_encrypted_vault(tmp_path: Path) -> None:
    records = build_records(10)
    assert len(records) == 10
    assert all(r.key.startswith("chorusface_sk_") for r in records)
    paths = write_vault(records, directory=tmp_path)
    assert paths["vault"].is_file()
    assert paths["master_key"].is_file()
    assert paths["handoff"].is_file()
    store = load_vault_keys(tmp_path)
    assert store.count == 10
    assert store.accepts(records[0].key)
    assert store.accepts(records[9].key)
    assert not store.accepts("chorusface_sk_not_real")


def test_token_accepted_helper() -> None:
    store = ApiKeyStore(["one"])
    assert token_accepted(store, "one")
    assert token_accepted(None, "legacy", "legacy")
    assert not token_accepted(store, "nope")


def test_vault_roundtrip_payload(tmp_path: Path) -> None:
    records = build_records(3)
    paths = write_vault(records, directory=tmp_path)
    master = paths["master_key"].read_bytes().strip()
    payload = decrypt_vault(paths["vault"].read_bytes(), master)
    assert payload["count"] == 3
    assert len(payload["keys"]) == 3
