# Local API keys (gitignored)

This folder holds the **encrypted** ChorusFace / PrismAPI sample keys.

## Generate (once per machine)

```powershell
pip install cryptography
python scripts/generate_api_keys.py
```

Creates (all gitignored except this README):

| File | Purpose |
| --- | --- |
| `.master_key` | Fernet master key — never share / never commit |
| `api_keys.vault.enc` | Encrypted vault of 10 keys |
| `api_keys.handoff.local.txt` | Plaintext keys to give integrators |
| `api_keys.hashes.json` | SHA-256 digests only |

## Use (one key → one AI / one system)

Integrators need **one** key from `api_keys.handoff.local.txt` **and** a stable
`client_id` (UUID) for their process. IP-only binding is optional and brittle;
exclusive **client_id lease** is the default.

```http
POST /auth/activate
Authorization: Bearer chorusface_sk_…
{"client_id":"<your-stable-uuid>"}
```

Then every call:

```http
Authorization: Bearer chorusface_sk_…
X-ChorusFace-Client-Id: <same-uuid>
POST /prism/speak
```

Embed:

```text
/stream.mjpg?token=chorusface_sk_…&client_id=<same-uuid>
```

A second system using the same key gets `403` (“API key in use by another system”).

| Env | Default | Meaning |
| --- | --- | --- |
| `CHORUSFACE_KEY_LEASE` | `1` | Exclusive client_id lease |
| `CHORUSFACE_KEY_BIND_IP` | `0` | Also sticky-bind activation IP |
| `CHORUSFACE_KEY_LEASE_TTL_S` | `900` | Idle lease expiry (seconds) |

The face service auto-loads `secrets/api_keys.vault.enc` when present.

## Rotate

```powershell
python scripts/generate_api_keys.py --force
```

Old keys stop working immediately after restart.
