#!/usr/bin/env python3
"""Generate 10 encrypted local API keys for ChorusFace / PrismAPI (never commit).

Writes under secrets/ (gitignored):
  .master_key                 Fernet master key
  api_keys.vault.enc          encrypted key vault
  api_keys.handoff.local.txt  plaintext keys for handoff to integrators
  api_keys.hashes.json        sha256 digests (no plaintext)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of sample keys (default 10)",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=None,
        help="Secrets directory (default: ./secrets or CHORUSFACE_SECRETS_DIR)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing vault",
    )
    args = parser.parse_args()

    from chorusface.api_keys import (
        HANDOFF_NAME,
        VAULT_NAME,
        build_records,
        secrets_dir,
        write_vault,
    )

    root = secrets_dir(args.dir)
    vault = root / VAULT_NAME
    if vault.is_file() and not args.force:
        print(f"Vault already exists: {vault}", file=sys.stderr)
        print("Re-run with --force to rotate all keys.", file=sys.stderr)
        return 2

    records = build_records(int(args.count))
    paths = write_vault(records, directory=root)
    print(f"Wrote {len(records)} encrypted API keys under {root}")
    for label, path in paths.items():
        print(f"  {label}: {path}")
    print()
    print("Handoff file (give one key per integrator — do not commit):")
    print(f"  {paths['handoff']}")
    print()
    print("Service will load the vault automatically when secrets/ is present.")
    print("Example:")
    print(
        f'  curl -H "Authorization: Bearer <key-from-{HANDOFF_NAME}>" '
        f'-H "Content-Type: application/json" '
        f'-d "{{\\"text\\":\\"Hello\\"}}" http://127.0.0.1:8766/prism/speak'
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
