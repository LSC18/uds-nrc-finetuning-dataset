#!/usr/bin/env python3
"""Write SHA-256 checksums for versioned dataset artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = sorted(
    path
    for directory in (ROOT / "full_v2", ROOT / "full_v3")
    for path in directory.glob("*")
)


def main() -> int:
    lines = []
    for path in FILES:
        if not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(ROOT)}")
    (ROOT / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} checksums")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
