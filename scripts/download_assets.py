#!/usr/bin/env python3
"""Download selected release assets and verify their SHA-256 digests."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("groups", nargs="*", default=["runtime", "checkpoints"])
    parser.add_argument("--manifest", type=Path, default=Path("assets/manifest.json"))
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    entries = json.loads(args.manifest.read_text(encoding="utf-8"))["assets"]
    selected = [item for item in entries if item["group"] in args.groups]
    if not selected:
        raise SystemExit(f"No assets matched groups: {', '.join(args.groups)}")
    for item in selected:
        target = args.root / item["path"]
        if not target.exists() and not args.verify_only:
            if not item.get("url"):
                raise SystemExit(f"No public URL has been assigned for {item['name']}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with urlopen(item["url"]) as response, target.open("wb") as output:
                while chunk := response.read(8 * 1024 * 1024):
                    output.write(chunk)
        if not target.exists():
            raise SystemExit(f"Missing {target}")
        if item.get("size") is not None and target.stat().st_size != item["size"]:
            raise SystemExit(
                f"Size mismatch for {target}: expected {item['size']}, got {target.stat().st_size}"
            )
        actual = digest(target)
        if item.get("sha256") and actual != item["sha256"]:
            raise SystemExit(f"Checksum mismatch for {target}: {actual}")
        print(f"verified {target}")


if __name__ == "__main__":
    main()
