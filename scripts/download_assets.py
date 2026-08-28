#!/usr/bin/env python3
"""Download selected release assets and verify their SHA-256 digests."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import tarfile
from urllib.request import urlopen


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def extract_archive(path: Path, destination: Path, strip_components: int = 0) -> None:
    """Safely extract a tar archive, optionally removing leading path components."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(path, "r:*") as archive:
        for original in archive.getmembers():
            parts = Path(original.name).parts
            if len(parts) <= strip_components:
                continue
            relative = Path(*parts[strip_components:])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe archive path in {path}: {original.name}")
            target = (root / relative).resolve(strict=False)
            if root != target and root not in target.parents:
                raise ValueError(f"Archive path escapes destination: {original.name}")
            member = copy.copy(original)
            member.name = relative.as_posix()
            if not (member.isfile() or member.isdir() or member.issym()):
                raise ValueError(f"Unsupported archive member: {original.name}")
            if member.issym():
                link_target = (target.parent / member.linkname).resolve(strict=False)
                if root != link_target and root not in link_target.parents:
                    raise ValueError(f"Archive link escapes destination: {original.name}")
            archive.extract(member, destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("groups", nargs="*", default=["runtime", "checkpoints"])
    parser.add_argument("--manifest", type=Path, default=Path("assets/manifest.json"))
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--extract", action="store_true",
        help="Extract archives into their manifest-defined runtime locations",
    )
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
            partial = target.with_name(target.name + ".part")
            print(f"downloading {item['name']} -> {target}", flush=True)
            with urlopen(item["url"]) as response, partial.open("wb") as output:
                total = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                next_report = 256 * 1024 * 1024
                while chunk := response.read(8 * 1024 * 1024):
                    output.write(chunk)
                    downloaded += len(chunk)
                    if downloaded >= next_report:
                        suffix = f" / {total} ({downloaded / total:.1%})" if total else ""
                        print(f"  {downloaded} bytes{suffix}", flush=True)
                        next_report += 256 * 1024 * 1024
            partial.replace(target)
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
        if args.extract and item.get("extract_to"):
            destination = args.root / item["extract_to"]
            extract_archive(target, destination, item.get("strip_components", 0))
            print(f"extracted {target} -> {destination}")


if __name__ == "__main__":
    main()
