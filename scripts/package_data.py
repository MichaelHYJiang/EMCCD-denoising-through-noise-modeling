#!/usr/bin/env python3
"""Create tar archives for external dataset hosting."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("release_archives"))
    args = parser.parse_args()
    if not args.source.is_dir():
        raise SystemExit(f"Not a directory: {args.source}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / f"{args.name}.tar.gz"
    with tarfile.open(archive, "w:gz", compresslevel=6) as tar:
        for path in sorted(args.source.rglob("*")):
            info = tar.gettarinfo(path, arcname=Path(args.name) / path.relative_to(args.source))
            info.uid = info.gid = 0; info.uname = info.gname = ""; info.mtime = 0
            if path.is_file():
                with path.open("rb") as handle:
                    tar.addfile(info, handle)
            else:
                tar.addfile(info)
    hasher = hashlib.sha256()
    with archive.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    sha = hasher.hexdigest()
    print(json.dumps({"path": str(archive), "size": archive.stat().st_size, "sha256": sha}, indent=2))


if __name__ == "__main__":
    main()
