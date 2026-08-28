import importlib.util
import io
from pathlib import Path
import tarfile

import pytest


SPEC = importlib.util.spec_from_file_location(
    "download_assets", Path("scripts/download_assets.py")
)
DOWNLOAD_ASSETS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DOWNLOAD_ASSETS)


def add_bytes(archive, name, value=b"data"):
    member = tarfile.TarInfo(name)
    member.size = len(value)
    archive.addfile(member, io.BytesIO(value))


def test_extract_archive_strips_wrapper_directory(tmp_path):
    source = tmp_path / "sample.tar.gz"
    with tarfile.open(source, "w:gz") as archive:
        add_bytes(archive, "testset/input/frame.tif")
    destination = tmp_path / "benchmark"
    DOWNLOAD_ASSETS.extract_archive(source, destination, strip_components=1)
    assert (destination / "input/frame.tif").read_bytes() == b"data"


def test_extract_archive_preserves_safe_relative_symlink(tmp_path):
    source = tmp_path / "benchmark.tar.gz"
    with tarfile.open(source, "w:gz") as archive:
        member = tarfile.TarInfo("testset/gt/frame.tif")
        member.type = tarfile.SYMTYPE
        member.linkname = "../gt_avg/frame.tif"
        archive.addfile(member)
        add_bytes(archive, "testset/gt_avg/frame.tif", b"reference")
    destination = tmp_path / "benchmark"
    DOWNLOAD_ASSETS.extract_archive(source, destination, strip_components=1)
    assert (destination / "gt/frame.tif").read_bytes() == b"reference"


def test_extract_archive_rejects_parent_traversal(tmp_path):
    source = tmp_path / "unsafe.tar.gz"
    with tarfile.open(source, "w:gz") as archive:
        add_bytes(archive, "wrapper/../../escape")
    with pytest.raises(ValueError, match="Unsafe archive path"):
        DOWNLOAD_ASSETS.extract_archive(source, tmp_path / "output", strip_components=1)
