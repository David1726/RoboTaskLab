"""Fetch the pinned official LIBERO source archive; verify before extracting."""

import argparse
import hashlib
import urllib.request
import zipfile
from pathlib import Path

REVISION = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
ARCHIVE_SHA256 = "53b6776952c61959f75ac9a9e413f0cd383ab7a9b98f24a899e50b61ec315fb6"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(".research"))
    args = parser.parse_args()
    directory = args.out.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / f"LIBERO-{REVISION}.zip"
    target = directory / f"LIBERO-{REVISION}"
    if target.exists():
        raise SystemExit(
            "Target already exists; use that checkout or choose a fresh output directory."
        )
    if not archive.exists():
        url = f"https://codeload.github.com/Lifelong-Robot-Learning/LIBERO/zip/{REVISION}"
        with urllib.request.urlopen(url, timeout=120) as response, archive.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    if checksum != ARCHIVE_SHA256:
        raise SystemExit("Archive checksum mismatch; extraction refused.")
    with zipfile.ZipFile(archive) as bundle:
        for item in bundle.infolist():
            resolved = (directory / item.filename).resolve()
            if not resolved.is_relative_to(target):
                raise SystemExit("Archive member is outside the pinned source directory.")
        bundle.extractall(directory)
    print(target)


if __name__ == "__main__":
    main()
