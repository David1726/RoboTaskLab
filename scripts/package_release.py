"""Package the checked public repository, excluding runtime and archive directories."""

import argparse
import hashlib
import tomllib
import zipfile
from pathlib import Path

from check_repository import check_repository, repository_files

from embodied_agent_harness.spec import write_json


def main():
    root = Path(__file__).resolve().parents[1]
    version = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "version"
    ]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path(f"dist/embodied-agent-harness-{version}.zip")
    )
    args = parser.parse_args()
    checks = check_repository(root)
    files = repository_files(root)
    out = args.out.resolve()
    if out.exists():
        raise SystemExit("Release already exists; choose a new output path.")
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(files):
            bundle.write(path, "embodied-agent-harness/" + path.relative_to(root).as_posix())
    write_json(
        out.with_suffix(".manifest.json"),
        {
            "artifact": out.name,
            "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "files": [p.relative_to(root).as_posix() for p in sorted(files)],
            "repository_checks": checks,
        },
    )
    print(f"{out} ({out.stat().st_size} bytes, {len(files)} files)")


if __name__ == "__main__":
    main()
