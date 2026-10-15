"""Check the public repository files, local links and portable data."""

import argparse
import json
import re
from pathlib import Path

ROOT_FILES = (
    "README.md",
    "LICENSE",
    "pyproject.toml",
    ".gitignore",
    ".gitattributes",
    "CONTRIBUTING.md",
    "THIRD_PARTY.md",
    "MANIFEST.in",
    "CHANGELOG.md",
)
DIRECTORIES = (
    "src",
    "tests",
    "scripts",
    "examples",
    "participant",
    "benchmarks",
    "docs",
    "assets",
    ".github",
)
EXCLUDED = {"__pycache__", ".venv", ".cache", ".research", "runs", "dist", "build", ".git"}
HOST_PATH = re.compile(
    r"(?i)[A-Z]:[\\/](?:Users|Windows|Program Files)[\\/]|/(?:Users|home)/[\w.-]+/"
)
TEXT_SUFFIXES = {".md", ".py", ".json", ".jsonl", ".html", ".yml", ".toml"}


def repository_files(root):
    root = root.resolve()
    files = [root / name for name in ROOT_FILES]
    for directory in DIRECTORIES:
        for path in (root / directory).rglob("*"):
            if path.is_file() and not any(
                p in EXCLUDED or p.endswith(".egg-info") for p in path.relative_to(root).parts
            ):
                files.append(path)
    return sorted(files)


def reject_constant(value):
    raise ValueError(f"nonfinite JSON value: {value}")


def check_repository(root):
    root = root.resolve()
    files = repository_files(root)
    for path in files:
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"file outside repository: {path.name}")
        if path.name.startswith(".env") or path.suffix in {".pem", ".key"}:
            raise ValueError(f"private file in public selection: {path.name}")
        if path.stat().st_size > 100_000_000:
            raise ValueError(f"oversized repository file: {path.name}")
        if path.suffix not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8")
        if "\ufffd" in text or HOST_PATH.search(text):
            raise ValueError(f"host path or invalid encoding: {path.relative_to(root)}")
        if path.suffix == ".json":
            json.loads(text, parse_constant=reject_constant)
        elif path.suffix == ".jsonl":
            for line in text.splitlines():
                json.loads(line, parse_constant=reject_constant)
        elif path.suffix in {".md", ".html"}:
            pattern = (
                r"\[[^\]]*\]\(<?([^)>\s]+)>?\)"
                if path.suffix == ".md"
                else r'(?:href|src)="([^"]+)"'
            )
            for target in re.findall(pattern, text):
                if re.match(r"^(?:[a-z]+:|#)", target) or "${" in target:
                    continue
                target = target.split("#", 1)[0].split("?", 1)[0]
                destination = (path.parent / target).resolve()
                if not destination.is_relative_to(root) or not destination.exists():
                    raise ValueError(f"broken local link in {path.relative_to(root)}: {target}")
    return {
        "files": len(files),
        "markdown_files": sum(p.suffix == ".md" for p in files),
        "public_bytes": sum(p.stat().st_size for p in files),
        "status": "pass",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parents[1])
    args = p.parse_args()
    print(json.dumps(check_repository(args.root)))


if __name__ == "__main__":
    main()
