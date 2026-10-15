# Contributing

Install `.[dev]` for protocol tests or `.[dev,manufacturing]` for simulation.
Use Python 3.12 and run:

```sh
python -m pip install numpy==1.26.4 imageio imageio-ffmpeg pillow
python -m pytest -q
ruff check src tests scripts examples participant
ruff format --check src tests scripts examples participant
python scripts/check_repository.py
```

The media dependencies enable the episode-recording tests. Without them, that
test module is skipped. Simulation tests additionally need the manufacturing extra.

Keep interface changes documented in [the benchmark protocol](docs/benchmark.md).
Include fresh simulator evidence for changes claimed to improve robotic behavior.
Report failed episodes and implementation errors, and distinguish state-based
control from visual perception. Changes to task predicates need an explicit
metric version and source recollection.

Published panels are reference examples. Generate and freeze separate development
and evaluation panels before tuning. Record model versions, dependency versions,
budgets and all paired outcomes for transfer experiments.

## Repository contents

Commit source, templates, panels, tests and portable reference evidence. Keep
virtual environments, caches, raw runs, upstream checkouts and credentials out
of the repository. `.gitignore` and the packaging allowlist exclude those files.
Run `scripts/check_repository.py` to check links, host information and public
file selection before release.

To build a clean source export:

```sh
python scripts/package_release.py --out dist/embodied-agent-harness-source.zip
```

The ZIP contains one repository directory. Extract it before initializing Git.

## Publish to GitHub

Create an empty GitHub repository, then run these commands inside the extracted
project. Replace the remote URL with the repository's actual URL.

```sh
git init
git add .
git commit -m "Initial release"
git branch -M main
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
```

The included GitHub Actions workflow runs protocol tests and linting; it does not
run simulator rendering. Attach large release archives as release assets rather
than committing `dist/`. Preserve the license and third-party attribution.
