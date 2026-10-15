"""Apply three host compatibility edits to a dedicated robosuite 1.4 installation.

No dynamics, controller, reward, or success predicates are changed. This is for
local scene previews, not an exact reproduction of RoboRSI's Linux evaluation.
"""

import importlib.metadata
import sys
from pathlib import Path


def main():
    if sys.platform != "win32":
        raise SystemExit("Windows preview setup only")
    dist = importlib.metadata.distribution("robosuite")
    if dist.version != "1.4.0":
        raise SystemExit("Expected robosuite==1.4.0")
    root = Path(dist.locate_file("robosuite")).resolve()
    edits = {
        root / "macros.py": [
            ("MUJOCO_GPU_RENDERING = True", "MUJOCO_GPU_RENDERING = False"),
            ('FILE_LOGGING_LEVEL = "DEBUG"', "FILE_LOGGING_LEVEL = None"),
        ],
        root / "utils/binding_utils.py": [
            (
                '    ctypes.WinDLL(os.path.join(os.path.dirname(__file__), "mujoco.dll"))',
                "    pass  # Native mujoco wheel already loads its packaged DLL on Windows.",
            ),
        ],
    }
    for path, replacements in edits.items():
        original = path.read_text(encoding="utf-8")
        modified = original
        for old, new in replacements:
            if old in modified:
                modified = modified.replace(old, new, 1)
            elif new not in modified:
                raise SystemExit(f"Unexpected upstream content: {path}")
        if modified != original:
            backup = path.with_suffix(path.suffix + ".preview-original")
            if not backup.exists():
                backup.write_text(original, encoding="utf-8")
            path.write_text(modified, encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
