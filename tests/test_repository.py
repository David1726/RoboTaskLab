from pathlib import Path

import pytest

from embodied_agent_harness.benchmark import load_module


def test_public_selection_excludes_runtime_and_rejects_bad_links_and_host_paths(tmp_path):
    module = load_module(
        Path(__file__).resolve().parents[1] / "scripts/check_repository.py", "repository_check_test"
    )
    for name in module.ROOT_FILES:
        (tmp_path / name).write_text("", encoding="utf-8")
    private = tmp_path / ".cache" / "private.json"
    private.parent.mkdir()
    private.write_text("not public JSON", encoding="utf-8")
    assert private not in module.repository_files(tmp_path)
    assert module.check_repository(tmp_path)["status"] == "pass"
    (tmp_path / "README.md").write_text("[missing](docs/missing.md)", encoding="utf-8")
    with pytest.raises(ValueError, match="broken local link"):
        module.check_repository(tmp_path)
    host_path = "C:" + chr(92) + "Users" + chr(92) + "example" + chr(92) + "file"
    (tmp_path / "README.md").write_text(host_path, encoding="utf-8")
    with pytest.raises(ValueError, match="host path"):
        module.check_repository(tmp_path)
