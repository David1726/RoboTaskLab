import importlib.util
import json
from pathlib import Path

import pytest

from embodied_agent_harness.cli import main
from embodied_agent_harness.evidence import compare, failure_packet, load_run
from embodied_agent_harness.roborsi import jobs
from embodied_agent_harness.spec import load_spec, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


def test_failed_backend_source_audit_cannot_return_success(tmp_path, monkeypatch):
    from embodied_agent_harness import cli

    spec = load_spec(ROOT / "examples/libero-small.json")
    count = sum(len(panel["seeds"]) for panel in spec["panels"].values())
    monkeypatch.setattr(
        cli,
        "execute",
        lambda *args: {
            "jobs": [
                {"status": "failed", "returncode": 0, "audit_returncode": 0} for _ in range(count)
            ]
        },
    )
    assert (
        main(
            [
                "run",
                "--spec",
                str(ROOT / "examples/libero-small.json"),
                "--python",
                "python",
                "--source",
                str(tmp_path / "source"),
                "--out",
                str(tmp_path / "out"),
                "--revision",
                "a" * 40,
                "--home",
                str(tmp_path / "backend-home"),
                "--execute",
            ]
        )
        == 2
    )


fixture_module = importlib.util.spec_from_file_location(
    "make_fixture", ROOT / "examples/make_fixture.py"
)
fixtures = importlib.util.module_from_spec(fixture_module)
fixture_module.loader.exec_module(fixtures)


@pytest.fixture
def scenario(tmp_path):
    spec = load_spec(ROOT / "examples/libero-small.json")
    base, candidate = tmp_path / "base", tmp_path / "candidate"
    fixtures.make_run(base, spec, fixtures.BASE_REVISION)
    fixtures.make_run(candidate, spec, fixtures.CANDIDATE_REVISION, improved=True)
    proposal = {
        "schema": "eah.proposal.v1",
        "baseline_revision": fixtures.BASE_REVISION,
        "candidate_revision": fixtures.CANDIDATE_REVISION,
        "diagnosis_hypothesis": "fixture hypothesis",
        "change": "fixture change",
        "rationale": "fixture rationale",
        "targets": [{"task": "libero_object/0", "seed": 21}],
    }
    return spec, base, candidate, proposal


def journal(root, panel="development", seed=21):
    return root / panel / f"seed-{seed}" / "episodes.jsonl"


def mutate_row(root, **changes):
    path = journal(root)
    row = json.loads(path.read_text())
    row.update(changes)
    path.write_text(json.dumps(row) + "\n")


def result(scenario):
    spec, base, candidate, proposal = scenario
    return compare(spec, load_run(base, spec), load_run(candidate, spec), proposal)


def test_paired_improvement_is_fixture_only(scenario):
    report = result(scenario)
    assert report["decision"] == "fixture_accept_for_review"
    assert report["panels"]["development"]["improvements"] == 1
    assert report["panels"]["heldout"]["paired"] == 2
    assert report["evidence_sha256"]


def test_heldout_regression_rejects_even_if_development_improves(scenario):
    spec, base, candidate, proposal = scenario
    regressed = candidate.parent / "regressed"
    fixtures.make_run(regressed, spec, fixtures.CANDIDATE_REVISION, improved=True, regressed=True)
    report = compare(spec, load_run(base, spec), load_run(regressed, spec), proposal)
    assert report["decision"] == "fixture_reject"
    assert report["panels"]["development"]["improvements"] == 1
    assert report["panels"]["heldout"]["regressions"] == 2


def test_no_improvement_rejects(scenario):
    mutate_row(scenario[2], success=False, verdict="failure")
    assert result(scenario)["decision"] == "fixture_reject"


def test_missing_candidate_seed_is_inconclusive(scenario):
    journal(scenario[2], "heldout", 32).unlink()
    report = result(scenario)
    assert report["decision"] == "fixture_inconclusive"
    assert report["panels"]["heldout"]["paired"] == 1


@pytest.mark.parametrize("verdict", ["infra", "implementation_error"])
def test_nonterminal_errors_are_not_task_failure_or_success(scenario, verdict):
    mutate_row(scenario[2], verdict=verdict, success=None, status="incomplete")
    report = result(scenario)
    assert report["decision"] == "fixture_inconclusive"
    assert report["panels"]["development"]["paired"] == 1


def test_infra_then_terminal_can_recover(scenario):
    path = journal(scenario[2])
    terminal = json.loads(path.read_text())
    infra = {**terminal, "verdict": "infra", "success": None, "status": "incomplete"}
    terminal["attempt"] = 2
    path.write_text(json.dumps(infra) + "\n" + json.dumps(terminal) + "\n")
    assert result(scenario)["decision"] == "fixture_accept_for_review"


@pytest.mark.parametrize(
    "change",
    [
        {"success": "true"},
        {"verdict": "unknown"},
        {"seed": 99},
        {"seed": True},
        {"run_mode": "evolve"},
        {"backend": "other"},
        {"task_key": "libero_object/99"},
        {"run_id": ""},
        {"attempt": True},
    ],
)
def test_invalid_journal_blocks_gate(scenario, change):
    mutate_row(scenario[2], **change)
    with pytest.raises(ValueError):
        result(scenario)


def test_duplicate_terminal_and_post_terminal_attempts_block_gate(scenario):
    path = journal(scenario[2])
    row = json.loads(path.read_text())
    row["attempt"] = 2
    with path.open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="post-terminal"):
        result(scenario)


def test_manifest_model_drift_blocks_gate(scenario):
    manifest = scenario[2] / "development/seed-21/campaign.json"
    data = read_json(manifest)
    data["models"]["planner"] = "different-model"
    write_json(manifest, data)
    with pytest.raises(ValueError, match="configuration drift"):
        result(scenario)


def test_runtime_drift_between_revisions_blocks_gate(scenario):
    for manifest in scenario[2].glob("*/seed-*/campaign.json"):
        data = read_json(manifest)
        data["runtime"]["libero_commit"] = "different-assets-or-simulator"
        write_json(manifest, data)
    with pytest.raises(ValueError, match="runtime mismatch"):
        result(scenario)


def test_summary_cannot_override_journal(scenario):
    write_json(scenario[2] / "development/seed-21/summary.json", {"success_rate": 1.0})
    mutate_row(scenario[2], verdict="failure", success=False)
    assert result(scenario)["decision"] == "fixture_reject"


def test_reused_baseline_execution_is_rejected(scenario):
    row = json.loads(journal(scenario[1]).read_text())
    mutate_row(scenario[2], run_id=row["run_id"])
    with pytest.raises(ValueError, match="reuses baseline"):
        result(scenario)


def test_fixture_cannot_silently_become_simulator_acceptance(scenario):
    spec, base, candidate, proposal = scenario
    for root in (base, candidate):
        receipt = read_json(root / "run.json")
        receipt["evidence_kind"] = "simulator_execution"
        write_json(root / "run.json", receipt)
    assert result(scenario)["decision"] == "inconclusive"


def test_mixed_fixture_and_live_evidence_blocks_gate(scenario):
    receipt = read_json(scenario[2] / "run.json")
    receipt["evidence_kind"] = "simulator_execution"
    write_json(scenario[2] / "run.json", receipt)
    with pytest.raises(ValueError, match="cannot be mixed"):
        result(scenario)


def test_proposal_cannot_target_heldout_or_success(scenario):
    scenario[3]["targets"] = [{"task": "libero_object/1", "seed": 31}]
    with pytest.raises(ValueError, match="development failures"):
        result(scenario)


def test_failure_packet_excludes_heldout_for_diagnosis(scenario):
    spec, base, _, _ = scenario
    packet = failure_packet(load_run(base, spec))
    assert len(packet["development_failures"]) == 1
    assert packet["development_failures"][0]["task"] == "libero_object/0"
    assert packet["evidence_kind"] == "fixture"


def test_overlapping_tasks_or_duplicate_seeds_are_invalid(tmp_path):
    path = tmp_path / "spec.json"
    spec = load_spec(ROOT / "examples/libero-small.json")
    spec["panels"]["heldout"]["tasks"] = spec["panels"]["development"]["tasks"]
    write_json(path, spec)
    with pytest.raises(ValueError, match="disjoint"):
        load_spec(path)
    spec["panels"]["heldout"]["tasks"] = ["libero_object/1"]
    spec["panels"]["heldout"]["seeds"] = [31, 31]
    write_json(path, spec)
    with pytest.raises(ValueError, match="duplicate seeds"):
        load_spec(path)


def test_plan_uses_one_seed_campaigns_and_does_not_execute(tmp_path, capsys, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run must not execute a subprocess")

    monkeypatch.setattr("subprocess.run", forbidden)
    out = tmp_path / "not-created"
    code = main(
        [
            "run",
            "--spec",
            str(ROOT / "examples/libero-small.json"),
            "--python",
            "python.exe",
            "--source",
            "source",
            "--out",
            str(out),
            "--home",
            "home",
            "--revision",
            fixtures.BASE_REVISION,
        ]
    )
    assert code == 0 and not out.exists()
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "not_executed"
    assert len(output["jobs"]) == 4
    for job in output["jobs"]:
        command = job["command"]
        assert command[command.index("--pass-at") + 1] == "1"
        assert command[command.index("--run-mode") + 1] == "frozen"
        assert "--require-complete" in job["audit_command"]


def test_cli_compare_writes_reviewable_report(scenario, tmp_path, capsys):
    spec, base, candidate, proposal = scenario
    spec_path, proposal_path, out = (
        tmp_path / "spec.json",
        tmp_path / "proposal.json",
        tmp_path / "report.json",
    )
    write_json(spec_path, spec)
    write_json(proposal_path, proposal)
    assert (
        main(
            [
                "compare",
                "--spec",
                str(spec_path),
                "--baseline",
                str(base),
                "--candidate",
                str(candidate),
                "--proposal",
                str(proposal_path),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert read_json(out)["decision"] == "fixture_accept_for_review"
    assert "fixture_accept_for_review" in capsys.readouterr().out


def test_command_arguments_preserve_paths_with_spaces(scenario):
    commands = jobs(
        scenario[0],
        Path("C:/Python With Spaces/python.exe"),
        Path("C:/RoboRSI source"),
        Path("C:/Run Outputs"),
    )
    assert (
        commands[0]["command"][0] == "C:\\Python With Spaces\\python.exe"
        if (__import__("os").name == "nt")
        else "C:/Python With Spaces/python.exe"
    )
    assert commands[0]["cwd"] == str(Path("C:/RoboRSI source"))
