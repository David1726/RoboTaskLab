from copy import deepcopy
from types import SimpleNamespace

import pytest

from embodied_agent_harness.benchmark import load_panel
from embodied_agent_harness.manufacturing import ManufacturingAdapter, insertion_success
from embodied_agent_harness.memory import content_id
from embodied_agent_harness.physical_icl import (
    physical_summary,
    select_physical_context,
    validate_protocol,
)
from embodied_agent_harness.spec import write_json


def physical_fixture():
    bank = {"source_tasks": ["nut_round"], "experiences": []}
    for i, gain in enumerate([0.6, 1.4]):
        physics = {"actuation_gain": gain, "mass_scale": 1, "friction": 0.6}
        bank["experiences"].append(
            {
                "task_id": "nut_round",
                "task_family": "nut_assembly",
                "world_id": f"w{i}",
                "source_layout": {"backend": "manufacturing", "seed": 24, "opening_m": None},
                "physics_sha256": content_id(physics),
                "instruction": "assemble nut",
                "trajectory_excerpt": [],
                "calls": [
                    {
                        "decision": {"name": "probe_cartesian"},
                        "feedback": {"status": "measured", "response_gain": gain},
                    },
                    {"decision": {"name": "assemble_nut"}, "feedback": {"status": "completed"}},
                ],
            }
        )
    case = {
        "id": "target",
        "task_id": "insert_round",
        "task_family": "insertion",
        "world_id": "w0",
        "physics": {"actuation_gain": 0.6, "mass_scale": 1, "friction": 0.6},
        "split": "evaluation",
    }
    return bank, case


def test_cross_family_and_matching_physics_are_required():
    bank, case = physical_fixture()
    panel = {"evaluation_track": "physical_icl", "cases": [case]}
    validate_protocol(panel, bank)
    case["task_family"] = "nut_assembly"
    with pytest.raises(ValueError, match="families overlap"):
        validate_protocol(panel, bank)
    case["task_family"] = "insertion"
    case["physics"]["friction"] = 0.9
    with pytest.raises(ValueError, match="matching physical"):
        validate_protocol(panel, bank)


def test_unrelated_context_changes_mechanism_and_hides_condition():
    bank, case = physical_fixture()
    matched = select_physical_context(bank, case, "source_context", 1)
    unrelated = select_physical_context(bank, case, "unrelated_physics", 1)
    assert matched["mode"] == unrelated["mode"] == "physical_context"
    assert matched["experiences"][0]["world_id"] == "w0"
    assert unrelated["experiences"][0]["world_id"] == "w1"
    with pytest.raises(ValueError, match="insufficient"):
        select_physical_context(bank, case, "source_context", 2)


def test_counterfactual_sources_cannot_change_layout_with_physics():
    bank, case = physical_fixture()
    bank["experiences"][1]["source_layout"]["seed"] = 25
    with pytest.raises(ValueError, match="layouts/geometry"):
        validate_protocol({"evaluation_track": "physical_icl", "cases": [case]}, bank)


def test_actor_frame_path_is_anonymous_and_cleaned(tmp_path, monkeypatch):
    from embodied_agent_harness.benchmark import SkillContext

    monkeypatch.chdir(tmp_path)
    output = tmp_path / "source_context" / "world-0" / "case-0"
    output.mkdir(parents=True)
    context = SkillContext(None, {}, "instruction", output, 10, None)
    path = context._actor_frame
    assert "source_context" not in str(path) and "world-0" not in str(path)
    path.write_bytes(b"unit test frame")
    context.save()
    assert not path.exists()


@pytest.mark.parametrize(
    "instruction,task",
    [
        ("把圆销插入圆孔并松开", "insert_round"),
        ("将方形销插入方孔并松开", "insert_square"),
        ("把小块堆叠到大块上并松开", "stack"),
    ],
)
def test_language_binding_accepts_only_defined_manufacturing_goals(instruction, task):
    from embodied_agent_harness.language import manufacturing_instruction_task

    assert manufacturing_instruction_task(instruction) == task
    with pytest.raises(ValueError, match="unsupported"):
        manufacturing_instruction_task("不要" + instruction)


def test_semantic_and_corrupted_controls_preserve_original_evidence():
    bank, case = physical_fixture()
    before = deepcopy(bank)
    language = select_physical_context(bank, case, "language_only", 1)
    assert "calls" not in language["experiences"][0]
    shuffled = select_physical_context(bank, case, "shuffled_feedback", 1)
    assert "response_gain" not in shuffled["experiences"][0]["calls"][0]["feedback"]
    assert bank == before


def test_insertion_oracle_rejects_hover_wrong_hole_tilt_and_held_object():
    args = ([0, 0, 0.876], [0, 0, 1], [0, 0, 0.852], 0.05, 0.01, 0.014, True)
    assert insertion_success(*args)
    assert not insertion_success([0, 0, 0.96], *args[1:])
    assert not insertion_success([0.02, 0, 0.876], *args[1:])
    assert not insertion_success(args[0], [1, 0, 0], *args[2:])
    assert not insertion_success(*args[:-1], False)


def test_physics_parameters_are_not_part_of_participant_observations():
    class Vector(list):
        def tolist(self):
            return list(self)

    adapter = ManufacturingAdapter.__new__(ManufacturingAdapter)
    adapter.case = {
        "task_id": "insert_round",
        "split": "evaluation",
        "physics": {"actuation_gain": 0.6},
    }
    adapter.env = SimpleNamespace(
        objects_dict={"peg": None, "hole": None},
        obj_body_id={"peg": 0, "hole": 1},
        robots=[SimpleNamespace(ee_force=Vector([0, 0, 0]), ee_torque=Vector([0, 0, 0]))],
    )
    adapter.sim = SimpleNamespace(data=SimpleNamespace(body_xquat=[Vector([1, 0, 0, 0])] * 2))
    observation = adapter.public_observation()
    assert (
        not {"physics", "world_id", "actuation_gain", "mass_scale", "friction"} & observation.keys()
    )


def test_world_family_macro_does_not_overweight_more_insertion_shapes():
    cases = [
        ("w0", "insertion"),
        ("w0", "insertion"),
        ("w0", "stacking"),
        ("w1", "insertion"),
        ("w1", "stacking"),
    ]

    def rows(values):
        return [
            {"case": {"world_id": w, "task_family": f}, "success": v}
            for (w, f), v in zip(cases, values, strict=True)
        ]

    summary = physical_summary(
        {"no_context": rows([0, 0, 1, 0, 1]), "source_context": rows([1, 1, 1, 1, 0])}
    )
    assert summary["source_context"]["world_family_macro_success"] == 0.75
    assert summary["source_context"]["paired_macro_delta"] == 0.25
    assert summary["source_context"]["paired_macro_delta_ci95"]["world_clusters"] == 2


def test_unknown_track_and_invalid_physical_intervention_are_rejected(tmp_path):
    import json
    from pathlib import Path

    panel = json.loads(Path("benchmarks/manufacturing-picl-v2.json").read_text())
    case = panel["cases"][0]
    case["physics"]["mass_scale"] = float("nan")
    path = tmp_path / "panel.json"
    path.write_text(json.dumps(panel), encoding="utf-8")
    with pytest.raises(ValueError, match="physics|nonfinite"):
        load_panel(path)
    case["physics"]["mass_scale"] = 1
    panel["evaluation_track"] = "physical_icl_typo"
    write_json(path, panel)
    with pytest.raises(ValueError, match="explicit"):
        load_panel(path)


def test_generator_keeps_source_task_geometry_and_layout_in_counterfactual_worlds():
    from pathlib import Path

    from embodied_agent_harness.benchmark import load_module

    module = load_module(Path("scripts/generate_manufacturing.py"), "manufacturing_generator_test")
    panel = module.generate(42, 3, 2, "random", "physical_icl")
    sources = [c for c in panel["cases"] if c["split"] == "development"]
    assert len({(c["task_id"], c["layout_seed"], c["opening_m"]) for c in sources}) == 1
    assert len({content_id(c["physics"]) for c in sources}) == 3
    assert not {c["task_family"] for c in sources} & {
        c["task_family"] for c in panel["cases"] if c["split"] == "evaluation"
    }
