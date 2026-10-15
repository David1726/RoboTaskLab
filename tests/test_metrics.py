"""Metric validity and failure modes, independent of participant implementation."""

import math

import pytest

from embodied_agent_harness.manufacturing import insertion_success
from embodied_agent_harness.metrics import force_assessment, force_summary, validate_force_limit
from embodied_agent_harness.physical_icl import physical_summary
from embodied_agent_harness.transfer import paired_comparison


def test_raw_force_and_legacy_example_never_create_quality_claims():
    for case in [{}, {"force_limit_N": 100}]:
        result = force_assessment(case, True, 30)
        assert result["force_qualified_success"] is None
    old = {"peak_wrist_force_N": 30, "force_qualified_success": True}
    assert force_summary([old])["force_qualified_success_rate"] is None


def test_declared_force_constraint_requires_measurement_and_task_success():
    case = {"force_limit_N": 25, "force_limit_basis": "experimental task specification A"}
    assert force_assessment(case, True, 20)["force_qualified_success"] is True
    assert force_assessment(case, True, 26)["force_qualified_success"] is False
    assert force_assessment(case, False, 20)["force_qualified_success"] is False
    assert force_assessment(case, True, None)["force_qualified_success"] is None
    rows = [dict(peak_wrist_force_N=20, **force_assessment(case, True, 20))]
    assert force_summary(rows)["force_qualified_success_rate"] == 1
    rows.append({"peak_wrist_force_N": None})
    assert force_summary(rows)["force_qualified_success_rate"] is None


@pytest.mark.parametrize(
    "case",
    [
        {"force_limit_N": float("nan")},
        {"force_limit_N": -1},
        {"force_limit_basis": "missing limit"},
        {"force_limit_N": 20, "force_limit_basis": " "},
    ],
)
def test_invalid_force_specifications_are_rejected(case):
    with pytest.raises(ValueError, match="force"):
        validate_force_limit(case)


def test_square_footprint_rejects_yaw_even_with_center_and_depth_correct():
    args = ([0, 0, 0.876], [0, 0, 1], [0, 0, 0.852], 0.05, 0.01, 0.012, True)
    identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    q = math.sqrt(0.5)
    yaw45 = [[q, -q, 0], [q, q, 0], [0, 0, 1]]
    assert insertion_success(*args, "square", identity)
    assert not insertion_success(*args, "square", yaw45)
    assert insertion_success(*args, "round", yaw45)


def test_world_coverage_mismatch_cannot_create_a_macro_gain():
    row = {"case": {"world_id": "w0", "task_family": "insertion"}, "success": True}
    with pytest.raises(ValueError, match="coverage"):
        physical_summary({"no_context": [row], "source_context": []})


def test_pairs_require_same_environment_and_metric_version():
    from copy import deepcopy

    row = {
        "case": {"id": "t", "task_id": "insert_round"},
        "success": True,
        "initial_state_sha256": "same",
        "init_asset_sha256": "modelA",
        "metric_version": "v2",
        "steps": 20,
        "skill_calls": 1,
    }
    other = deepcopy(row)
    other["init_asset_sha256"] = "modelB"
    with pytest.raises(ValueError, match="environment"):
        paired_comparison({"no_context": [row], "source_context": [other]})
    other = deepcopy(row)
    other["metric_version"] = "v1"
    with pytest.raises(ValueError, match="metric"):
        paired_comparison({"no_context": [row], "source_context": [other]})


def test_source_metric_version_drift_requires_new_collection():
    from embodied_agent_harness.physical_icl import validate_protocol

    with pytest.raises(ValueError, match="fresh source"):
        validate_protocol(
            {"evaluation_track": "physical_icl"},
            {"metric_versions": ["legacy_v1"]},
            expected_metric_version="eah.metrics.v2",
        )


def test_upside_down_diagnostic_and_legacy_report_remain_valid_json():
    import json

    from embodied_agent_harness.manufacturing import insertion_fit_margin
    from embodied_agent_harness.report import display_values

    assert (
        insertion_fit_margin([0, 0, 0.9], [0, 0, -1], [0, 0, 0.852], 0.01, 0.014, "round") is None
    )
    raw = {"evaluation": {"cross_section_margin_m": -math.inf}}
    shown = display_values(raw)
    assert (
        json.loads(json.dumps(shown, allow_nan=False))["evaluation"]["cross_section_margin_m"]
        is None
    )
    assert raw["evaluation"]["cross_section_margin_m"] == -math.inf


@pytest.mark.parametrize(
    "worlds,status", [(2, "insufficient_worlds"), (6, "degenerate_resampling")]
)
def test_small_or_degenerate_world_sample_has_no_misleading_confidence_interval(worlds, status):
    rows = [
        {"case": {"world_id": str(w), "task_family": "insertion"}, "success": True}
        for w in range(worlds)
    ]
    summary = physical_summary({"no_context": rows, "source_context": rows})
    ci = summary["source_context"]["paired_macro_delta_ci95"]
    assert ci["status"] == status
    assert ci["lower"] is None and ci["upper"] is None
