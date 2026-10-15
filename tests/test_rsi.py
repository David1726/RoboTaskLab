from copy import deepcopy

import pytest

from embodied_agent_harness.rsi import PickPlaceSkill, gate, propose


def evidence(successes, prefix):
    return [
        {
            "split": split,
            "seed": seed,
            "run_id": f"{prefix}-{seed}",
            "success": success,
            "initial_state_sha256": f"initial-{seed}",
        }
        for (split, seed), success in zip(
            [("development", 0), ("heldout", 1), ("heldout", 2)], successes, strict=True
        )
    ]


def test_revision_requires_observed_empty_grasp_and_clearance():
    feedback = {
        "failure": "empty_grasp",
        "close_vertical_clearance_m": 0.10,
        "close_xy_error_m": 0.002,
    }
    candidate, explanation = propose(PickPlaceSkill(), feedback)
    assert candidate.grasp_offset_m == pytest.approx(0.03)
    assert explanation["parent_skill"] != explanation["candidate_skill"]
    assert propose(candidate, {**feedback, "failure": "placement_failed"}) is None
    assert propose(candidate, {**feedback, "close_xy_error_m": 0.10}) is None
    assert propose(candidate, {**feedback, "close_vertical_clearance_m": 0.01}) is None


def test_gate_accepts_complete_paired_improvement():
    result = gate(evidence([False, False, False], "b"), evidence([True, True, True], "c"))
    assert result["accepted"] and result["candidate_successes"] == 3


def test_gate_rejects_regression_despite_development_gain():
    result = gate(evidence([False, True, True], "b"), evidence([True, True, False], "c"))
    assert not result["accepted"]
    assert result["changes"][-1]["change"] == "regression"


def test_missing_pairs_and_changed_initial_states_cannot_pass():
    base, cand = evidence([False] * 3, "b"), evidence([True] * 3, "c")
    with pytest.raises(ValueError, match="complete"):
        gate(base, cand[:-1])
    modified = deepcopy(cand)
    modified[0]["initial_state_sha256"] = "changed"
    with pytest.raises(ValueError, match="states differ"):
        gate(base, modified)


def test_no_replayed_ids_or_duplicate_pair():
    rows = evidence([True] * 3, "b")
    with pytest.raises(ValueError, match="fresh"):
        gate(rows, rows)
    with pytest.raises(ValueError, match="duplicate"):
        gate(rows, rows + [rows[0]])


@pytest.mark.parametrize("offset", [float("nan"), float("inf"), -0.5, 0.5])
def test_parameter_edits_remain_bounded(offset):
    with pytest.raises(ValueError):
        PickPlaceSkill(grasp_offset_m=offset)
