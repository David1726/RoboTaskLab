"""Bounded rule-agent skill refinement; no model, training, or arbitrary code edits."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class PickPlaceSkill:
    grasp_offset_m: float = 0.12
    transport_height_m: float = 0.27

    def __post_init__(self):
        if not math.isfinite(self.grasp_offset_m) or not -0.01 <= self.grasp_offset_m <= 0.15:
            raise ValueError("grasp_offset_m outside the bounded demo domain")
        if (
            not math.isfinite(self.transport_height_m)
            or not 0.22 <= self.transport_height_m <= 0.35
        ):
            raise ValueError("transport_height_m outside the bounded demo domain")

    @property
    def identity(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def propose(skill: PickPlaceSkill, feedback: dict) -> tuple[PickPlaceSkill, dict] | None:
    """Only development telemetry may enter here; no held-out outcomes or oracle goal."""
    if feedback["failure"] != "empty_grasp":
        return None
    clearance = feedback["close_vertical_clearance_m"]
    xy_error = feedback["close_xy_error_m"]
    if not math.isfinite(clearance) or not math.isfinite(xy_error):
        raise ValueError("nonfinite feedback")
    if xy_error > 0.025 or clearance < 0.035:
        return None
    # Align close height with the observed object center, retaining a small margin.
    offset = max(-0.005, min(0.03, skill.grasp_offset_m - clearance + 0.01))
    candidate = replace(skill, grasp_offset_m=offset)
    if candidate.identity == skill.identity:
        return None
    return candidate, {
        "diagnosis": "empty grasp with good XY alignment and excessive vertical clearance",
        "method": "bounded rule-based grasp-height parameter revision",
        "observed_close_clearance_m": clearance,
        "observed_close_xy_error_m": xy_error,
        "parent_skill": skill.identity,
        "candidate_skill": candidate.identity,
        "before": asdict(skill),
        "after": asdict(candidate),
    }


def gate(baseline: list[dict], candidate: list[dict]) -> dict:
    """Compare exact initialization pairs; missing, reused, or changed evidence is invalid."""

    def index(rows):
        result = {}
        ids = set()
        for row in rows:
            key = (row["split"], row["seed"])
            if not isinstance(row["run_id"], str) or not row["run_id"].strip():
                raise ValueError("nonempty execution identity required")
            if key in result or row["run_id"] in ids or type(row["success"]) is not bool:
                raise ValueError("duplicate or invalid rollout evidence")
            result[key] = row
            ids.add(row["run_id"])
        return result

    base, cand = index(baseline), index(candidate)
    if not base or base.keys() != cand.keys() or {k[0] for k in base} != {"development", "heldout"}:
        raise ValueError("complete development and heldout pairs are required")
    if {r["run_id"] for r in baseline} & {r["run_id"] for r in candidate}:
        raise ValueError("candidate must have fresh execution identities")
    changes = []
    for key, before in base.items():
        after = cand[key]
        if before["initial_state_sha256"] != after["initial_state_sha256"]:
            raise ValueError("initial simulator states differ")
        if before["success"] != after["success"]:
            changes.append(
                {
                    "split": key[0],
                    "seed": key[1],
                    "change": "improvement" if after["success"] else "regression",
                }
            )
    improvement = any(r["split"] == "development" and r["change"] == "improvement" for r in changes)
    regressions = [r for r in changes if r["change"] == "regression"]
    # This tiny demo also requires all candidate test episodes to solve the task.
    accepted = improvement and not regressions and all(r["success"] for r in candidate)
    return {
        "accepted": accepted,
        "changes": changes,
        "baseline_successes": sum(r["success"] for r in baseline),
        "candidate_successes": sum(r["success"] for r in candidate),
        "episodes_per_version": len(baseline),
        "criterion": "development improvement, no pair regression, all candidate episodes pass",
    }
