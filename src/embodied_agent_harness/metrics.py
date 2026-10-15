"""Owner-side metric definitions; raw wrist loads are diagnostics by default."""

import math

METRIC_VERSION = "eah.metrics.v2"
SUCCESS_HOLD_STEPS = 5


def task_metric_contract(case):
    task = case["task_id"]
    return {
        "metric_version": METRIC_VERSION,
        "task_predicate": (
            "released; upright axis_z > 0.995; tip depth 18..34 mm; "
            "tip lateral offset < opening minus peg radius/half-side; "
            "positive cross-section clearance at engaged wall planes"
            if task.startswith("insert_")
            else "robosuite 1.4 NutAssembly placement predicate plus not grasped"
            if task.startswith("nut_")
            else "robosuite 1.4 Stack contact/lift/release predicate"
        ),
        "success_hold_control_steps": SUCCESS_HOLD_STEPS,
        "success_hold_sample_span_s": (SUCCESS_HOLD_STEPS - 1) / 20,
        "scope": "Fixed coarse simulation task specification, not manufacturing certification",
    }


def validate_force_limit(case):
    limit = case.get("force_limit_N")
    basis = case.get("force_limit_basis")
    if limit is not None and (
        type(limit) not in (int, float) or not math.isfinite(limit) or limit <= 0
    ):
        raise ValueError("invalid wrist force limit")
    if basis is not None and (not isinstance(basis, str) or not basis.strip() or limit is None):
        raise ValueError("force limit basis requires a positive limit and nonempty rationale")


def force_assessment(case, success, peak):
    """An explicit owner rationale enables a task constraint, never a safety claim."""
    validate_force_limit(case)
    limit = case.get("force_limit_N")
    basis = case.get("force_limit_basis")
    measured = peak is not None and math.isfinite(peak) and peak >= 0
    passes = peak <= limit if measured and limit is not None else None
    status = "task_declared" if basis else "exploratory" if limit is not None else "diagnostic_only"
    return {
        "metric_version": METRIC_VERSION,
        "force_limit_N": limit,
        "force_limit_basis": basis,
        "force_metric_status": status,
        "force_limit_pass": passes,
        "force_qualified_success": bool(success and passes) if basis and measured else None,
        "force_measurement": {
            "source": "robosuite raw end-effector force sensor",
            "frame": "eef_sensor",
            "sample_rate_hz": 20,
            "statistic": "maximum sampled Euclidean force norm over executed control steps",
            "compensated": False,
            "interpretation": "Raw simulated wrist load; includes dynamic/load effects. "
            "Not isolated contact force, continuous-time peak, or safety certification.",
        },
    }


def force_summary(rows):
    """Missing measurements and uncalibrated historical thresholds stay unassessed."""
    peaks = [r["peak_wrist_force_N"] for r in rows if r.get("peak_wrist_force_N") is not None]
    assessed = [r for r in rows if r.get("force_metric_status") == "task_declared"]
    comparable = (
        bool(rows)
        and len(assessed) == len(rows)
        and all(r.get("force_qualified_success") is not None for r in rows)
    )
    specifications = {(r.get("force_limit_N"), r.get("force_limit_basis")) for r in assessed}
    return {
        "force_qualified_success_rate": (
            sum(r["force_qualified_success"] for r in assessed) / len(rows)
            if comparable and len(specifications) == 1
            else None
        ),
        "force_assessed_episodes": len(assessed),
        "force_measured_episodes": len(peaks),
        "mean_sampled_peak_wrist_force_N": sum(peaks) / len(peaks) if peaks else None,
        "max_sampled_peak_wrist_force_N": max(peaks) if peaks else None,
        "force_metric_note": "Raw 20Hz wrist-load diagnostic. Qualification requires an "
        "explicit comparable task specification; legacy illustrative limits are unassessed.",
    }
