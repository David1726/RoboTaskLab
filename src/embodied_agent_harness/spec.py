import json
import math
import os
import re
import tempfile
from pathlib import Path


def strict_json(text):
    def invalid_constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("nonfinite JSON number")
        return result

    return json.loads(
        text,
        parse_constant=invalid_constant,
        object_pairs_hook=unique_object,
        parse_float=finite_float,
    )


def read_json(path: Path, *, legacy_nonfinite=False) -> dict:
    text = path.read_text(encoding="utf-8")
    value = json.loads(text) if legacy_nonfinite else strict_json(text)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict) -> None:
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_spec(path: Path) -> dict:
    spec = read_json(path)
    expected = {
        "schema",
        "backend",
        "atomic",
        "models",
        "reasoning_effort",
        "tool_budget",
        "infra_retries",
        "panels",
    }
    if set(spec) != expected or spec["schema"] != "eah.experiment.v1":
        raise ValueError("unknown experiment schema or fields")
    if spec["backend"] not in {"libero", "libero-pro"}:
        raise ValueError("v0.1 supports RoboRSI libero/libero-pro only")
    if not isinstance(spec["atomic"], str) or not spec["atomic"]:
        raise ValueError("atomic must be a nonempty skill name")
    models = spec["models"]
    if (
        not isinstance(models, dict)
        or set(models) != {"planner", "engineer", "reviewer"}
        or not all(isinstance(v, str) and v.strip() for v in models.values())
    ):
        raise ValueError("explicit planner, engineer, reviewer models are required")
    if spec["reasoning_effort"] not in {"low", "medium", "high"}:
        raise ValueError("reasoning_effort must be low, medium, or high")
    for key, minimum in (("tool_budget", 1), ("infra_retries", 0)):
        if type(spec[key]) is not int or spec[key] < minimum:
            raise ValueError(f"invalid {key}")
    panels = spec["panels"]
    if not isinstance(panels, dict) or set(panels) != {"development", "heldout"}:
        raise ValueError("development and heldout panels required")
    for panel in panels.values():
        if not isinstance(panel, dict) or set(panel) != {"tasks", "seeds"}:
            raise ValueError("panels require tasks and seeds only")
        tasks, seeds = panel["tasks"], panel["seeds"]
        if (
            not isinstance(tasks, list)
            or not tasks
            or not all(
                isinstance(t, str)
                and re.fullmatch(r"libero_(spatial|object|goal)(_(task|object|swap|lan))?/\d+", t)
                for t in tasks
            )
            or len(tasks) != len(set(tasks))
        ):
            raise ValueError("invalid or duplicate short tasks")
        if (
            not isinstance(seeds, list)
            or not seeds
            or any(type(s) is not int or s < 0 for s in seeds)
            or len(seeds) != len(set(seeds))
        ):
            raise ValueError("invalid or duplicate seeds")
    if set(panels["development"]["tasks"]) & set(panels["heldout"]["tasks"]):
        raise ValueError("heldout tasks must be disjoint from development tasks")
    return spec


def expected_campaign(spec: dict, panel: str, seed: int) -> dict:
    return {
        "schema": "roborsi.libero_short_campaign.v1",
        "backend": spec["backend"],
        "atomic": spec["atomic"],
        "task_keys": spec["panels"][panel]["tasks"],
        "pass_at": 1,
        "seed_start": seed,
        "workers": 1,
        "tool_budget": spec["tool_budget"],
        "infra_retries": spec["infra_retries"],
        "models": spec["models"],
        "reasoning_effort": spec["reasoning_effort"],
        "atomic_compound_enabled": True,
        "agent_mode": "roborsi",
        "run_mode": "frozen",
    }
