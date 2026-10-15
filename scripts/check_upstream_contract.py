"""Static CLI check plus upstream audit of FIXTURES; never runs an agent/simulator."""

import argparse
import ast
import importlib.util
import json
from pathlib import Path

from embodied_agent_harness.roborsi import check_source, jobs
from embodied_agent_harness.spec import load_spec

AUDITED_REVISION = "e4e9024613e1060614964795206b64f15c422b06"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=Path("examples/libero-small.json"))
    parser.add_argument("--fixtures", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    check_source(source, AUDITED_REVISION)
    tree = ast.parse((source / "roborsi/cli/commands.py").read_text(encoding="utf-8-sig"))
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    available = {}
    for name in ("eval_suite", "eval_audit"):
        available[name] = {
            node.value
            for node in ast.walk(functions[name])
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.startswith("--")
        }
    spec = load_spec(args.spec)
    planned = jobs(spec, Path("python"), source, Path("out"))
    for job in planned:
        for field, function in (("command", "eval_suite"), ("audit_command", "eval_audit")):
            flags = {arg for arg in job[field] if arg.startswith("--")}
            if flags - available[function]:
                raise ValueError(f"unsupported upstream flags: {flags - available[function]}")
    # This file is stdlib-only. Loading it avoids installing RoboRSI's robot/model stack.
    module_spec = importlib.util.spec_from_file_location(
        "audited_roborsi_audit", source / "roborsi/evaluation/audit.py"
    )
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    results = []
    for manifest in sorted(args.fixtures.glob("*/*/seed-*/campaign.json")):
        report = module.audit_libero_short_suite(manifest.parent, check_media=True)
        if report["integrity_status"] != "pass" or report["campaign_status"] != "complete":
            raise ValueError(f"upstream rejected fixture shape: {report['errors']}")
        results.append(str(manifest.parent))
    if not results:
        raise ValueError("no fixture campaigns found")
    print(
        json.dumps(
            {
                "revision": AUDITED_REVISION,
                "cli_flags": "static_check_pass",
                "upstream_fixture_audits": len(results),
                "evidence_kind": "fixture",
                "simulator_episodes": 0,
                "model_calls": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
