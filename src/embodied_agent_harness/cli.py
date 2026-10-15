import argparse
import json
from pathlib import Path

from .evidence import compare, failure_packet, load_run
from .roborsi import execute, jobs
from .spec import load_spec, read_json, write_json


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Paired embodied skill regression evidence")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run", "inspect", "compare"):
        sub = commands.add_parser(name)
        sub.add_argument("--spec", type=Path, required=True)
        if name in {"plan", "run"}:
            sub.add_argument("--python", type=Path, required=True)
            sub.add_argument("--source", type=Path, required=True)
            sub.add_argument("--out", type=Path, required=True)
            if name == "run":
                sub.add_argument("--revision", required=True)
                sub.add_argument("--home", type=Path, required=True)
                sub.add_argument(
                    "--timeout",
                    type=int,
                    default=1800,
                    help="seconds per subprocess; not a monetary cap",
                )
                sub.add_argument(
                    "--execute",
                    action="store_true",
                    help="explicitly start configured model/simulator execution",
                )
        else:
            sub.add_argument("--baseline", type=Path, required=True)
            sub.add_argument("--out", type=Path, required=True)
            if name == "compare":
                sub.add_argument("--candidate", type=Path, required=True)
                sub.add_argument("--proposal", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        spec = load_spec(args.spec)
        if args.command in {"plan", "run"}:
            python, source, out = args.python.resolve(), args.source.resolve(), args.out.resolve()
            if args.command == "plan" or not args.execute:
                print(
                    json.dumps(
                        {"status": "not_executed", "jobs": jobs(spec, python, source, out)},
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                return 0
            home = args.home.resolve()
            if out.is_relative_to(source) or home.is_relative_to(source):
                raise ValueError("run output and ROBORSI_HOME must be outside the source checkout")
            result = execute(spec, python, source, args.revision, out, home, args.timeout)
            print(json.dumps({"receipt": str(out / "run.json"), "jobs": result["jobs"]}, indent=2))
            expected = sum(len(p["seeds"]) for p in spec["panels"].values())
            return (
                0
                if len(result["jobs"]) == expected
                and all(
                    j.get("status") == "finished"
                    and j.get("returncode") == 0
                    and j.get("audit_returncode") == 0
                    for j in result["jobs"]
                )
                else 2
            )
        baseline = load_run(args.baseline.resolve(), spec)
        if args.command == "inspect":
            write_json(args.out, failure_packet(baseline))
            print(f"Failure evidence packet: {args.out.resolve()}")
            return 2 if baseline["errors"] else 0
        candidate = load_run(args.candidate.resolve(), spec)
        result = compare(spec, baseline, candidate, read_json(args.proposal))
        write_json(args.out, result)
        print(
            json.dumps(
                {
                    "decision": result["decision"],
                    "reasons": result["reasons"],
                    "report": str(args.out.resolve()),
                },
                indent=2,
            )
        )
        if result["decision"].endswith("inconclusive"):
            return 2
        return 1 if result["decision"].endswith("reject") else 0
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
