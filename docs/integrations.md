# Integrations

## External agent process

`JsonProcessAgent` runs a JSON argv array without a shell. Each decision starts
one worker, sends one UTF-8 request on stdin and expects one response on stdout.
Each request carries the explicit state from the previous decision. The default
timeout is 60 seconds.

```json
{
  "schema": "eah.agent_request.v1",
  "observation": {
    "instruction": "Insert the round peg into the round hole and release it.",
    "remaining_steps": 600,
    "context": {"mode": "no_context", "experiences": []}
  },
  "feedback": null,
  "state": null
}
```

The example shows only some observation fields. The worker receives the full
[participant observation](benchmark.md#participant-interface).

```json
{
  "schema": "eah.agent_response.v1",
  "decision": {"type": "skill", "name": "insert", "arguments": {"motion_scale": 1.0}},
  "state": {"attempt": 1}
}
```

Use `--agent examples/process_agent.py` with `eah-bench` or `eah-transfer`.
The default worker is a local rule example. Set `EAH_AGENT_COMMAND` to a JSON
array such as `["python", "my_worker.py"]` to supply an inference worker.
Write logs to stderr and only the response JSON to stdout. Remote inference
workers must read and encode `image_path`; remote services cannot access it directly.

For a frozen-model experiment, record model revision/weight hash, prompt,
sampling settings, token limits, requests and costs. The bridge does not pin
remote model aliases or external dependencies. For large models, keep the model
loaded in a local service and let the short-lived worker call that service.

## Low-level policy or VLA

Wrap fixed-policy inference in `skills.execute`. Convert its output to Panda
`OSC_POSE` normalized seven-dimensional actions and expand action chunks through
`context.step`. The adapter owns frame conversion, units, gripper sign and
termination. The runner owns simulation and scoring; a second simulator cannot
provide the current episode's result.

The current observation protocol exposes privileged object states and does not
enforce vision-only evaluation, even if a model ignores those fields. Other
agent/harness runtimes can use the decision interface through an adapter that
preserves execution ownership and action semantics.

## RoboRSI campaign adapter

The separate `eah plan/run/inspect/compare` commands adapt RoboRSI's campaign CLI
and audit its evidence. They do not insert RoboRSI into the two-file control loop.
The adapter is based on revision `e4e9024613e1060614964795206b64f15c422b06`:
[CLI](https://github.com/nssmd/RoboRSI/blob/e4e9024613e1060614964795206b64f15c422b06/roborsi/cli/commands.py),
[evaluation suite](https://github.com/nssmd/RoboRSI/blob/e4e9024613e1060614964795206b64f15c422b06/roborsi/evaluation/suite.py)
and [audit](https://github.com/nssmd/RoboRSI/blob/e4e9024613e1060614964795206b64f15c422b06/roborsi/evaluation/audit.py).
Validation covers source/fixture compatibility. Full model-backed RoboRSI
execution remains unverified in this repository.

Fixture checks require only the development dependencies:

```sh
python examples/make_fixture.py --out runs/fixture
eah inspect --spec examples/libero-small.json --baseline runs/fixture/baseline --out runs/fixture/failures.json
eah compare --spec examples/libero-small.json --baseline runs/fixture/baseline --candidate runs/fixture/candidate --proposal runs/fixture/proposal.json --out runs/fixture/comparison.json
```

Fixtures are synthetic journal tests, not robot rollouts. For real campaigns,
install the pinned upstream source and dependencies separately, prepare assets
and configure models using the upstream installation guide. Replace model
placeholders in `examples/libero-small.json` and use separate source, home and
output directories for baseline and candidate runs.

```sh
eah plan --spec experiment.json --python /path/to/backend/python --source /path/to/source --out /path/to/result
```

Review the plan, then use `eah run` with the same path arguments, a full source
`--revision`, `--home /path/to/backend-home` and `--execute`.
`eah inspect` extracts development failures; `eah compare` evaluates fresh
baseline/candidate evidence and a revision proposal. `accept_for_review` records
a regression decision on the finite panel. It provides no statistical guarantee
and does not install code.

Provider configuration remains in RoboRSI's normal configuration location;
`ROBORSI_HOME` only selects embodied data paths. This adapter does not read or
export provider credentials. Model costs and remote configuration are managed
outside the harness. See command `--help` for flags and schemas.
