# Architecture

The runner initializes and runs simulation, enforces budgets, records evidence
and scores episodes.
Participants supply an agent and a skill implementation.

```text
panel + instruction
  → benchmark runner
  → agent.decide(observation, feedback)
  → skills.execute(call, context)
  → context.step(action)
  → simulator
  → evaluator + trajectory + video + report

source rollouts → experience bank → context selection
  → paired target rollouts → transfer statistics
```

## Modules

| Module | Responsibility |
|---|---|
| `benchmark.py` | Panel validation, participant loading, action/call budgets, episode recording and aggregation |
| `manufacturing.py` | Panda insertion, nut placement and stacking environments; layout and physical parameter interventions |
| `metrics.py` | Versioned task specifications and optional force constraints |
| `physical_icl.py` | Source/target family separation, physical-world matching, context controls and world/family aggregation |
| `memory.py` | Source evidence validation and bounded experience-bank construction |
| `transfer.py` | Paired target execution with frozen participant files |
| `report.py` | Static HTML reports with episode videos and metrics |
| `bridges.py` | UTF-8 JSON interface to an external inference process |
| `language.py` | Supported English/Chinese goal bindings |
| `libero_demo.py` | LIBERO environment setup and bounded skill-refinement execution |
| `rsi.py`, `agents.py` | Skill revision proposals, regression gate and revision-agent interface |
| `roborsi.py` | Pinned RoboRSI campaign command adapter |
| `evidence.py`, `spec.py`, `cli.py` | Campaign audit, configuration and command dispatch |

## Repository layout

```text
src/embodied_agent_harness/  library and CLI
participant/               agent and skill templates
benchmarks/                runnable example panels
examples/                  process worker and campaign fixtures
scripts/                   panel generation, setup, export and packaging
tests/                     contract, integrity and regression tests
docs/                      benchmark, metrics, architecture and integration guides
assets/                    reference reports and recorded simulation evidence
.github/workflows/         lightweight Python checks
```

Runtime directories `.venv/`, `.research/`, `.cache/`, `runs/`, `build/` and `dist/`
are ignored and excluded from source exports; they are not required repository content.

## Execution boundaries

Each target episode reloads participant modules and creates a new agent instance.
Only declared source context passes between episodes. Participants must also
reset imported helper modules and inference services if they retain state from
target episodes. The runner checks that each action has seven finite normalized
values, enforces budgets and evaluates the success predicate. Observations omit
reward, simulator termination and goal-oracle values.

Participant files run as trusted Python plugins in the runner's process. File
hashes detect changes to declared code, but provide no isolation from malicious
code or verification of remote model weights. External workers have a
per-request timeout and explicit JSON state. A worker can call a separately
started inference service to keep a large model loaded.

## Evidence

Each episode produces `result.json`, `trajectory.jsonl`, `initial-state.npy` and
`rollout.mp4`. Transfer experiments also store the exact input `context.json`.
Manifests record panels, participant hashes, dependency versions and campaign status.
Manufacturing episodes also record compiled masses, inertias and friction in
`physical-state.json`. Source reuse requires an identical execution implementation
and runtime, in addition to a matching metric version. Export manifests record
original file hashes separately from portable exported JSON hashes.

`scripts/verify_evidence.py` checks exported bytes, episode/summary agreement,
trajectory length and paired environment/initial-state/metric identity. It checks
record integrity; it does not replace simulator validation.
