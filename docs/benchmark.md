# Running a benchmark

Run commands from the repository root with the virtual environment activated.
Manufacturing tasks need the `manufacturing` extra; LIBERO tasks need `preview`
and a separately downloaded LIBERO source tree.

## Participant interface

| File | Entry point | Purpose |
|---|---|---|
| `participant/agent.py` | `create_agent()` → object with `decide(observation, feedback)` | Choose a skill, its arguments, or stop |
| `participant/skills.py` | `execute(call, context)` → feedback dictionary | Execute robot actions |

Example agent decision:

```python
{"type": "skill", "name": "insert", "arguments": {"motion_scale": 1.0}}
# To finish the episode:
{"type": "stop"}
```

Skills read `context.observe()` and call `context.step(action)`. Actions contain
normalized XYZ displacement, rotation displacement and gripper command: seven
finite values in `[-1, 1]`, consumed by Panda `OSC_POSE` at 20 Hz. Gripper `-1`
opens and `+1` closes. These are controller inputs, not absolute Cartesian poses
or joint positions.

Observations include `instruction`, `eef_pos`, `eef_quat_xyzw`, `gripper_qpos`,
world-coordinate `objects`, `image_path`, `remaining_steps` and `context`.
Manufacturing also supplies object orientation, task family, episode role and
raw wrist force/torque. Quaternions in `object_quat_wxyz` use WXYZ order; the
end-effector quaternion uses XYZW. Positions are metres. Wrist readings use the
end-effector sensor frame.

`completed` feedback means the skill sequence ended; the evaluator determines
whether the task succeeded. A participant may use feedback to recover within an
episode. The runner resets participant state before the next target.

## Agent track

```sh
eah-bench --panel benchmarks/manufacturing-agent-v2.json --split evaluation --out runs/agent
```

Open `runs/agent/report.html`. The example covers round/square insertion,
round/square nut placement and stacking under two physical parameter sets.
Exit code 1 indicates an unsuccessful episode. Inspect `termination` and `error`
to distinguish task failures from implementation errors. Output paths must be
new. Import and agent-factory failures are recorded as implementation errors.
Manifests distinguish running, failed and completed campaigns.

For development, generate a separate panel before editing the participant:

```sh
python scripts/generate_manufacturing.py --track agent --seed 42 --worlds 2 --out benchmarks/development.json
eah-bench --panel benchmarks/development.json --split development --out runs/development
```

## Physical ICL track

```sh
eah-bench --panel benchmarks/manufacturing-picl-v2.json --split development --out runs/source
eah-transfer --panel benchmarks/manufacturing-picl-v2.json --source-run runs/source --out runs/transfer
```

Source and target task families are disjoint. Each physical world fixes actuator
gain, object mass and scene-wide sliding friction. Equal coefficients on object,
table, fixture and gripper contacts prevent contact mixing from masking changes.
The source task, geometry and layout stay the same across worlds. Target case
definitions, environment hashes and initial states match across conditions.

| Condition | Source input |
|---|---|
| `no_context` | Empty source context |
| `source_context` | Matching-world calls, feedback and bounded trajectory excerpt |
| `action_effect` | Probe calls/effects and trajectory excerpt; task language and final label omitted |
| `language_only` | Source instruction and family |
| `unrelated_physics` | Same source task/layout under a different physical world |
| `shuffled_feedback` | Source calls with feedback reassigned |

The action-effect excerpt retains object-state semantics, so it does not fully
isolate physical information. Runs record context byte counts. Model experiments
also need matched token counts and inference budgets.

Source banks require a complete simulator campaign with matching skill hashes,
metric versions, execution implementation, runtime and agent identity. They
accept failed source tasks and reject unresolved implementation errors. Keep
target code, model weights and low-level policies fixed throughout a comparison.

## Randomized panels and language

```sh
python scripts/generate_manufacturing.py --track physical_icl --source-family random --physical-axis combined --worlds 6 --layouts 3 --seed 42 --out benchmarks/study.json
python scripts/generate_manufacturing.py --track agent --instruction "把圆销插入圆孔并松开" --worlds 2 --seed 24 --out benchmarks/language.json
```

`--source-family` accepts `insertion`, `nut_assembly`, `stacking` or `random`.
`--physical-axis` accepts `actuation`, `contact` or `combined`. Random source
selection chooses one family and variant per experiment. Robot geometry stays
fixed. Freeze the generated JSON before evaluation; seed reproduction also
requires the same generator version.

Language binding supports explicit English/Chinese single-task goals. It rejects
unsupported instructions, negation and mismatched goals. The example manufacturing
agent selects skills from the public task-family label, so its results do not
measure general language understanding.

## Reference results

The version 0.6.0 recordings use a frozen state-feedback rule policy after the
contact-friction correction. Source motion probes adjust a target skill's motion
scale; no model is trained. Both source tasks failed to complete but retained
valid probe trajectories, which are included in the source report.

| Experiment | Observed result |
|---|---|
| Agent, five tasks × two worlds | 8/10 task success |
| Six-condition transfer, six targets per condition | No context 3/6; matching source and action effect 4/6; other conditions 3/6 |
| World/family macro success, no context → matching source | 62.5% → 75% |
| Manufacturing-only wheel install, Chinese goals | 1/2 task success; PyTorch absent |

These reference panels were used during development and are not untouched
publication test sets. Only two physical worlds were evaluated. The observed
transfer concerns servo calibration and does not establish learned contact
dynamics. See [metrics](metrics.md) for scoring and uncertainty rules.

## LIBERO and bounded refinement

```sh
python -m pip install -e '.[preview]'
python scripts/fetch_libero.py
eah-bench --source .research/LIBERO-8f1084e3132a39270c3a13ebe37270a43ece2a01 --panel benchmarks/libero-object-pilot.json --split evaluation --out runs/libero
```

LIBERO placement panels are custom distributions and should not be reported as
official benchmark scores. `scripts/generate_benchmark.py` generates seeded
language/placement panels. `eah-rsi --help` provides bounded skill revision,
evaluation gating and saved-skill reuse. `--agent module:function` selects a
revision-agent factory; this is separate from the two-file benchmark interface.

## Reports and exports

```sh
eah-report runs/transfer
python scripts/export_benchmark.py --run runs/agent --out assets/agent
python scripts/export_transfer.py --run runs/transfer --source-run runs/source --out assets/transfer
python scripts/verify_evidence.py assets/agent assets/transfer
python -m http.server 8771 --bind 127.0.0.1 --directory assets
```

Open `http://127.0.0.1:8771/` to browse the included reports. Each episode has a
video, trajectory, result and initial state. Exports retain simulator evidence
while replacing host paths; raw run directories remain separate.
