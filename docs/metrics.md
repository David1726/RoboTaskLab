# Metrics

`metric_version = eah.metrics.v2` identifies the current task predicates.
The JSON record schema is versioned separately. There is no combined weighted
score and no default wrist-force limit.

## Task completion

The evaluator determines success; implementation errors count as unsuccessful
episodes. For manufacturing, the task predicate must hold at the final five
consecutive 20 Hz control samples, spanning 200 ms from first to last. This
rejects a single transient observation but does not establish continuous or
long-term stability.

| Task | Predicate |
|---|---|
| Round/square insertion | Released; tip depth 18–34 mm; axis Z > 0.995; conservative tip-centering condition; positive engaged cross-section clearance |
| Nut placement | robosuite 1.4 NutAssembly placement predicate, plus explicit absence of grasp |
| Stacking | robosuite 1.4 Stack contact/lift/release predicate |

Insertion uses a 100 mm peg with 10 mm radius or half-side. Hole radius/half-side
is declared by the panel. Axis Z > 0.995 bounds tilt to approximately 5.73°.
Tip lateral error must be less than `opening_m - 0.010`. Square corners are
projected into the hole planes, including yaw. A tilted round section uses a
conservative bounding circle. These are coarse simulation task specifications,
not manufacturing tolerance standards. The centering rule can reject some offset
square poses that otherwise fit. An upside-down cross-section diagnostic is null.

Terminal depth, lateral error, tilt, clearance and release state are diagnostic
fields. Nut placement does not model thread fastening or tightening torque.

## Transfer

Physical parameter checks use the compiled simulator model. The friction field
controls scene-wide sliding contact coefficients. Results from before the backend
correction cannot be reused as current source experience.

Report both pooled success and world/family macro success. The latter averages
within each task family, gives families equal weight within a world, then gives
worlds equal weight. Multiple insertion shapes therefore do not outweigh stacking.

A matching metric version confirms the goal definitions. Reusing source
experience also requires matching execution implementation and runtime, with
the same agent entrypoint.

Paired gain is the difference from `no_context` on identical target definitions,
environment hashes, initial states and metric versions. Positive/negative transfer
counts identify failed-to-successful and successful-to-failed pairs. Mixed metric
versions or unmatched conditions are rejected.

World-cluster bootstrap uses 2,000 resamples. Reports omit intervals when there
are fewer than five physical worlds or the resampling distribution is degenerate.
The five-world cutoff controls display only; it does not guarantee an adequate
sample size. Displayed intervals are exploratory. A study's claims still require
a sampling design with sufficient statistical power.

## Cost and errors

Record control steps, skill calls, implementation errors and termination causes.
Control steps correspond to simulated time, not wall-clock inference latency.
Source collection costs are reported separately. Compare efficiency on jointly
successful pairs as well as all episodes; early failure must not appear as an
efficiency improvement. Context bytes do not substitute for model token/cost logs.

## Wrist load

`peak_wrist_force_N` is the maximum Euclidean norm of the raw end-effector force
sensor at executed 20 Hz control samples. It is not gravity/inertia compensated
or sampled at every physics substep. A wrist sensor includes load and dynamic
effects; its reading is not an isolated workpiece contact force.
See [MuJoCo force sensors](https://mujoco.readthedocs.io/en/stable/XMLreference.html#sensor-force)
and [robosuite sensors](https://robosuite.ai/docs/modules/sensors.html).

By default, `force_metric_status` is `diagnostic_only` and
`force_qualified_success` is null. A task owner can declare both `force_limit_N`
and a nonempty `force_limit_basis` before freezing a panel. The basis should
describe the sensor, task requirements and calibration supporting the chosen
constraint. `task_declared` records a declaration; it does not verify that basis.

When a measured peak and declared constraint exist, force-qualified success means
task success with a peak at or below the limit. Missing measurements remain null.
An aggregate rate requires a complete, comparable constraint for every episode.
Exceeding the limit changes the score but does not terminate the robot. This
metric provides no real-robot safety or damage certification.

## Evidence

Episode results, original trajectories, source context, initial states and videos
are retained with hashes. Portable exports have their own hashes and retain the
original hashes for provenance. Undefined historical diagnostic values are shown
as null in browser reports; original records remain unchanged. Tests cover
false success, wrong square orientation, missing measurements, version drift,
paired environment mismatch and insufficient-world intervals.
