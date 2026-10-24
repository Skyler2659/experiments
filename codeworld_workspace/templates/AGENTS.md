# CodeWorld execution flow (required)

Read task/task.json, task/WORLD_FLOW.md, task/EXECUTION.md, world.json and
state/current.json. Every physical decision and judgment follows WORLD_FLOW.md.
RGB-D is a perception input: publish numerical world estimates and their uncertainty,
then calculate actions and judgments from that world. Use the reconstructed prior
for initial planning; obtain further evidence when relevant state may have changed. A reconstructed mesh does not
establish joints, contact, friction, attachment or dynamics; preserve unknowns.

## Perception and world updates

Use Robot.observe() to obtain RGB-D, calibration and public proprioception. Identify
objects, estimate candidate transforms/part states and create binary masks. Publish
measured poses with World.integrate_rgbd(capture, proposals). Inspect its numerical
fit and uncertainty report. Acquire informative observations when the next
decision needs information that is missing; uncertainty does not require fitting
the entire scene after every action.

World.update(..., basis='action_prediction') and fork() are for hypotheses only.
Never promote a target pose or an EXECUTED command to an observed pose. Direct
World.update(..., basis='observation_estimate'), assess(... 'observed') and
update_state(...) are unavailable in strict episodes.

World.assess({object_id: {'pose': 'pending'}}, note=...) may record a possible push,
occlusion or missing geometry. Maintain affected bystanders as well as the target.
After a physical action, sync_robot_feedback marks declared affected objects pending.
Refresh those objects before relying on their state. Unaffected validated geometry
can be reused for planning; full-scene reconstruction at every step is unnecessary.
Final completion requires valid supporting evidence. Unchanged objects retain
their observations; evidence is invalidated when its supporting state may change.

Part geometry may be proposed through an assembly, including its fixed body,
part meshes and T_object_part matrices. Candidate geometry remains hypothetical
until accepted through integrate_rgbd. Parameters alone do not move part geometry.
A closed exterior mesh does not establish an articulated interior or a joint axis.

## World computation and execution

Write compute(ctx) in a Python file and pass it to
World.compute_decision(path, purpose=..., object_ids=[...]). Available queries are
position, bounds, mesh, transform, object, state, metadata, uncertainty,
proprioception and measurement. NumPy, math and SciPy Rotation are provided. Raw RGB-D processing belongs to perception, not the decision computation. Every relevant object must be queried.
Ordinary pure Python helpers, comprehensions, conditions and loops are supported;
return a dictionary of actions/judgments. Position is the current mesh geometry center.
Geometry queries expose the reconstructed/current estimate and uncertainty; metadata and uncertainty describe
what is known and missing, and cannot establish physical state by themselves.

Return actions with op, args and affected_objects, and/or named judgments. Record
all potentially contacted bystanders. Compute targets, frames, directions, grasp,
rotation, placement, transport, collision/reachability, recovery and progress from
world queries. Public task semantics, tool dimensions and tolerances are declared
constants. Fixed tool orientations and gripper modes are also valid policy parameters.
Static dependency analysis is a diagnostic; it does not prove a physical formula.
A world query followed by an invented physical truth is not a valid measurement.
Robot-only sensing moves still require a computation over public proprioception.

Submit an action with Robot.execute_decision(receipt['decision'], index). The server
checks the recorded computation, relevant world/robot dependencies, payload and
actor. Recompute when its actual dependencies change. Direct robot action/IK/collision calls and
legacy move_from_world are unavailable in strict episodes. Multi-arm operations
use explicit arm IDs; inspect Robot.capabilities() for public tool conventions.

Controller, IK and collision reports enter CodeWorld through sync_robot_feedback.
The returned value is already stored in CodeWorld, with measurement_id and revision;
inspect it without an extra CLI call. Use ctx.measurement(measurement_id) in the
next decision computation. IK
success alone does not certify full-arm clearance or successful manipulation.

## Grasp feedback and physical state

These are optional diagnostics used when their evidence is needed.
World.check_grasp_geometry assesses reconstructed jaw engagement for a candidate;
WIDTH_COMPATIBLE does not prove a grasp. Robot.assess_grasp compares stored RGB-D
and TCP co-motion and publishes its result into CodeWorld. In a strict episode it
returns MEASUREMENT_IN_WORLD, a measurement_id and its published value. Interpret that measurement via
ctx.measurement; NOT_LIFTED and INCONCLUSIVE cannot establish attachment.

Use World.derive_state(decision_path, object_id, {field: judgment_key}) to publish
physical scalar or boolean results of a validated world computation. Never invent
on=True, held=True, an angle, or task_complete from a visual impression. The queried
measurements/geometry must support the claimed physical property. Unknown fields
remain unknown, and later changes invalidate dependent fields. Read a valid derived
field in later planning with ctx.state(object_id, field).

## Completion

Before the first motion, register every public task requirement using
World.define_completion(goals). Each named subgoal has a description and executable
conditions. State frames, reference objects and tolerances explicitly. Do not omit
an initially satisfied requirement. A signed-axis relation alone may not establish
adjacency, support or containment. The contract is an interpretation of the public
goal; do not inspect simulator object poses, task XML or private reward predicates.

Condition kinds (each requires object):
- relative_center: reference, unit axis, frame (W or reference), min/max.
- bounds_gap: same plus object_anchor and reference_anchor (min/max/center).
- distance: reference, optional coordinate axes, min/max in meters.
- state: a world-derived physical field and equals or a numeric min/max interval.

Robot.verify_completion(goal_ids=None) reports VERIFIED, NOT_SATISFIED or
INCONCLUSIVE. Query a subset for a subtask and all goals for the full task. Recheck
after later actions; stale, predicted or unresolved state cannot prove completion.
Use Robot.finish(verification=report['verification']) only after full VERIFIED.
For physical failure/inability, compute its reason from CodeWorld and call
Robot.finish(outcome='failed', reason=..., judgment=decision_path). Runtime exits use
Robot.finish(outcome='aborted', system_reason={'kind':'timeout','detail':'...'}).
Kinds: budget_exhausted, timeout, tool_error, process_error, interrupted. These exits
do not assert a physical outcome. Official benchmark success is recorded separately.

Cite world decision or verification records in progress/outcome reports. Author
your own task programs; do not read prior-run action programs or modify server,
configuration, ownership or evidence files. Never reset or teleport the environment.
