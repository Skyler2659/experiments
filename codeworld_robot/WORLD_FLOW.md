# CodeWorld flow for physical decisions

1. **Perception and update.** RGB-D identifies objects and estimates pose/part candidates
   and masks. Publish these through `World.integrate_rgbd(capture, proposals)`.
   Proposals map object IDs to `T_W_object` and `masks={view_id: mask_path}`; binary
   NPY/PNG masks are supported. Fit quality and uncertainty accompany the update.
   Use the reconstructed prior for initial planning. Update relevant objects when
   observations or actions indicate changed state. Reuse unaffected geometry.
   Images do not directly establish an action, direction, attachment or success.
2. **World computation.** Write ordinary pure Python `compute(ctx)` in an episode
   file, then `World.compute_decision(path, purpose=..., object_ids=[...])`.
   Recorded queries: `position`, `bounds`, `mesh`, `transform`, `object`, `metadata`,
   `uncertainty`, `proprioception`, `measurement` and `state(object_id, field)`.
   NumPy (`np`), `math` and SciPy `Rotation` are provided. Helpers, comprehensions,
   loops and normal Python output construction are supported. Keep raw observation
   IO in perception; physical decisions read CodeWorld. Query relevant objects.
   `position` is the current mesh geometry center. `metadata` contains labels and
   declared unknowns. `state` reads a derived field whose supporting inputs remain
   valid. Uncertainty may constrain a decision without forbidding every use of a mesh.
3. **Actions and policy parameters.** Return
   `{'actions':[{'op':..., 'args':..., 'affected_objects':[...] }], 'judgments':{...}}`.
   Targets, object relations and physical claims use the world. Public task semantics,
   tool conventions, fixed approach orientations, gripper modes and tolerances may be
   declared policy parameters. They need not each vary with a sensor reading.
   Static dependency hints are diagnostics, not a Python restriction or proof of
   correct reasoning. Runtime queries, source and results remain recorded for review.
4. **Execution and feedback.** `Robot.execute_decision(receipt['decision'], index)`
   submits the computed action. Dispatch checks its ownership, relevant world/robot
   dependencies and payload. Direct unbound physical commands are rejected.
   Omitted arm IDs are resolved during computation and saved explicitly in the
   payload; later active-arm selection cannot silently redirect an existing action.
   Controller, IK and collision results are stored as CodeWorld measurements before
   returning `status='MEASUREMENT_IN_WORLD'`, `measurement_id`, `state_revision` and
   `value`. The returned value may be inspected/logged immediately; use the same ID
   with `ctx.measurement` in subsequent computations. No extra CLI call is needed
   merely to view a result. `Robot.capabilities()` provides static tool conventions;
   these are backend-specific. RoboDojo X5's TCP +X approach convention must not
   be applied to LIBERO/Panda. If an axis is not explicitly advertised, derive
   it from the current public robot geometry and TCP binding rather than using
   another backend's convention.
   Use `ctx.proprioception()` for robot state in decisions.
5. **Freshness.** Refresh objects that an action may have changed; declare touched
   bystanders too. Unaffected objects retain their prior observations. A read-only
   query does not by itself invalidate physical state. Only stale dependencies need
   recomputation. Neither every controller tick nor every sensing move requires a
   full-scene reconstruction or pose fit. Commanded poses remain predictions.
6. **Physical state.** `World.derive_state(decision_path, object_id,
   {field:judgment_key})` publishes physical results from world computations.
   Read them later through `ctx.state`. Direct visual `on=True`, `held=True`, angles
   and success flags cannot become measurements. Use numerical world evidence for
   the claimed property. `World.check_grasp_geometry` and `Robot.assess_grasp` are
   optional diagnostics when their evidence is needed, not a required ritual for
   every grasp. Grasp feedback is also returned as an already-published measurement.
   Width compatibility or gripper closure alone does not establish attachment.
   The width diagnostic encloses every target surface in the pad band along the
   closing axis. For hollow or compound shapes, an INCOMPATIBLE result can include
   a distant opposite wall; it does not rule out a local rim or handle pinch.
   Evaluate such a contact region from CodeWorld geometry, check the actual finger
   path for collisions, and use observation-backed lift feedback for attachment.
7. **Completion and exit.** Register the public task requirements as named world
   conditions; check subgoals and final completion through `Robot.verify_completion`.
   A success exit requires the full verified report. A physical inability uses
   `Robot.finish(outcome='failed', reason=..., judgment=decision_path)`.
   Runtime termination uses `Robot.finish(outcome='aborted', system_reason={
   'kind':'timeout', 'detail':'...'})`; supported kinds are `budget_exhausted`,
   `timeout`, `tool_error`, `process_error`, `interrupted`. System termination does
   not establish a physical outcome. Official benchmark scoring remains separate.

This covers selection, coordinates, grasp/placement, transport, recovery, state
interpretation and completion. Planning from CodeWorld is required; image-based
physical judgments are not an alternative route. Unknown joints, hidden surfaces,
contact and friction remain unknown or explicitly modeled hypotheses. CodeWorld
records the agent's model and observations; it does not prove the model or every
physical formula correct. Never inspect simulator object poses/private predicates
or alter server/ownership/evidence files during an evaluated run.

### Objects whose shape or internal configuration changes through interaction
Treat the reconstructed mesh as an initial configuration, not a permanent rigid body.
This includes articulated/sliding objects (doors, drawers, scissors), elastic objects
(clips, springs), and deformable objects (bags, cloth). These examples are not a fixed
catalog: choose the representation from the intended interaction and observed response.
Distinguish whole-object motion from relative part motion and local deformation.

Choose contact points, motion direction/path, action scale and feedback cadence yourself
from CodeWorld geometry, task semantics and available measurements. Missing joint axes,
limits, stiffness or a deformable-body model do not block an interaction. Use an explicit
planning hypothesis when those properties are unknown; a complete mechanism model,
physics identification or full remeshing is not a prerequisite for action.

After an interaction that may change configuration, refresh the affected part/region
when needed for the next decision. Preserve the fixed body and unaffected geometry;
do not force changed geometry to fit one whole-object rigid transform. For relative
part motion, use a lightweight assembly and part transforms. For deformation, update
only task-relevant local geometry/shape and configuration descriptors from observations.
Use publish_parts/update_parts and RGB-D candidate-assembly integration as applicable;
scalar parameters alone do not deform a mesh or move a part. If a needed representation
is unsupported, retain that limitation and choose a supported local approximation or
another interaction rather than treating an old static mesh as the current shape.

Local, observation-supported configuration/shape updates are allowed maintenance of
CodeWorld, not a requirement to reconstruct the entire scene. Retain the original asset
as provenance. Do not import simulator geometry or ground-truth state. Select the least
representation detail sufficient for the next operation; adding joints is optional.
All physical actions and progress judgments still go through CodeWorld computations.
Keep predicted changes distinct from observed changes; commanded motion does not prove
opening, closing, insertion or deformation. Use relevant measured geometry/state for
progress, rather than requiring a joint angle for every configurable object. The private
official evaluator independently ends the episode as soon as it reports success.

### Component-aware interaction interface
For an object already represented as an assembly, ctx.mesh(object_id, part_id=...),
ctx.bounds(object_id, part_id=...), and ctx.transform(object_id, part_id=...) query
that part in world coordinates from the same recorded snapshot. Omit part_id for
unchanged whole-object queries. World.parts(object_id) lists the available parts.
Collision bodies for assemblies are named object_id::part_id; the rest of the object
remains present when one part moves. Prefer component-scoped intentional contacts
instead of exempting the whole cabinet/appliance from collision checks.
A proposed contact-driven part motion can be screened with an attachment containing
object_id, part_id, arm, and T_tcp_part. T_tcp_part is the hypothesized part-local-frame
transform relative to the TCP (derive it from CodeWorld and proprioception). Only that
part follows the sampled TCP path; fixed parts, other objects and the robot are still
checked. This is a collision prediction only: it does not attach simulator objects,
enforce a hinge, prove contact or change the measured world. Revise or discard the
hypothesis after observing the result. A push may lose contact; decide its validity
yourself. If the mesh contains separable components, use them without reconstructing
an entire kinematic model. Do not redefine bowl/mug rigid motion as component motion.
Choose grasp/pull, push, sliding contact or another feasible strategy independently.
Do not require an independent full-scene fit after every increment: update the changed
component as needed, preserve fixed-body evidence and use a bounded local estimate.
Do not make successful completion depend on optional tighter geometric thresholds
before attempting the remaining public task; official success ends the episode.

## Efficient interaction cadence
Use the current CodeWorld estimate to plan a useful action segment. An update is
needed when changed or uncertain geometry would materially change the next action;
an approach, micro-correction, closure and lift do not each mandate a separate fit.
A trajectory and a predicted payload may be planned under explicit uncertainty.
Verify grasp after a meaningful probe when needed, and verify task completion from
observed state. Neither width diagnostics nor lift diagnostics are mandatory gates.
The single-arm adapter screens alternative IK branches and reuses an identical
checked plan while robot state, geometry and planning options remain unchanged.
Use capabilities' finger_geometry in the bound TCP frame; joint-axis sign means
increasing coordinate, not automatically closing. Use actual finger collision meshes
for local contact geometry. Never interpret WIDTH_COMPATIBLE as a predicted grasp.
