# RCWM → agent workspace

Entry points (run export and geometry on Linux):

```sh
RCWM_EXPORT_OBJECTS=1 node scripts/export_codeworld.mjs RUNTIME RUN_NAME EXPORT_DIRECTORY
python -m codeworld_workspace.build --run RUNTIME/runs/RUN_NAME --export EXPORT_DIRECTORY --runtime RUNTIME --output NEW_WORKSPACE
```

The exporter records the actual evaluated Three.js scene graph and object matrixWorld values,
waits for asynchronous build completion, checks stable source bytes, and exports per-object GLBs.
The compiler inverts the executed object transform, preserves each local surface and material,
and checks vertex/bounds round trips. Source declarations remain available but never override
executed transforms. Nonrigid object-root transforms and nested object IDs are rejected.

The compiler snapshots the source and Three.js dependency and binds source, declarations and
geometry to one immutable geometry revision. Live pose is an overlay; it does not rewrite source
parameters. Source changes require a new export/publication. This resolves duplicate parent/child
pose declarations without inventing a fragile reverse mapping to arbitrary JavaScript assignments.
All original fields remain in the source snapshot. The object index is an agent-facing projection,
not a claim that every arbitrary JavaScript expression has a semantic equivalent.

Current scope: rigid geometry, static context, observed pose updates after robot manipulation;
no built-in joint estimator, contact physics or fine-tolerance certification. Agent-authored part transforms are supported.

Tests: `python -m unittest codeworld_workspace.tests.test_world` on Linux.
See templates/ for the exact Codex handoff and update contract shipped into each workspace.

CLI examples from a generated workspace:

```sh
PYTHONPATH=tools python -m codeworld_workspace . inspect
PYTHONPATH=tools python -m codeworld_workspace . object akita_black_bowl_1
PYTHONPATH=tools python -m codeworld_workspace . bounds akita_black_bowl_1
PYTHONPATH=tools python -m codeworld_workspace . export work/current.glb
```

Evidence boundaries: preservation covers the complete evaluated triangle surface and original
source files/parameters, not complete real-world geometry or automatic semantic understanding
of arbitrary source code. The builder does not reinterpret arbitrary fields as verified affordances.
The initial robot is a frozen asset. The execution adapter must supply current kinematics when
planning robot motion. Updating a pose does not simulate contacts or infer task success.

Current selected validation source is the original official GPT-6-astra reconstruction,
`libero_bowl_plate_drawer_static_20261016_v1`, configured in `configs/codeworld_workspace.json`.
`python -m codeworld_workspace.prepare --reuse-export` uses this selection and never calls an
LLM. Set a new workspace output path for a new build. Without --reuse-export it executes the
existing Three.js code first, which is rendering/export only, not a reconstruction-model run.

An alternative reconstruction backend is available without replacing this workspace or
the direct Codex robot interface:

```sh
# On the geometry host, with the pre-RCWM reconstruction package on PYTHONPATH:
python -m pipeline.reconstruct_legacy_measured --config /abs/measured_reconstruction.json
```

This runs SAM2, calibrated RGB-D sample consensus, shape fitting, surface completion,
and texture/export, then imports the new object meshes into this same CodeWorld.
It does not start a robot, import the old agent/controller, or reuse previous object meshes.
Original-image prompts may be reused when they match the input RGB; masks are regenerated.
Reconstruction recipes and mesh hierarchy are retained as source evidence. Import checks
surface preservation, not reconstruction accuracy.
`bounded_refinement=false` skips the expensive optional free-space surface refinement;
the support, topology, coordinate-export and texture checks still run. The first backend
integration trial uses that setting and records it in its stage report.

To rebuild a workspace from those finished assets, `prepare --config <config>` also accepts
`reconstruction.backend="legacy_measured"` and `reconstruction.source_scene` pointing to
the resulting `assets/scene.json`; validation mode remains `reuse_existing_reconstruction`
with `run_model_inference=false` and a new workspace output path.

Optional articulated/edited geometry is available via `World.publish_parts` and
`World.update_parts`; see DESIGN.md. These are generic agent-authored part transforms,
not a built-in joint estimator or physics model. State-only edits require no reconstruction.

### Scene changes and partial geometry

`World.assess({object_id: {aspect: status}}, evidence=[...], note=...)` records an
agent-owned assessment without changing any geometry/pose. Statuses: `pending`,
`occluded`, `observed`, `hypothesized`, `unreviewed`. Aspect names are open; use `pose`
for root pose and qualified names for local geometry/parts. `World.review_status()`
returns these records; absent assessments remain unreviewed. `object().current` and
`bounds().review` retain the metadata. Existing mesh queries remain available.
An `observation_estimate` pose update with references confirms only the root pose;
predictions, part updates and geometry publications do not clear unrelated concerns.
An agent can confirm an unchanged pose via `assess`, or defer an occluded object.
This supplies persistent uncertainty bookkeeping, not automatic image change detection.
New history entries carry `recorded_at_ns` (host wall clock, not simulation time).

### Geometry-driven decisions

`ctx = world.planning_context()` freezes current geometry/state for agent-written
planning. Query `bounds`, `point`, `local_point` or `mesh`; compute candidates in code.
`compare_routes(candidates, tool_half_extents_W=..., exclude={id: reason})` compares
continuous TCP polyline tool envelopes against world AABBs, ranking before selection.
It reports source unknowns/reviews and returns no selection if every route overlaps.
This optional broad phase is conservative, does not model the full arm, and cannot
resolve cavities. It does not replace task-specific mesh/contact reasoning.
Save the decision with `ctx.save`; `Robot.move_from_world` executes an indexed waypoint
and records decision identity/revision in the receipt. Ordinary agent-authored motion
code remains supported. RGB-D estimates should update the world before replanning.
