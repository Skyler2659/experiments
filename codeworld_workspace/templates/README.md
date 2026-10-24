# CodeWorld workspace

- source/: original reconstruction and its metadata.
- geometry/: object meshes in their local coordinates.
- objects/: object descriptions and source bindings.
- state/current.json: current object poses and their estimation/prediction basis.
- task/: task description, robot connection and results.
- work/: programs written by Codex for this task.
- scenarios/: optional alternative poses for planning.

Use `World(root)` to query geometry. Update current poses with `World.update` and label
whether they come from action prediction, agent estimation or sensor observations.
No perception model or motion planner is required by this interface. `verify_binding()`
is an explicit source-integrity audit, not a read-time or execution gate.
