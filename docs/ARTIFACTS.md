# Artifacts and video export

Everything below is relative to one sample directory. Results keep the original native
evaluator outcome; a rendered scene or successful export is not task success.

| Artifact | Location |
|---|---|
| Native result, policy timing and usage | `episode/result.json` and native `_result.json` under `episode/bench/eval_result/` |
| Exact policy prompt | `episode/prompt.txt` |
| Agent visible events and tool calls | `episode/codex_events.jsonl`, `tool_turns.json`, `calls.jsonl`, `codex_stderr.txt` |
| Reconstruction source and agent logs | `episode/privilege/reconstruction/runtime/runs/` |
| Initial visual GLB and objects | `episode/privilege/reconstruction/export/` |
| Live world | directory named by `episode/privilege/integration.json` |
| Historical CodeWorld snapshots and calls | `audit/states/`, `audit/calls.jsonl` |
| Saved RGB observations and camera metadata | `episode/steps/` |
| Provider retries and waits | `recovery/` |
| Continuous task-only camera MP4s | `episode/bench/eval_result/` |

```bash
worldrua export /path/to/batch/stack_bowls_s0 --bundle --videos
# For an imported episode without a batch manifest, also pass --runtime /path/to/runtime.
```

This writes `exports/current_scene.glb` from original visual meshes and recorded current
object/part transforms. Materials and textures are preserved; robot/camera models are
hidden. It also copies the original `initial_scene.glb` when present and native camera
MP4s. The policy-facing `world.export_scene()` remains the original computation-mesh
export; use the offline exporter for appearance-preserving publication assets.

`--bundle` writes `exports/artifacts.tar.gz`, including world assets/source, logs,
snapshots and observations. Authentication files, installed runtime dependencies and
symlinks are excluded. Experiment logs can contain scene-specific information and
absolute output paths; keep them outside the public code repository. The archive is
for analysis/handoff, not a physical simulator checkpoint. Its exported GLBs are
portable; replaying a relocated raw world may require rebinding recorded paths.

`--videos` writes:

- `videos/execution_observations.mp4`: recorded head-camera observations at 2 fps.
- `videos/comparison.mp4`: those observations alongside matched historical CodeWorld
  states. Original materials are used, robot/camera geometry is hidden, and missing
  historical pairs are reported in `videos/manifest.json` rather than replaced with
  final geometry.

These are observation-sequence videos, not full-rate continuous robot recordings.
The separate native camera MP4s are the continuous task-only recordings. A combined
task-plus-Agent-log video is not implemented in this release; the source logs are
included for analysis. No hidden model reasoning is inferred or reconstructed.

The exporter runs directly where the artifacts live, so it does not depend on SSH
syncing the old experiment directory. If copying results elsewhere, copy completed
artifacts first and verify the world path in `integration.json`.

## Live monitor

```bash
worldrua dashboard /path/to/batch --port 5191
worldrua comparisons /path/to/batch
```

The second command renders saved same-step RGB/CodeWorld comparisons in a separate
serial worker; it makes no robot calls. It can run in another terminal/tmux session.
The dashboard binds only to localhost. Use an SSH tunnel to view it remotely; no
public hosting service is required. A comparison image has its own recorded step and
time, which can lag the latest RGB. Final video export is an explicit offline command.
