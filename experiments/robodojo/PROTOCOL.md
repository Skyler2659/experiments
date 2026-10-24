# Frozen evaluation protocol

The default campaign contains 700 paired samples and four strategies: 2800 episodes. It is a selected task study, not the complete 42-cell RoboDojo leaderboard. The user-approved condition retains external multiview RGB-D and reports this observation augmentation explicitly.

| Strategy ID | Observation / representation | Executor |
|---|---|---|
| `roboprobe_l3` | Native RGB and robot state; no metric depth | Released, unmodified `RoboDojo_Agent_L3_Inspect_EEF` |
| `ours_no_codeworld` | Same initial external RGB-D scan and subsequent native RGB-D as Ours; direct reasoning, no persistent mesh scene or CodeWorld | Independent Codex CLI |
| `ours` | Fresh RGB-D reconstruction, programmable CodeWorld, observation-backed updates | Independent Codex CLI |
| `gt_codeworld` | Same RGB-D access plus native visual geometry, object identity and current rigid/link transforms | Independent Codex CLI using the same World/action interfaces |

GT defaults to `gt_mode: live_state`: native meshes are exported at initialization; rigid-object and articulated-link poses refresh when the agent observes. Links have separate meshes and a shared parent-articulation identity, so moving a lid does not leave its geometry at its initial pose. This is a geometric/state oracle upper bound, including identity and tracking benefits. It is not solely an initial-geometry ablation. `initial_geometry` exports the initial geometry and then requires the ordinary observation-backed update path. Native analytic primitives are tessellated; concave mesh faces, disconnected parts and holes are preserved. The oracle does not provide rewards, goal transforms, joint targets, trajectories, physical parameters or task-success predicates. Ground truth provenance is distinct from estimated sensor evidence.

## Public task information

All Ours-family goals include the native instance instruction plus the same released
`RoboDojo_Agent_L3_Inspect/recipes/<task>.md` text used by RoboProbe. Random variants
use the base task recipe, matching the released loader. The original instance
instruction is retained separately and remains authoritative. The recipe is saved
as `task/PUBLIC_TASK.md` and inside `task/task.json`; it contains public description
and scoring requirements, not hidden target transforms or an action program.
RoboProbe recipe inclusion is explicitly enabled. Tasks without a released recipe
record `available: false`; no task-specific instructions are invented.

The first Build Tower Astra-high diagnostic omitted the recipe from Ours-family
inputs. Its results are retained as pre-fix diagnostics, not an information-matched
ablation. The correction uses new episodes and independent agent contexts.

## Official samples and evaluator

Layouts come from the pinned RoboProbe publication's `astra_task_layouts.json`; only slot, variant and layout ID are read. Outcome fields are not selection inputs. Each strategy receives the identical native task, eval seed and layout ID for a paired slot. Some tasks use 25 standard plus 25 random variants. Published unstable-layout replacements are preserved; for example Fill Egg Holder includes layouts 50–53. A slot is one-based; native `layout_id` is the index in the evaluator's numerically sorted file list, not necessarily its filename suffix. Eval seed selects the layout directory and is a separate concept.

| Task | Native action limit |
|---|---:|
| Insert Tubes | 500 |
| Build Tower | 1050 |
| Pack Objects Into Box | 1300 |
| Hang Mugs | 800 |
| Fill Egg Holder | 700 |
| Organize Table | 1000 |
| Arrange Largest Number | 1050 |
| Swap T | 400 |
| Store Tools In Toolbox | 900 |
| Sweep Blocks | 1000 |

Limits are read from the native task at runtime. The table is documentation of the pinned sources, not a replacement scoring implementation. Native `reset`, `take_action`, reward registration, terminal checks, unstable filtering, normalization and `_result.json` remain authoritative. The adapter replaces policy transport; RoboProbe's EEF deploy already ignores its WebSocket client and runs its released policy in the simulator process. Ours supplies the independent CLI action queue inside native `run_eval`.

Native control frequency is 25 Hz, with physics dt 0.004 s: one official action advances 10 physics ticks. An Ours interface call can execute several waypoints and consume several official actions. IK checks, geometry queries and rendered observations are not actions. The framework logs interface requests, official actions, physics timing and model traces separately. Action interpolation and bounded settling consume native actions. No increased step limit, post-budget action, reward substitution or self-reported success is used.

## Observation acquisition and method controls

All three Ours-family strategies reset once, take the same fixed 18 external views and three native views with physics frozen, restore the head camera, then act in that same episode. Scan readiness and native-view revisit are checked; acquisition must not increment the native action counter. Reconstructed geometry uses the initial native view plus six fixed external fit views. Scanner views 3, 9 and 15 are reserved from fit for inspection/holdout use. The public X5 URDF is transformed from measured robot joints; target geometry and target poses never enter reconstruction.

Every default Ours episode reconstructs fresh. The shared runtime contains only software and fixed tools, not prior object meshes or prior task programs. The adaptive method retains depth 4, 8 cycles per node, up to 4 measured reviews, and one configured correction round with up to 2 extra reviews / 4 extra cycles. Reconstruction is not replaced by a template or coarse box model for handoff.

Ours, Ours minus CodeWorld and GT share the CLI executor model, reasoning effort, public robot kinematics, trajectory generation and dual-arm state interface. Removing CodeWorld also removes scene/payload collision geometry; robot self/inter-arm collision checks remain. This is a representation/interface ablation, not a new learned control policy. Live GT has additional privileged state information and must be reported separately.

The minus-CodeWorld arm skips the reconstruction stage and reasons from the raw RGB-D inputs. Report it as removal of the whole CodeWorld branch, including its preparation cost and scene-geometry capabilities. This four-strategy study does not isolate only the serialization format under identical reconstruction and total compute.

Default model is `gpt-6-astra`, reasoning `high`. RoboProbe uses its released launcher's 170-call budget, two-image horizon and 90-second isolated request deadline, RGB-only, with no environment step-limit override. Ours-family execution has a 14,400-second wall limit; fresh reconstruction has a separate 14,400-second wall limit. Ours does not claim an identical LLM-call budget to RoboProbe. Wall times, model usage, official actions and preparation cost must be reported separately. Short validation campaigns declare their modified wall limits and are excluded from the research campaign.

## Independence and reporting

The desktop supervisor prepares infrastructure and starts/stops lifecycle only. Ours-family task programs, planning, observations and world updates belong to an independent Codex CLI. Each episode uses fresh CLI context, disables memory and host skills, and permits only a fixed inspectable reconstruction toolkit. No cross-task experience, skill retrieval or skill saving is enabled. Working-directory isolation and policy instructions restrict access to evaluator data; this is not an operating-system security boundary against a malicious policy. Evaluator outputs live outside the episode workspace, and model traces remain available for auditing.

Success is the native boolean in `_result.json`; score is the native normalized score converted to 0–100. A render, export, prediction, collision clearance or CodeWorld completion assertion is not benchmark success. A native-only probe has `probe_only: true` and is excluded from all success aggregates. Method reconstruction failure counts against end-to-end success. Infrastructure/interruption/unstable outcomes remain unresolved. Completed failures cannot be retried for a favorable outcome. Attempts and infrastructure retry reasons are retained.

The manifest freezes task selection, model/budget settings, upstream commits and the native Assets release. The asset pin is [35efbc7](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo/commit/35efbc7dedfdbeeb6e95fb749bd885d73d483e41), also used on the validation host. Runtime paths and credentials are machine-local. Each attempt archives its official native layout JSON and source-version metadata. An Assets checkout with a known different revision is rejected; an exported folder without revision metadata is explicitly reported as unverified. The source pins and validation evidence accompany the handoff.

Primary references: [RoboDojo usage](https://robodojo-benchmark.com/doc/usage/), [RoboProbe source and published layout inventory](https://github.com/RoboProbe/RoboProbe), [Isaac Sim hardware requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html). NVIDIA does not officially support GPUs without RT cores for Isaac Sim; an H-card host needs an actual successful rendering probe before scheduling the campaign.

## Explicit cached-asset execution reruns

For debugging, an operator can set runtime `reuse_initial_artifacts` to a mapping
from strategy IDs to absolute source-attempt directories. This is opt-in and absent
from the default research configuration. The native task, layout, seed, representation,
GT mode and archived layout contents must match. Ours copies only the untouched
`preparation/workspace` source/geometry and revision-0 prior. GT reuses meshes and
reads poses from the new simulator. Prior execution code, final poses, histories and
completion contracts are not copied. RGB-D capture and execution use a new reset.
`artifact_reuse.json` and the reconstruction receipt identify the source and reuse.
Report these as cached-asset execution reruns, separately from fresh reconstruction
end-to-end evaluations. Preserve the superseded failed episodes.

## Fixed robot mounting contacts

The shared RoboDojo X5 adapter excludes only each fixed `base_link` against
`tabletop` as an installation contact, for all tasks and both reconstructed and
GT worlds. This is bound at collision initialization and reported in capabilities
as `fixed_mount_contacts`; agents need not rediscover these exceptions. Moving
links, fingers, payloads, other objects, self-collision and inter-arm collision
retain their checks. The no-CodeWorld arm has no scene/table collider. Other
robot backends do not inherit X5-specific contact exclusions.
