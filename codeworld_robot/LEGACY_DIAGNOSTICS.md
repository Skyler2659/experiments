# Legacy/manual diagnostics only

These historical examples are not the strict independent-agent protocol.

# CodeWorld robot execution

New episodes require the full **WORLD_FLOW.md** protocol: perception -> validated world
update -> world computation -> exact action dispatch -> feedback/world update -> world
judgment. This applies to every physical decision, not just completion. Read that file
before using any API. The older direct-call examples below describe legacy diagnostics;
strict independent episodes reject unbound direct robot commands and visual state claims.

# Direct Codex robot I/O

Codex reads CodeWorld and the task, writes programs, and operates the robot directly.
The default bridge has no task planner, pose estimator or SAM. Optional read-only grasp feedback is available in the client; see below.
RoboDojo can enable the sampled full-arm collision gate described below.
It imports no AgenticMeshWorld / CaP-X module. LIBERO supplies the robot controller and
physical contact dynamics; NumPy/SciPy provide coordinate arithmetic.

`Robot(root)`:
- `observe()` returns RGB-D, camera calibration and robot state.
- `state()` reads joint positions, TCP pose and finger positions.
- `move_tcp(pose_W, max_steps=150)` servos toward a world-frame pose.
- `gripper('OPEN'|'CLOSE'|'HOLD', steps=12)` commands the fingers.
- `step(actions)` executes arbitrary seven-dimensional LIBERO OSC actions.
- `finish(verification=...)` closes after CodeWorld verification; native evaluator output remains private.

LIBERO move_tcp uses the native OSC controller, stopping at its position/orientation tolerance
or the caller's step count. It performs no independent IK, mesh collision or path approval.
Raw step actions use simulator coordinates and normalized controller values; move_tcp
accepts world coordinates from the CodeWorld. Inputs must be finite and actuator commands
are clipped to the controller's range. Requests and receipts are retained to avoid
accidentally repeating a command after a communication timeout.

Codex chooses when and how to observe and maintain world state. Use World.update for
predicted or observed object poses. No probe, attach or fitted-pose report is required.
Task outcome is recorded separately from controller completion, using observed
CodeWorld state and the completion contract. The robot bridge does not expose native task-object poses or task success.

Start a fresh episode:

    PYTHONPATH=. maniskill_env/bin/python -m codeworld_robot.start \
      --workspace workspaces/milk_basket_direct_v2 \
      --episode episodes/NEW_NAME --config configs/codeworld_robot_milk_basket.json

The launcher uses the configured backend environment Python; geometry calculations can use
maniskill_env/bin/python. No robot_deps path or reconstruction-model call is needed.

State review is agent-owned: `World.assess` records affected bystanders or unknown geometry
without moving objects or invoking perception. `World.review_status` exposes unresolved
concerns; `Robot.observe` still returns sensor data, not tracked object poses. Reuse these
observations to update visible disturbances; choose further sensing only when useful.
See `codeworld_workspace/templates/AGENTS.md` for aspect-specific review and hidden interiors.

Spatial planning uses current CodeWorld geometry/poses as primary inputs; RGB-D updates
those inputs and verifies outcomes. `Robot.move_from_world(decision_file, waypoint_index,
quaternion_xyzw, max_steps=...)` executes a selected saved waypoint, rejects a stale world
revision and includes its decision reference in the receipt. It does not check full-arm
feasibility or prevent side effects. Candidate generation, contact reasoning, orientation,
review cadence and alternate planners remain under the agent's control.

Set `unlimited_episode: true` in episode configuration for long tasks. This passes
`ignore_done=True` to LIBERO/robosuite so its horizon does not terminate the episode.
Individual servo budgets still bound an attempted motion; the agent can observe and recover.

## Single- and multi-arm interface (v2)

`Robot.capabilities()` lists available arms and backend support. `state()` retains
legacy active-arm fields and adds `arms`; `state(arm=0)` reads a specific arm.
Arm identifiers are integer indices or exact advertised `arm_name` strings.
Pass `arm=...` to `move_tcp`, `gripper`, `joints`, and `move_from_world`.
Legacy calls without an arm use the active arm (initially 0); `select_arm` only
changes this default and does not reset either arm. Prefer explicit arms in new code.

```python
robot.move_tcp(target_left, arm=0)
robot.move_tcp_group([
    {'arm': 0, 'pose_W': target_left},
    {'arm': 1, 'pose_W': target_right},
])
robot.gripper_group([{'arm': 0, 'mode': 'HOLD'}, {'arm': 1, 'mode': 'CLOSE'}], steps=80)
robot.joints_group([
    {'arm': 0, 'joint_positions': left_q},
    {'arm': 1, 'joint_positions': right_q},
], duration_steps=100)
robot.check_ik(target_right, arm=1)  # optional, no motion or new capture
```

RoboDojo uses public X5 URDF IK with each robot's configured base and TCP offset.
TCP local +X is the approach direction. Do not reuse Panda tool orientations blindly.
Both arms share one joint interpolation clock and one native control packet per tick.
All target poses/IK solutions/joint limits are checked before a group starts; one
failed IK returns `IK_NOT_FOUND` with zero control steps. Uncommanded actuators keep
 their last commanded targets, including gripper closure. This does not establish that
an object is held. `HOLD` preserves the previous finger target; it does not infer a grasp.
RoboDojo `duration_steps` controls interpolation (default at least 60, enlarged for
0.006 rad/tick smoothstep increments), followed by 25 settling ticks. Its joint servo
does not use LIBERO's `max_steps` OSC budget. A TCP receipt reports per-arm measured
errors and `ARRIVED` or `TARGET_NOT_REACHED`; `EXECUTED` only means commands ran.
There is no whole-task step cap.

LIBERO retains OSC and raw `step`; singleton TCP/gripper groups work through the
same interface. Unsupported arms and duplicate arms fail before movement. LIBERO
does not implement joint targets or IK queries; discover this from the capabilities.
RoboDojo does not implement LIBERO raw OSC `step`.

Launch RoboDojo with `--config configs/codeworld_robot_robodojo.json`. The launcher
supports a Python executable **or a Python wrapper** in the `python` field. The
provided wrapper configures the existing headless Isaac runtime on l203. Configure
`task_name`, `seed`, `env_config`, and `world_origin_S_m` to match the source world;
changing the robot does not transform an existing reconstruction automatically.
The native scene must match the reconstruction used for the episode.

World geometry stays the planning source. The bridge returns robot proprioception,
RGB-D and action receipts, never simulator object poses. Codex maintains observed
object poses and predicted held-object motion through the existing World API; keep
arm identity and robot_state_ref in that evidence. No automatic attachment, handover,
per-frame reconstruction, or full-arm/inter-arm collision certification is added.
In particular, two reachable TCP targets are not proof of a collision-free path.

## RoboDojo official action budget

Set `official_evaluation: true` in the episode config to instantiate upstream
`EvalEnv` and execute every waypoint through `take_action`. The file queue replaces
only the websocket policy transport. Official reset, interpolation, counter, reward
and termination methods are unchanged. Multi-arm targets share one action. Long
commands consume multiple actions; further actuation stops when the evaluator ends.
`robot_io/official_actions.jsonl` records each action; `official_status.json` exposes
remaining budget, and `official_result.json` records the terminal result on close.
World estimates still come from reconstruction and selected RGB-D observations.
No native task-object state is supplied to the policy.

With the current ARX X5 config, dt=0.004 s and collect_freq=25 Hz, so each action
advances 10 physics steps. Videos use 25 fps with one frame per action in official
mode. The direct mode now also derives timestamps and frame stride from the actual
simulation config; old recordings made with the two-tick stride are slow motion.

## Independent Codex CLI executor

### Optional full-link motion backend

Fresh RoboDojo episodes can enable `full_arm_collision_check: true` and supply
`motion_dependencies` containing python-fcl and ruckig. This backend uses the public
X5 URDF collision meshes, all live robot joint/finger positions, and the current
reconstructed World. It checks non-adjacent self collisions, both arms including the
idle arm, environment surfaces, and agent-declared payloads along sampled joint FK.
Coverage is sampled (default maximum joint change 0.025 rad), not certified continuous
collision detection or a guarantee about unknown/incorrect reconstructed geometry.

`Robot.check_motion(commands, **options)` performs the same planned path check without
actuation and returns collision bodies/sample and estimated official actions. Commands
can contain `pose_W` or explicit `joint_positions`. `check_ik` also returns alternative
`joint_candidates`; candidate selection, task ordering and observation cadence remain
agent-owned. The new execution mode uses bounded Ruckig trajectories sampled at the
official action period, with at most `max_settle_actions` extra periods only when measured
joint error warrants them. The official evaluator still owns interpolation and counting.

Motion and gripper calls accept `allowed_contacts`, explicit pairs of body names such
as `[['arm0/link7', 'OBJECT_ID'], ['arm0/link8', 'OBJECT_ID']]`. These represent intentional
grasp/support contacts, not a global disable switch. The two fixed base/table mounting
contacts may be declared in configuration. Every other contact exception must be chosen
and recorded by the execution agent, with its physical reason in the task log.

Declare carried geometry in each relevant call with `attachments=[{'arm': 0,
'object_id': 'OBJECT_ID', 'T_tcp_object': MATRIX, 'tcp_offset_m': [0.145,0,0]}]`.
The relative pose is the agent's estimate, not an automatic grasp assertion. The payload
is removed from the static obstacle set, moved along robot FK, and checked against the
world, robot links, other arm and other payloads. Include the intentional gripper/payload
contact pairs. Without an attachment declaration the object remains a static obstacle.

This is a fixed implementation. No episode results or learned skills are imported
into future tasks. Source scene geometry is reused only when explicitly requested.

The desktop chat supervises; an independent `codex exec` process plans and operates.
Start a **fresh** episode with `codeworld_robot.start` (CLI ownership is now the default).
`--manual-diagnostic` is explicitly non-evaluation manual operation. Legacy episodes are
not retroactively protected; never relabel their results as CLI runs.

Create a transport JSON with `episode` (absolute episode path) and `python` (its Python).
For a remote episode add `ssh`, an argv array including the destination, e.g.
`["ssh", "-o", "BatchMode=yes", "l203"]`. Use the actual configured jump host when needed.
The local CLI's generated `remote.py script.py` executes its own Python in the episode,
with `episode/tools` on PYTHONPATH and the CLI run identity propagated.
It can read public files and copy observations using Python through this same bridge;
the supervisor must not supply per-step target coordinates or task programs.

```sh
python3 -m codeworld_robot.agent_cli run --config transport.json --run-dir agent_runs/NEW_RUN --model MODEL_ID
python3 -m codeworld_robot.agent_cli status --run-dir agent_runs/NEW_RUN
python3 -m codeworld_robot.agent_cli stop --run-dir agent_runs/NEW_RUN
```

Run `run` in a managed background terminal/process. Status only reads the lifecycle record;
stop asks that worker to terminate its CLI process group, revoke ownership and send only
Robot.finish(). It may wait for an in-flight server command to finish; this is not an
emergency hardware stop. Closed/assigned/unmanaged episodes are rejected. No automatic
retry, continuation, model substitution, or task resumption is performed.

Artifacts: prompt.txt, AGENTS.md, transport.json, events.jsonl, stderr.log, final.txt,
lifecycle.json (run id, requested model, CLI thread id, PID, exit status). EXITED means
CLI completion, **not task success**. CLI receipts and CodeWorld histories carry actor
identity; inspect the actual trace and decisions to establish substantive CodeWorld use.
The ownership check prevents accidental supervisor actions, not malicious access by
processes sharing the same OS account. Install updated server/client code on the simulator
host before starting a fresh episode; editing a local checkout does not update a live server.

## Panda and FR3 joint adapters

`robot_control: "joint_position"` explicitly opts LIBERO-PRO Panda into native
JointPositionController absolute joint goals. Default LIBERO OSC remains unchanged.
These controller protocols must be identified separately in benchmark comparisons.
Panda uses the loaded public MuJoCo robot geometry, including its actual mount and
LIBERO grip-position / hand-orientation convention. FetchBench binds the public
`franka_r3_cvx.urdf`, seven arm joints, two fingers and hand-local +Z 0.1034 m TCP.
Startup compares model FK against native proprioception and rejects frame mismatch.

Both adapters expose `check_ik`, `check_motion`, `joints`, `move_tcp` and gripper
commands through the same Robot client. Full-arm, non-adjacent self, environment,
explicit payload and finger sweep collision checks precede trajectories. Checks
sample joint-space paths; reconstruction error and between-sample contacts remain
outside this guarantee. Payload ownership and contact exceptions stay agent-owned.
The default seven-axis motion profile is quintic, with configurable velocity and
acceleration limits; optional Ruckig needs a matching Python runtime installation.
Queries do not advance physics. Budget exhaustion is a distinct result.

Example native diagnostic configurations:
`configs/codeworld_robot_libero_pro_panda_diagnostic.json` and
`configs/codeworld_robot_fetchbench_fr3_diagnostic.json`.
The former counts 20 Hz native policy cycles; the latter counts native physics
ticks and does not claim equivalence to a paper-level policy-action budget.
`motion_dependencies` must match the selected native Python ABI; the current
FR3 environment uses Python 3.8, while Panda uses Python 3.11.

`python -m codeworld_workspace.import_baked --export LEGACY_EXPORT --output NEW_WORLD`
imports explicitly world-baked legacy meshes with identity local transforms,
retains their lineage and marks source-program binding as unverified. It does not
rerun reconstruction or promote old geometry into new quality evidence.

FR3 geometry dependencies are pinned in `codeworld_robot/requirements-fr3.txt`.
Use the native Python with `pip install --no-deps --target TARGET -r FILE`;
trimesh 4.5.1 fixes invalid mesh indices seen with the native environment's older
3.23.5 GLB reader. Keep native simulator NumPy/Torch installations unchanged.
Fresh launchers execute the copied episode tools, so later repository edits do
not change an active episode's server implementation.

## Optional grasp geometry and measured lift feedback

`Robot.check_grasp_geometry(object_id, center_W=..., closing_axis_W=...,
approach_axis_W=..., max_opening_m=..., finger_depth_m=...,
finger_pad_height_m=..., expected_revision=...)` uses the current reconstructed
mesh and clips its triangles to the proposed finger pad band. The result is
`WIDTH_COMPATIBLE`, `INCOMPATIBLE`, or `INCONCLUSIVE`. An offset target may fail
although its total width fits. It reports required jaw opening and clearance.
Choose candidate heights/orientations and gripper dimensions from CodeWorld and
the public robot model. This query does not choose a grasp or check force closure;
use the existing IK and motion queries for robot/environment feasibility.
`World.check_grasp_geometry` exposes the same query without a connected robot.

`Robot.assess_grasp(object_id=..., before_capture=..., after_capture=...,
before_masks={view_id: mask_path}, after_masks={view_id: mask_path}, arm=...)`
reads stored RGB-D and measured TCP states. The first capture is after closure
and before an agent-chosen upward motion; the second follows that motion. Optional
`object_before_capture` uses an earlier unobstructed object baseline, with its
corresponding before_masks. For backends without last_gripper_command in captures (RoboDojo), supply
`closure_request="robot_io/requests/CLOSE_REQUEST_ID.json"`; its matching receipt
and measured counters are checked, and a later OPEN invalidates that evidence.
Masks are agent-selected binary object pixels, not
simulator segmentation; accepted formats are 2D NumPy boolean/0-1/0-255 arrays
and binary images. Paths must remain inside the episode. Depth must be calibrated
camera-z in meters. Retain the identity and mask evidence for review.

Results: `NOT_LIFTED` when the TCP rose but measured target surfaces did not;
`LIFT_EVIDENCE` when at least two stationary cameras at distinct locations agree
that the selected object rose with the TCP; `INCONCLUSIVE` for insufficient
motion/views, moved cameras, changed visibility, disagreement or TCP rotation.
The default probe evidence requires 20 mm TCP rise, 15 mm object rise, <=8 mm
co-motion discrepancy and <=5 degree TCP rotation. These are explicit adjustable
measurement criteria, not a required action sequence. Positive assessment always
requires at least two camera locations. Agent-selected masks can be incorrect;
the report is visible co-motion evidence, not proof of force closure, stable
payload attachment or long-term holding. Occluded objects may remain inconclusive.

Both functions are read-only: no robot commands, simulator stepping, automatic
World.update or attachment declaration. The agent decides whether/when to probe,
recover, observe again or declare an estimated payload. A gripper EXECUTED receipt
alone remains insufficient evidence of holding. Save the returned report before
using it to justify a carried-object estimate. Existing copied episodes do not
acquire these helpers when the repository is edited.

## CodeWorld completion contract
Independent CLI episodes now require a task/subtask condition contract before motion,
and observed CodeWorld verification before claiming completion. See the required
completion section of `codeworld_workspace/templates/AGENTS.md` for the condition API.
`Robot.verify_completion(goal_ids=None)` saves a query receipt; pass its `verification`
path to `Robot.finish`. All three server backends re-evaluate the receipt through shared
ownership dispatch. `finish(outcome='failed'|'aborted', reason=...)` closes without a
success claim. Supervisor interruption remains available. This changes the experiment
protocol; keep historical scores separate and start fresh episodes after deployment.
