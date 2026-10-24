"""Fixed, inspectable instructions; contains no task-specific action programs."""

NO_WORLD = '''You are the independent execution agent for Ours minus CodeWorld.
This is a representation ablation. Read task/task.json and task/ROBOT_BACKEND.md.
Use remote.py to write and run your own Python programs inside this episode.
from codeworld_robot import Robot; robot = Robot('.')
Use Robot.observe(), Robot.state(), Robot.capabilities(), Robot.check_ik(),
Robot.check_motion(), Robot.move_tcp(), Robot.move_tcp_group(), Robot.joints(),
Robot.joints_group(), Robot.gripper(), Robot.gripper_group(), and Robot.finish().
Gripper modes are the uppercase strings OPEN, CLOSE and HOLD. TCP poses use
{'translation_m':[x,y,z], 'quaternion_xyzw':[x,y,z,w]} in robot-base W. Arm IDs
are 0 or 1; use an explicit arm or Robot.select_arm(). Read measured feedback
after commands, including rejected IK/collision requests, before resubmitting.
The calibrated initial multiview RGB-D scan is preparation/capture/capture.json.
Capture paths in that JSON are relative to its directory. RGB and camera-z depth
and camera matrices are available. Use the same observations to calculate targets
and reason directly, using ordinary Python and in-episode notes. Preserve object
identity, initial positions and both arms' state across this episode as needed.
Do not construct, load or query a persistent CodeWorld, object mesh scene or scene
graph. The bridge retains public robot IK, trajectories, robot self-collision and
inter-arm collision checks; object/environment and payload geometry are unavailable.
No World.compute_decision or CodeWorld completion contract is required in this arm.
Use observed feedback for your judgments; a controller receipt is not object success.
The native scorer decides task success independently of your finish() self-report.
The official action limit cannot be increased. Do not reset, teleport or change the
simulator, controller, ownership, evaluator or settings. Never inspect task source,
private object poses, simulator assets, layouts, scores or reward predicates.
Write task-action programs yourself. Do not read any other episode, previous result,
skill library, memory or historical task-action code. Do not save or retrieve skills.
Local scratch code and within-episode history are allowed. No cross-episode learning.
Stop when the public interface says the episode has ended. On interruption stop.
'''

NO_WORLD_BACKEND = '''Backend: robodojo. Two X5 arms: left arm 0, right arm 1.
Read task/robot_home.json and Robot.state(). Each arm advertises measured episode-
initial home_joint_positions and home_gripper_qpos. Return both arms and grippers
to that configuration when the public task requires it; do not assume zero joints.
Robot.finish() does not perform return motions. Use direct commands and measured
feedback; the official evaluator checks task completion and robot return.
TCP local +X is the approach direction. Each arm has its own base transform and
TCP offset in Robot.capabilities(). Use X5 conventions, not Panda conventions.
Synchronized commands are available. Official action counts and physics ticks are
separate; one interface request can consume several official actions.
'''

GT_LIVE = '''
This episode is the explicitly privileged GT CodeWorld arm. The server exports
native visible meshes (with analytic primitives tessellated), object identities,
and current rigid/link transforms into the same World interface. Articulated links
are separate objects with parent_articulation metadata; their poses update
independently. No joint targets or private task predicates are supplied. Read the oracle
provenance and uncertainty fields; they are truth inputs, not RGB-D estimates.
Robot.observe() refreshes the oracle after actions. Compute all decisions through
World.compute_decision and submit Robot.execute_decision exactly as in Ours.
Use this provided oracle instead of re-fitting its poses from RGB-D. You still own
goal interpretation, object relations, attachment hypotheses and all task programs.
The oracle does not expose the reward, target poses, future actions, friction or mass.
Never inspect the native simulator or evaluator directly. Final task success remains
the native scorer's decision, distinct from CodeWorld's completion verification.
'''

COMMON = '''
Read the entire task/task.json goal and task/PUBLIC_TASK.md before planning.
The public task description and completion requirements apply to this episode;
partial progress or your own completion contract does not redefine task success.
This is a new evaluation episode. Work only in this episode and the current CLI
run directory. Do not read another run, global memory, research reports or previous
task programs. Use fixed local interfaces and write your own task code. Do not
retrieve, learn or save reusable skills. Native sample IDs and scorer files are
evaluation metadata, not planning inputs. Do not modify tools or episode settings.
API conventions: gripper mode is OPEN, CLOSE or HOLD (uppercase). Each command
inside move_tcp_group.commands includes its own arm and pose_W; pose_W contains
translation_m and quaternion_xyzw. Use explicit arm IDs 0/1 for dual-arm actions.
'''


def agent_instructions(original, representation, gt_mode):
    if representation == "none":
        return NO_WORLD + COMMON
    if representation == "ground_truth" and gt_mode == "live_state":
        original = original.replace("Use only public robot observations/proprioception and reconstructed geometry.",
                                    "Use public observations and the explicitly provided GT CodeWorld.")
        return original + GT_LIVE + COMMON
    return original + COMMON
