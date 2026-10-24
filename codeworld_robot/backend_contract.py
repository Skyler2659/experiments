"""Episode-specific robot conventions; never infer one backend from another."""
CONTRACTS = {
 'libero': '''Backend: libero. Robot: Panda, one arm (arm 0).
The TCP position is the native grip site; its orientation is the native end-effector BODY orientation, not necessarily the grip-site frame or a stock Panda URDF frame.
Use capabilities.arms[0].T_link_tcp and the exported public robot URDF. Finger joint axes and collision geometry in that binding define the actual closing direction and contact surfaces. No constant approach axis is assumed. Never use RoboDojo X5's +X convention here.
Use native joint-position control and native_policy_cycle counts when advertised. Robot configuration describes only the robot, not task-object ground truth.
''',
 'robodojo': '''Backend: robodojo. Robots: two X5 arms, left arm 0 and right arm 1.
All arms and grippers must return to their measured episode-initial configuration before agent-initiated successful finish. Read task/robot_home.json and ctx.proprioception(): each arm has home_joint_positions and home_gripper_qpos. These are measured initial coordinates, not assumed zeros. Plan and execute return paths through CodeWorld, keeping collision/payload checks active. World.define_completion automatically adds the required __robot_return_home goal; verify full completion after releasing objects and returning. Robot.finish does not execute return motions.
TCP local +X is the approach direction for this backend. Each arm has its own base transform and TCP offset, advertised in capabilities.arms. Use the corresponding public X5 geometry; do not use Panda wrist/finger conventions.
Use the advertised synchronized joint commands and official action counter, not LIBERO controller-step semantics.
''',
 'fetchbench': '''Backend: fetchbench. Robot: FR3. Use the current public URDF and TCP binding from capabilities. Do not import Panda or X5 frame conventions. Read the advertised counter domain.
''',
}

def execution_instructions(backend, common):
    contract = CONTRACTS[backend]
    if backend == 'libero':
        start = common.index('RoboDojo uses public X5 URDF')
        end = common.index('## Independent Codex CLI executor', start)
        common = common[:start] + common[end:]
        start = common.index('### Optional full-link motion backend')
        end = common.index('Motion and gripper calls accept', start)
        common = common[:start] + common[end:]
    elif backend == 'robodojo':
        start = common.index('## Panda and FR3 joint adapters')
        end = common.index('## Optional grasp geometry', start)
        common = common[:start] + common[end:]
        start = common.index('For the LIBERO joint-path adapter')
        end = common.index('RoboDojo uses public X5 URDF', start)
        common = common[:start] + common[end:]
    return '# Current episode robot contract\n\n' + contract + '\n' + common
