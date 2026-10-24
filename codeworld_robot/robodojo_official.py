"""File-policy transport around RoboDojo's unmodified evaluation environment.

Only the websocket model transport is replaced. Reset, action interpolation,
action counting, reward evaluation and episode termination remain upstream.
"""
import json
import math
from pathlib import Path
import numpy as np
from .io import atomic


class FilePolicyTransport:
    def __init__(self, **kwargs):
        pass

    def call(self, func_name, **kwargs):
        if func_name != 'reset':
            raise RuntimeError('Policy actions are supplied through the episode file queue')
        return None

    def close(self):
        pass


def create_official_env(cfg, app, seed, *, managed_loop=False, result_directory=None, policy_name='codeworld_file_policy'):
    from src.eval_client import eval_env
    from omegaconf import OmegaConf
    cfg.eval_cfg.policy_name = policy_name
    cfg.eval_cfg.eval_num = 1
    cfg.deploy_cfg = OmegaConf.create({'port': 1, 'policy_name': policy_name})
    original = eval_env.WsModelClient
    try:
        eval_env.WsModelClient = FilePolicyTransport
        env = eval_env.create_eval_env(cfg, app)
    finally:
        eval_env.WsModelClient = original
    if result_directory is not None:
        # Keep native evaluator files outside the policy workspace.
        env.save_dir = str(Path(result_directory).resolve())
        env._stream_dir = str(Path(env.save_dir) / '_stream')
        Path(env.save_dir).mkdir(parents=True, exist_ok=True)
    env.reset(seed=[seed])
    # The official run_eval registers these before invoking the policy.
    if not managed_loop:
        env.run_reward()
        if hasattr(env, 'get_score'):
            env.get_score()
    return env


class OfficialActions:
    def __init__(self, env, directory, *, expose_outcome=True):
        self.env = env
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.expose_outcome = expose_outcome
        self.interval = max(1, int(env.obs_manager.collect_interval))
        self.dt = float(env.dt)
        self.joints = {i: env.robot_manager.get_joint(r)[0].copy()
                       for i, r in enumerate(env.robot_manager.robot_list)}
        self.grippers = {}
        for i, r in enumerate(env.robot_manager.robot_list):
            q = float(env.robot_manager.get_end_effector_real_val(r)[0][0])
            lo, hi = r.gripper_scale
            val = (q-lo)/(hi-lo)
            self.grippers[i] = float(np.clip(val if r.gripper_move['sign'] == 1 else 1-val, 0, 1))
        self.write_status()

    @property
    def ended(self):
        return bool(self.env.end_flag[0])

    def status(self, *, private=False):
        result = {'official_actions': int(self.env.take_action_cnt[0]),
                'official_action_limit': int(self.env.step_lim),
                'remaining_actions': max(0, int(self.env.step_lim-self.env.take_action_cnt[0])),
                'physics_steps_per_action': self.interval,
                'physics_dt_s': self.dt,
                'simulation_time_s': float(self.env.take_action_cnt[0]*self.interval*self.dt),
                'terminated': self.ended}
        if private or self.expose_outcome:
            result['official_success'] = bool(self.env.success[0]) if self.ended else None
        return result

    def write_status(self):
        atomic(self.directory/'official_status.json', self.status())

    def take(self, joints=None, grippers=None):
        if self.ended:
            return False
        self.joints.update(joints or {})
        self.grippers.update(grippers or {})
        rm = self.env.robot_manager
        action = {}
        for i, robot in enumerate(rm.robot_list):
            action[rm.process_name(robot.arm_name)] = np.asarray(self.joints[i]).tolist()
            action[rm.process_name(robot.gripper_name)] = [self.grippers[i]]
        before = int(self.env.take_action_cnt[0])
        self.env.take_action(action)
        after = int(self.env.take_action_cnt[0])
        if after != before + 1:
            raise RuntimeError(f'Official action counter did not increment exactly once: {before}->{after}')
        with (self.directory/'official_actions.jsonl').open('a') as f:
            f.write(json.dumps({'action_index': after, 'action': action, **self.status()})+'\n')
        self.write_status()
        return True

    def finalize(self):
        result = self.status(private=True)
        result['process_score'] = ([float(x) for x in self.env.reward_manager.get_score()]
                                   if hasattr(self.env, 'get_score') else [])
        details = getattr(self.env, 'eval_result', {}).get('details', {})
        result['native_details'] = details
        result['native_result_path'] = str(Path(self.env.save_dir) / '_result.json') if hasattr(self.env, 'save_dir') else None
        result['outcome'] = ('SUCCESS' if result['official_success'] else 'FAILURE') if self.ended else 'STOPPED_EARLY'
        atomic(self.directory/'official_result.json', result)
        return result
