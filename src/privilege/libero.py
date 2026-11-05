"""LIBERO observation conventions, without changing the native action interface.

CodeWorld's existing attachment schema calls the single Panda hand ``right``.
This is an internal name only; LIBERO motion primitives take no arm argument.
"""
import copy

import numpy as np


def is_libero(robo):
    return tuple(getattr(robo, 'CAMERAS', ())) == ('agentview', 'wrist')


def robot_state(state):
    """Add a wxyz attachment frame to LIBERO's public xyzw proprioception."""
    result = copy.deepcopy(state)
    q = state['eef_quat']
    result['right'] = {
        'eef_pos': list(state['eef_pos']),
        'eef_quat': [q[3], q[0], q[1], q[2]],
        'gripper': float(np.clip(state['gripper_opening'] / .08, 0., 1.)),
    }
    return result


def camera(meta, image_shape):
    """RPent stores 256px calibration and 1024px RGB with the same framing.

    extrinsic_cam2world already uses OpenCV axes, unlike RoboDojo's matrix.
    Scale intrinsics from the *recorded* image size, never from a fresh camera.
    """
    k = np.asarray(meta['intrinsic_K'], dtype=float).copy()
    height, width = image_shape[:2]
    source_height, source_width = int(meta['height']), int(meta['width'])
    if min(source_height, source_width) <= 0 or k.shape != (3, 3):
        raise ValueError('Invalid LIBERO camera calibration')
    k[0] *= width / source_width
    k[1] *= height / source_height
    return k, np.asarray(meta['extrinsic_cam2world'], dtype=float)


def action_kwargs(tool, kwargs):
    """Translate bookkeeping only; the robot receives its original arguments."""
    result = {**kwargs, 'arm': 'right'}
    if tool == 'release':
        result['gripper'] = -1
    return result


def prompt(text):
    text = text.replace('the three paired views', 'the two paired views')
    text = text.replace("held={'left': object_id} (or right)", "held={'right': object_id}")
    text = text.replace("held={'left': None}", "held={'right': None}")
    text = text.replace("'cam_head'", "'agentview'").replace("'cam_right_wrist'", "'wrist'")
    return text + '''
LIBERO binding: one Panda arm. Use right ONLY as the CodeWorld attachment key;
native robo primitives have no arm argument. robo.state().eef_quat is xyzw.
The cameras are agentview and wrist, both calibrated at the recorded step.
The robot uses native LIBERO OSC servos. CodeWorld native-path collision checking
is not installed on this host; geometry queries do not certify executed paths.
For drawers and doors, publish separate fixed/moving parts and update their
transforms from RGB after interaction. Do not move the entire cabinet/microwave
when only its drawer/door moved. Part parameters are estimates, not simulator joints.
'''
