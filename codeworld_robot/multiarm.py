"""Simulator-independent synchronized joint control. No task policy."""
import numpy as np


def arm_index(value, names):
    if isinstance(value, str) and value in names:
        return names.index(value)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 0 <= value < len(names):
        raise ValueError(f'Unknown arm {value!r}; available: {names}')
    return int(value)


def resolve_commands(commands, names, active):
    if not isinstance(commands, list) or not commands:
        raise ValueError('commands must be a nonempty list')
    result = {}
    for command in commands:
        i = arm_index(command.get('arm', active), names)
        if i in result:
            raise ValueError('Duplicate arm in synchronized command')
        result[i] = command
    return result


def joint_frames(starts, targets, duration_steps=None, max_increment=.006):
    """Validate all targets before producing any frame; shared smoothstep clock."""
    checked = {}
    for i, target in targets.items():
        start = np.asarray(starts[i], dtype=float)
        target = np.asarray(target, dtype=float)
        if target.shape != start.shape or target.ndim != 1 or not np.isfinite(target).all():
            raise ValueError(f'Invalid joint target for arm {i}')
        checked[i] = (start, target)
    if not checked:
        raise ValueError('No joint targets')
    minimum = max(1, int(np.ceil(1.5 * max(np.max(abs(t-s)) for s,t in checked.values()) / max_increment)))
    if duration_steps is None:
        duration_steps = max(60, minimum)
    if isinstance(duration_steps, bool) or not isinstance(duration_steps, int) or duration_steps < minimum:
        raise ValueError(f'duration_steps must be an integer >= {minimum}')
    def frames():
        for k in range(1, duration_steps+1):
            u = k/duration_steps; u = u*u*(3-2*u)
            yield {i: s+u*(t-s) for i,(s,t) in checked.items()}
    return frames()


class CommandLatch:
    """Every simulation tick includes previous targets of uncommanded actuators."""
    def __init__(self, initial):
        self.targets = dict(initial)
    def merge(self, updates):
        self.targets.update(updates)
        return dict(self.targets)
