"""Task-independent, bounded joint trajectories sampled at the backend action period."""
import math
import numpy as np


def joint_trajectory(starts, targets, period, max_velocity=2.0, max_acceleration=8.0,
                     duration_s=None, profile='quintic', max_jerk=80.):
    """Ruckig or quintic rest-to-rest profile; limits apply to every commanded joint.

    No minimum 60-tick duration or trailing hold. This is time parameterization,
    not collision planning. All arms use the same clock.
    """
    if not targets or not np.isfinite(period) or period <= 0:
        raise ValueError('Nonempty targets and positive period required')
    if profile == 'ruckig':
        from ruckig import Ruckig, InputParameter, Trajectory, DurationDiscretization
        arms = list(targets)
        start = np.concatenate([np.asarray(starts[i], float) for i in arms])
        target = np.concatenate([np.asarray(targets[i], float) for i in arms])
        if start.shape != target.shape or not np.isfinite([start, target]).all():raise ValueError('Invalid joint targets')
        def limit(value):
            array=np.asarray(value,float)
            return (np.tile(array,len(arms)) if array.ndim else np.full(len(start),array)).tolist()
        inp=InputParameter(len(start));inp.current_position=start.tolist();inp.current_velocity=[0.]*len(start);inp.current_acceleration=[0.]*len(start)
        inp.target_position=target.tolist();inp.target_velocity=[0.]*len(start);inp.target_acceleration=[0.]*len(start)
        inp.max_velocity=limit(max_velocity);inp.max_acceleration=limit(max_acceleration);inp.max_jerk=limit(max_jerk)
        inp.duration_discretization=DurationDiscretization.Discrete
        if duration_s is not None:inp.minimum_duration=float(duration_s)
        trajectory=Trajectory(len(start));result=Ruckig(len(start),period).calculate(inp,trajectory)
        if result.value<0:raise ValueError(f'Ruckig trajectory failed: {result}')
        count=max(1,math.ceil((trajectory.duration-1e-9)/period))
        output=[]
        for k in range(1,count+1):
            q=np.asarray(trajectory.at_time(min(k*period,trajectory.duration))[0]);offset=0;frame={}
            for arm in arms:
                size=len(starts[arm]);frame[arm]=q[offset:offset+size];offset+=size
            output.append(frame)
        return output
    if profile != 'quintic':raise ValueError('Unknown trajectory profile')
    delta = {}
    minimum = float(period)
    for arm, value in targets.items():
        start, target = np.asarray(starts[arm], float), np.asarray(value, float)
        if start.shape != target.shape or start.ndim != 1 or not np.isfinite([start, target]).all():
            raise ValueError('Invalid joint target')
        v = np.broadcast_to(np.asarray(max_velocity, float), start.shape)
        a = np.broadcast_to(np.asarray(max_acceleration, float), start.shape)
        if not np.isfinite([v, a]).all() or np.any(v <= 0) or np.any(a <= 0):
            raise ValueError('Positive finite velocity/acceleration limits required')
        d = abs(target-start)
        # max derivative of 10u^3-15u^4+6u^5: 1.875; max second: 10/sqrt(3).
        minimum = max(minimum, float(np.max(1.875*d/v)),
                      float(np.max(np.sqrt((10/math.sqrt(3))*d/a))))
        delta[arm] = target-start
    if duration_s is not None:
        if not np.isfinite(duration_s) or duration_s < minimum:
            raise ValueError(f'duration_s must be >= {minimum:.6f}')
        minimum = float(duration_s)
    count = max(1, math.ceil(minimum/period))
    frames = []
    for k in range(1, count+1):
        u = k/count
        blend = u*u*u*(10+u*(-15+6*u))
        frames.append({arm: np.asarray(starts[arm])+blend*d for arm, d in delta.items()})
    return frames


def densify_joint_path(starts, frames, max_increment=.025):
    """Sample FK along joint-space segments, keeping uncommanded arms present."""
    current = {i: np.asarray(q, float).copy() for i, q in starts.items()}
    yield {i: q.copy() for i, q in current.items()}
    for frame in frames:
        target = {i: np.asarray(frame.get(i, q), float) for i, q in current.items()}
        count = max(1, math.ceil(max(float(np.max(abs(target[i]-q))) for i, q in current.items())/max_increment))
        for k in range(1, count+1):
            yield {i: q+(target[i]-q)*(k/count) for i, q in current.items()}
        current = target
