"""Benchmark infrastructure only: fixed-robot initialization, no task policy."""
import numpy as np


def settle_scene(env):
    sim=env.sim
    qi=[];vi=[]
    for name in sim.model.joint_names:
        if not name or not name.startswith(('robot0_','gripper0_')):continue
        qa=sim.model.get_joint_qpos_addr(name);va=sim.model.get_joint_qvel_addr(name)
        qi.extend(list(range(*qa)) if isinstance(qa,tuple) else [qa])
        vi.extend(list(range(*va)) if isinstance(va,tuple) else [va])
    fixed=sim.data.qpos[qi].copy();free=np.ones(sim.model.nv,bool);free[vi]=False;quiet=0
    for step in range(1500):
        sim.data.qpos[qi]=fixed;sim.data.qvel[vi]=0;sim.forward();sim.step()
        speed=float(np.max(abs(sim.data.qvel[free]))) if free.any() else 0.
        quiet=quiet+1 if speed<.005 else 0
        if step>=499 and quiet>=50:break
    sim.data.qpos[qi]=fixed;sim.data.qvel[vi]=0;sim.forward()
    if quiet<50:raise RuntimeError('Official initial state did not settle; no evaluation launched')
    return {'simulation_steps':step+1,'quiet_steps':quiet,'max_free_dof_speed':speed,
            'method':'physics settling with robot joints fixed; same rule as source reconstruction'}
