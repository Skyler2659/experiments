"""Run only in a fresh idle simulation: small wrist motions, no manipulation task."""
import json,sys,time
from pathlib import Path
import numpy as np
from codeworld_robot import Robot
from codeworld_robot.io import atomic
from codeworld_robot.pose import transform
from codeworld_robot.x5_ik import X5IK
root=Path(sys.argv[1]);deadline=time.monotonic()+240
while not (root/'robot_io/ready.json').exists():
    if time.monotonic()>deadline:raise TimeoutError('Simulator did not become ready')
    time.sleep(1)
r=Robot(root);evidence={}
try:
    initial=r.state();caps=r.capabilities();evidence['capabilities']=caps
    assert len(initial['arms'])==2
    errors=[]
    for row,meta in zip(initial['arms'],caps['arms']):
        model=X5IK('/private/lc/research/RoboDojo/Assets/Robots/x5/X5A.urdf',base=meta['base_T_W'],tcp_offset=meta['tcp_offset_link6_m'])
        errors.append(float(np.max(abs(model.fk(row['joint_positions'])-transform(row['eef_pose_W'])))))
    evidence['fk_matrix_max_errors']=errors
    assert max(errors)<.003, errors
    cmds=[]
    for row in initial['arms']:
        q=np.asarray(row['joint_positions']);q[-1]+=.02
        cmds.append({'arm':row['arm'],'joint_positions':q.tolist()})
    receipt=r.joints_group(cmds,duration_steps=60);evidence['synchronized_joints']=receipt
    for row,cmd in zip(receipt['robot_state']['arms'],cmds):
        assert np.max(abs(np.asarray(row['joint_positions'])-cmd['joint_positions']))<.015
    right_before=r.state(1)['joint_positions']
    left=np.asarray(r.state(0)['joint_positions']);left[-1]-=.02
    evidence['left_only']=r.joints(left.tolist(),arm=0,duration_steps=60)
    evidence['inactive_arm_drift_rad']=float(np.max(abs(np.asarray(r.state(1)['joint_positions'])-right_before)))
    assert evidence['inactive_arm_drift_rad']<.01
    before=r.state()['control_steps']
    evidence['ik_query']=r.check_ik(r.state(1)['eef_pose_W'],arm=1)
    assert evidence['ik_query']['status']=='IK_FOUND'
    assert r.state()['control_steps']==before
    bad={'translation_m':[100,100,100],'quaternion_xyzw':[0,0,0,1]}
    evidence['atomic_ik_failure']=r.move_tcp_group([{'arm':0,'pose_W':r.state(0)['eef_pose_W']},{'arm':1,'pose_W':bad}])
    assert evidence['atomic_ik_failure']['status']=='IK_NOT_FOUND'
    assert r.state()['control_steps']==before
    evidence['tcp_group']=r.move_tcp_group([{'arm':i,'pose_W':r.state(i)['eef_pose_W']} for i in (0,1)])
    assert evidence['tcp_group']['status']=='ARRIVED'
    evidence['grippers']=r.gripper_group([{'arm':0,'mode':'OPEN'},{'arm':1,'mode':'HOLD'}],steps=60)
    evidence['status']='PASS'
finally:
    evidence['finish']=r.finish();atomic(root/'smoke_result.json',evidence)
print(json.dumps({'status':evidence['status'],'fk_errors':errors,'inactive_arm_drift_rad':evidence['inactive_arm_drift_rad']}))
