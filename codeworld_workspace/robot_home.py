"""Initial public robot proprioception and measured return-to-home verification."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np

GOAL_ID = "__robot_return_home"

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def capture_home(robot_state):
    arms = copy.deepcopy(robot_state["arms"])
    if not arms or len({a["arm"] for a in arms}) != len(arms):
        raise ValueError("Distinct initial robot arms required")
    for arm in arms:
        for field in ("joint_positions","gripper_qpos"):
            a=np.asarray(arm[field],float)
            if a.ndim!=1 or not len(a) or not np.isfinite(a).all():raise ValueError("Invalid initial robot coordinates")
    return {"schema":"codeworld-robot-home-v1","source":"initial_public_robot_proprioception",
            "robot_state_ref":robot_state.get("robot_state_ref"),"arms":arms,
            "joint_tolerance_rad":0.05,"gripper_tolerance_m":0.005}

def required_goal(root):
    path=Path(root)/"task/robot_home.json"
    if not path.exists():return None
    home=json.loads(path.read_text())
    return {"description":"Release carried objects safely and return every arm and gripper to its recorded episode-initial configuration. Plan collision-checked return motions through CodeWorld; finish does not move the robot.",
            "conditions":[{"kind":"robot_home","object":"known_robot","baseline_digest":digest(home)}]}

def measure_home(root, robot_state, condition):
    home=json.loads((Path(root)/"task/robot_home.json").read_text())
    if condition["baseline_digest"]!=digest(home):raise ValueError("Initial robot baseline changed")
    measured=robot_state.get("arms",[])
    ids=[a.get("arm") for a in measured]
    expected=[a["arm"] for a in home["arms"]]
    if len(ids)!=len(set(ids)) or set(ids)!=set(expected):return {"status":"INCONCLUSIVE","reason":"Missing or duplicate arm feedback"}
    byid={a["arm"]:a for a in measured};checks=[]
    for initial in home["arms"]:
        row={"arm":initial["arm"]}
        for field,tol in [("joint_positions",home["joint_tolerance_rad"]),("gripper_qpos",home["gripper_tolerance_m"])]:
            actual=np.asarray(byid[initial["arm"]].get(field,[]),float);target=np.asarray(initial[field],float)
            if actual.shape!=target.shape or not np.isfinite(actual).all():return {"status":"INCONCLUSIVE","reason":"Incomplete robot coordinates"}
            error=float(np.max(np.abs(actual-target)));row[field]={"max_error":error,"tolerance":tol,"passed":error<=tol}
        checks.append(row)
    passed=all(c[f]["passed"] for c in checks for f in ("joint_positions","gripper_qpos"))
    return {"status":"SATISFIED" if passed else "NOT_SATISFIED","arms":checks,"baseline_digest":digest(home)}
