"""Time-paired grasp predictions and pose corrections; no robot control or waits."""

import copy
import numpy as np
from scipy.spatial.transform import Rotation


def arm_pose(state, arm):
    """World pose of the end-effector link, not the fingertip TCP."""
    if arm in state:
        pos, quat = state[arm]["eef_pos"], state[arm]["eef_quat"]
    else:
        raw = np.asarray(state[f"{arm}_ee_pose"], float).reshape(-1)
        pos, quat = raw[:3], raw[3:7]
    quat = np.asarray(quat, float)
    out = np.eye(4)
    out[:3, :3] = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
    out[:3, 3] = pos
    return out


def begin_action(state, robot_state, tool, kwargs):
    """Remember pre-motion geometry; a close is a candidate, not a grasp claim."""
    tracking = state.setdefault("tracking", {})
    tracking["robot_state"] = copy.deepcopy(robot_state)
    for arm in list(tracking.get("grasp_candidates", {})):
        if _measured_empty(robot_state, arm):
            tracking["grasp_candidates"].pop(arm)
    anchors = {}
    for arm in ("left", "right"):
        if arm not in robot_state and f"{arm}_ee_pose" not in robot_state:
            continue
        ee = arm_pose(robot_state, arm).tolist()
        anchors[arm] = {oid: {"T_W_E": ee, "T_W_object": copy.deepcopy(row["T_W_object"])}
                        for oid, row in state["objects"].items()}
    tracking["action_before"] = {"robot_state": copy.deepcopy(robot_state), "anchors": anchors,
                                 "tool": tool, "kwargs": copy.deepcopy(kwargs)}
    arm = kwargs.get("arm")
    if arm in anchors and kwargs.get("gripper") == 1 and arm not in state.get("attachments", {}):
        tracking.setdefault("grasp_candidates", {}).setdefault(arm, copy.deepcopy(anchors[arm]))


def correct_pose(state, oid, transform, robot_state=None):
    """Write a correction and its grasp offset together, under the caller's lock."""
    old = np.asarray(state["objects"][oid]["T_W_object"])
    new = np.asarray(transform)
    tracking = state.setdefault("tracking", {})
    if robot_state:
        tracking["robot_state"] = copy.deepcopy(robot_state)
    robot_state = robot_state or tracking.get("robot_state", {})
    for arm, held in state.get("attachments", {}).items():
        if held["object_id"] != oid:
            continue
        if arm in robot_state or f"{arm}_ee_pose" in robot_state:
            inverse_ee = np.linalg.inv(arm_pose(robot_state, arm))
        else:
            # Legacy/standalone worlds: recover the frame represented by the old
            # mutually consistent pose and attachment, without robot IO.
            inverse_ee = np.asarray(held["T_E_object"]) @ np.linalg.inv(old)
        held["T_E_object"] = (inverse_ee @ new).tolist()
    state["objects"][oid]["T_W_object"] = new.tolist()
    for arm, candidates in tracking.get("grasp_candidates", {}).items():
        if arm in robot_state or f"{arm}_ee_pose" in robot_state:
            candidates[oid] = {"T_W_E": arm_pose(robot_state, arm).tolist(), "T_W_object": new.tolist()}


def _release(state, arm, robot_state, reason, *, preserve_pose=False):
    held = state.setdefault("attachments", {}).pop(arm, None)
    state.setdefault("tracking", {}).setdefault("grasp_candidates", {}).pop(arm, None)
    if held is None:
        return
    oid = held["object_id"]
    state.setdefault("releases", {})[oid] = {**copy.deepcopy(held), "arm": arm,
        "host_step": robot_state.get("step"), "reason": reason,
        "T_W_object": copy.deepcopy(state["objects"][oid]["T_W_object"])}
    if preserve_pose:
        return
    row = state["objects"][oid]
    request_refresh(state, oid, reason)
    row["basis"] = "release_prediction"
    row.pop("sensor_validation", None)
    row.setdefault("review", {})["pose"] = {"status": "unresolved", "revision": state["revision"] + 1,
                                              "note": "Last carried pose; free motion/settling is unobserved"}
    if reason == "measured_empty_gripper":
        row["basis"] = "grasp_loss_prediction"
        row["review"]["pose"].update(status="unresolved",
            note="Measured empty gripper; retained last carried estimate, landing pose unknown")


def request_refresh(state, oid, reason):
    """Keep affected-object work visible until an observation actually resolves it."""
    row = state['objects'][oid]
    row['pose_refresh'] = {'reason': reason, 'since_revision': state['revision'] + 1,
                           'last_estimate_only': True}
    row.pop('sensor_validation', None)
    row.setdefault('review', {})['pose'] = {'status': 'unresolved',
        'revision': state['revision'] + 1, 'note': reason}


def _measured_empty(robot_state, arm):
    # Public PyRUA ArmState: exactly 0 is fully closed on nothing. Do not
    # interpret missing/raw feedback or choose a new grasp-width threshold.
    return robot_state.get(arm, {}).get("gripper") == 0.0


def predict(state, robot_state, *, released=(), reset_candidates=()):
    """Advance held geometry; release retains the last carried pose, not a retreat target."""
    affected = []
    for arm, held in list(state.setdefault("attachments", {}).items()):
        oid = held["object_id"]
        affected.append(oid)
        if arm in released:
            _release(state, arm, robot_state, "open_command")
            continue
        if _measured_empty(robot_state, arm):
            _release(state, arm, robot_state, "measured_empty_gripper")
            continue
        row = state["objects"][oid]
        row["T_W_object"] = (arm_pose(robot_state, arm) @ np.asarray(held["T_E_object"])).tolist()
        row["basis"] = "grasp_motion_prediction"
        if state.get('control_policy', {}).get('observation_consistency'):
            previous = held.get('confirmation', {})
            if previous.get('status') == 'motion_consistent':
                held['last_motion_confirmation'] = copy.deepcopy(previous)
            held['confirmation'] = {'object_id': oid, 'status': 'unconfirmed',
                                    'reason': 'Motion prediction since last observation'}
            held['basis'] = 'provisional_payload_for_collision'
            state.setdefault('grasp_review', {})[arm] = copy.deepcopy(held['confirmation'])
        row.pop("sensor_validation", None)
        row.setdefault("review", {})["pose"] = {"status": "predicted", "revision": state["revision"] + 1}
    candidates = state.setdefault("tracking", {}).setdefault("grasp_candidates", {})
    # Empty closes and executed reopens end an unconfirmed grasp attempt.
    # Partial opening does not, by itself, prove release of a confirmed payload.
    for arm in set(candidates) | set(released) | set(reset_candidates):
        if arm in released or arm in reset_candidates or _measured_empty(robot_state, arm):
            candidates.pop(arm, None)
    state.setdefault("tracking", {})["robot_state"] = copy.deepcopy(robot_state)
    return affected


def reconcile(state, robot_state, held, corrected_objects):
    """Confirm grasps using paired anchors; preserve existing offsets on repeated held."""
    tracking = state.setdefault("tracking", {})
    tracking["robot_state"] = copy.deepcopy(robot_state)
    attachments = state.setdefault("attachments", {})
    rejected = {}
    for arm, oid in held.items():
        if arm not in {"left", "right"}:
            raise ValueError("Held arm must be left or right")
        if oid is None:
            previous = attachments.get(arm, {}).get("object_id")
            _release(state, arm, robot_state, "agent_release_or_slip",
                     preserve_pose=previous in corrected_objects)
            continue
        if oid not in state["objects"]:
            raise ValueError("Unknown held object")
        if _measured_empty(robot_state, arm):
            # An agent's repeated held declaration cannot recreate a rigid
            # attachment contradicted by the measured empty hand. A supplied
            # visual pose still stands; nothing here pauses or rejects motion.
            previous = attachments.get(arm, {}).get("object_id")
            _release(state, arm, robot_state, "measured_empty_gripper",
                     preserve_pose=previous in corrected_objects)
            rejected[arm] = {"object_id": oid, "reason": "measured_empty_gripper"}
            continue
        check = state.get('grasp_review', {}).get(arm, {}) if state.get('control_policy', {}).get('observation_consistency') else None
        if check is not None and check.get('status') == 'contradicted':
            _release(state, arm, robot_state, 'observed_motion_contradicts_grasp',
                     preserve_pose=oid in corrected_objects)
            rejected[arm] = check
            continue
        if attachments.get(arm, {}).get("object_id") == oid:
            if check is not None:
                if check.get('status') == 'motion_consistent':
                    attachments[arm]['confirmation'] = copy.deepcopy(check)
                    attachments[arm]['basis'] = 'observation_consistent_grasp'
                else:
                    previous = attachments[arm].get('confirmation', {})
                    if previous.get('status') == 'motion_consistent':
                        attachments[arm]['last_motion_confirmation'] = copy.deepcopy(previous)
                    attachments[arm]['confirmation'] = copy.deepcopy(check)
                    attachments[arm]['basis'] = 'provisional_payload_for_collision'
                    rejected[arm] = check
            continue
        if arm in attachments:
            _release(state, arm, robot_state, "agent_changed_held_object",
                     preserve_pose=attachments[arm]["object_id"] in corrected_objects)
        row = state["objects"][oid]
        candidate = tracking.get("grasp_candidates", {}).get(arm, {}).get(oid)
        # action_before may describe an open/retreat or an invalidated empty
        # close. Only a surviving grasp candidate can supply a historical pair.
        if oid in corrected_objects or candidate is None:
            offset = np.linalg.inv(arm_pose(robot_state, arm)) @ np.asarray(row["T_W_object"])
            source = "current_pose"
        else:
            offset = np.linalg.inv(np.asarray(candidate["T_W_E"])) @ np.asarray(candidate["T_W_object"])
            row["T_W_object"] = (arm_pose(robot_state, arm) @ offset).tolist()
            row["basis"] = "grasp_motion_prediction"
            row.pop("sensor_validation", None)
            row.setdefault("review", {})["pose"] = {"status": "predicted", "revision": state["revision"] + 1}
            source = "paired_grasp_candidate"
        attachments[arm] = {"object_id": oid, "T_E_object": offset.tolist(),
                            "basis": "agent_confirmed_grasp", "initialization": source}
        if check is not None:
            attachments[arm]['confirmation'] = copy.deepcopy(check)
            attachments[arm]['basis'] = ('observation_consistent_grasp' if check.get('status') == 'motion_consistent'
                                        else 'provisional_payload_for_collision')
            if check.get('status') != 'motion_consistent':
                rejected[arm] = check
        state.get("releases", {}).pop(oid, None)
    ids = [h["object_id"] for h in attachments.values()]
    if len(ids) != len(set(ids)):
        raise ValueError("One object cannot be attached to two arms in this rigid grasp model")
    return rejected


def align_observation(state, proposals, packet, held=None):
    """Align image-time corrections; retain predictions when a release crosses the image.

    Returns deferred corrections instead of blocking the action loop. The next
    useful observation can correct the landing pose through the usual update API.
    """
    current = packet.get("robot_state", {})
    held = {arm: None if oid is not None and _measured_empty(current, arm) else oid
            for arm, oid in (held or {}).items()}
    attached = {arm: item["object_id"] for arm, item in state.get("attachments", {}).items()}
    released_now = {oid for arm, oid in attached.items() if arm in held and held[arm] is None}
    attached.update(held)
    metadata = packet.get("metadata", {})
    image_state = metadata.get("image_robot_state")
    step = metadata.get("host_step")
    image_step = metadata.get("image_host_step")
    delayed = any(v.get("timing", {}).get("pictures_behind", 0) for v in packet.get("views", []))
    deferred = {}
    for oid, proposal in list(proposals.items()):
        if proposal.get("time_basis", "current") != "image":
            continue
        event = state.get("releases", {}).get(oid)
        # A fresh post-action image can show the slip reported in THIS review.
        # Merely declaring held=None must not discard that landing correction.
        crossed_release = oid in released_now and delayed
        if event and oid not in attached.values():
            release_step = event.get("host_step")
            if image_step is not None and release_step is not None:
                crossed_release = image_step < release_step or (image_step == release_step and delayed)
            elif step is None or release_step is None or step <= release_step:
                arm = event["arm"]
                changed = (image_state and not np.allclose(arm_pose(image_state, arm), arm_pose(current, arm)))
                crossed_release = bool(delayed or changed)
        if crossed_release:
            deferred[oid] = "Image predates or overlaps release; retained last pose prediction"
            del proposals[oid]
            continue
        arms = [arm for arm, item in attached.items() if item == oid]
        if arms and image_state:
            arm = arms[0]
            proposal["image_T_W_object"] = copy.deepcopy(proposal["T_W_object"])
            proposal["T_W_object"] = (arm_pose(current, arm) @ np.linalg.inv(arm_pose(image_state, arm)) @
                                      np.asarray(proposal["T_W_object"])).tolist()
            proposal["propagated_from_image"] = True
    return deferred
