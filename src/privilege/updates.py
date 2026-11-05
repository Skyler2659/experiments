"""Agent feedback updates with explicit observation/prediction provenance."""

import copy
import fcntl
import time

from .observations import load
from .review import stamp_review
from .world import atomic, rigid


def integrate_rgb(world, capture, proposals, *, expected_revision, note):
    if world.branch:
        raise ValueError("Publish observed estimates to the current world, not a hypothesis branch")
    if type(expected_revision) is not int or not isinstance(note, str) or not note.strip():
        raise ValueError("Provide expected_revision and explain the RGB evidence")
    path, packet, _ = load(world.root, capture)
    if packet["frame"] != world.manifest["frame"]:
        raise ValueError("Observation and world frames differ")
    if not isinstance(proposals, dict) or not proposals:
        raise ValueError("Provide object pose proposals")
    from .rgb_pose import resolve_proposals
    proposals, fit_failed = resolve_proposals(world, packet, proposals)
    normalized = {}
    for oid, proposal in proposals.items():
        meta = world.object(oid)
        if meta.get("role") == "robot_snapshot":
            raise ValueError("The initial robot asset is not a live robot estimate")
        row = {"T_W_object": rigid(proposal["T_W_object"]).tolist()}
        if 'rgb_fit' in proposal:
            row['rgb_fit'] = copy.deepcopy(proposal['rgb_fit'])
        if "source" in proposal:
            row["source"] = str(proposal["source"])
        row["time_basis"] = proposal.get("time_basis", "current")
        if row["time_basis"] not in {"current", "image"}:
            raise ValueError("time_basis must be current or image")
        for key in ("translation_observed_axes", "orientation_observed_axes"):
            axes = proposal.get(key)
            if (
                not isinstance(axes, list)
                or len(axes) != 3
                or any(type(a) is not bool for a in axes)
            ):
                raise ValueError(key + " must contain three booleans")
            row[key] = axes
        normalized[oid] = row
    with (world.root / "state/update.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = world.state()
        if state["revision"] != expected_revision:
            raise ValueError("Stale state revision; reread and recompute")
        from .tracking import align_observation, correct_pose

        deferred = {**fit_failed, **align_observation(state, normalized, packet)}
        for oid, proposal in normalized.items():
            row = state["objects"][oid]
            previous = row.get("last_observation_recorded_at_ns", 0)
            if packet["recorded_at_ns"] < previous:
                raise ValueError("Observation predates the object's last accepted observation")
            correct_pose(state, oid, proposal["T_W_object"], packet.get("robot_state"))
            row.pop('pose_refresh', None)
            row.update(
                basis="rgb_agent_estimate",
                last_observation_recorded_at_ns=packet["recorded_at_ns"],
            )
            row.pop("oracle_validation", None)
            row["evidence"] = {"note": note, "references": [str(path.relative_to(world.root))]}
            row["sensor_validation"] = {
                "kind": "rgb_agent_estimate",
                "fit_performed": 'rgb_fit' in proposal,
                "metric_validation": False,
                "observation_id": packet["observation_id"],
                "translation_observed_axes": proposal["translation_observed_axes"],
                "orientation_observed_axes": proposal["orientation_observed_axes"],
                **{k: v for k, v in proposal.items() if k in
                   ("time_basis", "image_T_W_object", "propagated_from_image", "source", "rgb_fit")},
            }
            row["motion_epoch"] = row.get("motion_epoch", 0) + 1
            stamp_review(
                row, "pose", "observed", state["revision"] + 1, row["evidence"]["references"], note
            )
        for oid, reason in deferred.items():
            from .tracking import request_refresh
            request_refresh(state, oid, reason)
            stamp_review(state["objects"][oid], "pose", "unresolved", state["revision"] + 1,
                         [str(path.relative_to(world.root))], reason)
        state["revision"] += 1
        state["history"].append(
            {
                "revision": state["revision"],
                "kind": "rgb_agent_estimate",
                "updates": copy.deepcopy(normalized),
                "deferred": deferred,
                "note": note,
                "observation_id": packet["observation_id"],
                "recorded_at_ns": time.time_ns(),
            }
        )
        atomic(world.state_path, state)
    return {
        "status": "RGB_ESTIMATE",
        "revision": state["revision"],
        "objects": list(normalized),
        "deferred": deferred,
        "metric_validation": False,
    }


def commit_step(world, capture, *, poses, unchanged, unresolved, expected_revision, note, held=None,
                disturbed=None, relations=None):
    """Commit the step review, editing affected objects and retaining the others.

    Unchanged and unresolved are object->reason mappings. Unresolved can accompany
    a usable intermediate pose (e.g. a falling object), without declaring it settled. Neither an
    unchanged assertion nor a pose proposal is independently metric-validated.
    """
    if world.branch:
        raise ValueError("Commit action feedback to the current world")
    if type(expected_revision) is not int or not isinstance(note, str) or not note.strip():
        raise ValueError("Provide expected_revision and explain the observation")
    path, packet, _ = load(world.root, capture)
    if packet["frame"] != world.manifest["frame"]:
        raise ValueError("Observation and world frames differ")
    if not all(isinstance(x, dict) for x in (poses, unchanged, unresolved)):
        raise ValueError("poses, unchanged, unresolved must be object mappings")
    disturbed = disturbed or {}
    ids = {oid for oid in world.objects() if world.object(oid).get("role") != "robot_snapshot"}
    if (not isinstance(disturbed, dict) or not set(disturbed) <= ids or
            any(not isinstance(v, str) or not v.strip() for v in disturbed.values())):
        raise ValueError('disturbed maps affected object IDs to visual reasons')
    groups = [set(poses), set(unchanged), set(unresolved)]
    supplied = set.union(*groups)
    if not supplied <= ids or set(unchanged) & (set(poses) | set(unresolved)):
        raise ValueError("Use known non-robot objects; unchanged cannot overlap poses/unresolved")
    retained = ids - supplied
    held = held or {}
    if not isinstance(held, dict) or any(oid is not None and oid not in ids for oid in held.values()):
        raise ValueError("held maps arm to a non-robot object ID or None")
    from .rgb_pose import resolve_proposals
    poses, fit_failed = resolve_proposals(world, packet, poses)
    unresolved = {**unresolved, **fit_failed}
    normalized = {}
    for oid, proposal in poses.items():
        row = {"T_W_object": rigid(proposal["T_W_object"]).tolist()}
        if 'rgb_fit' in proposal:
            row['rgb_fit'] = copy.deepcopy(proposal['rgb_fit'])
        row["time_basis"] = proposal.get("time_basis", "current")
        if row["time_basis"] not in {"current", "image"}:
            raise ValueError("time_basis must be current or image")
        for key in ("translation_observed_axes", "orientation_observed_axes"):
            axes = proposal.get(key)
            if (
                not isinstance(axes, list)
                or len(axes) != 3
                or any(type(a) is not bool for a in axes)
            ):
                raise ValueError(key + " must contain three booleans")
            row[key] = axes
        normalized[oid] = row
    if any(
        not isinstance(reason, str) or not reason.strip()
        for reason in [*unchanged.values(), *unresolved.values()]
    ):
        raise ValueError("Explain each unchanged or unresolved object using this observation")
    with (world.root / "state/update.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = world.state()
        if state["revision"] != expected_revision:
            raise ValueError("Stale state revision; reread and recompute")
        revision = state["revision"] + 1
        # integrate_rgb and commit_step may be two calls on the same capture.
        # Preserve those observed estimates before unchanged/retained review edits.
        observed_poses = {oid: {"T_W_object": copy.deepcopy(row["T_W_object"]),
                               **copy.deepcopy(row["sensor_validation"])}
            for oid, row in state["objects"].items()
            if row.get("basis") == "rgb_agent_estimate"
            and row.get("sensor_validation", {}).get("observation_id") == packet["observation_id"]}
        refs = [str(path.relative_to(world.root))]
        from .tracking import align_observation, correct_pose, request_refresh

        deferred = align_observation(state, normalized, packet, held)
        unresolved = {**unresolved, **deferred}
        for oid, reason in disturbed.items():
            request_refresh(state, oid, reason)
        for oid, reason in unresolved.items():
            request_refresh(state, oid, reason)
        for oid in sorted(ids):
            row = state["objects"][oid]
            if oid in retained:
                # Passive occlusion does not invalidate a stable object. No all-object
                # re-estimation requirement; a held object's motion was predicted already.
                row["last_step_review"] = {"observation_id": packet["observation_id"],
                                           "revision": revision, "assessment": "retained"}
                if world.object(oid).get("movable") is False and not row.get('pose_refresh'):
                    stamp_review(row, "pose", "observed", revision, refs, "Stable background confirmed unchanged")
                    row["basis"] = "confirmed_unchanged"
                continue
            if packet["recorded_at_ns"] < row.get("last_observation_recorded_at_ns", 0):
                raise ValueError("Observation predates the object's last accepted observation")
            row.pop("sensor_validation", None)
            row.pop("oracle_validation", None)
            row["last_step_review"] = {
                "observation_id": packet["observation_id"],
                "revision": revision,
            }
            if oid in normalized:
                proposal = normalized[oid]
                correct_pose(state, oid, proposal["T_W_object"], packet["robot_state"])
                row.update(basis="rgb_agent_estimate")
                row["sensor_validation"] = {
                    "kind": "rgb_agent_estimate",
                    "fit_performed": 'rgb_fit' in proposal,
                    "metric_validation": False,
                    "observation_id": packet["observation_id"],
                    **{k: v for k, v in proposal.items() if k != "T_W_object"},
                }
                row["last_observation_recorded_at_ns"] = packet["recorded_at_ns"]
                stamp_review(row, "pose", "observed", revision, refs, note)
            elif oid in unchanged:
                row["basis"] = "rgb_agent_unchanged"
                row["last_observation_recorded_at_ns"] = packet["recorded_at_ns"]
                row["sensor_validation"] = {
                    "kind": "rgb_agent_unchanged",
                    "fit_performed": False,
                    "metric_validation": False,
                    "observation_id": packet["observation_id"],
                }
                stamp_review(row, "pose", "observed", revision, refs, unchanged[oid])
            if oid in unresolved:
                stamp_review(row, "pose", "unresolved", revision, refs, unresolved[oid])
            row["evidence"] = {
                "references": refs,
                "note": unresolved.get(oid, unchanged.get(oid, note)),
            }
        from .tracking import reconcile
        if state.get('control_policy', {}).get('observation_consistency'):
            from .control_review import review
            observed_poses.update(normalized)
            review_held = {a: h['object_id'] for a, h in state.get('attachments', {}).items()}
            review_held.update(held)
            review(state, packet, observed_poses, unchanged, unresolved, review_held, relations or {})
            held = review_held

        corrected_in_capture = {oid for oid, row in state['objects'].items()
            if (row.get('sensor_validation') or {}).get('observation_id') == packet['observation_id']
            and row.get('basis') == 'rgb_agent_estimate'}
        rejected_held = reconcile(state, packet["robot_state"], held, set(normalized) | corrected_in_capture)
        # Do this after detach: a same-review landing pose resolves a release,
        # whereas a release declaration on its own leaves a persistent refresh.
        refresh_resolved = set(normalized) | (set() if state.get('control_policy', {}).get('observation_consistency') else set(unchanged))
        for oid in refresh_resolved - set(unresolved):
            state['objects'][oid].pop('pose_refresh', None)
        # Reconciliation may detach and change the refresh reason. Preserve the
        # caller's explicit temporal uncertainty alongside its intermediate pose.
        for oid, reason in unresolved.items():
            request_refresh(state, oid, reason)
        accepted_held = {arm: None if arm in rejected_held else oid for arm, oid in held.items()}
        state["revision"] = revision
        state["history"].append(
            {
                "revision": revision,
                "kind": "action_observation_update",
                "observation_id": packet["observation_id"],
                "poses": normalized,
                "unchanged": unchanged,
                "unresolved": unresolved,
                "retained": sorted(retained),
                "held": held,
                "rejected_held": rejected_held,
                "deferred": deferred,
                "fit_failed": fit_failed,
                "disturbed": disturbed,
                "note": note,
                "recorded_at_ns": time.time_ns(),
            }
        )
        atomic(world.state_path, state)
    return {
        "status": "WORLD_STEP_UPDATED",
        "revision": revision,
        "metric_validation": False,
        "updated": list(normalized),
        "deferred": deferred,
        "unchanged": list(unchanged),
        "unresolved": list(unresolved),
        "retained": sorted(retained),
        "held": accepted_held,
        "rejected_held": rejected_held,
        "grasp_review": state.get("grasp_review", {}),
        "relations": state.get("relations", {}),
        "fit_failed": fit_failed,
        "pose_refresh": {oid: row['pose_refresh'] for oid, row in state['objects'].items()
                         if row.get('pose_refresh')},
    }
