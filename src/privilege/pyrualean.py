"""Read-only observation adapter and world injection for PyRUA-Lean.

Nothing here issues robot actions, changes primitives or decides task success.
"""

import copy
import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np

from .interaction import WorldInteraction
from .observations import load, register_views
from .reconstruction import reconstruct
from .robot_assets import export_robot
from .world import atomic, read, rigid


def plain(value):
    if is_dataclass(value):
        return plain(asdict(value))
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value.tolist() if hasattr(value, "tolist") else value


def capture(robo, root, frame):
    """Archive all host cameras with image-time calibration, without motion."""
    state = plain(robo.state())
    step = state["step"]
    views = []
    image_robot_state = None
    for camera in getattr(robo, "CAMERAS", ("cam_head",)):
        image = robo.image(camera, step=step)
        meta = robo.camera_meta(camera, step=step)
        if meta.get("image_robot_state"):
            image_robot_state = meta["image_robot_state"]
        if meta.get("shape") and tuple(meta["shape"][:2]) != tuple(np.asarray(image).shape[:2]):
            raise ValueError("Camera calibration resolution differs from captured RGB")
        transform = rigid(meta["extrinsic_matrix"]) @ np.diag([1., -1., -1., 1.])
        views.append({"view_id": camera, "image": image, "K": meta["intrinsic_matrix"],
                      "T_W_C": transform, "timing": meta.get("timing", {"host_step": step})})
    if plain(robo.state())["step"] != step:
        raise ValueError("Host moved during observation capture")
    return register_views(root, views, frame=frame, robot_state=state,
                          metadata={"host": "pyrualean", "host_step": step,
                                    "image_robot_state": image_robot_state,
                                    "calibration_source": "robo.camera_meta_image_time"})


def check_recovery_observation(reference, current_root, current_capture):
    """Check a restarted episode before reusing its interrupted reconstruction.

    This compares permitted observations only; it is not geometry validation.
    Small renderer noise is tolerated, but calibration and robot state must match.
    """
    from PIL import Image

    reference = Path(reference).resolve()
    _, before, before_image = load(reference.parents[2], reference)
    _, after, after_image = load(current_root, current_capture)

    def same(a, b):
        if isinstance(a, dict):
            return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
        if isinstance(a, (list, int, float)) and not isinstance(a, bool):
            try:
                aa, bb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
                return aa.shape == bb.shape and bool(np.allclose(aa, bb, atol=1e-5, rtol=0))
            except (TypeError, ValueError):
                return False
        return a == b

    if before['frame'] != after['frame'] or not same(before['robot_state'], after['robot_state']):
        raise ValueError('Recovery observation has different frame or public robot state')
    for key in ('K', 'T_W_C', 'image_size', 'camera_axes'):
        if not same(before['views'][0][key], after['views'][0][key]):
            raise ValueError('Recovery observation has different camera calibration: ' + key)
    with Image.open(before_image) as a, Image.open(after_image) as b:
        difference = np.abs(np.asarray(a.convert('RGB'), dtype=float) - np.asarray(b.convert('RGB'), dtype=float))
    mae = float(difference.mean())
    changed_fraction = float((difference.max(axis=2) > 32).mean())
    report = {'source_capture': str(reference), 'current_capture': current_capture,
              'rgb_mae_255': mae, 'rgb_changed_fraction_gt32': changed_fraction,
              'max_rgb_mae_255': 1.0, 'max_changed_fraction_gt32': 0.01,
              'matched': mae <= 1.0 and changed_fraction <= 0.01,
              'metric_geometry_validation': False}
    atomic(Path(current_root) / 'recovery_observation_check.json', report)
    if not report['matched']:
        raise ValueError('Restarted RGB differs from reconstruction source; refusing recovery reuse')
    return report


class AgentWorld:
    """World surface supplied to an agent; the robot remains the separate `robo`."""

    def __init__(self, world, robo, scratch):
        self._world = world
        self._robo = robo
        self._scratch = Path(scratch).resolve()
        self._loop = WorldInteraction(world, self._observe_raw, lambda: plain(robo.state()))
        self._requested_panels = None
        self._collision = None
        self._perception_mode = 'hybrid'  # standalone diagnostic API; prepare sets run policy

    def _configure_perception(self, mode):
        if mode not in {'codeworld', 'hybrid'}:
            raise ValueError('perception must be codeworld or hybrid')
        self._perception_mode = mode
        # RGB back_project internally invokes the same SAM/locate measurement.
        # Close that alternate route as well; baseline robots are untouched.
        self._robo._disabled_perception_tools = (
            frozenset({'locate', 'back_project', 'segment'}) if mode == 'codeworld' else frozenset())

    def _enable_collision(self, config):
        from .collision import SceneCollision, x5_spec
        install = getattr(self._robo._backend, 'set_motion_checker', None)
        if install is None:
            raise ValueError('This backend cannot check native motion paths')
        spec = config.get('robot_spec') or x5_spec(config['urdf'])
        self._collision = SceneCollision(self._world, spec,
            max_joint_increment=config.get('max_joint_increment_rad', .025),
            max_finger_increment_m=config.get('max_finger_increment_m', .002))
        install(self._collision.check)

    def expect_contacts(self, *, grasp=None, place=None, contacts=None):
        """Name intended contacts for the NEXT primitive, e.g. grasp={'right':'block'}.

        Grasp permits only that hand's fingers against the named object. Place
        permits the held object at the named support's top at the path endpoint
        (at most 2 mm vertical overlap). Arm bodies and other obstacles still count.
        General contacts=[{'first': selector, 'second': selector, 'reason': ...}]
        permits contact throughout this primitive, independent of direction/task.
        A selector is {'object': id, 'part': id} or {'object': id, 'bounds':
        [local_min, local_max]}, or {'arm': arm, 'link': link}. Optional bounds
        on a part/link are in that part/link frame; otherwise in object frame.
        Unselected surfaces/bodies remain checked. Contact is not a force or
        stability guarantee. Intent expires after this primitive, even a refusal.
        """
        if self._collision is None:
            raise ValueError('Scene collision checking is not configured')
        self._collision.expect_contacts(grasp=grasp, place=place, contacts=contacts)
        return {'expected_contacts': copy.deepcopy(self._collision.contacts)}

    def check_joint_path(self, frames, *, grippers=None):
        """Check joint waypoints [{arm: six_radians}, ...] without moving.

        Both arms use measured start joints; held meshes follow current grasp
        transforms. Optional grippers maps arms to normalized final opening 0..1.
        Result names collisions and world revision; CLEAR is sampled model clearance.
        """
        if self._collision is None:
            raise ValueError('Scene collision checking is not configured')
        live = plain(self._robo.state())
        raw = {}
        for arm in ('left', 'right'):
            raw[f'{arm}_arm_joint_state'] = live[arm]['joints']
            raw[f'{arm}_ee_joint_state'] = [live[arm]['gripper']]
        actions = []
        if set(grippers or {}) - {'left', 'right'}:
            raise ValueError('Gripper targets use left/right arm names')
        for frame in frames:
            if set(frame) - {'left', 'right'}:
                raise ValueError('Waypoints use left/right arm names')
            action = {f'{arm}_arm_joint_state': q for arm, q in frame.items()}
            action.update({f'{arm}_ee_joint_state': [value] for arm, value in (grippers or {}).items()})
            actions.append(action)
        return self._collision.check(raw, actions)

    def check_layout(self, poses):
        """Query proposed object transforms against fixed hardware, without motion.

        poses maps object IDs to proposed T_W_object matrices. Check every layer
        of a planned assembly. CLEAR does not establish stability or arm reach;
        check_joint_path covers the robot and carried objects along a path.
        """
        if self._collision is None:
            raise ValueError('Scene collision checking is not configured')
        return self._collision.check_layout(poses)

    def preview_plan(self, stages, *, access_margin_m=.04):
        """Forecast access to objects needed in later stages without changing the scene.

        stages=[{'name': ..., 'poses': {id: T_W_object}, 'access': [id, ...]}, ...].
        Envelopes flag proximity only; choose rearrangement and verify robot paths yourself.
        """
        from .planning import preview_plan
        return preview_plan(self._world, stages, access_margin_m=access_margin_m)

    def read_context(self):
        """Read the current scene before each action; unlocks only this revision/host step."""
        return self._loop.read_context()

    def commit_step(
        self, capture, *, poses=None, unchanged=None, unresolved=None, held=None, disturbed=None,
        expected_revision, note, relations=None
    ):
        """Review this step; update affected objects, retain others, confirm grasps via held."""
        from .updates import commit_step

        self._loop.check_capture(capture)
        result = commit_step(
            self._world,
            capture,
            poses=poses or {},
            unchanged=unchanged or {},
            unresolved=unresolved or {},
            expected_revision=expected_revision,
            note=note,
            held=held,
            disturbed=disturbed,
            relations=relations,
        )
        self._loop.committed()
        return result

    def _filter_action_feedback(self, tool, payload):
        if tool == 'grip' and self._world.state().get('control_policy', {}).get('observation_consistency'):
            payload = dict(payload)
            heuristic = payload.get('holding')
            payload['holding_heuristic'] = heuristic
            payload['holding'] = None if heuristic else False
            payload['holding_evidence'] = 'aperture_only; consult CodeWorld grasp_review after paired observation'
        return payload

    def _before_action(self, tool, kwargs):
        result = self._loop.before_action(tool, kwargs)
        if self._collision is not None:
            self._collision.begin_action()
        return result

    def _after_action(self, record):
        try:
            return self._loop.after_action(record)
        finally:
            if self._collision is not None:
                self._collision.end_action()

    def _action_rejected(self, record):
        self._loop._record_recovery(record.to_dict())
        self._loop.action_rejected()
        if self._collision is not None:
            self._collision.end_action()

    def _before_finish(self):
        return self._loop.before_finish()

    def _tool_feedback(self):
        context, images = self._loop.feedback()
        if self._collision is not None:
            context['collision_check'] = self._collision.last_report
        if self._requested_panels is not None:
            images, self._requested_panels = self._requested_panels, None
        return context, images

    def _observe_raw(self):
        return capture(self._robo, self._world.root, self._world.manifest["frame"])

    def objects(self):
        return self._world.objects()

    def state(self):
        return self._world.state()

    def object(self, object_id):
        return self._world.object(object_id)

    def snapshot(self):
        return self._world.snapshot()

    @property
    def live(self):
        """Numeric scene expressions that retain dependencies through arithmetic."""
        from .query import LiveQuery
        return LiveQuery(self._world)

    def compute(self, compute):
        """Reusable finite scalar/vector/matrix/path computed from one snapshot."""
        from .query import DerivedValue
        return DerivedValue(self._world, compute)

    def target(self, compute):
        """Cache xyz computed from a snapshot; refresh only its changed dependencies.

        Pass the result directly to robo.move_to. A copied list/array is a frozen
        value and cannot retain this dependency relationship.
        """
        from .query import DerivedTarget
        return DerivedTarget(self._world, compute)

    def mesh(self, object_id):
        return self._world.mesh(object_id)

    def bounds(self, object_id):
        return self._world.bounds(object_id)

    def parts(self, object_id):
        return self._world.parts(object_id)

    def part_mesh(self, object_id, part_id):
        return self._world.part_mesh(object_id, part_id)

    def points(self, object_id, points_local):
        return self._world.points(object_id, points_local)

    def nearest_surface(self, object_id, points_W):
        return self._world.nearest_surface(object_id, points_W)

    def aabb_relation(self, first, second):
        return self._world.aabb_relation(first, second)

    def observe(self):
        """Save main and both wrist RGB with image-time camera calibration."""
        cap = self._loop.refresh_observation()
        if cap is None:
            raise RuntimeError("Could not capture calibrated RGB; see CodeWorld feedback")
        return cap

    def compare_views(self, capture, object_ids=None):
        """Render this world with recorded cameras; show/save real-vs-world panels."""
        from .comparison import compare

        report, panels = compare(self._world, capture, object_ids=object_ids)
        self._requested_panels = panels
        return report

    def reconcile_location(self, object_id, estimate, *, tolerance_m=None, estimate_step=None, frame=None):
        """Compare PyRUA locate against current world without privileging either estimate.

        Compare footprint center XY and top_z, NOT table-relative center Z
        or height_m with the object's physical center/thickness. Call directly
        after locate; pass estimate_step when reusing an earlier result.
        Does not move the robot or change geometry. Real matched views correct the world.
        """
        if self._perception_mode == 'codeworld':
            raise RuntimeError('Location estimates are disabled; use RGB pose updates')
        estimate = plain(estimate)
        obj = self._world.snapshot()
        current_step = plain(self._robo.state())["step"]
        if estimate_step is not None and estimate_step != current_step:
            raise ValueError("locate estimate predates host motion; obtain a current estimate")
        if frame is not None and frame != self._world.manifest["frame"]:
            raise ValueError("locate and CodeWorld must use the same frame")
        mesh = obj.mesh(object_id)
        time_alignment = "current_world_assumed_static_since_image"
        if estimate.get("camera"):
            meta = self._robo.camera_meta(estimate["camera"], step=current_step)
            image_state = meta.get("image_robot_state")
            if image_state:
                from .tracking import arm_pose
                for arm, attachment in obj._state.get("attachments", {}).items():
                    if attachment["object_id"] == object_id:
                        image_pose = arm_pose(image_state, arm) @ np.asarray(attachment["T_E_object"])
                        mesh.apply_transform(image_pose @ np.linalg.inv(obj.transform(object_id)))
                        time_alignment = "held_prediction_at_image_time"
                        break
        bounds = np.asarray(mesh.bounds)
        from trimesh.bounds import oriented_bounds_2D
        footprint_transform, _ = oriented_bounds_2D(np.asarray(mesh.vertices)[:, :2])
        center_xy = np.linalg.inv(footprint_transform)[:2, 2]
        threshold = float(tolerance_m) if tolerance_m is not None else max(.02, .25 * float(np.max(bounds[1] - bounds[0])))
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("tolerance_m must be positive")
        delta = {}
        if estimate.get("center_xyz") is not None:
            delta["footprint_center_distance_m"] = float(np.linalg.norm(np.asarray(estimate["center_xyz"])[:2] - center_xy))
        if estimate.get("top_z") is not None:
            delta["top_difference_m"] = abs(float(estimate["top_z"]) - float(bounds[1, 2]))
        conflict = any(d > threshold for d in delta.values())
        return {"object_id": object_id, "revision": obj.revision, "conflict": conflict,
                "threshold_m": threshold, **delta,
                "preferred": "no_default" if conflict else ("no_large_conflict" if delta else "not_comparable"),
                "host_step": current_step, "frame": self._world.manifest["frame"],
                "time_basis": "caller_current_locate" if estimate_step is None else "checked_host_step",
                "geometry_time_alignment": time_alignment,
                "codeworld": {"footprint_center_xy": center_xy.tolist(), "top_z": float(bounds[1, 2]),
                              "review": self._world.state()["objects"][object_id].get("review")},
                "locate": estimate,
                "rule": "Measured RGB overrides either estimate after camera/time alignment; correct pose first."}

    def integrate_location(self, capture, object_id, estimate, *, expected_revision, note,
                           estimate_step=None):
        """Apply an agent-identified locate estimate as a translation-only correction.

        Reuse the existing capture at the locate step. Fit footprint XY and top_z
        to the existing shape, preserving rotation; center Z/height_m are not
        physical center/thickness. The agent chooses identity and usable estimates.
        No locate call, new capture, motion, or independent accuracy claim is made.
        """
        if self._perception_mode == 'codeworld':
            raise RuntimeError('Location estimates are disabled; use RGB pose updates')
        estimate = plain(estimate)
        _, packet, _ = load(self._world.root, capture)
        step = packet.get("metadata", {}).get("host_step")
        if estimate_step is not None and estimate_step != step:
            raise ValueError("locate estimate and capture must refer to the same host step")
        if not estimate.get("found", True) or (estimate.get("center_xyz") is None and estimate.get("top_z") is None):
            return {"status": "NO_LOCATION", "objects": [], "revision": self._world.state()["revision"]}
        snapshot = self._world.snapshot()
        transform = snapshot.transform(object_id)
        mesh = snapshot.mesh(object_id)
        image_state = packet.get("metadata", {}).get("image_robot_state")
        if image_state:
            from .tracking import arm_pose
            for arm, attachment in snapshot._state.get("attachments", {}).items():
                if attachment["object_id"] == object_id:
                    image_pose = arm_pose(image_state, arm) @ np.asarray(attachment["T_E_object"])
                    mesh.apply_transform(image_pose @ np.linalg.inv(transform))
                    transform = image_pose
                    break
        from trimesh.bounds import oriented_bounds_2D
        footprint, _ = oriented_bounds_2D(np.asarray(mesh.vertices)[:, :2])
        delta = np.zeros(3)
        axes = [False, False, False]
        if estimate.get("center_xyz") is not None:
            delta[:2] = np.asarray(estimate["center_xyz"], float)[:2] - np.linalg.inv(footprint)[:2, 2]
            axes[:2] = [True, True]
        if estimate.get("top_z") is not None:
            delta[2] = float(estimate["top_z"]) - mesh.bounds[1, 2]
            axes[2] = True
        transform[:3, 3] += delta
        return self.integrate_rgb(capture, {object_id: {
            "T_W_object": transform.tolist(), "time_basis": "image",
            "translation_observed_axes": axes, "orientation_observed_axes": [False] * 3,
            "source": "pyrualean_locate",
        }}, expected_revision=expected_revision, note=note)

    def integrate_rgb(self, capture, proposals=None, *, poses=None, expected_revision, note):
        """Submit RGB estimates: integrate_rgb(capture, proposals, expected_revision=..., note=...).

        poses= is an alias for proposals, matching commit_step's spelling.
        """
        if proposals is not None and poses is not None:
            raise ValueError('Specify proposals or poses, not both')
        proposals = poses if proposals is None else proposals
        if proposals is None:
            raise ValueError('Supply object pose proposals, e.g. poses={object_id: proposal}')
        _, packet, _ = load(self._world.root, capture)
        if packet.get("metadata", {}).get("host_step") != plain(self._robo.state())["step"]:
            raise ValueError("Capture predates host motion; observe again before updating")
        return self._world.integrate_rgb(
            capture, proposals, expected_revision=expected_revision, note=note
        )

    def update(self, updates, *, basis="agent_estimate", note=None, expected_revision=None):
        if basis not in {"agent_estimate", "action_prediction"}:
            raise ValueError("Use integrate_rgb for observed poses, or fork for hypotheses")
        return self._world.update(
            updates, basis=basis, note=note, expected_revision=expected_revision
        )

    def record_feedback(self, affected_objects, *, note, result=None, expected_revision=None):
        return self._world.record_feedback(
            affected_objects, note=note, result=plain(result), expected_revision=expected_revision
        )

    def publish_parts(self, object_id, parts, **options):
        parts = copy.deepcopy(parts)
        for spec in parts.values():
            if "source_faces" in spec:
                continue
            path = (self._scratch / spec["mesh"]).resolve()
            path.relative_to(self._scratch)
            spec["mesh"] = str(path)
        return self._world.publish_parts(object_id, parts, **options)

    def update_parts(self, object_id, transforms=None, **options):
        return self._world.update_parts(object_id, transforms, **options)

    def fork(self, name):
        api = AgentWorld(self._world.fork(name), self._robo, self._scratch)
        api._perception_mode = self._perception_mode
        return api

    def set_hypothesis(self, updates, expected_revision):
        return self._world.set_hypothesis(updates, expected_revision)

    def review_status(self, object_ids=None):
        return self._world.review_status(object_ids)

    def export_scene(self, output):
        path = (self._scratch / output).resolve()
        path.relative_to(self._scratch)
        path.parent.mkdir(parents=True, exist_ok=True)
        return self._world.export_scene(path)


PROMPT = """
You also have `world`, a persistent reconstructed geometry workspace. Robot actions,
native actions and episode evaluation use the original `robo`; CodeWorld qualifies grasp evidence
and intercepts unsafe home fallback. The original finish tool retains its evaluator.
Read `world.objects()` and `world.object(id)`. `world.mesh(id)` returns a trimesh mesh
(vertices/faces in world meters); `world.bounds(id)` returns bounds_W_m and revision.
Use `s = world.snapshot()` for one consistent estimate: s.mesh(id, part_id=None),
s.bounds(id, part_id=None), s.transform(id, part_id=None), s.position(id),
s.local_point(id, point, part_id=None), s.uncertainty(id), s.revision. Write normal
Python/NumPy computations directly in your cell. Keep reusable scene computations live:
goal = world.live.position(object_id) + [0, 0, .08]
orientation = world.live.transform(object_id)[:3, :3]
Numeric arithmetic, slicing, matrix products and NumPy operations such as stack/inv
retain dependencies. For arbitrary numeric computation (including scalar angles or
waypoint arrays), use expr = world.compute(lambda s: ...), querying the supplied s.
world.target(lambda s: ...) is the xyz-only shorthand. Pass live xyz expressions
directly to robo.move_to; expr.value() resolves other numeric results. Reusing path[i]
preserves the expression; converting the whole path to a list before multiple actions
freezes it. Only queried dependencies changing triggers recomputation; unrelated scene
updates do not. Operands are evaluated together from one current snapshot.
Do not capture old world arrays inside the callback. snapshot(), np.asarray(expr),
list(expr), float(expr), expr.value() and iteration results are intentional numeric
snapshots; keep the expression for reuse. Python branch decisions are evaluated now;
put state-dependent branching inside world.compute when reusing its result later.
Fixed task-space goals can remain constants. Geometry is estimated, and hidden surfaces
may be reconstructed hypotheses.
The initial robot mesh is a frozen reference; current robot state comes from robo.state().

The host captures post-action RGB and requires the scene review described below.
Pose entries use {'T_W_object': your_4x4_matrix,
'translation_observed_axes': [bool, bool, bool],
'orientation_observed_axes': [bool, bool, bool]}; use actual booleans for axes supported
by the observation. An explicit transform is your estimate; image_points proposals use
the RGB fitter described below. Neither certifies identity or metric accuracy.
`world.update({id: T}, basis='action_prediction', note=...)` records only a
prediction and does not complete the required post-action update. Reuse unaffected geometry.

For local changes use world.parts(id), world.part_mesh(id, part_id),
world.publish_parts(id, {part_id: {'mesh': 'workspace-relative.glb',
'T_object_part': matrix}}, parameters=..., note=...) including the fixed body,
or world.update_parts(id, {part_id: matrix}, parameters=..., note=...). These are
agent-authored geometry/configuration estimates, not automatic joint or deformation tracking.
`branch = world.fork(name)` and branch.set_hypothesis({id: T}, expected_revision=...)
keep hypothetical changes separate. world.export_scene('current.glb') exports for inspection.
world.nearest_surface(id, points_W), world.points(id, local_points), and
world.aabb_relation(a, b) are optional geometry queries. AABB overlap is not mesh contact.
"""


def prepare(config_path, *, robo, robot, out_dir, scratch, cancelled=None):
    """Host setup hook; returns extra namespace bindings, prompt and run metadata."""
    if robot not in {"robodojo", "robodojo_xpl", "robodojo_xpl_rgb"}:
        raise ValueError("Privilege's live adapter currently supports PyRUA RoboDojo robots")
    path = Path(config_path).expanduser().resolve()
    config = read(path)

    def resolve(value):
        return (path.parent / Path(value).expanduser()).resolve()

    directory = Path(out_dir).resolve() / "privilege"
    directory.mkdir(exist_ok=False)
    frame = config.get("frame", "pyrualean_world")
    initial = capture(robo, directory, frame)
    started = time.monotonic()
    if config.get("world"):
        from .storage import import_world

        if config.get("recovery_observation"):
            check_recovery_observation(resolve(config["recovery_observation"]), directory, initial["capture"])
        world = import_world(
            resolve(config["world"]),
            directory / "world",
            frame=frame,
            T_host_world=config.get("T_host_world"),
        )
        mode = "imported_prior"
    else:
        asset = config.get("robot_asset")
        if asset:
            asset = copy.deepcopy(asset)
            asset["urdf"] = str(resolve(asset["urdf"]))
            asset["packages"] = {k: str(resolve(v)) for k, v in asset.get("packages", {}).items()}
            robot_glb = export_robot(asset, initial["robot_state"], directory / "known_robot.glb")
        elif config.get("robot_glb"):
            robot_glb = resolve(config["robot_glb"])
            # An externally posed snapshot must identify the public state it represents.
            binding = read(resolve(config["robot_state_file"]))
            if binding != initial["robot_state"]:
                raise ValueError(
                    "Supplied robot GLB state differs from the captured public robot state"
                )
        else:
            raise ValueError(
                "Configure robot_asset (public URDF bindings) or robot_glb + robot_state_file"
            )
        reconstruction = dict(config["reconstruction"])
        for key in ("runtime", "codex_home", "codex_config"):
            if reconstruction.get(key):
                reconstruction[key] = str(resolve(reconstruction[key]))
        world = reconstruct(
            directory,
            initial["capture"],
            robot_glb,
            directory / "reconstruction",
            reconstruction,
            cancelled=cancelled,
        )
        mode = "fresh_calibrated_rgb"
    alignment = None
    alignment_config = config.get("height_alignment", {})
    if alignment_config is not False:
        from .alignment import align_table_height, table_object
        from pyrualean.hosts.robodojo_kinematics import TABLE_Z

        alignment = align_table_height(
            world, table_object_id=table_object(world, alignment_config.get("table_object_id")),
            target_z_m=TABLE_Z, source="pyrualean.hosts.robodojo_kinematics.TABLE_Z",
        )
    from .fixtures import setup_fixed_obstacles
    _, initial_packet, _ = load(directory, initial['capture'])
    fixture_setup = setup_fixed_obstacles(world, config, initial_packet, robot)
    fixed_obstacles = fixture_setup['added']
    api = AgentWorld(world, robo, scratch)
    api._configure_perception(config.get('perception', 'codeworld'))
    state = world.state()
    state['control_policy'] = {'observation_consistency': True, 'version': 'observation-control-v1'}
    atomic(world.state_path, state)
    control = getattr(getattr(robo, '_backend', None), 'set_codeworld_control', None)
    if control is not None:
        control(True)
    collision_config = config.get('collision')
    if collision_config:
        collision_config = copy.deepcopy(collision_config)
        if 'urdf' in collision_config:
            collision_config['urdf'] = str(resolve(collision_config['urdf']))
        if 'robot_spec' in collision_config:
            collision_config['robot_spec']['urdf'] = str(resolve(collision_config['robot_spec']['urdf']))
            collision_config['robot_spec']['packages'] = {
                k: str(resolve(v)) for k, v in collision_config['robot_spec'].get('packages', {}).items()}
        api._enable_collision(collision_config)
    record = {
        "mode": mode,
        "frame": frame,
        "world": str(world.root),
        "setup_s": time.monotonic() - started,
        "initial_observation": initial["capture"],
        "metric_validation": False,
        "robot_control": "native_actions_with_codeworld_observation_control",
        "interaction_mode": "mandatory",
        "perception": api._perception_mode,
        "disabled_perception_tools": sorted(robo._disabled_perception_tools),
        "height_alignment": alignment,
        "fixed_obstacles_added": fixed_obstacles,
        "fixed_obstacle_setup": fixture_setup,
        "collision": {"enabled": bool(api._collision), "config": collision_config,
                      "coverage": "sampled_URDF_and_reconstructed_meshes" if api._collision else None},
    }
    from .runtime import implementation_receipt
    record['implementation'] = implementation_receipt()
    atomic(directory / "integration.json", record)
    # Source declarations / part names are useful once, not in every tool response.
    initial_context = json.dumps(
        {**api._loop.scene.packet(detailed=True), "interaction": api._loop.status()},
        ensure_ascii=False,
    )
    return (
        {"world": api},
        OBSERVATION_CONTROL_PROMPT + PROMPT + STRICT_PROMPT + (CODEWORLD_ONLY_PROMPT if api._perception_mode == 'codeworld' else '')
        + (COLLISION_PROMPT if api._collision else '') + "\nInitial CodeWorld:\n" + initial_context,
        record,
    )


COLLISION_PROMPT = """
Scene collision checking is enabled for native move_to, set_gripper and home paths.
It tests live URDF links, both arms and attached CodeWorld meshes against current
scene geometry. Collision feedback names the bodies and state revision. A zero-motion
refusal leaves no pending scene update; choose a different path or intermediate pose.
An action stopped after partial motion still needs the normal post-action review.
You choose the motion and the intended contact geometry from the task and RGB/CodeWorld.
There is no task-specific action template or required approach direction. For general
contact-rich motion, declare the contact pairs for your next primitive:
world.expect_contacts(contacts=[{
    'first': {'object': carried_id, 'part': chosen_contact_part},
    'second': {'object': target_id, 'part': chosen_target_part},
    'reason': 'your task-specific reason for this contact'}])
Each selector can name an object+part, an object+local bounds [min_xyz,max_xyz],
or an exact robot arm+link. Parts and robot links also accept optional local bounds.
For a mesh without named parts, use {'object': object_id, 'bounds': [local_min, local_max]}.
The key is bounds (local_bounds is an alias); do not use an object ID as a part name.
Bounds are in the selected part/link frame, or object frame when no part is selected.
Choose regions/parts from observed geometry; publish useful parts if needed. They may
face any direction. The declaration allows the selected surface intersections throughout
this ONE primitive, including native controller segments. Other parts, links, self
collisions and obstacles remain checked. Local regions are geometrically clipped, so
a coarse triangle crossing a region boundary is not by itself a rejection.
This expresses intended interaction, not force, depth, stability or success. Choose
contact extent and motion yourself. A refusal reports the object/part that remains in
collision: inspect it, revise your geometry estimate/contact scope or choose another path;
do not repeatedly submit the same blocked path or exempt an entire scene.
Legacy shortcuts remain optional: grasp={'right': object_id} permits finger/object
contact; place={'right': support_id} describes only shallow support-top contact at the
endpoint. Use general contacts for interactions those shortcuts cannot describe.
Intent expires after ONE primitive, including a refusal; redeclare for another primitive
when intended. Changing selected mesh geometry invalidates its old contact intent with
an explicit report; pose/part-frame changes carry local regions with them automatically.
Existing confirmed grasp finger contacts are automatic.
After detaching, existing finger/object contact is retained for opening in place in
the immediately next primitive; it does not permit arm motion or new contact pairs.
world.check_joint_path([{ 'right': six_joint_angles }, ...]) is an optional offline query.
Fixed hardware is part of CodeWorld, including objects with role=fixed_obstacle.
When choosing an assembly site, consider every layer and the carried object's
approach/withdrawal space, not just the first support's reachable point.
world.check_layout({object_id: proposed_T_W_object, ...}) optionally tests proposed
assembly poses against fixed hardware. It neither moves objects nor changes state.
The returned clearances help compare sites: a nearly touching final assembly may
leave insufficient space for fingers, wrist rotation and carried-object motion.
Empty space between fixture components remains usable; proximity alone is not collision.
These are sampled checks of estimated geometry; contact/slip still needs RGB corrections.
"""

STRICT_PROMPT = """
Mandatory read/update, freely chosen action computation:
Read the compact current CodeWorld in every response and call world.read_context() before
EACH state-changing robo primitive (not each servo tick). Robot actions remain native PyRUA.
After the action inspect the three paired views: REAL RGB left, same-camera CodeWorld
geometry right. K/extrinsics are recorded at image time, including delayed wrist images.
The world render uses identification colors and omits the initial frozen robot: ignore
lighting/material differences and normal gripper/arm occlusion. Inspect the held object,
released object, contact partners, and other genuinely changed objects. Stable table/floor/
walls and passively occluded objects may retain their established state, without repeated
per-object justification or re-estimation.

Before the next action or finish, commit the step review:
ctx = world.read_context()
world.commit_step(ctx['interaction']['pending_update']['capture'],
    poses={...}, unchanged={...}, unresolved={...}, held={...}, disturbed={...},
    expected_revision=ctx['revision'], note='what changed / stable scene confirmation')
All mappings are optional: stable omitted objects retain their state. A pending pose_refresh
persists across reviews/actions until an actual pose correction or an explicit visual
confirmation of that SAME placement. Inspect that list before planning from those objects.
For visible slip/release/tilt, submit the landing pose in THIS review along with held=None;
detaching alone does not locate the object. Include displaced neighbours in poses too.
disturbed={object_id: 'visible change'} records affected neighbours when their pose cannot
yet be recovered; unresolved={object_id: 'occluded/ambiguous'} also persists. These neither
add action gates nor demand waits/captures. Do not mark an uncertain landing unchanged.
Do not fill every stable object with boilerplate.
At a confirmed grasp, set held={'left': object_id} (or right), with a pose correction if
needed. The host remembers paired geometry at the first close, so confirmation after a
lift uses the grasp-time offset instead of mixing an old object pose with the new arm pose.
The host then predicts that object's position AND orientation from measured arm
motion; finger occlusion alone is not a reason to leave it at its old position. On release,
slip or failed grasp use held={'left': None}; a full open command also clears the attachment.
Measured fully closed empty fingers also clear the attachment, retaining the last carried
estimate as unresolved until corrected. A held declaration cannot override that reading;
commit_step reports rejected_held and keeps any supplied visual pose correction.
Confirm the intended object visually, not just gripper closure. Correct the predicted pose
from main/wrist images when it disagrees. No mandatory geometric calculation is required.
Pose proposals are current-world T_W_object by default. For a held object's pose measured
in a delayed image, set time_basis='image' in that pose proposal: the host propagates it
using image-time and current end-effector poses. Do not treat a pre-release image as a
post-release position measurement; retain the prediction until a useful view arrives.
All pose correction APIs also correct the grasp offset, so subsequent motion preserves
the visual correction. Delayed pre-release corrections retain the last prediction and
report deferred objects without blocking the next action or requiring another capture.

Use world.compare_views(capture, object_ids=[...]) to inspect corrections from the same
recorded cameras. Pose changes are common; shape changes need actual silhouette/size
evidence across useful views. For an opening drawer, change its part transform/state via
update_parts; use publish_parts when the original shape/part decomposition is wrong.

For a visible changed object, you can fit its pose directly inside poses instead of
guessing 3D coordinates. Supply the same physical mesh-local landmarks in recorded views:
poses={object_id: {'image_points': {'mode': 'rigid', 'points': [
    {'local_xyz': [x,y,z], 'pixels': {'cam_head': [column,row],
                                   'cam_right_wrist': [column,row]}}, ...]}}}
Use at least three non-collinear landmarks, each visible in at least two CURRENT views.
Coordinates refer to the original RGB, not the resized comparison panel. Get local mesh
coordinates by applying inverse(snapshot.transform(id)) to snapshot.mesh(id).vertices.
Match real corners/surface features, not the center of an occluded silhouette or an AABB
corner with no corresponding surface. The solver triangulates and fits full rigid pose;
it does not assume the object is flat or sitting on the table. mode='translation' uses
one or more matched landmarks and preserves orientation ONLY when images support that.
Fitting runs within the existing commit and writes its result before the next read/plan.
fit_failed leaves a persistent unresolved entry; residuals are saved in sensor_validation.
Identity/correspondence errors can still give wrong fits. Inspect the result against RGB.
Outside a pending action, reuse the current capture with the same proposal format:
world.integrate_rgb(capture, poses={object_id: proposal},
                    expected_revision=world.read_context()['revision'], note='RGB evidence')
The second positional argument can also supply proposals instead of poses=.
Do not use world.update(..., basis='rgb_agent_estimate') for this operation.
Do not add a robot motion simply to make another correction possible.
After committing, read_context again before an action. Return from the Python cell when
new images are needed to decide. Raw geometry/source details remain queryable on demand.
"""

CODEWORLD_ONLY_PROMPT = """
This run uses RGB + CodeWorld perception. robo.locate, robo.back_project and robo.segment
are disabled, including the implicit RGB measurement behind back_project. Do not use
world.integrate_location/reconcile_location or start an external detector. Use the returned
RGB, its camera calibration, public robot feedback and the persistent reconstructed world.
Baseline PyRUA runs retain their original perception APIs.
"""


OBSERVATION_CONTROL_PROMPT = """
CodeWorld observation-control-v1:
- In CodeWorld mode, grip.holding=None means aperture suggests contact but the payload is
  unconfirmed; holding=False means no holding evidence. Native aperture is a heuristic.
  Report current observed object poses in commit_step, including observed axes. New held
  claims remain provisional until two current pose observations show object/hand motion
  consistency. Provisional geometry is carried as a payload hypothesis for collision queries,
  not a statement that the object is physically held. Inspect grasp_review; a contradicted
  grasp is detached. Do not copy predicted poses as if they were independent RGB evidence.
- After release, inspect the object independently of the fingers and after withdrawal.
  An unchanged sentence does not resolve a release landing pose. Use observed poses or
  unresolved. Optional relations={id:{'kind':'supported' or 'inserted','target':id,
  'note':'visible contact evidence'}} records an observed claim, not certified support,
  insertion, stability or task success. Check the handle/hole/support relation in the
  calibrated views; a similar silhouette alone is insufficient.
- home refuses a failed clearance lift under CodeWorld control. Select intermediate
  retreat poses and check their paths instead of retrying the same home blindly.
- Before building a multi-stage structure, list the objects needed in later stages.
  world.preview_plan([{'name':'stage','poses':{id:T_W},'access':[future_id]}, ...])
  forecasts proximity to future access envelopes without moving anything. Consider
  rearranging crowded loose pieces before a layer obstructs access. This is a geometric
  warning, not a visibility/IK/stability proof; check candidate robot paths separately.
- Use recovery_history to identify repeated empty closes, blocked approaches and IK
  refusals. Before repeating, compare measured opening, object motion, grasp surface,
  contact height and reachability. Revise the pose/shape from RGB when mismatched;
  change the grasp surface/approach or preserve a reachable intermediate placement.
  Avoid pushing a thin object into a less accessible state without a recovery plan.
  Leave room and budget for release verification and withdrawal. You choose the plan.
"""

OBSERVATION_CONTROL_PROMPT += """
Geometry queries deliberately omit textures. To publish parts without losing appearance,
use world.publish_parts(id, {'part': {'source_faces': original_face_indices,
'T_object_part': T}}, expected_revision=...). Face indices address the original bound
object mesh in object coordinates, before any previous assembly replacement. Query
triangles preserve face order for an unsplit object. Include every required body part.
Do not reuse assembly/reordered face indices as original indices. Imported mesh parts
also recover source appearance when all their object-frame triangles match uniquely;
new/deformed surfaces must provide their own material. Inspect appearance_source.
Source unknowns remain in read_context: hidden thickness is not a measured clearance.
Before stacking layers or moving beside a completed structure, revisit uncertain support
height/thickness with the allowed calibrated RGB observations and compare predicted vs
observed top surfaces. Do not infer true clearance from CLEAR alone.
"""
