"""X5 command dispatch, separated from simulator startup and queue transport."""

from dataclasses import dataclass
from typing import Any, Callable
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .multiarm import arm_index, resolve_commands
from .pose import transform
from .request_loop import complete_receipt


@dataclass
class SessionState:
    active: int = 0
    stopped: bool = False
    collision: Any = None


@dataclass
class X5Dispatcher:
    session: SessionState
    root: Path
    names: list[str]
    solvers: Any
    rm: Any
    validate_motion: Callable
    execute_joints: Callable
    state: Callable
    settings: dict
    official: bool
    official_actions: Any
    official_take: Callable
    tick: Callable
    benchmark: bool
    env: Any
    capture: Callable

    def __call__(self, packet):
        from .ownership import authorize

        source = authorize(self.root, packet)
        op = packet["op"]
        args = packet.get("args", {})
        result = {"actor": source}
        if op in (
            "move_tcp",
            "move_tcp_group",
            "joints",
            "joints_group",
            "gripper",
            "gripper_group",
            "check_ik",
            "check_motion",
        ):
            commands = resolve_commands(
                (
                    args["commands"]
                    if op.endswith("_group") or op == "check_motion"
                    else [args]
                ),
                self.names,
                self.session.active,
            )
            result["commanded_arms"] = list(commands)
            if op.startswith("move_tcp") or op in ("check_ik", "check_motion"):
                targets = {}
                solutions = {}
                all_candidates = {}
                for i, c in commands.items():
                    if op == "check_motion" and "joint_positions" in c:
                        solutions[i] = np.asarray(c["joint_positions"], float)
                        continue
                    T = transform(c["pose_W"])
                    targets[i] = T
                    candidates = self.solvers[i].solve_candidates(
                        T, self.rm.get_joint(self.rm.robot_list[i])[0]
                    )
                    all_candidates[str(i)] = [q.tolist() for q in candidates]
                    solutions[i] = candidates[0] if candidates else None
                failed = [i for (i, q) in solutions.items() if q is None]
                if failed:
                    result.update(status="IK_NOT_FOUND", failed_arms=failed, steps=0)
                elif op == "check_ik":
                    result.update(
                        status="IK_FOUND",
                        joint_targets={
                            str(i): q.tolist() for (i, q) in solutions.items()
                        },
                        joint_candidates=all_candidates,
                        collision_checked=False,
                    )
                elif op == "check_motion":
                    if not self.session.collision:
                        raise ValueError(
                            "Full-arm collision checking is not configured"
                        )
                    (_, report) = self.validate_motion(solutions, args)
                    result.update(
                        status=report["status"],
                        collision=report,
                        joint_targets={
                            str(i): q.tolist() for (i, q) in solutions.items()
                        },
                        steps=0,
                    )
                else:
                    (n, report) = self.execute_joints(
                        solutions, args.get("duration_steps"), args
                    )
                    rows = self.state()["arms"]
                    errors = {}
                    for i, T in targets.items():
                        actual = transform(rows[i]["eef_pose_W"])
                        errors[str(i)] = {
                            "position_error_m": float(
                                np.linalg.norm(actual[:3, 3] - T[:3, 3])
                            ),
                            "rotation_error_rad": float(
                                Rotation.from_matrix(
                                    actual[:3, :3].T @ T[:3, :3]
                                ).magnitude()
                            ),
                        }
                    arrived = all(
                        (
                            e["position_error_m"] < 0.004
                            and e["rotation_error_rad"] < 0.06
                            for e in errors.values()
                        )
                    )
                    result.update(
                        status=(
                            "COLLISION_BLOCKED"
                            if report and report["status"] == "COLLISION"
                            else "ARRIVED" if arrived else "TARGET_NOT_REACHED"
                        ),
                        steps=n,
                        arm_errors=errors,
                        collision=report,
                    )
                    if len(errors) == 1:
                        result.update(next(iter(errors.values())))
            elif op.startswith("joints"):
                (n, report) = self.execute_joints(
                    {
                        i: np.asarray(c["joint_positions"], dtype=float)
                        for (i, c) in commands.items()
                    },
                    args.get("duration_steps"),
                    args,
                )
                result.update(
                    status=(
                        "COLLISION_BLOCKED"
                        if report and report["status"] == "COLLISION"
                        else "EXECUTED"
                    ),
                    steps=n,
                    collision=report,
                )
            else:
                n = args.get("steps", 80)
                if isinstance(n, bool) or not isinstance(n, int) or n < 1:
                    raise ValueError("steps must be a positive integer")
                for c in commands.values():
                    if c["mode"] not in ("OPEN", "CLOSE", "HOLD"):
                        raise ValueError("Invalid gripper mode")
                if self.session.collision:
                    starts = {
                        i: self.rm.get_joint(r)[0].copy()
                        for (i, r) in enumerate(self.rm.robot_list)
                    }
                    fingers = {
                        i: self.rm.get_end_effector_real_val(r)[0].copy()
                        for (i, r) in enumerate(self.rm.robot_list)
                    }
                    finger_targets = {}
                    for i, c in commands.items():
                        if c["mode"] == "HOLD":
                            continue
                        r = self.rm.robot_list[i]
                        val = 1.0 if c["mode"] == "OPEN" else 0.0
                        if r.gripper_move["sign"] != 1:
                            val = 1 - val
                        val = (
                            val * (r.gripper_scale[1] - r.gripper_scale[0])
                            + r.gripper_scale[0]
                        )
                        finger_targets[i] = np.array(
                            [
                                val,
                                val * r.gripper_move["mimic"][1]
                                + r.gripper_move["mimic"][2],
                            ]
                        )
                    samples = max(
                        1,
                        int(
                            np.ceil(
                                max(
                                    [0.0]
                                    + [
                                        np.max(abs(v - fingers[i]))
                                        for (i, v) in finger_targets.items()
                                    ]
                                )
                                / 0.002
                            )
                        ),
                    )
                    report = self.session.collision.check_path(
                        starts,
                        [starts] * samples,
                        fingers,
                        allowed_contacts=list(self.settings.get("allowed_contacts", []))
                        + list(args.get("allowed_contacts", [])),
                        attachments=args.get("attachments", []),
                        finger_targets=finger_targets,
                    )
                    result["collision"] = report
                    if report["status"] == "COLLISION":
                        result.update(status="COLLISION_BLOCKED", steps=0)
                        return complete_receipt(
                            result,
                            self.state(),
                            official_status=(
                                self.official_actions.status()
                                if self.official
                                else None
                            ),
                            actor=source,
                            decision_ref=args.get("decision_ref"),
                        )
                if self.official:
                    import math

                    values = {
                        i: 1.0 if c["mode"] == "OPEN" else 0.0
                        for (i, c) in commands.items()
                        if c["mode"] != "HOLD"
                    }
                    for _ in range(math.ceil(n / self.official_actions.interval)):
                        if not self.official_take(grippers=values):
                            break
                else:
                    plans = {
                        i: list(
                            self.rm.plan_endeffector_joint(
                                0,
                                self.rm.robot_list[i].arm_name,
                                target_val=1.0 if c["mode"] == "OPEN" else 0.0,
                                control_step_num=n,
                            )
                        )
                        for (i, c) in commands.items()
                        if c["mode"] != "HOLD"
                    }
                    for k in range(max([n] + [len(plan) for plan in plans.values()])):
                        control = {}
                        for plan in plans.values():
                            if k < len(plan):
                                control.update(plan[k])
                        self.tick(control)
                result.update(status="EXECUTED")
        elif op == "select_arm":
            self.session.active = arm_index(args["arm"], self.names)
            result = {"status": "SELECTED", "active_arm": self.session.active}
        elif op == "observe":
            result = {"status": "OBSERVED"}
        elif op == "finish":
            self.session.stopped = True
            result = {"status": "CLOSED"}
        else:
            raise ValueError("Unknown direct I/O operation " + op)
        if self.official:
            result.update(self.official_actions.status())
            if self.official_actions.ended and op not in (
                "finish",
                "observe",
                "select_arm",
                "check_ik",
            ):
                result["status"] = (
                    "EPISODE_ENDED"
                    if self.benchmark
                    else "EPISODE_SUCCESS" if self.env.success[0] else "EPISODE_LIMIT"
                )
        if op not in ("check_ik", "check_motion"):
            result["capture"] = self.capture()
        return complete_receipt(
            result,
            self.state(),
            official_status=self.official_actions.status() if self.official else None,
            actor=source,
            decision_ref=args.get("decision_ref"),
        )
