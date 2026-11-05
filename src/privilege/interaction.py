"""Mandatory read/action/observation/update ordering, independent of robot control."""

import copy
import fcntl
import time

from .context import SceneContext
from .observations import load
from .world import atomic


class WorldInteraction:
    def __init__(self, world, observe, robot_state):
        self.world = world
        self.observe = observe
        self.robot_state = robot_state
        self.scene = SceneContext(world)
        self.read_at = None
        self.pending = None
        self.action_count = 0
        self.updated_count = 0
        self._action_read = None
        self._action_kwargs = {}
        self._comparison_cache = None
        self._delivered_images = None

    def status(self):
        return {
            "mode": "mandatory",
            "action_count": self.action_count,
            "updated_count": self.updated_count,
            "read_at": copy.deepcopy(self.read_at),
            "pending_update": copy.deepcopy(self.pending),
        }

    def _save(self):
        atomic(self.world.root / "state/interaction.json", self.status())

    def context(self, packet=None):
        status = self.status()
        if status["pending_update"]:
            p = status["pending_update"]
            status["pending_update"] = {k: p.get(k) for k in
                ("action", "phase", "capture", "capture_error", "focus_objects", "state_error") if p.get(k) is not None}
        return {**(self.scene.packet() if packet is None else packet), "interaction": status}

    def read_context(self):
        packet = self.scene.packet()
        self.read_at = {"revision": packet["revision"], "host_step": self.robot_state()["step"]}
        self._save()
        return self.context(packet)

    def before_action(self, tool, kwargs):
        if self.pending is not None:
            raise RuntimeError(
                "CodeWorld update required: call world.commit_step with the "
                "post-action RGB before another action"
            )
        live = self.robot_state()
        now = {"revision": self.world.state()["revision"], "host_step": live["step"]}
        if self.read_at != now:
            raise RuntimeError(
                "Read the current CodeWorld with world.read_context() before this action"
            )
        # Save paired inputs before motion, without changing geometry or requiring
        # another read. Confirmation may arrive only after a diagnostic lift.
        from .tracking import begin_action
        with (self.world.root / "state/update.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = self.world.state()
            if self.read_at != {"revision": state["revision"], "host_step": live["step"]}:
                raise RuntimeError("World changed before dispatch; read_context and use current targets")
            self._tracking_before = copy.deepcopy(state.get("tracking"))
            begin_action(state, live, tool, kwargs)
            atomic(self.world.state_path, state)
        self._action_read = copy.deepcopy(self.read_at)
        self._action_kwargs = copy.deepcopy(kwargs)
        self.read_at = None
        self.action_count += 1
        self.pending = {
            "action": self.action_count,
            "phase": "executing",
            "tool": tool,
            "kwargs": copy.deepcopy(kwargs),
            "codeworld_read": self._action_read,
            "capture": None,
            "capture_error": None,
        }
        self._save()

    def _record_recovery(self, record):
        if not self.world.state().get('control_policy', {}).get('observation_consistency'):
            return
        summary = record.get('summary') or {}
        kw = self._action_kwargs
        reason = ('empty_close' if kw.get('gripper') == 1 and summary.get('holding') is False else
                  'blocked_or_unreachable' if summary.get('blocked') or summary.get('reached') is False else
                  'tool_rejected' if not record.get('ok', True) else None)
        if reason is None:
            return
        with (self.world.root / 'state/update.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = self.world.state()
            history = state.setdefault('recovery_history', [])
            history.append({'host_step': self.robot_state()['step'], 'reason': reason,
                            'action': copy.deepcopy(kw), 'feedback': copy.deepcopy(summary),
                            'error': record.get('error'),
                            'interpretation': 'Review current object pose/contact and change the failed approach; outcome is not proof of object motion'})
            state['recovery_history'] = history[-8:]
            atomic(self.world.state_path, state)

    def action_rejected(self):
        """A zero-motion collision refusal does not create an observation/update debt."""
        with (self.world.root / "state/update.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = self.world.state()
            if self._tracking_before is None:
                state.pop("tracking", None)
            else:
                state["tracking"] = self._tracking_before
            atomic(self.world.state_path, state)
        self.action_count -= 1
        self.read_at = copy.deepcopy(self._action_read)
        self.pending = None
        self._save()

    def after_action(self, record):
        self._record_recovery(record.to_dict())
        self.pending = {
            "action": self.action_count,
            "phase": "awaiting_update",
            "record": record.to_dict(),
            "codeworld_read": self._action_read,
            "after_action_ns": time.time_ns(),
            "capture": None,
            "capture_error": None,
        }
        self.read_at = None
        try:
            from .tracking import predict

            kw = self._action_kwargs
            live = self.robot_state()
            opening = live.get(kw.get("arm"), {}).get("gripper")
            opened = opening >= .9 if opening is not None else record.to_dict().get("ok", False)
            released = ([kw["arm"]] if opened and
                        kw.get("arm") in {"left", "right"} and kw.get("gripper") == -1 and
                        (kw.get("opening") is None or kw["opening"] >= .9) else [])
            feedback = self.pending["record"]
            executed = (feedback["env_steps"] > 0 if feedback.get("env_steps") is not None
                        else feedback.get("ok", False))
            reset_candidates = ([kw["arm"]] if executed and
                                kw.get("arm") in {"left", "right"} and
                                kw.get("gripper") == -1 else [])
            with (self.world.root / "state/update.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                state = self.world.state()
                if feedback.get("env_steps") == 0:
                    # A failed IK/close that never executed must not seed a new
                    # grasp candidate. Retain candidates from earlier real motion.
                    state["tracking"] = copy.deepcopy(self._tracking_before or {})
                    released = reset_candidates = []
                ids = predict(state, live, released=released, reset_candidates=reset_candidates)
                state["revision"] += 1
                state["history"].append({"revision": state["revision"], "kind": "agent_feedback",
                    "affected_objects": ids, "result": {"primitive": record.to_dict(),
                    "codeworld_read": self._action_read}, "recorded_at_ns": time.time_ns()})
                atomic(self.world.state_path, state)
            self.pending["focus_objects"] = sorted(set(ids) | {
                oid for oid, row in state['objects'].items() if row.get('pose_refresh')})
        except Exception as exc:
            self.pending["state_error"] = str(exc)
        self.refresh_observation()

    def refresh_observation(self):
        try:
            cap = self.observe()
            if self.pending is not None:
                self.pending.update(capture=cap["capture"], capture_error=None)
            return cap
        except Exception as exc:
            if self.pending is not None:
                self.pending["capture_error"] = f"{type(exc).__name__}: {exc}"
            return None
        finally:
            self._save()

    def check_capture(self, capture):
        if self.pending is None:
            raise ValueError("No action awaits a CodeWorld step update")
        if self.pending["phase"] != "awaiting_update":
            raise ValueError("Action is still executing; wait for its post-action observation")
        _, packet, _ = load(self.world.root, capture)
        if packet["recorded_at_ns"] < self.pending["after_action_ns"]:
            raise ValueError("Use an observation captured after this action")
        if packet.get("metadata", {}).get("host_step") != self.robot_state()["step"]:
            raise ValueError("Capture predates host motion; observe again")
        return packet

    def committed(self):
        self.updated_count += 1
        self.pending = None
        self.read_at = None
        self._save()

    def before_finish(self):
        if self.pending is not None:
            raise RuntimeError("Update CodeWorld with world.commit_step before finish")

    def feedback(self):
        context = self.context()
        png = None
        if self.pending and self.pending.get("capture"):
            _, packet, image = load(self.world.root, self.pending["capture"])
            context["post_action_observation"] = {
                "observation_id": packet["observation_id"],
                "capture": self.pending["capture"],
                "views": [{k: v[k] for k in ("view_id", "K", "T_W_C", "image_size", "timing") if k in v}
                          for v in packet["views"]],
            }
            from .comparison import compare

            key = (self.pending["capture"], context["revision"])
            try:
                if getattr(self, '_comparison_timed_out', None) == key:
                    raise TimeoutError('Comparison rendering already timed out for this observation; use current RGB')
                if self._comparison_cache is None or self._comparison_cache[0] != key:
                    self._comparison_cache = (key, compare(self.world, key[0], wall_budget_s=30))
                report, panels = self._comparison_cache[1]
                context["comparison"] = {"revision": report["revision"], "files": report["files"],
                                         "format": "real RGB left, same-camera world geometry right"}
                if self._delivered_images != key:
                    png = panels
                    self._delivered_images = key
            except Exception as exc:
                if isinstance(exc, TimeoutError):
                    self._comparison_timed_out = key
                context["comparison_error"] = f"{type(exc).__name__}: {exc}"
                if self._delivered_images != key:
                    from pathlib import Path
                    folder = Path(self.world.root / self.pending["capture"]).parent
                    images = [(folder / v["rgb_uri"]).read_bytes() for v in packet["views"]]
                    names = [v["view_id"] for v in packet["views"]]
                    png = {"images": images, "originals": images,
                           "camera_names": [n + ".png" for n in names], "labels": names}
                    self._delivered_images = key
        return context, png
