"""Require a bounded recovery review before CodeWorld abandons a live task.

Strategy diversity and irrecoverability remain the agent's judgments. The host
checks that referenced observations/actions actually ran; it does not infer task
success or supply a task-specific recovery plan.
"""
from __future__ import annotations

RECOVERY_INSTRUCTIONS = """CodeWorld stopping protocol:
A missed grasp or blocked path is a local failure, not proof that the task cannot
be completed. If budget remains, inspect current RGB/state, revise the failure
hypothesis, and try a meaningfully different recovery strategy. Preserve partial
progress. Do not repeatedly issue an unchanged failed command or substitute arm
homing for task recovery. Finish with success only when observations support it.
An early finish(failure/stuck) opens a recovery review instead of ending the run.
After that checkpoint, try two distinct recovery strategies and cite their
returned tool_turn_index values in recovery_trials, each with strategy and
observation. If current observations demonstrate that recovery is impossible,
use unrecoverable={reason, observation, turn_index, alternatives_ruled_out};
cite a fresh tool turn and explain at least two ruled-out alternatives. This is
your observation-based assessment, not a simulator success signal. A terminal
environment or an exhausted decision budget may finish immediately.
A rejected finish is not permission to end with a text answer: continue the same
episode. Never reset the scene or replay the previous attempt's actions.
"""

RECOVERY_PROPERTIES = {
    "recovery_trials": {
        "type": "array", "description": "Two distinct recovery trials after the requested review.",
        "items": {"type": "object", "properties": {
            "strategy": {"type": "string"}, "turn_index": {"type": "integer"},
            "observation": {"type": "string"}},
            "required": ["strategy", "turn_index", "observation"]}},
    "unrecoverable": {
        "type": "object", "description": "Observed irreversible obstacle; not merely a failed grasp.",
        "properties": {"reason": {"type": "string"}, "observation": {"type": "string"},
            "turn_index": {"type": "integer"},
            "alternatives_ruled_out": {"type": "array", "items": {"type": "string"}, "minItems": 2}},
        "required": ["reason", "observation", "turn_index", "alternatives_ruled_out"]},
}


def text(value):
    return isinstance(value, str) and bool(value.strip())


class FinishRecovery:
    def __init__(self):
        self.checkpoint = None
        self.last_rejection = None
        self.audit = []

    def check(self, args, turns, *, terminal, remaining_calls):
        status = args.get("status")
        if status not in {"success", "failure", "stuck"}:
            return {"error": "Invalid finish status", "accepted": False}
        if status == "success" or terminal or remaining_calls == 0:
            self.last_rejection = None
            return None
        if self.checkpoint is None:
            self.checkpoint = len(turns)
            return self.reject("Review the current scene and test alternative recovery strategies.", remaining_calls)

        def observed(index, motion=False):
            if type(index) is not int or not self.checkpoint < index < len(turns):
                return False
            row = turns[index]
            return (row.get("tool") in {"python", "run_program"} and row.get("ok")
                    and (not motion or (row.get("env_steps") or 0) > 0))

        evidence = args.get("unrecoverable")
        if isinstance(evidence, dict):
            alternatives = evidence.get("alternatives_ruled_out", [])
            if (text(evidence.get("reason")) and text(evidence.get("observation"))
                    and observed(evidence.get("turn_index")) and isinstance(alternatives, list)
                    and len({x.strip().casefold() for x in alternatives if text(x)}) >= 2):
                self.audit.append({"accepted": True, "basis": "agent_observed_irrecoverability", "evidence": evidence})
                self.last_rejection = None
                return None
        trials = args.get("recovery_trials", [])
        if isinstance(trials, list):
            valid = [x for x in trials if isinstance(x, dict) and text(x.get("strategy"))
                     and text(x.get("observation")) and observed(x.get("turn_index"), motion=True)]
            if (len({x['turn_index'] for x in valid}) >= 2
                    and len({x['strategy'].strip().casefold() for x in valid}) >= 2):
                self.audit.append({"accepted": True, "basis": "agent_reviewed_recovery_trials", "evidence": valid})
                self.last_rejection = None
                return None
        return self.reject("Recovery evidence is missing or does not reference two executed trials after the checkpoint. "
                           "Read-only calculations, repeated finish calls, and pre-checkpoint attempts are insufficient. "
                           "If motion is impossible, provide fresh observed irrecoverability evidence instead.", remaining_calls)

    def reject(self, reason, remaining):
        response = {"error": "early_finish_requires_recovery", "accepted": False,
                    "reason": reason, "checkpoint_turn_index": self.checkpoint,
                    "remaining_decision_calls": remaining, "recovery_protocol": RECOVERY_INSTRUCTIONS}
        self.last_rejection = response
        self.audit.append(response)
        return response


def resume_argv(argv, session_id):
    """Preserve model/MCP/config; resume the same Codex history without stale images."""
    result = [argv[0], "exec", "resume"]
    index = 2
    while index < len(argv) - 1:
        arg = argv[index]
        if arg in {"-C", "--cd", "-s", "--sandbox", "-i", "--image"}:
            index += 2
        else:
            result.append(arg)
            index += 1
    return [*result, session_id, "-"]
