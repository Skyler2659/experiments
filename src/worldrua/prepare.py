"""Optional offline evaluation tracing, never added to agent observations."""

import ast
import shutil
from pathlib import Path

from .config import RESOURCE


def prepare_truth_audit(folder, cfg):
    from pyrualean.hosts.xpolicylab_robodojo import lay_out

    bench = lay_out(
        folder / "episode/bench",
        Path(cfg["robodojo_root"]),
        Path(cfg["xpolicylab_root"]),
        depth=False,
        suite="arx_x5",
    )
    shutil.copy2(RESOURCE / "truth_recorder.py", bench / "utils/pyrualean_truth_recorder.py")
    replacements = {
        "src/eval_client/eval_env.py": [
            (
                "            data = self.obs_manager.get_obs(env_idx_list=env_idx_list)\n",
                "            from utils.pyrualean_truth_recorder import record as audit_record\n"
                "            audit_record(self, data, phase='observation')\n",
            )
        ],
        "XPolicyLab/policy/PyRUA_Lean/deploy.py": [
            (
                "                TASK_ENV.take_action(action)\n",
                "                from utils.pyrualean_truth_recorder import record as audit_record\n"
                "                audit_record(TASK_ENV, phase='after_native_action')\n",
            ),
            (
                "        _judge_as_it_stands(TASK_ENV)\n",
                "        from utils.pyrualean_truth_recorder import record as audit_record\n"
                "        audit_record(TASK_ENV, phase='final')\n",
            ),
        ],
    }
    for rel, pairs in replacements.items():
        path = bench / rel
        text = path.read_text()
        for needle, insertion in pairs:
            if needle + insertion in text:
                continue
            if text.count(needle) != 1:
                raise ValueError("Unsupported simulator revision for audit hook: " + rel)
            text = text.replace(needle, needle + insertion)
        ast.parse(text)
        path.write_text(text)
