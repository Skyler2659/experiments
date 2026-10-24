"""Load the same released public recipe used by RoboProbe, without policy imports."""
from pathlib import Path


def load_task_brief(roboprobe_root, task_name):
    if not task_name or Path(task_name).name != task_name or task_name in (".", ".."):
        raise ValueError("Invalid public task name")
    relative = Path("policy/RoboDojo_Agent_L3_Inspect/recipes") / (task_name.removesuffix("_random") + ".md")
    path = Path(roboprobe_root) / relative
    return {"source": "RoboProbe released public task recipe", "path": str(relative),
            "task_name": task_name, "available": path.is_file(),
            "text": path.read_text(encoding="utf-8") if path.is_file() else "",
            "instance_instruction_authoritative": True}


def apply_task_brief(task, instruction, brief):
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Native public task instruction is missing")
    task["native_instruction"] = instruction
    task["public_task_brief"] = brief
    task["goal"] = instruction + ("\n\nTASK RECIPE:\n" + brief["text"].rstrip() if brief["text"] else "")
    return task
