"""Public task goals shared by baseline PyRUA and PyRUA + RePrivilege."""

from functools import lru_cache
import json
from pathlib import Path


@lru_cache(maxsize=1)
def _tasks():
    return json.loads(Path(__file__).with_name("robodojo_public_tasks.json").read_text())


def public_description(robot, task):
    if robot not in {"robodojo", "robodojo_xpl", "robodojo_xpl_rgb"}:
        return None
    key = str(task).replace("-", "_").lower()
    if key.endswith("_random"):
        key = key[:-7]
    return _tasks().get(key)


def render_description(robot, task):
    item = public_description(robot, task)
    if item is None:
        return ""
    lines = ["", "Public task description and scoring:", item["description"],
             f"Source: {item['source']} (snapshot {item['retrieved']})"]
    lines += [f"- {row['score']:g} points: {row['condition']}" for row in item["scoring"]]
    lines.append("These are public completion conditions, not a live score. The instance instruction "
                 "remains authoritative for instance-specific choices. Check every full-completion "
                 "condition before finish; partial structures are not task success.")
    return "\n".join(lines)
