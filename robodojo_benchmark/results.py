"""Authoritative native outcomes, independent of policy and media lifecycles."""

from pathlib import Path
from .catalog import read


def native_result(path, layout_id):
    result = read(path)
    details = list(result.get("details", {}).values())
    if len(details) != 1 or details[0].get("layout_id") != layout_id:
        raise ValueError("Native result does not contain exactly the requested layout")
    d = details[0]
    if (
        type(d.get("success")) is not bool
        or type(d.get("score")) not in (int, float)
        or not 0 <= d["score"] <= 1
    ):
        raise ValueError("Malformed native success/score")
    return {
        "official_success": d["success"],
        "score_0_100": float(d["score"]) * 100,
        "native_result": str(Path(path).resolve()),
        "layout_id": layout_id,
    }


def saved_native_result(attempt, job):
    """Return a finalized native outcome, or None when no native result exists."""
    evaluator = Path(attempt) / "evaluator"
    path = evaluator / "_result.json"
    if not path.is_file():
        return None
    result = {"status": "SCORED", **native_result(path, job["layout_id"])}
    config = evaluator / "server_config.json"
    if config.is_file():
        result["probe_only"] = bool(read(config).get("probe_native_only", False))
    actions = evaluator / "official_result.json"
    if actions.is_file():
        try:
            detail = read(actions)
            result.update(
                {
                    key: detail[key]
                    for key in ("official_actions", "official_action_limit")
                    if key in detail
                }
            )
        except (OSError, ValueError, TypeError) as exc:
            result["action_receipt_error"] = str(exc)
    return result
