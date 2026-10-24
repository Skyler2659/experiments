"""File-queue transport; policy dispatch and simulation remain separate."""

import time
import traceback
from .io import atomic, read


def serve_requests(app, root, dispatch, timeline, *, should_stop, steps):
    io = root / "robot_io"
    while app.is_running() and not should_stop():
        for path in sorted(
            (io / "requests").glob("*.json"), key=lambda p: p.stat().st_mtime_ns
        ):
            destination = io / "responses" / path.name
            if destination.exists():
                continue
            try:
                request = read(path)
                timeline.operation = {"request_id": path.stem, **request}
                result = dispatch(request)
            except Exception as exc:
                result = {
                    "status": "ERROR",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            result["request_id"] = path.stem
            atomic(destination, result)
            print(path.stem, result["status"], steps(), flush=True)
            if should_stop():
                break
        time.sleep(0.05)


def complete_receipt(
    result, robot_state, *, official_status=None, actor=None, decision_ref=None
):
    """Include clock/budget provenance even when execution is rejected before motion."""
    if official_status is not None:
        result.update(official_status)
    result.update(robot_state=robot_state, actor=actor)
    if decision_ref is not None:
        result["decision_ref"] = decision_ref
    return result
