"""Loopback-only live monitor; reads files without invoking a robot."""

import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import time
from urllib.parse import unquote, urlsplit

from .config import RESOURCE, read


def snapshot(root):
    data = read(root / "status.json")
    data["telemetry_age_s"] = time.time() - data.get("time", 0)
    for row in data.get("runs", []):
        folder = root / row["id"]
        row.pop("result", None)
        row["images"] = {}
        steps = sorted((folder / "episode/steps").glob("[0-9]*/cam_head.png"))
        if steps:
            row["step"] = int(steps[-1].parent.name)
            row["image_time"] = steps[-1].stat().st_mtime
            row["images"] = {
                p.stem: str(p.relative_to(root)) for p in steps[-1].parent.glob("cam_*.png")
            }
        row["recovery"] = [
            read(p)
            for p in (folder / "recovery").glob("*.json")
            if read(p).get("status") in ("WAITING_PROVIDER", "WAITING_TOOL", "NEEDS_ATTENTION")
        ]
        compare = read(folder / "live_comparison/latest.json")
        row["comparisons"] = {
            k: str(Path(v).relative_to(root))
            for k, v in compare.get("images", {}).items()
            if Path(v).is_relative_to(root)
        }
        row["comparison_step"] = compare.get("step")
        row["comparison_time"] = compare.get("time")
        row["videos"] = [str(p.relative_to(root)) for p in (folder / "videos").glob("*.mp4")]
        row["videos"] += [
            str(p.relative_to(root))
            for p in (folder / "episode/bench/eval_result").rglob("*cam_head*.mp4")
        ]
        result = read(folder / "episode/result.json")
        row["stateful_calls"] = result.get("stateful_calls")
        row["env_steps"] = result.get("env_steps")
    return data


def public_file(root, relative):
    parts = Path(relative).parts
    if not parts or any(x in parts for x in ("..", "agent-home", ".codex-home", "auth.json")):
        raise ValueError("Private or invalid path")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path.suffix.lower() not in (".mp4", ".png", ".jpg"):
        raise ValueError("Only generated media is served")
    return path


def serve(root, port):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            target = urlsplit(self.path).path
            if target == "/":
                body = (RESOURCE / "dashboard.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif target == "/api/status":
                body = json.dumps(snapshot(root), ensure_ascii=False).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif target.startswith("/media/"):
                try:
                    path = public_file(root, unquote(target[len("/media/") :]))
                    size = path.stat().st_size
                    start, end = 0, size - 1
                    partial = self.headers.get("Range")
                    if partial:
                        unit, positions = partial.split("=")
                        lo, hi = positions.split("-")
                        if unit != "bytes" or not lo:
                            raise ValueError("Unsupported range")
                        start, end = int(lo), min(int(hi) if hi else end, end)
                        if not 0 <= start <= end < size:
                            raise ValueError("Invalid range")
                    self.send_response(206 if partial else 200)
                    self.send_header(
                        "Content-Type",
                        mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                    )
                    self.send_header("Accept-Ranges", "bytes")
                    if partial:
                        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                    self.send_header("Content-Length", str(end - start + 1))
                    self.end_headers()
                    with path.open("rb") as f:
                        f.seek(start)
                        remaining = end - start + 1
                        while remaining:
                            block = f.read(min(1024 * 1024, remaining))
                            if not block:
                                break
                            self.wfile.write(block)
                            remaining -= len(block)
                except (ValueError, OSError):
                    self.send_error(404)
            else:
                self.send_error(404)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"WorldRUA dashboard: http://127.0.0.1:{server.server_port}/", flush=True)
    server.serve_forever()
