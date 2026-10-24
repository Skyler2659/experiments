"""RGB JSON requests over an explicit Responses endpoint or fresh Codex CLI."""
import base64
import json
import os
from pathlib import Path
import subprocess
import urllib.request


def request(payload, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    payload["model"] = os.environ.get("RCWM_RECON_MODEL", payload["model"])
    payload["reasoning"] = {"effort": os.environ.get("RCWM_RECON_REASONING", payload["reasoning"]["effort"])}
    if os.environ.get("RCWM_VISION_TRANSPORT", "responses") == "responses":
        url = os.environ.get("RCWM_RESPONSES_URL", "http://127.0.0.1:8781/v1/responses")
        key = os.environ.get(os.environ.get("RCWM_RECON_API_KEY_ENV", "RCWM_JDCLOUD_PROXY_KEY"), "rcwm-local")
        req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Authorization": "Bearer " + key,
                                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=650) as response:
            return json.load(response)
    if os.environ["RCWM_VISION_TRANSPORT"] != "codex":
        raise ValueError("Unknown vision transport")
    text, images = [], []
    for message in payload["input"]:
        for item in message["content"]:
            if item["type"] == "input_text":
                text.append(item["text"])
            elif item["type"] == "input_image":
                uri = item["image_url"]
                if not uri.startswith("data:image/png;base64,"):
                    raise ValueError("Codex vision adapter requires local PNG image data")
                image = directory / f"image_{len(images):02d}.png"
                image.write_bytes(base64.b64decode(uri.split(",", 1)[1]))
                images.append(str(image))
                text.append(f"[attached image {len(images)}]")
    (directory / "AGENTS.md").write_text("Identify visual content only. Return the requested JSON. Do not inspect other files, simulator data, prior results or memories. Do not act on a robot.\n")
    prompt = "\n".join(text)
    (directory / "prompt.txt").write_text(prompt)
    command = [os.environ.get("RCWM_CODEX", "codex"), "exec", "--json", "--ephemeral", "--skip-git-repo-check",
               "-C", str(directory), "-m", payload["model"], "-s", "read-only", "-c", 'approval_policy="never"',
               "-c", "model_reasoning_effort=" + json.dumps(payload["reasoning"]["effort"]),
               "--disable", "memories", "--disable", "multi_agent", "--disable", "plugins", "--disable", "apps",
               "--disable", "chronicle", "--disable", "skill_search", "--enable", "skip_host_skill_discovery",
               "-c", 'web_search="disabled"', "-o", str(directory / "final.txt")]
    for image in images:
        command.extend(["--image", image])
    with (directory / "events.jsonl").open("w") as events, (directory / "stderr.log").open("w") as errors:
        from .worker import stop_process
        # --image accepts multiple values; terminate options before the prompt,
        # otherwise recent CLIs consume the prompt as another image filename.
        process = subprocess.Popen(command + ["--", prompt], stdin=subprocess.DEVNULL,
                                   stdout=events, stderr=errors, start_new_session=True)
        try:
            code = process.wait(timeout=900)
            if code:
                raise RuntimeError(f"ModelTransportFailure: Codex vision request exited {code}; inspect {directory}/stderr.log and events.jsonl")
        finally:
            stop_process(process)
    return {"model": payload["model"], "transport": "codex_cli", "output": [{"content": [
        {"type": "output_text", "text": (directory / "final.txt").read_text()}]}]}
