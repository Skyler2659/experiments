"""Preserve native execution video; align reconstructed snapshots at recorded host steps."""

import argparse, json, subprocess, os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import imageio_ffmpeg

ROOT = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument("run")
p.add_argument("--limit", type=int)
p.add_argument("--runtime", required=True)
args = p.parse_args()
run = Path(args.run).resolve()
out = run / "videos"
out.mkdir(exist_ok=True)
steps = sorted(
    [
        p
        for p in (run / "episode/steps").glob("[0-9]*")
        if p.is_dir() and (p / "cam_head.png").exists()
    ]
)
steps = steps[: args.limit] if args.limit else steps
integration = json.loads((run / "episode/privilege/integration.json").read_text())
world = Path(integration["world"])
world = world if world.is_absolute() else (run / "episode/privilege" / world).resolve()
runtime = Path(args.runtime).expanduser().resolve()
manifest = {
    "export_version": 3,
    "fps": 2,
    "alignment": "One recorded host observation per frame. Last audited CodeWorld state at the same host step within its chronological observation interval.",
    "hidden_objects": [],
    "frames": [],
}
if not steps:
    raise RuntimeError("No recorded RGB frames")
ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()


def encode(folder, name, count):
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            "2",
            "-i",
            str(folder / "%05d.png"),
            "-frames:v",
            str(count),
            "-c:v",
            "libx264",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out / name),
        ],
        check=True,
    )


execution = out / "execution_frames"
execution.mkdir(exist_ok=True)
for i, step in enumerate(steps):
    im = Image.open(step / "cam_head.png").convert("RGB")
    panel = Image.new("RGB", (im.width, im.height + 36), "#f2f3e9")
    panel.paste(im, (0, 36))
    ImageDraw.Draw(panel).text(
        (12, 12),
        f"Execution | host step {int(step.name)} | observation sequence, 2 fps",
        fill="#263a2b",
    )
    panel.save(execution / f"{i:05d}.png")
encode(execution, "execution_observations.mp4", len(steps))
if not (world / "world.json").exists():
    (out / "manifest.json").write_text(
        json.dumps(dict(manifest, error="No reconstructed world available"), indent=2)
    )
    raise SystemExit(0)
# A cached robot_state can refer to an old requested observation (e.g. state(0)).
# Restrict each state to its actual chronological observation interval.
audit = []
for line in (run / "audit/calls.jsonl").read_text().splitlines():
    try:
        row = json.loads(line)
    except ValueError:
        continue
    h = row.get("world_state")
    if h and (run / "audit/states" / f"{h}.json").exists():
        audit.append(row)
by_step = {}
all_steps = sorted(
    [p for p in (run / "episode/steps").glob("[0-9]*") if (p / "observation.json").exists()]
)
for j, step in enumerate(all_steps):
    start_ns = json.loads((step / "observation.json").read_text())["recorded_at_ns"]
    end_ns = (
        json.loads((all_steps[j + 1] / "observation.json").read_text())["recorded_at_ns"]
        if j + 1 < len(all_steps)
        else float("inf")
    )
    candidates = [
        r
        for r in audit
        if start_ns <= r["time_ns"] < end_ns
        and (r.get("robot_state") or {}).get("step") == int(step.name)
    ]
    if candidates:
        by_step[int(step.name)] = max(candidates, key=lambda r: r["time_ns"])
objects = json.loads((world / "world.json").read_text())["objects"]
records = []
for oid, path in objects.items():
    obj = json.loads((world / path).read_text())
    hidden = (
        oid in {"mounting_plate", "mounting_bracket", "mounting_post"}
        or obj.get("role") == "robot_snapshot"
        or any(
            t in oid.lower()
            for t in (
                "camera",
                "known_robot",
                "sensor",
                "tripod",
                "cam_head",
                "cam_left",
                "cam_right",
            )
        )
    )
    if hidden:
        manifest["hidden_objects"].append(oid)
    else:
        records.append(dict(id=oid, mesh=obj["geometry"]["local_mesh"]))
for step in steps:
    s = int(step.name)
    if s not in by_step:
        continue
    row = by_step[s]
    obs = json.loads((step / "observation.json").read_text())
    cam = obs["cameras"]["cam_head"]
    size = Image.open(step / "cam_head.png").size
    manifest["frames"].append(
        dict(
            step=s,
            rgb=str(step / "cam_head.png"),
            state=json.loads((run / "audit/states" / (row["world_state"] + ".json")).read_text()),
            view=dict(
                image_size=size,
                K=cam["intrinsic_matrix"],
                T_W_C=(
                    np.asarray(cam["extrinsic_matrix"]) @ np.diag([1.0, -1.0, -1.0, 1.0])
                ).tolist(),
            ),
            state_hash=row["world_state"],
        )
    )
if not manifest["frames"]:
    raise RuntimeError("No paired historical world states; refusing to substitute final state")
packet = dict(world=str(world), objects=records, frames=manifest["frames"])
(out / "render_input.json").write_text(json.dumps(packet))
subprocess.run(
    [
        str(runtime / ".render-tools/node/bin/node"),
        str(ROOT / "render_history.mjs"),
        str(out / "render_input.json"),
    ],
    check=True,
    env=dict(os.environ, CODEWORLD_RENDER_RUNTIME=str(runtime)),
    timeout=1800,
)
frames = out / "comparison_frames"
frames.mkdir(exist_ok=True)
for i, row in enumerate(manifest["frames"]):
    im = Image.open(row["rgb"]).convert("RGB")
    recon = Image.open(out / "reconstruction_frames" / f"{i:05d}.png").convert("RGB")
    w, h = im.size
    panel = Image.new("RGB", (w * 2, h + 64), "#f2f3e9")
    panel.paste(im, (0, 40))
    panel.paste(recon, (w, 40))
    d = ImageDraw.Draw(panel)
    d.text((12, 8), f"Execution | host step {row['step']}", fill="#253b2b")
    d.text(
        (w + 12, 8), f"CodeWorld r{row['state']['revision']} | original materials", fill="#253b2b"
    )
    d.text(
        (12, h + 47),
        "Action-aligned observations, 2 fps | Robot and camera models hidden on reconstruction side",
        fill="#576b53",
    )
    panel.save(frames / f"{i:05d}.png")
encode(frames, "comparison.mp4", len(manifest["frames"]))
manifest["appearance"] = (
    "Original GLB materials, texture maps and vertex colors; no identity palette or simulator textures."
)
manifest["unpaired_steps"] = [int(s.name) for s in steps if int(s.name) not in by_step]
for row in manifest["frames"]:
    row.pop("state", None)
(out / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps({"frames": len(manifest["frames"]), "hidden": manifest["hidden_objects"]}))
