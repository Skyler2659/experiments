"""Render same-step observation/CodeWorld comparisons without calling the robot."""

import concurrent.futures, json, os, shutil, subprocess, time, traceback, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(sys.argv[1]).resolve()
RESOURCE = Path(__file__).resolve().parent
RUNTIME = json.loads((ROOT / "manifest.json").read_text())["config"]["runtime"]


def read(p):
    return json.loads(p.read_text())


def atomic(p, d):
    t = p.with_suffix(".tmp")
    t.write_text(json.dumps(d, ensure_ascii=False))
    t.replace(p)


def render(ident):
    folder = ROOT / ident
    world = Path(read(folder / "episode/privilege/integration.json")["world"])
    spec = read(world / "world.json")
    out = folder / "live_comparison"
    out.mkdir(exist_ok=True)
    row = None
    state = None
    observations = sorted((folder / "episode/steps").glob("[0-9]*/observation.json"))
    intervals = {}
    for j, p in enumerate(observations):
        intervals[int(p.parent.name)] = (
            read(p)["recorded_at_ns"],
            read(observations[j + 1])["recorded_at_ns"]
            if j + 1 < len(observations)
            else float("inf"),
        )
    for line in reversed((folder / "audit/calls.jsonl").read_text().splitlines()):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        s = (r.get("robot_state") or {}).get("step")
        key = r.get("world_state")
        if s not in intervals or not key or not intervals[s][0] <= r["time_ns"] < intervals[s][1]:
            continue
        p = folder / "audit/states" / (key + ".json")
        if not p.exists():
            continue
        st = read(p)
        if st["geometry_revision"] != spec["geometry_revision"]:
            continue
        row = r
        state = st
        break
    if row is None:
        return
    step = row["robot_state"]["step"]
    stepdir = folder / "episode/steps" / f"{step:04d}"
    receipt = out / "latest.json"
    if receipt.exists():
        old = read(receipt)
        if (
            old.get("render_version") == 4
            and old["state_hash"] == row["world_state"]
            and old["step"] == step
        ):
            return
    work = out / "work"
    work.mkdir(exist_ok=True)
    geometry = work / "world"
    geometry.mkdir(exist_ok=True)
    objects = []
    hidden = []
    for oid, path in spec["objects"].items():
        obj = read(world / path)
        if (
            oid in {"mounting_plate", "mounting_bracket", "mounting_post"}
            or obj.get("role") == "robot_snapshot"
            or any(
                t in oid.lower()
                for t in (
                    "robot",
                    "camera",
                    "sensor",
                    "tripod",
                    "cam_head",
                    "cam_left",
                    "cam_right",
                )
            )
        ):
            hidden.append(oid)
            continue
        mesh = obj["geometry"]["local_mesh"]
        objects.append({"id": oid, "mesh": mesh})
        live = state["objects"].get(oid, {})
        meshes = (
            [part["local_mesh"] for part in live["assembly"]["parts"].values()]
            if live.get("assembly")
            else [mesh]
        )
        for asset in meshes:
            source = world / asset
            dest = geometry / asset
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
    if read(world / "world.json")["geometry_revision"] != spec["geometry_revision"]:
        return
    obs = read(stepdir / "observation.json")
    frames = []
    cams = []
    for name in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
        rgb = stepdir / (name + ".png")
        if not rgb.exists() or name not in obs["cameras"]:
            continue
        cam = obs["cameras"][name]
        size = Image.open(rgb).size
        frames.append(
            {
                "state": state,
                "view": {
                    "image_size": size,
                    "K": cam["intrinsic_matrix"],
                    "T_W_C": (
                        np.asarray(cam["extrinsic_matrix"]) @ np.diag([1.0, -1.0, -1.0, 1.0])
                    ).tolist(),
                },
            }
        )
        cams.append(name)
    atomic(
        work / "render_input.json", {"world": str(geometry), "objects": objects, "frames": frames}
    )
    env = dict(os.environ, CODEWORLD_RENDER_RUNTIME=RUNTIME)
    libs = Path(RUNTIME) / ".system-libs/usr/lib/x86_64-linux-gnu"
    if libs.is_dir():
        env["LD_LIBRARY_PATH"] = str(libs) + ":" + env.get("LD_LIBRARY_PATH", "")
    # CPU software rendering avoids taking a simulator's GPU slot.
    env["CUDA_VISIBLE_DEVICES"] = ""
    with (out / "render.log").open("w") as log:
        subprocess.run(
            [
                RUNTIME + "/.render-tools/node/bin/node",
                str(RESOURCE / "render_history.mjs"),
                str(work / "render_input.json"),
            ],
            env=env,
            check=True,
            timeout=100,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    images = {}
    version = f"v4_{step:04d}_{row['world_state'][:12]}"
    for i, name in enumerate(cams):
        real = Image.open(stepdir / (name + ".png")).convert("RGB")
        recon = Image.open(work / "reconstruction_frames" / f"{i:05d}.png").convert("RGB")
        w, h = real.size
        panel = Image.new("RGB", (2 * w, h + 48), "#f3f3eb")
        panel.paste(real, (0, 36))
        panel.paste(recon, (w, 36))
        draw = ImageDraw.Draw(panel)
        draw.text((12, 12), f"Real observation | STEP {step}", fill="#23352e")
        draw.text(
            (w + 12, 12), f"CodeWorld | STEP {step} | revision {state['revision']}", fill="#23352e"
        )
        path = out / (version + "_" + name + ".png")
        panel.save(path)
        images[name] = str(path)
    atomic(
        receipt,
        {
            "render_version": 4,
            "id": ident,
            "step": step,
            "state_hash": row["world_state"],
            "revision": state["revision"],
            "geometry_revision": state["geometry_revision"],
            "time": time.time(),
            "image_time": (stepdir / "cam_head.png").stat().st_mtime,
            "images": images,
            "hidden_objects": hidden,
            "appearance": "original reconstructed GLB materials and textures",
            "alignment": "same recorded host step; latest audited reconstructed state at that step",
            "material_report": read(work / "material_report.json"),
        },
    )
    print(ident, "comparison step", step, flush=True)


def main():
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    jobs = {}
    last = {}
    while True:
        try:
            status = read(ROOT / "status.json")
            session = read(ROOT / "manifest.json")["created_at"]
            candidates = [
                r
                for r in status["runs"]
                if r.get("stage") == "POLICY"
                and (r["status"] == "RUNNING" or r.get("finished_at", 0) >= session)
            ]
            for ident, f in list(jobs.items()):
                if not f.done():
                    continue
                try:
                    f.result()
                except Exception:
                    (ROOT / ident / "comparison_error.txt").write_text(traceback.format_exc())
                last[ident] = time.time()
                del jobs[ident]
            if not jobs:
                due = [r for r in candidates if time.time() - last.get(r["id"], 0) > 25]
                if due:
                    r = min(due, key=lambda r: (r["status"] != "RUNNING", last.get(r["id"], 0)))
                    ident = r["id"]
                    jobs[ident] = pool.submit(render, ident)
            if status.get("all_finished") and not jobs:
                break
        except Exception:
            print(traceback.format_exc(), flush=True)
        time.sleep(5)


if __name__ == "__main__":
    main()
