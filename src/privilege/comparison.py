"""Calibrated, CPU geometry rendering for agent visual feedback (no simulator)."""

import hashlib
import io
import time

import numpy as np
from PIL import Image, ImageDraw

from .observations import load
from .world import atomic


def _clip_near(triangle, near=0.001):
    output = []
    for a, b in zip(triangle, np.roll(triangle, -1, axis=0)):
        if a[2] >= near:
            output.append(a)
        if (a[2] >= near) != (b[2] >= near):
            output.append(a + (b - a) * ((near - a[2]) / (b[2] - a[2])))
    return [np.array([output[0], output[i], output[i + 1]]) for i in range(1, len(output) - 1)]


def render(snapshot, view, object_ids, *, transforms=None, deadline=None):
    """Z-buffer triangles using exactly the recorded OpenCV K and T_W_C.

    Colors identify objects, not recovered illumination/materials. The initial
    frozen robot is omitted; real-arm occlusion is visible in the paired RGB.
    """
    w, h = view["image_size"]
    canvas = np.full((h, w, 3), 235, np.uint8)
    depth = np.full((h, w), np.inf)
    labels = np.full((h, w), -1, np.int32)
    cam = np.linalg.inv(np.asarray(view["T_W_C"], float))
    k = np.asarray(view["K"], float)
    for index, oid in enumerate(object_ids):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError('CodeWorld comparison rendering exceeded its feedback budget; use current RGB')
        mesh = snapshot.mesh(oid)
        vertices = np.asarray(mesh.vertices)
        if transforms and oid in transforms:
            delta = transforms[oid] @ np.linalg.inv(snapshot.transform(oid))
            vertices = vertices @ delta[:3, :3].T + delta[:3, 3]
        vertices = vertices @ cam[:3, :3].T + cam[:3, 3]
        seed = hashlib.sha256(oid.encode()).digest()
        color = np.array([65 + b % 150 for b in seed[:3]], float)
        for face_index, face in enumerate(mesh.faces):
            if face_index % 64 == 0 and deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError('CodeWorld comparison rendering exceeded its feedback budget; use current RGB')
            for tri in _clip_near(vertices[face]):
                pixel = tri @ k.T
                pixel = pixel[:, :2] / pixel[:, 2:]
                lo = np.maximum(np.floor(pixel.min(axis=0)).astype(int), [0, 0])
                hi = np.minimum(np.ceil(pixel.max(axis=0)).astype(int), [w - 1, h - 1])
                if np.any(hi < lo):
                    continue
                a, b, c = pixel
                ab, ac = b - a, c - a
                denom = ab[0] * ac[1] - ab[1] * ac[0]
                if abs(denom) < 1e-9:
                    continue
                xx, yy = np.meshgrid(np.arange(lo[0], hi[0] + 1) + .5,
                                     np.arange(lo[1], hi[1] + 1) + .5)
                u = ((b[0] - xx) * (c[1] - yy) - (b[1] - yy) * (c[0] - xx)) / denom
                v = ((c[0] - xx) * (a[1] - yy) - (c[1] - yy) * (a[0] - xx)) / denom
                t = 1 - u - v
                inv_z = u / tri[0, 2] + v / tri[1, 2] + t / tri[2, 2]
                z = np.divide(1., inv_z, out=np.full_like(inv_z, np.inf), where=inv_z > 0)
                region = np.s_[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]
                mask = (u >= -1e-7) & (v >= -1e-7) & (t >= -1e-7) & (z < depth[region])
                normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
                shade = .65 + .35 * abs(normal[2]) / max(np.linalg.norm(normal), 1e-12)
                depth[region][mask] = z[mask]
                labels[region][mask] = index
                canvas[region][mask] = (color * shade).astype(np.uint8)
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    for index, oid in enumerate(object_ids):
        yy, xx = np.where(labels == index)
        if len(xx) > 30:
            draw.text((int(np.median(xx)), int(np.median(yy))), oid, fill="black",
                      stroke_width=1, stroke_fill="white")
    return image, labels


def compare(world, capture, *, object_ids=None, wall_budget_s=None):
    """Return/save one real-vs-world panel per camera; no automatic pose claim."""
    deadline = None if wall_budget_s is None else time.monotonic() + float(wall_budget_s)
    path, packet, _ = load(world.root, capture)
    snapshot = world.snapshot()
    ids = list(object_ids) if object_ids is not None else [
        oid for oid in snapshot.objects() if world.object(oid).get("role") != "robot_snapshot"
    ]
    overrides = {}
    from .tracking import arm_pose

    image_state = packet.get("metadata", {}).get("image_robot_state")
    if image_state:
        for arm, held in snapshot._state.get("attachments", {}).items():
            overrides[held["object_id"]] = arm_pose(image_state, arm) @ np.asarray(held["T_E_object"])
    panels, originals, files = [], [], []
    for i, view in enumerate(packet["views"]):
        rgb_path = path.parent / view["rgb_uri"]
        with Image.open(rgb_path) as rgb:
            real = rgb.convert("RGB")
        predicted, _ = render(snapshot, view, ids, transforms=overrides, deadline=deadline)
        w, h = real.size
        panel = Image.new("RGB", (2 * w, h + 24), "white")
        panel.paste(real, (0, 24)); panel.paste(predicted, (w, 24))
        draw = ImageDraw.Draw(panel)
        draw.text((5, 5), view["view_id"] + " | REAL RGB", fill="black")
        draw.text((w + 5, 5), f"CodeWorld r{snapshot.revision} | geometry; robot omitted", fill="black")
        buffer = io.BytesIO(); panel.save(buffer, format="PNG")
        filename = f"comparison_r{snapshot.revision}_{i}.png"
        (path.parent / filename).write_bytes(buffer.getvalue())
        panels.append(buffer.getvalue()); originals.append(rgb_path.read_bytes())
        files.append(str((path.parent / filename).relative_to(world.root)))
    report = {"revision": snapshot.revision, "capture": capture, "files": files,
              "cameras": [v["view_id"] for v in packet["views"]],
              "objects": ids, "camera_match": "recorded_K_and_T_W_C",
              "interpretation": "Compare object pose/shape; ignore lighting and real-robot occlusion."}
    atomic(path.parent / f"comparison_r{snapshot.revision}.json", report)
    return report, {"images": panels, "originals": originals, "camera_names": [c + ".png" for c in report["cameras"]],
                    "labels": [c + "_real_and_codeworld" for c in report["cameras"]]}
