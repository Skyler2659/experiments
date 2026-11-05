"""Calibrated RGB records. Camera convention: +x right, +y down, +z forward."""

import copy
import hashlib
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from .world import atomic, read, rigid


def calibration(K, T_W_C, image_size):
    k = np.asarray(K, dtype=float)
    if (
        k.shape != (3, 3)
        or not np.isfinite(k).all()
        or k[0, 0] <= 0
        or k[1, 1] <= 0
        or not np.allclose(k[2], [0, 0, 1])
    ):
        raise ValueError("Expected finite pinhole K with positive focal lengths")
    if len(image_size) != 2 or any(type(v) is not int or v <= 0 for v in image_size):
        raise ValueError("image_size must be [width, height]")
    return k.tolist(), rigid(T_W_C).tolist()


def register(root, image, *, K, T_W_C, frame, robot_state=None, metadata=None):
    """Copy one observation into the world; recording does not estimate object poses."""
    return register_views(root, [{"view_id": "rgb", "image": image, "K": K,
                                  "T_W_C": T_W_C}], frame=frame,
                          robot_state=robot_state, metadata=metadata)


def register_views(root, views, *, frame, robot_state=None, metadata=None):
    """Archive synchronized named RGB views and their image-time calibration."""
    if not isinstance(frame, str) or not frame:
        raise ValueError("Name the common metric world frame")
    if not views or len({v["view_id"] for v in views}) != len(views):
        raise ValueError("Provide unique named views")
    ident = uuid.uuid4().hex
    directory = Path(root) / "observations" / ident
    directory.mkdir(parents=True)
    saved = []
    for index, view in enumerate(views):
        rgb = _rgb(view["image"])
        size = list(rgb.size)
        k, transform = calibration(view["K"], view["T_W_C"], size)
        filename = "rgb.png" if index == 0 else f"rgb_{index}.png"
        rgb.save(directory / filename)
        saved.append({"view_id": view["view_id"], "rgb_uri": filename,
                      "image_size": size, "K": k, "T_W_C": transform,
                      "camera_axes": "opencv",
                      "timing": copy.deepcopy(view.get("timing", {})),
                      "rgb_sha256": hashlib.sha256((directory / filename).read_bytes()).hexdigest()})
    packet = {"schema": "privilege-calibrated-rgb-v1", "observation_id": ident,
              "frame": frame, "recorded_at_ns": time.time_ns(),
              "robot_state": copy.deepcopy(robot_state or {}),
              "metadata": copy.deepcopy(metadata or {}), "views": saved}
    atomic(directory / "capture.json", packet)
    return {"capture": f"observations/{ident}/capture.json", **packet}


def _rgb(image):
    if isinstance(image, (str, Path)):
        with Image.open(image) as src:
            rgb = src.convert("RGB")
    elif isinstance(image, Image.Image):
        rgb = image.convert("RGB")
    else:
        array = np.asarray(image)
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("Expected uint8 HxWx3 RGB")
        rgb = Image.fromarray(array)
    return rgb


def load(root, capture):
    root = Path(root).resolve()
    path = (root / capture).resolve()
    path.relative_to(root / "observations")
    packet = read(path)
    if packet.get("schema") != "privilege-calibrated-rgb-v1":
        raise ValueError("Unsupported observation schema")
    if not packet.get("views"):
        raise ValueError("Expected calibrated RGB views")
    images = [_validate_view(path, view) for view in packet["views"]]
    return path, packet, images[0]  # primary image; packet retains every validated view


def _validate_view(path, view):
    if view.get("camera_axes") != "opencv":
        raise ValueError("Expected OpenCV camera axes")
    if view.get("depth_uri"):
        raise ValueError("This backend is calibrated RGB, not RGB-D")
    image = (path.parent / view["rgb_uri"]).resolve()
    image.relative_to(path.parent)
    if hashlib.sha256(image.read_bytes()).hexdigest() != view["rgb_sha256"]:
        raise ValueError("Observation image changed")
    with Image.open(image) as rgb:
        if list(rgb.size) != view["image_size"]:
            raise ValueError("Observation image size changed")
    calibration(view["K"], view["T_W_C"], view["image_size"])
    return image
