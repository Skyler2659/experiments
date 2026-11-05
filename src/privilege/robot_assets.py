"""Public URDF visual snapshots only. No IK, controller, simulator or motion API."""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from .world import rigid


def _origin(element):
    t = np.eye(4)
    if element is not None:
        t[:3, 3] = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
        t[:3, :3] = Rotation.from_euler(
            "xyz", np.fromstring(element.get("rpy", "0 0 0"), sep=" ")
        ).as_matrix()
    return t


def _value(state, binding):
    if isinstance(binding, (float, int)):
        return float(binding)
    value = state
    for key in binding["state"].split("."):
        value = value[int(key)] if isinstance(value, (list, tuple)) else value[key]
    result = float(value) * binding.get("scale", 1.0) + binding.get("offset", 0.0)
    if not np.isfinite(result):
        raise ValueError("Nonfinite public joint state")
    return result


def export_robot(spec, state, destination):
    """spec: urdf, instances[{T_W_base, joints:{URDF_name: {state,scale,offset}}}].

    Bind every independent movable joint explicitly, including fingers. A
    normalized gripper measurement must be mapped with its public model's units.
    Optional validate_link/position_state compares FK with public link position.
    """
    urdf = Path(spec["urdf"]).resolve()
    xml = ET.parse(urdf).getroot()
    joints = xml.findall("joint")
    roots = {x.get("name") for x in xml.findall("link")} - {
        j.find("child").get("link") for j in joints
    }
    if len(roots) != 1 or not spec.get("instances"):
        raise ValueError("Expected one URDF root and at least one robot instance")
    meshes = []
    for instance in spec["instances"]:
        values = {name: _value(state, binding) for name, binding in instance["joints"].items()}
        # Resolve mimic joints in dependency order, rather than relying on XML order.
        pending_mimics = [j for j in joints if j.find("mimic") is not None]
        while pending_mimics:
            ready = [j for j in pending_mimics if j.find("mimic").get("joint") in values]
            if not ready:
                raise ValueError("Unbound or cyclic mimic joint")
            for j in ready:
                m = j.find("mimic")
                values[j.get("name")] = values[m.get("joint")] * float(
                    m.get("multiplier", "1")
                ) + float(m.get("offset", "0"))
                pending_mimics.remove(j)
        transforms = {next(iter(roots)): rigid(instance["T_W_base"])}
        pending = list(joints)
        while pending:
            ready = [j for j in pending if j.find("parent").get("link") in transforms]
            if not ready:
                raise ValueError("Disconnected URDF")
            for joint in ready:
                kind = joint.get("type")
                delta = np.eye(4)
                if kind != "fixed":
                    value = values[joint.get("name")]
                    element = joint.find("axis")
                    axis = np.fromstring(
                        element.get("xyz", "1 0 0") if element is not None else "1 0 0", sep=" "
                    )
                    if axis.shape != (3,) or np.linalg.norm(axis) == 0:
                        raise ValueError("Invalid URDF axis")
                    axis /= np.linalg.norm(axis)
                    if kind in ("revolute", "continuous"):
                        delta[:3, :3] = Rotation.from_rotvec(axis * value).as_matrix()
                    elif kind == "prismatic":
                        delta[:3, 3] = axis * value
                    else:
                        raise ValueError("Unsupported URDF joint: " + str(kind))
                transforms[joint.find("child").get("link")] = (
                    transforms[joint.find("parent").get("link")]
                    @ _origin(joint.find("origin"))
                    @ delta
                )
                pending.remove(joint)
        if instance.get("validate_link"):
            expected = state
            for key in instance["position_state"].split("."):
                expected = expected[key]
            error = np.linalg.norm(
                transforms[instance["validate_link"]][:3, 3] - np.asarray(expected)
            )
            if error > instance.get("max_position_error_m", 0.003):
                raise ValueError(f"Robot asset FK disagrees with public state: {error} m")
        for link in xml.findall("link"):
            for visual in link.findall("visual"):
                geometry = visual.find("geometry")
                if geometry is None:
                    continue
                mesh_spec = geometry.find("mesh")
                if mesh_spec is not None:
                    filename = mesh_spec.get("filename")
                    if filename.startswith("package://"):
                        package, relative = filename[10:].split("/", 1)
                        path = Path(spec["packages"][package]) / relative
                    else:
                        path = urdf.parent / filename.removeprefix("file://")
                    mesh = trimesh.load(path, force="scene", process=False).to_geometry()
                    mesh.apply_scale(np.fromstring(mesh_spec.get("scale", "1 1 1"), sep=" "))
                elif geometry.find("box") is not None:
                    mesh = trimesh.creation.box(
                        np.fromstring(geometry.find("box").get("size"), sep=" ")
                    )
                elif geometry.find("cylinder") is not None:
                    element = geometry.find("cylinder")
                    mesh = trimesh.creation.cylinder(
                        float(element.get("radius")), float(element.get("length"))
                    )
                elif geometry.find("sphere") is not None:
                    mesh = trimesh.creation.icosphere(
                        radius=float(geometry.find("sphere").get("radius"))
                    )
                else:
                    raise ValueError("Unsupported URDF visual geometry")
                mesh.apply_transform(transforms[link.get("name")] @ _origin(visual.find("origin")))
                meshes.append(mesh)
    if not meshes:
        raise ValueError("URDF has no visual geometry")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    trimesh.util.concatenate(meshes).export(destination)
    return destination
