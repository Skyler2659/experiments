import numpy as np
import pytest
import trimesh
from privilege.robot_assets import export_robot


def test_public_urdf_state_positions_visuals_without_control(tmp_path):
    urdf = tmp_path / "robot.urdf"
    urdf.write_text("""<robot name="r"><link name="base"/><link name="finger">
    <visual><geometry><box size="0.1 0.1 0.1"/></geometry></visual></link>
    <joint name="slide" type="prismatic"><parent link="base"/><child link="finger"/>
    <axis xyz="1 0 0"/></joint></robot>""")
    t = np.eye(4)
    t[2, 3] = 1
    spec = {
        "urdf": str(urdf),
        "instances": [
            {
                "T_W_base": t.tolist(),
                "joints": {"slide": {"state": "arm.gripper", "scale": 0.1}},
                "validate_link": "finger",
                "position_state": "arm.position",
            }
        ],
    }
    result = export_robot(
        spec, {"arm": {"gripper": 0.5, "position": [0.05, 0, 1]}}, tmp_path / "robot.glb"
    )
    mesh = trimesh.load(result, force="mesh")
    assert np.allclose(mesh.centroid, [0.05, 0, 1])
    with pytest.raises(ValueError, match="FK disagrees"):
        export_robot(spec, {"arm": {"gripper": 0.5, "position": [0, 0, 0]}}, tmp_path / "bad.glb")
