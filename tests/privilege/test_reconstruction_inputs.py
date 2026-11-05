import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from privilege.reconstruction_inputs import ensure_inputs, snapshot_inputs


class AssignedInputsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "reference.png").write_bytes(b"assigned rgb")
        (self.root / "known_robot.glb").write_bytes(b"posed public robot")
        self.metadata = {
            "input.json": {"reference": "reference.png"},
            "views.json": {"views": [{"rgb_uri": "reference.png", "rgb_sha256": hashlib.sha256(b"assigned rgb").hexdigest()}]},
            "camera_metadata.json": {"K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
            "robot_binding.json": {"robot_node": {"asset": {"sha256": hashlib.sha256(b"posed public robot").hexdigest()}}},
        }
        snapshot_inputs(self.root, self.metadata)

    def test_missing_calibration_restored_exactly(self):
        for name in ("views.json", "camera_metadata.json"):
            (self.root / name).unlink()
        self.assertEqual(set(ensure_inputs(self.root)["restored"]), {"views.json", "camera_metadata.json"})
        for name, expected in self.metadata.items():
            self.assertEqual(json.loads((self.root / name).read_text()), expected)
        self.assertEqual(ensure_inputs(self.root)["restored"], [])

    def test_changed_reference_rejects_repair(self):
        (self.root / "views.json").unlink()
        (self.root / "reference.png").write_bytes(b"different scene")
        with self.assertRaisesRegex(ValueError, "RGB differs"):
            ensure_inputs(self.root)
        self.assertFalse((self.root / "views.json").exists())

    def test_readable_metadata_is_not_overwritten(self):
        (self.root / "views.json").unlink()
        (self.root / "camera_metadata.json").write_text('{"K": []}')
        with self.assertRaisesRegex(ValueError, "metadata changed"):
            ensure_inputs(self.root)
        self.assertFalse((self.root / "views.json").exists())
        self.assertEqual(json.loads((self.root / "camera_metadata.json").read_text()), {"K": []})


if __name__ == "__main__":
    unittest.main()
