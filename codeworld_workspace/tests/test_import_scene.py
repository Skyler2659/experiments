import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import trimesh

from codeworld_workspace.import_scene import import_scene
from codeworld_workspace.world import World


class ImportSceneTests(unittest.TestCase):
    def test_y_up_textured_parts_and_recipe_survive_current_world_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'source'
            obj = source/'objects/body'
            obj.mkdir(parents=True)
            scene = trimesh.Scene()
            for i, color in enumerate([(255, 0, 0), (0, 0, 255)]):
                mesh = trimesh.creation.box([.1, .2, .3])
                mesh.visual = trimesh.visual.texture.TextureVisuals(
                    uv=np.zeros((len(mesh.vertices), 2)),
                    material=trimesh.visual.material.PBRMaterial(
                        name=str(i), baseColorTexture=Image.new('RGB', (2, 2), color)))
                transform = np.eye(4)
                transform[0, 3] = i*.4
                scene.add_geometry(mesh, node_name='part'+str(i), transform=transform)
            scene.export(obj/'mesh.glb')
            recipe = {'family': 'fixture_two_parts', 'observed_dimensions_m': [.5, .2, .3]}
            (obj/'geometry_recipe.json').write_text(json.dumps(recipe))
            mt = np.eye(4)
            mt[:3, 3] = [.02, 0, .03]
            (source/'scene.json').write_text(json.dumps({
                'frame': 'robot_base', 'units': 'm', 'nodes': [{
                    'object_id': 'body', 'parent_id': None,
                    'pose_parent_object': {'translation_m': [.6, .2, .1],
                        'quaternion_xyzw': [0, 0, math.sqrt(.5), math.sqrt(.5)]},
                    'asset': {'mesh_ref': 'objects/body/mesh.glb',
                              'T_object_mesh': mt.tolist(), 'up_axis': 'Y'}}]}))
            out = Path(tmp)/'workspace'
            import_scene(source, out)
            world = World(out)
            np.testing.assert_allclose(world.mesh('body').bounds,
                [[.45, .17, .03], [.75, .67, .23]], atol=2e-6)
            asset = trimesh.load(out/'geometry/body.glb', force='scene', process=False)
            colors = {tuple(np.asarray(g.visual.material.baseColorTexture)[0, 0, :3])
                      for g in asset.geometry.values()}
            self.assertEqual(colors, {(255, 0, 0), (0, 0, 255)})
            self.assertEqual(len(world.parts('body')['nodes']), 2)
            self.assertEqual(world.object('body')['declared']['reconstruction_recipe'], recipe)
            world.verify_binding()
            (out/'source/reconstruction/body/geometry_recipe.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'changed'):
                world.verify_binding()


if __name__ == '__main__':
    unittest.main()
