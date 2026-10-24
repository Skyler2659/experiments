"""Numerical and cost-structure regressions for indexed pose-update geometry."""
import unittest
from unittest.mock import patch
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from codeworld_workspace.surface_index import SurfaceIndex, object_surface, pose_observability


class SurfaceIndexTests(unittest.TestCase):
    def test_inverse_point_query_matches_transformed_mesh_distance(self):
        mesh=trimesh.creation.icosphere(subdivisions=2,radius=.04)
        points=np.random.default_rng(7).normal(size=(40,3))*.05
        surface=SurfaceIndex(mesh)
        tree=mesh.triangles_tree
        for rot,offset in [([0,0,0],[0,0,0]),([.4,-.2,1.1],[.1,-.03,.02]),([0,0,np.pi],[0,0,.1])]:
            T=np.eye(4);T[:3,:3]=Rotation.from_rotvec(rot).as_matrix();T[:3,3]=offset
            expected_mesh=mesh.copy();expected_mesh.apply_transform(T)
            expected=trimesh.proximity.closest_point_naive(expected_mesh,points)[1]
            np.testing.assert_allclose(surface.distances(points,T),expected,rtol=1e-10,atol=1e-12)
            self.assertIs(tree,mesh.triangles_tree)
        self.assertEqual(surface.queries,3)

    def test_geometry_cache_reuses_index_across_pose_updates(self):
        # Use the publication fixture to check cache identity, not a wall-clock
        # threshold affected by unrelated jobs on a shared workstation.
        from codeworld_robot.tests.test_world_sensor_audit import WorldSensorAuditTests
        fixture=WorldSensorAuditTests();fixture.setUp()
        try:
            state=fixture.world.state();first=object_surface(fixture.world,'box',state)
            state['objects']['box']['T_W_object'][0][3]+=.3
            with patch.object(fixture.world,'_mesh',side_effect=AssertionError('geometry unexpectedly reloaded')):
                second=object_surface(fixture.world,'box',state)
            self.assertIs(first,second)
        finally:
            fixture.doCleanups()

    def test_optional_diagnostic_identifies_box_symmetry_without_false_unique_pose(self):
        mesh=trimesh.creation.box([.1,.06,.04]);surface=SurfaceIndex(mesh)
        # Complete surface evidence, including exact geometry symmetries.
        points=np.concatenate([mesh.vertices,mesh.triangles_center])
        result=pose_observability(surface,np.eye(4),points,[np.zeros(3),np.array([1.,0,0])],.001)
        self.assertFalse(result['orientation_observed'])
        self.assertEqual(result['orientation_geometry_equivalent_axes'],[True,True,True])
        symmetric=[v for v in result['orientation_alternatives'] if v['angle_deg']==180]
        self.assertTrue(all(v['geometry_equivalent'] for v in symmetric))
        self.assertTrue(all(v['p90_m']<1e-12 for v in symmetric))


if __name__=='__main__':unittest.main()
