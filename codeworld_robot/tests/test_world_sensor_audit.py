"""Offline regressions for RGB-D publication and grasp-evidence freshness."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import trimesh

from codeworld_robot.io import atomic
from codeworld_workspace import World
from codeworld_workspace.completion import state_ref


class WorldSensorAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        atomic(self.root/'task/world_flow.json',{'required':True})
        atomic(self.root/'task/robot_connection.json',{'io_directory':str(self.root/'robot_io')})
        self.robot={'robot_state_ref':'fixture:0','timestamp_ns':0}
        atomic(self.root/'robot_io/robot_state.json',self.robot)
        (self.root/'geometry').mkdir()
        trimesh.creation.box([.1,.2,.3]).export(self.root/'geometry/box.ply')
        atomic(self.root/'objects/box.json',{'geometry':{'local_mesh':'geometry/box.ply'}})
        atomic(self.root/'world.json',{'geometry_revision':'fixture','objects':{'box':'objects/box.json'},'bound_files':{}})
        atomic(self.root/'state/current.json',{'revision':0,'history':[],'geometry_revision':'fixture',
            'objects':{'box':{'T_W_object':np.eye(4).tolist(),'basis':'imported_reconstruction'}}})
        self.world=World(self.root)
        self.pose=np.eye(4);self.pose[2,3]=1.15

    def capture(self,second_bad_view=False,robot=None):
        folder=self.root/'robot_io/captures/test';folder.mkdir(parents=True,exist_ok=True)
        np.save(folder/'depth.npy',np.ones((64,64)))
        mask=np.zeros((64,64),bool);mask[27:37,27:37]=True
        np.save(self.root/'mask.npy',mask)
        view={'view_id':'cam','depth_uri':str(folder/'depth.npy'),'depth_convention':'camera_z_m',
              'K':[[100,0,32],[0,100,32],[0,0,1]],'T_W_C':np.eye(4).tolist()}
        views=[view]
        if second_bad_view:
            np.save(folder/'bad_depth.npy',np.full((64,64),np.nan))
            views.append({**copy.deepcopy(view),'view_id':'bad','depth_uri':str(folder/'bad_depth.npy')})
        path=folder/'capture.json'
        atomic(path,{'robot_state':self.robot if robot is None else robot,'views':views})
        return str(path)

    def proposals(self,second_bad_view=False):
        masks={'cam':'mask.npy'}
        if second_bad_view:masks['bad']='mask.npy'
        return {'box':{'T_W_object':self.pose.tolist(),'masks':masks}}

    def advance_robot(self):
        atomic(self.root/'robot_io/robot_state.json',{'robot_state_ref':'fixture:1','timestamp_ns':1})

    def test_invalid_declared_view_cannot_be_silently_dropped(self):
        result=self.world.integrate_rgbd(self.capture(True),self.proposals(True))
        self.assertEqual(result['objects']['box']['status'],'INCONCLUSIVE')
        self.assertEqual(len(result['objects']['box']['views']),2)
        self.assertNotIn('sensor_validation',self.world.state()['objects']['box'])

    def test_measurement_thresholds_must_be_finite_and_typed(self):
        capture=self.capture()
        for options in ({'max_p90_m':float('nan')},{'max_p90_m':float('inf')},
                        {'max_p90_m':True},{'max_p90_m':0},{'max_p90_m':-.01},{'min_points':float('nan')},
                        {'min_points':32.5},{'min_points':True}):
            with self.subTest(options=options),self.assertRaises(ValueError):
                self.world.integrate_rgbd(capture,self.proposals(),**options)

    def test_surface_patch_does_not_establish_unique_translation(self):
        capture=self.capture()
        mask=np.zeros((64,64),bool);mask[28:36,31:35]=True
        np.save(self.root/'mask.npy',mask)
        for offset in (0.,.02):
            proposals=self.proposals();proposals['box']['T_W_object'][0][3]=offset
            result=self.world.integrate_rgbd(capture,proposals,diagnose_observability=True)['objects']['box']
            self.assertEqual(result['status'],'SURFACE_CONSISTENT')
            self.assertFalse(result['translation_observed'])
            self.assertEqual(len(result['translation_alternatives']),6)
            self.assertFalse(self.world.state()['objects']['box']['sensor_validation']['translation_observed'])

    def test_normal_update_keeps_fit_quality_without_mandatory_ambiguity_search(self):
        result=self.world.integrate_rgbd(self.capture(),self.proposals())['objects']['box']
        self.assertEqual(result['status'],'SURFACE_CONSISTENT')
        self.assertEqual(result['surface_queries'],1)
        self.assertEqual(result['observability']['status'],'NOT_EVALUATED')
        self.assertIsNone(result['translation_observed'])
        self.assertIsNone(result['orientation_observed'])
        self.assertEqual(result['translation_alternatives'],[])
        stamp=self.world.state()['objects']['box']['sensor_validation']
        self.assertLess(stamp['surface_error_m'],1e-6)
        self.assertEqual(stamp['acceptance_tolerance_m'],.008)

    def test_explicit_coarse_tolerance_preserves_actual_error(self):
        # A 12-mm geometric mismatch fails the default check, but can be retained
        # as an explicitly coarse world estimate without hiding its residual.
        capture=self.capture();proposal=self.proposals()
        proposal['box']['T_W_object'][2][3]+=.012
        rejected=self.world.integrate_rgbd(capture,proposal)['objects']['box']
        self.assertEqual(rejected['status'],'INCONCLUSIVE')
        accepted=self.world.integrate_rgbd(capture,proposal,max_p90_m=.02)['objects']['box']
        self.assertEqual(accepted['status'],'SURFACE_CONSISTENT')
        self.assertAlmostEqual(accepted['p90_surface_error_m'],.012,places=5)
        self.assertFalse(accepted['within_default_surface_tolerance'])
        self.assertAlmostEqual(self.world.state()['objects']['box']['sensor_validation']['surface_error_m'],.012,places=5)
        proposal['box']['T_W_object'][2][3]+=.1
        self.assertEqual(self.world.integrate_rgbd(capture,proposal,max_p90_m=.02)['objects']['box']['status'],'INCONCLUSIVE')

    def test_capture_requires_nonempty_backend_version(self):
        atomic(self.root/'robot_io/robot_state.json',{})
        with self.assertRaises(ValueError):
            self.world.integrate_rgbd(self.capture(robot={}),self.proposals())

    def test_candidate_assembly_missing_hash_cannot_corrupt_published_state(self):
        capture=self.capture();self.world.sync_robot_feedback();before=self.world.state()
        proposals=self.proposals()
        proposals['box']['assembly']={'geometry_revision':'candidate','parameters':{'on':True},
            'parts':{'body':{'local_mesh':'geometry/box.ply','T_object_part':np.eye(4).tolist()}}}
        with self.assertRaisesRegex(ValueError,'sha256'):
            self.world.integrate_rgbd(capture,proposals)
        self.assertEqual(self.world.state(),before)

    def test_rgbd_commit_rechecks_live_robot_after_fitting(self):
        from codeworld_robot.grasp import _cloud
        capture=self.capture();self.world.sync_robot_feedback();before=self.world.state()
        def moving_cloud(*args,**kwargs):
            value=_cloud(*args,**kwargs);self.advance_robot();return value
        with patch('codeworld_robot.grasp._cloud',side_effect=moving_cloud),self.assertRaises(ValueError):
            self.world.integrate_rgbd(capture,self.proposals())
        self.assertEqual(self.world.state(),before)

    def test_unrelated_feedback_update_is_preserved_when_fit_commits(self):
        from codeworld_robot.grasp import _cloud
        capture=self.capture();self.world.sync_robot_feedback()
        def unrelated_update(*args,**kwargs):
            result=_cloud(*args,**kwargs)
            state=self.world.state();state['revision']+=1
            state['objects']['unrelated']={'T_W_object':np.eye(4).tolist(),'basis':'other_update'}
            state.setdefault('measurements',{})['new_feedback']={'value':'preserve this'}
            state['history'].append({'kind':'unrelated_feedback'})
            atomic(self.world.state_path,state)
            return result
        with patch('codeworld_robot.grasp._cloud',side_effect=unrelated_update):
            result=self.world.integrate_rgbd(capture,self.proposals())
        self.assertEqual(result['objects']['box']['status'],'SURFACE_CONSISTENT')
        final=self.world.state()
        self.assertEqual(final['objects']['unrelated']['basis'],'other_update')
        self.assertEqual(final['measurements']['new_feedback']['value'],'preserve this')
        self.assertEqual(final['history'][-2]['kind'],'unrelated_feedback')

    def test_same_object_change_while_fitting_is_not_overwritten(self):
        from codeworld_robot.grasp import _cloud
        capture=self.capture();self.world.sync_robot_feedback()
        def target_update(*args,**kwargs):
            result=_cloud(*args,**kwargs)
            state=self.world.state();state['revision']+=1
            state['objects']['box']['T_W_object'][0][3]=.33
            atomic(self.world.state_path,state)
            return result
        with patch('codeworld_robot.grasp._cloud',side_effect=target_update),self.assertRaisesRegex(ValueError,'Target object changed'):
            self.world.integrate_rgbd(capture,self.proposals())
        self.assertEqual(self.world.state()['objects']['box']['T_W_object'][0][3],.33)
        self.assertNotIn('sensor_validation',self.world.state()['objects']['box'])

    def test_changed_target_geometry_while_fitting_is_rejected(self):
        from codeworld_robot.grasp import _cloud
        capture=self.capture();self.world.sync_robot_feedback()
        def geometry_update(*args,**kwargs):
            result=_cloud(*args,**kwargs)
            trimesh.creation.box([.2,.2,.3]).export(self.root/'geometry/box.ply')
            return result
        with patch('codeworld_robot.grasp._cloud',side_effect=geometry_update),self.assertRaisesRegex(ValueError,'Target geometry changed'):
            self.world.integrate_rgbd(capture,self.proposals())
        self.assertNotIn('sensor_validation',self.world.state()['objects']['box'])

    def test_old_grasp_capture_cannot_be_restamped_as_current(self):
        capture=self.capture();self.advance_robot()
        with patch('codeworld_robot.grasp.assess_grasp') as assess,self.assertRaises(ValueError):
            self.world.ingest_grasp_feedback(after_capture=capture,object_id='box')
        assess.assert_not_called()
        self.assertFalse(self.world.state().get('measurements'))

    def test_grasp_commit_rechecks_live_robot_after_assessment(self):
        capture=self.capture();self.world.sync_robot_feedback();before=self.world.state()
        def moving_assessment(*args,**kwargs):
            self.advance_robot();return {'status':'LIFT_EVIDENCE'}
        with patch('codeworld_robot.grasp.assess_grasp',side_effect=moving_assessment),self.assertRaises(ValueError):
            self.world.ingest_grasp_feedback(after_capture=capture,object_id='box')
        self.assertEqual(self.world.state(),before)

    def test_fresh_grasp_measurement_keeps_capture_version(self):
        capture=self.capture()
        with patch('codeworld_robot.grasp.assess_grasp',return_value={'status':'INCONCLUSIVE'}):
            result=self.world.ingest_grasp_feedback(after_capture=capture,object_id='box')
        measurement=self.world.state()['measurements'][result['measurement_id']]
        self.assertEqual(measurement['robot_state_ref'],state_ref(self.robot))
        self.assertEqual(Path(measurement['after_capture']).resolve(),Path(capture).resolve())


if __name__=='__main__':unittest.main()
