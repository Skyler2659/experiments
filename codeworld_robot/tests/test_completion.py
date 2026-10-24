import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
import trimesh
from codeworld_workspace import World
from codeworld_robot.io import atomic,read
from codeworld_robot.ownership import authorize
from codeworld_robot.completion import enforce


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        atomic(self.root/'task/task.json',{'goal':'cup on plate and pudding beside plate'})
        atomic(self.root/'task/robot_connection.json',{'io_directory':str(self.root/'robot_io')})
        atomic(self.root/'robot_io/robot_state.json',{'robot_state_ref':'episode:10'})
        self.capture='robot_io/captures/obs/capture.json'
        atomic(self.root/self.capture,{'views':[{'view_id':'front'}],'robot_state':{'robot_state_ref':'episode:10'}})
        atomic(self.root/'world.json',{'geometry_revision':'fixture','objects':{x:f'objects/{x}.json' for x in ['cup','plate','pudding']},'bound_files':{}})
        state={'revision':0,'geometry_revision':'fixture','history':[],'objects':{}}
        for name,pos in [('plate',[0,0,0]),('cup',[0,0,.1]),('pudding',[.5,.1,0])]:
            mesh=trimesh.creation.box([.05,.05,.05]);g=self.root/'geometry'/f'{name}.ply';g.parent.mkdir(exist_ok=True);mesh.export(g)
            T=np.eye(4);T[:3,3]=pos
            atomic(self.root/f'objects/{name}.json',{'geometry':{'local_mesh':str(g.relative_to(self.root))}})
            state['objects'][name]={'T_W_object':T.tolist(),'basis':'observation_estimate','review':{'pose':{'status':'observed','revision':0,'evidence':[self.capture]}}}
        atomic(self.root/'state/current.json',state);self.w=World(self.root)
        self.goals={
            'cup':{'description':'cup near plate','conditions':[{'kind':'distance','object':'cup','reference':'plate','axes':[0,1],'max':.05}]},
            'pudding':{'description':'pudding on declared right and adjacent to plate','conditions':[
                {'kind':'relative_center','object':'pudding','reference':'plate','axis':[0,1,0],'frame':'W','min':.06,'max':.15},
                {'kind':'distance','object':'pudding','reference':'plate','axes':[0,1],'max':.2}]}}
        self.w.define_completion(self.goals)
        self.actor={'role':'codex_cli','run_id':'test'}
    def verify(self,ids=None):return self.w.verify_completion(ids,robot_state=read(self.root/'robot_io/robot_state.json'))
    def satisfied(self):
        s=self.w.state();s['objects']['pudding']['T_W_object'][0][3]=0;atomic(self.w.state_path,s)
    def packet(self,**args):return {'op':'finish','args':args,'actor':self.actor}
    def receipt(self,r):
        atomic(self.root/'work/check.json',r);return 'work/check.json'
    def test_l07_direction_alone_does_not_prove_adjacency(self):
        r=self.verify();self.assertEqual(r['status'],'NOT_SATISFIED')
        self.assertEqual(r['goals']['pudding']['checks'][0]['status'],'SATISFIED')
        self.assertEqual(r['goals']['pudding']['checks'][1]['status'],'NOT_SATISFIED')
    def test_subtask_does_not_allow_finishing_entire_task(self):
        r=self.verify(['cup']);self.assertEqual(r['status'],'VERIFIED')
        with self.assertRaises(ValueError):enforce(self.root,self.packet(verification=self.receipt(r)))
    def test_valid_full_report_and_direct_dispatch_bare_finish(self):
        self.satisfied();r=self.verify();self.assertEqual(r['status'],'VERIFIED')
        enforce(self.root,self.packet(verification=self.receipt(r)))
        with self.assertRaises(ValueError):authorize(self.root,self.packet())
    def test_obsolete_report_and_observation_rejected_after_motion(self):
        self.satisfied();r=self.verify();ref=self.receipt(r)
        atomic(self.root/'robot_io/robot_state.json',{'robot_state_ref':'episode:11'})
        self.assertEqual(self.verify()['status'],'INCONCLUSIVE')
        with self.assertRaises(ValueError):enforce(self.root,self.packet(verification=ref))
    def test_pose_prediction_and_pending_review_cannot_count(self):
        self.satisfied();T=self.w.state()['objects']['cup']['T_W_object']
        self.w.update({'cup':T},basis='action_prediction')
        self.assertEqual(self.verify()['status'],'INCONCLUSIVE')
        self.w.update({'cup':T},basis='observation_estimate')
        self.assertEqual(self.verify()['status'],'INCONCLUSIVE')
        self.w.update({'cup':T},basis='observation_estimate',evidence=[self.capture])
        self.assertEqual(self.verify()['status'],'VERIFIED')
        self.w.assess({'cup':{'pose':'pending'}})
        self.assertEqual(self.verify()['status'],'INCONCLUSIVE')
    def test_equivalent_observation_bookkeeping_does_not_invalidate_success(self):
        self.satisfied();ref=self.receipt(self.verify())
        self.w.assess({'cup':{'pose':'observed'}},evidence=[self.capture])
        enforce(self.root,self.packet(verification=ref))
    def test_fail_stop_remain_available_without_success(self):
        enforce(self.root,self.packet(outcome='failed',reason='cannot proceed'))
        authorize(self.root,{'op':'finish','actor':{'role':'supervisor_stop'}})
        with self.assertRaises(ValueError):enforce(self.root,self.packet(outcome='failed'))
    def test_contract_cannot_be_omitted_or_rewritten_after_motion(self):
        p={'op':'gripper','actor':self.actor,'args':{'mode':'OPEN'}}
        enforce(self.root,p)
        g=copy.deepcopy(self.goals);del g['pudding']
        with self.assertRaises(ValueError):self.w.define_completion(g)
        (self.root/'task/completion.json').unlink()
        with self.assertRaises(ValueError):enforce(self.root,p)
        with self.assertRaises(ValueError):self.w.define_completion(self.goals)
    def test_numeric_and_discrete_state_need_field_specific_current_evidence(self):
        # A current angle update must not refresh an older closed/open field.
        g={'drawer':{'description':'physical drawer state','conditions':[{'kind':'state','object':'cup','field':'angle','max':.05},{'kind':'state','object':'cup','field':'closed','equals':True}]}}
        (self.root/'task/completion.json').unlink();self.w.define_completion(g)
        self.w.update_state({'cup':{'angle':0.,'closed':True}},evidence=[self.capture])
        self.assertEqual(self.verify()['status'],'VERIFIED')
        atomic(self.root/'robot_io/robot_state.json',{'robot_state_ref':'episode:11'})
        cap2='robot_io/captures/new/capture.json';atomic(self.root/cap2,{'views':[{}],'robot_state':{'robot_state_ref':'episode:11'}})
        self.w.update_state({'cup':{'angle':0.}},evidence=[cap2])
        self.assertEqual(self.verify()['status'],'INCONCLUSIVE')
    def test_arbitrary_visual_boolean_and_hypothesis_rejected(self):
        g={'bad':{'description':'visual claim','conditions':[{'kind':'visual_success','object':'cup','success':True}]}}
        with self.assertRaises(ValueError):self.w.define_completion(g)
        (self.root/'scenarios').mkdir();branch=self.w.fork('plan')
        with self.assertRaises(ValueError):branch.verify_completion(robot_state={'robot_state_ref':'episode:10'})

if __name__=='__main__':unittest.main()
