import json
from types import SimpleNamespace
import numpy as np
import pytest
from robodojo_benchmark.ground_truth import triangulate, GroundTruthWorld, local_mesh
from codeworld_workspace import World
from codeworld_workspace.completion import measured_pose_supported, oracle_covers_feedback


def test_concave_polygon_and_hole_face_are_not_convexified():
    p=np.array([[0,0,0],[3,0,0],[3,1,0],[1,1,0],[1,3,0],[0,3,0]],float)
    f=triangulate(p,[6,3],[0,1,2,3,4,5,0,1,2],holes={1})
    areas=np.linalg.norm(np.cross(p[f[:,1]]-p[f[:,0]],p[f[:,2]]-p[f[:,0]]),axis=1)/2
    assert len(f)==4 and np.isclose(areas.sum(),5)
    triangles=np.tile([[0,1,2],[3,4,5]],(50000,1))
    actual=triangulate(p,np.full(len(triangles),3),triangles.ravel(),holes={3,7})
    np.testing.assert_array_equal(actual,np.delete(triangles,[3,7],axis=0))


def test_usd_transform_export_and_live_oracle(tmp_path):
    pxr=pytest.importorskip('pxr')
    from pxr import Usd,UsdGeom,Gf
    stage=Usd.Stage.CreateInMemory();UsdGeom.SetStageMetersPerUnit(stage,1.)
    parent=UsdGeom.Xform.Define(stage,'/object')
    parent.AddTranslateOp().Set(Gf.Vec3d(1,2,3));parent.AddScaleOp().Set(Gf.Vec3f(2,3,4))
    cube=UsdGeom.Cube.Define(stage,'/object/cube');cube.GetSizeAttr().Set(.1)
    mesh=local_mesh(stage,'/object')
    np.testing.assert_allclose(mesh.extents,[.2,.3,.4],atol=1e-6)
    pos=np.array([1.,2.,3.])
    obj=SimpleNamespace(prim_path='/object',get_world_pose=lambda:(pos.copy(),np.array([1.,0,0,0])))
    sm=SimpleNamespace(stage=stage,_rigid_and_dynamic_objects=[{'cube':obj}],_geometry_objects=[{}])
    env=SimpleNamespace(scene_manager=sm)
    gt=GroundTruthWorld(env,tmp_path,np.array([1.,2.,2.]))
    gt.sync({'robot_state_ref':'episode:0','control_steps':0,'official_actions':0})
    world=World(tmp_path);world.verify_binding()
    np.testing.assert_allclose(np.mean(world.bounds('cube')['bounds_W_m'],axis=0),[0,0,1],atol=1e-6)
    state=world.state();assert measured_pose_supported(state,'cube')
    assert state['objects']['cube']['basis']=='oracle_state'
    assert 'sensor_validation' not in state['objects']['cube']
    pos[0]+=.5;gt.sync({'robot_state_ref':'episode:20','control_steps':20,'official_actions':2})
    np.testing.assert_allclose(np.mean(World(tmp_path).bounds('cube')['bounds_W_m'],axis=0),[.5,0,1],atol=1e-6)
    state=world.state()
    assert oracle_covers_feedback(state,'cube',{'robot_state_ref':'episode:10','control_steps':10,'official_actions':1})
    assert not oracle_covers_feedback(state,'cube',{'robot_state_ref':'another:10','control_steps':10,'official_actions':1})
    assert not oracle_covers_feedback(state,'cube',{'robot_state_ref':'episode:30','control_steps':30,'official_actions':3})
    state['objects']['cube']['T_W_object'][0][3]+=1
    assert not measured_pose_supported(state,'cube')
    initial=tmp_path/'initial';other=GroundTruthWorld(env,initial,[1,2,2],mode='initial_geometry')
    other.sync({'robot_state_ref':'episode:20','control_steps':20})
    assert not measured_pose_supported(World(initial).state(),'cube')


def test_articulation_links_keep_independent_geometry_and_live_pose(tmp_path):
    pytest.importorskip('pxr')
    from pxr import Usd, UsdGeom, UsdPhysics, Gf
    stage=Usd.Stage.CreateInMemory();UsdGeom.SetStageMetersPerUnit(stage,1.)
    root=UsdGeom.Xform.Define(stage,'/holder')
    root.AddTranslateOp().Set(Gf.Vec3d(1,2,3))
    base=UsdGeom.Xform.Define(stage,'/holder/base');UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
    cube=UsdGeom.Cube.Define(stage,'/holder/base/box');cube.GetSizeAttr().Set(.2)
    # Nesting checks that a child's surface is not duplicated into its parent.
    lid=UsdGeom.Xform.Define(stage,'/holder/base/lid');UsdPhysics.RigidBodyAPI.Apply(lid.GetPrim())
    move=lid.AddTranslateOp();move.Set(Gf.Vec3d(0,0,.3))
    spin=lid.AddRotateYOp();spin.Set(0)
    geom=UsdGeom.Cube.Define(stage,'/holder/base/lid/box');geom.GetSizeAttr().Set(1)
    geom.AddScaleOp().Set(Gf.Vec3f(.3,.2,.02))
    obj=SimpleNamespace(prim_path='/holder',category_name='egg_holder')
    sm=SimpleNamespace(stage=stage,_rigid_and_dynamic_objects=[{}],_geometry_objects=[{}],
                       _articulation_objects=[{'holder':obj}])
    gt=GroundTruthWorld(SimpleNamespace(scene_manager=sm),tmp_path,[1,2,3])
    world=World(tmp_path);world.verify_binding()
    assert set(world.state()['objects'])=={'holder_base','holder_base_lid'}
    gt.sync({'robot_state_ref':'egg:0','control_steps':0})
    np.testing.assert_allclose(np.diff(world.bounds('holder_base')['bounds_W_m'],axis=0),[[.2,.2,.2]],atol=1e-6)
    np.testing.assert_allclose(np.mean(world.bounds('holder_base_lid')['bounds_W_m'],axis=0),[0,0,.3],atol=1e-6)
    move.Set(Gf.Vec3d(.1,0,.4));spin.Set(90)
    gt.sync({'robot_state_ref':'egg:10','control_steps':10})
    np.testing.assert_allclose(np.diff(world.bounds('holder_base_lid')['bounds_W_m'],axis=0),[[.02,.2,.3]],atol=1e-6)
    np.testing.assert_allclose(np.mean(world.bounds('holder_base_lid')['bounds_W_m'],axis=0),[.1,0,.4],atol=1e-6)
    np.testing.assert_allclose(np.mean(world.bounds('holder_base')['bounds_W_m'],axis=0),[0,0,0],atol=1e-6)
    assert len(world.state()['history'])==2
    meta=json.loads((tmp_path/'objects/holder_base_lid.json').read_text())
    assert meta['parent_articulation']=='holder' and meta['role']=='articulation_link'
