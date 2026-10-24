"""Expand measured tabletop coverage using saved fit RGB-D, not task assets."""
import copy,json,shutil
from pathlib import Path
import numpy as np
from .mesh_world import MeshCodeWorld,pose_matrix,matrix_pose,sha256,generated_scene_code


def expand(scene_path,capture_path,*,scale=1.6,grid_m=.004,plane_tolerance_m=.003,observations=None):
    import trimesh
    from PIL import Image
    from .quality_reconstruction import points,inside
    scene_path=Path(scene_path);cap=Path(capture_path);root=scene_path.parent
    if not np.isfinite([scale,grid_m,plane_tolerance_m]).all() or scale<=1 or grid_m<=0 or plane_tolerance_m<=0:raise ValueError('Require positive resolution and scale greater than one')
    world=MeshCodeWorld.load(scene_path);scene=world.source
    if any(n['object_id']=='measured_tabletop' for n in scene['nodes']):raise ValueError('Table already expanded; do not compound its scale')
    node=world.node('unassigned_observed_context');context=world.mesh(node['object_id'])
    horizontal=abs(context.face_normals[:,2])>.95
    heights=context.triangles_center[:,2];z=heights[horizontal]
    if not len(z):raise ValueError('No measured horizontal support; request table observation')
    bins=np.arange(z.min(),z.max()+.0021,.002);hist,edges=np.histogram(z,bins=bins,weights=context.area_faces[horizontal])
    plane_z=edges[hist.argmax()]+.001
    table_faces=horizontal&(abs(heights-plane_z)<=plane_tolerance_m)
    original_points=context.triangles[table_faces].reshape(-1,3)
    old_bounds=np.array([original_points.min(0),original_points.max(0)])
    center=old_bounds[:,:2].mean(0);half=np.ptp(old_bounds[:,:2],axis=0)*scale/2
    bounds=np.array([[*(center-half),plane_z-plane_tolerance_m],[*(center+half),plane_z+plane_tolerance_m]])
    data=copy.deepcopy(observations if observations is not None else json.loads(cap.read_text()))
    # A measured point cloud does not make a flattened Delaunay patch measured.
    # Triangulate only adjacent sensor pixels and preserve their actual heights.
    # Keep all old context: horizontal proximity alone cannot classify or erase
    # an object bottom. Instance review can disambiguate it in a later version.
    from .projective_surface import reconstruct
    table,backend=reconstruct(data,cap.parent,bounds,stride=2,max_edge_m=grid_m*4,min_facing=.02,consistent_color=False)
    horizontal_ids=np.flatnonzero(abs(table.face_normals[:,2])>.95)
    if not len(horizontal_ids):raise ValueError('Insufficient observed horizontal support')
    table=table.submesh([horizontal_ids],append=True,repair=False)
    used=[r['view_id'] for r in backend['views']]
    table_path=root/'assets/measured_tabletop.glb';table.export(table_path)
    receipt={'source':'fit RGBD observations from current capture','requested_scale':scale,'old_bounds_W_m':old_bounds.tolist(),
        'new_bounds_W_m':table.bounds.tolist(),'grid_m':grid_m,'plane_z_W_m':plane_z,'fit_views':used,
        'construction':'adjacent original FIT depth pixels; original vertex heights; no Delaunay, flattening or gap bridging',
        'surface_backend':backend,'context_source_faces_removed':0,
        'plane_used_for_region_selection_only':True,'foreground_identity_review_required':True,
        'task_assets_used':False,'full_real_table_boundary_verified':False}
    scene['nodes'].append({'object_id':'measured_tabletop','parent_id':None,'role':'support_surface','state':'ESTIMATED',
        'pose_parent_object':matrix_pose(np.eye(4)),'asset':{'mesh_ref':'assets/measured_tabletop.glb','sha256':sha256(table_path),'up_axis':'Z','T_object_mesh':np.eye(4).tolist(),'provenance':receipt}})
    scene['support_surface']=receipt
    scene_path.write_text(json.dumps(scene,indent=2));return receipt


def augment(scene,capture,output,scale=1.6):
    out=Path(output)
    if out.exists():raise ValueError('Choose a new world version')
    shutil.copytree(Path(scene).parent,out)
    before=json.loads((out/'scene.json').read_text());before['world_version']=before.get('world_version',1)+1
    (out/'scene.json').write_text(json.dumps(before,indent=2))
    receipt=expand(out/'scene.json',capture,scale=scale)
    world=MeshCodeWorld.load(out/'scene.json');world.export_glb(out/'scene.glb');world.render(world.source['canonical_camera'],out/'expanded_table.png')
    (out/'scene.py').write_text(generated_scene_code());(out/'table_expansion.json').write_text(json.dumps(receipt,indent=2));return receipt

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--capture',required=True);p.add_argument('--output',required=True);p.add_argument('--scale',type=float,default=1.6);a=p.parse_args()
    print(json.dumps(augment(a.scene,a.capture,a.output,a.scale),indent=2))
