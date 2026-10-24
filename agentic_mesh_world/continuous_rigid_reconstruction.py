"""Fit continuous bottle/bowl/plate bodies to observed RGB-D, not source fragments.

VLM chooses the whole-object shape family and original-image masks. Geometry
fits metric position and a radial profile. This replaces fragment repair as the
main output for these simple upright rigid objects. Source evidence is archived.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from .mesh_world import MeshCodeWorld, sha256, pose_matrix, matrix_pose
from .quality_reconstruction import points, write


def fit_profile(xyz, family, *, bins=32):
    if family not in {'bottle','bowl','plate'}:raise ValueError('Unsupported whole-object family')
    if len(xyz)<100:raise ValueError('Insufficient measured object points')
    z0,z1=np.quantile(xyz[:,2],[.002,.998]);height=z1-z0
    if height<=0:raise ValueError('Degenerate object height')
    rim=xyz[xyz[:,2]>z1-.12*height] if family!='bottle' else xyz
    initial=(np.quantile(rim[:,:2],.01,axis=0)+np.quantile(rim[:,:2],.99,axis=0))/2
    # Fit a common axis to height-conditioned circles. Radius varies with z, so
    # a bottle shoulder cannot drag the axis away from the body center.
    band=np.clip(((xyz[:,2]-z0)/height*bins).astype(int),0,bins-1)
    initial_r=np.array([np.median(np.linalg.norm(xyz[band==i,:2]-initial,axis=1)) if np.any(band==i) else .01 for i in range(bins)])
    probe=np.linspace(0,len(xyz)-1,min(30000,len(xyz)),dtype=int)
    def residual(values):return np.linalg.norm(xyz[probe,:2]-values[:2],axis=1)-values[2:][band[probe]]
    result=least_squares(residual,np.r_[initial,initial_r],loss='soft_l1',f_scale=.002,max_nfev=100)
    center=result.x[:2];radius=np.linalg.norm(xyz[:,:2]-center,axis=1)
    outer=[];inner=[]
    for i in range(bins):
        selected=band==i
        if selected.sum()<10:continue
        z=float(np.median(xyz[selected,2])-z0)
        outer.append([float(np.quantile(radius[selected],.9)),z])
        inner.append([float(np.quantile(radius[selected],.1)),z])
    if len(outer)<8:raise ValueError('Insufficient profile coverage')
    outer=np.asarray(outer);inner=np.asarray(inner)
    # Remove isolated radial bumps, while keeping broad shoulders and the rim.
    from scipy.ndimage import median_filter
    outer[:,0]=median_filter(outer[:,0],size=3,mode='nearest')
    outer[0,1]=0;outer[-1,1]=height
    assumptions=['upright axisymmetric whole-object hypothesis from VLM; asymmetric details are not represented']
    if family=='bottle':
        profile=np.vstack([[0,0],outer,[0,height]])
    else:
        rim_radius=float(outer[-1,0]);wall=float(np.clip(np.median(outer[:,0]-inner[:,0]),.001,.004))
        bottom_points=xyz[radius<.3*rim_radius]
        floor=float(np.quantile(bottom_points[:,2],.05)-z0) if len(bottom_points)>20 else .08*height
        floor=float(np.clip(floor,.001,.35*height))
        inside=outer.copy();inside[:,0]=np.maximum(inside[:,0]-wall,0);inside=inside[inside[:,1]>=floor]
        profile=np.vstack([[0,0],outer,inside[::-1],[0,floor],[0,0]])
        assumptions.append('unseen wall/base continuity completed; thickness estimated from radial bands')
    return profile,np.r_[center,z0],assumptions


def lathe_mesh(profile,segments=96):
    import trimesh
    mesh=trimesh.creation.revolve(profile,sections=segments)
    mesh.remove_unreferenced_vertices()
    if mesh.volume<0:mesh.invert()
    return mesh


def run(config):
    import open3d as o3d
    cfg=json.loads(Path(config).read_text());cap=Path(cfg['capture']);data=json.loads(cap.read_text())
    source=MeshCodeWorld.load(cfg['source_scene']);out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
    views={v['view_id']:v for v in data['views']};nodes=[];reports=[];used_views={}
    for spec in cfg['objects']:
        oid=spec['object_id'];ann_path=Path(spec['annotations']);packet=json.loads(ann_path.read_text())
        clouds=[];colors=[];bindings=[]
        bounds=source.mesh(oid).bounds;selected=set(spec.get('view_ids',[]))
        for row in packet['views']:
            v=views[row['view_id']]
            if selected and v['view_id'] not in selected:continue
            if oid not in row['masks']:continue
            if v['role']!='fit':raise ValueError('Completion fitting cannot consume holdout')
            if not v.get('surface_enabled',True) or not v.get('identity_enabled',True):continue
            rgb=cap.parent/v['rgb_uri'];depth_file=cap.parent/v['depth_uri'];ref=row['masks'][oid];mask_file=ann_path.parent/ref['uri']
            if sha256(rgb)!=row['rgb_sha256'] or sha256(mask_file)!=ref['sha256']:raise ValueError('Original mask/RGB changed')
            if v.get('depth_sha256') and sha256(depth_file)!=v['depth_sha256']:raise ValueError('Original depth changed')
            depth=np.load(depth_file);mask=np.asarray(Image.open(mask_file).convert('L'))>127
            xyz=points(depth,np.asarray(v['K']),pose_matrix(v['T_W_C']))
            # Source bounds only select a local fitting workspace. Masks decide
            # identity; bounds do not provide hidden/native object geometry.
            margin=float(spec.get('workspace_margin_m',.008))
            keep=mask&np.isfinite(depth)&(depth>0)&np.all(xyz>=bounds[0]-margin,axis=-1)&np.all(xyz<=bounds[1]+margin,axis=-1)
            iy,ix=np.nonzero(keep);index=np.linspace(0,len(ix)-1,min(12000,len(ix)),dtype=int) if len(ix) else []
            clouds.append(xyz[iy[index],ix[index]]);colors.append(np.asarray(Image.open(rgb).convert('RGB'))[iy[index],ix[index]])
            bindings.append({'view_id':v['view_id'],'rgb_sha256':sha256(rgb),'depth_sha256':sha256(depth_file),'mask_sha256':sha256(mask_file)})
            used_views[v['view_id']]=v
        cloud=np.concatenate(clouds);color=np.concatenate(colors)
        profile,origin,assumptions=fit_profile(cloud,spec['family']);mesh=lathe_mesh(profile)
        _,nearest=cKDTree(cloud-origin).query(mesh.vertices)
        mesh.visual.vertex_colors=color[nearest];file=out/f'{oid}.glb';mesh.export(file)
        world_mesh=mesh.copy();world_mesh.apply_translation(origin)
        ray=o3d.t.geometry.RaycastingScene();ray.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(world_mesh.vertices),o3d.utility.Vector3iVector(world_mesh.faces))))
        errors=ray.compute_distance(o3d.core.Tensor(cloud.astype('float32'))).numpy()
        pose=np.eye(4);pose[:3,3]=origin
        nodes.append({'object_id':oid,'parent_id':None,'pose_parent_object':matrix_pose(pose),'state':'ESTIMATED',
                      'asset':{'mesh_ref':file.name,'sha256':sha256(file),'T_object_mesh':np.eye(4).tolist(),'up_axis':'Z',
                               'provenance':{'source':'continuous measured-profile reconstruction hypothesis','generated_asset_used':True,
                                             'original_scene_sha256':sha256(cfg['source_scene']),'observed_inputs':bindings,'assumptions':assumptions}}})
        reports.append({'object_id':oid,'family':spec['family'],'faces':len(mesh.faces),'components':len(mesh.split(only_watertight=False)),
                        'watertight':bool(mesh.is_watertight),'bounds_W_m':world_mesh.bounds.tolist(),
                        'measured_points':len(cloud),'observed_point_to_candidate_p95_m':float(np.quantile(errors,.95)),
                        'observed_point_to_candidate_median_m':float(np.median(errors)),'assumptions':assumptions,
                        'independent_acceptance':False})
        write(out/f'{oid}_profile.json',{'profile_radius_z_m':profile.tolist(),'origin_W_m':origin.tolist(),'shape_family':spec['family']})
        (out/f'{oid}_model.py').write_text(
            'import json\nfrom pathlib import Path\nimport numpy as np\n'
            'from agentic_mesh_world.continuous_rigid_reconstruction import lathe_mesh\n'
            f'PROFILE=Path(__file__).with_name({(oid+"_profile.json")!r})\n'
            'def make_mesh(world_frame=True):\n'
            '    data=json.loads(PROFILE.read_text())\n'
            '    mesh=lathe_mesh(np.array(data["profile_radius_z_m"]))\n'
            '    if world_frame: mesh.apply_translation(data["origin_W_m"])\n'
            '    return mesh\n')
        print('continuous body',oid,flush=True)
    scene={'schema_version':'mesh-code-v1','frame':'robot_base','units':'m','world_id':source.source['world_id']+'_continuous',
           'world_version':source.source['world_version']+1,'nodes':nodes,'quality':{'reconstruction_acceptance':'PENDING','metric_contact_ready':False},
           'observations':[]}
    evidence=out/'evidence';evidence.mkdir()
    import shutil
    for vid,v in used_views.items():
        target=evidence/f'{vid}.png';shutil.copy2(cap.parent/v['rgb_uri'],target)
        scene['observations'].append({**v,'rgb_ref':str(target.relative_to(out))})
    write(out/'scene.json',scene);write(out/'config.json',cfg)
    report={'status':'CONTINUOUS_RIGID_CANDIDATES','objects':reports,'holdout_used':False,'code_world_schema_fixed':False,
            'scope':'whole-object hypotheses fitted to visible RGB-D; raw evidence retained; new independent validation required'}
    write(out/'report.json',report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    print(json.dumps(run(p.parse_args().config)))
