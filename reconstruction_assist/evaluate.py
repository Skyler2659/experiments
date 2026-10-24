"""Occlusion-aware FIT geometry feedback for an executed RCWM component (Linux)."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion,binary_dilation
import trimesh,open3d as o
from agentic_mesh_world.mesh_world import pose_matrix

def ray_scene(mesh):
    r=o.t.geometry.RaycastingScene();r.add_triangles(o.t.geometry.TriangleMesh.from_legacy(o.geometry.TriangleMesh(o.utility.Vector3dVector(mesh.vertices),o.utility.Vector3iVector(mesh.faces))));return r

def compare_depth(hit,depth,mask,tolerance=.006):
    valid=np.isfinite(depth)&(depth>0);inside=binary_erosion(mask,iterations=1)&valid
    visible=np.isfinite(hit)&valid&(hit<=depth+tolerance)
    excess=visible&~binary_dilation(mask,iterations=1)&(hit<depth-tolerance)
    missing=inside&(~np.isfinite(hit)|(hit>depth+tolerance))
    delta=np.abs(hit-depth);matched=inside&np.isfinite(hit)&(delta<=tolerance)
    return {'false_occupied_fraction':float(excess.sum()/max(1,visible.sum())),
            'missing_observed_fraction':float(missing.sum()/max(1,inside.sum())),
            'matched_observed_fraction':float(matched.sum()/max(1,inside.sum())),
            'observed_pixels':int(inside.sum())},excess,missing,matched

def evaluate(export,evidence,scope=None,save_overlays=True):
    out=Path(export);ev=Path(evidence);cfg=json.loads((ev/'config.json').read_text());cap=Path(cfg['capture']);data=json.loads(cap.read_text());ann=Path(cfg['annotations']);packet=json.loads(ann.read_text());by={v['view_id']:v for v in data['views']};summary=json.loads((ev/'model/evidence.json').read_text());plane=np.array(summary['support']['plane_W']);rows=[]
    inventory=json.loads((out/'render.json').read_text())['objects'];expected=set(scope or [])
    for oid in inventory:
        if oid not in summary['objects']:continue
        mesh=trimesh.load(out/(oid+'.glb'),force='scene',process=False).to_geometry();ray=ray_scene(mesh)
        xyz=np.load(ev/'samples'/(oid+'.npz'))['xyz_W_m'];xyz=xyz[np.linspace(0,len(xyz)-1,min(15000,len(xyz)),dtype=int)]
        dist=ray.compute_distance(o.core.Tensor(xyz.astype('float32')),nthreads=2).numpy();views=[]
        for row in packet['views']:
            if oid not in row['masks']:continue
            v=by[row['view_id']]
            if v['role']!='fit' or not v.get('surface_enabled',True):raise ValueError('Feedback cannot read holdout')
            d=np.load(cap.parent/v['depth_uri'])[::4,::4];h,w=d.shape;yy,xx=np.mgrid[:h,:w];cam=pose_matrix(v['T_W_C']);directions=np.c_[xx.ravel()*4,yy.ravel()*4,np.ones(h*w)]@np.linalg.inv(v['K']).T@cam[:3,:3].T
            rays=np.c_[np.broadcast_to(cam[:3,3],directions.shape),directions].reshape(h,w,6).astype('float32')
            hit=ray.cast_rays(o.core.Tensor(rays),nthreads=2)['t_hit'].numpy();mask=np.asarray(Image.open(ann.parent/row['masks'][oid]['uri']).convert('L'))[::4,::4]>127
            metrics,excess,missing,matched=compare_depth(hit,d,mask)
            if metrics['observed_pixels']<8:continue
            file=oid+'_'+v['view_id']+'_overlay.png'
            if save_overlays:
                image=np.array(Image.open(cap.parent/v['rgb_uri']).convert('RGB'))[::4,::4].copy();image[excess]=[255,40,40];image[missing]=[40,80,255];image[matched]=(image[matched]*.5+[10,110,35]).astype('uint8');Image.fromarray(image).resize((512,512),Image.Resampling.NEAREST).save(out/file)
            views.append({'view_id':v['view_id'],**metrics,'overlay':file})
        p95=float(np.quantile(dist,.95));missing=float(np.mean([v['missing_observed_fraction'] for v in views])) if views else 1.;excess=float(np.mean([v['false_occupied_fraction'] for v in views])) if views else 1.
        score=p95/.01+missing+excess
        rows.append({'object_id':oid,'point_surface_p95_m':p95,'point_surface_median_m':float(np.median(dist)),'mean_missing_fraction':missing,'mean_false_occupied_fraction':excess,'score':score,'min_support_plane_gap_m':float(np.min(mesh.vertices@plane[:3]+plane[3])),'views':views,'needs_revision':p95>.006 or missing>.10 or excess>.05})
    support_review=[]
    if scope and 'tabletop' in scope:
        from .support_review import analyze
        support_review=analyze(out,inventory)
        by_support={r['object_id']:r for r in support_review}
        for row in rows:
            support=by_support.get(row['object_id'])
            if support:
                row['support_contact']=support
                if support['floating']:
                    row['needs_revision']=True
                    gap=support['minimum_vertical_support_gap_m']
                    row['score']+=gap/.01 if gap is not None else 10.
    missing_ids=sorted(expected-set(inventory));bad=[r['object_id'] for r in rows if r['needs_revision']]
    return {'status':'FIT_REVIEW_REQUIRED' if bad or missing_ids else 'FIT_CHECKS_PASSED','objects':rows,'missing_object_ids':missing_ids,'revision_objects':bad,'score':float(np.mean([r['score'] for r in rows]))+len(missing_ids)*10 if rows else 100.,'support_review':support_review,'holdout_used':False,'independent_acceptance':False,'overlay_legend':'red=false occupied in observed free space; blue=missing observed surface; green=matched; foreground occlusion ignored'}
