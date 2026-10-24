"""Frozen-output comparison. Held-out views enter ONLY this final evaluator."""
import argparse,json
from pathlib import Path
import numpy as np
import trimesh,open3d as o
from scipy.ndimage import binary_dilation
from .evaluate import evaluate,ray_scene
from agentic_mesh_world.mesh_world import MeshCodeWorld,pose_matrix

def main(config):
 cfg=json.loads(Path(config).read_text());out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False);ev=Path(cfg['evidence']);summary=json.loads((ev/'model/evidence.json').read_text());plane=np.array(summary['support']['plane_W']);ecfg=json.loads((ev/'config.json').read_text());ids=[x['object_id'] for x in ecfg['objects']];bounds=np.array([summary['objects'][i]['visible_bounds_W_m'] for i in ids]);low=bounds[:,0].min(0)-.025;high=bounds[:,1].max(0)+.025;cap=Path(cfg['evaluation_capture']);views=json.loads(cap.read_text())['views'];result={}
 for method in cfg['methods']:
  name=method['name'];source=Path(method['source']);dest=out/name;dest.mkdir();meshes={}
  if method['kind']=='legacy':
   world=MeshCodeWorld.load(source/'scene.json');meshes={i:world.mesh(i) for i in ids};robot=world.mesh('known_robot')
  else:
   inv={x['object_id']:x for x in json.loads((source/'object_inventory.json').read_text())};meshes={i:trimesh.load(source/inv[i]['mesh_ref'],force='scene',process=False).to_geometry() for i in ids if i in inv};robot=trimesh.load(source/inv['known_robot']['mesh_ref'],force='scene',process=False).to_geometry()
  for oid,m in meshes.items():m.export(dest/(oid+'.glb'))
  (dest/'render.json').write_text(json.dumps({'objects':list(meshes)}));fit=evaluate(dest,ev,ids,save_overlays=False)
  props=ray_scene(trimesh.util.concatenate(list(meshes.values())));rr=ray_scene(robot);held=[]
  for v in views:
   if v['role']!='holdout':continue
   d=np.load(cap.parent/v['depth_uri'])[::2,::2];h,w=d.shape;yy,xx=np.mgrid[:h,:w];t=pose_matrix(v['T_W_C']);directions=np.c_[xx.ravel()*2,yy.ravel()*2,np.ones(h*w)]@np.linalg.inv(v['K']).T@t[:3,:3].T;xyz=(t[:3,3]+directions*d.reshape(-1,1)).reshape(h,w,3);rays=np.c_[np.broadcast_to(t[:3,3],directions.shape),directions].reshape(h,w,6).astype('float32');r=o.core.Tensor(rays);hit=props.cast_rays(r,nthreads=2)['t_hit'].numpy();rh=rr.cast_rays(r,nthreads=2)['t_hit'].numpy()
   valid=np.isfinite(d)&(d>0);inroi=np.all((xyz>=low)&(xyz<=high),axis=2);nonrobot=~(np.isfinite(rh)&(rh<=d+.006));observed=valid&inroi&nonrobot&(xyz@plane[:3]+plane[3]>.004)
   visible=np.isfinite(hit)&valid&nonrobot&(hit<=d+.006);delta=abs(hit-d);match=observed&np.isfinite(hit)&(delta<=.006);miss=observed&(~np.isfinite(hit)|(hit>d+.006));excess=visible&~binary_dilation(observed,iterations=1)&(hit<d-.006)
   residual=delta[observed&np.isfinite(hit)];held.append({'view_id':v['view_id'],'observed_prop_pixels':int(observed.sum()),'matched_fraction':float(match.sum()/max(1,observed.sum())),'missing_fraction':float(miss.sum()/max(1,observed.sum())),'false_occupied_fraction':float(excess.sum()/max(1,visible.sum())),'finite_hit_depth_p95_m':float(np.quantile(residual,.95)) if len(residual) else None})
  result[name]={'fit':fit,'heldout':held,'summary':{'fit_object_mean_p95_mm':float(np.mean([r['point_surface_p95_m'] for r in fit['objects']]))*1000,'fit_object_mean_missing_fraction':float(np.mean([r['mean_missing_fraction'] for r in fit['objects']])),'fit_object_mean_false_occupied_fraction':float(np.mean([r['mean_false_occupied_fraction'] for r in fit['objects']])),'holdout_mean_matched_fraction':float(np.mean([r['matched_fraction'] for r in held])),'holdout_mean_missing_fraction':float(np.mean([r['missing_fraction'] for r in held])),'holdout_mean_false_occupied_fraction':float(np.mean([r['false_occupied_fraction'] for r in held]))}}
  (dest/'metrics.json').write_text(json.dumps(result[name],indent=2)+'\n');print(name,result[name]['summary'],flush=True)
 report={'methods':result,'heldout_roi':'observed RGB-D within common FIT-derived workspace bounds, >4mm above measured table; recorded public robot occlusion excluded; no semantic holdout masks','tolerance_m':.006,'scope':'single-sample geometric comparison; this does not certify hidden geometry or task success'};(out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');main(p.parse_args().config)
