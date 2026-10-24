"""Build model-readable FIT evidence from raw RGB-D and cached image masks (Linux)."""
import argparse,json,re
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial import ConvexHull
from agentic_mesh_world.static_scene_assets import collect
from agentic_mesh_world.masked_sample_consensus import run as consensus
from agentic_mesh_world.static_scene_constraints import support_samples
from agentic_mesh_world.static_surface_constraints import fit_plane
from agentic_mesh_world.static_asset_geometry import box_part
from agentic_mesh_world.quality_reconstruction import points
from agentic_mesh_world.mesh_world import pose_matrix

def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def run(config):
    cfg=json.loads(Path(config).read_text());out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
    cap=Path(cfg['capture']);ann=Path(cfg['annotations']);data=json.loads(cap.read_text());packet=json.loads(ann.read_text())
    by={v['view_id']:v for v in data['views']};specs=cfg['objects']+cfg.get('regions',[])
    for row in packet['views']:
        if by[row['view_id']]['role']!='fit' or not by[row['view_id']].get('surface_enabled',True):raise ValueError('Only surface-enabled FIT observations allowed')
    collection={'capture':str(cap),'annotations':str(ann),'output':str(out/'samples_raw')}
    write(out/'collect.json',collection);collect(out/'collect.json')
    consensus(cap,ann,out/'samples_raw',out/'samples')
    stats=json.loads((out/'samples/samples.json').read_text());bounds=np.array([s['bounds_005_995_m'] for s in stats['objects'].values()])
    plane_points,plane_views=support_samples(cap,ann,[bounds[:,0].min(0),bounds[:,1].max(0)])
    plane,plane_report=fit_plane(plane_points)
    table=[]
    for row in packet['views']:
        v=by[row['view_id']];d=np.load(cap.parent/v['depth_uri']);xyz=points(d,np.asarray(v['K']),pose_matrix(v['T_W_C']))
        exclude=np.zeros(d.shape,bool)
        for ref in row['masks'].values():exclude|=np.asarray(Image.open(ann.parent/ref['uri']))>127
        keep=np.isfinite(xyz).all(2)&(d>0)&~exclude&(abs(xyz@plane[:3]+plane[3])<.0015)
        table.append(xyz[keep][::20])
    table=np.concatenate(table);q=np.quantile(table[:,:2],[.002,.998],axis=0);inside=np.all((table[:,:2]>=q[0])&(table[:,:2]<=q[1]),axis=1);xy=table[inside,:2];hull=xy[ConvexHull(xy).vertices]
    model=out/'model';model.mkdir();rows={}
    for spec in specs:
        oid=spec['object_id'];z=np.load(out/'samples'/(oid+'.npz'));xyz=z['xyz_W_m'];rgb=z['rgb'];part,_=box_part(xyz)
        candidates=[]
        for row in packet['views']:
            if oid not in row['masks']:continue
            m=np.asarray(Image.open(ann.parent/row['masks'][oid]['uri']))>127
            yy,xx=np.nonzero(m)
            if not len(xx):continue
            candidates.append((len(xx),row,m,[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]))
        candidates.sort(key=lambda r:r[0],reverse=True);picked=[]
        for item in candidates:
            if len(picked)==3:break
            current=re.search(r'_(\d+)$',item[1]['view_id'])
            first=re.search(r'_(\d+)$',picked[0][1]['view_id']) if picked else None
            if not picked or current is None or first is None or abs(int(current[1])-int(first[1]))>=3:picked.append(item)
        crops=[]
        for _,r,m,box in picked:
            v=by[r['view_id']];im=Image.open(cap.parent/v['rgb_uri']).convert('RGB');x0,y0,x1,y1=box;pad=12;box=[max(0,x0-pad),max(0,y0-pad),min(im.width,x1+pad),min(im.height,y1+pad)]
            im=im.crop(box);f=model/(oid+'_'+v['view_id']+'.png');im.save(f)
            crops.append({'view_id':v['view_id'],'image':f.name,'crop_xyxy':box,'crop_pixel_to_original':'u_original=u_crop+x0, v_original=v_crop+y0; no resize','K_original':v['K'],'T_W_C':v['T_W_C']})
        rows[oid]={'description':spec['description'],'parent_id':spec.get('parent_id'),'visible_points':len(xyz),'visible_bounds_W_m':np.quantile(xyz,[.005,.995],axis=0).tolist(),'upright_box_initialization':part,'median_observed_rgb':np.median(rgb,axis=0).tolist(),'surface_samples_W_m':xyz[np.linspace(0,len(xyz)-1,min(192,len(xyz)),dtype=int)].tolist(),'fit_view_ids':[b['view_id'] for b in stats['objects'][oid]['bindings']],'crops':crops,'limitations':['box is a conditional visible-envelope initialization, not the true shape or certified size','unobserved surfaces and masked-out ownership remain unknown']}
    summary={'frame':'robot_base_W','units':'m','pixel_centers':'integer; use K directly, no +0.5 when unprojecting','objects':rows,'support':{'plane_W':plane.tolist(),'observed_footprint_xy_W_m':hull.tolist(),'plane_fit':plane_report,'thickness':'unobserved; any slab thickness is a hypothesis'},'provenance':{'raw_rgbd_reprocessed':True,'cached_sam_masks':str(ann),'historical_meshes_used':False,'holdout_used':False}}
    write(model/'evidence.json',summary);write(out/'config.json',cfg);print(json.dumps({'output':str(out),'objects':len(rows)}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');run(p.parse_args().config)
