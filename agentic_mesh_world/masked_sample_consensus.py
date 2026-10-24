"""Cross-view support for FIT object samples, without inventing empty space."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
from .mesh_world import pose_matrix,sha256
from .quality_reconstruction import write


def support_counts(xyz, observations, tolerance_m=.006):
    votes=np.zeros(len(xyz),np.uint16)
    for view,depth,mask in observations:
        t=pose_matrix(view['T_W_C']);p=(xyz-t[:3,3])@t[:3,:3]
        uv=p@np.asarray(view['K']).T;uv=uv[:,:2]/np.maximum(p[:,2,None],1e-12)
        h,w=depth.shape;valid=(p[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
        x=np.clip(np.rint(uv[:,0]),0,w-1).astype(int);y=np.clip(np.rint(uv[:,1]),0,h-1).astype(int)
        d=depth[y,x]
        # A foreground occluder cannot provide support for the surface behind it.
        votes+=valid&np.isfinite(d)&(d>0)&(abs(d-p[:,2])<=tolerance_m)&mask[y,x]
    return votes


def run(capture, annotations, samples, output, min_views=2):
    cap=Path(capture);ann=Path(annotations);src=Path(samples);out=Path(output)
    data=json.loads(cap.read_text());packet=json.loads(ann.read_text());stats=json.loads((src/'samples.json').read_text())
    if packet['capture_sha256']!=sha256(cap) or stats['capture_sha256']!=sha256(cap):raise ValueError('Sample capture changed')
    if stats['annotations_sha256']!=sha256(ann):raise ValueError('Sample masks changed')
    views={v['view_id']:v for v in data['views']};by={};reports=[]
    for row in packet['views']:
        v=views[row['view_id']]
        if v['role']!='fit':raise ValueError('Consensus may use FIT only')
        depth_file=cap.parent/v['depth_uri']
        if sha256(depth_file)!=v['depth_sha256']:raise ValueError('Depth changed')
        depth=np.load(depth_file)
        for oid,ref in row['masks'].items():
            f=ann.parent/ref['uri']
            if sha256(f)!=ref['sha256']:raise ValueError('Mask changed')
            by.setdefault(oid,[]).append((v,depth,np.asarray(Image.open(f).convert('L'))>127))
    out.mkdir(exist_ok=False)
    for oid,item in stats['objects'].items():
        z=np.load(src/(oid+'.npz'));xyz=z['xyz_W_m'];rgb=z['rgb']
        votes=support_counts(xyz,by.get(oid,[]));keep=votes>=min_views
        if int(keep.sum())<100:raise ValueError('Insufficient cross-view object evidence '+oid)
        np.savez_compressed(out/(oid+'.npz'),xyz_W_m=xyz[keep],rgb=rgb[keep])
        # Rejected samples remain explicit unknown ownership, not free-space evidence.
        np.savez_compressed(out/(oid+'_unsupported.npz'),xyz_W_m=xyz[~keep],support_count=votes[~keep])
        reports.append({'object_id':oid,'input_points':len(xyz),'retained_points':int(keep.sum()),'removed_points':int((~keep).sum()),'support_histogram':np.bincount(votes.astype(int)).tolist()})
        item.update(points=int(keep.sum()),bounds_005_995_m=np.quantile(xyz[keep],[.005,.995],axis=0).tolist(),
                    sample_policy=f'at least {min_views} depth-consistent same-ID FIT masks; unsupported ownership retained separately',cross_view_verified=min_views>=2)
        print('sample consensus',oid,int(keep.sum()),'/',len(xyz),flush=True)
    write(out/'samples.json',stats)
    cfg=json.loads((src/'config.json').read_text());cfg['output']=str(out);write(out/'config.json',cfg)
    write(out/'consensus_report.json',{'objects':reports,'min_views':min_views,'depth_tolerance_m':.006,'holdout_used':False,'unsupported_is_empty':False,
        'source_samples':str(src),'source_samples_sha256':sha256(src/'samples.json'),'capture_sha256':sha256(cap),'annotations_sha256':sha256(ann)})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--capture',required=True);p.add_argument('--annotations',required=True);p.add_argument('--samples',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.capture,a.annotations,a.samples,a.output)
