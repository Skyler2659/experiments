"""Validate RGB-D pose proposals against measured surfaces before world publication.

This checks visible surface consistency, not hidden shape or unique pose observability.
"""
import copy,json
import numpy as np
from .completion import path_in,digest,state_ref


def validate_candidate_assembly(world, assembly):
    """Check executable geometry inputs; free parameters remain unmeasured hypotheses."""
    import hashlib
    import re
    from .world import rigid
    if (not isinstance(assembly,dict) or not isinstance(assembly.get('geometry_revision'),str)
            or not assembly['geometry_revision'] or not isinstance(assembly.get('parameters'),dict)
            or not isinstance(assembly.get('parts'),dict) or not assembly['parts']):
        raise ValueError('Candidate assembly requires geometry_revision, parameters and nonempty parts')
    for part_id,part in assembly['parts'].items():
        if (not isinstance(part_id,str) or not part_id or not isinstance(part,dict)
                or not isinstance(part.get('local_mesh'),str)
                or not isinstance(part.get('sha256'),str)
                or not re.fullmatch(r'[0-9a-f]{64}',part['sha256'])
                or 'T_object_part' not in part):
            raise ValueError('Candidate part requires local_mesh, sha256 and T_object_part')
        path=path_in(world.root,part['local_mesh'])
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=part['sha256']:
            raise ValueError('Candidate part mesh is missing or does not match its sha256')
        rigid(part['T_object_part'])
    # parameters are intentionally not used for the surface or observability tests.


def integrate_rgbd(world,capture,proposals,*,max_p90_m=.008,min_points=32,diagnose_observability=False):
    from .world import atomic,read,rigid
    import fcntl
    from codeworld_robot.grasp import _cloud,_capture
    world.sync_robot_feedback();path,cap=_capture(world.root,capture)
    capture_ref=state_ref(cap.get('robot_state',{}))
    if not capture_ref or capture_ref!=state_ref(world._robot_state()):raise ValueError('Stale or unversioned RGB-D capture')
    if (not proposals or isinstance(max_p90_m,bool) or not isinstance(max_p90_m,(int,float))
            or not np.isfinite(max_p90_m) or max_p90_m<=0
            or isinstance(min_points,bool) or not isinstance(min_points,int) or min_points<32
            or not isinstance(diagnose_observability,bool)):
        raise ValueError('Invalid measurement thresholds')
    from .surface_index import object_surface,pose_observability,geometry_key
    s=world.state();reports={};updated=copy.deepcopy(s);surfaces={}
    for oid,spec in proposals.items():
        T=rigid(spec['T_W_object']);candidate=copy.deepcopy(s);candidate['objects'][oid]['T_W_object']=T.tolist()
        if 'assembly' in spec:
            validate_candidate_assembly(world,spec['assembly'])
            candidate['objects'][oid]['assembly']=copy.deepcopy(spec['assembly'])
        surface=object_surface(world,oid,candidate);surfaces[oid]=surface;stats=[];clouds=[];camera_origins=[]
        query_start=surface.queries
        masks=spec.get('masks',{})
        if not isinstance(masks,dict) or set(masks)-{v['view_id'] for v in cap['views']}:
            raise ValueError('Masks must refer to views in the supplied capture')
        for view in cap['views']:
            mask=masks.get(view['view_id'])
            if mask is None:continue
            # Reuse the calibrated mask validation, then extract a bounded sample.
            cloud=_cloud(world.root,path,view,mask,min_points)
            if cloud is None:
                stats.append({'view_id':view['view_id'],'status':'INCONCLUSIVE',
                              'reason':'Too few valid target depth pixels'})
                continue
            maskpath=path_in(world.root,mask)
            if maskpath.suffix=='.npy':mask=np.load(maskpath,allow_pickle=False).astype(bool)
            else:
                from PIL import Image
                mask=np.asarray(Image.open(maskpath)).astype(bool)
            depth_path=Path_or_relative(world.root,path.parent,view['depth_uri']);depth=np.load(depth_path,allow_pickle=False)
            y,x=np.where(mask & np.isfinite(depth) & (depth>0));indices=np.linspace(0,len(x)-1,min(len(x),256)).astype(int);x=x[indices];y=y[indices]
            points=(np.column_stack([x,y,np.ones_like(x)])@np.linalg.inv(np.asarray(view['K'])).T)*depth[y,x,None]
            camera=np.asarray(view['T_W_C']);points=points@camera[:3,:3].T+camera[:3,3]
            clouds.append(points);camera_origins.append(camera[:3,3])
            dist=surface.distances(points,T)
            error=float(np.quantile(dist,.9))
            stats.append({'view_id':view['view_id'],'points':len(points),
                          'status':'SURFACE_CONSISTENT' if np.isfinite(error) and error<=max_p90_m else 'INCONCLUSIVE',
                          'p90_surface_error_m':error})
        accepted=bool(stats) and all(v['status']=='SURFACE_CONSISTENT' for v in stats)
        # Common updates publish a checked estimate plus its limitations. A unique
        # six-DoF explanation is unnecessary for reading geometry or planning.
        observability={'status':'NOT_EVALUATED','translation_observed':None,'orientation_observed':None,
            'translation_observed_axes':[None,None,None], 'orientation_observed_axes':[None,None,None],
            'orientation_geometry_equivalent_axes':[None,None,None],
            'translation_alternatives':[], 'orientation_alternatives':[]}
        if accepted and diagnose_observability:
            observability=pose_observability(surface,T,np.concatenate(clouds),camera_origins,max_p90_m)
        fit_error=max((v['p90_surface_error_m'] for v in stats if 'p90_surface_error_m' in v),default=None)
        reports[oid]={**observability,'observability':observability,
            'status':'SURFACE_CONSISTENT' if accepted else 'INCONCLUSIVE','views':stats,
            'max_p90_m':max_p90_m,'p90_surface_error_m':fit_error,
            'within_default_surface_tolerance':bool(fit_error is not None and fit_error<=.008),
            'position_evidence':'visible_surface_fit' if accepted else 'insufficient',
            'orientation_evidence':'prior_with_visible_surface_fit' if accepted else 'insufficient',
            'surface_queries':surface.queries-query_start,
            'limitations':'Surface residual validates consistency of this estimate with the supplied masked depth. It does not establish object identity, unique pose, hidden geometry, or task success. A larger acceptance tolerance permits a coarser estimate, not stronger completion evidence.'}
        row=updated['objects'][oid]
        if accepted:
            row.update(T_W_object=T.tolist(),basis='observation_estimate',evidence={'references':[str(path)]})
            if 'assembly' in spec:row['assembly']=copy.deepcopy(spec['assembly'])
            row.setdefault('review',{})['pose']={'status':'observed','revision':s['revision']+1,'evidence':[str(path)]}
            row['sensor_validation']={'pose_digest':digest(row['T_W_object']),'assembly_digest':digest(row['assembly']) if row.get('assembly') else None,'capture':str(path),'report':reports[oid],
                'orientation_observed':observability['orientation_observed'],'translation_observed':observability['translation_observed'],
                'translation_observed_axes':observability['translation_observed_axes'], 'orientation_observed_axes':observability['orientation_observed_axes'],
                'orientation_geometry_equivalent_axes':observability['orientation_geometry_equivalent_axes'],
                'position_evidence':'visible_surface_fit','orientation_evidence':'prior_with_visible_surface_fit',
                'surface_error_m':fit_error,'acceptance_tolerance_m':max_p90_m}
        else:
            row.pop('sensor_validation',None);row.setdefault('review',{})['pose']={'status':'pending','evidence':[str(path)],'reason':'RGB-D proposal did not fit measured surface'}
    with (world.root/'state/update.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        current=world.state()
        # Read-only feedback and unrelated object updates do not change the fit's
        # inputs. Merge into the current snapshot instead of discarding useful
        # work or replacing unrelated concurrent updates with the old snapshot.
        for oid,spec in proposals.items():
            if current['objects'].get(oid)!=s['objects'].get(oid):
                raise ValueError('Target object changed during RGB-D fitting: '+oid)
            candidate=copy.deepcopy(current)
            candidate['objects'][oid]['T_W_object']=rigid(spec['T_W_object']).tolist()
            if 'assembly' in spec:candidate['objects'][oid]['assembly']=copy.deepcopy(spec['assembly'])
            if geometry_key(world,oid,candidate)!=surfaces[oid].geometry_key:
                raise ValueError('Target geometry changed during RGB-D fitting: '+oid)
        if capture_ref!=state_ref(world._robot_state()):raise ValueError('Robot changed during RGB-D fitting; acquire a current capture')
        for oid in proposals:
            current['objects'][oid]=updated['objects'][oid]
            if reports[oid]['status']=='SURFACE_CONSISTENT':
                current['objects'][oid]['review']['pose']['revision']=current['revision']+1
        current['revision']+=1;current.setdefault('history',[]).append({'revision':current['revision'],'kind':'validated_rgbd_update','capture':str(path),'reports':reports});atomic(world.state_path,current)
        updated=current
    return {'state_revision':updated['revision'],'objects':reports}


def Path_or_relative(root,base,value):
    from pathlib import Path
    p=Path(value);p=p if p.is_absolute() else base/p
    return path_in(root,p)
