"""Observed surface identity, independent of geometry generation and object names.

AABB is deliberately absent from the labeling rule. Unknown, occluded and
unannotated pixels never become negative identity evidence.
"""
from pathlib import Path
import json
import numpy as np
from PIL import Image
from .mesh_world import pose_matrix, sha256


def load_annotations(path, capture, capture_root):
    path=Path(path); packet=json.loads(path.read_text()); views={v['view_id']:v for v in capture['views']}
    records=[]; seen=set()
    for record in packet['views']:
        vid=record['view_id']
        if vid in seen: raise ValueError('Duplicate annotation view')
        seen.add(vid); v=views[vid]
        if record['rgb_sha256']!=v['rgb_sha256'] or sha256(Path(capture_root)/v['rgb_uri'])!=v['rgb_sha256']:
            raise ValueError('Instance annotation RGB mismatch')
        masks={}
        for oid,ref in record['masks'].items():
            if capture.get('object_bounds_from_masked_depth') and oid not in capture['object_bounds_from_masked_depth']:raise ValueError('Unknown instance identity')
            if not isinstance(ref,dict) or 'sha256' not in ref: raise ValueError('Mask requires content binding')
            file=path.parent/ref['uri']
            if sha256(file)!=ref['sha256']: raise ValueError('Instance mask changed')
            mask=np.asarray(Image.open(file).convert('L'))>127
            if list(mask.shape[::-1])!=v['image_size']: raise ValueError('Mask resolution mismatch')
            masks[oid]=mask
        records.append((v,masks))
    return records


def reviewed_mask_completeness(capture_path,annotations_path,review_ref,*,role='fit'):
    """Bind visible-mask completeness to the exact RGB/mask/capture revision."""
    path=Path(review_ref['path'])
    if not path.is_file() or sha256(path)!=review_ref['sha256']:raise ValueError('Mask completeness review changed')
    review=json.loads(path.read_text());cap=json.loads(Path(capture_path).read_text())
    ann=json.loads(Path(annotations_path).read_text());views={v['view_id']:v for v in cap['views']}
    masks={v['view_id']:v for v in ann['views']}
    if review.get('capture_sha256')!=sha256(capture_path) or review.get('annotations_sha256')!=sha256(annotations_path):
        raise ValueError('Mask completeness review belongs to another capture/annotation revision')
    if not review.get('reviewer'):raise ValueError('Mask completeness reviewer required')
    complete={};seen=set()
    for row in review['masks']:
        vid,oid=row['view_id'],row['object_id'];key=(vid,oid)
        if key in seen:raise ValueError('Duplicate mask completeness review')
        seen.add(key);view=views[vid];ref=masks[vid]['masks'][oid]
        if view['role']!=role or row['rgb_sha256']!=view['rgb_sha256'] or row['mask_sha256']!=ref['sha256']:
            raise ValueError('Mask completeness evidence does not bind this '+role+' image and mask')
        if not isinstance(row.get('reason'),str) or not row['reason'].strip():raise ValueError('Mask review needs a reason')
        if (row.get('identity_verified') is True and row.get('all_visible_surfaces_covered') is True
                and row.get('foreign_pixels_present') is False):
            complete.setdefault(vid,[]).append(oid)
    return complete


def reviewed_negative_masks(capture_path,annotations_path,review_ref):
    return reviewed_mask_completeness(capture_path,annotations_path,review_ref,role='fit')


def project_samples(samples, view, depth, tolerance_m):
    t=pose_matrix(view['T_W_C']); k=np.asarray(view['K'])
    # Flatten four samples per triangle for two large BLAS products instead of
    # millions of separate 4x3 products. Projection and rounding rules stay the
    # same. The deployment benchmark checks exact pixels/visibility on source
    # RGB-D samples; identity tests check serial/threaded vote parity.
    p=((samples.reshape(-1,3)-t[:3,3])@t[:3,:3]).reshape(samples.shape)
    z=p[...,2];uv=(p.reshape(-1,3)@k.T).reshape(samples.shape)
    xy=np.rint(uv[...,:2]/np.maximum(z[...,None],1e-12)).astype(np.int64)
    ok=(z>0)&(xy[...,0]>=0)&(xy[...,0]<depth.shape[1])&(xy[...,1]>=0)&(xy[...,1]<depth.shape[0])
    x=np.clip(xy[...,0],0,depth.shape[1]-1);y=np.clip(xy[...,1],0,depth.shape[0]-1)
    d=depth[y,x];ok &= np.isfinite(d)&(d>0)&(abs(d-z)<=tolerance_m)
    return x,y,ok


def assign(mesh, records, root, object_ids, *, tolerance_m=.003,min_views=2,
           min_score=.65,min_margin=.2,min_baseline_m=.02,chunk=50000,propagation_distance_m=.01,
           enable_negative_evidence=False,boundary_margin_px=2,max_negative_fraction=.35,
           negative_policy='reviewed_complete', negative_complete_masks=None, workers=1,
           surface_bridge_distance_m=0,progress_hook=None):
    """Votes from center and inset corners. Mixed faces remain UNKNOWN.

Keep global source-face indices through submesh/export; no component deletion.
Camera duplicates cannot inflate independent support counts.

Optional negative evidence does not treat all unannotated pixels as background.
The object must have a nonempty mask in that view. A depth-consistent sample
outside that mask, farther than boundary_margin_px from it and outside mask
overlaps, then contributes negative identity evidence. Occlusion, absent masks
and silhouette uncertainty remain unknown. Positive votes and their original
minimum-view, identity-fraction and identity-margin gates remain unchanged.

This is conditional on mask identity/completeness: a confidently wrong or badly
undersegmented mask can supply false negative evidence. Pixel margins are not a
metric uncertainty model, and camera-position separation does not establish
statistical independence. This gate is therefore an optional A/B repair tool,
not semantic certification or permission to erase geometry.
"""
    if not object_ids or len(set(object_ids))!=len(object_ids): raise ValueError('Unique objects required')
    if min_views<2 or not 0<min_score<=1 or not 0<=min_margin<=1:raise ValueError('Invalid identity limits')
    if not isinstance(enable_negative_evidence,(bool,np.bool_)):raise ValueError('Negative evidence enable flag must be boolean')
    if negative_policy not in {'reviewed_complete','legacy_outside_mask'}:raise ValueError('Unknown negative evidence policy')
    complete = {vid:set(oids) for vid,oids in (negative_complete_masks or {}).items()}
    if any(set(oids)-set(object_ids) for oids in complete.values()):raise ValueError('Unknown identity in complete-mask review')
    if not np.isfinite(boundary_margin_px) or boundary_margin_px<0 or not 0<=max_negative_fraction<=1:raise ValueError('Invalid negative evidence limits')
    if not isinstance(workers,int) or not 1<=workers<=8:raise ValueError('Identity workers must be an integer in [1,8]')
    if not np.isfinite(surface_bridge_distance_m) or not 0<=surface_bridge_distance_m<=.001:
        raise ValueError('Measured sheet identity bridge must be at most 1 mm')
    n=len(mesh.faces); scores=np.zeros((n,len(object_ids)),np.float32)
    count=np.zeros_like(scores,dtype=np.uint16);seen=np.zeros(n,np.uint16);directions=[];used=[]
    negative_scores=np.zeros_like(scores);negative_count=np.zeros_like(count)
    core_count=np.zeros_like(count)
    prepared=[]
    eligible=[(i,v,m) for i,(v,m) in enumerate(records) if v['role']=='fit' and v.get('surface_enabled',True) and v.get('identity_enabled',True)]
    # A new higher-resolution observation at an existing camera position must
    # be usable, without counting the same position twice as independent support.
    selected=[];centers=[]
    def resolution(row):
        _,view,masks=row
        return np.prod(view.get('image_size',next(iter(masks.values())).shape[::-1] if masks else [0,0]))
    for i,view,masks in sorted(eligible,key=resolution,reverse=True):
        camera=np.asarray(view['T_W_C']['translation_m'])
        if any(np.linalg.norm(camera-q)<min_baseline_m for q in centers):continue
        centers.append(camera);selected.append((i,view,masks))
    for _,view,masks in sorted(selected,key=lambda row:row[0]):
        camera=np.asarray(view['T_W_C']['translation_m'])
        directions.append(camera);used.append(view['view_id'])
        depth=np.load(Path(root)/view['depth_uri']); overlap=np.zeros(depth.shape,np.uint16)
        for mask in masks.values():overlap+=mask
        outside={};core_masks={}
        from scipy.ndimage import distance_transform_edt
        for oid in object_ids:
            if oid in masks and masks[oid].any():
                # Diagnostic only: count support safely inside the same mask.
                # Original positive scores and ownership gates are unchanged.
                core_masks[oid]=distance_transform_edt(masks[oid])>boundary_margin_px
                if enable_negative_evidence and (negative_policy=='legacy_outside_mask' or oid in complete.get(view['view_id'],set())):
                    outside[oid]=distance_transform_edt(~masks[oid])>boundary_margin_px
        prepared.append((view,masks,depth,overlap,outside,core_masks))
    # Workers own disjoint source-face ranges. The camera accumulation order is
    # unchanged for every face, preserving votes and thresholds bit-for-bit.
    # Cache source triangles before entering threads: trimesh cache writes are
    # not part of the worker path.
    triangles=mesh.triangles
    def vote_chunk(start):
        end=min(n,start+chunk);tri=triangles[start:end];center=tri.mean(1)
        samples=np.concatenate([center[:,None],.8*tri+.2*center[:,None]],axis=1)
        for view,masks,depth,overlap,outside,core_masks in prepared:
            x,y,visible=project_samples(samples,view,depth,tolerance_m)
            usable=visible&(overlap[y,x]==1)
            seen[start:end]+=visible.any(1)
            for j,oid in enumerate(object_ids):
                if oid not in masks:continue
                positive=(usable&masks[oid][y,x]).sum(1)/4.
                scores[start:end,j]+=positive
                count[start:end,j]+=positive>=.5
                if oid in core_masks:
                    core_count[start:end,j]+=(usable&core_masks[oid][y,x]).sum(1)>=2
                if oid in outside:
                    reliable_negative=visible&(overlap[y,x]<=1)&(~masks[oid][y,x])&outside[oid][y,x]
                    negative=reliable_negative.sum(1)/4.
                    negative_scores[start:end,j]+=negative
                    negative_count[start:end,j]+=negative>=.5
    if workers==1:
        for number,start in enumerate(range(0,n,chunk),1):
            vote_chunk(start)
            if progress_hook:progress_hook(min(number*chunk,n),n)
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for number,_ in enumerate(pool.map(vote_chunk,range(0,n,chunk)),1):
                if progress_hook:progress_hook(min(number*chunk,n),n)
    total=scores.sum(1); fractions=scores/np.maximum(total[:,None],1e-12)
    best=fractions.argmax(1); top=fractions[np.arange(n),best]
    second=np.partition(fractions,-2,axis=1)[:,-2] if len(object_ids)>1 else np.zeros(n)
    support=count[np.arange(n),best]
    valid=(support>=min_views)&(top>=min_score)&((top-second)>=min_margin)
    denominator=scores+negative_scores
    object_confidence=scores/np.maximum(denominator,1e-12)
    negative_fraction=negative_scores/np.maximum(denominator,1e-12)
    rejected=np.zeros_like(scores,dtype=bool)
    if enable_negative_evidence:
        rejected=(negative_scores>0)&((negative_fraction>max_negative_fraction)|(object_confidence<min_score))
        valid &= ~rejected[np.arange(n),best]
    labels=np.where(valid,best,-1).astype(np.int32)
    direct=labels.copy()
    if propagation_distance_m>0:
        if progress_hook:progress_hook(n,n,'identity_surface_propagation')
        labels=propagate_identity(mesh,labels,scores,fractions,propagation_distance_m,
            negative_rejected=rejected,bridge_distance_m=surface_bridge_distance_m)
    return labels,{'workers':workers,'direct_owner':direct,
                   'surface_bridge_distance_m':float(surface_bridge_distance_m),
                   'propagated_faces':int(np.sum((direct<0)&(labels>=0))),'scores':scores,'support_views':count,'visible_views':seen,'confidence':top,
                   'core_support_views':core_count,
                   'core_support_policy':{'boundary_margin_px':float(boundary_margin_px),'required_samples':2,
                       'sample_count':4,'scope':'diagnostic eroded-mask support; not a replacement identity vote'},
                   'negative_scores':negative_scores,'negative_views':negative_count,'object_confidence':object_confidence,
                   'negative_fraction':negative_fraction,'negative_rejected':rejected,
                   'negative_evidence':{'enabled':bool(enable_negative_evidence),'boundary_margin_px':float(boundary_margin_px),
                       'policy':negative_policy,'reviewed_complete_masks':{k:sorted(v) for k,v in complete.items()},
                       'max_negative_fraction':float(max_negative_fraction),'denominator':'positive plus reliable negative sample votes',
                       'negative_view_count_rule':'at least two of the four samples in a distinct camera view',
                       'limitation':'wrong or incomplete instance masks can still create false negative identity evidence'},
                   'margin':top-second,'fit_view_ids':used,'object_ids':object_ids,
                   'camera_support_policy':'prefer highest original pixel resolution; one vote per separated camera position; all observations remain available for audit',
                   'policy':'depth-visible multi-view mask votes; uncertain/conflicting surfaces retained'}


def propagate_identity(mesh,labels,scores,fractions,max_distance_m,*,negative_rejected=None,bridge_distance_m=0):
    """Transfer identity over a short observed continuous surface, not empty space.

Every propagated face still needs its own positive image evidence. Multi-view
seeds lend identity support through compatible local adjacency; they do not
invent geometry or turn an entirely unseen surface into an observed one.
Explicit reliable-negative rejection is a per-face/per-object exclusion: a
rejected face cannot receive or relay that object's identity from nearby seeds.
"""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    if negative_rejected is None:negative_rejected=np.zeros_like(scores,dtype=bool)
    negative_rejected=np.asarray(negative_rejected,dtype=bool)
    if negative_rejected.shape!=scores.shape:raise ValueError('Negative rejection must bind every face and object')
    result=labels.copy();edges=mesh.face_adjacency
    if not len(edges) and not bridge_distance_m:return result
    centers=mesh.triangles_center
    length=np.linalg.norm(centers[edges[:,0]]-centers[edges[:,1]],axis=1)
    compatible=np.sum(mesh.face_normals[edges[:,0]]*mesh.face_normals[edges[:,1]],axis=1)>.5
    for j in range(scores.shape[1]):
        allowed=((labels<0)|(labels==j))&(scores[:,j]>0)&(fractions[:,j]>=.85)&~negative_rejected[:,j]
        use=compatible&allowed[edges[:,0]]&allowed[edges[:,1]]
        e=edges[use];seeds=np.flatnonzero(labels==j)
        lengths=length[use]
        if bridge_distance_m:
            # Separate sensor sheets can sample the same local surface without
            # sharing vertex indices. Transfer identity only over sub-mm steps
            # between measured, similarly oriented faces that each have their
            # own unambiguous positive RGB evidence. Never add a triangle or
            # count this inference as another camera observation.
            from scipy.spatial import cKDTree
            ids=np.flatnonzero(allowed)
            if len(ids)>1:
                distance,nearest=cKDTree(centers[ids]).query(centers[ids],k=min(4,len(ids)))
                a=np.broadcast_to(ids[:,None],nearest.shape).ravel();b=ids[nearest.ravel()]
                gap=distance.ravel();keep=(a<b)&(gap<=bridge_distance_m)
                keep&=(mesh.face_normals[a]*mesh.face_normals[b]).sum(1)>.5
                e=np.concatenate([e,np.column_stack([a[keep],b[keep]])])
                lengths=np.concatenate([lengths,gap[keep]])
        if not len(e) or not len(seeds):continue
        graph=coo_matrix((np.maximum(lengths,1e-12),(e[:,0],e[:,1])),shape=(len(labels),len(labels))).tocsr()
        distances=dijkstra(graph,directed=False,indices=seeds,min_only=True,limit=max_distance_m)
        take=(labels<0)&allowed&np.isfinite(distances)
        result[take]=j
    return result


def ray_scene(meshes):
    import open3d as o
    scene=o.t.geometry.RaycastingScene();ids=[]
    for mesh in meshes:
        if not len(mesh.faces):ids.append(None);continue
        tri=o.geometry.TriangleMesh(o.utility.Vector3dVector(mesh.vertices),o.utility.Vector3iVector(mesh.faces))
        ids.append(scene.add_triangles(o.t.geometry.TriangleMesh.from_legacy(tri)))
    return scene,ids


def audit(meshes, object_ids, records, root, *, role='holdout',stride=2,tolerance_m=.005,
          min_pixels=30,min_views=2,min_recall=.98,min_precision=.98,min_iou=.95,
          complete_masks=None):
    """Raycast identity, not the union of surfaces inside an object's box.

The whole scene supplies occlusion, while the hit geometry ID supplies ownership.
Masks are visible-instance annotations, never amodal guesses. A sparse view is
non-evaluable, not a pass; each object must still have enough evaluable views.
"""
    import open3d as o
    scene,ids=ray_scene(meshes); rows={oid:[] for oid in object_ids};positions={}
    for view,masks in records:
        if view['role']!=role or not view.get('surface_enabled',True) or not view.get('identity_enabled',True):continue
        position=np.asarray(view['T_W_C']['translation_m'])
        positions[view['view_id']]=position
        d=np.load(Path(root)/view['depth_uri'])[::stride,::stride];k=np.asarray(view['K']);t=pose_matrix(view['T_W_C'])
        yy,xx=np.indices(d.shape);directions=np.stack([(xx*stride-k[0,2])/k[0,0],(yy*stride-k[1,2])/k[1,1],np.ones_like(d)],-1)@t[:3,:3].T
        rays=np.concatenate([np.broadcast_to(t[:3,3],directions.shape),directions],-1).astype('float32')
        hit=scene.cast_rays(o.core.Tensor(rays),nthreads=8);z=hit['t_hit'].numpy();gid=hit['geometry_ids'].numpy()
        valid=np.isfinite(d)&(d>0);occluded=np.isfinite(z)&(z>d+tolerance_m)
        # Mask intersections have ambiguous reference identity: do not count them.
        overlap=np.zeros(d.shape,np.uint16)
        for m in masks.values():overlap+=m[::stride,::stride]
        for j,oid in enumerate(object_ids):
            if oid not in masks:continue
            gt=masks[oid][::stride,::stride]&valid&(overlap==1)
            pred=(gid==ids[j])&valid&~occluded&(overlap<=1) if ids[j] is not None else np.zeros_like(gt)
            pixels=int(gt.sum())
            if pixels<min_pixels:
                rows[oid].append({'view_id':view['view_id'],'status':'INSUFFICIENT_VISIBLE_PIXELS','target_pixels':pixels});continue
            correct=gt&pred&(abs(z-d)<=tolerance_m);union=gt|pred
            residual=abs(z[gt&pred]-d[gt&pred])
            complete=complete_masks is None or oid in complete_masks.get(view['view_id'],[])
            precision=float(correct.sum()/max(1,pred.sum()));iou=float((gt&pred).sum()/max(1,union.sum()))
            rows[oid].append({'view_id':view['view_id'],'status':'EVALUATED' if complete else 'EVALUATED_PARTIAL_MASK','target_pixels':pixels,
                'union_recall':float((gt&np.isfinite(z)&(abs(z-d)<=tolerance_m)).sum()/pixels),
                'recall':float(correct.sum()/pixels),'precision':precision if complete else None,
                'iou':iou if complete else None,
                'raw_mask_precision':precision,'raw_mask_iou':iou,
                'mask_scope':'complete visible instance' if complete else 'positive visible fragment; outside identity unknown',
                'p95_error_m':float(np.quantile(residual,.95)) if len(residual) else None,
                'wrong_owner_pixels':int((gt&(gid!=ids[j])&np.isfinite(z)&(abs(z-d)<=tolerance_m)).sum())})
    failures=[]
    for oid,values in rows.items():
        checked=[r for r in values if r['status']=='EVALUATED']
        independent=[]
        for row in checked:
            position=positions[row['view_id']]
            if all(np.linalg.norm(position-p)>=.02 for p in independent):independent.append(position)
        if len(independent)<min_views:failures.append({'object_id':oid,'kind':'NEED_OBSERVATION','reason':'insufficient distinct complete-mask validation views'})
        for r in [x for x in values if x['status'].startswith('EVALUATED')]:
            if (r['recall']<min_recall or (r['precision'] is not None and r['precision']<min_precision)
                    or (r['iou'] is not None and r['iou']<min_iou) or r['p95_error_m'] is None or r['p95_error_m']>tolerance_m):
                reason=('visible geometry also absent from complete scene' if r['union_recall']<min_recall else
                        'measured surface exists but belongs to another node' if r['recall']<min_recall else
                        'extra predicted surface, depth disagreement or annotation silhouette mismatch')
                failures.append({'object_id':oid,'view_id':r['view_id'],'kind':'INSTANCE_REPAIR_REQUIRED',
                    'action':'resegment' if r['union_recall']>=min_recall else 'surface_repair','reason':reason,
                    'measurements':{k:r[k] for k in ('recall','precision','iou','union_recall','p95_error_m','wrong_owner_pixels')}})
    return {'status':'INSTANCE_CHECK_PASSED' if not failures else 'INSTANCE_REVIEW_FAILED','role':role,'objects':rows,'failures':failures,
            'scope':'visible annotated surfaces only; no backside or contact certification',
            'mask_completeness_policy':'bound complete masks required for precision/IoU and minimum validation views' if complete_masks is not None else 'legacy caller assumes complete masks',
            'limits':{'min_recall':min_recall,'min_precision':min_precision,'min_iou':min_iou,'tolerance_m':tolerance_m,'min_complete_views':min_views}}


def choose_views(capture, count=8, role='fit', exclude=()):
    eligible=[v for v in capture['views'] if v['role']==role and v.get('surface_enabled',True) and v['view_id'] not in exclude]
    chosen=[]
    while eligible and len(chosen)<count:
        def separation(v):
            p=np.asarray(v['T_W_C']['translation_m'])
            return min((np.linalg.norm(p-np.asarray(c['T_W_C']['translation_m'])) for c in chosen),default=1.)
        v=max(eligible,key=separation);chosen.append(v);eligible.remove(v)
    return chosen
