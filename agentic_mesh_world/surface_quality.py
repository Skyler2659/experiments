"""Evidence-aware mesh detail QA. Small does not mean spurious.

Disconnected measured sheets may be legitimate. Cleaning requires free-space
contradiction; visual/continuity issues must remain explicit after cleaning.
"""
from pathlib import Path
import numpy as np
from .mesh_world import pose_matrix


def topology(mesh):
    import open3d as o
    import trimesh
    # GLB texture/normal seams duplicate vertices at identical coordinates.
    # Count geometric connectivity on an analysis copy; otherwise a textured
    # continuous basket can appear to have hundreds of thousands of fragments.
    # Exact welding adds no bridge, moves no source point and changes no face ID.
    vertices,inverse=np.unique(np.asarray(mesh.vertices),axis=0,return_inverse=True)
    analysis=trimesh.Trimesh(vertices=vertices,faces=inverse[mesh.faces],process=False)
    om=o.geometry.TriangleMesh(o.utility.Vector3dVector(analysis.vertices),o.utility.Vector3iVector(analysis.faces))
    labels,sizes,areas=om.cluster_connected_triangles()
    labels=np.asarray(labels);sizes=np.asarray(sizes);areas=np.asarray(areas)
    counts=np.bincount(analysis.edges_unique_inverse)
    boundary=analysis.edges_unique[counts==1]
    length=float(np.linalg.norm(analysis.vertices[boundary[:,0]]-analysis.vertices[boundary[:,1]],axis=1).sum())
    report={'faces':len(mesh.faces),'vertices':len(mesh.vertices),'components':len(sizes),
        'largest_component_area_fraction':float(areas.max()/max(areas.sum(),1e-20)),
        'small_components_le_10_faces':int(np.sum(sizes<=10)),
        'small_component_area_fraction':float(areas[sizes<=10].sum()/max(areas.sum(),1e-20)),
        'boundary_edges':len(boundary),'boundary_length_m':length,
        'watertight':bool(analysis.is_watertight),
        'analysis_vertices':len(vertices),'duplicate_position_vertices':len(mesh.vertices)-len(vertices),
        'connectivity_analysis':'exact coincident vertex welding on analysis copy; source faces unchanged',
        'interpretation':'open boundaries may be true thin sheets or missing surfaces; not automatically holes to fill'}
    return report,labels,sizes,areas


def evidence(mesh,data,root,tolerance_m=.001,chunk=100000):
    n=len(mesh.faces);support=np.zeros(n,np.uint16);contradictions=np.zeros(n,np.uint16)
    centers=mesh.triangles_center;normals=mesh.face_normals
    for v in data['views']:
        if v['role']!='fit' or not v.get('surface_enabled',True):continue
        depth=np.load(Path(root)/v['depth_uri']);t=pose_matrix(v['T_W_C']);k=np.asarray(v['K'])
        for start in range(0,n,chunk):
            stop=min(n,start+chunk);p=(centers[start:stop]-t[:3,3])@t[:3,:3];z=p[:,2]
            uv=p@k.T;xy=np.rint(uv[:,:2]/np.maximum(z[:,None],1e-12)).astype(np.int64)
            ray=t[:3,3]-centers[start:stop];ray/=np.maximum(np.linalg.norm(ray,axis=1,keepdims=True),1e-12)
            angle=abs(np.sum(normals[start:stop]*ray,axis=1))>.15
            valid=angle&(z>0)&(xy[:,0]>=0)&(xy[:,0]<depth.shape[1])&(xy[:,1]>=0)&(xy[:,1]<depth.shape[0])
            idx=np.flatnonzero(valid);d=depth[xy[idx,1],xy[idx,0]];ok=np.isfinite(d)&(d>0);idx=idx[ok];d=d[ok]
            support[start+idx]+=abs(z[idx]-d)<=tolerance_m
            contradictions[start+idx]+=z[idx]<d-tolerance_m
    return support,contradictions


def clean(mesh,data,root, *, tolerance_m=.001,max_component_area_m2=2e-5,min_contradictions=2):
    before,labels,sizes,areas=topology(mesh)
    candidates=(areas<=max_component_area_m2)[labels]
    support=np.zeros(len(mesh.faces),np.uint16);bad=support.copy()
    if candidates.any():
        candidate_mesh=mesh.submesh([np.flatnonzero(candidates)],append=True,repair=False)
        sp,bc=evidence(candidate_mesh,data,root,tolerance_m)
        support[candidates]=sp;bad[candidates]=bc
    face_areas=mesh.area_faces
    unsupported=(support<2)&(bad>=min_contradictions)
    contradicted_area=np.bincount(labels,weights=face_areas*unsupported,minlength=len(sizes))
    # Preserve long thin parts and all strongly supported patches, irrespective of
    # triangle count. Only wholly suspect components may be removed.
    selected=(areas<=max_component_area_m2)&(contradicted_area/np.maximum(areas,1e-20)>.9)
    drop=selected[labels]
    if np.all(drop):raise ValueError('Cleaning would remove all geometry')
    cleaned=mesh.submesh([np.flatnonzero(~drop)],append=True,repair=False) if drop.any() else mesh.copy()
    after,_,_,_=topology(cleaned) if drop.any() else (before,None,None,None)
    receipt={'before':before,'after':after,'removed_faces':int(drop.sum()),'removed_components':int(selected.sum()),
        'removed_area_m2':float(face_areas[drop].sum()),'policy':'small component AND >90% area lacks two supporting views AND contradicts at least two fit depth views',
        'tolerance_m':tolerance_m,'positive_support_histogram':np.bincount(support[candidates].astype(int)).tolist(),
        'support_histogram_scope':'only small-component candidate faces',
        'unknown_small_components_preserved':int(np.sum((sizes<=10)&~selected)),
        'missing_surface_completion':False}
    return cleaned,receipt


def detail_gate(report, *, max_small_components=100,max_small_area_fraction=.002):
    failures=[]
    if report['small_components_le_10_faces']>max_small_components and report['small_component_area_fraction']>max_small_area_fraction:
        failures.append('excess disconnected speckles; acquire denser observations or use a thin-surface backend')
    return {'status':'DETAIL_REPAIR_REQUIRED' if failures else 'DETAIL_NUMERIC_CHECK_PASSED',
        'failures':failures,'visual_review_required':True,
        'limits':{'max_small_components':max_small_components,'max_small_area_fraction':max_small_area_fraction},
        'scope':'speckle screening, not a completeness or closed-solid certificate'}


def deduplicate_small(mesh, *, tolerance_m=.00002, normal_cosine=.99):
    """Remove tiny duplicate patches only when a larger surface covers them.

    This is independent of evidence cleaning: real thin parts without an
    equivalent retained surface are never discarded for being small.
    """
    import open3d as o
    before,labels,sizes,areas=topology(mesh)
    candidate=(sizes<=10)[labels];drop=np.zeros(len(mesh.faces),bool)
    if candidate.any() and (~candidate).any():
        target=mesh.submesh([np.flatnonzero(~candidate)],append=True,repair=False)
        om=o.geometry.TriangleMesh(o.utility.Vector3dVector(target.vertices),o.utility.Vector3iVector(target.faces))
        scene=o.t.geometry.RaycastingScene();scene.add_triangles(o.t.geometry.TriangleMesh.from_legacy(om))
        idx=np.flatnonzero(candidate);tri=mesh.triangles[idx]
        samples=np.concatenate([tri,tri.mean(1)[:,None,:]],axis=1)
        closest=scene.compute_closest_points(o.core.Tensor(samples.astype('float32')))
        distance=np.linalg.norm(closest['points'].numpy()-samples,axis=-1)
        normals=closest['primitive_normals'].numpy()
        aligned=abs(np.sum(normals*mesh.face_normals[idx,None,:],axis=-1))>=normal_cosine
        equivalent=(distance<=tolerance_m).all(1)&aligned.all(1)
        goodarea=np.bincount(labels[idx],weights=mesh.area_faces[idx]*equivalent,minlength=len(sizes))
        duplicate=(sizes<=10)&(goodarea/np.maximum(areas,1e-20)>.999)
        drop=duplicate[labels]
    cleaned=mesh.submesh([np.flatnonzero(~drop)],append=True,repair=False) if drop.any() else mesh.copy()
    return cleaned,{'before':before,'after':topology(cleaned)[0] if drop.any() else before,
        'removed_faces':int(drop.sum()),'tolerance_m':tolerance_m,'normal_cosine':normal_cosine,
        'policy':'all vertices and face centres of tiny component have a retained, aligned larger surface within tolerance',
        'thin_distinct_parts_preserved':True}


def render_detail_views(world,oid,folder,size=640,smooth=False):
    """Fixed multi-angle inspection, including elevated views of internal sheets."""
    import os
    os.environ.setdefault('PYOPENGL_PLATFORM','egl')
    import pyrender
    from PIL import Image,ImageDraw
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    mesh=world.mesh(oid);center=mesh.bounds.mean(0);scale=max(mesh.extents);renderer=pyrender.OffscreenRenderer(size,size)
    images=[]
    try:
        scene=pyrender.Scene(bg_color=[240,242,245,255],ambient_light=[.6,.6,.6])
        for piece in world.mesh_pieces(oid):scene.add(pyrender.Mesh.from_trimesh(piece,smooth=smooth))
        camera_node=scene.add(pyrender.PerspectiveCamera(yfov=np.deg2rad(40)))
        light_node=scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=2))
        for i in range(8):
            az=2*np.pi*(i%4)/4+.3;el=np.deg2rad(30 if i<4 else 65)
            pos=center+scale*2.1*np.array([np.cos(az)*np.cos(el),np.sin(az)*np.cos(el),np.sin(el)])
            z=(pos-center)/np.linalg.norm(pos-center);x=np.cross([0,0,1],z);x/=np.linalg.norm(x);pose=np.eye(4);pose[:3,:3]=np.column_stack([x,np.cross(z,x),z]);pose[:3,3]=pos
            scene.set_pose(camera_node,pose=pose);scene.set_pose(light_node,pose=pose)
            rgb,_=renderer.render(scene,flags=pyrender.RenderFlags.SKIP_CULL_FACES)
            im=Image.fromarray(rgb);im.save(folder/f'view_{i:02d}.png');images.append(im)
    finally:renderer.delete()
    panel=Image.new('RGB',(size*4,(size+24)*2),'white');draw=ImageDraw.Draw(panel)
    for i,im in enumerate(images):
        x=(i%4)*size;y=(i//4)*(size+24);panel.paste(im,(x,y+24));draw.text((x+5,y+5),f'{oid} / view {i} / two-sided surface inspection',fill='black')
    panel.save(folder/'contact_sheet.jpg')
    return {'views':8,'contact_sheet':str(folder/'contact_sheet.jpg'),'display':'two-sided measured surface; no gap filling'}
