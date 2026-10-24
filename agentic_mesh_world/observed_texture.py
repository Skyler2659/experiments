"""Bake calibrated fit RGB into an atlas without changing measured triangles."""
from pathlib import Path
import numpy as np
from PIL import Image
from .instance_surface import project_samples
from .mesh_world import pose_matrix


def bake(mesh, views, root, *, tolerance_m=.003,max_views=8,max_atlas_size=4096):
    import trimesh
    selected=[v for v in views if v['role']=='fit' and v.get('surface_enabled',True)][:max_views]
    if not selected:raise ValueError('Observed texture requires fit RGBD')
    n=len(mesh.faces);best=np.full(n,-1,np.int32);quality=np.full(n,-np.inf)
    projected=[];images=[]
    for vi,v in enumerate(selected):
        d=np.load(Path(root)/v['depth_uri']);t=pose_matrix(v['T_W_C']);k=np.asarray(v['K'])
        image=Image.open(Path(root)/v['rgb_uri']).convert('RGB');images.append(image)
        p=(mesh.vertices-t[:3,3])@t[:3,:3];uv=p@k.T;uv=uv[:,:2]/np.maximum(p[:,2,None],1e-12);projected.append(uv)
        for start in range(0,n,50000):
            stop=min(n,start+50000);tri=mesh.triangles[start:stop];centers=tri.mean(1)
            samples=np.concatenate([centers[:,None],tri],1)
            _,_,visible=project_samples(samples,v,d,tolerance_m)
            delta=t[:3,3]-centers;dist=np.linalg.norm(delta,axis=1)
            angle=abs(np.sum(mesh.face_normals[start:stop]*delta/np.maximum(dist[:,None],1e-12),1))
            score=angle*k[0,0]/np.maximum(dist,1e-9)
            valid=visible.all(1)&(angle>.2)&(score>quality[start:stop])
            best[start:stop][valid]=vi;quality[start:stop][valid]=score[valid]
    # One tile per view plus a per-face observed-color fallback tile.
    columns=int(np.ceil(np.sqrt(len(images)+1)));tile=min(max(max(i.size) for i in images),max_atlas_size//columns)
    if tile<32:raise ValueError('Atlas budget too small')
    atlas=Image.new('RGB',(columns*tile,columns*tile));uvs=[]
    for i,im in enumerate(images):atlas.paste(im.resize((tile,tile)),((i%columns)*tile,(i//columns)*tile))
    missing=np.flatnonzero(best<0);side=max(1,int(np.ceil(np.sqrt(len(missing)))))
    if side>tile:raise ValueError('Fallback tile budget insufficient; increase atlas resolution')
    fallback=np.zeros((tile,tile,3),np.uint8)
    colors=np.asarray(mesh.visual.vertex_colors)[:,:3][mesh.faces[missing]].mean(1).astype('uint8')
    if len(missing):fallback[np.arange(len(missing))//side,np.arange(len(missing))%side]=colors
    atlas.paste(Image.fromarray(fallback),((len(images)%columns)*tile,(len(images)//columns)*tile))
    # Reuse vertices within a source view. Seams duplicate vertices, never faces.
    vertices=[];faces=np.empty_like(mesh.faces);cursor=0
    for vi in range(len(images)):
        ids=np.flatnonzero(best==vi)
        if not len(ids):continue
        old,inverse=np.unique(mesh.faces[ids].ravel(),return_inverse=True)
        vertices.append(mesh.vertices[old]);faces[ids]=inverse.reshape(-1,3)+cursor;cursor+=len(old)
        uv=(projected[vi][old]+.5)/np.asarray(images[vi].size)
        uv=(uv+np.array([vi%columns,vi//columns]))/columns;uv[:,1]=1-uv[:,1];uvs.append(uv)
    if len(missing):
        vertices.append(mesh.triangles[missing].reshape(-1,3));faces[missing]=np.arange(len(missing)*3).reshape(-1,3)+cursor
        xy=np.column_stack([np.arange(len(missing))%side,np.arange(len(missing))//side])+.5
        uv=(xy/tile+np.array([len(images)%columns,len(images)//columns]))/columns;uv[:,1]=1-uv[:,1]
        uvs.append(np.repeat(uv,3,axis=0))
    result=trimesh.Trimesh(np.concatenate(vertices),faces,process=False)
    result.visual=trimesh.visual.TextureVisuals(uv=np.concatenate(uvs),material=trimesh.visual.material.PBRMaterial(baseColorTexture=atlas,metallicFactor=0.,roughnessFactor=1.))
    if not np.allclose(result.triangles,mesh.triangles,atol=1e-12):raise ValueError('Texture changed geometry')
    return result,{'source':'fit RGB only','view_ids':[v['view_id'] for v in selected],
                   'projected_face_fraction':float(np.mean(best>=0)),'fallback':'original measured vertex colors',
                   'atlas_size':list(atlas.size),'geometry_unchanged':True,'photometric_relighting_solved':False}
