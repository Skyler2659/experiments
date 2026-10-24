"""Measured depth-grid triangulation for thin/nonplanar observed surfaces.

Unlike signed-distance fusion, this preserves distinct sheets. Faces only join
neighbouring sensor pixels; depth jumps and grazing triangles are excluded.
No unseen gaps are capped. Overlapping measured patches remain explicit.
"""
from pathlib import Path
import numpy as np
from .mesh_world import pose_matrix
from .quality_reconstruction import points,inside


def reconstruct(data,root,bounds, *, stride=2,max_edge_m=.0015,min_facing=.12,consistent_color=True):
    import trimesh
    from PIL import Image
    meshes=[];rows=[]
    for v in data['views']:
        if v['role']!='fit' or not v.get('surface_enabled',True):continue
        depth=np.load(Path(root)/v['depth_uri']);t=pose_matrix(v['T_W_C'])
        full_points=points(depth,np.asarray(v['K']),t);p=full_points[::stride,::stride]
        rgb=np.asarray(Image.open(Path(root)/v['rgb_uri']).convert('RGB'))[::stride,::stride]
        full_valid=inside(full_points,bounds)&np.isfinite(full_points).all(-1)&np.isfinite(depth)&(depth>0)
        if v.get('robot_mask_uri'):
            robot=np.load(Path(root)/v['robot_mask_uri'],allow_pickle=False)
            if robot.shape!=depth.shape:raise ValueError('Robot/depth dimensions differ')
            full_valid&=~robot
        valid=full_valid[::stride,::stride]
        ids=np.arange(valid.size).reshape(valid.shape)
        a,b,c,d=ids[:-1,:-1],ids[:-1,1:],ids[1:,:-1],ids[1:,1:]
        faces=np.concatenate([np.stack([a,b,c],-1).reshape(-1,3),np.stack([b,d,c],-1).reshape(-1,3)])
        if stride>1:
            # Downsampling cannot turn a missing/foreground pixel between
            # sampled corners into an observed triangle. Check the complete
            # original pixel cell, retaining uncertainty at its boundary.
            bad=np.pad((~full_valid).astype('int32'),((1,0),(1,0))).cumsum(0).cumsum(1)
            ys=np.arange(0,depth.shape[0],stride);xs=np.arange(0,depth.shape[1],stride)
            count=(bad[np.ix_(ys[1:]+1,xs[1:]+1)]-bad[np.ix_(ys[:-1],xs[1:]+1)]
                   -bad[np.ix_(ys[1:]+1,xs[:-1])]+bad[np.ix_(ys[:-1],xs[:-1])])
            cells=count.ravel()==0;faces=faces[np.r_[cells,cells]]
        faces=faces[valid.ravel()[faces].all(-1)]
        xyz=p.reshape(-1,3);tri=xyz[faces]
        edges=np.linalg.norm(tri-np.roll(tri,1,axis=1),axis=2).max(-1)
        normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);length=np.linalg.norm(normal,axis=1)
        ray=t[:3,3]-tri.mean(1);ray/=np.maximum(np.linalg.norm(ray,axis=1,keepdims=True),1e-12)
        facing=abs(np.sum(normal*ray,axis=1))/np.maximum(length,1e-20)
        keep=(edges<=max_edge_m)&(facing>=min_facing)&(length>1e-15)
        faces=faces[keep]
        reverse=np.sum(normal[keep]*ray[keep],axis=1)<0
        faces[reverse]=faces[reverse][:,[0,2,1]]
        if not len(faces):continue
        mesh=trimesh.Trimesh(xyz,faces,vertex_colors=rgb.reshape(-1,3),process=False);mesh.remove_unreferenced_vertices()
        meshes.append(mesh);rows.append({'view_id':v['view_id'],'faces':len(faces)})
    if not meshes:raise ValueError('No locally continuous measured surface; require additional observations')
    mesh=trimesh.util.concatenate(meshes)
    appearance=harmonize_colors(mesh,data,root) if consistent_color else {'mode':'original per-view RGB'}
    return mesh,{'backend':'measured_projective_surface','views':rows,
        'limits':{'stride':stride,'max_edge_m':max_edge_m,'min_facing':min_facing},
        'construction':'adjacent sensor pixels only; preserve thin and curved sheets; no gap filling',
        'overlap':'multi-view measured patches; no closed-solid or part-identity claim','appearance':appearance}


def harmonize_colors(mesh,data,root,normal_distance_m=.00005,chunk=200000):
    """Choose the same well-facing measured RGB for coincident view patches.

    Avoid multi-view colour flicker without inventing textures. Occluded points
    and pixels crossing a different thin sheet cannot supply appearance.
    """
    from PIL import Image
    xyz=mesh.vertices;normals=mesh.vertex_normals
    score=np.full(len(xyz),-1.);color=mesh.visual.vertex_colors.copy()
    for v in data['views']:
        if v['role']!='fit' or not v.get('surface_enabled',True):continue
        depth=np.load(Path(root)/v['depth_uri']);rgb=np.asarray(Image.open(Path(root)/v['rgb_uri']).convert('RGB'))
        k=np.asarray(v['K']);t=pose_matrix(v['T_W_C'])
        for start in range(0,len(xyz),chunk):
            end=min(start+chunk,len(xyz));p=(xyz[start:end]-t[:3,3])@t[:3,:3];z=p[:,2]
            uv=p@k.T;xy=np.rint(uv[:,:2]/np.maximum(z[:,None],1e-12)).astype('int64')
            valid=(z>0)&(xy[:,0]>=0)&(xy[:,1]>=0)&(xy[:,0]<depth.shape[1])&(xy[:,1]<depth.shape[0])
            ids=np.flatnonzero(valid);ray=t[:3,3]-xyz[start:end];ray/=np.maximum(np.linalg.norm(ray,axis=1,keepdims=True),1e-12)
            facing=abs(np.sum(ray*normals[start:end],axis=1))
            obs=depth[xy[ids,1],xy[ids,0]]
            good=(abs(z[ids]-obs)*facing[ids]<=normal_distance_m)&(facing[ids]>.2)&(facing[ids]>score[start+ids])
            ids=ids[good];score[start+ids]=facing[ids];color[start+ids,:3]=rgb[xy[ids,1],xy[ids,0]]
    mesh.visual.vertex_colors=color
    return {'mode':'best-facing depth-consistent observed RGB','normal_distance_m':normal_distance_m,
            'recolored_vertex_fraction':float(np.mean(score>=0)),'generated_texture_used':False}
