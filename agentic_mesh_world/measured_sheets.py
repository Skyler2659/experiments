"""Observed planar-sheet reconstruction for thin surfaces that TSDF erases.

RANSAC planes come from measured RGB-D, not object templates. Triangulation is
restricted by observation spacing; it never fills a convex hull wholesale.
Unmodelled points are returned as residual evidence, never silently accepted.
"""
from pathlib import Path
import numpy as np
from .mesh_world import pose_matrix
from .quality_reconstruction import points,inside


def reconstruct(data,root,bounds, *, stride=3,point_voxel_m=.0003,plane_tolerance_m=.00015,
                max_planes=40,min_plane_points=200,max_edge_m=.002):
    import open3d as o
    import trimesh
    from scipy.spatial import Delaunay,cKDTree
    from PIL import Image
    xyz=[];colors=[];view_counts=[]
    for v in data['views']:
        if v['role']!='fit' or not v.get('surface_enabled',True):continue
        d=np.load(Path(root)/v['depth_uri']);p=points(d,np.asarray(v['K']),pose_matrix(v['T_W_C']))[::stride,::stride]
        rgb=np.asarray(Image.open(Path(root)/v['rgb_uri']).convert('RGB'))[::stride,::stride]
        valid=inside(p,bounds)&np.isfinite(p).all(-1)&(d[::stride,::stride]>0)
        if v.get('robot_mask_uri'):
            robot=np.load(Path(root)/v['robot_mask_uri']) if str(v['robot_mask_uri']).endswith('.npy') else np.asarray(Image.open(Path(root)/v['robot_mask_uri']).convert('L'))>0
            valid &= ~robot[::stride,::stride].astype(bool)
        xyz.append(p[valid]);colors.append(rgb[valid]);view_counts.append({'view_id':v['view_id'],'points':int(valid.sum())})
    cloud=o.geometry.PointCloud(o.utility.Vector3dVector(np.concatenate(xyz)));cloud.colors=o.utility.Vector3dVector(np.concatenate(colors)/255.)
    cloud=cloud.voxel_down_sample(point_voxel_m);allpoints=np.asarray(cloud.points).copy();total=len(allpoints)
    o.utility.random.seed(7301);pieces=[];records=[];unmeshed=[]
    for i in range(max_planes):
        if len(cloud.points)<min_plane_points:break
        plane,inliers=cloud.segment_plane(distance_threshold=plane_tolerance_m,ransac_n=3,num_iterations=1000,probability=.999)
        if len(inliers)<min_plane_points:break
        xyz=np.asarray(cloud.points)[inliers];rgb=np.asarray(cloud.colors)[inliers]
        normal=np.asarray(plane[:3]);normal/=np.linalg.norm(normal);origin=xyz.mean(0)
        axis=np.eye(3)[np.argmin(abs(normal))];u=np.cross(normal,axis);u/=np.linalg.norm(u);v=np.cross(normal,u)
        uv=np.column_stack([(xyz-origin)@u,(xyz-origin)@v])
        # Merge close samples in-plane to avoid almost coincident sliver triangles.
        cell=np.floor(uv/point_voxel_m).astype('int64');_,inverse=np.unique(cell,axis=0,return_inverse=True)
        n=int(inverse.max()+1);weights=np.bincount(inverse);uv2=np.column_stack([np.bincount(inverse,weights=uv[:,j])/weights for j in range(2)])
        color=np.column_stack([np.bincount(inverse,weights=rgb[:,j])/weights for j in range(3)])
        try:faces=Delaunay(uv2).simplices
        except Exception:
            unmeshed.append(xyz)
            cloud=cloud.select_by_index(inliers,invert=True);continue
        tri=uv2[faces];edge=np.linalg.norm(tri-np.roll(tri,1,axis=1),axis=2).max(axis=1)
        # Reject triangles whose centre has no nearby measured point as well.
        distance=cKDTree(uv2).query(tri.mean(1))[0]
        faces=faces[(edge<=max_edge_m)&(distance<=max_edge_m/2)]
        used=np.zeros(n,bool);used[np.unique(faces)]=True
        unmeshed.append(xyz[~used[inverse]])
        vertices=origin+uv2[:,0,None]*u+uv2[:,1,None]*v
        mesh=trimesh.Trimesh(vertices,faces,vertex_colors=np.column_stack([np.clip(color*255,0,255),np.full(n,255)]).astype('uint8'),process=False)
        mesh.remove_unreferenced_vertices()
        if len(mesh.faces):
            pieces.append(mesh);records.append({'id':f'measured_sheet_{i:03d}','normal_W':normal.tolist(),'origin_W':origin.tolist(),
                'measured_points':len(inliers),'faces':len(mesh.faces),'plane_residual_p95_m':float(np.quantile(abs((xyz-origin)@normal),.95)),
                'construction':'measured plane with spacing-limited triangulation; no inferred thickness'})
        cloud=cloud.select_by_index(inliers,invert=True)
    if not pieces:raise ValueError('No measured planar sheets; request another backend rather than invent planes')
    mesh=trimesh.util.concatenate(pieces)
    residual=np.concatenate([np.asarray(cloud.points),*unmeshed])
    return mesh,{'backend':'measured_planar_sheets','views':view_counts,'sheets':records,'input_downsampled_points':total,
        'residual_points':len(residual),'residual_fraction':len(residual)/max(total,1),
        'limits':{'plane_tolerance_m':plane_tolerance_m,'point_voxel_m':point_voxel_m,'max_edge_m':max_edge_m},
        'not_recovered':'unobserved geometry, thickness, nonplanar residuals'},residual
