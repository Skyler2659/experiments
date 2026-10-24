"""Allocate a measured regional volume; integrate original uncut depth rays.

Cropping allocates computation only. A depth return behind the region must still
update its existing voxels. Occlusion, invalid depth and robot pixels do not
become empty-space observations. Uses Open3D's VoxelBlockGrid integration.
"""
from pathlib import Path
import numpy as np
from PIL import Image
from .mesh_world import pose_matrix
from .quality_reconstruction import points,inside


def fuse(data,root,bounds,voxel,*,truncation_voxels=4,device='CPU:0',weight_threshold=.5):
    import open3d as o
    root=Path(root);dev=o.core.Device(device);views=[v for v in data['views'] if v['role']=='fit' and v.get('surface_enabled',True)]
    def image(a):return o.t.geometry.Image(o.core.Tensor(np.ascontiguousarray(a))).to(dev)
    def camera(v):return (o.core.Tensor(np.asarray(v['K']),o.core.Dtype.Float64),o.core.Tensor(np.linalg.inv(pose_matrix(v['T_W_C'])),o.core.Dtype.Float64))
    def depth(v):
        d=np.load(root/v['depth_uri']).astype('float32');d[~np.isfinite(d)|(d<=0)]=0
        if v.get('robot_mask_uri'):d[np.load(root/v['robot_mask_uri'])]=0
        return d
    attrs={'attr_names':['tsdf','weight','color'],'attr_dtypes':[o.core.Dtype.Float32]*3,'attr_channels':[(1,),(1,),(3,)],
        'voxel_size':voxel,'block_resolution':8,'device':dev}
    scratch=o.t.geometry.VoxelBlockGrid(**attrs,block_count=1024);coords=[];counts=[]
    for v in views:
        d=depth(v);valid=(d>0)&inside(points(d,np.asarray(v['K']),pose_matrix(v['T_W_C'])),np.asarray(bounds))
        counts.append({'view_id':v['view_id'],'valid_pixels':int(valid.sum()),'integration_original_valid_pixels':int((d>0).sum())})
        if not valid.any():continue
        k,t=camera(v);block=scratch.compute_unique_block_coordinates(image(np.where(valid,d,0)),k,t,1.,5.,truncation_voxels)
        coords.append(block.cpu().numpy())
    if not coords:raise ValueError('No observed depth activates the regional volume')
    unique=np.unique(np.concatenate(coords),axis=0).astype('int32');del scratch
    grid=o.t.geometry.VoxelBlockGrid(**attrs,block_count=max(1024,int(len(unique)*1.3)+1));blocks=o.core.Tensor(unique,device=dev)
    grid.hashmap().activate(blocks)
    for v in views:
        d=depth(v);rgb=np.asarray(Image.open(root/v['rgb_uri']).convert('RGB'),dtype='float32')
        if rgb.shape[:2]!=d.shape:raise ValueError('Original RGB-D dimensions differ')
        k,t=camera(v)
        # These are measured returns, including background outside the ROI.
        # No virtual clipping plane or candidate surface is integrated.
        grid.integrate(blocks,image(d),image(rgb),k,k,t,1.,5.,float(truncation_voxels))
        print('bounded original-depth integrate',v['view_id'],len(unique),flush=True)
    mesh=grid.extract_triangle_mesh(weight_threshold=weight_threshold).to_legacy()
    mesh.remove_degenerate_triangles();mesh.remove_duplicated_triangles();mesh.remove_unreferenced_vertices()
    return mesh,counts
