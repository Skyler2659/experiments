"""Nearest real mesh triangle per calibrated image pixel (no proxy geometry)."""
import numpy as np
from .mesh_world import pose_matrix


def visible_faces(mesh, view, width, height):
    camera=np.linalg.inv(pose_matrix(view['T_W_C']))
    points=np.asarray(mesh.vertices)@camera[:3,:3].T+camera[:3,3]
    projected=points@np.asarray(view['K']).T
    uv=projected[:,:2]/np.maximum(projected[:,2,None],1e-12)
    depth=np.full((height,width),np.inf)
    faces=np.full((height,width),-1,dtype=np.int32)
    for fid,indices in enumerate(mesh.faces):
        z=points[indices,2]
        if np.any(z<=.001):continue
        triangle=uv[indices]
        lo=np.maximum(np.floor(triangle.min(0)).astype(int),0)
        hi=np.minimum(np.ceil(triangle.max(0)).astype(int),[width-1,height-1])
        if np.any(hi<lo):continue
        a,b,c=triangle
        denominator=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
        if abs(denominator)<1e-10:continue
        y,x=np.mgrid[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
        # Calibrated RGB-D uses integer pixel centers.
        wa=((b[1]-c[1])*(x-c[0])+(c[0]-b[0])*(y-c[1]))/denominator
        wb=((c[1]-a[1])*(x-c[0])+(a[0]-c[0])*(y-c[1]))/denominator
        wc=1-wa-wb
        inside=(wa>=-1e-8)&(wb>=-1e-8)&(wc>=-1e-8)
        distance=1/np.maximum(wa/z[0]+wb/z[1]+wc/z[2],1e-12)
        region=np.s_[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
        keep=inside&(distance<depth[region])
        depth[region][keep]=distance[keep];faces[region][keep]=fid
    return faces
