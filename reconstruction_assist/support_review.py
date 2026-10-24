"""Scene-only vertical support check on delivered meshes; no simulator geometry."""
from pathlib import Path
import numpy as np
import trimesh
import open3d as o

def analyze(directory,inventory,tolerance_m=.006):
    directory=Path(directory)
    if 'tabletop' not in inventory:return []
    meshes={oid:trimesh.load(directory/(oid+'.glb'),force='scene',process=False).to_geometry() for oid in inventory if oid!='known_robot'}
    rows=[]
    for oid,mesh in meshes.items():
        if oid=='tabletop' or not len(mesh.vertices):continue
        bottom=mesh.vertices[mesh.vertices[:,2]<=mesh.vertices[:,2].min()+.003]
        bottom=bottom[np.linspace(0,len(bottom)-1,min(len(bottom),256),dtype=int)]
        scene=o.t.geometry.RaycastingScene();support_ids=[]
        for other,support in meshes.items():
            if other==oid or not len(support.faces):continue
            scene.add_triangles(o.t.geometry.TriangleMesh.from_legacy(o.geometry.TriangleMesh(o.utility.Vector3dVector(support.vertices),o.utility.Vector3iVector(support.faces))));support_ids.append(other)
        origins=bottom+np.array([0,0,.001]);rays=np.c_[origins,np.broadcast_to([0,0,-1],origins.shape)].astype('float32');hits=scene.cast_rays(o.core.Tensor(rays),nthreads=2);d=hits['t_hit'].numpy();finite=np.isfinite(d)
        gap=float(np.min(d[finite])-.001) if finite.any() else None
        idx=int(hits['geometry_ids'].numpy()[np.argmin(d)]) if finite.any() else None
        rows.append({'object_id':oid,'support_object_id':support_ids[idx] if idx is not None and idx<len(support_ids) else None,'minimum_vertical_support_gap_m':gap,'floating':gap is None or gap>tolerance_m,'tolerance_m':tolerance_m,'scope':'delivered geometry contact diagnostic; does not certify mechanical stability or hidden support'})
    return rows
