"""Mesh-derived object-frame evidence for structural proposals, never task truth."""
import numpy as np


def object_frame(world, oid):
    node=world.node(oid)
    transform=world.object_transform(oid)@np.asarray(node['asset']['T_object_mesh'])
    transform=np.asarray(getattr(world,'deltas',{}).get(oid,np.eye(4)))@transform
    # T_object_mesh acts on the upright mesh before source Y-up conversion.
    u,_,vt=np.linalg.svd(transform[:3,:3]);rotation=u@vt
    result=np.eye(4);result[:3,:3]=rotation;result[:3,3]=transform[:3,3]
    return result


def profile(world, oid):
    frame=object_frame(world,oid);mesh=world.mesh(oid).copy();mesh.apply_transform(np.linalg.inv(frame))
    low,high=mesh.bounds;extent=np.maximum(high-low,1e-12)
    centers=(mesh.triangles_center-low)/extent;areas=mesh.area_faces;normals=mesh.face_normals
    features=[]
    for axis in range(3):
        selected=abs(normals[:,axis])>.94
        bins=np.clip((centers[:,axis]*40).astype(int),0,39)
        weighted=np.bincount(bins[selected],weights=areas[selected],minlength=40)
        peaks=[]
        for index in np.argsort(weighted)[-10:][::-1]:
            take=selected&(bins==index)
            if not take.any() or weighted[index]/mesh.area<.002:continue
            peaks.append({'normalized_coordinate':float(np.average(centers[take,axis],weights=areas[take])),
                          'band':[float(index/40),float((index+1)/40)],
                          'surface_area_fraction':float(weighted[index]/mesh.area)})
        features.append({'axis':'xyz'[axis],'parallel_surface_peaks':peaks})
    return {'object_frame_T_W':frame.tolist(),'bounds_in_object_frame_m':mesh.bounds.tolist(),
            'size_in_object_frame_m':mesh.extents.tolist(),'axis_aligned_surface_evidence':features,
            'selector_note':'object_box low/high use normalized PARENT bounds in this shared upright object frame; ordinary box uses world axes. Peaks describe generated surfaces, not semantic or physical truth.'}
