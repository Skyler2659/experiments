"""Shared boundary and support constraints for static reconstruction."""
import copy
import numpy as np


SUPPORT_REPLAY_SOURCE='''import importlib.util,json
from pathlib import Path
from agentic_mesh_world.static_surface_constraints import extend_hidden_base
def make_mesh():
    p=Path(__file__).parent
    s=importlib.util.spec_from_file_location("prior",p/"pre_support_model.py")
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    r=json.loads((p/"support_completion.json").read_text())
    return extend_hidden_base(m.make_mesh(),r["plane_W"],max_gap=r.get("max_gap_m",.025),force_implicit=r.get("method")=="bounded_signed_distance_material_union")[0]
'''


def extruded_polygon(polygon, thickness, thin_axis=0):
    import manifold3d as m
    import trimesh
    solid=m.CrossSection([np.asarray(polygon)],m.FillRule.EvenOdd).extrude(float(thickness))
    raw=solid.to_mesh64()
    mesh=trimesh.Trimesh(np.array(raw.vert_properties[:,:3],copy=True),np.asarray(raw.tri_verts),process=False)
    # Cross-section coordinates are horizontal span and vertical height.
    v=mesh.vertices.copy();span=1-thin_axis
    mesh.vertices[:,thin_axis]=v[:,2]-thickness/2
    mesh.vertices[:,span]=v[:,0];mesh.vertices[:,2]=v[:,1]
    if mesh.volume<0:mesh.invert()
    return mesh


def fit_panel_boundary(part, xyz):
    """Fit upper contour of a thin vertical panel, not a rectangular envelope."""
    if part['kind']!='box':return part,{'status':'not_applicable'}
    size=np.asarray(part['size_m']);axis=int(np.argmin(size));span=1-axis
    if axis==2 or size[axis]/min(size[span],size[2])>.10 or size[span]<.03:return part,{'status':'not_applicable'}
    t=np.asarray(part['T_W_part']);p=(xyz-t[:3,3])@t[:3,:3]
    valid=(abs(p[:,axis])<max(size[axis]*.8,.003))&(abs(p[:,span])<size[span]/2+.001)&(p[:,2]>-size[2]/2-.002)&(p[:,2]<size[2]/2+.006)
    p=p[valid];edges=np.linspace(-size[span]/2,size[span]/2,25);x=[];z=[];counts=[]
    for a,b in zip(edges[:-1],edges[1:]):
        q=p[(p[:,span]>=a)&(p[:,span]<b)]
        if len(q)>=12:
            x.append((a+b)/2);z.append(float(np.quantile(q[:,2],.985)));counts.append(len(q))
    coverage=len(x)/24
    receipt={'status':'insufficient_boundary_evidence','coverage':coverage,'points':len(p),'thin_axis':axis}
    if coverage<.65:return part,receipt
    from scipy.optimize import least_squares
    x=np.asarray(x);z=np.asarray(z)
    result=least_squares(lambda a:a[0]*x+a[1]-z,[0,np.median(z)],loss='soft_l1',f_scale=.002)
    line=result.x[0]*x+result.x[1];residual=float(np.quantile(abs(line-z),.9))
    variation=float(abs(result.x[0])*size[span])
    receipt.update(slope=float(result.x[0]),line_residual_p90_m=residual,edge_height_change_m=variation)
    if variation<.008 or residual>.006:
        receipt['status']='flat_boundary_supported' if residual<.004 and variation<.008 else 'flat_or_non_linear_boundary_requires_review'
        return part,receipt
    ends=np.array([-size[span]/2,size[span]/2]);top=result.x[0]*ends+result.x[1]+.001
    top=np.clip(top,-size[2]/2+.004,size[2]/2+.004)
    polygon=[[ends[0],-size[2]/2],[ends[1],-size[2]/2],[ends[1],top[1]],[ends[0],top[0]]]
    new={'kind':'profile_panel','thin_axis':axis,'thickness_m':float(size[axis]),'polygon_span_z_m':np.asarray(polygon).tolist(),'T_W_part':part['T_W_part'],'boundary_evidence':receipt}
    receipt['status']='observed_sloped_boundary_fitted'
    return new,receipt


def conform_recipe(recipe, xyz):
    new=copy.deepcopy(recipe);records=[]
    for i,p in enumerate(new['parts']):
        fitted,report=fit_panel_boundary(p,xyz);new['parts'][i]=fitted
        if report['status']!='not_applicable':records.append({'part_index':i,**report})
    new['boundary_constraints']=records
    if any(r['status']=='observed_sloped_boundary_fitted' for r in records):
        new['assumptions'].append('thin-panel upper contours fitted to measured depth envelope; hidden thickness and extrapolated endpoints remain hypotheses')
    return new


def fit_plane(xyz):
    """Robust shared support plane with orientation and residual evidence."""
    import open3d as o
    p=np.asarray(xyz);cloud=o.geometry.PointCloud(o.utility.Vector3dVector(p))
    coeff,idx=cloud.segment_plane(distance_threshold=.0015,ransac_n=3,num_iterations=1500)
    q=p[idx];center=q.mean(0);_,v=np.linalg.eigh(np.cov(q-center,rowvar=False));n=v[:,0]
    if n[2]<0:n=-n
    plane=np.r_[n,-n@center]
    if n[2]<.95:raise ValueError('Support is not an upward work surface')
    return plane,{'inliers':len(idx),'residual_p95_m':float(np.quantile(abs(q@n+plane[3]),.95))}


def contact_metrics(mesh, plane, tolerance=.002):
    plane=np.asarray(plane);h=mesh.vertices@plane[:3]+plane[3]
    base=dominant_base_height(mesh,plane)
    face_h=mesh.triangles_center@plane[:3]+plane[3]
    patch=(abs(face_h)<tolerance)&(mesh.face_normals@plane[:3]<-.3)
    area=float(mesh.area_faces[patch].sum())
    status='penetrating' if h.min() < -tolerance else 'floating' if h.min()>tolerance else 'contact_geometry_consistent' if area>=tolerance**2 or base<=tolerance else 'support_contact_uncertain'
    return {'signed_min_gap_m':float(h.min()),'near_plane_vertices':int(np.sum(abs(h)<tolerance)),
            'contact_patch_area_m2':area,'dominant_base_gap_m':base,'penetration_m':float(max(0,-h.min())),'status':status,
            'physical_stability':'not_evaluated'}


def dominant_base_height(mesh,plane):
    n=np.asarray(plane[:3]);h=mesh.vertices@n+plane[3];centers=mesh.triangles_center@n+plane[3]
    eligible=(mesh.face_normals@n<-.7)&(centers<h.min()+.025)
    if not eligible.any():return float(h.min())
    bins=np.arange(h.min()-.001,h.min()+.026,.001);weights,_=np.histogram(centers[eligible],bins,weights=mesh.area_faces[eligible]);i=int(np.argmax(weights));mask=eligible&(centers>=bins[i])&(centers<bins[i+1])
    return float(np.average(centers[mask],weights=mesh.area_faces[mask])) if mask.any() else float(h.min())


def extend_hidden_base(mesh, plane, max_gap=.025,force_implicit=False):
    """Add an inferred lower base; never translate observed object geometry."""
    import manifold3d as m
    import trimesh
    from .static_asset_geometry import union
    vertices,inverse=np.unique(mesh.vertices,axis=0,return_inverse=True);mesh=trimesh.Trimesh(vertices,inverse[mesh.faces],process=False)
    contact=contact_metrics(mesh,plane)
    if contact['status']=='contact_geometry_consistent':return mesh,{'status':'not_needed','contact':contact}
    if contact['status']=='support_contact_uncertain':raise ValueError('Ambiguous contact: do not turn feet into a solid base')
    plane=np.asarray(plane);n=plane[:3];n=n/np.linalg.norm(n);u=np.cross([0,1,0],n);u/=np.linalg.norm(u);v=np.cross(n,u);basis=np.column_stack([u,v,n]);center=mesh.bounds.mean(0);origin=center-(center@n+plane[3])*n
    local=mesh.copy();local.vertices=(mesh.vertices-origin)@basis;gap=dominant_base_height(mesh,plane)
    if gap<=.002:return mesh,{'status':'not_needed_or_penetrating','gap_m':gap}
    if gap>max_gap:raise ValueError('Gap exceeds evidence-supported hidden-base completion budget')
    level=gap+.0015;section=local.section(plane_origin=[0,0,level],plane_normal=[0,0,1])
    if section is None:raise ValueError('No closed base cross-section')
    contours=[np.asarray(c)[:,:2] for c in section.discrete if len(c)>3]
    if not contours:raise ValueError('No base contour; do not use a convex hull across legs')
    cross=m.CrossSection(contours,m.FillRule.EvenOdd).offset(.000001);solid=cross.extrude(level+.0001);raw=solid.to_mesh64()
    foot=trimesh.Trimesh(np.array(raw.vert_properties[:,:3],copy=True),np.asarray(raw.tri_verts),process=False);foot.vertices[:,2]-=.0001
    method='double_precision_material_union';remesh_pitch=None
    try:
        if force_implicit:raise ValueError('Requested serialization-stable field union')
        result=union([local,foot])
        if not result.is_watertight or len(result.split())!=1:raise ValueError('Nonclosed material union')
    except ValueError:
        # Dense field/UV assets can contain numerical tangencies. A bounded
        # signed-distance union closes the material without translating it.
        result,remesh_pitch=implicit_material_union(local,foot);method='bounded_signed_distance_material_union'
    result.vertices=result.vertices@basis.T+origin
    return result,{'status':'hidden_base_completion_hypothesis','max_gap_m':max_gap,'method':method,'remesh_pitch_m':remesh_pitch,'dominant_base_gap_before_m':gap,'section_height_m':level,'contours':len(contours),'numerical_overlap_m':.000001,'pose_changed':False,
                  'assumption':('implicit material union remeshes the full body; inferred base is not measured' if remesh_pitch else 'visually inferred resting object; unobserved base extended to measured support; physical stability not established')}


def implicit_material_union(first,second):
    import open3d as o
    from .static_bounded_asset import field_mesh,simplify_mesh
    low=np.minimum(first.bounds[0],second.bounds[0]);high=np.maximum(first.bounds[1],second.bounds[1]);pitch=.0015 if np.prod(high-low)>.002 else .001
    low-=3*pitch;high+=3*pitch;shape=np.ceil((high-low)/pitch).astype(int)+1
    scenes=[]
    for mesh in [first,second]:
        scene=o.t.geometry.RaycastingScene();scene.add_triangles(o.t.geometry.TriangleMesh.from_legacy(o.geometry.TriangleMesh(o.utility.Vector3dVector(mesh.vertices),o.utility.Vector3iVector(mesh.faces))));scenes.append(scene)
    field=np.empty(int(np.prod(shape)),np.float32)
    for start in range(0,len(field),60000):
        flat=np.arange(start,min(start+60000,len(field)));coords=np.column_stack(np.unravel_index(flat,shape))*pitch+low;query=o.core.Tensor(coords.astype('float32'))
        field[start:start+len(flat)]=np.minimum(*[s.compute_signed_distance(query,nthreads=2).numpy() for s in scenes])
    field=field.reshape(shape)
    for axis in range(3):
        sl=[slice(None)]*3;sl[axis]=0;field[tuple(sl)]=pitch;sl[axis]=-1;field[tuple(sl)]=pitch
    mesh=field_mesh(field,low,pitch);mesh=simplify_mesh(mesh,120000)
    if not mesh.is_watertight or len(mesh.split())!=1:raise ValueError('Implicit material union failed continuity')
    return mesh,pitch
