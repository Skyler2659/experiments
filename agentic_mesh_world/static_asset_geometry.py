"""Continuous static solids from whole-instance calibrated samples.

Recipes describe geometry hypotheses, not a fixed manipulation-world ontology.
The generators preserve metric transforms and hollow material shells.
"""
import numpy as np


def union(parts):
    import trimesh
    import manifold3d as m
    if len(parts)==1:return parts[0]
    solids=[]
    for part in parts:
        vertices,inverse=np.unique(part.vertices,axis=0,return_inverse=True)
        part=trimesh.Trimesh(vertices,inverse[part.faces],process=False);part.remove_unreferenced_vertices()
        if part.volume<0:part.invert()
        if not part.is_watertight:raise ValueError('Boolean input must be a closed material solid')
        solid=m.Manifold(m.Mesh64(np.ascontiguousarray(part.vertices,dtype=np.float64),np.ascontiguousarray(part.faces,dtype=np.uint64)))
        if solid.is_empty():raise ValueError('Manifold rejected a recipe solid')
        solids.append(solid)
    solid=solids[0]
    for other in solids[1:]:solid=solid+other
    solid=solid.simplify(1e-7)
    out=solid.to_mesh64()
    mesh=trimesh.Trimesh(np.asarray(out.vert_properties)[:,:3],np.asarray(out.tri_verts),process=False)
    # Float32 Boolean intersections can collapse an edge to one exact position.
    # Weld coincident positions and remove its zero-area triangles before export.
    mesh.merge_vertices(digits_vertex=10);mesh.update_faces(mesh.nondegenerate_faces(height=1e-10));mesh.update_faces(mesh.unique_faces());mesh.remove_unreferenced_vertices()
    if not mesh.is_watertight:
        # A removed collinear Boolean face can leave a T junction: one long
        # edge opposite two short edges. Split its adjacent triangle at the
        # existing middle vertex; do not create a cap or move any vertex.
        edges,count=np.unique(mesh.edges_sorted,axis=0,return_counts=True);boundary=edges[count==1]
        for group in trimesh.graph.connected_components(boundary,min_len=3):
            if len(group)!=3:continue
            pairs=np.array([[group[0],group[1]],[group[1],group[2]],[group[2],group[0]]]);length=np.linalg.norm(mesh.vertices[pairs[:,0]]-mesh.vertices[pairs[:,1]],axis=1)
            a,c=pairs[np.argmax(length)];b=next(i for i in group if i not in {a,c});delta=mesh.vertices[c]-mesh.vertices[a]
            alpha=np.dot(mesh.vertices[b]-mesh.vertices[a],delta)/np.dot(delta,delta)
            if not 0<alpha<1 or np.linalg.norm(mesh.vertices[b]-mesh.vertices[a]-alpha*delta)>1e-7:continue
            matches=np.flatnonzero(np.any(mesh.faces==a,axis=1)&np.any(mesh.faces==c,axis=1))
            if len(matches)!=1:continue
            index=matches[0];face=mesh.faces[index];pos=next(j for j in range(3) if {face[j],face[(j+1)%3]}=={a,c});u,v,d=face[pos],face[(pos+1)%3],face[(pos+2)%3]
            keep=np.ones(len(mesh.faces),bool);keep[index]=False;mesh.faces=np.vstack([mesh.faces[keep],[u,b,d],[b,v,d]])
        if not mesh.is_watertight:trimesh.repair.fill_holes(mesh)
    return mesh


def field_slab(x,y,top,thickness):
    """Closed sheet with sampled top, parallel underside, and stitched perimeter."""
    import trimesh
    x=np.asarray(x);y=np.asarray(y);z=np.asarray(top);ny,nx=z.shape
    xx,yy=np.meshgrid(x,y);upper=np.column_stack([xx.ravel(),yy.ravel(),z.ravel()]);n=len(upper)
    lower=upper.copy();lower[:,2]-=thickness;faces=[]
    for j in range(ny-1):
        for i in range(nx-1):
            a=j*nx+i;b=a+1;c=a+nx;d=c+1
            faces.extend([[a,b,d],[a,d,c],[a+n,d+n,b+n],[a+n,c+n,d+n]])
    ring=list(range(nx))+[j*nx+nx-1 for j in range(1,ny)]+list(range(n-2,n-nx-1,-1))+[j*nx for j in range(ny-2,0,-1)]
    for a,b in zip(ring,ring[1:]+ring[:1]):faces.extend([[a,b+n,b],[a,a+n,b+n]])
    mesh=trimesh.Trimesh(np.vstack([upper,lower]),faces,process=False)
    if mesh.volume<0:mesh.invert()
    return mesh


def tray_shell(width,depth,height,wall,floor,segments=96):
    """D-shaped container; watertight ceramic/plastic material, open inner cavity."""
    import trimesh
    theta=np.linspace(0,np.pi,segments)
    boundary=np.column_stack([width/2*np.cos(theta),-depth/2+depth*np.sin(theta)])
    inner=boundary*np.array([(width-2*wall)/width,(depth-2*wall)/depth])+[0,wall/2]
    n=len(boundary);vertices=np.vstack([np.c_[boundary,np.zeros(n)],np.c_[boundary,np.full(n,height)],np.c_[inner,np.full(n,height)],np.c_[inner,np.full(n,floor)],[[0,0,0],[0,0,floor]]]);faces=[]
    for i in range(n):
        j=(i+1)%n
        for a,b in [(0,n),(n,2*n),(2*n,3*n)]:faces.extend([[a+i,a+j,b+j],[a+i,b+j,b+i]])
        faces.extend([[4*n,j,i],[4*n+1,3*n+i,3*n+j]])
    mesh=trimesh.Trimesh(vertices,faces,process=False);mesh.fix_normals()
    if mesh.volume<0:mesh.invert()
    return mesh


def outline_shell(outline,height,wall,floor):
    import trimesh
    outline=np.asarray(outline);center=outline.mean(0);radial=outline-center
    inner=center+radial*(1-wall/np.maximum(np.linalg.norm(radial,axis=1),wall*2))[:,None]
    n=len(outline);v=np.vstack([np.c_[outline,np.zeros(n)],np.c_[outline,np.full(n,height)],np.c_[inner,np.full(n,height)],np.c_[inner,np.full(n,floor)],[list(center)+[0],list(center)+[floor]]]);f=[]
    for i in range(n):
        j=(i+1)%n
        for a,b in [(0,n),(n,2*n),(2*n,3*n)]:f.extend([[a+i,a+j,b+j],[a+i,b+j,b+i]])
        f.extend([[4*n,j,i],[4*n+1,3*n+i,3*n+j]])
    mesh=trimesh.Trimesh(v,f,process=False);mesh.fix_normals()
    if mesh.volume<0:mesh.invert()
    return mesh


def make_recipe(recipe):
    import trimesh
    from .continuous_rigid_reconstruction import lathe_mesh
    parts=[]
    for p in recipe['parts']:
        kind=p['kind']
        if kind=='box':mesh=trimesh.creation.box(p['size_m'])
        elif kind=='profile_panel':
            from .static_surface_constraints import extruded_polygon
            mesh=extruded_polygon(p['polygon_span_z_m'],p['thickness_m'],p['thin_axis'])
        elif kind=='cylinder':mesh=trimesh.creation.cylinder(p['radius_m'],p['height_m'],sections=64)
        elif kind=='torus':mesh=trimesh.creation.torus(p['major_radius_m'],p['minor_radius_m'],major_sections=96,minor_sections=12)
        elif kind=='lathe':mesh=lathe_mesh(np.asarray(p['profile_radius_z_m']))
        elif kind=='field_slab':mesh=field_slab(p['x_m'],p['y_m'],p['top_z_m'],p['thickness_m'])
        elif kind=='tray':mesh=tray_shell(**p['dimensions'])
        elif kind=='outline_shell':mesh=outline_shell(p['outline_xy_m'],**p['dimensions'])
        elif kind=='loop_handle':
            from .static_asset_shape_extensions import loop_handle
            mesh=loop_handle(**p['dimensions'])
        elif kind=='carton_roof':
            from .static_asset_shape_extensions import carton_roof
            mesh=carton_roof(**p['dimensions'])
        elif kind=='polygon_slab':
            from .static_asset_shape_extensions import polygon_slab
            mesh=polygon_slab(**p['dimensions'])
        elif kind=='convex':mesh=trimesh.convex.convex_hull(np.asarray(p['vertices_W_m']))
        else:raise ValueError('Unsupported recipe primitive: '+kind)
        mesh.apply_transform(np.asarray(p.get('T_W_part',np.eye(4))))
        parts.append(mesh)
    return union(parts)


def rectangle_frame(xyz):
    """Find a metric upright rectangle from robust 2-D extents; no native pose."""
    xy=np.asarray(xyz)[:,:2];best=None
    for angle in np.linspace(0,np.pi/2,361):
        c,s=np.cos(angle),np.sin(angle);r=np.array([[c,-s],[s,c]])
        projected=xy@r;lo,hi=np.quantile(projected,[.003,.997],axis=0);area=np.prod(hi-lo)
        if best is None or area<best[0]:best=(area,r,lo,hi)
    _,r,lo,hi=best;rotation=np.eye(3);rotation[:2,:2]=r
    return rotation


def box_part(xyz,rotation=None):
    rotation=rectangle_frame(xyz) if rotation is None else rotation
    local=xyz@rotation;lo,hi=np.quantile(local,[.002,.998],axis=0)
    t=np.eye(4);t[:3,:3]=rotation;t[:3,3]=((lo+hi)/2)@rotation.T
    return {'kind':'box','size_m':(hi-lo).tolist(),'T_W_part':t.tolist()},rotation


def shelf_part(xyz,thickness=.007):
    import open3d as o
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter
    cloud=o.geometry.PointCloud(o.utility.Vector3dVector(xyz))
    plane,idx=cloud.segment_plane(distance_threshold=.0025,ransac_n=3,num_iterations=1500)
    n=np.asarray(plane[:3]);n*=1 if n[2]>0 else -1
    if n[2]<.55:raise ValueError('Shelf fit is not an upward tilted surface')
    u=np.cross([0,0,1],n)
    if np.linalg.norm(u)<.01:u=np.array([1.,0,0])
    u/=np.linalg.norm(u);v=np.cross(n,u);rotation=np.column_stack([u,v,n]);origin=np.mean(xyz[idx],axis=0)
    local=(xyz-origin)@rotation;visible=local[np.abs(local[:,2])<.005]
    lo,hi=np.quantile(visible[:,:2],[.003,.997],axis=0)
    x=np.linspace(lo[0],hi[0],65);y=np.linspace(lo[1],hi[1],33);xx,yy=np.meshgrid(x,y)
    z=griddata(visible[:,:2],visible[:,2],(xx,yy),method='linear')
    missing=~np.isfinite(z);z[missing]=griddata(visible[:,:2],visible[:,2],(xx[missing],yy[missing]),method='nearest')
    z=gaussian_filter(np.clip(z,-.004,.004),.6)
    t=np.eye(4);t[:3,:3]=rotation;t[:3,3]=origin
    return {'kind':'field_slab','x_m':x.tolist(),'y_m':y.tolist(),'top_z_m':z.tolist(),'thickness_m':thickness,'T_W_part':t.tolist()},rotation


def connect_rack(recipe):
    """Complete short occluded attachment regions, retaining observed planes."""
    post=max((p for p in recipe['parts'] if p['kind']=='box'),key=lambda p:p['size_m'][2])
    pt=np.asarray(post['T_W_part']);added=[]
    for shelf in [p for p in recipe['parts'] if p['kind']=='field_slab']:
        st=np.asarray(shelf['T_W_part']);n=st[:3,2];point=pt[:3,3].copy()
        point[2]=st[2,3]-np.dot(n[:2],point[:2]-st[:2,3])/n[2]
        local=(point-st[:3,3])@st[:3,:3]
        x=np.asarray(shelf['x_m']);y=np.asarray(shelf['y_m'])
        target=np.array([np.clip(local[0],x[0]+.002,x[-1]-.002),np.clip(local[1],y[0]+.002,y[-1]-.002),-.004])
        start=point-n*.004;end=st[:3,3]+st[:3,:3]@target;delta=end-start;length=np.linalg.norm(delta)
        if length<.001:continue
        if length>.04:raise ValueError('Rack connection exceeds short occlusion prior')
        direction=delta/length;side=np.cross(direction,n);side/=np.linalg.norm(side)
        t=np.eye(4);t[:3,:3]=np.column_stack([side,direction,n]);t[:3,3]=(start+end)/2
        added.append({'kind':'box','size_m':[float(min(post['size_m'][:2])),float(length+.008),.010],'T_W_part':t.tolist()})
    recipe['parts'].extend(added)
    recipe['assumptions'].append('short unobserved shelf-to-post attachment solids completed from rigid assembly hypothesis; observed geometry is not moved')
    return recipe


def fit_recipe(family,xyz,rgb,regions=None):
    """Fit broad shape + visible major structure, with explicit completion priors."""
    if family in {'box','book','carton','can','mug','caddy','basket','convex'}:
        from .static_asset_shape_extensions import fit
        from .static_surface_constraints import conform_recipe
        recipe,r=fit(family,xyz,rgb,regions or {})
        return conform_recipe(recipe,xyz),r
    from .continuous_rigid_reconstruction import fit_profile
    assumptions=[];regions=regions or {}
    if family in {'bottle','bowl','plate'}:
        profile,origin,assumptions=fit_profile(xyz,family)
        t=np.eye(4);t[:3,3]=origin
        parts=[{'kind':'lathe','profile_radius_z_m':profile.tolist(),'T_W_part':t.tolist()}];r=np.eye(3)
    elif family in {'package','hotplate'}:
        part,r=box_part(xyz);parts=[part]
        assumptions=['upright closed rectangular body fitted to visible RGB-D; hidden faces completed']
        # Coil appearance is carried by original-image texture, avoiding invented
        # raised obstacle geometry on a pattern visible only in RGB.
    elif family=='cabinet':
        silver=(rgb.min(1)>45)&((rgb.max(1)-rgb.min(1))<26)
        wood=(rgb.mean(1)<105)&~silver
        handles=xyz[silver];body=xyz[wood]
        r=rectangle_frame(body)
        if len(handles)>80:
            delta=handles[:,:2].mean(0)-body[:,:2].mean(0)
            front=int(np.argmax(np.abs(delta@r[:2,:2])));other=1-front
            u=r[:,other].copy();v=r[:,front].copy()
            if delta@v[:2]<0:v=-v
            if np.linalg.det(np.column_stack([u,v,[0,0,1]]))<0:u=-u
            r=np.column_stack([u,v,[0,0,1]])
        part,_=box_part(body,r);t=np.asarray(part['T_W_part']);size=np.asarray(part['size_m']);parts=[]
        thickness=.006
        # Carcass: back, sides, roof and floor. Drawer fronts are fitted inside
        # the opening instead of sealing the front with an outer bounding box.
        for dims,offset in [
            ([size[0],thickness,size[2]],[0,-size[1]/2+thickness/2,0]),
            ([thickness,size[1],size[2]],[-size[0]/2+thickness/2,0,0]),
            ([thickness,size[1],size[2]],[size[0]/2-thickness/2,0,0]),
            ([size[0],size[1],thickness],[0,0,-size[2]/2+thickness/2]),
            ([size[0],size[1],thickness],[0,0,size[2]/2-thickness/2])]:
            p=t.copy();p[:3,3]+=r@offset;parts.append({'kind':'box','size_m':list(map(float,dims)),'T_W_part':p.tolist()})
        if len(handles)>80:
            from scipy.cluster.vq import kmeans2
            h=(handles-t[:3,3])@r;centers,labels=kmeans2(h[:,2,None],np.linspace(h[:,2].min(),h[:,2].max(),3)[:,None],minit='matrix',iter=30)
            groups=[h[labels==i] for i in np.argsort(centers[:,0])]
        else:groups=[np.array([[0,size[1]/2-.01,z]]) for z in np.linspace(-size[2]/3,size[2]/3,3)]
        local=(body-t[:3,3])@r
        for group in groups:
            z=float(np.median(group[:,2]));handle_front=float(np.quantile(group[:,1],.8))
            face=local[(abs(local[:,0])<size[0]*.4)&(abs(local[:,2]-z)<size[2]*.12)&(local[:,1]>0)]
            bins=np.arange(0,size[1]/2+.002,.001);hist,edges=np.histogram(face[:,1],bins=bins)
            front=float((edges[np.argmax(hist)]+edges[np.argmax(hist)+1])/2) if len(face)>20 else size[1]/2-.025
            panel=t.copy();panel[:3,3]+=r@np.array([0,front-.003,z])
            parts.append({'kind':'box','size_m':[float(size[0]-.008),.006,float(size[2]/3*.94)],'T_W_part':panel.tolist()})
            handle=t.copy();handle[:3,:3]=np.column_stack([r[:,1],r[:,2],r[:,0]])
            handle[:3,3]+=r@np.array([float(np.median(group[:,0])),handle_front,z])
            length=float(np.clip(np.ptp(group[:,0]),size[0]*.3,size[0]*.5))
            parts.append({'kind':'cylinder','radius_m':.0024,'height_m':length,'T_W_part':handle.tolist()})
            for x in [-length/2,length/2]:
                mount=t.copy();mount[:3,3]+=r@np.array([x+float(np.median(group[:,0])),(handle_front+front)/2,z])
                parts.append({'kind':'box','size_m':[.004,float(max(.006,handle_front-front+.004)),.004],'T_W_part':mount.tolist()})
        assumptions=['static current cabinet appearance; three drawer fronts identified in original RGB; no joints','unseen cabinet shell and handle mounts completed']
    elif family=='rack':
        parts=[];r=None
        for name in ['rack_lower_shelf','rack_upper_shelf']:
            p,rr=shelf_part(regions[name][0]);parts.append(p)
            if r is None:
                u=rr[:,0];v=np.cross([0,0,1],u);r=np.column_stack([u,v,[0,0,1]])
        post,post_rgb=regions['rack_post'];dark=(post_rgb.mean(1)<80)&(post[:,2]>.045)
        post=post[dark]
        # Tall support is selected by the persistent upper column footprint;
        # lower shelf pixels incorrectly selected by SAM do not widen the post.
        high=post[post[:,2]>np.quantile(post[:,2],.8)]
        local=post@r;hl=high@r;lo,hi=np.quantile(hl[:,:2],[.01,.99],axis=0)
        keep=np.all(local[:,:2]>=lo-.003,1)&np.all(local[:,:2]<=hi+.003,1)
        base_cloud,base_rgb=regions['rack_base'];base_cloud=base_cloud[base_rgb.mean(1)<110]
        base,r0=box_part(base_cloud,r);parts.append(base)
        base_top=np.asarray(base['T_W_part'])[2,3]+base['size_m'][2]/2
        p,_=box_part(post[keep],r);p['size_m'][2]=float(np.quantile(post[:,2],.998)-base_top+.002)
        pt=np.asarray(p['T_W_part']);pt[2,3]=base_top-.002+p['size_m'][2]/2;p['T_W_part']=pt.tolist();parts.append(p)
        # A shelf's occluded rear edge may stop a few millimetres before the
        # visible post. Complete that local connection in the recipe, without
        # moving the measured shelf plane or the post.
        from scipy.interpolate import RegularGridInterpolator
        for shelf in parts[:2]:
            st=np.asarray(shelf['T_W_part']);contact=(pt[:3,3]-st[:3,3])@st[:3,:3]
            x=np.asarray(shelf['x_m']);y=np.asarray(shelf['y_m']);z=np.asarray(shelf['top_z_m'])
            newx=x.copy();newy=y.copy()
            for axis,old in [(0,x),(1,y)]:
                target=contact[axis]
                if old[0]-.015<target<old[0]:new=np.linspace(target-.002,old[-1],len(old))
                elif old[-1]<target<old[-1]+.015:new=np.linspace(old[0],target+.002,len(old))
                else:new=old
                if axis==0:newx=new
                else:newy=new
            xx,yy=np.meshgrid(newx,newy);interp=RegularGridInterpolator((y,x),z)
            newz=interp(np.c_[np.clip(yy.ravel(),y[0],y[-1]),np.clip(xx.ravel(),x[0],x[-1])]).reshape(z.shape)
            shelf.update(x_m=newx.tolist(),y_m=newy.tolist(),top_z_m=newz.tolist())
        assumptions=['two tilted grooved shelves fitted from region RGB-D; unobserved sheet thickness completed','post and base completed as closed wood solids; no articulated state']
        assumptions.append('short occluded shelf-to-post connections completed within 15 mm of observed rear edge')
    elif family=='tray':
        from scipy.spatial import ConvexHull,cKDTree
        # Outer wall and high middle structure have different heights. Fit the
        # measured footprint instead of assuming a stretched semicircle.
        high=xyz[xyz[:,2]>np.quantile(xyz[:,2],.8)]
        _,eigen=np.linalg.eigh(np.cov(high[:,:2].T));u=np.r_[eigen[:,-1],0];v=np.cross([0,0,1],u)
        r=np.column_stack([u,v,[0,0,1]]);local=xyz@r
        base=float(np.quantile(local[:,2],.002));center=np.median(local[:,:2],axis=0)
        low=local[local[:,2]<np.quantile(local[:,2],.6)];hull=ConvexHull(low[:,:2]);outline=low[hull.vertices,:2]
        edge_dist=np.full(len(local),np.inf)
        for a,b in zip(outline,np.roll(outline,-1,axis=0)):
            ab=b-a;alpha=np.clip((local[:,:2]-a)@ab/(ab@ab),0,1);dist=np.linalg.norm(local[:,:2]-a-alpha[:,None]*ab,axis=1);edge_dist=np.minimum(edge_dist,dist)
        perimeter=local[edge_dist<.003]
        angle=np.mod(np.arctan2(perimeter[:,1]-center[1],perimeter[:,0]-center[0]),2*np.pi)
        bins=np.floor(angle/(2*np.pi)*48).astype(int)
        upper=[np.quantile(perimeter[bins==i,2],.95) for i in range(48) if np.sum(bins==i)>3]
        rim=float(np.median(upper));height=max(.009,rim-base)
        t=np.eye(4);t[:3,:3]=r;t[:3,3]=np.r_[center,base]@r.T
        parts=[{'kind':'outline_shell','outline_xy_m':(outline-center).tolist(),'dimensions':{'height':height,'wall':.002,'floor':.002},'T_W_part':t.tolist()}]
        divider_cloud=local[local[:,2]>rim+.004];lo,hi=np.quantile(divider_cloud,[.01,.99],axis=0)
        # Measured high surfaces determine the static divider / crossbar; its
        # unobserved connection to the floor is a continuity hypothesis.
        dt=t.copy();dt[:3,3]=np.r_[(lo[:2]+hi[:2])/2,(base+hi[2])/2]@r.T
        parts.append({'kind':'box','size_m':[float(hi[0]-lo[0]),float(hi[1]-lo[1]),float(hi[2]-base)],'T_W_part':dt.tolist()})
        assumptions=['rounded footprint fitted to depth samples; low outer wall and high middle structure are distinct','unseen wall/floor thickness and connection under the middle structure completed']
    else:raise ValueError('Unknown whole-object family')
    recipe={'family':family,'frame':'robot_base','units':'m','parts':parts,'assumptions':assumptions}
    if family=='rack':recipe=connect_rack(recipe)
    from .static_surface_constraints import conform_recipe
    return conform_recipe(recipe,xyz),r
