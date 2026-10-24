"""Additional whole-body families fitted to RGB-D, without native task assets."""
import numpy as np


def loop_handle(width,height,tube,corner=.004):
    """Closed rounded rectangle tube in local XZ; opening remains empty."""
    import trimesh
    c=min(corner,width/3,height/3);path=[]
    for x,z,start in [(width/2-c,height/2-c,0),(-width/2+c,height/2-c,90),(-width/2+c,-height/2+c,180),(width/2-c,-height/2+c,270)]:
        a=np.deg2rad(np.linspace(start,start+90,16,endpoint=False))
        path.extend(np.c_[x+c*np.cos(a),np.zeros(len(a)),z+c*np.sin(a)])
    path=np.asarray(path);tangent=np.roll(path,-1,axis=0)-np.roll(path,1,axis=0);tangent/=np.linalg.norm(tangent,axis=1)[:,None]
    side=np.cross(tangent,[0,1,0]);theta=np.linspace(0,2*np.pi,12,endpoint=False)
    v=path[:,None,:]+tube*(side[:,None,:]*np.cos(theta)[None,:,None]+np.array([0,1,0])[None,None,:]*np.sin(theta)[None,:,None])
    faces=[];n=len(path)
    for i in range(n):
        for j in range(12):
            a=i*12+j;b=((i+1)%n)*12+j;cc=((i+1)%n)*12+(j+1)%12;d=i*12+(j+1)%12
            faces.extend([[a,b,cc],[a,cc,d]])
    m=trimesh.Trimesh(v.reshape(-1,3),faces,process=True);m.fix_normals()
    if m.volume<0:m.invert()
    return m


def carton_roof(width,depth,height):
    import trimesh
    v=np.array([[-width/2,-depth/2,0],[width/2,-depth/2,0],[width/2,depth/2,0],[-width/2,depth/2,0],[0,-depth/2,height],[0,depth/2,height]])
    m=trimesh.Trimesh(v,[[0,2,1],[0,3,2],[0,1,4],[3,5,2],[0,4,5],[0,5,3],[1,2,5],[1,5,4]],process=True)
    m.fix_normals();return m


def polygon_slab(outline,z,thickness):
    import trimesh
    xy=np.asarray(outline);n=len(xy);center=xy.mean(0)
    v=np.vstack([np.c_[xy,np.full(n,z)],np.c_[xy,np.full(n,z-thickness)],[list(center)+[z],list(center)+[z-thickness]]]);f=[]
    for i in range(n):
        j=(i+1)%n;f.extend([[i,j,n+j],[i,n+j,n+i],[2*n,j,i],[2*n+1,n+i,n+j]])
    m=trimesh.Trimesh(v,f,process=True);m.fix_normals();return m


def fit_panels(xyz,r,thickness=.0035):
    """Fit individual orthogonal material panels, retaining open compartments."""
    from scipy.spatial import cKDTree
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    import open3d as o
    p=np.unique(np.round(xyz@r,5),axis=0)
    _,idx=cKDTree(p).query(p,k=16);near=p[idx];center=near.mean(1);delta=near-center[:,None,:]
    cov=np.einsum('nki,nkj->nij',delta,delta)/16;values,vectors=np.linalg.eigh(cov);normal=vectors[:,:,0]
    valid=values[:,0]<1e-6;lo,hi=np.quantile(p,[.002,.998],axis=0);parts=[];planes={}
    for axis in range(3):
        s=p[valid&(abs(normal[:,axis])>.9)]
        bins=np.arange(lo[axis]-.004,hi[axis]+.005,.0008);hist,edges=np.histogram(s[:,axis],bins);smooth=gaussian_filter1d(hist.astype(float),.7)
        peaks,_=find_peaks(smooth,prominence=max(15,smooth.max()*.03),distance=4)
        groups=[]
        for peak in peaks:
            c=(edges[peak]+edges[peak+1])/2;points=s[abs(s[:,axis]-c)<.0016]
            if len(points)<80:continue
            c=float(np.median(points[:,axis]))
            if groups and c-groups[-1][0]<.006:
                old=groups.pop();points=np.vstack([old[1],points]);c=float(np.median(points[:,axis]))
            groups.append((c,points))
        planes[axis]=groups
    cuts=sorted(set([lo[0],hi[0]]+[c for c,_ in planes[0]]));cuts=[v for i,v in enumerate(cuts) if i==0 or v-cuts[i-1]>.008]
    if cuts[-1]<hi[0]-.008:cuts.append(hi[0])
    def add(points,axis,c):
        lower,upper=np.quantile(points,[.003,.997],axis=0);size=upper-lower
        other=[a for a in range(3) if a!=axis]
        if min(size[other])<.009:return
        size[other]+=thickness;size[axis]=thickness
        center=(lower+upper)/2;center[axis]=c;t=np.eye(4);t[:3,:3]=r;t[:3,3]=center@r.T
        parts.append({'kind':'box','size_m':size.tolist(),'T_W_part':t.tolist()})
    for axis in [0,1]:
        for c,points in planes[axis]:
            if axis==1:
                for a,b in zip(cuts,cuts[1:]):
                    q=points[(points[:,0]>=a-.002)&(points[:,0]<=b+.002)]
                    if len(q)>80:add(q,axis,c)
            else:add(points,axis,c)
    # A rim is a narrow ring of observations, not a filled horizontal lid.
    for c,points in planes[2]:
        pc=o.geometry.PointCloud(o.utility.Vector3dVector(points));labels=np.asarray(pc.cluster_dbscan(eps=.006,min_points=20))
        for label in set(labels)-{-1}:
            q=points[labels==label];a,b=np.quantile(q[:,:2],[.003,.997],axis=0);size=b-a
            if min(size)<.015:continue
            cells=np.unique(np.floor((q[:,:2]-a)/.003).astype(int),axis=0)
            coverage=len(cells)/max(1,np.prod(np.ceil(size/.003)))
            if coverage>.45:add(q,2,c)
    # Hidden underside joins measured panels, with explicit completion origin.
    t=np.eye(4);t[:3,:3]=r;t[:3,3]=np.r_[(lo[:2]+hi[:2])/2,lo[2]+thickness/2]@r.T
    parts.append({'kind':'box','size_m':[float(hi[0]-lo[0]+thickness),float(hi[1]-lo[1]+thickness),thickness],'T_W_part':t.tolist()})
    if len(parts)<5:raise ValueError('Insufficient observed panel structure')
    return parts


def caddy_prior(xyz,r,regions):
    """Continuous RGB-structure prior; bounded refinement fits actual surfaces."""
    local=xyz@r;lo,hi=np.quantile(local,[.003,.997],axis=0);origin=np.r_[(lo[:2]+hi[:2])/2,lo[2]];w,d,h=hi-lo;th=.0035;parts=[]
    def panel(size,center):
        t=np.eye(4);t[:3,:3]=r;t[:3,3]=(origin+center)@r.T
        parts.append({'kind':'box','size_m':list(map(float,size)),'T_W_part':t.tolist()})
    panel([w,d,th],[0,0,th/2]);groups=[]
    for key in ['caddy_left','caddy_middle','caddy_right']:
        p=regions[key][0]@r-origin;groups.append((np.median(p[:,0]),p))
    groups.sort(key=lambda a:a[0]);heights=[float(np.quantile(p[:,2],.99)) for _,p in groups]
    # Region medians shift with occlusion. Locate actual vertical boards near
    # the RGB three-column prior using interior depth-plane peaks instead.
    local=local-origin;interior=local[abs(local[:,1])<d/2-.012]
    edges=np.arange(-w/2-.002,w/2+.003,.0007);counts,_=np.histogram(interior[:,0],edges);centers=(edges[:-1]+edges[1:])/2
    cuts=[]
    for expected in np.linspace(-w/2,w/2,4):
        candidate=np.flatnonzero(abs(centers-expected)<.018)
        cuts.append(float(centers[candidate[np.argmax(counts[candidate])]]) if len(candidate) and counts[candidate].max()>50 else float(expected))
    for i,(a,b) in enumerate(zip(cuts,cuts[1:])):
        height=heights[i]
        strip=local[(local[:,0]>a+.005)&(local[:,0]<b-.005)&(local[:,1]>d/2-.008)]
        front=float(np.quantile(strip[:,2],.99)) if len(strip)>50 else (height if i!=1 else height*.52)
        front=float(np.clip(front,th,height))
        panel([b-a+th,th,height],[(a+b)/2,-d/2+th/2,height/2]);panel([b-a+th,th,front],[(a+b)/2,d/2-th/2,front/2])
    for i,x in enumerate(cuts):
        height=heights[0] if i==0 else heights[-1] if i==3 else max(heights[i-1:i+1]);panel([th,d,height],[x,0,height/2])
    mid=(cuts[1]+cuts[2])/2;panel([cuts[2]-cuts[1],th,heights[1]*.55],[mid,0,heights[1]*.275])
    # Visible compartments can have raised floors; a single bottom slab is
    # insufficient. Horizontal samples set each interior support height.
    candidates=[]
    for p in fit_panels(xyz,r):
        if p['size_m'][2]>th*1.1:continue
        center=np.asarray(p['T_W_part'])[:3,3]@r-origin
        if center[2]<.008:continue
        candidates.append((center,p['size_m'][0]*p['size_m'][1]))
    for i,(a,b) in enumerate(zip(cuts,cuts[1:])):
        ys=[(-d/2,d/2)] if i!=1 else [(-d/2,0),(0,d/2)]
        for y0,y1 in ys:
            matches=[(c,area) for c,area in candidates if a-.005<c[0]<b+.005 and y0-.005<c[1]<y1+.005 and c[2]<heights[i]*.8]
            if matches:
                c,_=max(matches,key=lambda q:q[1]);panel([b-a+th,y1-y0+th,th],[(a+b)/2,(y0+y1)/2,c[2]])
    return parts


def fit(family,xyz,rgb,regions):
    from .static_asset_geometry import box_part,rectangle_frame
    from .continuous_rigid_reconstruction import fit_profile
    parts=[];assumptions=[]
    if family=='convex':
        import trimesh
        lo,hi=np.quantile(xyz,[.002,.998],axis=0);p=xyz[np.all(xyz>=lo-.001,1)&np.all(xyz<=hi+.001,1)]
        hull=trimesh.convex.convex_hull(p);parts=[{'kind':'convex','vertices_W_m':hull.vertices.tolist()}];r=np.eye(3)
        assumptions=['closed convex whole-solid envelope fitted to masked metric samples; unseen surfaces and small page gaps completed','only suitable for whole solid objects without task-relevant cavity/handle openings']
    elif family in {'box','book','carton'}:
        part,r=box_part(xyz)
        if family=='book':
            import trimesh
            transform,size=trimesh.bounds.oriented_bounds(xyz,angle_digits=2)
            t=np.linalg.inv(transform);part={'kind':'box','size_m':size.tolist(),'T_W_part':t.tolist()};r=t[:3,:3]
        if family=='carton':
            t=np.asarray(part['T_W_part']);size=np.asarray(part['size_m']);roof=float(size[2]*.07)
            body=t.copy();body[2,3]-=roof/2
            part.update(size_m=[float(size[0]),float(size[1]),float(size[2]-roof+.001)],T_W_part=body.tolist())
            rt=t.copy();rt[2,3]+=size[2]/2-roof-.0005
            parts.append({'kind':'carton_roof','dimensions':{'width':float(size[0]),'depth':float(size[1]),'height':roof+.0005},'T_W_part':rt.tolist()})
            assumptions.append('gable fold is a whole-carton hypothesis; roof height is estimated from observed upper envelope')
        parts.append(part);assumptions.append('closed rectangular body from FIT depth; hidden faces completed')
    elif family=='can':
        z0,z1=np.quantile(xyz[:,2],[.002,.998]);side=xyz[(xyz[:,2]>z0+.1*(z1-z0))&(xyz[:,2]<z1-.1*(z1-z0))];xy=side[:,:2]
        from scipy.optimize import least_squares
        center=(np.quantile(xy,.01,axis=0)+np.quantile(xy,.99,axis=0))/2
        init=np.r_[center,np.median(np.linalg.norm(xy-center,axis=1))]
        q=least_squares(lambda a:np.linalg.norm(xy-a[:2],axis=1)-a[2],init,loss='soft_l1',f_scale=.001).x
        t=np.eye(4);t[:3,3]=np.r_[q[:2],(z0+z1)/2];r=np.eye(3)
        parts=[{'kind':'cylinder','radius_m':float(q[2]),'height_m':float(z1-z0),'T_W_part':t.tolist()}]
        assumptions=['upright cylindrical can fitted to measured side surface; unobserved top/bottom completed']
    elif family=='mug':
        body=regions['mug_body'][0];handle=regions['mug_handle'][0]
        profile,origin,assumptions=fit_profile(body,'bowl');t=np.eye(4);t[:3,3]=origin
        parts=[{'kind':'lathe','profile_radius_z_m':profile.tolist(),'T_W_part':t.tolist()}]
        u=np.r_[handle[:,:2].mean(0)-origin[:2],0.];u/=np.linalg.norm(u);v=np.cross([0,0,1],u);r=np.column_stack([u,v,[0,0,1]])
        local=(handle-origin)@r;lo,hi=np.quantile(local,[.01,.99],axis=0)
        radius=float(np.median(profile[profile[:,0]>0,0]));tube=float(np.clip((hi[1]-lo[1])/2,.002,.004))
        # Extend only the occluded attachment into the fitted body wall.
        inner=radius-.004;outer=hi[0]-tube;center=(inner+outer)/2
        ht=np.eye(4);ht[:3,:3]=r;ht[:3,3]=origin+r@np.array([center,0,(lo[2]+hi[2])/2])
        parts.append({'kind':'loop_handle','dimensions':{'width':float(outer-inner),'height':float(hi[2]-lo[2]-2*tube),'tube':tube,'corner':.006},'T_W_part':ht.tolist()})
        assumptions+=['handle frame and opening fitted to separate measured region; hidden mounts completed into body']
    elif family=='caddy':
        part,r=box_part(xyz);size=np.asarray(part['size_m'])
        if size[0]<size[1]:r=np.column_stack([r[:,1],-r[:,0],r[:,2]])
        # The middle front wall is lower than the rear wall. Infer the sign
        # from measured middle-column heights, rather than a camera/world axis.
        p=xyz@r;middle=regions['caddy_middle'][0]@r;ulo,uhi=np.quantile(middle[:,0],[.01,.99]);vlo,vhi=np.quantile(p[:,1],[.003,.997])
        central=p[(p[:,0]>ulo+.004)&(p[:,0]<uhi-.004)]
        rear=central[central[:,1]<vlo+.006];front=central[central[:,1]>vhi-.006]
        if len(rear)>30 and len(front)>30 and np.quantile(front[:,2],.95)>np.quantile(rear[:,2],.95)+.01:r=np.column_stack([-r[:,0],-r[:,1],r[:,2]])
        parts=caddy_prior(xyz,r,regions)
        assumptions=['RGB three-column/middle-divider structure with measured metric envelope; this is a shape prior, not accurate visible-panel fitting','unobserved bottom, panel thickness and short overlaps completed; refine inaccurate visible spans from RGB-D']
    elif family=='basket':
        part,r=box_part(xyz);t=np.asarray(part['T_W_part']);size=np.asarray(part['size_m']);t[2,3]-=size[2]/2
        w,d,h=size;c=min(w,d)*.07;outline=[]
        for x,y,start in [(w/2-c,d/2-c,0),(-w/2+c,d/2-c,90),(-w/2+c,-d/2+c,180),(w/2-c,-d/2+c,270)]:
            a=np.deg2rad(np.linspace(start,start+90,24,endpoint=False));outline.extend(np.c_[x+c*np.cos(a),y+c*np.sin(a)])
        parts=[{'kind':'outline_shell','outline_xy_m':np.asarray(outline).tolist(),'dimensions':{'height':float(h),'wall':.006,'floor':.004},'T_W_part':t.tolist()}]
        assumptions=['open rounded rectangular basket fitted to measured envelope; weave/cloth appearance retained by RGB texture','unobserved wall/floor thickness completed; detailed cloth folds are approximated']
    else:raise ValueError(family)
    return {'family':family,'frame':'robot_base','units':'m','parts':parts,'assumptions':assumptions},r
