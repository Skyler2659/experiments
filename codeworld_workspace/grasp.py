"""Candidate-local gripper clearance query; not a grasp or force-closure planner."""
import numpy as np


def _positive(value, name):
    value=float(value)
    if not np.isfinite(value) or value<=0:raise ValueError(name+' must be positive and finite')
    return value


def _axis(value):
    a=np.asarray(value,float)
    if a.shape!=(3,) or not np.isfinite(a).all() or np.linalg.norm(a)<1e-9:raise ValueError('Expected finite nonzero axis')
    return a/np.linalg.norm(a)


def _clip(poly, axis, limit, sign):
    result=[]
    for p,q in zip(poly,poly[1:]+poly[:1]):
        dp=sign*p[axis]-limit;dq=sign*q[axis]-limit
        if dp<=0:result.append(p)
        if (dp<=0)!=(dq<=0):result.append(p+(q-p)*(dp/(dp-dq)))
    return result


def check_grasp_geometry(world, object_id, *, center_W, closing_axis_W,
                         approach_axis_W, max_opening_m, finger_depth_m,
                         finger_pad_height_m, margin_m=.002, expected_revision=None):
    """Clip actual mesh triangles to an explicit jaw engagement band.

    Axes and finger dimensions come from the agent/public gripper model. This
    answers enclosing jaw-width compatibility, not collision, friction or holding
    success. The band is unbounded along the closing axis: disconnected surfaces
    on the far side of a hollow object are included. INCOMPATIBLE therefore does
    not rule out a local rim/handle pinch, which needs its own contact-region
    analysis, finger collision check, and observed lift evidence.
    """
    state=world.state()
    if expected_revision is not None and state['revision']!=expected_revision:raise ValueError('Stale grasp candidate')
    opening=_positive(max_opening_m,'max_opening_m');depth=_positive(finger_depth_m,'finger_depth_m');height=_positive(finger_pad_height_m,'finger_pad_height_m')
    margin=float(margin_m)
    if not np.isfinite(margin) or margin<0 or 2*margin>=opening:raise ValueError('Invalid margin_m')
    center=np.asarray(center_W,float)
    if center.shape!=(3,) or not np.isfinite(center).all():raise ValueError('Invalid center_W')
    closing=_axis(closing_axis_W);approach=_axis(approach_axis_W)
    if abs(closing@approach)>1e-5:raise ValueError('Closing and approach axes must be orthogonal')
    lateral=np.cross(approach,closing);basis=np.column_stack([closing,approach,lateral])
    triangles=(np.asarray(world.mesh(object_id).triangles)-center)@basis
    # Most faces are remote from the local pad band. Keep intersecting faces,
    # including faces whose vertices straddle the band without lying inside it.
    keep=((triangles[:,:,1].min(1)<=depth/2)&(triangles[:,:,1].max(1)>=-depth/2)&
          (triangles[:,:,2].min(1)<=height/2)&(triangles[:,:,2].max(1)>=-height/2))
    points=[]
    for triangle in triangles[keep]:
        poly=list(triangle)
        for axis,limit in [(1,depth/2),(2,height/2)]:
            for sign in [1,-1]:
                if poly:poly=_clip(poly,axis,limit,sign)
        points.extend(poly)
    result={'schema':'codeworld-grasp-width-v1','object_id':object_id,'state_revision':state['revision'],
            'geometry_revision':state.get('geometry_revision',world.manifest.get('geometry_revision')),
            'center_W_m':center.tolist(),'closing_axis_W':closing.tolist(),'approach_axis_W':approach.tolist(),
            'max_opening_m':opening,'margin_m':margin,'finger_depth_m':depth,'finger_pad_height_m':height,
            'scope':'enclosing_cross_section_unbounded_along_closing_axis',
            'local_patch_grasp_evaluated':False,
            'holding_verified':False,'collision_checked':False,'force_closure_checked':False}
    if world.state()['revision']!=state['revision']:raise ValueError('World changed while checking grasp')
    if not points:return {**result,'status':'INCONCLUSIVE','reason':'No target surface intersects the declared finger pad band'}
    x=np.asarray(points)[:,0];lo=float(x.min());hi=float(x.max());span=hi-lo
    required=2*max(abs(lo),abs(hi))+2*margin
    fits=required<=opening
    centered=lo<0<hi
    return {**result,'status':'WIDTH_COMPATIBLE' if fits and centered else 'INCOMPATIBLE',
            'reason':'Candidate-local width fits; holding still requires sensor evidence' if fits and centered else 'Off-center or wider than the available jaw opening',
            'span_m':span,'required_centered_opening_m':required,'closing_extent_relative_center_m':[lo,hi],
            'jaw_clearance_m':opening-required,'surfaces_on_both_sides':centered}
