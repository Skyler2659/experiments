"""Snapshot geometry for agent-written candidate generation and route comparison."""
import copy
from pathlib import Path
import numpy as np


def segment_hits_box(a, b, bounds):
    """Continuous segment/AABB test; no sampling gaps."""
    lo, hi = 0., 1.
    for x, delta, lower, upper in zip(a, b-a, bounds[0], bounds[1]):
        if abs(delta) < 1e-12:
            if x < lower or x > upper: return False
        else:
            u, v = sorted(((lower-x)/delta, (upper-x)/delta))
            lo, hi = max(lo, u), min(hi, v)
            if lo > hi: return False
    return True


class PlanningContext:
    """One consistent world snapshot. Agent owns candidates, tool model and task policy.

    AABB route comparison is conservative tool-envelope screening, not arm IK,
    contact feasibility or a certificate for unknown geometry.
    """
    def __init__(self, world):
        self.world = world
        self.snapshot = copy.deepcopy(world.state())
        self.queries = []
        self.cache = {}

    def bounds(self, object_id):
        if object_id not in self.cache:
            self.cache[object_id] = self.world._mesh(object_id, self.snapshot).bounds
        value = self.cache[object_id].copy()
        self.queries.append({'op':'bounds', 'object_id':object_id, 'bounds_W_m':value.tolist()})
        return value

    def point(self, object_id, fraction, offset_W=(0,0,0)):
        """Point relative to current world AABB: fraction 0=lower, 1=upper.

        This is an envelope anchor, not necessarily a point on the mesh surface.
        For contact candidates use mesh/surface geometry and a tool/contact model.
        """
        b=self.bounds(object_id);f=np.asarray(fraction,float);o=np.asarray(offset_W,float)
        if f.shape!=(3,) or o.shape!=(3,) or not np.isfinite([f,o]).all():raise ValueError('Invalid anchor')
        p=b[0]+f*(b[1]-b[0])+o
        self.queries.append({'op':'bounds_anchor','object_id':object_id,'fraction':f.tolist(),
                             'offset_W_m':o.tolist(),'result_W_m':p.tolist()})
        return p

    def local_point(self, object_id, point_object, part_id=None):
        """Transform an agent-selected local feature using current object/part state."""
        row=self.snapshot['objects'][object_id];T=np.asarray(row['T_W_object'],float)
        if part_id is not None:T=T@np.asarray(row['assembly']['parts'][part_id]['T_object_part'],float)
        p=np.asarray(point_object,float)
        if p.shape!=(3,) or not np.isfinite(p).all():raise ValueError('Invalid local point')
        result=T[:3,:3]@p+T[:3,3]
        self.queries.append({'op':'local_point','object_id':object_id,'part_id':part_id,
                             'point_object_m':p.tolist(),'result_W_m':result.tolist()})
        return result

    def mesh(self, object_id):
        mesh=self.world._mesh(object_id,self.snapshot)
        self.queries.append({'op':'mesh','object_id':object_id,'vertices':len(mesh.vertices),'faces':len(mesh.faces)})
        return mesh

    def compare_routes(self, candidates, *, tool_half_extents_W, exclude=None):
        """Rank agent-proposed TCP polylines by obstacle overlaps then path length.

        Each candidate is {name, points_W_m, rationale}. Exclusions map object IDs
        to explicit reasons (e.g. target contact or held object represented by tool).
        Tool envelope must cover all orientations and any carried object along route.
        Unknown/pending state is exposed, never silently certified by this screening.
        """
        from .world import read
        half=np.asarray(tool_half_extents_W,float)
        if half.shape!=(3,) or not np.isfinite(half).all() or (half<=0).any():raise ValueError('Invalid tool envelope')
        exclude=dict(exclude or {})
        for oid,reason in exclude.items():
            if oid not in self.snapshot['objects'] or not isinstance(reason,str) or not reason.strip():raise ValueError('Invalid exclusion')
        obstacles={};review={}
        for oid,row in self.snapshot['objects'].items():
            metadata=read(self.world.root/self.world.manifest['objects'][oid])
            review[oid]={'aspects':copy.deepcopy(row.get('review',{'pose':{'status':'unreviewed'}})),
                         'source_unknowns':metadata.get('declared',{}).get('unknowns',[])}
            if metadata.get('role')=='robot_snapshot':
                exclude.setdefault(oid,'Static robot snapshot excluded; full arm collision is not evaluated')
            if oid not in exclude:obstacles[oid]=self.bounds(oid)
        results=[];names=set()
        for c in candidates:
            name=c['name'];points=np.asarray(c['points_W_m'],float)
            if not isinstance(name,str) or not name or name in names:raise ValueError('Unique candidate names required')
            names.add(name)
            if points.ndim!=2 or points.shape[1]!=3 or len(points)<2 or not np.isfinite(points).all():raise ValueError('Invalid polyline')
            hits=[]
            for oid,b in obstacles.items():
                expanded=np.array([b[0]-half,b[1]+half])
                segments=[i for i in range(len(points)-1) if segment_hits_box(points[i],points[i+1],expanded)]
                if segments:hits.append({'object_id':oid,'segments':segments})
            results.append({'name':name,'points_W_m':points.tolist(),'rationale':c.get('rationale'),
                'overlaps':hits,'path_length_m':float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum())})
        if not results:raise ValueError('No candidates')
        ranked=sorted(results,key=lambda c:(len(c['overlaps']),c['path_length_m'],c['name']))
        return {'schema':'codeworld-route-decision-v1','state_revision':self.snapshot['revision'],
                'geometry_revision':self.snapshot['geometry_revision'],'branch':self.world.branch,
                'queries':copy.deepcopy(self.queries),'candidates':results,
                'selected':ranked[0]['name'] if not ranked[0]['overlaps'] else None,
                'selection_rule':'Prefer zero conservative tool-envelope overlaps, then shortest polyline',
                'tool_half_extents_W_m':half.tolist(),'excluded_objects':exclude,'review':review,
                'limitations':['World AABBs can reject traversable cavities; use part/mesh-specific reasoning for insertion.',
                               'No full-arm collision, IK, contact, or unknown-surface certification.']}

    def save(self, decision, output):
        from .world import atomic
        if self.world.state()['revision']!=self.snapshot['revision']:raise ValueError('Stale planning snapshot')
        output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);atomic(output,decision)
        return output
