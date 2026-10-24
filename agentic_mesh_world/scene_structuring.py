"""Mandatory scene-wide second stage after reviewed mesh reconstruction.

Every node gets an executable, exact-mesh descriptor. Atomic vs decomposable
is an explicit decision, never inferred from object-name substrings.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from .mesh_world import MeshCodeWorld, sha256
from .structure_program import freeze, verify_freeze, write


def object_preview(world,oid,path):
    import os
    os.environ.setdefault('PYOPENGL_PLATFORM','egl')
    import pyrender
    from PIL import Image
    pieces=world.mesh_pieces(oid);mesh=world.mesh(oid);center=mesh.bounds.mean(0);size=float(max(mesh.extents))
    pos=center+np.array([1.4,-1.7,1.4])*size
    z=(pos-center)/np.linalg.norm(pos-center);x=np.cross([0,0,1],z);x/=np.linalg.norm(x)
    pose=np.eye(4);pose[:3,:3]=np.column_stack([x,np.cross(z,x),z]);pose[:3,3]=pos
    scene=pyrender.Scene(bg_color=[240,242,245,255],ambient_light=[.6,.6,.6])
    for piece in pieces:
        rendered=pyrender.Mesh.from_trimesh(piece,smooth=False)
        if world.node(oid)['asset'].get('provenance',{}).get('surface_type')=='thin_sheet':
            for primitive in rendered.primitives:primitive.material.doubleSided=True
        scene.add(rendered)
    scene.add(pyrender.PerspectiveCamera(yfov=np.deg2rad(40)),pose=pose)
    scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=2),pose=pose)
    renderer=pyrender.OffscreenRenderer(512,512)
    try:rgb,_=renderer.render(scene)
    finally:renderer.delete()
    Image.fromarray(rgb).save(path)


def describe(world, oid):
    node=world.node(oid);pieces=world.mesh_pieces(oid);mesh=world.mesh(oid)
    from .structure_features import object_frame
    frame = object_frame(world, oid); intrinsic = mesh.copy(); intrinsic.apply_transform(np.linalg.inv(frame))
    materials=[]
    for i,piece in enumerate(pieces):
        visual=piece.visual;entry={'primitive':i,'visual_kind':visual.kind}
        if visual.kind=='texture':
            material=visual.material;img=getattr(material,'baseColorTexture',None)
            if img is None:img=getattr(material,'image',None)
            factor=getattr(material,'baseColorFactor',None)
            entry['base_color_factor']=np.asarray(factor).tolist() if factor is not None else None
            if img is not None:
                pixels=np.asarray(img.convert('RGBA'));rgb=pixels[...,:3]
                entry.update(texture_size=list(img.size),texture_embedded_in_mesh=True,
                             texture_mean_srgb=rgb.reshape(-1,3).mean(axis=0).tolist(),
                             color_authority='per-surface UV texture and material in mesh; mean is only a summary')
            entry['uv_count']=len(visual.uv) if visual.uv is not None else 0
        else:
            colors=np.asarray(visual.vertex_colors)
            entry.update(mean_rgba=colors.mean(axis=0).tolist(),color_authority='mesh vertex/face colors')
        materials.append(entry)
    return {'object_id':oid,'parent_id':node.get('parent_id'),
            'children':[n['object_id'] for n in world.source['nodes'] if n.get('parent_id')==oid],
            'geometry':{'representation':'full_textured_triangle_mesh','asset':node['asset'],
                        'faces':len(mesh.faces),'vertices':len(mesh.vertices),'surface_area_m2':float(mesh.area),
                        'watertight':bool(mesh.is_watertight),'bounds_W_m':mesh.bounds.tolist(),
                        'size_W_aabb_m':mesh.extents.tolist(),
                        'intrinsic_frame_W':frame.tolist(),'intrinsic_size_m':intrinsic.extents.tolist(),
                        'note':'mesh retains rim, cavity, curvature and underside where reconstructed; no primitive substitution'},
            'placement':{'T_W_object':world.object_transform(oid).tolist(),'pose_parent_object':node['pose_parent_object'],
                         'joint':node.get('joint'),'units':'m','frame':'robot_base'},
            'appearance':materials,'regions':node.get('regions',[]),
            'evidence':{'view_ids':node.get('evidence_view_ids',[]),'state':node.get('state','UNKNOWN'),
                        'fit':node.get('fit'), 'precision':'exact representation of reconstructed candidate; real-world accuracy unverified'}}


def existing_public_assembly(world, oid):
    """Use retained known-robot provenance, never a task-object asset catalog."""
    node=world.node(oid);provenance=node['asset'].get('provenance') or {}
    if provenance.get('source')!='public robosuite Panda only':return None
    groups={};cursor=0
    for geometry in provenance.get('geometries',[]):
        if not geometry.get('body_name') or 'source_face_range' not in geometry:return None
        start,stop=geometry['source_face_range']
        if start!=cursor or stop<=start:raise ValueError('public assembly face provenance is not contiguous')
        cursor=stop;groups.setdefault(geometry['body_name'],[]).append([start,stop])
    if cursor!=len(world.mesh(oid).faces):raise ValueError('public assembly provenance does not cover the frozen asset')
    return {'source':'known public robot assembly metadata bound to frozen asset',
            'frozen_asset_sha256':node['asset']['sha256'],
            'preferred_tree':{'id':oid,'semantic':'known robot assembly at measured state','children':[
                {'id':name,'semantic':'public robot body '+name,
                 'selector':{'kind':'source_face_ranges','ranges':ranges},
                 'evidence':['Existing public robot link membership; not discovered task geometry.']}
                for name,ranges in groups.items()]},
            'instruction':'Preserve this existing source membership instead of rediscovering known links with spatial boxes.'}


def run(scene, output, routes=None, *, require_features=False, development_source=False):
    scene=Path(scene).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    source=json.loads(scene.read_text())
    quality=source.get('quality',{})
    source_accepted=quality.get('reconstruction_acceptance') not in {'PENDING','REJECTED','GEOMETRY_REVIEW_REQUIRED'} and quality.get('metric_contact_ready') is not False
    if not source_accepted and not development_source:
        raise ValueError('Independent source geometry review required')
    if source.get('geometry_validation') and source['geometry_validation'].get('independent_acceptance') is not True:
        source_accepted=False
        if not development_source:raise ValueError('Independent source geometry review required')
    if source.get('quality',{}).get('requires_instance_acceptance'):
        receipt=scene.parent.parent/'acceptance.json'
        if not receipt.exists() or json.loads(receipt.read_text()).get('scene_sha256')!=sha256(scene) or json.loads(receipt.read_text()).get('status')!='GEOMETRY_AND_IDENTITY_ACCEPTED':
            source_accepted=False
            if not development_source:
                raise ValueError('INSTANCE_ACCEPTANCE_REQUIRED before structural code generation')
    frozen=output/'frozen'
    if not frozen.exists():freeze(scene,frozen)
    verify_freeze(frozen)
    if sha256(scene)!=sha256(frozen/'scene.json'):raise ValueError('stage-one scene changed: use a new stage-two output')
    previous_path=output/'status.json'
    if previous_path.exists():
        previous=json.loads(previous_path.read_text())
        if previous.get('functional_geometry_required') and previous.get('status')=='SCENE_STRUCTURED':
            from .functional_geometry import FEATURE_COMPILER_VERSION
            if previous.get('functional_compiler_version')!=FEATURE_COMPILER_VERSION:
                raise ValueError('functional implementation changed: create a new stage-two package')
            for path,digest in previous['functional_bindings'].items():
                if sha256(output/path)!=digest:raise ValueError('published functional package changed')
            return previous
    world=MeshCodeWorld.load(frozen/'scene.json');routes=routes or {};rows=[]
    for node in world.source['nodes']:
        oid=node['object_id'];folder=output/'objects'/oid;folder.mkdir(parents=True,exist_ok=True)
        desc=describe(world,oid);write(folder/'object.json',desc)
        (folder/'object.py').write_text(
            '# Full geometry, placement and appearance from one frozen authority.\n'
            'from pathlib import Path\nfrom agentic_mesh_world.mesh_world import MeshCodeWorld\n'
            'from agentic_mesh_world.scene_structuring import describe\n'
            'WORLD = MeshCodeWorld.load(Path(__file__).resolve().parents[2]/"frozen/scene.json")\n'
            f'OBJECT_ID = {oid!r}\n'
            'def geometry(): return WORLD.mesh_pieces(OBJECT_ID)\n'
            'def properties(): return describe(WORLD, OBJECT_ID)\n'
            'def surface_distance(point_W): return WORLD.surface_distance(OBJECT_ID, point_W)\n')
        decision_path=folder/'decision.json'
        if oid in routes and not decision_path.exists():write(decision_path,routes[oid])
        decision=json.loads(decision_path.read_text()) if decision_path.exists() else None
        if not decision:
            status='NEED_STRUCTURE_DECISION'
            if not (folder/'asset.png').exists():object_preview(world,oid,folder/'asset.png')
            write(folder/'request.json',{'kind':'classify','object_id':oid,'descriptor':'object.json',
                'options':['atomic','compound','existing_assembly','uncertain'],
                'instruction':'Inspect asset and observations. Atomic keeps the full shape, cavity and texture. Do not decide by ID alone.'})
        else:
            if not decision.get('reason') or decision.get('kind') not in {'atomic','compound','existing_assembly','uncertain'}:
                raise ValueError(f'invalid structure decision: {oid}')
            kind=decision['kind']
            if kind=='atomic':status='ATOMIC_STRUCTURED'
            elif kind=='existing_assembly':
                status='ASSEMBLY_STRUCTURED' if desc['children'] else 'NEED_STRUCTURE_DECISION'
            elif kind=='compound':
                from .structure_workflow import run as run_parts
                if not (folder/'asset.png').exists():object_preview(world,oid,folder/'asset.png')
                case=folder/'parts_workflow';case.mkdir(exist_ok=True)
                if not (case/'frozen').exists():(case/'frozen').symlink_to(os.path.relpath(frozen,case),target_is_directory=True)
                if not (case/'stage_one_hints.json').exists():
                    known=existing_public_assembly(world,oid)
                    if known:write(case/'stage_one_hints.json',known)
                (case/'proposal_prompt.txt').write_text(PROPOSAL_PROMPT.replace('{object_id}',oid))
                try:
                    result=run_parts(case,target_object_id=oid);status=result['status']
                except Exception as exc:
                    status='STRUCTURE_BLOCKED'
                    write(folder/'error.json',{'status':status,'reason':str(exc),'candidate_preserved':True})
            else:status='NEED_OBSERVATION'
        program=f'objects/{oid}/object.py'
        if status=='STRUCTURALIZED':program=f'objects/{oid}/parts_workflow/rounds/{result["round"]:02d}/structure.py'
        rows.append({'object_id':oid,'status':status,'descriptor':f'objects/{oid}/object.json',
            'program':program,'decision':decision})
    complete=all(r['status'] in {'ATOMIC_STRUCTURED','ASSEMBLY_STRUCTURED','STRUCTURALIZED'} for r in rows)
    result={'schema_version':'scene-structure-batch-v1','status':'SCENE_STRUCTURED' if complete else 'STRUCTURING_PENDING',
            'source_formally_accepted':source_accepted,'development_source':bool(development_source),
            'objects':rows,'coverage':{'total':len(rows),'complete':sum(r['status'] in {'ATOMIC_STRUCTURED','ASSEMBLY_STRUCTURED','STRUCTURALIZED'} for r in rows)},
            'scope':'all scene nodes enumerated; physical precision inherited from stage one'}
    write(output/'status.json',result);write(output/'agent_scene.json',result)
    (output/'scene.py').write_text('from pathlib import Path\nfrom agentic_mesh_world.scene_structuring import StructuredScene\nWORLD = StructuredScene(Path(__file__).parent)\n')
    if complete and require_features:
        from .functional_geometry import run_object
        structured = StructuredScene(output)
        features = [run_object(structured, n['object_id'], output/'objects'/n['object_id']/'functional') for n in world.source['nodes']]
        result['functional_geometry'] = features
        result['functional_geometry_required'] = True
        if any(row['status'] != 'FEATURES_REVIEWED' for row in features):
            result['status'] = 'FUNCTIONAL_GEOMETRY_PENDING'
        else:
            from .functional_geometry import FEATURE_COMPILER_VERSION
            result['functional_compiler_version']=FEATURE_COMPILER_VERSION
            result['functional_bindings']={str(path.relative_to(output)):sha256(path) for n in world.source['nodes']
                for path in (output/'objects'/n['object_id']/'functional/accepted.json',output/'objects'/n['object_id']/'functional/requirements.json')}
        write(output/'status.json', result); write(output/'agent_scene.json', result)
    return result


class StructuredScene(MeshCodeWorld):
    """Same all-object geometry for Agent queries and rendering."""
    def __init__(self,root):
        from .structure_program import StructuralWorld
        self.package=Path(root).resolve();verify_freeze(self.package/'frozen')
        base=MeshCodeWorld.load(self.package/'frozen/scene.json');super().__init__(base.source,base.root)
        self.catalog=json.loads((self.package/'agent_scene.json').read_text())
        self.structured={r['object_id']:StructuralWorld((self.package/r['program']).parent) for r in self.catalog['objects'] if r['status']=='STRUCTURALIZED'}
        self.functional = {}
        for row in self.catalog['objects']:
            path = self.package/'objects'/row['object_id']/'functional/accepted.json'
            if path.exists():
                data = json.loads(path.read_text())
                if data['asset_sha256'] != self.node(row['object_id'])['asset']['sha256']:
                    raise ValueError('functional geometry belongs to another asset')
                self.functional[row['object_id']] = data

    def mesh_pieces(self,object_id):
        if object_id in self.structured:return self.structured[object_id].mesh_pieces(object_id)
        return super().mesh_pieces(object_id)

    def object_info(self,object_id):
        import copy
        from .functional_geometry import feature_context
        row=next(r for r in self.catalog['objects'] if r['object_id']==object_id)
        from .reconstruction_quality import object_quality
        return {'structuring_status':row['status'],**describe(self,object_id),
                'quality': object_quality(self.node(object_id)),
                'functional_geometry': feature_context(self.functional.get(object_id, {'state': 'NOT_EXTRACTED'}))}

    def feature_info(self, object_id, feature_id):
        from .functional_geometry import feature_context
        for feature in self.functional.get(object_id, {}).get('features', []):
            if feature['id'] == feature_id:
                return feature_context({'features':[feature]})['features'][0]
        raise ValueError('unknown functional feature')

    def part_info(self,object_id,part_id):
        if object_id not in self.structured:raise ValueError('object has no accepted internal part program')
        return self.structured[object_id].part_info(part_id)

    def planner(self):
        """Ground Macro/Micro on this accepted package, including its parts."""
        from .mesh_planner import MeshPlanner
        return MeshPlanner(self.package)


PROPOSAL_PROMPT='''Propose a recursive surface partition for object {object_id} from the frozen asset and observations. Do not call tools or other agents. Return proposal_json string and reason. Schema: {object_id,tree:{id,semantic,evidence:[strings],children:[nodes]},relations:[],limitations:[strings]}. Prefer object_box for walls/covers/dividers using supplied object-frame surface peaks; separate meaningful internal dividers rather than hiding them in a broad remainder. Each child has selector {kind:"box",low:[x,y,z],high:[x,y,z]} normalized in PARENT world-axis bounds; or {kind:"object_box",low:[x,y,z],high:[x,y,z]} normalized in PARENT bounds in the supplied upright object frame; or {kind:"plane_band",height_band:[low,high],half_thickness_m:number}; or {kind:"remaining"}. When supplied with geometry-verified original assembly provenance, selector {kind:"source_face_ranges",ranges:[[start,exclusive_end],...]} can retain those parts exactly; never invent index ranges. Children are processed in order and are disjoint. Unassigned triangles remain unresolved. Relations support {kind:"above",a,b,minimum_m} or {kind:"near",a,b,tolerance_m}. Use geometry profiles to locate parts, not screen-axis guesses. Preserve uncertain geometry, source texture, and part connections. Refine only observed meaningful parts; no physics or motion laws.'''


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',required=True);p.add_argument('--output',required=True);p.add_argument('--routes')
    p.add_argument('--require-features',action='store_true')
    a=p.parse_args();r=run(a.scene,a.output,json.loads(Path(a.routes).read_text()) if a.routes else None,require_features=a.require_features)
    print(json.dumps({'status':r['status'],'coverage':r['coverage']}))
