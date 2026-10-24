"""Reusable measured-detail repair: fit-only candidates, topology gate, heldout audit.

Does not advance task state, use task meshes, or certify unseen surfaces. Prior
world versions and rejected candidates remain intact.
"""
import json,shutil
from pathlib import Path
import numpy as np
from .mesh_world import MeshCodeWorld,sha256,generated_scene_code
from .quality_reconstruction import fuse,write,evaluate,assess_instance_regions,validate_capture
from .surface_quality import clean,topology,detail_gate,render_detail_views,deduplicate_small


def _mesh(om):
    import trimesh
    return trimesh.Trimesh(np.asarray(om.vertices),np.asarray(om.triangles),vertex_colors=np.column_stack([
        np.clip(np.asarray(om.vertex_colors)*255,0,255),np.full(len(om.vertices),255)]).astype('uint8'),process=False)


def run(config):
    config=Path(config).resolve();cfg=json.loads(config.read_text());out=Path(cfg['output']).resolve();out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(config,out/'config.json');base=MeshCodeWorld.load(cfg['base_scene'])
    shutil.copytree(base.root,out/'world');scene=json.loads((out/'world/scene.json').read_text());scene['world_version']=scene.get('world_version',1)+1
    reports={};evaluations={};pending=[]
    oldcap=Path(cfg['base_capture']);olddata=json.loads(oldcap.read_text())
    from .reconstruction_robot import prepare,attach
    # Validate the robot handoff before spending time on surface reconstruction.
    robot_binding=prepare(olddata,oldcap.parent,cfg.get('robot')) if not scene.get('robot_integration') else None
    write(out/'status.json',{'status':'RUNNING','stage':'detail_repair'})
    for node in scene['nodes']:
        # Robot FK is measured context, not a surface-generation repair target.
        if node.get('role')=='known_robot' or node['asset'].get('provenance',{}).get('source')=='public robosuite Panda only':continue
        oid=node['object_id'];folder=out/'objects'/oid;folder.mkdir(parents=True);oldmesh=base.mesh(oid)
        write(folder/'before_topology.json',topology(oldmesh)[0])
        spec=cfg.get('objects',{}).get(oid)
        if spec:
            cap=Path(spec['capture'])
            if not cap.exists() and spec.get('capture_command'):
                import subprocess
                command=spec['capture_command']
                if not isinstance(command,list) or not all(isinstance(x,str) for x in command):raise ValueError('capture_command must be an argv list')
                with (folder/'capture.log').open('w') as log:subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
            data=json.loads(cap.read_text());validate_capture(data,cap.parent)
            bounds=np.asarray(data['object_bounds_from_masked_depth'][oid])+np.array([[-.003,-.003,0],[.003,.003,.003]])
            voxel=spec.get('voxel_m',.0005)
            om,counts=fuse(data,cap.parent,bounds,voxel,truncation_voxels=3)
            mesh=_mesh(om);mesh.export(folder/'tsdf_raw.glb');mesh,audit=clean(mesh,data,cap.parent,tolerance_m=2*voxel)
            mesh.export(folder/'tsdf_clean.glb');write(folder/'tsdf_audit.json',audit)
            backend='tsdf';sheet_report=None
            if detail_gate(audit['after'])['status']=='DETAIL_REPAIR_REQUIRED':
                from .measured_sheets import reconstruct
                candidate,sheet_report,residual=reconstruct(data,cap.parent,bounds,**spec.get('sheet_options',{}))
                candidate.export(folder/'sheets_raw.glb');np.save(folder/'unmodelled_points.npy',residual);write(folder/'measured_sheets.json',sheet_report)
                candidate,sheet_audit=clean(candidate,data,cap.parent,tolerance_m=2*voxel);candidate.export(folder/'sheets_clean.glb');write(folder/'sheets_audit.json',sheet_audit)
                # No holdout has been read when making this choice.
                if sheet_report['residual_fraction']<=spec.get('max_unmodelled_fraction',.05) and sheet_audit['after']['small_component_area_fraction']<audit['after']['small_component_area_fraction']:
                    mesh,audit,backend=candidate,sheet_audit,'measured_sheets'
            if detail_gate(audit['after'])['status']=='DETAIL_REPAIR_REQUIRED':
                from .projective_surface import reconstruct as projective
                candidate,projective_report=projective(data,cap.parent,bounds,**spec.get('projective_options',{}))
                candidate.export(folder/'projective_raw.glb')
                candidate,projective_audit=clean(candidate,data,cap.parent,tolerance_m=2*voxel)
                candidate,duplicates=deduplicate_small(candidate)
                projective_audit['duplicate_patches']=duplicates
                projective_audit['after']=duplicates['after']
                candidate.export(folder/'projective_clean.glb');write(folder/'projective_audit.json',projective_audit)
                fit_data={**data,'views':[{**v,'role':'holdout'} for v in data['views'] if v['role']=='fit' and v.get('surface_enabled',True)]}
                fit_rows=evaluate(candidate,fit_data,cap.parent,bounds)
                # Empty target views do not support or oppose the candidate.
                checked=[r for r in fit_rows if r['target_pixels']>=30]
                fit_ok=bool(checked) and all(r['coverage']>=.98 and r['p95_error_m'] is not None and r['p95_error_m']<=.001 for r in checked)
                write(folder/'projective_surface.json',{**projective_report,'fit_validation':fit_rows,'fit_gate_passed':fit_ok})
                if fit_ok and projective_audit['after']['small_component_area_fraction']<audit['after']['small_component_area_fraction']:
                    mesh,audit,backend=candidate,projective_audit,'measured_projective_surface'
            gate=detail_gate(audit['after']);write(folder/'candidate_choice.json',{'backend':backend,'selection':'fit observations and topology only','gate':gate})
            # Evaluate after selection, not as the candidate-selection objective.
            rows=evaluate(mesh,data,cap.parent,bounds)
            evaluations[oid]={'measured':rows,'previous':evaluate(oldmesh,data,cap.parent,bounds)}
            write(folder/'validation.json',evaluations[oid])
            if gate['status']=='DETAIL_REPAIR_REQUIRED':pending.append(oid)
            provenance={'source':'measured detail repair','backend':backend,'capture':str(cap),'capture_sha256':sha256(cap),
                'native_task_asset_used':False,'surface_type':'thin_sheet' if backend in {'measured_sheets','measured_projective_surface'} else 'measured_surface',
                'topology_ref':str((folder/'candidate_choice.json').relative_to(out)),
                'unmodelled_fraction':sheet_report['residual_fraction'] if backend=='measured_sheets' else None}
            # New observations, not old structure decisions, follow the new asset.
            for v in data['views']:
                vid=f'detail_{oid}_{v["view_id"]}';dest=out/'world/evidence'/f'{vid}.png';shutil.copy2(cap.parent/v['rgb_uri'],dest)
                scene.setdefault('observations',[]).append({**v,'view_id':vid,'rgb_ref':str(dest.relative_to(out/'world')),'mask_refs':{}})
        else:
            mesh,audit=clean(oldmesh,olddata,oldcap.parent,tolerance_m=.003)
            provenance={**node['asset'].get('provenance',{}),'detail_cleaning':'only contradicted tiny components; rest retained'}
        reports[oid]=audit;write(folder/'cleaning.json',audit)
        if len(mesh.faces)==len(oldmesh.faces) and not spec and audit['removed_faces']==0:
            continue
        provenance.pop('support_ref',None);provenance.pop('low_support_face_fraction',None)
        mesh.apply_transform(np.linalg.inv(base.object_transform(oid)))
        path=out/'world/assets'/f'{oid}_detail.glb';mesh.export(path)
        node['asset']={'mesh_ref':str(path.relative_to(out/'world')),'sha256':sha256(path),'up_axis':'Z','T_object_mesh':np.eye(4).tolist(),'provenance':provenance}
        node['state']='ESTIMATED';node.pop('fit',None)
    if not scene.get('robot_integration'):
        attach(scene,out/'world',robot_binding)
    write(out/'world/scene.json',scene);(out/'world/scene.py').write_text(generated_scene_code());world=MeshCodeWorld.load(out/'world/scene.json');world.export_glb(out/'world/scene.glb')
    # Make the exported viewer honour zero-thickness measured sheets from both sides.
    export_two_sided(world,out/'world/scene.glb')
    for node in scene['nodes']:render_detail_views(world,node['object_id'],out/'detail_views'/node['object_id'])
    world.render(scene['canonical_camera'],out/'canonical.png')
    acceptance=assess_instance_regions(evaluations,**cfg.get('acceptance',{'max_p95_m':.005,'min_coverage':.98}))
    write(out/'acceptance.json',acceptance);write(out/'detail_quality.json',reports)
    status={'status':'GEOMETRY_REPAIR_REQUIRED' if pending or acceptance['failures'] else 'DETAIL_VISUAL_REVIEW_REQUIRED',
        'detail_failures':pending,'world':str(out/'world/scene.json'),'scope':'new detail candidate; not a closed-solid or semantic-part acceptance'}
    from .scene_structuring import run as structure
    stage=structure(out/'world/scene.json',out/'structure',require_features=True);status['structure_status']=stage['status']
    write(out/'status.json',status);return status


def export_two_sided(world,path):
    import trimesh
    from trimesh.exchange.gltf import export_glb
    scene=trimesh.Scene()
    for n in world.source['nodes']:
        for i,piece in enumerate(world.mesh_pieces(n['object_id'])):scene.add_geometry(piece,node_name=f'{n["object_id"]}_{i}')
    def materials(tree):
        mats=tree.setdefault('materials',[])
        mats.append({'doubleSided':True,'pbrMetallicRoughness':{'baseColorFactor':[1,1,1,1],'metallicFactor':0,'roughnessFactor':1}})
        for mat in mats:mat['doubleSided']=True
        for mesh in tree.get('meshes',[]):
            for primitive in mesh['primitives']:primitive.setdefault('material',len(mats)-1)
    Path(path).write_bytes(export_glb(scene,tree_postprocessor=materials))
