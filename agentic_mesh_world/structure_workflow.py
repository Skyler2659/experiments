"""Resumable stage-two loop: propose, execute, inspect, revise, publish.

Proposal/review are explicit model or human handoffs. Rejected rounds remain
on disk. No stage-one generators, poses or cameras are rewritten.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from .structure_program import (StructuralWorld, compile_program, verify_program,
                                publish_agent_bundle, verify_freeze, write, sha256)
from .mesh_world import MeshCodeWorld


def previews(root):
    import os
    os.environ.setdefault('PYOPENGL_PLATFORM','egl')
    import pyrender
    from PIL import Image, ImageDraw
    from trimesh.visual import ColorVisuals
    w=StructuralWorld(root)
    nodes=[n for n in w.parts.values() if 'mesh_ref' in n]
    colors=[[74,145,218,255],[237,153,51,255],[79,179,112,255],[201,92,164,255],[153,117,222,255],[228,94,79,255],[65,184,184,255]]
    bounds=w.mesh(w.program['object_id']).bounds;c=bounds.mean(axis=0);d=float(np.max(bounds[1]-bounds[0]))
    root=Path(root);folder=root/'renders';folder.mkdir(exist_ok=True)
    for view,(angle,elev) in enumerate([(25,25),(120,20),(220,30)]):
        a,e=np.deg2rad([angle,elev]);pos=c+2.5*d*np.array([np.cos(a)*np.cos(e),np.sin(a)*np.cos(e),np.sin(e)])
        z=(pos-c)/np.linalg.norm(pos-c);x=np.cross([0,0,1],z);x/=np.linalg.norm(x);y=np.cross(z,x)
        pose=np.eye(4);pose[:3,:3]=np.column_stack([x,y,z]);pose[:3,3]=pos
        images=[]
        for mode in ('textured','parts','exploded'):
            scene=pyrender.Scene(bg_color=[240,242,245,255],ambient_light=[.65,.65,.65])
            for i,n in enumerate(nodes):
                mesh=w.part_pieces(n['id'])[0].copy()
                if mode!='textured':mesh.visual=ColorVisuals(mesh,face_colors=([255,0,0,255] if n['unknown'] else colors[i%len(colors)]))
                if mode=='exploded':
                    v=mesh.bounds.mean(axis=0)-c
                    mesh.apply_translation(v*.6)
                scene.add(pyrender.Mesh.from_trimesh(mesh,smooth=False))
            scene.add(pyrender.PerspectiveCamera(yfov=np.deg2rad(40)),pose=pose)
            scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=2),pose=pose)
            renderer=pyrender.OffscreenRenderer(600,600)
            try:rgb,_=renderer.render(scene)
            finally:renderer.delete()
            images.append(Image.fromarray(rgb))
        canvas=Image.new('RGB',(1800,700),'white');draw=ImageDraw.Draw(canvas)
        for i,(label,img) in enumerate(zip(['Textured source partition','Semantic parts (red = unresolved)','Exploded diagnostic, NOT observation'],images)):
            canvas.paste(img,(600*i,35));draw.text((600*i+10,10),label,fill='black')
        for i,n in enumerate(nodes):
            col=tuple(colors[i%len(colors)][:3]) if not n['unknown'] else (255,0,0)
            xx=(i%5)*350;yy=640+(i//5)*22;draw.rectangle([xx,yy,xx+12,yy+12],fill=col);draw.text((xx+18,yy),n['id'],fill='black')
        canvas.save(folder/f'view_{view}.png')
    write(folder/'legend.json',{n['id']:colors[i%len(colors)] for i,n in enumerate(nodes)})


def run(root,max_rounds=5,target_object_id=None):
    root=Path(root).resolve();verify_freeze(root/'frozen')
    context_path=root/'input_context.json'
    if not context_path.exists() or json.loads(context_path.read_text()).get('feature_version')!=2:
        if context_path.exists():(root/'input_context_v1.json').write_text(context_path.read_text())
        from .structure_features import profile as object_profile
        world=MeshCodeWorld.load(root/'frozen/scene.json');profile=[]
        for node in world.source['nodes']:
            if target_object_id is not None and node['object_id']!=target_object_id:continue
            mesh=world.mesh(node['object_id']);normalized=(mesh.vertices-mesh.bounds[0])/np.maximum(mesh.extents,1e-12)
            slices=[]
            for low in np.arange(0,1,.05):
                points=normalized[(normalized[:,2]>=low)&(normalized[:,2]<=low+.05)]
                if len(points):slices.append({'z_band':[float(low),float(low+.05)],'min':points.min(0).tolist(),'max':points.max(0).tolist()})
            profile.append({'object_id':node['object_id'],'bounds_W':mesh.bounds.tolist(),'faces':len(mesh.faces),'normalized_height_slices':slices,**object_profile(world,node['object_id'])})
        write(root/'input_context.json',{'feature_version':2,'frame':'robot_base','units':'m','source':'frozen reconstructed candidate meshes; measured/generated source is recorded in scene asset provenance; NOT ground truth','objects':profile})
    history=[]
    for i in range(max_rounds):
        proposal=root/'proposals'/f'round_{i:02d}.json';candidate=root/'rounds'/f'{i:02d}'
        if not proposal.exists():
            status={'status':'NEED_PROPOSAL' if i==0 else 'NEED_REVISION','round':i,'proposal_path':str(proposal),'history':history,
                    'instruction':'Inspect previous part/whole renders and verification; revise only affected subtree selectors or relations. Keep unknown surfaces explicit.'}
            break
        if not candidate.exists():
            compile_program(root/'frozen',json.loads(proposal.read_text()),candidate)
        elif sha256(proposal)!=sha256(candidate/'proposal.json'):
            # JSON may have different whitespace; compare parsed content.
            if json.loads(proposal.read_text())!=json.loads((candidate/'proposal.json').read_text()):
                raise ValueError('do not modify completed proposals; submit next round')
        report=verify_program(candidate);write(candidate/'verification.json',report)
        if not (candidate/'renders/view_2.png').exists():previews(candidate)
        review_path=candidate/'review.json'
        if not review_path.exists():
            status={'status':'NEED_VISUAL_REVIEW','round':i,'candidate':str(candidate),'checks':report,'history':history,
                    'required_review':{'program_sha256':sha256(candidate/'structure.json'),'accepted':'boolean','reason':'part boundaries, omitted parts, parent relations and original observations; do not certify by appearance alone'}}
            break
        review=json.loads(review_path.read_text())
        if review.get('program_sha256')!=sha256(candidate/'structure.json'):raise ValueError('stale review')
        if review.get('verification_sha256') and review['verification_sha256']!=sha256(candidate/'verification.json'):
            review_path.rename(candidate/('review_stale_'+review['verification_sha256'][:12]+'.json'))
            status={'status':'NEED_VISUAL_REVIEW','round':i,'candidate':str(candidate),'checks':report,'history':history,
                    'reason':'Verification changed; previous model verdict preserved and must be reassessed.'}
            break
        history.append({'round':i,'checks':report['status'],'review':review})
        if report['status']=='STRUCTURE_CHECKED' and review.get('accepted'):
            publish_agent_bundle(candidate,review)
            w=StructuralWorld(candidate);w.export_glb(candidate/'scene.glb')
            status={'status':'STRUCTURALIZED','candidate':str(candidate),'round':i,'history':history,
                    'scope':'reviewed structure of frozen reconstructed asset; geometry limitations inherited'}
            break
    else:status={'status':'BUDGET_EXHAUSTED','history':history}
    write(root/'status.json',status)
    return status


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--max-rounds',type=int,default=5)
    args=p.parse_args();result=run(args.root,args.max_rounds);print(json.dumps({k:v for k,v in result.items() if k not in {'history','checks'}},ensure_ascii=False))
