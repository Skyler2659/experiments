"""One remote geometry service call; runs only inside the isolated reconstruction run."""
import argparse,json,os,subprocess
from pathlib import Path
from .fit_tools import execute
from .evaluate import evaluate
from .refinement_policy import resolve_children

def run(root,name,node,evidence,output,mode,require_parts=False,require_exportable_materials=False):
    root=Path(root);nd=root/'runs'/name/'fractal'/node;out=Path(output);out.mkdir(parents=True,exist_ok=False)
    if mode=='fit':
        req=json.loads((nd/'fit_requests.json').read_text());result=execute(req['requests'],evidence)
        (out/'fit_results.json').write_text(json.dumps(result,indent=2)+'\n');return
    cfg=json.loads((Path(evidence)/'config.json').read_text());script=Path(__file__).with_name('render_node.mjs')
    env=os.environ.copy();env['PLAYWRIGHT_BROWSERS_PATH']=str((root/'.browsers').resolve()) if (root/'.browsers').exists() else env.get('PLAYWRIGHT_BROWSERS_PATH',str(Path.home()/'.cache/rcwmworld-browsers'))
    libraries=root/'.system-libs/usr/lib/x86_64-linux-gnu'
    if libraries.exists():env['LD_LIBRARY_PATH']=str(libraries)+os.pathsep+env.get('LD_LIBRARY_PATH','')
    command=[str(root/'.render-tools/node/bin/node'),str(script),str(root),name,node,cfg['capture'],str(out)]
    with (out/'renderer.log').open('w') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env,timeout=180)
    if r.returncode:
        result={'status':'PROGRAM_ERROR','score':1000.,'error':(out/'renderer.log').read_text()[-5000:],'revision_objects':[]}
    else:
        scope=None
        if node=='scene':scope=[r['object_id'] for r in cfg['objects']]+['tabletop','known_robot']
        elif node in json.loads((Path(evidence)/'model/evidence.json').read_text())['objects']:scope=[node]
        rendered=json.loads((out/'render.json').read_text())
        try:
            children=resolve_children(nd,root/'runs'/name)
            reference_error=None
        except (ValueError,OSError) as error:children=[];reference_error=str(error)
        declared_parts=[c for c,path in children if (path/'node_scope.json').exists()]
        expected_parts=[c for c,_ in children] if require_parts else declared_parts
        missing_parts=sorted(set(expected_parts)-set(rendered.get('part_inventory',{})))
        material_issues=rendered.get('material_export_issues',[])
        if reference_error:
            result={'status':'PROGRAM_ERROR','score':1000.,'error':reference_error,'revision_objects':[]}
        elif require_exportable_materials and material_issues:
            affected=sorted({i['object_id'] for i in material_issues if i.get('object_id')})
            result={'status':'PROGRAM_ERROR','score':1000.,'error':'GLB appearance would lose custom shaders: '+json.dumps(material_issues)+'. Replace ShaderMaterial/onBeforeCompile effects with actual texture maps (DataTexture/CanvasTexture), supported PBR materials or vertex colors. Preserve observed appearance in the exported asset. Route the correction to the named object/subpart; do not delete the pattern merely to pass this check.','revision_objects':affected}
        elif missing_parts:
            result={'status':'PROGRAM_ERROR','score':1000.,'error':'Declared subparts were not built/tagged in the assembly: '+', '.join(missing_parts),'revision_objects':[]}
        else:
            result=evaluate(out,evidence,scope)
            result['executed_subparts']=rendered.get('part_inventory',{})
    (out/'feedback.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runtime',required=True);p.add_argument('--name',required=True);p.add_argument('--node',required=True);p.add_argument('--evidence',required=True);p.add_argument('--output',required=True);p.add_argument('--mode',choices=['fit','review'],default='review');p.add_argument('--require-parts',action='store_true');p.add_argument('--require-exportable-materials',action='store_true');a=p.parse_args();run(a.runtime,a.name,a.node,a.evidence,a.output,a.mode,a.require_parts,a.require_exportable_materials)
