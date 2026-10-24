#!/usr/bin/env python3
"""Local orchestration only: transfer source, call Linux tools, return node feedback.
Exit 10 asks the original recursive runner to resume this node; exit 20 is infrastructure failure.
"""
import fcntl,json,os,re,shlex,shutil,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from refinement_policy import subpart_scope,decomposition_required,should_retry,child_source_revision,update_child_revision
import adaptive_refinement as adaptive
import correction_review as correction

def call(cmd,**kw):return subprocess.run(cmd,check=True,**kw)
def source_files(nd):return [p for p in nd.iterdir() if p.is_file() and (p.suffix in {'.js','.mjs','.json','.html'} or p.name=='account.md') and not p.name.startswith(('feedback','fit_requests','selection','review_state','refinement_pending','node_context'))]

def inherited_residuals(result,nd,run):
    """An unchanged exhausted child is not a new parent assembly error."""
    if nd.name!='scene' or result.get('missing_object_ids') or result['status']=='PROGRAM_ERROR':return []
    inherited=[]
    children=json.loads((nd/'part.json').read_text()).get('children',[])
    for child in children:
        if not isinstance(child,str):continue
        selection=run/'fractal'/child/'selection.json'
        if not selection.exists():continue
        selected=json.loads(selection.read_text())
        if not selected.get('budget_exhausted'):continue
        prior=json.loads((run/selected['selected_feedback']/'feedback.json').read_text())
        for p in prior.get('objects',[]):
            for current in result.get('objects',[]):
                if current['object_id']==p['object_id'] and current['needs_revision'] and current['score']<=p['score']*1.01+1e-6:
                    inherited.append(current['object_id'])
    return sorted(set(inherited))
def run():
    chain,node,cycle=sys.argv[1:4]
    if not re.fullmatch(r'[\w-]+',node):raise ValueError('Node IDs must be flat')
    runtime=Path(os.environ['RCWM_ROOT']);run=runtime/chain;nd=run/'fractal'/node
    cfg=json.loads(Path(os.environ['RCWM_ASSIST_CONFIG']).read_text());remote=cfg['remote_runtime'];name=run.name
    local_transport=cfg.get('transport')=='local'
    if len(sys.argv)>4 and sys.argv[4]=='dispatch':
        if adaptive.enabled(cfg):
            context=json.loads((nd/'node_context.json').read_text())
            if adaptive.dispatch(nd,run,context['parent_id']):return 11
        return 0
    if len(sys.argv)>4 and sys.argv[4]=='delegate':
        if adaptive.enabled(cfg) and not adaptive.can_delegate(nd,run):
            (nd/'feedback.md').write_text('ADAPTIVE_DESCENT_BLOCKED: first deliver a complete object program and obtain its measured review. Only a failing object with decision REFINE may create/revisit subparts. A passing or stopped object must be reused; unused depth is not permission to split it. Do not bypass a stopped owner by creating new IDs.\n')
            return 10
        return 0
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ControlPath=none','-o','ProxyCommand=ssh -S /tmp/rcwm-drawer-final -W %h:%p sz.2.gpu','l203']
    rsync_shell='ssh -o ControlPath=none -o "ProxyCommand=ssh -S /tmp/rcwm-drawer-final -W %h:%p sz.2.gpu"'
    with (run/'.feedback.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        mode='fit' if (nd/'fit_requests.json').exists() else 'review'
        if mode=='review' and not (nd/'part.json').exists():return 0
        if mode=='review' and cfg.get('refinement',{}).get('enabled'):
            try:scope=subpart_scope(nd,run)
            except (ValueError,OSError) as error:
                (nd/'refinement_pending.json').write_text(json.dumps({'status':'SUBPART_SCOPE_REQUIRED','error':str(error)})+'\n')
                (nd/'feedback.md').write_text('PROGRAM_ERROR in subpart ownership: '+str(error)+'\nRead your actual node_context.json and parent brief/scope. Write node_scope.json in YOUR flat fractal/<node>/ directory. Reuse the assigned semantic owner and actual direct parent, never invent a new object. Do not change geometry to compensate for missing scope metadata.\n')
                return 10
            if scope:
                (nd/'refinement_pending.json').unlink(missing_ok=True)
                (nd/'selection.json').write_text(json.dumps({'selected_status':'DEFERRED_TO_PARENT','owner_object_id':scope['owner_object_id'],'parent_node':scope['parent_node'],'independent_acceptance':False})+'\n')
                return 0
            if decomposition_required(nd,cfg):
                (nd/'refinement_pending.json').write_text(json.dumps({'status':'DECOMPOSITION_REQUIRED','node':node})+'\n')
                (nd/'feedback.md').write_text('DECOMPOSITION_REQUIRED: this structural object must be reconstructed by at least two real subpart children. Read the deep reconstruction prompt. Create scoped child briefs, node_scope.json and children.json as a JSON LIST, then END. Retain the observed gaps, cavities, edges and surface detail. Increasing tessellation or moving the same whole-object primitive is not decomposition. Parent assembly must import and use the child builds; it will be evaluated against the full object.\n')
                return 10
            (nd/'refinement_pending.json').unlink(missing_ok=True)
        if not local_transport:call(['rsync','-a','--exclude=*.log','--exclude=feedback/***','--exclude=.best','--exclude=.adaptive_best','--exclude=.sid','-e',rsync_shell,str(run/'fractal')+'/',f'l203:{remote}/runs/{name}/fractal/'])
        stamp=f'{node}_{mode}_{time.time_ns()}';remote_out=cfg['remote_work']+'/feedback/'+stamp
        target=nd/'feedback'/stamp
        if local_transport:remote_out=str(target)
        geometry_path=cfg.get('geometry_pythonpath','/private/lc/research/agentic_mesh_world_deps/geometry' if not local_transport else '')
        env={'PYTHONPATH':cfg['remote_source']+(os.pathsep+geometry_path if geometry_path else ''),'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYOPENGL_PLATFORM':'egl'}
        python=cfg.get('evidence_python',sys.executable if local_transport else '/private/lc/conda_envs/tabletopgen/bin/python')
        cmd=['env',*[k+'='+v for k,v in env.items()],python,'-m','reconstruction_assist.remote_feedback','--runtime',remote,'--name',name,'--node',node,'--evidence',correction.active_evidence(run,cfg),'--output',remote_out,'--mode',mode]
        if cfg.get('refinement',{}).get('enabled') and not adaptive.enabled(cfg) and node in cfg['refinement'].get('structural_objects',[]):cmd.append('--require-parts')
        if cfg.get('refinement',{}).get('enabled'):cmd.append('--require-exportable-materials')
        if local_transport:call(cmd)
        else:
            call(ssh+[shlex.join(cmd)])
            target.mkdir(parents=True)
            call(['rsync','-a','-e',rsync_shell,f'l203:{remote_out}/',str(target)+'/'])
        if mode=='fit':
            incoming=json.loads((target/'fit_results.json').read_text())
            previous=json.loads((nd/'fit_results.json').read_text()) if (nd/'fit_results.json').exists() else {'results':[]}
            merged={r['id']:r for r in previous['results']}
            merged.update({r['id']:r for r in incoming['results']})
            incoming['results']=list(merged.values())
            (nd/'fit_results.json').write_text(json.dumps(incoming,indent=2)+'\n');shutil.move(nd/'fit_requests.json',target/'fit_requests.json')
            (nd/'feedback.md').write_text('Fitting tools completed. Read fit_results.json. Use successful recipe parameters through input/measurement/recipe_geometry.mjs or build equivalent parameter-driven geometry. Failed fits are evidence, not permission to invent a measured shape. Continue your assigned scope.\n');return 10
        result=json.loads((target/'feedback.json').read_text())
        if adaptive.enabled(cfg):
            inherited=inherited_residuals(result,nd,run)
            failed={row['object_id'] for row in result.get('objects',[]) if row.get('needs_revision')}
            only_inherited=bool(inherited) and failed<=set(inherited) and not result.get('missing_object_ids') and result['status']!='PROGRAM_ERROR'
            retry,state=adaptive.finish_review(nd,run,correction.effective_config(nd,cfg),result,str(target.relative_to(run)),only_inherited)
            retry,state,correction_note=correction.review(nd,run,cfg,state,state.get('best_feedback',str(target.relative_to(run))),shared_lock=lock)
            lines=['Linux geometry feedback: '+result['status'],
                   'Adaptive decision: '+state['decision'],
                   'Read '+str(target.relative_to(runtime))+'/feedback.json and its overlays.',
                   'Only revise the failed object or its observed residual region. Passing objects are reused, not rebuilt.',
                   'Diagnose structural failures only; do not review or polish textures, colors or packaging graphics.']
            if correction_note:lines.append(correction_note)
            for row in result.get('objects',[]):
                if row.get('needs_revision'):lines.append(f"{row['object_id']}: p95={row['point_surface_p95_m']*1000:.2f} mm, missing={row['mean_missing_fraction']:.3f}, excess={row['mean_false_occupied_fraction']:.3f}.")
            for row in result.get('objects',[]):
                contact=row.get('support_contact',{})
                if contact.get('floating'):
                    gap=contact.get('minimum_vertical_support_gap_m')
                    lines.append(f"{row['object_id']}: FLOATING_SUPPORT; vertical gap={gap} m, supporting candidate={contact.get('support_object_id')}. Inspect support_contact in feedback.json. Preserve measured visible surfaces; repair the assembly or explicitly hypothesized hidden contact structure rather than blindly translating every object.")
            if result.get('error'):lines.append(result['error'])
            if retry:
                lines.append('Read adaptive_targets.json before revisiting object children. Target only unresolved owners with remaining budget. At an object node, correct local parameters first; create scoped subparts only for residual structure that the current representation cannot express. Do not repeat unchanged whole-scene delivery or bypass stopped scopes with fresh IDs.')
            else:
                lines.append('This scope is stopped. Reuse its selected subtree; report unresolved residuals without claiming a pass. A parent must not restart this local budget.')
            (nd/'feedback.md').write_text('\n'.join(lines)+'\n')
            return 10 if retry else 0
        state_file=nd/'review_state.json';state=json.loads(state_file.read_text()) if state_file.exists() else {'reviews':0,'best_score':float('inf')};state['reviews']+=1
        child_changed=False
        if cfg.get('refinement',{}).get('enabled'):
            child_changed=update_child_revision(state,child_source_revision(nd,run,source_files))
        if result['score']<state['best_score']:
            state.update(best_score=result['score'],best_feedback=str(target.relative_to(run)),best_status=result['status']);best=nd/'.best';shutil.rmtree(best,ignore_errors=True);best.mkdir()
            for p in source_files(nd):shutil.copy2(p,best/p.name)
        state_file.write_text(json.dumps(state,indent=2)+'\n')
        inherited=inherited_residuals(result,nd,run)
        only_inherited=bool(inherited) and set(result.get('revision_objects',[]))<=set(inherited) and not result.get('missing_object_ids')
        retry=should_retry(result,state,cfg,only_inherited)
        lines=['Linux geometry feedback: '+result['status'], 'Read '+str(target.relative_to(runtime))+'/feedback.json and its overlays.', 'Red means false occupied; blue means missing; green matches. Correct only your assigned node; measured evidence and fit hypotheses can be revised, camera calibration cannot.']
        for row in result.get('objects',[]):
            if row['needs_revision']:lines.append(f"{row['object_id']}: p95={row['point_surface_p95_m']*1000:.2f} mm, missing={row['mean_missing_fraction']:.3f}, excess={row['mean_false_occupied_fraction']:.3f}. Inspect worst view overlays before changing structure.")
        if result.get('error'):lines.append(result['error'])
        if child_changed:lines.append('Child source revision changed. The previous parent-only best snapshot does not describe this assembly, so its score was invalidated; consumed review budget is unchanged.')
        if cfg.get('refinement',{}).get('enabled'):
            lines.append('Inspect actual *_render.png images alongside the same original FIT RGB views, not only the colored error overlays. Correct visible material color, rim bands, wood grain and repeated grooves when supported by the image. Appearance is not certified by the depth score.')
            if retry and result['status']!='PROGRAM_ERROR':
                lines.append('STRUCTURAL_REFINEMENT_REQUIRED: unresolved regions must be assigned to scoped subpart children when depth remains. Use children.json to create/revisit only affected children; put their repair instructions in your own refinement_requests.json keyed by child node ID. Children read that parent file on resume. Do not accept inherited child errors as a stopping condition. After child delivery reassemble and run the full-object review again. At maximum depth, revise your own local program. Report any unresolved residuals at the review limit.')
        (nd/'feedback.md').write_text('\n'.join(lines)+'\n')
        if retry:return 10
        restored=state['best_score']<result['score']
        if restored:
            for p in source_files(nd):
                if p.name=='account.md' and not (nd/'.best/account.md').exists():
                    p.write_text('Source was restored to an earlier lower-error delivery; the model narrative below describes the later attempted revision. See selection.json for the selected evidence.\n\n'+p.read_text())
                else:p.unlink()
            for p in (nd/'.best').iterdir():shutil.copy2(p,nd/p.name)
        (nd/'selection.json').write_text(json.dumps({'reviews':state['reviews'],'selected_feedback':state['best_feedback'],'selected_score':state['best_score'],'selected_status':state['best_status'],'restored_best':restored,'budget_exhausted':result['status']!='FIT_CHECKS_PASSED','inherited_child_residuals':inherited,'independent_acceptance':False},indent=2)+'\n')
        return 0
if __name__=='__main__':
    try:
        chain,node=sys.argv[1:3]
        if not re.fullmatch(r'[\w-]+',node):raise ValueError('Node IDs must be flat')
        lock_path=Path(os.environ['RCWM_ROOT'])/chain/'fractal'/node/'.correction_node.lock'
        with lock_path.open('a') as node_lock:
            fcntl.flock(node_lock,fcntl.LOCK_EX)
            sys.exit(run())
    except Exception as e:
        print('REMOTE_FEEDBACK_ERROR: '+repr(e),file=sys.stderr);sys.exit(20)
