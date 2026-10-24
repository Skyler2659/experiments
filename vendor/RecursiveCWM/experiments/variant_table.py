#!/usr/bin/env python3
"""Structural ablation table: compute metrics from each <scene>-<variant>-r1 chain's final render, with usage and wall time from trace/events.jsonl.
Write runs/pilot/metrics/variants.csv and variants-table.tex (RCWM_VARIANTS_TEX, default runs/pilot/metrics/)."""
import sys, os, csv, json, glob, subprocess, datetime, collections
def _env(name, hint):
    v=os.environ.get(name)
    if not v: raise SystemExit(f"set {name}: {hint}")
    return v
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import pickers
ROOT=_env('RCWM_ROOT','runtime root that holds runs/ (see docs/environment.md)'); C=os.environ.get('RCWM_REFS', os.path.join(os.path.dirname(os.path.abspath(__file__)),'pilot-scenes'))  # all ten reference images (see pilot-scenes/REFERENCES.md)
SCENES=['school-block','medieval-village']
SHOW_COST=False  # User request, 2026-09-09: hide Cost/Time columns for now (still write data to variants.csv).
VARS=[('flat','Flat + zoom'),('localglobal',r'Local $\rightarrow$ global'),('globallocal',r'Global $\rightarrow$ local'),('twolevel',r'\textbf{Recursive, fixed 2-level}'),('recursive',r'\textbf{Recursive, free depth}')]
def pick(chain):
    d=f'{chain}/fractal/scene'
    # Trust the final-render path declared in part.json first.
    try:
        ev=json.load(open(f'{d}/part.json')).get('evidence',{})
        for k in ('final_render','render','final'):
            v=ev.get(k) if isinstance(ev,dict) else None
            if isinstance(v,str) and os.path.isfile(f'{d}/{v}') and pickers._img_ok(f'{d}/{v}'): return f'{d}/{v}'
    except Exception: pass
    for n in ['final.png','FINAL.png','render.png','evidence/final.png','evidence/render-final.png','review/final.png']:
        if os.path.isfile(f'{d}/{n}') and pickers._img_ok(f'{d}/{n}'): return f'{d}/{n}'
    return pickers.newest([f'{d}/round*[0-9].png',f'{d}/assembled*[0-9].png',f'{d}/handoff.png',f'{d}/blockout*.png',f'{d}/evidence/integrated*[0-9].png'])
def tokens_from_logs(chain):
    """codex prints usage on the 'tokens used' line or the next; sum it across node logs."""
    import re, glob as g
    tot=0
    for f in g.glob(f'{chain}/fractal/*/codex-run.log'):
        lines=open(f,errors='ignore').read().splitlines()
        for i,l in enumerate(lines):
            if 'tokens used' in l:
                m=re.search(r'([0-9][0-9,]*)\s*$',l) or (re.match(r'\s*([0-9][0-9,]*)\s*$',lines[i+1]) if i+1<len(lines) else None)
                if m: tot+=int(m.group(1).replace(',',''))
    return tot
def trace(chain):
    ev=[]
    for l in open(f'{chain}/trace/events.jsonl'):
        try: ev.append(json.loads(l.replace('"usage_total":}','"usage_total":0}')))  # The old runner corrupted this field when usage was empty.
        except Exception: pass  # Skip lines corrupted by interleaved writes.
    # Time = union of wall-clock intervals with active sessions (exclude gaps from quota interruptions/manual restarts, preserving parallelism benefits).
    open_=collections.defaultdict(list); iv=[]
    for e in ev:
        ts=datetime.datetime.fromisoformat(e['ts'])
        if e['event']=='session_start': open_[e['node_id']].append(ts)
        elif e['event']=='session_end' and open_[e['node_id']]: iv.append((open_[e['node_id']].pop(0),ts))
    iv.sort(); merged=[]
    for s,en in iv:
        if merged and s<=merged[-1][1]: merged[-1]=(merged[-1][0],max(merged[-1][1],en))
        else: merged.append((s,en))
    mins=sum((en-s).total_seconds() for s,en in merged)/60
    done=os.path.isfile(f'{chain}/fractal/scene/part.json') and any(e['event']=='stop' and e['node_id']=='scene' for e in ev)
    return tokens_from_logs(chain), mins, len([e for e in ev if e['event']=='session_end']), done
rows=[]
for SC in SCENES:
    for v,name in VARS:
        ch=f'{ROOT}/runs/pilot/{SC}-{v}-r1'
        if not os.path.isdir(ch): continue
        p=pick(ch)
        m={}
        if p:
            out=subprocess.run([f'{ROOT}/.venv/bin/python',os.path.join(os.path.dirname(os.path.abspath(__file__)),'eval_metrics.py'),f'{C}/{SC}.png',p],capture_output=True,text=True,timeout=1200)
            try: m=json.loads(out.stdout.strip().splitlines()[-1])
            except Exception: print('metric fail',ch,out.stderr[-200:])
        tok,mins,sess,done=trace(ch)
        rows.append({'scene':SC,'variant':v,'name':name,'render':p or '','final':done,'psnr':m.get('psnr'),'ssim':m.get('ssim'),'detail_ssim':m.get('detail_crop_ssim_mean'),'lpips':m.get('lpips'),'edge_f1':m.get('edge_f1'),'tokens':tok,'minutes':round(mins,1),'sessions':sess})
        print(SC,v,os.path.relpath(p,ROOT) if p else None,done,m.get('psnr'),m.get('ssim'),m.get('detail_crop_ssim_mean'),tok,round(mins,1))
os.makedirs(f'{ROOT}/runs/pilot/metrics',exist_ok=True)
with open(f'{ROOT}/runs/pilot/metrics/variants.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
def fmt(v,f):
    return '--' if v in (None,'') else f%float(v)
for r in rows:  # Final paper: leave unfinished rows blank and exclude them from bolding (retain them in CSV).
    if not r['final']: r.update({'psnr':None,'ssim':None,'detail_ssim':None,'lpips':None})
L=[]
for SC in SCENES:
    rs=[r for r in rows if r['scene']==SC]
    if not rs: continue
    best={k:(max if k!='lpips' else min)((r[k] for r in rs if r[k] is not None),default=None) for k in ['psnr','ssim','detail_ssim','lpips']}
    bt=min((r['tokens'] for r in rs if r['final'] and r['tokens']),default=None); bm=min((r['minutes'] for r in rs if r['final']),default=None)
    L.append(r'\multicolumn{%d}{@{}l}{\emph{%s}}\\'%(7 if SHOW_COST else 5,SC))
    for r in rs:
        def c(k,f):
            s=fmt(r[k],f); return r'\textbf{%s}'%s if s!='--' and r[k]==best[k] else s
        tk='%.0fk'%(r['tokens']/1000) if r['tokens'] else '--'; mn='%.0f'%r['minutes']
        if r['final'] and r['tokens']==bt: tk=r'\textbf{%s}'%tk
        if r['final'] and r['minutes']==bm: mn=r'\textbf{%s}'%mn
        dag=''
        tail=f" & {tk} & {mn}" if SHOW_COST else ''
        L.append(f"{r['name']}{dag} & {c('psnr','%.1f')} & {c('ssim','%.2f')} & {c('detail_ssim','%.2f')} & {c('lpips','%.3f')}{tail} \\\\")
    L.append(r'\midrule')
L=L[:-1]
COST_CAP=r" Cost is total model usage (thousand tokens, summed over every session of every node) and time is the wall clock during which at least one solver session was running, both over the full run, including sessions of attempts that were interrupted by provider quota limits and resumed; $\dagger$ marks runs not yet at their own stopping state, so their cost and time are lower bounds."
NOCOST_CAP=r""
tex=r"""\begin{table}[h]
\caption{Structural controls on two scenes, single runs on the same base model and scene compiler. Global fidelity is whole-frame PSNR/SSIM/LPIPS against the reference; local fidelity is the mean SSIM over eight fixed $2\times$ detail windows."""+(COST_CAP if SHOW_COST else NOCOST_CAP)+r""" Best per scene in bold.}
\label{tab:variants}
\small
\begin{tabular}{@{}l"""+('cccccc' if SHOW_COST else 'cccc')+r"""@{}}
\toprule
Variant & PSNR$\uparrow$ & SSIM$\uparrow$ & Local SSIM$\uparrow$ & LPIPS$\downarrow$"""+(r" & Cost$\downarrow$ & Time (min)$\downarrow$" if SHOW_COST else '')+r""" \\
\midrule
"""+'\n'.join(L)+r"""
\bottomrule
\end{tabular}
\end{table}
"""
OUT=os.environ.get('RCWM_VARIANTS_TEX',f'{ROOT}/runs/pilot/metrics/variants-table.tex'); open(OUT,'w').write(tex); print('tex written',OUT)
