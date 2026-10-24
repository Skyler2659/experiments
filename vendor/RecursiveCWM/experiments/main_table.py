#!/usr/bin/env python3
"""Main table: runs/pilot/metrics/all-methods-full.csv -> runs/pilot/metrics/casemetrics-table.tex; also replace it in the paper's 04_experiments.tex (RCWM_PAPER).
Bold the best value per scene and metric; mark final=False rows with a dagger; omit the Tokens column for now at the user's request."""
import csv, collections, re, sys
from pathlib import Path
import os
def _env(name, hint):
    v=os.environ.get(name)
    if not v: raise SystemExit(f"set {name}: {hint}")
    return v
ROOT=_env('RCWM_ROOT','runtime root that holds runs/ (see docs/environment.md)'); CSV=os.environ.get('RCWM_METRICS_CSV',f'{ROOT}/runs/pilot/metrics/all-methods-full.csv')
NO_PAPER=os.environ.get('RCWM_NO_PAPER','0')=='1'  # 1 = write the table next to the csv only, do not touch the paper
PAPER=None if NO_PAPER else Path(_env('RCWM_PAPER','paper repo checkout whose sections/04_experiments.tex holds the table'))/'sections/04_experiments.tex'
WHOLE=['city-full','snow-village','island-harbor','medieval-village','japan-island']
CROPS=['school-block','police-corner','park-lake','shop-row','valley-village']
NAME={'ours':'Ours','seig':'SEIG','viga':'VIGA','img2threejs':'img2threejs'}
rows=[r for r in csv.DictReader(open(CSV)) if r.get('psnr') and r.get('final')=='True']  # Include only finished runs in the final paper.
by=collections.defaultdict(dict)
for r in rows: by[r['scene']][r['method']]=r
def emit(scenes):
    out=[]
    for SC in scenes:
        ms=by.get(SC,{})
        if not ms: continue
        def best(key,hi=True):
            vals={m:float(r[key]) for m,r in ms.items() if r.get(key) not in ('None','',None)}
            return ((max if hi else min)(vals,key=vals.get)) if vals else None
        B={'psnr':best('psnr'),'ssim':best('ssim'),'edge_f1':best('edge_f1'),'lpips':best('lpips',False),'clip_sim':best('clip_sim')}
        first=True
        for m in ['ours','seig','viga','img2threejs']:
            if m not in ms: continue
            r=ms[m]
            def cell(key,fmt,bm):
                v=r.get(key)
                if v in ('None','',None): return '--'
                s=fmt%float(v); return r'\textbf{'+s+'}' if m==bm else s
            out.append(f"{SC if first else ''} & {NAME[m]} & {cell('psnr','%.1f',B['psnr'])} & {cell('ssim','%.2f',B['ssim'])} & {cell('edge_f1','%.2f',B['edge_f1'])} & {cell('lpips','%.3f',B['lpips'])} & {cell('clip_sim','%.2f',B['clip_sim'])} \\\\")
            first=False
        out.append(r'\midrule')
    return '\n'.join(out[:-1])
tex=(r"""\begin{table}[t]
\caption{Record-only metrics for every method with a completed run; best per scene in bold.
Single runs on a shared base model with identical raw inputs (all reference annotations
kept by every method); SEIG is our from-paper reproduction.}
\label{tab:casemetrics}
\small
\begin{tabularx}{\linewidth}{@{}l l Y Y Y Y Y@{}}
\toprule
Scene & Method & PSNR$\uparrow$ & SSIM$\uparrow$ & Edge $F_1\uparrow$ & LPIPS$\downarrow$ & CLIP$\uparrow$ \\
\midrule
\multicolumn{7}{@{}l}{\emph{Whole scenes}}\\
""" + emit(WHOLE) + r"""
\midrule
\multicolumn{7}{@{}l}{\emph{Local crops}}\\
""" + emit(CROPS) + r"""
\bottomrule
\end{tabularx}
\end{table}
""")
OUTDIR=os.path.dirname(CSV); Path(f'{OUTDIR}/casemetrics-table.tex').write_text(tex)
# a markdown copy for reading outside LaTeX
md=['| Scene | Method | PSNR↑ | SSIM↑ | Edge F1↑ | LPIPS↓ | CLIP↑ |','|---|---|---|---|---|---|---|']
for l in tex.splitlines():
    if l.count('&')==6 and 'Scene & Method' not in l: md.append('| '+' | '.join(c.strip().replace('\\textbf{','**').replace('}','**').replace('\\\\','') for c in l.split('&'))+' |')
Path(f'{OUTDIR}/casemetrics-table.md').write_text('\n'.join(md)+'\n')
n=0
if PAPER:
    t=PAPER.read_text()
    t2,n=re.subn(r'\\begin\{table\}\[t\]\n\\caption\{Record-only.*?\\end\{table\}\n', lambda m: tex, t, count=1, flags=re.S)
    if n: PAPER.write_text(t2)
print('table rows',sum(1 for l in tex.splitlines() if l.endswith('\\\\') and '&' in l),'swapped into paper' if n else ('local only' if NO_PAPER else 'NOT swapped'))
