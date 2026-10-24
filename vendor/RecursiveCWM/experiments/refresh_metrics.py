#!/usr/bin/env python3
"""Recompute the full table using shared pickers selection: runs/pilot/metrics/all-methods-full.csv (record render paths and final flags)."""
import sys, os, csv, json, subprocess
def _env(name, hint):
    v=os.environ.get(name)
    if not v: raise SystemExit(f"set {name}: {hint}")
    return v
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import pickers
ROOT=_env('RCWM_ROOT','runtime root that holds runs/ (see docs/environment.md)'); C=os.environ.get('RCWM_REFS', os.path.join(os.path.dirname(os.path.abspath(__file__)),'pilot-scenes'))  # all ten reference images (see pilot-scenes/REFERENCES.md)
CSV=os.environ.get('RCWM_METRICS_CSV',f'{ROOT}/runs/pilot/metrics/all-methods-full.csv')
SCENES=['city-full','snow-village','island-harbor','medieval-village','japan-island','school-block','police-corner','park-lake','shop-row','valley-village']
M={'ours':pickers.ours,'seig':pickers.seig,'viga':pickers.viga,'img2threejs':pickers.i2t}
rows={(r['scene'],r['method']):r for r in csv.DictReader(open(CSV))} if os.path.isfile(CSV) else {}
keys=list(next(iter(rows.values())).keys()) if rows else ['scene','method','final']
for k in ['render','final']:
    if k not in keys: keys.append(k)
for SC in SCENES:
    for m,fn in M.items():
        p,fin=fn(ROOT,SC)
        if not p:
            rows.pop((SC,m),None); continue
        r=rows.get((SC,m))
        if r and r.get('render')==p and r.get('final')==str(fin) and r.get('psnr'): continue
        out=subprocess.run([f'{ROOT}/.venv/bin/python',os.path.join(os.path.dirname(os.path.abspath(__file__)),'eval_metrics.py'),f'{C}/{SC}.png',p],capture_output=True,text=True,timeout=1200)
        try: d=json.loads(out.stdout.strip().splitlines()[-1])
        except Exception: print('FAIL',SC,m,out.stderr[-300:]); continue
        r={'scene':SC,'method':m,'render':p,'final':str(fin)}; r.update({k:str(v) for k,v in d.items()})
        for k in r:
            if k not in keys: keys.append(k)
        rows[(SC,m)]=r; print('ok',SC,m,os.path.relpath(p,os.path.dirname(ROOT)),fin,d.get('psnr'),d.get('ssim'),d.get('edge_f1'))
with open(CSV,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=keys); w.writeheader()
    for SC in SCENES:
        for m in M:
            if (SC,m) in rows: w.writerow({k:rows[(SC,m)].get(k,'') for k in keys})
print('written',CSV)
