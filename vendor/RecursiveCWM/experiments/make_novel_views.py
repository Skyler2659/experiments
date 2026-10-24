#!/usr/bin/env python3
"""Novel-view grid: one row per scene = final reference-camera render + five novel views (+/-35 degrees, high orbit, two close-ups), at equal height and centered within columns.
Usage: make_novel_views.py <out.png>"""
import sys, os, glob
def _env(name, hint):
    v=os.environ.get(name)
    if not v: raise SystemExit(f"set {name}: {hint}")
    return v
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import pickers
from PIL import Image, ImageDraw, ImageFont, ImageOps
ROOT=_env('RCWM_ROOT','runtime root that holds runs/ (see docs/environment.md)')
BOLD='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
FONT=ImageFont.truetype(BOLD,52); FONT2=ImageFont.truetype(BOLD,44)
SCENES=['city-full','snow-village','medieval-village','japan-island','island-harbor','school-block','police-corner','park-lake','shop-row','valley-village']
COLS=[('Reference cam',None),('$-35^\\circ$','view-L35.png'),('$+35^\\circ$','view-R35.png'),('High orbit','view-orbit.png'),('Close-up A','view-close1.png'),('Close-up B','view-close2.png')]
COLS=[('Reference cam',None),('-35°','view-L35.png'),('+35°','view-R35.png'),('High orbit','view-orbit.png'),('Close-up A','view-close1.png'),('Close-up B','view-close2.png')]
def content_box(im):
    import numpy as np
    a=np.asarray(im.convert('RGB')); m=np.any(a<245,axis=2)
    if not m.any(): raise ValueError('blank view has no image content')
    def span(bits):
        edges=np.diff(np.r_[False,bits,False].astype(int))
        runs=list(zip(np.where(edges==1)[0],np.where(edges==-1)[0]))
        # A detached thin frame rule after a white footer is not scene content.
        thin=max(2,round(len(bits)*0.01))
        while len(runs)>1 and runs[0][1]-runs[0][0]<=thin: runs.pop(0)
        while len(runs)>1 and runs[-1][1]-runs[-1][0]<=thin: runs.pop()
        return int(runs[0][0]),int(runs[-1][1])
    top,bottom=span(m.any(1)); left,right=span(m[top:bottom].any(0))
    return left,top,right,bottom
def cb(im):
    return im.crop(content_box(im))
def fit_content(im,size):
    # Cover fitting can remove an object's extremity and expose another blank
    # edge. Refine the source box, always resampling from the original pixels.
    box=tuple(map(float,content_box(im)))
    for _ in range(20):
        left,top,right,bottom=box; w,h=right-left,bottom-top
        scale=min(w/size[0],h/size[1])
        left=max(0.0,left+(w-size[0]*scale)/2); top=max(0.0,top+(h-size[1]*scale)/2)
        box=(left,top,min(im.width,left+size[0]*scale),min(im.height,top+size[1]*scale))
        fitted=im.resize(size,Image.Resampling.LANCZOS,box=box)
        l,t,r,b=content_box(fitted)
        if max(l,t,size[0]-r,size[1]-b)<=1: return fitted
        box=(left+l*scale,top+t*scale,left+r*scale,top+b*scale)
    raise ValueError('content crop did not converge')
def load(SC,fn):
    if fn is None:
        p,_=pickers.ours(ROOT,SC); return Image.open(p).convert('RGB') if p else None
    p=f'{pickers.ours_chain(ROOT,SC)}/fractal/scene/novel-views/{fn}'
    return Image.open(p).convert('RGB') if os.path.isfile(p) else None
def main(out,RH=520,PAD=10,COLLBL=64,ROWLBL=56):
    rows=[]
    for SC in SCENES:
        cells=[load(SC,fn) for _,fn in COLS]
        if any(c is None for c in cells): raise ValueError(f'{SC}: all six views are required')
        cells=[cb(c) for c in cells]  # trim at native resolution, before any scaling
        rows.append((SC,cells))
    # One fixed width per column; median aspect avoids an outlier's excessive crop.
    import statistics
    colw=[round(RH*statistics.median(cells[j].width/cells[j].height for _,cells in rows)) for j in range(len(COLS))]
    colw=[max(w,round(FONT.getlength(name))+2*PAD) for w,(name,_) in zip(colw,COLS)]
    # Scene names occupy a snug left gutter, so rows need only the 10 px separator.
    ROWLBL=max(ROWLBL,max(FONT2.getbbox(SC)[2] for SC,_ in rows)+2*PAD)
    W=ROWLBL+sum(colw)+PAD*(len(COLS)+1); H=COLLBL+len(rows)*(RH+PAD)
    sheet=Image.new('RGB',(W,H),'white'); d=ImageDraw.Draw(sheet); x=ROWLBL+PAD
    header_y=6-min(FONT.getbbox(name)[1] for name,_ in COLS)
    for j,(name,_) in enumerate(COLS):
        tw=d.textlength(name,font=FONT); d.text((x+(colw[j]-tw)//2,header_y),name,fill='black',font=FONT); x+=colw[j]+PAD
    y=COLLBL; starts=[]
    for SC,cells in rows:
        x=ROWLBL+PAD; starts.append(y)
        for j,c in enumerate(cells):
            c=fit_content(c,(colw[j],RH))
            sheet.paste(c,(x,y))
            x+=colw[j]+PAD
        left,top,right,bottom=d.textbbox((0,0),SC,font=FONT2)
        d.text((ROWLBL-PAD-right,y+(RH-(bottom-top))//2-top),SC,fill='black',font=FONT2); y+=RH+PAD
    sheet.save(out,optimize=True); print(out,sheet.size,len(rows),'rows','row starts',starts,'row height',RH,'column widths',colw,'label width',ROWLBL,flush=True)
if __name__=='__main__': main(sys.argv[1])
