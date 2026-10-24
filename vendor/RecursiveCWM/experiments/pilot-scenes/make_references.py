#!/usr/bin/env python3
"""Rebuild the five WorldClaw-derived references from page renders of arXiv:2608.05248 (see REFERENCES.md).
usage: make_references.py <dir with fig04.png fig09.png fig10.png fig12.png fig15.png> <out dir>"""
import sys, os, hashlib
from PIL import Image
TABLE={'medieval-village':('fig09.png',(2208,804,4200,2888),(1400,963)),
       'snow-village':('fig10.png',(2200,796,4200,2888),(1400,963)),
       'island-harbor':('fig04.png',(2196,796,4200,2888),(1400,963)),
       'japan-island':('fig12.png',(2192,800,4200,2888),(1400,963)),
       'valley-village':('fig15.png',(76,2196,1987,1447),None)}
src,out=sys.argv[1],sys.argv[2]; os.makedirs(out,exist_ok=True)
for scene,(f,(x,y,w,h),size) in TABLE.items():
    p=os.path.join(src,f)
    if not os.path.isfile(p): print('missing',p); continue
    im=Image.open(p).convert('RGB')
    if im.width!=8516: print(f'note: {f} is {im.width}px wide; crop boxes assume 8516px page renders (scaling by {im.width/8516:.3f})')
    s=im.width/8516; box=(round(x*s),round(y*s),round((x+w)*s),round((y+h)*s))
    c=im.crop(box)
    if size: c=c.resize(size,Image.LANCZOS)
    q=os.path.join(out,scene+'.png'); c.save(q)
    print(scene, c.size, hashlib.sha256(open(q,'rb').read()).hexdigest()[:16])
