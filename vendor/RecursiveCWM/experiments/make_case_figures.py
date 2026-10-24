#!/usr/bin/env python3
import sys, os, glob, numpy as np, cv2
def _env(name, hint):
    v=os.environ.get(name)
    if not v: raise SystemExit(f"set {name}: {hint}")
    return v
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); import pickers
from PIL import Image, ImageDraw, ImageFont, ImageOps
from make_novel_views import content_box, fit_content
from skimage.metrics import structural_similarity as ssim
ROOT=_env('RCWM_ROOT','runtime root that holds runs/ (see docs/environment.md)')
C=os.environ.get('RCWM_REFS', os.path.join(os.path.dirname(os.path.abspath(__file__)),'pilot-scenes'))  # all ten reference images (see pilot-scenes/REFERENCES.md)
BOLD='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
FONT=ImageFont.truetype(BOLD,52); FONT2=ImageFont.truetype(BOLD,44)
PANELS=[('Reference',None),('Ours',pickers.ours),('SEIG',pickers.seig),('VIGA',pickers.viga),('img2threejs',pickers.i2t)]
def cb(im):
    a=np.asarray(im.convert('RGB')).astype(int); bg=a[2,2]; m=(abs(a-bg).sum(2)>30)
    if m.sum()<100: return im
    ys,xs=m.any(1).nonzero()[0],m.any(0).nonzero()[0]
    return im.crop((max(xs[0]-8,0),max(ys[0]-8,0),min(xs[-1]+8,im.width),min(ys[-1]+8,im.height)))
def cover(im,size,preserve_frame=False):
    if preserve_frame:
        # Last resort for the source frame's a/b/c/d annotations: removing its
        # footer changes the aspect slightly. Keep the letters rather than crop
        # through them, but permit at most 12 px of symmetric padding per edge.
        content=im.crop(content_box(im)); contained=ImageOps.contain(content,size,Image.Resampling.LANCZOS)
        if max(size[0]-contained.width,size[1]-contained.height)<=24:
            return ImageOps.pad(content,size,method=Image.Resampling.LANCZOS,color='white',centering=(0.5,0.5))
    return fit_content(im,size)
def align_to(refnp,img,transforms=None):
    CANW,CANH=refnp.shape[1],refnp.shape[0]
    def remember(M):
        if transforms is not None:
            # Analysis registration maps native source pixels to reference pixels.
            native=np.diag([CANW/img.width,CANH/img.height,1.0])
            transforms.append(np.vstack([M,[0,0,1]])@native)
    src=np.asarray(img.convert('RGB').resize((CANW,CANH),Image.LANCZOS))
    g1=cv2.cvtColor(refnp,cv2.COLOR_RGB2GRAY); g2=cv2.cvtColor(src,cv2.COLOR_RGB2GRAY)
    mk=np.zeros_like(g1); mk[int(CANH*0.12):int(CANH*0.86),int(CANW*0.10):int(CANW*0.90)]=255
    try:
        # ignore the frame furniture (view tabs, badge) near the borders: it is identical in every render and would pin the alignment to identity
        orb=cv2.ORB_create(2000); k1,d1=orb.detectAndCompute(g1,mk); k2,d2=orb.detectAndCompute(g2,mk)
        ms=sorted(cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True).match(d2,d1),key=lambda m:m.distance)[:400]
        if len(ms)>=30:
            p2=np.float32([k2[m.queryIdx].pt for m in ms]); p1=np.float32([k1[m.trainIdx].pt for m in ms])
            M,inl=cv2.estimateAffinePartial2D(p2,p1,ransacReprojThreshold=6)
            if M is not None and inl.sum()>20:
                remember(M)
                return cv2.warpAffine(src,M,(CANW,CANH),borderValue=(245,245,245))
    except Exception: pass
    try:   # fallback: intensity-based affine registration on downscaled edge maps (works when features do not match across renderers)
        sc=400/CANW; e1=cv2.Canny(g1,60,140); e2=cv2.Canny(g2,60,140); e1[mk==0]=0; e2[mk==0]=0
        a=cv2.resize(e1,None,fx=sc,fy=sc).astype(np.float32)/255; b=cv2.resize(e2,None,fx=sc,fy=sc).astype(np.float32)/255
        a=cv2.GaussianBlur(a,(0,0),3); b=cv2.GaussianBlur(b,(0,0),3); M=np.eye(2,3,dtype=np.float32)
        cc,M=cv2.findTransformECC(a,b,M,cv2.MOTION_AFFINE,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,200,1e-5),None,5)
        if cc>0.3:
            M[:,2]/=sc; remember(cv2.invertAffineTransform(M))
            return cv2.warpAffine(src,M,(CANW,CANH),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,borderValue=(245,245,245))
    except Exception: pass
    remember(np.eye(2,3))
    return src
def native_detail(im,transform,box,size,trim=False):
    """Sample the original registered window directly from the native render.

    Keep the scientific registration intact. The only layout transform is a
    uniform, centered cover fit in that existing reference coordinate system.
    No downscaled analysis pixels are used to draw the magnified detail.
    """
    x,y,w,h=box; inv=np.linalg.inv(transform)
    # First extract the complete chosen field at native sampling density. Even
    # high-resolution overrides never pass through the downscaled score canvas.
    density=max(1.0,float(np.linalg.svd(inv[:2,:2],compute_uv=False)[0]))
    native_size=(int(np.ceil(w*density)),int(np.ceil(h*density)))
    sample=np.array([[w/native_size[0],0,x],[0,h/native_size[1],y],[0,0,1]])
    coeffs=(inv@sample)[:2].ravel()
    patch=im.transform(native_size,Image.Transform.AFFINE,coeffs,Image.Resampling.BICUBIC,fillcolor=(245,245,245))
    # Paired sheets must also remove screenshot/registration margins within a
    # selected window. Full comparison detail fields retain their existing frame.
    if trim: return cover(patch,size)
    return ImageOps.fit(patch,size,method=Image.Resampling.LANCZOS,centering=(0.5,0.5))
SPEC={'city-full':[0.40,0.32,0.26,0.22,0.18,0.16],'medieval-village':[0.50,0.40,0.32,0.26,0.20,0.16],'japan-island':[0.55,0.55,0.38,0.38]}
SPEC_OURS=[0.5,0.42,0.34,0.34]   # reference-vs-ours figures: four windows, side by side under the two big views
FINAL_ONLY=True  # Final paper: leave unfinished methods blank, without * or pending labels.
OVERRIDE=dict(kv.split('=',1) for kv in os.environ.get('RCWM_OURS_OVERRIDE','').split(',') if '=' in kv)  # 'scene=/path/render.png': force the Ours render used in that scene's figure
def load(SC,name,fn):
    if fn is None: return Image.open(f'{C}/{SC}.png').convert('RGB'),True
    if name=='Ours' and SC in OVERRIDE: return Image.open(OVERRIDE[SC]).convert('RGB'),True
    p,fin=fn(ROOT,SC)
    if not p or (FINAL_ONLY and not fin): return None,False
    try: return Image.open(p).convert('RGB'),fin
    except Exception: return None,False
def one_scene(SC,out,PAD=10,LBL=66,GUT=84,PANELS=PANELS,CW=760,WSL=None,RH=None):
    WSL=WSL or SPEC.get(SC,[0.55,0.55])
    imgs={}; fins={}
    for name,fn in PANELS:
        im,fin=load(SC,name,fn); imgs[name]=im; fins[name]=fin   # full frames, same framing for every column
    ref=imgs['Reference']; CANW=int(os.environ.get('RCWM_CASE_CANW','1280')); CANH=int(ref.height*CANW/ref.width)   # alignment canvas; raise it when high-resolution renders are available
    refnp=np.asarray(ref.resize((CANW,CANH),Image.LANCZOS)); aligned={'Reference':refnp}
    transforms={'Reference':np.diag([CANW/ref.width,CANH/ref.height,1.0])}
    for name,_ in PANELS:
        if name!='Reference' and imgs.get(name) is not None:
            found=[]; aligned[name]=align_to(refnp,imgs[name],found); transforms[name]=found[0]
    edges=cv2.Canny(refnp,80,160)>0
    reff=refnp.astype(float)/255; ourf=aligned.get('Ours'); ourf=ourf.astype(float)/255 if ourf is not None else None
    blf=[aligned[k].astype(float)/255 for k in aligned if k not in ('Reference','Ours')]
    rects=[]  # (fx,fy,WS); keep window centers at least 0.4 * the large window side apart; zoomed windows must not share its center.
    def far(fx,fy,WS):
        return all(abs(fx+WS/2-(a+w/2))>=0.4*max(WS,w) or abs(fy+WS/2-(b+w/2))>=0.4*max(WS,w) for a,b,w in rects)
    for WS in WSL:
        wpx,hpx=int(CANW*WS),int(CANH*WS); cands=[]
        for fy in np.linspace(0,1-WS,7):
            for fx in np.linspace(0,1-WS,7):
                x,y=int(fx*CANW),int(fy*CANH); ed=edges[y:y+hpx,x:x+wpx].mean(); r=reff[y:y+hpx,x:x+wpx]
                so=ssim(r,ourf[y:y+hpx,x:x+wpx],channel_axis=2,data_range=1.0) if ourf is not None else 0
                sb=max((ssim(r,b[y:y+hpx,x:x+wpx],channel_axis=2,data_range=1.0) for b in blf),default=0.0)
                cands.append((so-sb,so,ed,fx,fy))
        if not blf: pick=next(((fx,fy,WS) for gap,so,ed,fx,fy in sorted(cands,key=lambda c:-c[2]) if far(fx,fy,WS)),None)   # no baselines: most structured windows, spread out
        else: pick=next(((fx,fy,WS) for gap,so,ed,fx,fy in sorted(cands,key=lambda c:-c[0]) if ed>=(0.06 if WS>=0.5 else 0.12) and so>=0.3 and far(fx,fy,WS)),None)
        if pick is None: pick=next(((fx,fy,WS) for gap,so,ed,fx,fy in sorted(cands,key=lambda c:-c[2]) if far(fx,fy,WS)),None)
        if pick is None: pick=max(cands,key=lambda c:c[2])[3:]+(WS,)
        rects.append(pick)
    # hand-picked windows: RCWM_CASE_FIX='scene=idx:fx,fy,ws;idx:fx,fy,ws' replaces the automatic choice at those positions
    for kv in os.environ.get('RCWM_CASE_FIX','').split('|'):
        if kv.startswith(SC+'='):
            for item in kv.split('=',1)[1].split(';'):
                i,vals=item.split(':'); fx,fy,ws=map(float,vals.split(',')); i=int(i)
                while len(rects)<=i: rects.append(rects[-1])
                rects[i]=(fx,fy,ws)
    print('windows',SC,[(round(a,3),round(b,3),c) for a,b,c in rects],file=sys.stderr)
    H1=RH if RH is not None else int(CW*ref.height/ref.width)
    cols=[]
    for name,_ in PANELS:
        im=imgs[name]
        if im is None:
            w=Image.new('RGB',(CW,H1),'white')
        else: w=cover(im,(CW,H1),preserve_frame=SC in ('snow-village','island-harbor','japan-island','medieval-village'))
        cols.append((name,w,fins[name]))
    rows=[]; heights=[]
    for fx,fy,WS in rects:
        wpx,hpx=int(CANW*WS),int(CANH*WS); x,y=int(fx*CANW),int(fy*CANH); row=[]
        rh=RH if RH is not None else int(hpx*CW/wpx); heights.append(rh)
        for name,_ in PANELS:
            a=aligned.get(name)
            if a is None: row.append(None); continue
            row.append(native_detail(imgs[name],transforms[name],(x,y,wpx,hpx),(CW,rh),trim=RH is not None))
        rows.append(row)
    W=GUT+len(cols)*(CW+PAD)+PAD
    H=LBL+H1+sum(heights)+PAD*max(0,len(rows)-1)+PAD
    sheet=Image.new('RGB',(W,int(H)),'white'); d=ImageDraw.Draw(sheet); x=GUT
    header_y=6-min(FONT.getbbox(name)[1] for name,_,_ in cols)
    for name,w,fin in cols:
        lab=name; tw=d.textlength(lab,font=FONT)
        d.text((x+(CW-tw)//2,header_y),lab,fill='black',font=FONT); sheet.paste(w,(x,LBL)); x+=CW+PAD
    def gutter(text,h,y):
        g=Image.new('RGB',(h,GUT-14),'white'); gd=ImageDraw.Draw(g); tw=gd.textlength(text,font=FONT2)
        gd.text(((h-tw)//2,6),text,fill='black',font=FONT2); sheet.paste(g.rotate(90,expand=True),(4,y))
    gutter('Whole',H1,LBL); y=LBL+H1; starts=[LBL]
    for k,row in enumerate(rows):
        rh=heights[k]; starts.append(y); gutter(f'Detail {k+1}',rh,y); x=GUT
        for c in row:
            if c is not None: sheet.paste(c,(x,y))
            x+=CW+PAD
        y+=rh+PAD
    sheet.save(out,optimize=True); print(out,sheet.size,'row starts',starts,'row heights',[H1]+heights,flush=True)
def grid(scenes,out,RH=520,PAD=10,COLLBL=64,ROWLBL=52):
    cells={}
    for SC in scenes:
        for name,fn in PANELS:
            im,fin=load(SC,name,fn)
            if im is None:
                im=Image.new('RGB',(int(RH*1.3),RH),'white')
            else:
                im=cb(im); im=im.resize((int(im.width*RH/im.height),RH),Image.LANCZOS)
            cells[(SC,name)]=im
    colw={name:max(cells[(SC,name)].width for SC in scenes) for name,_ in PANELS}
    W=sum(colw.values())+PAD*(len(PANELS)+1); H=COLLBL+len(scenes)*(RH+PAD+ROWLBL)+PAD
    sheet=Image.new('RGB',(W,H),'white'); d=ImageDraw.Draw(sheet); x=PAD
    for name,_ in PANELS:
        tw=d.textlength(name,font=FONT); d.text((x+(colw[name]-tw)//2,8),name,fill='black',font=FONT); x+=colw[name]+PAD
    y=COLLBL
    for SC in scenes:
        x=PAD
        for name,_ in PANELS:
            im=cells[(SC,name)]; sheet.paste(im,(x+(colw[name]-im.width)//2,y)); x+=colw[name]+PAD
        d.text((PAD,y+RH+4),SC,fill='black',font=FONT2); y+=RH+PAD+ROWLBL
    sheet.save(out,optimize=True); print(out)
if __name__=='__main__':
    # make_case_figures.py <outdir> [--full S1,S2] [--ours S3,S4,...] [crops]
    F=sys.argv[1]; args=sys.argv[2:]
    full=[]; ours=[]; want=[]
    i=0
    while i<len(args):
        if args[i]=='--full': full=args[i+1].split(','); i+=2
        elif args[i]=='--ours': ours=args[i+1].split(','); i+=2
        else: want.append(args[i]); i+=1
    if not full and not ours and not want: want=['city-full','snow-village','island-harbor','medieval-village','japan-island','crops']
    for SC in full+[w for w in want if w!='crops']: one_scene(SC,f'{F}/case-{SC}.png')
    for SC in ours: one_scene(SC,f'{F}/case-ours-{SC}.png',PANELS=[('Reference',None),('Ours',pickers.ours)],CW=1100,WSL=SPEC_OURS,RH=756)
    if 'crops' in want: grid(['school-block','police-corner','park-lake','shop-row','valley-village'],f'{F}/case-grid-crops.png')
