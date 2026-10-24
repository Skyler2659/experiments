"""Wait for stationary RGB frames to settle without advancing physics."""
import time
import numpy as np

def wait_ready(render, read, *, initial=False, timeout=180):
 start=time.monotonic();previous=None;stable=0;history=[];minimum=80 if initial else 5
 for frame in range(1000):
  render();data=read();images=[]
  for r in (data.values() if isinstance(data,dict) else data):
   im=np.asarray(r['rgb'][0]['data'])[:,:,:3].astype(float);h,w=im.shape[:2];im=im[:h-h%8,:w-w%8];images.append(im.reshape(h//8,8,w//8,8,3).mean(axis=(1,3)))
  valid=all(im.size and im.std()>1 for im in images)
  delta=max((float(np.mean(abs(a-b))) for a,b in zip(images,previous)),default=1e9) if previous is not None else 1e9
  if frame%10==0:print('RENDER_SETTLE',frame+1,'delta',round(delta,3),'elapsed',round(time.monotonic()-start,2),flush=True)
  stable=stable+1 if valid and delta<=1.0 else 0
  history.append({'frame':frame+1,'elapsed_s':time.monotonic()-start,'max_rgb_mae':delta if np.isfinite(delta) else None})
  if frame+1>=minimum and (not initial or time.monotonic()-start>=10) and stable>=5:return data,{'frames':frame+1,'elapsed_s':time.monotonic()-start,'stable_frames':stable,'tail':history[-8:]}
  if time.monotonic()-start>timeout:raise TimeoutError('RGB rendering did not stabilize: '+str(history[-8:]))
  previous=images
 raise TimeoutError('RGB frame limit before stability')
