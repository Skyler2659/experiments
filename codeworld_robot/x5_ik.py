"""Continuous seeded IK using the public X5 URDF, in the CodeWorld frame."""
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
class X5IK:
 def __init__(self,path,arm=0,base=None,tcp_offset=None):
  xml=ET.parse(path).getroot();self.chain=[];self.limits=[]
  self.tcp_offset=np.asarray([.145,0,0] if tcp_offset is None else tcp_offset,dtype=float)
  for i in range(1,7):
   j=xml.find(f"joint[@name='joint{i}']");limit=j.find('limit');self.limits.append((float(limit.get('lower')),float(limit.get('upper'))));o=j.find('origin');T=np.eye(4);T[:3,:3]=R.from_euler('xyz',np.fromstring(o.get('rpy','0 0 0'),sep=' ')).as_matrix();T[:3,3]=np.fromstring(o.get('xyz'),sep=' ');self.chain.append((T,np.fromstring(j.find('axis').get('xyz'),sep=' ')))
  self.base=np.eye(4);self.base[:3,:3]=R.from_euler('z',np.pi/2).as_matrix();self.base[:3,3]=[-.3 if arm==0 else .3,0,0]
  if base is not None:self.base=np.asarray(base,dtype=float)
 def fk(self,q):
  q=np.asarray(q,dtype=float)
  if q.shape!=(6,) or not np.isfinite(q).all():raise ValueError('Expected six finite X5 joint angles')
  T=self.base.copy()
  for angle,(origin,axis) in zip(q,self.chain):
   A=np.eye(4);A[:3,:3]=R.from_rotvec(axis*angle).as_matrix();T=T@origin@A
  T[:3,3]+=T[:3,:3]@self.tcp_offset;return T
 def solve_candidates(self,target,current):
  current=np.asarray(current,dtype=float);self.fk(current)
  target=np.asarray(target,dtype=float)
  if target.shape!=(4,4) or not np.isfinite(target).all():raise ValueError('Expected finite target transform')
  def fun(q):
   T=self.fk(q);return np.r_[T[:3,3]-target[:3,3],.25*R.from_matrix(T[:3,:3].T@target[:3,:3]).as_rotvec()]
  lower,upper=np.asarray(self.limits).T
  candidates=[]
  for seed in [current,np.array([-.2,1.6,1.7,-1.6,0,2.8]),np.array([-.6,1.,1.,-1.5,0,-2.])]:
   fit=least_squares(fun,np.clip(seed,lower+1e-6,upper-1e-6),bounds=(lower,upper),max_nfev=180,xtol=1e-9,ftol=1e-9,gtol=1e-9)
   if np.linalg.norm(fun(fit.x))<.001:
    # Equivalent 2*pi configurations are considered ONLY within actual URDF limits.
    q=fit.x.copy()
    for i in range(len(q)):
     equivalents=[q[i]+2*np.pi*k for k in range(-3,4) if lower[i]<=q[i]+2*np.pi*k<=upper[i]]
     q[i]=min(equivalents,key=lambda v:abs(v-current[i]))
    if not any(np.linalg.norm(q-old)<1e-4 for old in candidates):candidates.append(q)
  return sorted(candidates,key=lambda q:(np.max(abs(q-current)),np.linalg.norm(q-current)))
 def solve(self,target,current):
  candidates=self.solve_candidates(target,current)
  return candidates[0] if candidates else None
