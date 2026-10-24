"""Public robot-only URDF kinematics, independent of simulator task state."""
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares


def origin(element):
    t=np.eye(4)
    if element is not None:
        t[:3,3]=np.fromstring(element.get('xyz','0 0 0'),sep=' ')
        t[:3,:3]=Rotation.from_euler('xyz',np.fromstring(element.get('rpy','0 0 0'),sep=' ')).as_matrix()
    return t


def mesh_path(urdf, filename, packages=None):
    path=Path(urdf).resolve()
    if filename.startswith('package://'):
        package,relative=filename[len('package://'):].split('/',1)
        if package in (packages or {}):return Path(packages[package])/relative
        for parent in path.parents:
            if parent.name==package:return parent/relative
        raise ValueError('Unknown URDF mesh package: '+package)
    return path.parent/(filename[7:] if filename.startswith('file://') else filename)


class URDFKinematics:
    def __init__(self,urdf,*,joint_names=None,finger_names=None,root_link=None,tcp_link=None,T_link_tcp=None,base=None):
        self.path=Path(urdf);self.xml=ET.parse(urdf).getroot();self.joints=[];self.limits_by_name={}
        for j in self.xml.findall('joint'):
            name=j.get('name');kind=j.get('type');p=j.find('parent').get('link');c=j.find('child').get('link')
            a=j.find('axis');axis=np.fromstring(a.get('xyz','1 0 0'),sep=' ') if a is not None else np.array([1.,0,0])
            if np.linalg.norm(axis)==0:raise ValueError('Zero URDF joint axis')
            self.joints.append((name,kind,p,c,origin(j.find('origin')),axis/np.linalg.norm(axis),j.find('mimic')))
            limit=j.find('limit')
            if kind in ('revolute','prismatic','continuous'):
                self.limits_by_name[name]=(-2*np.pi,2*np.pi) if kind=='continuous' or limit is None else (float(limit.get('lower')),float(limit.get('upper')))
        roots={x.get('name') for x in self.xml.findall('link')}-{j[3] for j in self.joints}
        if root_link is None:
            if len(roots)!=1:raise ValueError('Expected a single URDF root')
            root_link=next(iter(roots))
        self.root_link=root_link
        self.joint_names=list(joint_names) if joint_names is not None else [j[0] for j in self.joints if j[1] in ('revolute','continuous')]
        self.finger_names=list(finger_names) if finger_names is not None else [j[0] for j in self.joints if j[1]=='prismatic' and j[6] is None]
        if any(n not in self.limits_by_name for n in self.joint_names+self.finger_names):raise ValueError('Unknown movable joint binding')
        if len(set(self.joint_names+self.finger_names))!=len(self.joint_names+self.finger_names):raise ValueError('Joint bindings overlap')
        self.limits=[self.limits_by_name[n] for n in self.joint_names]
        children={j[0]:j[3] for j in self.joints}
        self.tcp_link=tcp_link or ('link6' if 'link6' in {x.get('name') for x in self.xml.findall('link')} else (children[self.joint_names[-1]] if self.joint_names else self.root_link))
        self.T_link_tcp=np.eye(4) if T_link_tcp is None else np.asarray(T_link_tcp,float)
        self.base=np.eye(4) if base is None else np.asarray(base,float)
        self.packages={}

    def transforms(self,q,fingers=None):
        q=np.asarray(q,float)
        if q.shape!=(len(self.joint_names),) or not np.isfinite(q).all():raise ValueError('Expected finite bound robot joint vector')
        fingers=np.zeros(len(self.finger_names)) if fingers is None else np.asarray(fingers,float)
        if fingers.shape!=(len(self.finger_names),) or not np.isfinite(fingers).all():raise ValueError('Invalid bound finger vector')
        values=dict(zip(self.joint_names,q));values.update(zip(self.finger_names,fingers))
        for name,_,_,_,_,_,mimic in self.joints:
            if mimic is not None and name not in values:values[name]=values[mimic.get('joint')]*float(mimic.get('multiplier','1'))+float(mimic.get('offset','0'))
        tf={self.root_link:self.base};pending=list(self.joints)
        while pending:
            ready=[j for j in pending if j[2] in tf]
            if not ready:raise ValueError('Disconnected URDF or invalid root binding')
            for j in ready:
                name,kind,p,c,t,axis,_=j;a=np.eye(4)
                if kind in ('revolute','continuous'):a[:3,:3]=Rotation.from_rotvec(axis*values[name]).as_matrix()
                elif kind=='prismatic':a[:3,3]=axis*values[name]
                elif kind!='fixed':raise ValueError('Unsupported URDF joint '+kind)
                tf[c]=tf[p]@t@a;pending.remove(j)
        return tf

    def fk(self,q,fingers=None):return self.transforms(q,fingers)[self.tcp_link]@self.T_link_tcp

    def solve_candidates(self,target,current):
        target=np.asarray(target,float);current=np.asarray(current,float);self.fk(current)
        if target.shape!=(4,4) or not np.isfinite(target).all():raise ValueError('Invalid target transform')
        lower,upper=np.array(self.limits).T;middle=(lower+upper)/2;span=upper-lower
        def residual(q):
            t=self.fk(q);return np.r_[t[:3,3]-target[:3,3],.25*Rotation.from_matrix(t[:3,:3].T@target[:3,:3]).as_rotvec()]
        seeds=[current,middle,(current+middle)/2]
        seeds += [middle+.23*span*np.sin(np.arange(len(current))+phase) for phase in (0.,1.7,3.4)]
        candidates=[]
        for seed in seeds:
            fit=least_squares(residual,np.clip(seed,lower+1e-8,upper-1e-8),bounds=(lower,upper),max_nfev=160,xtol=1e-9,ftol=1e-9,gtol=1e-9)
            q=fit.x;t=self.fk(q)
            if np.linalg.norm(t[:3,3]-target[:3,3])>.001 or Rotation.from_matrix(t[:3,:3].T@target[:3,:3]).magnitude()>.005:continue
            for i in range(len(q)):
                equivalent=[q[i]+2*np.pi*k for k in range(-3,4) if lower[i]<=q[i]+2*np.pi*k<=upper[i]]
                q[i]=min(equivalent,key=lambda v:abs(v-current[i]))
            if not any(np.linalg.norm(q-old)<1e-4 for old in candidates):candidates.append(q)
        return sorted(candidates,key=lambda q:(np.max(abs(q-current)),np.linalg.norm(q-current)))

    def solve(self,target,current):
        candidates=self.solve_candidates(target,current);return candidates[0] if candidates else None

    def finger_geometry(self):
        """Public finger joint geometry in the bound TCP frame (zero coordinates).

        Axis sign is increasing joint coordinate, not a claim about CLOSE.
        Contact surfaces remain the named links' URDF collision meshes.
        """
        q=np.zeros(len(self.joint_names));f=np.zeros(len(self.finger_names))
        transforms=self.transforms(q,f);tcp=self.fk(q,f);rows=[]
        for name,kind,parent,child,origin,axis,_ in self.joints:
            if name not in self.finger_names:continue
            joint=transforms[parent]@origin
            rows.append({'joint':name,'type':kind,'child_link':child,
                'axis_tcp_increasing_coordinate':(tcp[:3,:3].T@joint[:3,:3]@axis).tolist(),
                'origin_tcp_at_zero_m':(tcp[:3,:3].T@(joint[:3,3]-tcp[:3,3])).tolist(),
                'limits':list(self.limits_by_name[name])})
        return {'reference':'zero_finger_coordinates','frame':'bound_tcp','joints':rows,
            'contact_geometry':'URDF collision meshes on finger links and their fixed descendants',
            'holding_verified':False}

    def binding(self):
        return {'finger_geometry':self.finger_geometry(),'urdf':str(self.path),'joint_names':self.joint_names,'finger_joint_names':self.finger_names,
                'root_link':self.root_link,'tcp_link':self.tcp_link,'T_link_tcp':self.T_link_tcp.tolist(),
                'base_T_W':self.base.tolist(),'joint_limits_rad':self.limits}
