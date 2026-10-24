import numpy as np
from scipy.spatial.transform import Rotation

def transform(p):
    if np.asarray(p['translation_m']).shape!=(3,) or np.asarray(p['quaternion_xyzw']).shape!=(4,):raise ValueError('Expected xyz translation and xyzw quaternion')
    t=np.eye(4);t[:3,3]=p['translation_m'];t[:3,:3]=Rotation.from_quat(p['quaternion_xyzw']).as_matrix()
    if not np.isfinite(t).all():raise ValueError('Pose must be finite')
    return t

def pose(t):return {'translation_m':np.asarray(t)[:3,3].tolist(),'quaternion_xyzw':Rotation.from_matrix(np.asarray(t)[:3,:3]).as_quat().tolist()}
