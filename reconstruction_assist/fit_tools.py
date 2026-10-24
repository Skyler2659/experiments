"""Fit generic shape hypotheses requested by a model; never choose an object strategy."""
import json
from pathlib import Path
import numpy as np
from agentic_mesh_world.continuous_rigid_reconstruction import fit_profile
from agentic_mesh_world.static_asset_geometry import box_part,shelf_part

def execute(requests,evidence_dir):
    root=Path(evidence_dir);result=[]
    available=json.loads((root/'model/evidence.json').read_text())['objects']
    for req in requests:
        oid=req['evidence_id'];family=req['family'];key=req['id']
        if oid not in available:raise ValueError('Unknown evidence ID')
        xyz=np.load(root/'samples'/(oid+'.npz'))['xyz_W_m']
        try:
            if family in {'bottle','bowl','plate'}:
                profile,origin,assumptions=fit_profile(xyz,family);t=np.eye(4);t[:3,3]=origin
                parts=[{'kind':'lathe','profile_radius_z_m':profile.tolist(),'T_W_part':t.tolist()}]
            elif family=='upright_box':
                part,_=box_part(xyz);parts=[part];assumptions=['upright rectangular visible envelope; unseen extent uncertain']
            elif family=='tilted_slab':
                thickness=float(req.get('thickness_m',.007))
                if not .001<=thickness<=.03:raise ValueError('Thickness outside tool limits')
                part,_=shelf_part(xyz,thickness);parts=[part];assumptions=['observed tilted top; chosen hidden underside thickness']
            else:raise ValueError('Unsupported fitting family '+family)
            result.append({'id':key,'evidence_id':oid,'status':'CANDIDATE','recipe':{'parts':parts,'assumptions':assumptions},'frame':'W','fit_views':available[oid]['fit_view_ids']})
        except ValueError as e:result.append({'id':key,'evidence_id':oid,'status':'FAILED','reason':str(e)})
    return {'results':result,'scope':'conditional candidates; model selects and can revise program parameters; no old object mesh'}
