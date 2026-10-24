"""Inspect dominant planar surface inclination in height bands of a mesh.

Height bands are reviewed part regions, not hidden simulator segmentation.
Normal-area evidence is retained, so a horizontal panel cannot pass by name.
"""
import numpy as np

def panel_pitch(mesh, bands=((.15,.53),(.55,.94))):
    triangles=np.asarray(mesh.vertices)[np.asarray(mesh.faces)]
    centers=triangles.mean(axis=1)
    cross=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    length=np.linalg.norm(cross,axis=1);area=length/2
    normals=cross/np.maximum(length[:,None],1e-15)
    lo,hi=mesh.bounds[:,2];height=(centers[:,2]-lo)/(hi-lo)
    angles=np.degrees(np.arccos(np.clip(np.abs(normals[:,2]),0,1)))
    results=[]
    for low,high in bands:
        chosen=(height>=low)&(height<=high)&(angles<75)
        if chosen.sum()<2:raise ValueError('Insufficient panel faces in reviewed height band')
        hist,_=np.histogram(angles[chosen],bins=np.arange(0,77,2),weights=area[chosen]);peak=int(np.argmax(np.convolve(hist,np.ones(5),mode='same')))*2+1
        selected=chosen&(np.abs(angles-peak)<=10);order=np.argsort(angles[selected]);aa=angles[selected][order];ww=area[selected][order];pitch=float(aa[np.searchsorted(np.cumsum(ww),ww.sum()/2)])
        results.append({'height_band':[low,high],'pitch_deg':pitch,'surface_area_support':float(area[selected].sum()),'histogram_area':hist.tolist(),'method':'area-weighted median near dominant plane-normal mode; absolute inclination only'})
    return results

def check_panel_constraints(mesh, constraints):
    reports=[]
    for rule in constraints:
        if rule['kind']!='panel_pitch':raise ValueError('Unsupported structural constraint')
        measured=panel_pitch(mesh,tuple(map(tuple,rule['height_bands'])))
        if len(measured)!=len(rule['expected_deg']):raise ValueError('Panel count/constraint mismatch')
        checks=[]
        for result,expected in zip(measured,rule['expected_deg']):
            checks.append({**result,'expected_deg':expected,'tolerance_deg':rule['tolerance_deg'],'accepted':abs(result['pitch_deg']-expected)<=rule['tolerance_deg']})
        entry={'kind':'panel_pitch','source':rule['evidence'],'checks':checks,'accepted':all(c['accepted'] for c in checks)}
        if rule.get('normal_W') is not None:
            feature=plane_feature(mesh,tuple(rule['feature_band']));normal=np.array(feature['normal']);expected=np.array(rule['normal_W']);expected/=np.linalg.norm(expected)
            angle=float(np.degrees(np.arccos(np.clip(abs(np.dot(normal,expected)),0,1))))
            distance=float(np.linalg.norm(np.array(feature['center'])-rule['center_W']))
            entry.update(normal_error_deg=angle,plane_center_error_m=distance)
            entry['accepted'] &= angle<=rule['tolerance_deg'] and distance<=rule['center_tolerance_m']
        reports.append(entry)
    return reports

def plane_feature(mesh, band=(.55,.94)):
    triangles=np.asarray(mesh.vertices)[np.asarray(mesh.faces)];centers=triangles.mean(1)
    cross=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]);weights=np.linalg.norm(cross,axis=1)/2
    normals=cross/np.maximum(2*weights[:,None],1e-15);normals[normals[:,2]<0]*=-1
    height=(centers[:,2]-mesh.bounds[0,2])/mesh.extents[2]
    pitch=panel_pitch(mesh,(band,))[0]['pitch_deg'];angles=np.degrees(np.arccos(np.clip(normals[:,2],0,1)))
    chosen=(height>=band[0])&(height<=band[1])&(np.abs(angles-pitch)<10)
    # Exclude tiny decorative faces; retain geometry-derived dominant plane.
    n=np.average(normals[chosen],axis=0,weights=weights[chosen]);n/=np.linalg.norm(n)
    center=np.average(centers[chosen],axis=0,weights=weights[chosen])
    return {'normal':n.tolist(),'center':center.tolist(),'band':list(band)}
