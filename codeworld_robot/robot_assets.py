"""Bindings for known robot geometry. No task-object geometry is exported here."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from .kinematics import URDFKinematics


def franka_model(urdf,base=None,tcp_offset=.1034):
    t=np.eye(4);t[2,3]=tcp_offset
    return URDFKinematics(urdf,joint_names=['panda_joint%d'%i for i in range(1,8)],
        finger_names=['panda_finger_joint1','panda_finger_joint2'],tcp_link='panda_hand',T_link_tcp=t,base=base)


def export_mujoco_robot(sim,*,joint_names,finger_names,root_body,tcp_site,output,T_W_S,tcp_rotation_body=None):
    """Export only public robot descendants from the compiled native robot model.

    This preserves the exact mount, fixed bodies and grip-site frame used by LIBERO,
    instead of assuming that another Panda URDF has an identical tool frame.
    """
    import trimesh
    m=sim.model;d=sim.data;output=Path(output);output.mkdir(parents=True,exist_ok=True)
    root_id=m.body_name2id(root_body);ids={root_id}
    for i in range(m.nbody):
        if int(m.body_parentid[i]) in ids:ids.add(i)
    def transform(p,q):
        t=np.eye(4);t[:3,3]=p;t[:3,:3]=Rotation.from_quat(np.asarray(q)[[1,2,3,0]]).as_matrix();return t
    def attributes(t):return {'xyz':' '.join(map(str,t[:3,3])),'rpy':' '.join(map(str,Rotation.from_matrix(t[:3,:3]).as_euler('xyz')))}
    xml=ET.Element('robot',name='libero_panda_public');body_names={i:m.body_id2name(i) or 'body_%d'%i for i in ids}
    for i in sorted(ids):
        link=ET.SubElement(xml,'link',name=body_names[i])
        for gi in range(int(m.body_geomadr[i]),int(m.body_geomadr[i]+m.body_geomnum[i])):
            if not (m.geom_contype[gi] or m.geom_conaffinity[gi]):continue
            kind=int(m.geom_type[gi]);size=m.geom_size[gi]
            if kind==7:
                mid=int(m.geom_dataid[gi]);v=int(m.mesh_vertadr[mid]);nv=int(m.mesh_vertnum[mid]);f=int(m.mesh_faceadr[mid]);nf=int(m.mesh_facenum[mid])
                mesh=trimesh.Trimesh(vertices=np.array(m.mesh_vert[v:v+nv]),faces=np.array(m.mesh_face[f:f+nf]),process=False)
            elif kind==6:mesh=trimesh.creation.box(2*size)
            elif kind==2:mesh=trimesh.creation.icosphere(subdivisions=2,radius=size[0])
            elif kind==5:mesh=trimesh.creation.cylinder(radius=size[0],height=2*size[1])
            elif kind==3:
                mesh=trimesh.creation.capsule(radius=size[0],height=2*size[1]);mesh.apply_translation(-mesh.bounds.mean(0))
            elif kind==4:mesh=trimesh.creation.icosphere(subdivisions=2);mesh.apply_scale(size)
            else:raise ValueError('Unsupported robot geom type '+str(kind))
            filename='geom_%d.stl'%gi;mesh.export(output/filename)
            col=ET.SubElement(link,'collision');ET.SubElement(col,'origin',**attributes(transform(m.geom_pos[gi],m.geom_quat[gi])))
            ET.SubElement(ET.SubElement(col,'geometry'),'mesh',filename=filename)
        if i==root_id:continue
        joints=list(range(int(m.body_jntadr[i]),int(m.body_jntadr[i]+m.body_jntnum[i])))
        if len(joints)>1:raise ValueError('Multiple joints on a robot body need an explicit binding')
        body_t=transform(m.body_pos[i],m.body_quat[i]);parent=body_names[int(m.body_parentid[i])]
        if joints:
            j=joints[0];kind={3:'revolute',2:'prismatic'}.get(int(m.jnt_type[j]))
            if kind is None:raise ValueError('Unsupported robot joint type')
            # Joint axes are expressed in the body frame, with rotation about jnt_pos.
            offset=np.eye(4);offset[:3,3]=m.jnt_pos[j]
            intermediate=body_names[i]+'_joint_frame';ET.SubElement(xml,'link',name=intermediate)
            elem=ET.SubElement(xml,'joint',name=m.joint_id2name(j),type=kind)
            ET.SubElement(elem,'parent',link=parent);ET.SubElement(elem,'child',link=intermediate)
            ET.SubElement(elem,'origin',**attributes(body_t@offset));ET.SubElement(elem,'axis',xyz=' '.join(map(str,m.jnt_axis[j])))
            lo,hi=m.jnt_range[j];ET.SubElement(elem,'limit',lower=str(lo),upper=str(hi),effort='1000',velocity='10')
            fixed=ET.SubElement(xml,'joint',name=body_names[i]+'_body_fixed',type='fixed')
            ET.SubElement(fixed,'parent',link=intermediate);ET.SubElement(fixed,'child',link=body_names[i]);ET.SubElement(fixed,'origin',**attributes(np.linalg.inv(offset)))
        else:
            elem=ET.SubElement(xml,'joint',name=body_names[i]+'_fixed',type='fixed')
            ET.SubElement(elem,'parent',link=parent);ET.SubElement(elem,'child',link=body_names[i]);ET.SubElement(elem,'origin',**attributes(body_t))
    path=output/'robot.urdf';ET.ElementTree(xml).write(str(path),encoding='utf-8',xml_declaration=True)
    sid=m.site_name2id(tcp_site) if isinstance(tcp_site,str) else int(tcp_site)
    base=np.eye(4);base[:3,3]=d.body_xpos[root_id];base[:3,:3]=d.body_xmat[root_id].reshape(3,3)
    tcp=transform(m.site_pos[sid],m.site_quat[sid])
    if tcp_rotation_body is not None:
        # robosuite eef_pos is grip_site, while eef_quat is eef BODY orientation.
        bid=m.body_name2id(tcp_rotation_body)
        tcp[:3,:3]=d.body_xmat[int(m.site_bodyid[sid])].reshape(3,3).T@d.body_xmat[bid].reshape(3,3)
    model=URDFKinematics(path,joint_names=joint_names,finger_names=finger_names,root_link=root_body,
        tcp_link=body_names[int(m.site_bodyid[sid])],T_link_tcp=tcp,base=np.asarray(T_W_S)@base)
    return model
