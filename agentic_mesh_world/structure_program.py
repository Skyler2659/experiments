"""Stage two: recursive, textured part programs over a frozen reconstruction.

No generators or simulator state are called here. Selectors partition source
triangles; unknown triangles are retained. Program edits and rendering share
one part hierarchy. Structural hypotheses are distinct from metric truth.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from .mesh_world import MeshCodeWorld, sha256, nearest_mesh_surface
from .panel_geometry import plane_feature


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def freeze(scene, output):
    """Freeze the stage-one representation, including every referenced asset."""
    scene, output = Path(scene).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("freeze output already exists; preserve previous versions")
    world = MeshCodeWorld.load(scene)
    output.mkdir(parents=True)
    shutil.copy2(scene, output / "scene.json")
    for node in world.source["nodes"]:
        ref = node["asset"]["mesh_ref"]
        target = output / ref
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(world.root / ref, target)
        support = node["asset"].get("provenance", {}).get("support_ref")
        if support:
            support_path = Path(support)
            if support_path.is_absolute() or ".." in support_path.parts: raise ValueError("invalid support evidence path")
            (output / support_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(world.root / support_path, output / support_path)
    if (world.root / "evidence").is_dir():
        shutil.copytree(world.root / "evidence", output / "evidence")
    if (world.root / 'robot').is_dir():
        shutil.copytree(world.root/'robot',output/'robot',dirs_exist_ok=True)
    # Surface ownership and exported local-to-source face IDs are code authority.
    identity_paths=[]
    for node in world.source['nodes']:
        provenance=node['asset'].get('provenance',{})
        for key in ('source_face_ref','source_mesh_ref','ownership_ref','surface_support_ref'):
            ref=provenance.get(key)
            if not ref:continue
            rel=Path(ref)
            if rel.is_absolute() or '..' in rel.parts:raise ValueError('Invalid ownership reference')
            dst=output/rel;dst.parent.mkdir(parents=True,exist_ok=True)
            if not dst.exists():shutil.copy2(world.root/rel,dst)
            identity_paths.append(dst)
    if (world.root/'source').is_dir():
        shutil.copytree(world.root/'source',output/'source',dirs_exist_ok=True)
        identity_paths.extend(p for p in (output/'source').rglob('*') if p.is_file())
    receipt=world.root.parent/'acceptance.json'
    if receipt.is_file():
        record=json.loads(receipt.read_text())
        if record.get('scene_sha256')!=sha256(scene):raise ValueError('acceptance belongs to another source scene')
        shutil.copy2(receipt,output/'acceptance.json');identity_paths.append(output/'acceptance.json')
    code_dir=output/"stage_one_code";code_dir.mkdir()
    for source_file in sorted(Path(__file__).parent.glob('*.py')):
        shutil.copy2(source_file, code_dir/source_file.name)
    # Do not assume assets always live under assets/ or use GLB.
    paths = list(dict.fromkeys([output / "scene.json"] + [output/n["asset"]["mesh_ref"] for n in world.source["nodes"]]))
    paths += [output/n["asset"]["provenance"]["support_ref"] for n in world.source["nodes"] if n["asset"].get("provenance", {}).get("support_ref")]
    paths += [p for folder in (output/"evidence",output/'robot',code_dir) if folder.exists() for p in folder.rglob('*') if p.is_file()]
    paths += identity_paths
    record = {"source_scene": str(scene), "files": {str(p.relative_to(output)): sha256(p) for p in paths},
              "scope": "frozen stage-one geometry/state; not a new reconstruction acceptance"}
    write(output / "freeze.json", record)
    return record


def verify_freeze(root):
    root = Path(root)
    record = json.loads((root / "freeze.json").read_text())
    for name, digest in record["files"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"frozen stage-one input changed: {name}")


def select(mesh, selector, object_reference=None):
    if selector['kind']=='object_box':
        if object_reference is None:raise ValueError('object_box needs source object frame')
        local=mesh.copy();local.apply_transform(np.linalg.inv(object_reference))
        selected,feature=select(local,{**selector,'kind':'box'})
        return selected,{**feature,'kind':'object_box','object_frame_T_W':object_reference.tolist()}
    centers = mesh.triangles_center
    kind = selector["kind"]
    if kind == "remaining":
        return np.ones(len(mesh.faces), dtype=bool), {"kind": kind}
    if kind == "box":
        lo, hi = mesh.bounds
        normalized = (centers-lo)/np.maximum(hi-lo, 1e-12)
        low, high = np.asarray(selector["low"]), np.asarray(selector["high"])
        if low.shape != (3,) or high.shape != (3,) or not np.isfinite([low,high]).all() or np.any(low>high):
            raise ValueError("invalid normalized box")
        return ((normalized >= low)&(normalized <= high)).all(axis=1), {"kind":kind,"low":low.tolist(),"high":high.tolist()}
    if kind == "plane_band":
        f = plane_feature(mesh, tuple(selector["height_band"]))
        n, c = np.asarray(f["normal"]), np.asarray(f["center"])
        thickness = float(selector["half_thickness_m"])
        if not np.isfinite(thickness) or thickness <= 0:
            raise ValueError("plane thickness must be positive")
        return abs((centers-c)@n) <= thickness, {**f,"kind":kind,"half_thickness_m":thickness}
    raise ValueError(f"unsupported selector {kind}")


def _submesh(mesh, ids):
    # trimesh keeps UVs and the source material when selecting faces.
    return mesh.submesh([np.asarray(ids, dtype=int)], append=True, repair=False)


def compile_program(frozen, proposal, output):
    """Execute a scene-specific recursive partition; retain all source faces."""
    frozen, output = Path(frozen).resolve(), Path(output).resolve()
    verify_freeze(frozen)
    if output.exists():
        raise ValueError("candidate output exists")
    world = MeshCodeWorld.load(frozen / "scene.json")
    oid = proposal["object_id"]
    pieces = world.mesh_pieces(oid)
    if len(pieces) != 1:
        raise ValueError("multi-material multi-primitive input needs explicit per-primitive proposals")
    mesh = pieces[0]
    from .structure_features import object_frame
    reference=object_frame(world,oid)
    output.mkdir(parents=True)
    nodes, errors, seen = [], [], set()

    def visit(spec, ids, parent):
        pid = spec["id"]
        if not isinstance(pid,str) or not pid.replace('_','').isalnum() or pid in seen:
            raise ValueError("part IDs must be unique identifiers")
        seen.add(pid)
        if len(ids) == 0:
            errors.append({"part":pid,"reason":"empty part"})
            return
        part = _submesh(mesh, ids)
        node = {"id":pid,"parent":parent,"semantic":spec.get("semantic","unresolved"),
                "evidence":spec.get("evidence",[]),"face_count":len(ids),"area_m2":float(part.area),
                "bounds_W":part.bounds.tolist(),"pivot_W":part.bounds.mean(axis=0).tolist(),
                "parameters":{"translation_m":[0,0,0],"rotation_deg_xyz":[0,0,0],"scale_xyz":[1,1,1]},
                "selector":spec.get("selector"),"unknown":bool(spec.get('unknown',False) or
                    spec.get("semantic")=="unresolved" or 'unresolved' in pid.split('_'))}
        nodes.append(node)
        if spec.get("children"):
            available = np.ones(len(ids),dtype=bool)
            for child in spec["children"]:
                if child['selector']['kind']=='source_face_ranges':
                    ranges=child['selector']['ranges'];selected=np.zeros(len(ids),dtype=bool)
                    for start,stop in ranges:
                        if not (isinstance(start,int) and isinstance(stop,int) and 0<=start<stop<=len(mesh.faces)):
                            raise ValueError('invalid source face range')
                        selected|=(ids>=start)&(ids<stop)
                    feature={'kind':'source_face_ranges','ranges':ranges}
                else:
                    selected, feature = select(part, child["selector"], reference)
                selected &= available
                child = copy.deepcopy(child)
                child["selector_evaluation"] = feature
                visit(child, ids[selected], pid)
                available[selected] = False
            if available.any():
                visit({"id":pid+"_unresolved","semantic":"unresolved","evidence":[]},ids[available],pid)
        else:
            ref = f"parts/{pid}.glb"
            (output / "parts").mkdir(exist_ok=True)
            # Store in world coordinates at rest; transforms are about explicit pivots.
            part.export(output / ref)
            node["mesh_ref"], node["mesh_sha256"] = ref, sha256(output / ref)
            node["source_face_ids"] = ids.tolist()
        if "selector_evaluation" in spec:
            node["selector_evaluation"] = spec["selector_evaluation"]

    visit(proposal["tree"], np.arange(len(mesh.faces)), None)
    program = {"schema_version":"structure-program-v1","source":os.path.relpath(frozen,output),
               "source_digest":sha256(frozen/"freeze.json"),"object_id":oid,"nodes":nodes,
               "relations":proposal.get("relations",[]),"source_faces":len(mesh.faces),
               "proposal_errors":errors,"evidence_status":"STRUCTURAL_HYPOTHESIS",
               "limitations":proposal.get("limitations",[])}
    write(output / "structure.json", program)
    write(output / "proposal.json", proposal)
    # Explicit, per-scene construction calls, rather than one generic load wrapper.
    lines = ["# Generated recursive structure; defaults are the measured mesh partition.",
             "# Edit via verified parameters; source geometry remains frozen.",
             "from pathlib import Path", "from agentic_mesh_world.structure_program import PartAssembly", "",
             "def build(overrides=None):", "    root = Path(__file__).parent",
             "    a = PartAssembly(root, overrides or {})"]
    for n in nodes:
        lines.append(f"    a.part({n['id']!r}, parent={n['parent']!r}, semantic={n['semantic']!r})")
    lines.extend(["    return a.finish()", "", "WORLD = build()", ""])
    (output / "structure.py").write_text("\n".join(lines))
    return program


class PartAssembly:
    """Named scene-program calls are checked against the structural authority."""
    def __init__(self, root, overrides):
        self.world = StructuralWorld(root, overrides)
        self.registered = []

    def part(self, name, *, parent, semantic):
        node = self.world.parts[name]
        if parent != node["parent"] or semantic != node["semantic"] or name in self.registered:
            raise ValueError("construction program disagrees with structural authority")
        if parent is not None and parent not in self.registered:
            raise ValueError("parent must be constructed first")
        self.registered.append(name)

    def finish(self):
        if set(self.registered) != set(self.world.parts):
            raise ValueError("program omitted parts")
        return self.world


class StructuralWorld(MeshCodeWorld):
    def __init__(self, root, overrides=None):
        self.program_root = Path(root).resolve()
        self.program = json.loads((self.program_root/"structure.json").read_text())
        frozen = (self.program_root / self.program["source"]).resolve()
        verify_freeze(frozen)
        if sha256(frozen/"freeze.json") != self.program["source_digest"]:
            raise ValueError("source manifest changed")
        base = MeshCodeWorld.load(frozen/"scene.json")
        super().__init__(base.source, base.root)
        self.parts = {n["id"]:copy.deepcopy(n) for n in self.program["nodes"]}
        for name, patch in (overrides or {}).items():
            if name not in self.parts or set(patch)-{"translation_m","rotation_deg_xyz","scale_xyz"}:
                raise ValueError("unknown part or parameter")
            self.parts[name]["parameters"].update(patch)
        for n in self.parts.values():
            for key,value in n["parameters"].items():
                a=np.asarray(value,dtype=float)
                if a.shape!=(3,) or not np.isfinite(a).all() or (key=="scale_xyz" and np.any(a<=0)):
                    raise ValueError("invalid part transform")
            if "mesh_ref" in n and sha256(self.program_root/n["mesh_ref"])!=n["mesh_sha256"]:
                raise ValueError("part asset changed")

    def part_transform(self, name):
        n=self.parts[name]; p=n["parameters"]; pivot=np.asarray(n["pivot_W"])
        t=np.eye(4)
        t[:3,:3]=Rotation.from_euler("xyz",p["rotation_deg_xyz"],degrees=True).as_matrix()@np.diag(p["scale_xyz"])
        t[:3,3]=pivot-t[:3,:3]@pivot+np.asarray(p["translation_m"])
        return self.part_transform(n["parent"])@t if n["parent"] else t

    def descendants(self,name):
        return [n for n in self.parts if self.parts[n]["parent"]==name]

    def part_pieces(self,name):
        n=self.parts[name]
        if "mesh_ref" in n:
            mesh=trimesh.load(self.program_root/n["mesh_ref"],force="mesh",process=False)
            mesh.apply_transform(self.part_transform(name))
            return [mesh]
        return [m for child in self.descendants(name) for m in self.part_pieces(child)]

    def part_info(self,name):
        n=self.parts[name]; m=trimesh.util.concatenate(self.part_pieces(name))
        t=self.part_transform(name);feature=copy.deepcopy(n.get('selector_evaluation'))
        if feature and 'normal' in feature:
            normal=np.linalg.inv(t[:3,:3]).T@np.asarray(feature['normal']);normal/=np.linalg.norm(normal)
            feature['normal']=normal.tolist();feature['center']=(t@np.r_[feature['center'],1])[:3].tolist()
        return {"id":name,"semantic":n["semantic"],"parent":n["parent"],"children":self.descendants(name),
                "bounds_W":m.bounds.tolist(),"parameters":n["parameters"],"evidence":n["evidence"],
                "size_W_m":m.extents.tolist(),"pivot_W":(t@np.r_[n['pivot_W'],1])[:3].tolist(),"geometric_feature":feature,
                "status":"HYPOTHESIS" if n["unknown"] else "MESH_DERIVED_NOT_CONTACT_VALIDATED"}

    def mesh_pieces(self,object_id):
        if object_id==self.program["object_id"]:
            return [m for n in self.parts.values() if n["parent"] is None for m in self.part_pieces(n["id"])]
        return super().mesh_pieces(object_id)

    def export_glb(self,path):
        scene=trimesh.Scene()
        for node in self.source['nodes']:
            oid=node['object_id']
            if oid==self.program['object_id']:
                for part in self.parts.values():
                    if 'mesh_ref' in part:
                        scene.add_geometry(self.part_pieces(part['id'])[0],node_name=f"{oid}/{part['id']}")
            else:
                for i,piece in enumerate(self.mesh_pieces(oid)):
                    scene.add_geometry(piece,node_name=f'{oid}/{i}')
        scene.export(path)

    def part_surface_distance(self,name,point):
        mesh=trimesh.util.concatenate(self.part_pieces(name))
        nearest,dist=nearest_mesh_surface(mesh,point)
        return {"part":name,"distance_m":dist,"nearest_W":nearest.tolist(),"kind":"unsigned_mesh_surface"}


def verify_program(root, *, overrides=None, max_unknown_fraction=.05):
    """Whole/part/whole checks; hypotheses are not physical certificates."""
    w=StructuralWorld(root,overrides)
    base=MeshCodeWorld.load(w.root/"scene.json")
    source=base.mesh(w.program["object_id"])
    leaves=[n for n in w.parts.values() if "mesh_ref" in n]
    ids=np.concatenate([n["source_face_ids"] for n in leaves])
    coverage=len(ids)==len(source.faces) and np.array_equal(np.sort(ids),np.arange(len(source.faces)))
    unknown=sum(n["area_m2"] for n in leaves if n["unknown"])/source.area
    errors=list(w.program["proposal_errors"])
    if not coverage: errors.append({"reason":"source triangles omitted or duplicated"})
    if unknown>max_unknown_fraction: errors.append({"reason":"unresolved surface area exceeds budget","value":unknown})
    checks=[]
    for r in w.program["relations"]:
        a=trimesh.util.concatenate(w.part_pieces(r["a"]))
        b=trimesh.util.concatenate(w.part_pieces(r["b"]))
        if r["kind"]=="above":
            value=float(a.bounds.mean(axis=0)[2]-b.bounds.mean(axis=0)[2]);ok=value>r.get("minimum_m",0)
        elif r["kind"]=="near":
            # Vertex-to-triangle distance, not centre or AABB overlap.
            pts=a.vertices[::max(1,len(a.vertices)//1000)]
            value=float(trimesh.proximity.closest_point(b,pts)[1].min());ok=value<=r["tolerance_m"]
        else: raise ValueError("unknown relation")
        checks.append({**r,"value_m":value,"accepted":bool(ok)})
        if not ok:errors.append({"reason":"relation failed","relation":r,"value_m":value})
    part_checks=[]
    for n in leaves:
        mesh=trimesh.util.concatenate(w.part_pieces(n["id"]))
        connected=trimesh.graph.connected_components(mesh.face_adjacency,nodes=np.arange(len(mesh.faces)),min_len=1)
        indexed_largest=max((float(mesh.area_faces[c].sum()) for c in connected),default=0)/mesh.area
        # UV/normal seams duplicate vertices in GLB. Diagnose geometric
        # connectivity on a copy; do not alter the textured authority.
        geometric=mesh.copy();geometric.merge_vertices(merge_tex=True,merge_norm=True,digits_vertex=7)
        connected=trimesh.graph.connected_components(geometric.face_adjacency,nodes=np.arange(len(geometric.faces)),min_len=1)
        largest=max((float(geometric.area_faces[c].sum()) for c in connected),default=0)/geometric.area
        part_checks.append({"id":n["id"],"area_m2":float(mesh.area),"largest_component_area_fraction":largest,
                            "indexed_component_area_fraction":indexed_largest,
                            "connectivity_note":"position-welded diagnostic at 1e-7 m; disconnected boundary sheets are not proof of wrong semantic grouping; not a closed-solid guarantee",
                            "textured":mesh.visual.kind=="texture","face_count":len(mesh.faces)})
    # Rest-state surface and UV fidelity: exact partition is checked by source IDs;
    # GLB roundtrip positions also need numerical agreement.
    max_error=0.; max_uv_error=0.; texture_preserved=True
    if not overrides:
        for n in leaves:
            m=trimesh.util.concatenate(w.part_pieces(n["id"]))
            expected=source.triangles[np.asarray(n["source_face_ids"])]
            max_error=max(max_error,float(np.max(abs(m.triangles-expected))))
            if source.visual.kind=='texture':
                if m.visual.kind!='texture':texture_preserved=False
                else:
                    expected_uv=source.visual.uv[source.faces[np.asarray(n['source_face_ids'])]]
                    max_uv_error=max(max_uv_error,float(np.max(abs(m.visual.uv[m.faces]-expected_uv))))
                    def pixels(material):
                        img=getattr(material,'baseColorTexture',None)
                        if img is None:img=getattr(material,'image',None)
                        return np.asarray(img.convert('RGBA')) if img is not None else None
                    texture_preserved &= np.array_equal(pixels(source.visual.material),pixels(m.visual.material))
        if max_error>1e-6:errors.append({"reason":"rest assembly changed source geometry","error_m":max_error})
        if max_uv_error>1e-6 or not texture_preserved:errors.append({'reason':'texture or UV changed'})
    result={"status":"STRUCTURE_CHECKED" if not errors else "NEEDS_REVISION","errors":errors,
            "triangle_partition_exact":bool(coverage),"unresolved_area_fraction":float(unknown),
            "rest_max_coordinate_error_m":max_error if not overrides else None,"relations":checks,"parts":part_checks,
            "rest_max_uv_error":max_uv_error if not overrides else None,"texture_pixels_preserved":bool(texture_preserved) if not overrides else None,
            "scope":"structural partition of frozen candidate, not validation against physical truth"}
    return result


def evaluate_edit(root, overrides, output):
    """Execute a hypothetical part edit without changing the observed package."""
    output=Path(output)
    if output.exists():raise ValueError('edit output exists')
    output.mkdir(parents=True)
    w=StructuralWorld(root,overrides)
    report=verify_program(root,overrides=overrides)
    write(output/'parameters.json',overrides)
    write(output/'verification.json',report)
    write(output/'status.json',{'status':'REJECTED' if report['errors'] else 'NEED_OBSERVATION_REVIEW',
                              'kind':'hypothetical structure edit; no real-world state update'})
    w.export_glb(output/'hypothesis.glb')
    return report


def publish_agent_bundle(root, visual_review):
    root=Path(root)
    report=verify_program(root)
    if report["status"]!="STRUCTURE_CHECKED" or not visual_review.get("accepted"):
        raise ValueError("unresolved geometry or visual review")
    if visual_review.get("program_sha256")!=sha256(root/"structure.json"):
        raise ValueError("review belongs to a different program")
    w=StructuralWorld(root)
    bundle={"schema_version":"structured-agent-world-v1","object_id":w.program["object_id"],
            "program":"structure.py","program_data":"structure.json", "parts":[w.part_info(n) for n in w.parts],
            "relations":report["relations"],"limitations":w.program["limitations"]+[
                "Source reconstruction defects and metric uncertainty are inherited; exact reassembly is not a fresh reconstruction validation.",
                "Parts are triangle-surface partitions; cut interfaces may be open or approximate, and detachable solids are not established.",
                "No physical simulation, world evolution, or downstream manipulation was validated."],
            "available_queries":["part_info","part_pieces","part_surface_distance","bounds","render"],
            "edit_policy":"overrides are hypothetical until verified and separately published",
            "physical_validity":"NOT_VALIDATED"}
    write(root/"verification.json",report);write(root/"visual_review.json",visual_review);write(root/"agent_world.json",bundle)
    return bundle


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest="command",required=True)
    a=s.add_parser("freeze");a.add_argument("--scene",required=True);a.add_argument("--output",required=True)
    a=s.add_parser("compile");a.add_argument("--frozen",required=True);a.add_argument("--proposal",required=True);a.add_argument("--output",required=True)
    a=s.add_parser("verify");a.add_argument("--root",required=True)
    args=p.parse_args()
    if args.command=="freeze":result=freeze(args.scene,args.output)
    elif args.command=="compile":result=compile_program(args.frozen,json.loads(Path(args.proposal).read_text()),args.output)
    else:
        result=verify_program(args.root);write(Path(args.root)/"verification.json",result)
    print(json.dumps({k:v for k,v in result.items() if k not in {"nodes","files"}},ensure_ascii=False)[:6000])


if __name__=="__main__":main()
