"""Export synchronized evidence from recorded frame indices, never inferred timestamps."""
import json
import os
from pathlib import Path
import subprocess
import textwrap
import numpy as np
from PIL import Image, ImageDraw
from .campaign import write


def world_panel(root, snapshot, bounds=None):
    from codeworld_workspace import World
    world = World(root)
    state = json.loads(Path(snapshot).read_text())
    # Fixed orthographic view; all triangles of actual CodeWorld meshes are used.
    basis = np.array([[.707, -.408, .577], [-.707, -.408, .577], [0, .816, .577]])
    meshes = [world._mesh(obj, state) for obj in world.objects() if obj in state['objects']
              and json.loads((world.root/world.manifest['objects'][obj]).read_text()).get('role') != 'robot_snapshot']
    if not meshes:
        raise ValueError('Snapshot has no renderable CodeWorld objects')
    vertices = [np.asarray(m.vertices) @ basis for m in meshes]
    points = np.concatenate(vertices)
    if bounds is None:
        bounds = (points[:, :2].min(0), points[:, :2].max(0))
    low, high = bounds
    scale = 520 / max(float(np.max(high-low)), .01)
    center = (low+high)/2
    triangles = []
    palette = ['#82b8e8','#f2ba70','#91c99d','#d497d9','#ccc48b']
    for i,(mesh, verts) in enumerate(zip(meshes, vertices)):
        xy = (verts[:,:2]-center)*scale
        xy[:,0] += 320
        xy[:,1] = 275-xy[:,1]
        for face in mesh.faces:
            triangles.append((float(verts[face,2].mean()), xy[face], palette[i%len(palette)]))
    result = Image.new('RGB',(640,480),'#17202b');draw=ImageDraw.Draw(result)
    for _,triangle,color in sorted(triangles,key=lambda x:x[0]):
        draw.polygon([tuple(p) for p in triangle],fill=color)
    draw.rectangle((0,0,640,45),fill='#17202b')
    draw.text((12,10),'CodeWorld | latest recorded state (held between updates)',fill='white')
    draw.text((12,28),f"Revision: {state.get('revision','unknown')} | fixed view; static robot mesh omitted",fill='white')
    return result, bounds


def export(attempt):
    import cv2
    import imageio_ffmpeg
    attempt = Path(attempt).resolve()
    root = attempt/'episode';io=root/'robot_io';out=attempt/'artifacts';out.mkdir(exist_ok=True)
    manifest={'status':'INCOMPLETE','clock':'recorded task frames; inference wall time excluded',
              'codeworld_semantics':'last recorded observation state; no interpolation',
              'paths':{'codeworld_code':'episode/source','codeworld_interface':'episode/tools/codeworld_workspace','geometry':'episode/geometry',
                       'agent_events':'agent/events.jsonl','robot_operations':'episode/robot_io/requests',
                       'timeline':'episode/robot_io/timeline.jsonl'}}
    write(out/'manifest.json',manifest)
    timeline=io/'timeline.jsonl';source=io/'continuous.mp4'
    if not timeline.exists() or not source.exists():
        manifest.update(status='UNAVAILABLE',error='No frame-index timeline/video; historical alignment is not fabricated')
        write(out/'manifest.json',manifest);return manifest
    rows=[json.loads(line) for line in timeline.read_text().splitlines()]
    if not rows:raise ValueError('Empty frame timeline')
    if [r['frame'] for r in rows]!=list(range(len(rows))):raise ValueError('Non-contiguous frame timeline')
    job=json.loads((attempt/'job.json').read_text())
    has_world=job['strategy'] in ('ours','gt_codeworld')
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    names=['task','agent_task_sync']+(['codeworld_task_sync'] if has_world else [])
    cap=cv2.VideoCapture(str(source));fps=cap.get(cv2.CAP_PROP_FPS) or 25
    writers={};logs=[]
    try:
        for name in names:
            log=(out/f'{name}.ffmpeg.log').open('wb');logs.append(log)
            width=640 if name=='task' else 1280
            writers[name]=subprocess.Popen([ffmpeg,'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
                '-s',f'{width}x480','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast',
                '-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(out/f'{name}.partial.mp4')],
                stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log)
        previous=None;panel=None;bounds=None
        for row in rows:
            ok,frame=cap.read()
            if not ok:raise ValueError(f"Video ended before timeline frame {row['frame']}")
            task=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).resize((640,480))
            writers['task'].stdin.write(task.tobytes())
            agent=Image.new('RGB',(640,480),'#17202b');draw=ImageDraw.Draw(agent)
            operation=json.dumps(row['operation'],ensure_ascii=True,indent=2)
            lines=['Agent operation | recorded request / native action',f"Frame {row['frame']} | {row['clock']}",'Full request and response retained in robot_io/']
            lines += textwrap.wrap(operation,82)[:23]
            draw.multiline_text((12,12),'\n'.join(lines),fill='white',spacing=5)
            pair=Image.new('RGB',(1280,480));pair.paste(task,(0,0));pair.paste(agent,(640,0))
            writers['agent_task_sync'].stdin.write(pair.tobytes())
            if has_world:
                ref=row.get('world_snapshot')
                if ref!=previous or panel is None:
                    if ref:
                        panel,bounds=world_panel(root,io/ref,bounds)
                    else:
                        panel=Image.new('RGB',(640,480),'#17202b')
                        ImageDraw.Draw(panel).text((20,20),'CodeWorld not yet available at this frame',fill='white')
                    previous=ref
                pair.paste(panel,(640,0));writers['codeworld_task_sync'].stdin.write(pair.tobytes())
        if cap.read()[0]:raise ValueError('Video has extra frames not recorded in timeline')
        for name,process in writers.items():
            process.stdin.close()
            if process.wait(timeout=120):raise RuntimeError(f'{name}: FFmpeg failed; see log')
        for name in names:os.replace(out/f'{name}.partial.mp4',out/f'{name}.mp4')
        manifest.update(status='COMPLETE',frames=len(rows),fps=fps,
                        videos={n:f'artifacts/{n}.mp4' for n in names},
                        codeworld_video='recorded' if has_world else 'not_applicable')
    except Exception as exc:
        manifest.update(status='ERROR',error=str(exc))
        raise
    finally:
        cap.release()
        for process in writers.values():
            if process.poll() is None:
                process.kill();process.wait()
        for log in logs:log.close()
        write(out/'manifest.json',manifest)
    return manifest


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',required=True)
    print(json.dumps(export(parser.parse_args().attempt),indent=2))

if __name__=='__main__':main()
