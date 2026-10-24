#!/usr/bin/env python3
"""Standalone RCWM entry point. Does not import the previous reconstruction code."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'vendor' / 'RecursiveCWM'


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def png_info(path):
    data = path.read_bytes()
    if data[:8] != b'\x89PNG\r\n\x1a\n' or data[12:16] != b'IHDR':
        raise ValueError('Use an actual PNG reference image')
    return hashlib.sha256(data).hexdigest(), list(struct.unpack('>II', data[16:24]))


def prepare(args):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', args.scene):
        raise ValueError('Scene must contain letters, digits, underscore or hyphen')
    image = Path(args.image).resolve()
    digest, size = png_info(image)
    metadata = None
    if args.capture:
        cap_path = Path(args.capture).resolve()
        capture = json.loads(cap_path.read_text())
        matches = [v for v in capture['views'] if v['view_id'] == args.view]
        if len(matches) != 1:
            raise ValueError('Select one capture view with --view')
        view = matches[0]
        expected = view.get('rgb_sha256')
        if not expected:
            raise ValueError('Capture view needs rgb_sha256 to verify the cached image')
        if expected != digest:
            raise ValueError('Cached image does not match the recorded camera RGB')
        metadata = {'capture_source': str(cap_path), 'observation_id': capture.get('observation_id'),
                    'view': view, 'solver_input_role': 'available_auxiliary_metadata'}
    dest = ROOT / 'inputs' / args.scene
    dest.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(image, dest / 'reference.png')
    write(dest / 'input.json', {'scene': args.scene, 'image_source': str(image),
          'reference': 'reference.png', 'sha256': digest, 'image_size': size,
          'input_kind': 'single_camera_rgb', 'backend': 'RCWM',
          'previous_reconstruction_used': False})
    if metadata:
        write(dest / 'camera_metadata.json', metadata)
    print(dest)


def environment():
    env = os.environ.copy()
    runtime = Path(env.get('RCWM_ROOT', str(ROOT / 'runtime'))).resolve()
    env['RCWM_ROOT'] = str(runtime)
    # The Codex sandbox requires the real writable directory, not a symlink.
    env['PLAYWRIGHT_BROWSERS_PATH'] = str(Path(env.get(
        'PLAYWRIGHT_BROWSERS_PATH', str(runtime / '.browsers'))).resolve())
    libdir = runtime / '.system-libs/usr/lib/x86_64-linux-gnu'
    if libdir.is_dir():
        env['LD_LIBRARY_PATH'] = str(libdir) + os.pathsep + env.get('LD_LIBRARY_PATH', '')
    env['PATH'] = os.pathsep.join([str(runtime / 'bin'), str(runtime / '.render-tools/node/bin'),
                    str(runtime / '.codex-cli/node_modules/.bin'), env.get('PATH', '')])
    return runtime, env


def preflight(runtime, env):
    errors = []
    if platform.system() != 'Linux' and not (runtime / 'platform.json').exists():
        errors.append('On macOS run scripts/setup_macos.sh to install and verify the native port')
    for rel in ['.venv/bin/python', '.render-tools/node/bin/node',
                '.render-tools/node_modules/three/build/three.module.js',
                '.render-tools/node_modules/playwright/index.mjs']:
        if not (runtime / rel).exists():
            errors.append('Missing runtime dependency: ' + rel)
    for cmd in ['codex', 'python3', 'sha256sum', 'bash']:
        if not shutil.which(cmd, path=env['PATH']):
            errors.append('Missing command: ' + cmd)
    return errors


def run(args):
    cfg_path = Path(args.config).resolve()
    cfg = json.loads(cfg_path.read_text())
    name = args.name or cfg['scene']
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name):
        raise ValueError('Invalid run name')
    ref = ROOT / cfg['reference']
    digest, size = png_info(ref)
    binding = json.loads((ref.parent / 'input.json').read_text())
    if digest != binding['sha256']:
        raise ValueError('Prepared reference image changed')
    depth, cycles = cfg['max_depth'], cfg['max_cycles']
    if type(depth) is not int or depth < 0 or type(cycles) is not int or cycles < 1:
        raise ValueError('Invalid recursion limits')
    runtime, env = environment()
    env.update(RCWM_MODEL=cfg['model'], RCWM_REASONING=cfg['reasoning'])
    prompt = ROOT / cfg['solver_prompt'] if cfg.get('solver_prompt') else UPSTREAM / 'solver/solver-template.md'
    if not prompt.is_file(): raise ValueError('Missing solver prompt')
    env['RCWM_PROMPT'] = str(prompt.resolve())
    assistance = cfg.get('assistance')
    if assistance:
        transport = ROOT / assistance['transport_config']
        evidence = ROOT / assistance['measurement']
        if not transport.is_file() or not evidence.is_file():
            raise ValueError('Missing measured assistance configuration or evidence')
        env['RCWM_ASSIST_CONFIG'] = str(transport.resolve())
        env['RCWM_FEEDBACK_HOOK'] = str(ROOT / 'reconstruction_assist/feedback_hook.py')
        env['RCWM_ADAPTIVE_DISPATCH'] = '1' if json.loads(transport.read_text()).get('refinement', {}).get('strategy') == 'adaptive' else '0'
    known = cfg.get('known_robot')
    if known:
        robot_asset, robot_binding = ROOT / known['asset'], ROOT / known['binding']
        robot_record = json.loads(robot_binding.read_text())
        if robot_record['reference_sha256'] != digest:
            raise ValueError('Robot observation is not bound to this reference')
        if hashlib.sha256(robot_asset.read_bytes()).hexdigest() != robot_record['robot_node']['asset']['sha256']:
            raise ValueError('Known robot asset changed')
    run_dir = runtime / 'runs' / name
    command = ['bash', str(run_dir / 'runner_source/rcwm.sh'), str(ref), name, str(depth), str(cycles)]
    plan = {'backend': 'RCWM-hybrid' if known else 'RCWM', 'solver_prompt': str(prompt), 'known_robot': known, 'reference': str(ref), 'image_size': size,
            'command': command, 'runtime': str(runtime),
            'upstream': json.loads((ROOT / 'upstream.json').read_text()),
            'model': cfg['model'], 'reasoning': cfg['reasoning'], 'assistance': assistance,
            'previous_reconstruction_used': False,
            'preflight_errors': preflight(runtime, env)}
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    if plan['preflight_errors']:
        raise RuntimeError('\n'.join(plan['preflight_errors']))
    run_dir.mkdir(parents=True, exist_ok=False)
    # Running shell scripts must not change under a live interpreter.
    shutil.copytree(UPSTREAM, run_dir / 'runner_source',
                    ignore=shutil.ignore_patterns('.git', 'runtime', '__pycache__', '*.pyc'))
    shutil.copy2(prompt, run_dir / 'solver_prompt.md')
    env['RCWM_PROMPT'] = str(run_dir / 'solver_prompt.md')
    if assistance:
        shutil.copytree(ROOT / 'reconstruction_assist', run_dir / 'assist_source',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        transport_config = json.loads((ROOT / assistance['transport_config']).read_text())
        if transport_config.get('refinement', {}).get('strategy') == 'adaptive':
            transport_config.setdefault('agent_correction', {
                'enabled': True, 'max_rounds': 1, 'extra_reviews': 2,
                'extra_cycles': 4, 'timeout_seconds': 660})
        write(run_dir / 'assist_transport.json', transport_config)
        env['RCWM_ASSIST_CONFIG'] = str(run_dir / 'assist_transport.json')
        env['RCWM_FEEDBACK_HOOK'] = str(run_dir / 'assist_source/feedback_hook.py')
    shutil.copytree(ref.parent, run_dir / 'input')
    if known:
        shutil.copyfile(robot_asset, run_dir / 'input/known_robot.glb')
        shutil.copyfile(robot_binding, run_dir / 'input/robot_binding.json')
    write(run_dir / 'launch.json', plan)
    write(run_dir / 'status.json', {'status': 'RUNNING'})
    with (run_dir / 'launch.log').open('w') as log:
        code = subprocess.call(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    part = run_dir / 'fractal/scene/part.json'
    valid = False
    if part.exists():
        with (run_dir / 'artifact_check.log').open('w') as log:
            valid = subprocess.call([str(runtime / '.venv/bin/python'),
                str(UPSTREAM / 'runner/check_part.py'), str(part)],
                stdout=log, stderr=subprocess.STDOUT) == 0
    selection_path = run_dir / 'fractal/scene/selection.json'
    selection = json.loads(selection_path.read_text()) if selection_path.exists() else None
    write(run_dir / 'status.json', {'status': 'PROGRAM_DELIVERED' if code == 0 and valid else 'FAILED',
          'returncode': code, 'part_valid': valid, 'metric_geometry_verified': False,
          'robot_interface_connected': False, 'measured_feedback_selection': selection})
    print(run_dir)
    if code or not valid:
        raise RuntimeError('RCWM did not deliver a valid program; inspect launch.log')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--image', required=True)
    p.add_argument('--scene', required=True)
    p.add_argument('--capture')
    p.add_argument('--view')
    p.set_defaults(func=prepare)
    p = sub.add_parser('run')
    p.add_argument('--config', default=str(ROOT / 'configs/robodojo_insert_tubes.json'))
    p.add_argument('--name')
    p.add_argument('--dry-run', action='store_true')
    p.set_defaults(func=run)
    args = parser.parse_args()
    try:
        args.func(args)
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        parser.exit(1, str(exc) + '\n')


if __name__ == '__main__':
    main()
