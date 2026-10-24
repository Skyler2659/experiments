"""Provision private environments; never alter an existing simulator environment."""
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def call(args,**kwargs):
    print('+',' '.join(map(str,args)),flush=True)
    subprocess.run(list(map(str,args)),check=True,**kwargs)

def clone(url,revision,path):
    if path.exists() and any(path.iterdir()):
        actual=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
        if actual!=revision:raise ValueError(f'{path} is {actual}; expected {revision}')
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    call(['git','clone',url,path]);call(['git','-C',path,'checkout','--detach',revision])

def pip(python,*args):call(['uv','pip','install','--python',python,*args])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sim-python',help='Existing official Isaac Sim 5.1 / RoboDojo Python; recommended on a configured host')
    p.add_argument('--bootstrap-sim',action='store_true',help='Download and build Isaac Sim 5.1, IsaacLab and CuRobo in this folder')
    p.add_argument('--assets',help='Existing official RoboDojo Assets directory; otherwise run the official downloader')
    p.add_argument('--skip-reconstruction',action='store_true',help='Install GT/no-CodeWorld/RoboProbe only')
    a=p.parse_args()
    if not a.sim_python and not a.bootstrap_sim:p.error('Pass --sim-python /path/to/bin/python or --bootstrap-sim')
    rd=ROOT/'vendor/RoboDojo';rp=ROOT/'vendor/RoboProbe'
    pins=json.loads((ROOT/'experiments/robodojo/platform_sources.json').read_text())
    call([sys.executable,ROOT/'experiments/robodojo/setup_links.py'])
    assets=rd/'Assets'
    if a.assets:
        target=Path(a.assets).expanduser().resolve()
        if not (target/'Eval_Layout/RoboDojo').is_dir():raise ValueError('Official evaluation layouts are missing from --assets')
        if not assets.exists():assets.symlink_to(target,target_is_directory=True)
        elif assets.resolve()!=target:raise ValueError('A different Assets tree already exists')
    elif not (assets/'Eval_Layout/RoboDojo').exists():
        # The official downloader's fresh shallow clone only fetches main.
        # Fetch the selected asset revision before asking it to checkout that pin.
        cache=rd/'.cache/robodojo_assets_repo'
        env={**os.environ,'GIT_LFS_SKIP_SMUDGE':'1'}
        if not (cache/'.git').exists():
            cache.parent.mkdir(parents=True,exist_ok=True)
            call(['git','clone','--depth','1','--sparse',pins['Assets']['url'],cache],env=env)
        call(['git','-C',cache,'fetch','--depth','1','origin',pins['Assets']['commit']],env=env)
        call(['bash',rd/'scripts/init_assets.sh'],env={**os.environ,'HF_REVISION':pins['Assets']['commit']})
    if a.sim_python:
        native=Path(a.sim_python).expanduser().absolute()
        call([native,'-c','import importlib.util; assert importlib.util.find_spec("isaacsim"); assert importlib.util.find_spec("isaaclab")'])
        simdir=ROOT/'.venv-simulator'
        if not (simdir/'bin/python').exists():call([native,'-m','venv','--system-site-packages',simdir])
    else:
        simdir=ROOT/'.venv-simulator'
        if not (simdir/'bin/python').exists():call(['uv','venv','--python','3.11',simdir])
        py=simdir/'bin/python'
        pip(py,'pip','wheel==0.45.1','setuptools','packaging==23.0','numpy==1.26.0')
        pip(py,'torch==2.7.0','torchvision==0.22.0','torchaudio==2.7.0','--index-url','https://download.pytorch.org/whl/cu128')
        pip(py,'isaacsim[all,extscache]==5.1.0','--extra-index-url','https://pypi.nvidia.com')
        pip(py,'-r',rd/'scripts/requirements.txt','opencv-python-headless==4.11.0.86','scipy==1.15.3','numpy==1.26.0','omegaconf','trimesh==4.5.1','shapely','transforms3d','ruckig==0.19.4','python-fcl==0.7.0.11')
        for name in ('IsaacLab','curobo'):
            clone(pins[name]['url'],pins[name]['commit'],rd/'third_party'/name)
        env={**os.environ,'PATH':str(simdir/'bin')+os.pathsep+os.environ['PATH'],'VIRTUAL_ENV':str(simdir),'OMNI_KIT_ACCEPT_EULA':'YES','ACCEPT_EULA':'Y','TERM':'xterm-256color'}
        call(['bash',rd/'third_party/IsaacLab/isaaclab.sh','--install','none'],env=env)
        pip(py,'-e',str(rd/'third_party/curobo')+'[cu12]','--no-build-isolation')
    sim=simdir/'bin/python'
    # L3 runs inside the simulator; upstream pins this SDK and Pillow version.
    # Overlay installation leaves the supplied native environment untouched.
    # RoboProbe permits newer NumPy/OpenCV, but Isaac Sim's Kit extensions load
    # NumPy 1.x modules. Keep the native Docker versions in this overlay too.
    pip(sim,'-e',rp,'numpy==1.26.0','scipy==1.15.3','opencv-python-headless==4.11.0.86',
        'trimesh==4.5.1','python-fcl==0.7.0.11','ruckig==0.19.4','rtree','pytest')
    control=ROOT/'.venv-benchmark'
    if not (control/'bin/python').exists():call(['uv','venv','--python','3.12',control])
    pip(control/'bin/python','-r',ROOT/'experiments/robodojo/requirements.txt')
    runtime=ROOT/'runtime'
    if not a.skip_reconstruction:
        recon=ROOT/'.venv-reconstruction'
        if not (recon/'bin/python').exists():call(['uv','venv','--python','3.12',recon])
        py=recon/'bin/python'
        pip(py,'torch==2.6.0','torchvision==0.21.0','--index-url','https://download.pytorch.org/whl/cu124')
        pip(py,'-r',ROOT/'experiments/robodojo/requirements-reconstruction.txt')
        sam=ROOT/'vendor/sam2';clone(pins['sam2']['url'],pins['sam2']['commit'],sam)
        pip(py,'-e',sam,'--no-build-isolation')
        env={**os.environ,'RCWM_PYTHON':str(control/'bin/python')}
        call(['bash',ROOT/'scripts/setup.sh'],env=env)
    codex=shutil.which('codex')
    if not codex or subprocess.check_output([codex,'--version'],text=True).strip()!='codex-cli 0.160.0':
        node=runtime/'.render-tools/node/bin'
        npm=str(node/'npm') if (node/'npm').exists() else shutil.which('npm')
        if not npm:raise RuntimeError('Install Node/npm, or install the reconstruction runtime with its Node distribution')
        env={**os.environ,'PATH':str(node)+os.pathsep+os.environ['PATH']}
        call([npm,'install','--prefix',ROOT/'.codex-cli','@openai/codex@0.160.0'],env=env)
        codex=str(ROOT/'.codex-cli/node_modules/.bin/codex')
    local=ROOT/'experiments/robodojo/runtime.local.json'
    cfg=json.loads((local if local.exists() else ROOT/'experiments/robodojo/runtime.example.json').read_text())
    cfg.update(sim_python=str(sim),roboprobe_sim_python=str(sim),control_python=str(control/'bin/python'),codex=codex)
    cfg.setdefault('env',{})['PATH']=str(runtime/'.render-tools/node/bin')+os.pathsep+os.environ['PATH']
    local.write_text(json.dumps(cfg,indent=2)+'\n')
    print('Installed. Login to Codex, configure the RoboProbe API endpoint/key, then run doctor and probe. Runtime:',local)
if __name__=='__main__':main()
