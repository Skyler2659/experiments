import json,os,tempfile
from pathlib import Path

def read(path):return json.loads(Path(path).read_text())
def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(dir=path.parent,prefix='.write-')
    try:
        with os.fdopen(fd,'w') as f:json.dump(value,f,indent=2,allow_nan=False)
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)
