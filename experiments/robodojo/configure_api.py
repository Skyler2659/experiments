"""Configure a Responses-compatible provider; credentials remain in environment."""
import argparse
import json
from pathlib import Path
from urllib.parse import urlparse


def configure(runtime_path, base_url, key_env='OPENAI_API_KEY'):
    import re
    if urlparse(base_url).scheme not in ('https','http') or not urlparse(base_url).netloc:
        raise ValueError('Use the API base URL, e.g. https://api.openai.com/v1')
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',key_env):raise ValueError('Invalid environment variable name')
    base_url=base_url.rstrip('/')
    if base_url.endswith('/responses'):raise ValueError('Use base URL without /responses')
    runtime_path=Path(runtime_path).resolve()
    config=json.loads(runtime_path.read_text())
    provider=runtime_path.with_name('provider.local.toml')
    provider.write_text('model_provider = "experiment"\n[model_providers.experiment]\nname = "Experiment API"\n'
        +f'base_url = {json.dumps(base_url)}\nenv_key = {json.dumps(key_env)}\nwire_api = "responses"\n')
    config.update(codex_provider_config=str(provider),vision_transport='responses',
        reconstruction_base_url=base_url,reconstruction_api_key_env=key_env,
        roboprobe_base_url=base_url,roboprobe_api_key_env=key_env)
    runtime_path.write_text(json.dumps(config,indent=2)+'\n')
    return runtime_path

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base-url',required=True)
    p.add_argument('--key-env',default='OPENAI_API_KEY')
    p.add_argument('--runtime',default=str(Path(__file__).with_name('runtime.local.json')))
    a=p.parse_args();print(configure(a.runtime,a.base_url,a.key_env))
