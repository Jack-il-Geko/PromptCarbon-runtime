"""Create the runtime environment; install dependencies and pinned model assets."""
import argparse
import json
import platform
import subprocess
import sys
import time
import venv
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def prepare_env():
    """Create the optional API-key file without replacing existing settings."""
    try:
        with (ROOT / '.env').open('x', encoding='utf-8') as file:
            file.write('OPENAI_API_KEY=your_key_here\n')
    except FileExistsError:
        pass


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--automatic-constraints',action='store_true')
    parser.add_argument('--allow-untested-platform',action='store_true')
    args=parser.parse_args()
    if sys.version_info[:2]!=(3,12):parser.error('Use Python 3.12 for the tested runtime.')
    tested=platform.system()=='Darwin' and platform.machine()=='arm64'
    if not tested and not args.allow_untested_platform:
        parser.error('Only macOS Apple Silicon has been tested. Other platforms require --allow-untested-platform and are not certified.')
    started=time.perf_counter()
    env=ROOT/'.venv'
    if not env.exists():venv.EnvBuilder(with_pip=True).create(env)
    python=env/('Scripts/python.exe' if platform.system()=='Windows' else 'bin/python')
    version=subprocess.check_output([str(python),'-c','import sys; print(str(sys.version_info.major)+"."+str(sys.version_info.minor))'],text=True).strip()
    if version!='3.12':parser.error('Existing .venv uses Python '+version+'. Use the clean runtime export; the research environment will not be modified.')
    subprocess.run([str(python),'-m','pip','install','-r',str(ROOT/'requirements.txt')],check=True)
    groups=['task','reasoning','tokenizer']+(['constraint'] if args.automatic_constraints else [])
    command='from runtime_assets import ensure_assets; ensure_assets('+repr(tuple(groups))+')'
    subprocess.run([str(python),'-c',command],cwd=ROOT,check=True)
    prepare_env()
    result=dict(seconds=time.perf_counter()-started,platform=platform.platform(),python=sys.version,automatic_constraints=args.automatic_constraints)
    (ROOT/'.cache').mkdir(exist_ok=True)
    (ROOT/'.cache/install_metrics.json').write_text(json.dumps(result,indent=2))
    print('Installed. Start with:',python,'run.py')
    print('Installation seconds:',round(result['seconds'],2))


if __name__=='__main__':main()
