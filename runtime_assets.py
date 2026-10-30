"""Download only pinned runtime assets, verify SHA256 and cache locally."""
import hashlib
import json
import os
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def ensure_assets(groups=("task", "reasoning", "tokenizer")):
    """Keep existing checkpoints; fetch missing groups from the release or local mirror."""
    manifest_path=ROOT/'runtime_manifest.json'
    if not manifest_path.exists():
        return  # Research checkout already contains its runtime assets.
    manifest=json.loads(manifest_path.read_text())
    base=os.environ.get('PROMPTCARBON_ASSET_BASE_URL',manifest['base_url']).rstrip('/')
    cache=Path(os.environ.get('PROMPTCARBON_DOWNLOAD_CACHE',str(ROOT/'.cache/downloads')))
    for name in groups:
        asset=manifest['assets'][name]
        if all((ROOT/p).is_file() and (ROOT/p).stat().st_size==meta['bytes'] and digest(ROOT/p)==meta['sha256'] for p,meta in asset['files'].items()):
            continue
        cache.mkdir(parents=True,exist_ok=True)
        archive=cache/asset['archive']
        if not archive.exists() or digest(archive)!=asset['sha256']:
            url=base+'/'+asset['archive']
            partial=archive.with_suffix('.partial')
            print(f"Downloading {name}: {asset['bytes']/1e6:.1f} MB",flush=True)
            try:
                with urllib.request.urlopen(url,timeout=60) as src,partial.open('wb') as dst:
                    shutil.copyfileobj(src,dst)
                if digest(partial)!=asset['sha256']:raise RuntimeError('Asset checksum mismatch: '+name)
                partial.replace(archive)
            except (OSError,urllib.error.URLError) as exc:
                partial.unlink(missing_ok=True)
                raise RuntimeError('Cannot download '+name+'. The runtime release assets must be published first, or set PROMPTCARBON_ASSET_BASE_URL to a verified local mirror. '+str(exc)) from exc
        with tempfile.TemporaryDirectory(dir=cache) as tmp:
            temp=Path(tmp)
            with zipfile.ZipFile(archive) as z:
                if set(z.namelist())!=set(asset['files']):raise RuntimeError('Unexpected archive contents: '+name)
                for relative,meta in asset['files'].items():
                    part=Path(relative)
                    if part.is_absolute() or '..' in part.parts:raise RuntimeError('Unsafe asset path')
                    target=temp/part;target.parent.mkdir(parents=True,exist_ok=True)
                    with z.open(relative) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst)
                    if target.stat().st_size!=meta['bytes'] or digest(target)!=meta['sha256']:
                        raise RuntimeError('File checksum mismatch: '+relative)
                for relative in asset['files']:
                    target=ROOT/relative;target.parent.mkdir(parents=True,exist_ok=True)
                    (temp/relative).replace(target)
        # Expanded files are the local cache; retaining duplicate ZIPs is unnecessary.
        archive.unlink(missing_ok=True)
