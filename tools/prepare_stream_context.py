#!/usr/bin/env python3
"""Prepare an offline Docker context from public NumPy files and a local Rust probe."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    import numpy
    if sys.version_info[:2]!=(3,12) or numpy.__version__!='2.3.5':
        p.error('requires existing CPython 3.12 and NumPy 2.3.5; no auto-install')
    root=Path(__file__).resolve().parents[1];probe=root/'target/release/examples/stream_fixture_probe'
    if not probe.is_file():p.error('build stream_fixture_probe first')
    out=a.output;out.mkdir(parents=True,exist_ok=False)
    module=Path(numpy.__file__).parent
    shutil.copytree(module,out/'numpy',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(module.parent/'numpy.libs',out/'numpy.libs')
    shutil.copyfile(probe,out/'probe');shutil.copyfile(root/'docker/synthetic-stream/Dockerfile',out/'Dockerfile')
    (out/'context.json').write_text(json.dumps({'numpy':numpy.__version__,'Python_ABI':'cp312','probe_sha256':hashlib.sha256(probe.read_bytes()).hexdigest(),'scope':'no model files or credentials copied'},indent=2)+'\n')
    print(out)

if __name__=='__main__':main()
