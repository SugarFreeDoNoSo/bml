#!/usr/bin/env python3
"""Compare formatted base with head so format-only changes do not widen mutation scope."""
import difflib
from pathlib import Path
import subprocess
import sys

base, output = sys.argv[1:]
subprocess.run(['git','rev-parse','--verify',base],check=True,capture_output=True)
files = subprocess.check_output(['git','diff','--name-only','--diff-filter=ACMR',base,'--','*.rs'],text=True).splitlines()
files += subprocess.check_output(['git','ls-files','--others','--exclude-standard','--','*.rs'],text=True).splitlines()
patch=[]
for name in sorted(set(files)):
    previous=subprocess.run(['git','show',f'{base}:{name}'],capture_output=True,text=True)
    old='' if previous.returncode else subprocess.run(['rustfmt','--edition','2021','--config','skip_children=true','--emit','stdout'],input=previous.stdout,capture_output=True,text=True,check=True).stdout
    new=Path(name).read_text()
    patch.extend(difflib.unified_diff(old.splitlines(keepends=True),new.splitlines(keepends=True),fromfile='a/'+name,tofile='b/'+name,n=0))
Path(output).parent.mkdir(parents=True,exist_ok=True)
Path(output).write_text(''.join(patch))
