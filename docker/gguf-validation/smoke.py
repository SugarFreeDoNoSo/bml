#!/usr/bin/env python3
"""Exercise the runner on tiny synthetic GGUFs, on host or inside Docker."""
import argparse
import json
import os
import shutil
from pathlib import Path
import struct
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent

def fixture(dtype=2):
    def string(s):
        b=s.encode(); return struct.pack('<Q',len(b))+b
    b=b'GGUF'+struct.pack('<IQQ',3,1,1)
    b+=string('general.architecture')+struct.pack('<I',8)+string('gemma4')
    b+=string('token_embd.weight')+struct.pack('<IQQIQ',2,64,3,dtype,0)
    b+=bytes((-len(b))%32)
    if dtype==2:
        for i in range(6):
            b+=struct.pack('<e', [0.5,-1.0,2.0][i%3])+bytes(j|((15-j)<<4) for j in range(16))
    else:
        b+=struct.pack('<192f',*range(192))
    return b

def main():
    p=argparse.ArgumentParser(); p.add_argument('--image'); p.add_argument('--evidence-dir'); a=p.parse_args()
    with tempfile.TemporaryDirectory(prefix='bml-gguf-smoke-') as tmp:
        root=Path(tmp); models=root/'models'; models.mkdir(); results=root/'results'; results.mkdir()
        (models/'fixture.gguf').write_bytes(fixture())
        (models/'f32.gguf').write_bytes(fixture(0))
        (models/'truncated.gguf').write_bytes(b'GGUF')
        (models/'badref').write_text('#!/usr/bin/env python3\nprint("0\\n"*64,end="")\n')
        (models/'badref').chmod(0o755)
        cases=[('q4','fixture.gguf',[],0),('missing','absent.gguf',[],2),('dtype','f32.gguf',[],2),
               ('bounds','fixture.gguf',['--row-start','3'],2),('truncated','truncated.gguf',[],2),('mismatch','fixture.gguf',[],2)]
        for name, filename, extra, expected in cases:
            out=results/name; out.mkdir()
            if a.image:
                cmd=['docker','--host=unix:///var/run/docker.sock','run','--rm','--network=none','--read-only',
                     '--cap-drop=ALL','--security-opt=no-new-privileges','--memory=512m','--cpus=2','--pids-limit=64',
                     '--user',f'{os.getuid()}:{os.getgid()}',
                     '--mount',f'type=bind,src={models},dst=/models,readonly',
                     '--mount',f'type=bind,src={out},dst=/results',a.image,
                     '--model',f'/models/{filename}','--output','/results']
            else:
                cmd=['python3',str(ROOT/'run.py'),'--model',str(models/filename),'--output',str(out)]
            cmd+=['--rows','3','--blocks','2']+extra
            env=os.environ.copy()
            if a.image:
                for key in ['DOCKER_HOST','DOCKER_CONTEXT','DOCKER_TLS','DOCKER_TLS_VERIFY','DOCKER_CERT_PATH']:
                    env.pop(key,None)
            if name == 'mismatch':
                if a.image:
                    pos=cmd.index(a.image); cmd[pos:pos]=['-e','BML_REFERENCE=/models/badref']
                else:
                    env['BML_REFERENCE']=str(models/'badref')
            r=subprocess.run(cmd,capture_output=True,text=True,env=env,timeout=90)
            assert r.returncode==expected,(name,r.returncode,r.stdout,r.stderr)
            report=json.loads((out/'result.json').read_text())
            assert report['status']==('passed' if expected==0 else 'failed'),report
            if name == 'mismatch':
                assert 'comparison failed' in report.get('error', ''), report
            if expected==0:
                assert report['architecture']=='gemma4'
                assert report['bml_execution_architecture_status'].startswith('unsupported')
                assert len(report['samples'])==3 and all(s['passed'] for s in report['samples'])
                assert report['full_model_inference_validated'] is False
            print(name+': PASS (expected '+report['status']+')')
        if a.evidence_dir:
            shutil.copytree(results, a.evidence_dir)
    print('6 smoke cases passed; synthetic data only')

if __name__=='__main__': main()
