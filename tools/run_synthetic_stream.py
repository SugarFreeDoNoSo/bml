#!/usr/bin/env python3
"""Serial controller: hard-limit entire compile/reference job, then clean AOT runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from synthetic_stream import positive_finite

os.environ['OPENBLAS_NUM_THREADS']='1'
try:
    import numpy as np
except ImportError:
    np = None

DOCKER=['docker','--host=unix:///var/run/docker.sock']
JOB_BYTES=7*1024**3
TOTAL_BUDGET=8*1024**3


def host_preflight():
    info=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    available=int(info['MemAvailable'].split()[0])*1024
    cg=Path('/sys/fs/cgroup');limit=(cg/'memory.max').read_text().strip();current=int((cg/'memory.current').read_text())
    stats=dict(line.split() for line in (cg/'memory.stat').read_text().splitlines())
    inactive=int(stats.get('inactive_file',0))
    if limit!='max':available=min(available,max(0,int(limit)-current)+inactive)
    if available<TOTAL_BUDGET:raise RuntimeError('less than 8 GiB conservative headroom, including reclaimable inactive file cache')
    return {'MemAvailable_bytes':int(info['MemAvailable'].split()[0])*1024,'cgroup_limit':limit,
            'cgroup_current_bytes':current,'inactive_file_reclaimable_bytes':inactive,
            'conservative_headroom_bytes':available,'job_limit_bytes':JOB_BYTES,
            'controller_and_runtime_margin_bytes':TOTAL_BUDGET-JOB_BYTES}


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--job-image',default='bml-stream-job:local')
    p.add_argument('--clean-image',default='bml-aot-clean:local')
    p.add_argument('--sizes',type=int,nargs='+',choices=[2560,5120,10240,20480],default=[2560,5120,10240,20480])
    p.add_argument('--mm-seconds-at-1280',type=positive_finite,default=0.982674173)
    a=p.parse_args();
    if np is None: p.error('NumPy reference required; never installed automatically')
    a.results.mkdir(parents=True,exist_ok=False);a.work.mkdir(parents=True,exist_ok=False)
    (a.work/'tmp').mkdir()
    src=Path(__file__).resolve().parents[1]
    env=os.environ.copy()
    for key in ['DOCKER_HOST','DOCKER_CONTEXT','DOCKER_TLS','DOCKER_TLS_VERIFY','DOCKER_CERT_PATH']:env.pop(key,None)
    history=[];calibration=a.mm_seconds_at_1280
    for n in a.sizes:
        initial=host_preflight()
        name=f'bml-stream-{os.getpid()}-{n}'
        command=DOCKER+['run','--name',name,'--rm','--network=none','--read-only','--memory=7g','--memory-swap=7g','--cpus=2','--pids-limit=64',
            '--cap-drop=ALL','--security-opt=no-new-privileges','--user',f'{os.getuid()}:{os.getgid()}',
            '--tmpfs','/tmp:size=16m,mode=1777','-e','TMPDIR=/work/tmp',
            '--mount',f'type=bind,src={src},dst=/src,readonly',
            '--mount',f'type=bind,src={a.work},dst=/work',
            '--mount',f'type=bind,src={a.results},dst=/results',a.job_image,
            'python3','/src/tools/synthetic_stream.py','--n',str(n),'--work',f'/work/n{n}',
            '--results',f'/results/n{n}','--mm-seconds-at-1280',str(calibration)]
        t=time.perf_counter()
        try:r=subprocess.run(command,capture_output=True,text=True,env=env,timeout=420)
        except subprocess.TimeoutExpired:
            stopped=subprocess.run(DOCKER+['rm','-f',name],capture_output=True,env=env,timeout=20)
            (a.results/f'n{n}.timeout.json').write_text(json.dumps({'status':'controller_timeout','named_container_stop_requested':name,'stop_exit':stopped.returncode}))
            raise RuntimeError(f'controller timeout; stop request exit {stopped.returncode}; reconcile before any retry')
        (a.results/f'n{n}.job.log').write_text(r.stdout+'\n'+r.stderr)
        item={'n':n,'host_preflight':initial,'container_command':command,'container_exit':r.returncode,'job_wall_seconds':time.perf_counter()-t}
        report_path=a.results/f'n{n}'/'result.json'
        if report_path.exists():item['job']=json.loads(report_path.read_text())
        if r.returncode or item.get('job',{}).get('status')!='completed':
            history.append(item);(a.results/'summary.json').write_text(json.dumps(history,indent=2)+'\n');return 2
        binary=a.work/f'n{n}'/'kernel';clean=[]
        op=next(o for o in item['job']['operations'] if o['operation']=='matvec')
        assert op['status']=='passed'
        assert not (a.work/f'n{n}'/'fixture.gguf').exists()
        for variant in [0,1]:
            inp=a.work/'input-clean.bin';out=a.work/'output-clean.bin'
            rng=np.random.Generator(np.random.PCG64(20261031+variant))
            with inp.open('wb') as f:
                for row in range(0,n,64):f.write((rng.integers(-64,65,size=(min(64,n-row),1),dtype=np.int16).astype('<f4')/8).tobytes())
            clean_name=f'bml-stream-clean-{os.getpid()}-{n}-{variant}'
            cmd=DOCKER+['run','--name',clean_name,'--rm','-i','--network=none','--read-only','--memory=7g','--memory-swap=7g','--cpus=1','--pids-limit=32',
                '--cap-drop=ALL','--security-opt=no-new-privileges','--user',f'{os.getuid()}:{os.getgid()}',
                '--mount',f'type=bind,src={binary},dst=/kernel,readonly',a.clean_image,'/kernel','--matvec']
            t=time.perf_counter()
            with inp.open('rb') as f,out.open('wb') as target:
                try:
                    c=subprocess.run(cmd,stdin=f,stdout=target,stderr=subprocess.PIPE,env=env,timeout=75)
                except subprocess.TimeoutExpired:
                    stopped=subprocess.run(DOCKER+['rm','-f',clean_name],capture_output=True,env=env,timeout=20)
                    raise RuntimeError(f'clean container timeout; stop request exit {stopped.returncode}; reconcile before retry')
            if c.returncode:raise RuntimeError(f'clean run failed {c.returncode}: {c.stderr[:300]}')
            assert sha(out)==op['inputs'][variant]['output_sha256']
            clean.append({'variant':variant,'passed':True,'output_sha256':sha(out),
                          'container_wall_seconds':time.perf_counter()-t,'metrics':json.loads(c.stderr),
                          'mounts':['executable only; input through stdin'],'GGUF_available':False,'BML_available':False})
            inp.unlink();out.unlink()
        item['clean_matvec']=clean
        for mm in item['job']['operations']:
            if mm['operation']=='matmul' and mm['status']=='passed':
                calibration=max(v['kernel_seconds'] for v in mm['inputs'])/(n/1280)**3
        # All generated GB data are expendable; compact reports/hashes are retained.
        shutil.rmtree(a.work/f'n{n}')
        item['large_intermediates_and_binary_deleted']=True
        history.append(item);(a.results/'summary.json').write_text(json.dumps(history,indent=2)+'\n')
        print(json.dumps({'n':n,'matvec':op['status'],'clean_runs':len(clean),'GGUF_bytes':item['job']['GGUF_bytes']}),flush=True)

if __name__=='__main__':raise SystemExit(main())
