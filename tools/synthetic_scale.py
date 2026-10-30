#!/usr/bin/env python3
"""Safe medium-scale extension: fixed F32 GGUF A, dynamic x/B, linked binary weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import struct
import subprocess
import time

# Driver reference only. It is neither linked to nor required by the executable.
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
try:
    import numpy as np
except ImportError:
    np = None
from gguf_fixture_reader import read_fixture
from synthetic_aot import string, child_limits, memory_preflight

MAX_WEIGHT_BYTES = 64*1024**2
MAX_KERNEL_BYTES = 128*1024**2
MAX_PARENT_BYTES = 512*1024**2
KERNEL_HEADROOM = 16*1024**2


def sha(b):
    return hashlib.sha256(b).hexdigest()


def workspace(n, cols):
    return 4*n*n+16*n*cols+KERNEL_HEADROOM


def gguf(n, seed):
    if 4*n*n > MAX_WEIGHT_BYTES:
        raise ValueError('64 MiB linked-weight/build budget exceeded before materialization')
    # Estimated driver temporaries: binary/NumPy weights and reference conversion.
    if 64*n*n > MAX_PARENT_BYTES:
        raise ValueError('512 MiB conservative parent workspace estimate exceeded')
    if np is None:
        raise RuntimeError("medium-scale driver requires NumPy; no automatic installation")
    rng = np.random.Generator(np.random.PCG64(seed))
    a = rng.integers(-128,129,size=(n,n),dtype=np.int16).astype('<f4')/16
    meta={'general.architecture':'bml-synthetic-matrix','bml.test.seed':str(seed),
          'bml.test.generator':'NumPy PCG64 integers[-128,128]/16; F32'}
    b=b'GGUF'+struct.pack('<IQQ',3,1,len(meta))
    for k,v in meta.items(): b+=string(k)+struct.pack('<I',8)+string(v)
    b+=string('matrix.weight')+struct.pack('<IQQIQ',2,n,n,0,0)
    b+=bytes((-len(b))%32)
    b+=a.tobytes()
    assert len(b)<=3_000_000_000
    return b,a


def kernel_source(n):
    return r'''#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <time.h>
#include <sys/resource.h>
#define N DIMENSION
extern const float A[];
static double seconds(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
int main(int argc,char **argv) {
 if(argc!=2) return 2;
 int cols=strcmp(argv[1],"--matmul")==0?N:1;
 if(cols==1 && strcmp(argv[1],"--matvec")!=0) return 2;
 size_t count=(size_t)N*cols;
 double *B=malloc(count*sizeof(double)), *C=calloc(count,sizeof(double));
 if(!B||!C) return 2;
 for(size_t i=0;i<count;i++) if(scanf("%lf",B+i)!=1 || !isfinite(B[i])) return 2;
 int ch; while((ch=getchar())!=EOF) if(!isspace((unsigned char)ch)) return 2;
 double begin=seconds();
 for(int r=0;r<N;r++) for(int k=0;k<N;k++) {
  double w=A[(size_t)r*N+k];
  for(int c=0;c<cols;c++) C[(size_t)r*cols+c]+=w*B[(size_t)k*cols+c];
 }
 double elapsed=seconds()-begin;
 puts("[");
 for(size_t i=0;i<count;i++) { if(!isfinite(C[i])) return 2; printf("%s%.17g",i?",":"",C[i]); }
 puts("]");
 struct rusage u; getrusage(RUSAGE_SELF,&u);
 fprintf(stderr,"{\"kernel_seconds\":%.9g,\"peak_rss_kib\":%ld}\n",elapsed,u.ru_maxrss);
 free(B);free(C);return 0;
}
'''.replace('DIMENSION',str(n))


def run(cmd, data=None, timeout=10, expected=0):
    env=os.environ.copy()
    for k in ['DOCKER_HOST','DOCKER_CONTEXT','DOCKER_TLS','DOCKER_TLS_VERIFY','DOCKER_CERT_PATH']:
        env.pop(k,None)
    before=time.perf_counter()
    r=subprocess.run(cmd,input=data,text=True,capture_output=True,timeout=timeout,env=env,
                     preexec_fn=None if cmd[0]=='docker' else child_limits)
    elapsed=time.perf_counter()-before
    if r.returncode!=expected:
        raise RuntimeError(f'{cmd[0]} exited {r.returncode}: {r.stderr[:500]}')
    return r,elapsed


def validate_mutations(blob, a, exporter, folder):
    info=read_fixture(blob,max_dimension=2560,decode_values=False)
    results=[]
    # GGUF does not contain an automatic content checksum; the expected seeded
    # payload hash is what identifies corruption or an unexpected transpose.
    for label,payload in [('byte_corruption',bytes([blob[info['data_offset']]^1])+blob[info['data_offset']+1:]),
                          ('layout_transpose',a.T.copy().tobytes())]:
        bad=folder/(label+'.gguf')
        bad.write_bytes(blob[:info['data_offset']]+payload)
        dest=folder/(label+'.bin')
        run([str(exporter),str(bad),str(dest)],timeout=30)
        assert sha(dest.read_bytes())!=sha(a.tobytes()),label
        results.append(label+': rejected by expected payload identity')
    rank_pos=info['descriptor_offset']+len(string('matrix.weight'))
    damaged=bytearray(blob);struct.pack_into('<Q',damaged,rank_pos+4,0)
    (folder/'bad_dimension.gguf').write_bytes(damaged)
    for label,path in [('bad_dimension',folder/'bad_dimension.gguf'),('missing_file',folder/'absent.gguf')]:
        run([str(exporter),str(path),str(folder/'bad.bin')],expected=2)
        results.append(label+': importer rejected')
    try: read_fixture(blob[:-1],max_dimension=2560,decode_values=False)
    except ValueError: results.append('truncation: independent reader rejected')
    else: raise AssertionError('truncation accepted')
    return results


def measure(n, seed, folder, exporter, image, enabled):
    preflight=memory_preflight()
    folder.mkdir()
    t=time.perf_counter();blob,a=gguf(n,seed);generation=time.perf_counter()-t
    info=read_fixture(blob,max_dimension=2560,decode_values=False)
    assert info['dims']==[n,n]
    model=folder/'fixture.gguf';model.write_bytes(blob)
    raw=folder/'weights.bin'
    _,import_time=run([str(exporter),str(model),str(raw)],timeout=30)
    assert raw.read_bytes()==a.tobytes()
    checks=validate_mutations(blob,a,exporter,folder) if n==160 else []
    src=folder/'kernel.c';src.write_text(kernel_source(n))
    asm=folder/'weights.S'
    asm.write_text('.section .rodata\n.balign 32\n.global A\n.type A,@object\nA:\n.incbin '+json.dumps(str(raw))+'\n.size A,.-A\n.section .note.GNU-stack,"",@progbits\n')
    binary=folder/'kernel'
    _,compile_time=run(['cc','-std=c11','-O2','-fno-fast-math',str(src),str(asm),'-lm','-o',str(binary)],timeout=30)
    model.unlink();raw.unlink()
    report={'n':n,'seed':seed,'generator':'NumPy PCG64; integer[-128,128]/16','numpy_reference_version':np.__version__,
            'fixture_sha256':sha(blob),'payload_sha256':sha(a.tobytes()),'binary_sha256':sha(binary.read_bytes()),
            'binary_bytes':binary.stat().st_size,'generated_C_bytes':src.stat().st_size,'embedding':'ELF .rodata via assembler .incbin; no C weight literals',
            'generation_seconds':generation,'import_seconds':import_time,'compile_seconds':compile_time,
            'memory_preflight':preflight,'mutation_checks':checks,'operations':[],
            'reference':'NumPy float64 dot/matmul (single-thread BLAS), independent of generated classical C loops',
            'scope':'fixed F32 A, dynamic float64 x/B; not transformer or full BML backend',
            'linkage':run(['ldd',str(binary)])[0].stdout}
    for name,cols in [('matvec',1),('matmul',n)]:
        budget=workspace(n,cols)
        if name not in enabled or budget>MAX_KERNEL_BYTES:
            report['operations'].append({'operation':name,'status':'not_executed','reason': 'stopped at an earlier time budget' if name not in enabled else 'workspace plus 16 MiB headroom exceeds 128 MiB runtime budget','estimated_workspace_bytes':budget})
            enabled.discard(name);continue
        op={'operation':name,'estimated_workspace_bytes':budget,'inputs':[]}
        try:
            # Two distinct dynamic inputs, same compiled weights and executable.
            for variant in [0,1]:
                rng=np.random.Generator(np.random.PCG64(seed+1+variant))
                b=rng.integers(-64,65,size=(n,cols),dtype=np.int16).astype(np.float64)/8
                t=time.perf_counter();expected=np.asarray(a,dtype=np.float64)@b;reference_seconds=time.perf_counter()-t
                data=' '.join(map(str,b.ravel()))+'\n'
                r,process_seconds=run([str(binary),'--'+name],data)
                actual=np.asarray(json.loads(r.stdout)).reshape(n,cols)
                assert np.allclose(actual,expected,atol=1e-9,rtol=1e-9)
                metrics=json.loads(r.stderr)
                entry={'input_variant':variant,'input_sha256':sha(b.tobytes()),'output_sha256':sha(actual.tobytes()),
                       'max_abs_error':float(np.max(np.abs(actual-expected))),'reference_seconds':reference_seconds,
                       'process_seconds_including_IO':process_seconds,**metrics}
                if image:
                    cmd=['docker','--host=unix:///var/run/docker.sock','run','--rm','-i','--network=none','--read-only',
                         '--memory=128m','--memory-swap=128m','--cpus=1','--pids-limit=32','--cap-drop=ALL','--security-opt=no-new-privileges',
                         '--user',f'{os.getuid()}:{os.getgid()}', '--mount',f'type=bind,src={binary},dst=/kernel,readonly',image,'/kernel','--'+name]
                    clean,_=run(cmd,data,timeout=15)
                    assert np.array_equal(np.asarray(json.loads(clean.stdout)).reshape(n,cols),actual)
                    entry['clean']=json.loads(clean.stderr)
                op['inputs'].append(entry)
            assert op['inputs'][0]['output_sha256']!=op['inputs'][1]['output_sha256']
            op['status']='passed';op['different_inputs_change_output']=True
        except subprocess.TimeoutExpired as e:
            op['status']='stopped_timeout';op['timeout_seconds']=e.timeout;enabled.discard(name)
        report['operations'].append(op)
    # Delivery fixtures restored after proving absence during runtime.
    model.write_bytes(blob)
    (folder/'measurement.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--exporter',type=Path,default=Path('target/release/examples/synthetic_export'))
    p.add_argument('--clean-image')
    p.add_argument('--seed',type=int,default=20261030)
    a=p.parse_args();memory_preflight();a.output.mkdir(parents=True,exist_ok=False)
    reports=[];enabled={'matvec','matmul'};n=160
    while enabled:
        if 4*n*n>MAX_WEIGHT_BYTES or 64*n*n>MAX_PARENT_BYTES:
            stop={'n_not_materialized':n,'reason':'64 MiB binary-weight or 512 MiB conservative driver workspace limit'};break
        reports.append(measure(n,a.seed,a.output/f'n{n}',a.exporter.resolve(),a.clean_image,enabled));n*=2
    else: stop={'n_not_materialized':n,'reason':'both operation budgets exhausted'}
    summary={'reports':reports,'stop':stop,'limits':{'weights_bytes':MAX_WEIGHT_BYTES,'kernel_bytes':MAX_KERNEL_BYTES,'driver_estimate_bytes':MAX_PARENT_BYTES,'operation_wall_timeout_seconds':10,'container_wall_timeout_seconds':15},'real_model_inference':False}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'sizes':[r['n'] for r in reports],'stop':stop}))

if __name__=='__main__':main()
