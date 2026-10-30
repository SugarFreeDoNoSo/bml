#!/usr/bin/env python3
"""Bounded synthetic F32 GGUF -> C AOT experiment, not transformer inference."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import resource
import struct
import subprocess
import time
from gguf_fixture_reader import read_fixture

MAX_N = 128
MAX_FILE_BYTES = 3_000_000_000

def planned_sizes():
    sizes=[]; n=10
    while 4*n*n+65536 <= MAX_FILE_BYTES:
        sizes.append(n); n*=2
    return sizes

def string(s):
    b = s.encode(); return struct.pack('<Q', len(b)) + b

def fixture(n, seed):
    if not 1 <= n <= MAX_N:
        raise ValueError('materialization limited to dimensions 1..128')
    rng = random.Random(seed)
    weights = [rng.randint(-128, 128)/16 for _ in range(n*n)]
    metadata = {'general.architecture': 'bml-synthetic-matrix',
                'bml.test.seed': str(seed), 'bml.test.generator': 'Python Random.randint(-128,128)/16; F32 exact'}
    b = b'GGUF' + struct.pack('<IQQ', 3, 1, len(metadata))
    for k, v in metadata.items():
        b += string(k) + struct.pack('<I', 8) + string(v)
    b += string('matrix.weight') + struct.pack('<IQQIQ', 2, n, n, 0, 0)
    b += bytes((-len(b)) % 32)
    return b + struct.pack(f'<{n*n}f', *weights), weights

def c_source(n, weights):
    # Model resources are compiled into the executable; stdin supplies only activations.
    constants = ','.join(float(v).hex()+'f' for v in weights)
    return f'''#include <stdio.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <sys/resource.h>
#define N {n}
static const float A[N*N] = {{{constants}}};
int main(int argc,char **argv) {{
    if(argc!=2) return 2;
    int mm=strcmp(argv[1],"--matmul")==0;
    if(!mm && strcmp(argv[1],"--matvec")!=0) return 2;
    double B[N*N],C[N*N];
    int cols=mm?N:1;
    for(int i=0;i<N*cols;i++) if(scanf("%lf",&B[i])!=1 || !isfinite(B[i])) return 2;
    int ch; while((ch=getchar())!=EOF) if(!isspace((unsigned char)ch)) return 2;
    for(int r=0;r<N;r++) for(int c=0;c<cols;c++) {{
        double sum=0;
        for(int k=0;k<N;k++) sum+=(double)A[r*N+k]*B[k*cols+c];
        if(!isfinite(sum)) return 2;
        C[r*cols+c]=sum;
    }}
    printf("[");
    for(int i=0;i<N*cols;i++) printf("%s%.17g",i?",":"",C[i]);
    puts("]");
    struct rusage usage; if(getrusage(RUSAGE_SELF,&usage)==0) fprintf(stderr,"peak_rss_kib=%ld\\n",usage.ru_maxrss);
    return 0;
}}
'''

def reference(a, b, n, cols):
    # Independent Python float64 oracle, not generated C or the BML scalar IR.
    return [math.fsum(a[r*n+k]*b[k*cols+c] for k in range(n))
            for r in range(n) for c in range(cols)]

def estimate(n, bandwidth_gbs=20.0, peak_gflops=10.0):
    if n < 1 or bandwidth_gbs <= 0 or peak_gflops <= 0:
        raise ValueError('positive dimension/bandwidth/peak required')
    a = 4*n*n
    mv_bytes = a+8*n
    mm_min_bytes = 12*n*n
    mv_flops, mm_flops = 2*n*n, 2*n*n*n
    return {'kind': 'analytical_estimate_not_measurement', 'n': n,
            'f32_A_bytes': a, 'matvec_working_bytes': mv_bytes,
            'matmul_working_bytes': mm_min_bytes,
            'matvec_flops_conventional': mv_flops, 'matmul_flops_conventional': mm_flops,
            'matvec_min_traffic_bytes': mv_bytes, 'matmul_min_traffic_bytes': mm_min_bytes,
            'matmul_naive_traffic_bytes': 8*n*n*n+4*n*n,
            'assumptions': {'bandwidth_GB_per_second_decimal': bandwidth_gbs,
                            'compute_GFLOP_per_second': peak_gflops,
                            'hardware_calibrated': False,'complexity_scope':'implemented classical dense algorithm with fixed A and dynamic x/B; not a universal specialization lower bound',
                            'traffic': 'ideal once-only A/B reads and C write; excludes cache misses/packing/OS',
                            'flops': '2 per multiply-add convention; exact scalar count subtract n (matvec) or n^2 (matmul)',
                            'storage': 'F32 A/B/C; excludes executable/allocator/metadata; emitted kernel uses float64 B/C'},
            'matvec_ideal_lower_bound_seconds': max(mv_flops/(peak_gflops*1e9), mv_bytes/(bandwidth_gbs*1e9)),
            'matmul_ideal_lower_bound_seconds': max(mm_flops/(peak_gflops*1e9), mm_min_bytes/(bandwidth_gbs*1e9))}

def child_limits():
    resource.setrlimit(resource.RLIMIT_AS, (512*1024*1024, 512*1024*1024))
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))

def measure(n, seed, out, exporter, image):
    if not 1 <= n <= MAX_N:
        raise ValueError('real runs limited to n=1..128; larger sizes are analytical only')
    out.mkdir(parents=True, exist_ok=False)
    generation_start = time.perf_counter()
    original, weights = fixture(n, seed)
    generation_seconds = time.perf_counter()-generation_start
    if len(original)>MAX_FILE_BYTES:
        raise ValueError('GGUF file exceeds 3 GB budget')
    independent=read_fixture(original)
    assert independent['dims']==[n,n] and independent['values']==weights
    model = out/'fixture.gguf'; model.write_bytes(original)
    commands = []
    def run(cmd, data=None):
        commands.append(cmd)
        env = os.environ.copy()
        if cmd[0] == 'docker':
            for key in ['DOCKER_HOST','DOCKER_CONTEXT','DOCKER_TLS','DOCKER_TLS_VERIFY','DOCKER_CERT_PATH']:
                env.pop(key, None)
        r = subprocess.run(cmd, input=data, text=True, capture_output=True, env=env,
                           timeout=60, preexec_fn=None if cmd[0] == 'docker' else child_limits)
        run.last_stderr = r.stderr
        if r.returncode:
            raise RuntimeError(f'command failed {r.returncode}: {r.stderr}')
        return r.stdout
    raw = out/'imported.bin'
    import_start = time.perf_counter()
    assert int(run([str(exporter), str(model), str(raw)])) == n
    imported = struct.unpack(f'<{n*n}f', raw.read_bytes())
    assert list(imported) == weights, 'BML GGUF import differs from generated oracle'
    import_seconds = time.perf_counter()-import_start
    codegen_start = time.perf_counter()
    src = out/'kernel.c'; src.write_text(c_source(n, imported))
    codegen_seconds = time.perf_counter()-codegen_start
    binary = out/'kernel'
    before = time.perf_counter()
    run(['cc','-std=c11','-O2','-fno-fast-math',str(src),'-lm','-o',str(binary)])
    build_seconds = time.perf_counter()-before
    # Destructive removal of our own temporary fixture/import is intentional evidence:
    # only compiled weights remain; no GGUF or BML path goes to runtime.
    model.unlink(); raw.unlink()
    rng = random.Random(seed+1)
    comparisons = []
    for cols, op in [(1,'--matvec'),(n,'--matmul')]:
        injected = [rng.randint(-64,64)/8 for _ in range(n*cols)]
        expected = reference(weights, injected, n, cols)
        data = ' '.join(map(str,injected))+'\n'
        before = time.perf_counter(); actual = json.loads(run([str(binary),op], data))
        elapsed = time.perf_counter()-before
        kernel_rss = int(re.search(r'peak_rss_kib=(\d+)',run.last_stderr).group(1))
        assert len(actual)==len(expected)
        err = max(abs(x-y) for x,y in zip(actual,expected))
        assert all(math.isfinite(x) and abs(x-y)<=1e-9+1e-9*abs(y) for x,y in zip(actual,expected))
        clean = None
        if image:
            # Fresh container receives just one executable; no GGUF, source or BML mount.
            cmd = ['docker','--host=unix:///var/run/docker.sock','run','--rm','-i',
                   '--network=none','--read-only','--memory=128m','--memory-swap=128m',
                   '--cpus=1','--pids-limit=32','--cap-drop=ALL','--security-opt=no-new-privileges',
                   '--user',f'{os.getuid()}:{os.getgid()}',
                   '--mount',f'type=bind,src={binary},dst=/kernel,readonly',image,'/kernel',op]
            clean_values=json.loads(run(cmd,data))
            assert clean_values==actual
            clean_rss = int(re.search(r'peak_rss_kib=(\d+)',run.last_stderr).group(1))
            clean={'kernel_peak_rss_kib':clean_rss,'passed':True,'image':image,'mounts':['/kernel readonly'],
                   'network':'none','memory_bytes':128*1024*1024,
                   'bml_or_gguf_available_to_runtime':False}
        comparisons.append({'kernel_peak_rss_kib':kernel_rss,'operation':op[2:],'kind':'measured','n':n,'max_abs_error':err,
                            'process_wall_seconds_including_startup_and_io':elapsed,
                            'input':injected,'expected':expected,'actual':actual,'clean_environment':clean})
    for bad in ['', 'nan '+ '0 '*(n-1), '0 '*(n+1)]:
        r = subprocess.run([str(binary),'--matvec'],input=bad,text=True,capture_output=True,timeout=5,preexec_fn=child_limits)
        assert r.returncode==2, 'invalid injected activations must be rejected'
    ldd = subprocess.run(['ldd',str(binary)],capture_output=True,text=True)
    # Restore an exact reproducible fixture for delivery only AFTER isolated runtime tests.
    model.write_bytes(original)
    result={'independent_gguf_reader':'tools/gguf_fixture_reader.py; separate Python GGUF v3 subset implementation','roundtrip_passed':True,'fixture_bytes':len(original),'rss':{'parent_peak_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'children_peak_kib_cumulative':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,'scope':'Linux process high-water marks, cumulative for this driver; container has its own 128 MiB hard limit'},'kind':'measured_synthetic_AOT','n':n,'seed':seed,'fixture_sha256':hashlib.sha256(original).hexdigest(),
            'fixture_metadata_contract':'custom test metadata, NOT a standardized GGUF operation graph',
            'gguf_deleted_before_runtime':True,'weights_embedded':True,'transformer_inference':False,
            'exporter':str(exporter),'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
            'compiler':subprocess.check_output(['cc','--version'],text=True).splitlines()[0],
            'target':subprocess.check_output(['cc','-dumpmachine'],text=True).strip(),
            'dynamic_dependencies':ldd.stdout,'linkage':'system libc/libm allowed; not proven fully static',
            'kernel_resources':{'A':'embedded F32','B_C':'float64 scratch','max_n':MAX_N},
            'fixture_generation_wall_seconds':generation_seconds,'BML_import_wall_seconds':import_seconds,'codegen_wall_seconds':codegen_seconds,'compile_wall_seconds':build_seconds,'static_operand':'A fixed and embedded at compilation; x/B dynamic at runtime','binary_bytes':binary.stat().st_size,
            'atol':1e-9,'rtol':1e-9,'commands':commands,'comparisons':comparisons}
    (out/'measurement.json').write_text(json.dumps(result,indent=2)+'\n')
    return {'fixture_generation_wall_seconds':generation_seconds,'BML_import_wall_seconds':import_seconds,'codegen_wall_seconds':codegen_seconds,'n':n,'binary_bytes':result['binary_bytes'],'compile_wall_seconds':build_seconds,
            'operations':[{k:v for k,v in c.items() if k in ['operation','kernel_peak_rss_kib','max_abs_error','process_wall_seconds_including_startup_and_io','clean_environment']} for c in comparisons]}

def memory_preflight():
    info = dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(info['MemAvailable'].split()[0])*1024
    result = {'host_available_bytes': available}
    cg = Path('/sys/fs/cgroup')
    if (cg/'memory.max').exists():
        maximum = (cg/'memory.max').read_text().strip()
        current = int((cg/'memory.current').read_text())
        result.update(cgroup_limit=maximum, cgroup_current_bytes=current)
        if (cg/'memory.peak').exists():
            result['cgroup_memory_peak_bytes'] = int((cg/'memory.peak').read_text())
            result['cgroup_peak_scope'] = 'entire cloud cgroup lifetime, includes other processes; not isolated kernel peak'
        if maximum != 'max':
            available = min(available, int(maximum)-current)
    result['conservative_available_bytes'] = available
    if available < 1024**3:
        raise RuntimeError('preflight requires at least 1 GiB headroom; no matrix materialized')
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sizes',type=int,nargs='+',default=[10,20,40,80])
    p.add_argument('--seed',type=int,default=20261030)
    p.add_argument('--exporter',type=Path,default=Path('target/release/examples/synthetic_export'))
    p.add_argument('--clean-image',help='Existing minimal Linux image with no BML; never pulled automatically')
    a=p.parse_args()
    # Reject unsafe materialization before creating anything or computing estimates.
    if not all(1<=n<=MAX_N for n in a.sizes):
        p.error('measurement sizes must be 1..128; larger doubling sizes are analytical only')
    preflight = memory_preflight()
    a.output.mkdir(parents=True,exist_ok=False)
    # Fixed dimensions cap A to 64 KiB; compiler child memory <=512 MiB, CPU <=30 s.
    measurements=[measure(n,a.seed,a.output/f'n{n}',a.exporter.resolve(),a.clean_image) for n in a.sizes]
    estimates=[dict(estimate(n),execution_status='measured separately' if n in a.sizes else 'not executed; analytical only',conservative_GGUF_bytes_bound=4*n*n+65536) for n in planned_sizes()]
    summary={'scaling_sequence':'10 * 2^k; replaces former million-square proposal','max_matrix_and_GGUF_bytes_decimal':MAX_FILE_BYTES,'last_planned_dimension':planned_sizes()[-1],'next_dimension_rejected_by_storage_budget':40960,'limits':{'real_dimension_max':128,'compiler_address_space_bytes':512*1024*1024,'child_cpu_seconds':30,'subprocess_timeout_seconds':60,'container_memory_bytes':128*1024*1024,'docker_cli_address_space_capped':False},'memory_preflight':preflight,'seed':a.seed,'max_measured_n':MAX_N,'measurements':measurements,'estimates':estimates,
             'cost_scope':'synthetic arithmetic only; wall time includes process/IO, not kernel benchmark or inference TPS'}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
