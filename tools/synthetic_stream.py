#!/usr/bin/env python3
"""One bounded streaming experiment inside its own <=7 GiB container."""
import argparse
import hashlib
import json
import mmap
import math
import os
from pathlib import Path
import resource
import shutil
import struct
import subprocess
import time

os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
try:
    import numpy as np
except ImportError:
    np = None
from gguf_fixture_reader import read_fixture
from synthetic_aot import string

JOB_MEMORY=7*1024**3
MAX_FILE=3_000_000_000
BASELINE_MM_SECONDS=0.982674173


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def snapshot():
    cg=Path('/sys/fs/cgroup')
    return {k:(cg/k).read_text().strip() for k in ['memory.max','memory.current','memory.peak'] if (cg/k).exists()}


def header(n,seed):
    meta={'general.architecture':'bml-synthetic-matrix','bml.test.seed':str(seed),
          'bml.test.generator':'PCG64 int16 stream row batches64, integers[-128,128]/16, F32'}
    b=b'GGUF'+struct.pack('<IQQ',3,1,len(meta))
    for k,v in meta.items():b+=string(k)+struct.pack('<I',8)+string(v)
    b+=string('matrix.weight')+struct.pack('<IQQIQ',2,n,n,0,0)
    return b+bytes((-len(b))%32)


def positive_finite(value):
    try:
        result=float(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError('calibration must be positive and finite') from error
    if not math.isfinite(result) or result<=0:
        raise argparse.ArgumentTypeError('calibration must be positive and finite')
    return result


def operation_plan(n, calibration):
    calibration=positive_finite(calibration)
    plan=[]
    for name,cols in [('matvec',1),('matmul',n)]:
        predicted=2*calibration*(n/1280)**3 if name=='matmul' else None
        memory=4*n*n+40*n*cols+512*1024**2
        reason=None
        if predicted is not None and predicted>60:
            reason='2x calibrated cubic estimate exceeds 60 s/case'
        elif memory>6*1024**3:
            reason='conservative workspace >6 GiB within 7 GiB job'
        plan.append({'operation':name,'cols':cols,'planned':reason is None,
                     'reason':reason,'predicted_kernel_seconds_with_2x_margin':predicted,
                     'conservative_memory_estimate_bytes':memory})
    return plan


def disk_budget(n, calibration):
    plan=operation_plan(n,calibration)
    active=[p for p in plan if p['planned']]
    # Both variants of every active operation are prepared before embedding.
    # Each input is F32 (4 bytes), each reference/output F64 (8 bytes).
    retained=sum(2*(4+8)*n*p['cols'] for p in active)
    peak=retained
    for p in active:
        peak=max(peak,retained+8*n*p['cols'])  # one live actual output
        retained-=2*(4+8)*n*p['cols']
    # embed_in_place rejects launchers >1 MiB; include alignment and footer.
    resource=max(len(header(n,20261030))+4*n*n,4*n*n+1024**2+4096+40)
    reserve=512*1024**2
    return {'disk_required_bytes':resource+peak+reserve,
            'resource_disk_bound_bytes':resource,'peak_auxiliary_disk_bytes':peak,
            'disk_reserve_bytes':reserve,'operations':plan}


def preflight(n,folder,calibration=BASELINE_MM_SECONDS):
    size=len(header(n,20261030))+4*n*n
    if n not in [2560,5120,10240,20480] or size>MAX_FILE:raise ValueError('outside authorized doubling/file limit')
    cg=snapshot()
    if cg.get('memory.max') in [None,'max'] or int(cg['memory.max'])>JOB_MEMORY:
        raise RuntimeError('requires enforced container memory.max <=7 GiB; refusing host execution')
    budget=disk_budget(n,calibration)
    required=budget['disk_required_bytes']
    free=shutil.disk_usage(folder).free
    if free<required:raise RuntimeError('insufficient disk for resource, prepared inputs/references, output and reserve')
    return {**budget,'GGUF_bytes':size,'disk_free_bytes':free,'disk_required_bytes':required,'cgroup':cg,
            'memory_budget_bytes':JOB_MEMORY,'generation_batch_rows':64}


def kernel(n):
    return r'''#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/resource.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#define N DIMENSION
typedef struct {char magic[8];uint64_t n,offset,bytes,fnv;} Footer;
_Static_assert(sizeof(Footer)==40,"footer ABI");
static uint64_t fnv(const unsigned char *b,size_t n){uint64_t h=UINT64_C(14695981039346656037);for(size_t i=0;i<n;i++){h^=b[i];h*=UINT64_C(1099511628211);}return h;}
static const float *A;
static void *mapping;
static size_t mapping_bytes;
static int open_resource(void){
 int fd=open("/proc/self/exe",O_RDONLY);struct stat st;
 if(fd<0||fstat(fd,&st)||st.st_size<40)return 2;
 mapping_bytes=(size_t)st.st_size;mapping=mmap(NULL,mapping_bytes,PROT_READ,MAP_PRIVATE,fd,0);close(fd);
 if(mapping==MAP_FAILED)return 2;
 Footer f;memcpy(&f,(const char*)mapping+mapping_bytes-40,40);
 if(memcmp(f.magic,"BMLF32V1",8)||f.n!=N||f.offset>mapping_bytes-40||f.bytes!=mapping_bytes-40-f.offset||f.bytes!=(uint64_t)N*N*4||f.offset%4096)return 2;
 const unsigned char *data=(const unsigned char*)mapping+f.offset;
 if(fnv(data,(size_t)f.bytes)!=f.fnv)return 2;
 A=(const float*)data;return 0;
}
static double secs(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(int argc,char **argv){
#ifdef BUILD_FNV
 if(argc!=4)return 2;
 FILE *fp=fopen(argv[1],"rb");if(!fp)return 2;
 uint64_t offset=strtoull(argv[2],NULL,10),remain=strtoull(argv[3],NULL,10),h=UINT64_C(14695981039346656037);
 if(fseeko(fp,(off_t)offset,SEEK_SET))return 2;
 unsigned char block[65536];
 while(remain){size_t want=remain<sizeof(block)?(size_t)remain:sizeof(block);if(fread(block,1,want,fp)!=want)return 2;for(size_t i=0;i<want;i++){h^=block[i];h*=UINT64_C(1099511628211);}remain-=want;}
 fclose(fp);printf("%llu\n",(unsigned long long)h);return 0;
#endif
 if(argc!=2)return 2;
 int cols=strcmp(argv[1],"--matmul")==0?N:1;
 if(cols==1 && strcmp(argv[1],"--matvec")!=0)return 2;
 double integrity_start=secs();if(open_resource())return 2;double integrity=secs()-integrity_start;
 size_t count=(size_t)N*cols;
 float *B=malloc(count*sizeof(float));double *C=calloc(count,sizeof(double));
 if(!B||!C||fread(B,sizeof(float),count,stdin)!=count||fgetc(stdin)!=EOF)return 2;
 for(size_t i=0;i<count;i++)if(!isfinite(B[i]))return 2;
 double t=secs();
 for(int r=0;r<N;r++)for(int k=0;k<N;k++){
  double w=A[(size_t)r*N+k];
  for(int c=0;c<cols;c++)C[(size_t)r*cols+c]+=w*(double)B[(size_t)k*cols+c];
 }
 double elapsed=secs()-t;
 for(size_t i=0;i<count;i++)if(!isfinite(C[i]))return 2;
 if(fwrite(C,sizeof(double),count,stdout)!=count)return 2;
 struct rusage u;getrusage(RUSAGE_SELF,&u);
 fprintf(stderr,"{\"kernel_seconds\":%.9g,\"integrity_seconds\":%.9g,\"peak_rss_kib\":%ld}\n",elapsed,integrity,u.ru_maxrss);
 free(B);free(C);munmap(mapping,mapping_bytes);return 0;
}
'''.replace('DIMENSION',str(n))


def child_limits():
    resource.setrlimit(resource.RLIMIT_CPU,(60,60))
    resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,6*1024**3))


def embed_in_place(model, launcher, destination, source_offset, n, checksum):
    prefix=launcher.read_bytes()
    if len(prefix)>1024**2 or destination.exists():raise ValueError('unexpected launcher/destination')
    payload=4*n*n;offset=(len(prefix)+4095)//4096*4096
    if offset<source_offset or model.stat().st_size-source_offset!=payload:raise ValueError('invalid resource layout')
    # Move backwards to preserve overlapping bytes; only 4 MiB temporary RAM.
    with model.open('r+b') as f:
        f.truncate(offset+payload+40)
        remaining=payload
        while remaining:
            count=min(4*1024**2,remaining);begin=remaining-count
            f.seek(source_offset+begin);block=f.read(count)
            if len(block)!=count:raise ValueError('short resource read')
            f.seek(offset+begin);f.write(block);remaining=begin
        f.seek(0);f.write(prefix);f.write(bytes(offset-len(prefix)))
        f.seek(offset+payload);f.write(struct.pack('<8sQQQQ',b'BMLF32V1',n,offset,payload,checksum))
    model.rename(destination);destination.chmod(0o755)
    return offset


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n',type=int,required=True)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20261030)
    p.add_argument('--mm-seconds-at-1280',type=positive_finite,default=BASELINE_MM_SECONDS)
    a=p.parse_args();
    if np is None: p.error('NumPy reference required; never installed automatically')
    a.work.mkdir(parents=True,exist_ok=False);a.results.mkdir(parents=True,exist_ok=False)
    n=a.n;report={'n':n,'status':'started','seed':a.seed,'reference_numpy':np.__version__,
        'operations':[],'scope':'static F32 A; binary stdin F32 x/B; F64 output/accumulation; not model inference'}
    result=a.results/'result.json'
    def save():result.write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        report['preflight']=preflight(n,a.work,a.mm_seconds_at_1280);save()
        model=a.work/'fixture.gguf';rng=np.random.Generator(np.random.PCG64(a.seed))
        t=time.perf_counter()
        with model.open('wb') as f:
            f.write(header(n,a.seed))
            for start in range(0,n,64):
                rows=min(64,n-start)
                block=rng.integers(-128,129,size=(rows,n),dtype=np.int16).astype('<f4')/16
                f.write(block.tobytes())
        report['generation_seconds']=time.perf_counter()-t
        report['GGUF_bytes']=model.stat().st_size;report['GGUF_sha256']=digest(model)
        t=time.perf_counter()
        r=subprocess.run(['/opt/stream/probe',str(model)],capture_output=True,text=True,timeout=30,check=True)
        imported=json.loads(r.stdout)
        report['BML_metadata_probe_seconds']=time.perf_counter()-t
        f=model.open('rb');mm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
        independent=read_fixture(mm,max_dimension=20480,decode_values=False)
        assert independent['dims']==[n,n] and independent['data_offset']==imported['offset']
        matrix=np.ndarray((n,n),dtype='<f4',buffer=mm,offset=imported['offset'])
        indices=[0,n-1,n*n//2,n*n-1]
        assert imported['samples']==[float(matrix.ravel()[i]) for i in indices]
        report['independent_roundtrip']='mmap Python GGUF header/layout and BML bounded samples agree; no global F32 expansion'
        source=a.work/'kernel.c';source.write_text(kernel(n))
        launcher=a.work/'launcher';binary=a.work/'kernel';hasher=a.work/'fnv';t=time.perf_counter()
        commands=[['cc','-std=c11','-O2','-fno-fast-math',str(source),'-lm','-o',str(launcher)],
                  ['cc','-std=c11','-O2','-DBUILD_FNV',str(source),'-lm','-o',str(hasher)]]
        for command in commands:
            c=subprocess.run(command,capture_output=True,text=True,timeout=120,preexec_fn=child_limits)
            if c.returncode:raise RuntimeError('compile failed: '+c.stderr[:300])
        report['compile_seconds']=time.perf_counter()-t
        checksum=int(subprocess.check_output([str(hasher),str(model),str(imported['offset']),str(4*n*n)],text=True,timeout=60).strip())
        hasher.unlink()
        # Compute independent references before consuming the GGUF in place.
        prepared=[]
        for planned in operation_plan(n,a.mm_seconds_at_1280):
            name=planned['operation'];cols=planned['cols']
            op={k:v for k,v in planned.items() if k not in ['cols','planned']}
            op['inputs']=[]
            if not planned['planned']:
                op['status']='preflight_not_executed';report['operations'].append(op);save();continue
            for variant in [0,1]:
                bfile=a.work/f'{name}-{variant}.input';efile=a.work/f'{name}-{variant}.expected'
                brng=np.random.Generator(np.random.PCG64(a.seed+1+variant))
                b=np.memmap(bfile,dtype='<f4',mode='w+',shape=(n,cols))
                for row in range(0,n,64):b[row:row+64]=brng.integers(-64,65,size=(min(64,n-row),cols),dtype=np.int16).astype('<f4')/8
                b.flush();ref=np.memmap(efile,dtype='<f8',mode='w+',shape=(n,cols))
                t=time.perf_counter();b64=np.asarray(b,dtype=np.float64)
                for row in range(0,n,64):ref[row:row+64]=np.asarray(matrix[row:row+64],dtype=np.float64)@b64
                ref.flush();del b64
                op['inputs'].append({'variant':variant,'input_sha256':digest(bfile),'reference_sha256':digest(efile),'reference_seconds':time.perf_counter()-t,'status':'reference_prepared'})
                del ref,b
            prepared.append((op,cols))
        del matrix;mm.close();f.close()
        t=time.perf_counter()
        offset=embed_in_place(model,launcher,binary,imported['offset'],n,checksum)
        launcher.unlink()
        report.update(embedding_seconds=time.perf_counter()-t,embedding='single-file in-place normalized F32 resource; aligned/versioned footer; self mmap; FNV64 accidental-corruption check plus external SHA256',binary_bytes=binary.stat().st_size,binary_sha256=digest(binary),C_source_bytes=source.stat().st_size,resource_offset=offset,resource_bytes=4*n*n,resource_fnv64=checksum)
        report['linkage']=subprocess.check_output(['ldd',str(binary)],text=True)
        report['cgroup_after_embedding']=snapshot();save()
        for op,cols in prepared:
            name=op['operation']
            for entry in op['inputs']:
                variant=entry['variant'];bfile=a.work/f'{name}-{variant}.input';efile=a.work/f'{name}-{variant}.expected';ofile=a.work/'actual.bin'
                ref=np.memmap(efile,dtype='<f8',mode='r',shape=(n,cols));begin=time.perf_counter()
                try:
                    with bfile.open('rb') as inp,ofile.open('wb') as out:
                        proc=subprocess.run([str(binary),'--'+name],stdin=inp,stdout=out,stderr=subprocess.PIPE,timeout=60,preexec_fn=child_limits)
                    process_seconds=time.perf_counter()-begin
                    if proc.returncode:raise RuntimeError(f'kernel exit {proc.returncode}')
                    actual=np.memmap(ofile,dtype='<f8',mode='r',shape=(n,cols))
                    max_error=0.0
                    for row in range(0,n,64):
                        x=actual[row:row+64];y=ref[row:row+64]
                        max_error=max(max_error,float(np.max(np.abs(x-y))))
                        if not np.all(np.isfinite(x)) or not np.allclose(x,y,atol=1e-9,rtol=1e-9):raise RuntimeError('reference mismatch')
                    entry.update(status='passed',process_seconds_including_binary_IO=process_seconds,max_abs_error=max_error,output_sha256=digest(ofile),**json.loads(proc.stderr))
                    del actual
                except subprocess.TimeoutExpired:
                    entry.update(status='timeout',wall_timeout_seconds=60)
                del ref
                for path in [bfile,efile,ofile]:path.unlink(missing_ok=True)
                if entry['status']=='timeout':break
            op['status']='passed' if all(v['status']=='passed' for v in op['inputs']) else 'timeout'
            if op['status']=='passed':
                assert op['inputs'][0]['output_sha256']!=op['inputs'][1]['output_sha256']
                op['different_runtime_inputs_change_output']=True
            report['operations'].append(op);report['cgroup_after_operation']=snapshot();save()
        # Keep binary only for controller's independent clean-container matvec check,
        # then controller deletes it. No GB artifacts copied into results/source.
        report.update(status='completed',GGUF_deleted=True,cgroup_final=snapshot(),
                      driver_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      atol=1e-9,rtol=1e-9)
    except Exception as e:
        report.update(status='failed',error=type(e).__name__+': '+str(e))
    save();print(json.dumps({'n':n,'status':report['status']}))
    return 0 if report['status']=='completed' else 2

if __name__=='__main__':raise SystemExit(main())
