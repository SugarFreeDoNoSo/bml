#!/usr/bin/env python3
"""Offline bounded Q4_0 inspection, not model inference or a TPS benchmark."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--output', default='/results')
    p.add_argument('--tensor', default='token_embd.weight')
    p.add_argument('--row-start', type=int, default=0)
    p.add_argument('--rows', type=int, choices=range(1, 9), default=2)
    p.add_argument('--blocks', type=int, choices=range(1, 129), default=1)
    p.add_argument('--timeout', type=int, default=60)
    p.add_argument('--hash-model', action='store_true', help='Opt in to reading the entire model to hash it')
    a = p.parse_args()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    report = {'schema': 1, 'status': 'failed', 'scope': 'bounded rank-2 Q4_0 tensor inspection/dequantization/projection',
              'full_model_inference_validated': False, 'tps': None, 'platform': platform.platform(),
              'machine': platform.machine(), 'arguments': vars(a), 'commands': [],
              'reference': json.loads((ROOT/'reference/provenance.json').read_text()),
              'build': json.loads((ROOT/'build.json').read_text()) if (ROOT/'build.json').exists() else {'kind': 'host development'},
              'limits': {'max_rows': 8, 'max_blocks_per_row': 128, 'decoded_elements_per_row': a.blocks*32}}
    logs = []
    def run(cmd):
        report['commands'].append(cmd)
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
        logs.append(json.dumps({'command': cmd, 'exit': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr}))
        if r.returncode:
            raise RuntimeError(f'command exited {r.returncode}: {r.stderr.strip()}')
        return r.stdout
    try:
        if a.row_start < 0 or not 1 <= a.timeout <= 600:
            raise ValueError('row-start must be nonnegative; timeout must be 1..600 seconds')
        if sha(ROOT/'reference/q4_0.inc') != report['reference']['extracted_function_sha256']:
            raise ValueError('reference source hash mismatch')
        report['runner_sha256'] = sha(__file__)
        report['reference_binary_sha256'] = sha(os.environ.get('BML_REFERENCE', '/opt/bml/q4_reference'))
        report['probe_binary_sha256'] = sha(os.environ.get('BML_PROBE', '/opt/bml/bounded_probe'))
        model = Path(a.model)
        if not model.is_file():
            raise ValueError('model file is missing or is not a regular file')
        report['model_size'] = model.stat().st_size
        if a.hash_model:
            report['model_sha256'] = sha(model)
        report['model_hash_scope'] = 'full file' if a.hash_model else 'sample bytes only; full model hash not computed'
        cmd = [os.environ.get('BML_PROBE', '/opt/bml/bounded_probe'), str(model), a.tensor, str(a.row_start), str(a.rows), str(a.blocks)]
        lines = run(cmd).splitlines()
        m = lines[0].split('\t')
        if m[0] != 'metadata' or len(m) != 8:
            raise ValueError('invalid probe metadata response')
        arch = bytes.fromhex(m[1]).decode('utf-8')
        report['metadata'] = dict(zip(['version', 'kv_count', 'tensor_count', 'ne0', 'ne1', 'file_bytes'], map(int, m[2:])))
        report['architecture'] = arch
        report['bml_execution_architecture_status'] = 'experimental llama; not validated inference' if arch == 'llama' else 'unsupported for BML inference; tensor inspection only'
        samples = []
        for line in lines[1:]:
            kind, row, encoded, decoded, dot = line.split('\t')
            if kind != 'row':
                raise ValueError('invalid sample response')
            raw = bytes.fromhex(encoded)
            sample_path = out/f'row-{int(row)}.q4_0'
            sample_path.write_bytes(raw)
            reference = [float(v) for v in run([os.environ.get('BML_REFERENCE', '/opt/bml/q4_reference'), str(sample_path)]).splitlines()]
            actual = list(map(float, decoded.split(',')))
            if len(actual) != a.blocks*32 or len(reference) != len(actual):
                raise ValueError('decoded sample length mismatch')
            activation = [(i % 17 - 8)/8 for i in range(len(reference))]
            expected_dot = math.fsum(v*x for v, x in zip(reference, activation))
            actual_dot = float(dot)
            close = lambda x, y: math.isfinite(x) and math.isfinite(y) and abs(x-y) <= 1e-6 + 1e-6*abs(y)
            ok = all(close(x, y) for x, y in zip(actual, reference)) and close(actual_dot, expected_dot)
            samples.append({'row': int(row), 'sample_sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw),
                            'bml_dot': actual_dot, 'reference_dot': expected_dot, 'passed': ok,
                            'activation': 'x[i]=((i%17)-8)/8; independent float64 dot over upstream decoded values',
                            'atol': 1e-6, 'rtol': 1e-6})
        report['samples'] = samples
        if len(samples) != a.rows or not all(s['passed'] for s in samples):
            raise ValueError('dequantization/projection comparison failed')
        report['status'] = 'passed'
    except Exception as e:
        report['error'] = f'{type(e).__name__}: {e}'
    (out/'run.log').write_text('\n'.join(logs)+'\n')
    (out/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': report['status'], 'result': str(out/'result.json')}))
    return 0 if report['status'] == 'passed' else 2

if __name__ == '__main__':
    sys.exit(main())
