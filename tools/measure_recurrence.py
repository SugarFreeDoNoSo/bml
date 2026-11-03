#!/usr/bin/env python3
"""Record instrumented equal-work recurrence metrics with source provenance."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def run(command):
    return subprocess.check_output(command, cwd=ROOT, text=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    command = ['cargo', 'run', '-p', 'bml-compiler', '--example',
               'recurrence_metrics', '--release', '--locked']
    rows = json.loads(run(command))
    expected = {(variant, steps, initial)
                for variant in ['direct_scalar', 'rpn_deduplicated', 'rpn_unshared']
                for steps in [10, 101, 1000, 10001] for initial in [1, 2]}
    actual = {(r['variant'], r['steps'], r['initial']) for r in rows}
    if actual != expected or len(rows) != len(expected):
        raise RuntimeError('Incomplete or duplicated measurement cases')
    for row in rows:
        if row['absolute_error'] != 0 or len(row['samples']) != 7:
            raise RuntimeError('Incorrect result or incomplete samples')
        for sample in row['samples']:
            if sample['ns'] <= 0:
                raise RuntimeError('Invalid elapsed time')
    sources = ['crates/compiler/examples/recurrence_metrics.rs',
               'crates/compiler/benches/support/recurrence.rs',
               'crates/compiler/src/rpn.rs', 'Cargo.toml', 'Cargo.lock',
               'rust-toolchain.toml', 'tools/measure_recurrence.py']
    report = {
        'schema': 1,
        'measured_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'command': command,
        'revision': run(['git', 'rev-parse', 'HEAD']).strip(),
        'worktree_dirty': bool(run(['git', 'status', '--porcelain'])),
        'compiler': run(['rustc', '-vV']),
        'flags': {name: os.environ.get(name)
                  for name in ['RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS']},
        'hardware': run(['lscpu']),
        'source_sha256': {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()
                          for f in sources},
        'scope': 'Single-thread instrumented execution, 7 samples of 10 evaluations. '
                 'Construction, correctness checks and JSON formatting excluded. '
                 'Allocator forwards to System and counts successful calls including realloc; '
                 'requested bytes include complete replacement sizes, not just growth. '
                 'Peak is additional requested live heap, not RSS or allocator overhead. '
                 'Bytecode/pool sizes and capacities are data, not machine-code size. '
                 'Timers include allocator instrumentation. VM may have concurrent load; '
                 'these are not dedicated-hardware latency or inference claims.',
        'cases': rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Recorded {len(rows)} cases in {args.output}')


if __name__ == '__main__':
    main()
