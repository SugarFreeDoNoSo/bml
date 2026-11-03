#!/usr/bin/env python3
"""Measure linked hot-loop machine code, not rlib size or L1 residency."""
import argparse
import datetime
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]

NETWORK_SYMBOL = re.compile(
    r'bml_runtime::net::|std::net::|std::sys::net::|'
    r'\b(?:socket|socketpair|connect|accept4?|bind|listen|sendto|sendmsg|'
    r'recvfrom|recvmsg|getaddrinfo|freeaddrinfo|shutdown)(?:@\S*)?\b')


def network_symbols(symbols):
    """Report linked/imported networking symbols, preserving diagnostic lines."""
    return [line.strip() for line in symbols.splitlines()
            if NETWORK_SYMBOL.search(line)]


def run(args):
    return subprocess.run(args, cwd=ROOT, check=True, text=True,
                          stdout=subprocess.PIPE).stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build = ['cargo', 'build', '-p', 'bml-runtime', '--example', 'hot_loop_probe',
             '--release', '--locked', '--config', 'profile.release.strip=false',
             '--message-format=json']
    messages = run(build)
    executables = [json.loads(line).get('executable') for line in messages.splitlines()
                   if line.startswith('{') and json.loads(line).get('reason') == 'compiler-artifact'
                   and json.loads(line).get('target', {}).get('name') == 'hot_loop_probe']
    executable = Path(next(path for path in executables if path))
    compiler = run(['rustc', '-vV'])
    host = next(line.split(': ', 1)[1] for line in compiler.splitlines() if line.startswith('host: '))
    sysroot = Path(run(['rustc', '--print', 'sysroot']).strip())
    llvm = sysroot / 'lib' / 'rustlib' / host / 'bin'
    symbols = run([str(llvm / 'llvm-nm'), '--print-size', '--size-sort', str(executable)])
    named_symbols = run([str(llvm / 'llvm-nm'), '--demangle', str(executable)])
    dynamic_symbols = run([str(llvm / 'llvm-nm'), '--dynamic', '--demangle', str(executable)])
    network = network_symbols(named_symbols + '\n' + dynamic_symbols)
    matches = []
    for line in symbols.splitlines():
        fields = line.split()
        if len(fields) == 4 and fields[2].lower() == 't' and 'dispatch_ops' in fields[3]:
            matches.append({'symbol': fields[3], 'address': int(fields[0], 16),
                            'bytes': int(fields[1], 16)})
    if not matches or any(item['bytes'] == 0 for item in matches):
        raise RuntimeError('No sized dispatch_ops machine-code symbol; measurement unproven')
    artifact_dir = ROOT / 'target' / 'quality'
    artifact_dir.mkdir(parents=True, exist_ok=True)
    disassembly = run([str(llvm / 'llvm-objdump'), '--disassemble', '--demangle', str(executable)])
    assembly_file = artifact_dir / 'hot-loop-probe.asm'
    assembly_file.write_text(disassembly)
    sources = ['crates/runtime/src/hot_loop.rs', 'crates/runtime/examples/hot_loop_probe.rs',
               'crates/runtime/src/buffer.rs', 'Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml',
               'tools/measure_hot_loop.py']
    evidence = {
        'schema': 1,
        'measured_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'revision': run(['git', 'rev-parse', 'HEAD']).strip(),
        'worktree_dirty': bool(run(['git', 'status', '--porcelain'])),
        'compiler': compiler.strip(), 'build_command': build,
        'extra_flags': {name: os.environ.get(name) for name in ['RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS']},
        'probe_output': run([str(executable)]).strip(),
        'executable_sha256': hashlib.sha256(executable.read_bytes()).hexdigest(),
        'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources},
        'symbols': matches, 'total_dispatch_symbol_bytes': sum(s['bytes'] for s in matches),
        'threshold_bytes': 32768,
        'dispatch_symbols_below_threshold': sum(s['bytes'] for s in matches) < 32768,
        'scope': 'Linked probe, release optimization and fat LTO; stripping disabled solely to retain symbols. '
                 'Counts only retained dispatch_ops symbols, not inlined copies, transitive callees, '
                 'jump-table data, network code isolation, cache residency or instruction-cache misses.',
        'disassembly': str(assembly_file.relative_to(ROOT)),
        'disassembly_sha256': hashlib.sha256(disassembly.encode()).hexdigest(),
        'pmu': {'status': 'not_measured', 'reason': 'This tool measures static symbol sizes only'},
        'network_linkage': {
            'network_symbols': network,
            'no_network_symbols_in_linked_probe': not network,
            'symbol_table_sha256': hashlib.sha256(named_symbols.encode()).hexdigest(),
            'dynamic_symbol_table_sha256': hashlib.sha256(dynamic_symbols.encode()).hexdigest(),
            'scope': 'Scans the complete unstripped probe and its dynamic imports. '
                     'This establishes that the standalone hot-loop probe does not link '
                     'named RPC/socket routines. It does not establish transitive call '
                     'isolation in the worker/server executable, exclude raw syscalls, '
                     'or measure instruction-cache residency.',
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps({'symbols': matches, 'output': str(args.output)}))


if __name__ == '__main__':
    main()
