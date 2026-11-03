#!/usr/bin/env python3
"""Audit the retained x86-64 worker dispatcher, resolving GOT calls and jump tables."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[1]
HOT = 'bml_runtime::hot_loop::dispatch_ops'
DIRECT = {HOT, '<alloc::raw_vec::RawVec<f64>>::grow_one', 'core::panicking::panic_bounds_check'}
MATH = {'exp2', 'log2'}


def run(args):
    return subprocess.check_output(args, cwd=ROOT, text=True)


def mapped_bytes(binary, address, size):
    header = struct.unpack_from('<16sHHIQQQIHHHHHH', binary)
    if header[0][:6] != b'\x7fELF\x02\x01' or header[2] != 62:
        raise ValueError('Only ELF64 little-endian x86-64 is supported')
    for index in range(header[10]):
        item = struct.unpack_from('<IIQQQQQQ', binary, header[5] + index * header[9])
        kind, _, offset, virtual, _, file_size, _, _ = item
        if kind == 1 and virtual <= address and address + size <= virtual + file_size:
            start = offset + address - virtual
            result = binary[start:start+size]
            if len(result) != size:
                raise ValueError('Truncated ELF mapping')
            return result
    raise ValueError('Jump table is outside file-backed ELF segments')


def audit(binary, assembly, symbols, relocations, variants):
    sized = {}
    for line in symbols.splitlines():
        fields = line.split(maxsplit=3)
        if len(fields) == 4 and fields[2].lower() == 't':
            sized[fields[3]] = (int(fields[0], 16), int(fields[1], 16))
    if HOT not in sized or 'bml_runtime::net::send_msg' not in sized:
        raise ValueError('Worker must retain both dispatcher and RPC sender symbols')
    start, size = sized[HOT]
    if not 0 < size < 32768:
        raise ValueError('Dispatcher size is missing or exceeds 32 KiB')
    relocation_map = {}
    for line in relocations.splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[1].startswith('R_X86_64_'):
            relocation_map[int(fields[0], 16)] = (fields[1], fields[2])
    imported = {value[1] for value in relocation_map.values()}
    if not {'socket', 'accept4', 'send', 'recv'}.issubset(imported):
        raise ValueError('Missing network imports: this is not the expected worker binary')
    match = re.search(r'^([0-9a-f]+) <' + re.escape(HOT) + r'>:\n(.*?)(?=\n[0-9a-f]+ <|\Z)', assembly, re.M | re.S)
    if match is None or int(match[1], 16) != start:
        raise ValueError('Missing or inconsistent dispatcher disassembly')
    instructions = []
    for line in match[2].splitlines():
        item = re.match(r'\s*([0-9a-f]+):\s+(\S+)\s*(.*)', line)
        if item:
            address = int(item[1], 16)
            if start <= address < start + size:
                instructions.append((address, item[2], item[3]))
    addresses = {item[0] for item in instructions}
    calls, tables = [], []
    for position, (address, mnemonic, args) in enumerate(instructions):
        if mnemonic in {'syscall', 'sysenter', 'sysret', 'sysretq', 'int', 'intq', 'int3', 'lcall', 'lcallq', 'ljmp', 'ljmpq', 'iret', 'iretq', 'xbegin'}:
            raise ValueError('System call/trap in dispatcher')
        if mnemonic.startswith('call'):
            if args.startswith('*'):
                got = re.match(r'\*(-?0x[0-9a-f]+)\(%rip\)\s*#\s*0x([0-9a-f]+)', args)
                if got is None or position + 1 >= len(instructions):
                    raise ValueError('Unresolved indirect call in dispatcher')
                if instructions[position+1][0] + int(got[1], 16) != int(got[2], 16):
                    raise ValueError('GOT address disagrees with RIP displacement')
                relocation = relocation_map.get(int(got[2], 16))
                if relocation is None or relocation[0] != 'R_X86_64_GLOB_DAT' or relocation[1] not in MATH:
                    raise ValueError('Unapproved indirect call target')
                target = relocation[1]
            else:
                direct = re.match(r'0x([0-9a-f]+) <(.+)>$', args)
                if direct is None or direct[2] not in DIRECT or sized.get(direct[2], (None,))[0] != int(direct[1], 16):
                    raise ValueError('Unapproved direct call target')
                target = direct[2]
            calls.append({'address': address, 'target': target})
        if mnemonic.startswith(('j', 'loop')):
            if not args.startswith('*'):
                target = re.match(r'0x([0-9a-f]+)', args)
                if target is None or int(target[1], 16) not in addresses:
                    raise ValueError('Branch escapes dispatcher or instruction boundaries')
                continue
            if mnemonic != 'jmpq' or args != '*%rcx':
                raise ValueError('Unrecognized indirect jump')
            window = instructions[max(0, position-4):position]
            # Exact compiler idiom: load signed i32 table displacement, add
            # the table base, then jump. Optional unrelated stack spill allowed.
            bases = [(i, re.search(r'#\s*0x([0-9a-f]+)', op)) for i, (_, code, op) in enumerate(window)
                     if code == 'leaq' and '(%rip), %rdx' in op]
            if not bases or bases[-1][1] is None:
                raise ValueError('No proved jump-table base')
            index, table = bases[-1]
            following = window[index+1:]
            if len(following) not in (2, 3) or following[0][1:] != ('movslq', '(%rdx,%rcx,4), %rcx') or following[1][1:] != ('addq', '%rdx, %rcx'):
                raise ValueError('Unrecognized jump-table address computation')
            if len(following) == 3 and (following[2][1] != 'movq' or not following[2][2].endswith('(%rsp)')):
                raise ValueError('Jump-table register may be overwritten')
            base = int(table[1], 16)
            displacement = re.match(r'(-?0x[0-9a-f]+)\(%rip\), %rdx', window[index][2])
            if displacement is None or following[0][0] + int(displacement[1], 16) != base:
                raise ValueError('Table base disagrees with RIP displacement')
            offsets = struct.unpack('<' + 'i' * variants, mapped_bytes(binary, base, 4 * variants))
            targets = [base + offset for offset in offsets]
            if any(target not in addresses for target in targets):
                raise ValueError('Jump-table target escapes dispatcher')
            tables.append({'branch_address': address, 'table_address': base, 'targets': targets})
    if not calls or not tables:
        raise ValueError('Expected calls/jump-table evidence missing')
    return {'symbol': HOT, 'address': start, 'bytes': size, 'calls': calls,
            'jump_tables': tables, 'worker_network_imports': sorted(imported & {'socket', 'accept4', 'send', 'recv'})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    build = ['cargo', 'build', '-p', 'bml-worker', '--release', '--locked', '--config', 'profile.release.strip=false']
    run(build)
    binary = ROOT / 'target/release/bml-worker'
    compiler = run(['rustc', '-vV'])
    host = next(line.split(': ', 1)[1] for line in compiler.splitlines() if line.startswith('host: '))
    llvm = Path(run(['rustc', '--print', 'sysroot']).strip()) / 'lib/rustlib' / host / 'bin'
    assembly = run([str(llvm/'llvm-objdump'), '--disassemble', '--demangle', '--no-show-raw-insn', str(binary)])
    symbols = run([str(llvm/'llvm-nm'), '--demangle', '--print-size', str(binary)])
    relocations = run([str(llvm/'llvm-objdump'), '-R', str(binary)])
    source = (ROOT/'crates/compiler/src/rpn.rs').read_text().split('pub enum RpnOp {', 1)[1].split('/// Programa RPN:', 1)[0]
    variants = re.findall(r'^    (\w+)(?:,|\(| \{)', source, re.M)
    if len(variants) != 14:
        raise ValueError('RpnOp layout changed; review the binary audit before updating it')
    result = audit(binary.read_bytes(), assembly, symbols, relocations, len(variants))
    artifacts = ROOT/'target/quality'; artifacts.mkdir(parents=True, exist_ok=True)
    for name, data in [('worker-audit.asm', assembly), ('worker-audit.nm', symbols), ('worker-audit.reloc', relocations)]:
        (artifacts/name).write_text(data)
    files = ['crates/runtime/src/hot_loop.rs', 'crates/runtime/src/buffer.rs', 'crates/runtime/src/graph.rs',
             'crates/runtime/src/graph_rpc.rs', 'crates/runtime/src/net.rs', 'crates/worker/src/main.rs',
             'crates/compiler/src/rpn.rs', 'Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', 'tools/audit_hot_loop_binary.py']
    report = {'schema': 1, 'date_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'compiler': compiler, 'build_command': build, 'result': result,
              'flags': {key: os.environ.get(key) for key in ['RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS']},
              'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
              'source_sha256': {f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files},
              'disassembly_sha256': hashlib.sha256(assembly.encode()).hexdigest(),
              'scope': 'Retained x86-64 dispatcher inside the real worker, with network routines linked. '
                       'All calls resolve to recursion, exp2/log2, Vec growth or bounds panic; all direct '
                       'branches and computed jump-table targets stay inside the dispatcher. No RPC/socket '
                       'calls or syscalls occur in it. Allocator/panic slow paths are identified, not '
                       'transitively verified; arbitrary panic hooks, allocator replacement or dynamic '
                       'library interposition are outside this audit. ValidatedGraph reserves the peak '
                       'stack and validates references before execution. No PMU/cache-residency claim.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(f"Audited {result['bytes']} dispatcher bytes, {len(result['calls'])} calls and {len(result['jump_tables'])} jump-table branches")


if __name__ == '__main__':
    main()
