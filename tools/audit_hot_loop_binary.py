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
DIRECT = {HOT, '<alloc::raw_vec::RawVec<f64>>::grow_one',
          'core::panicking::panic_bounds_check', 'core::panicking::panic_fmt'}
MATH = {'exp2', 'log2'}
RPN_VARIANTS = ('One', 'Zero', 'Bml', 'Dup', 'Var', 'Const', 'VarIndexed',
                'ContextVarIndexed', 'ContextConstIndexed', 'StoreResult', 'Loop',
                'FAdd', 'FMul', 'Pick', 'Drop', 'Swap', 'FMax')


def run(args):
    return subprocess.check_output(args, cwd=ROOT, text=True)


def worker_artifact(messages):
    """Use Cargo's actual executable path, including configured target directories."""
    paths = []
    for line in messages.splitlines():
        message = json.loads(line)
        target = message.get('target', {})
        if (message.get('reason') == 'compiler-artifact'
                and target.get('name') == 'bml-worker' and target.get('kind') == ['bin']
                and message.get('executable')):
            paths.append(Path(message['executable']))
    if len(paths) != 1 or not paths[0].is_absolute():
        raise ValueError('Expected one absolute bml-worker executable path from Cargo')
    return paths[0]


def rpn_variants(source):
    match = re.search(r'^pub enum RpnOp \{\n(.*?)^\}', source, re.M | re.S)
    variants = tuple(re.findall(r'^    (\w+)(?:,|\(| \{)', match[1], re.M)) if match else ()
    if variants != RPN_VARIANTS:
        raise ValueError('RpnOp layout changed; review the binary audit before updating it')
    return variants


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


def function_instructions(assembly, symbol, start, size):
    match = re.search(r'^([0-9a-f]+) <' + re.escape(symbol) + r'>:\n(.*?)(?=\n[0-9a-f]+ <|\Z)', assembly, re.M | re.S)
    if match is None or int(match[1], 16) != start:
        raise ValueError('Missing or inconsistent function disassembly: ' + symbol)
    instructions = []
    for line in match[2].splitlines():
        item = re.match(r'\s*([0-9a-f]+):\s+(\S+)\s*(.*)', line)
        if item:
            address = int(item[1], 16)
            if start <= address < start + size:
                instructions.append((address, item[2], item[3]))
    if not instructions or instructions[0][0] != start:
        raise ValueError('Missing first instruction: ' + symbol)
    return instructions


def audit_local_trunc(assembly, start, size):
    """Review the static trunc leaf emitted by the pinned x86-64 toolchain."""
    if not 0 < size < 128:
        raise ValueError('Unreviewed local trunc size')
    instructions = function_instructions(assembly, 'trunc', start, size)
    addresses = {item[0] for item in instructions}
    branches = []
    allowed = {'movq', 'shrq', 'andl', 'cmpl', 'ja', 'leal', 'movabsq',
               'sarq', 'cmovaeq', 'andq', 'retq'}
    for address, mnemonic, args in instructions:
        if mnemonic not in allowed:
            raise ValueError('Unreviewed instruction in local trunc helper')
        if mnemonic == 'ja':
            target = re.match(r'0x([0-9a-f]+)', args)
            if target is None or int(target[1], 16) not in addresses:
                raise ValueError('Branch escapes local trunc helper')
            branches.append({'address': address, 'target': int(target[1], 16)})
    if instructions[-1][1] != 'retq':
        raise ValueError('Local trunc helper must end in return')
    return {'symbol': 'trunc', 'address': start, 'bytes': size, 'branches': branches}


def audit(binary, assembly, symbols, relocations, variants):
    if variants != len(RPN_VARIANTS):
        raise ValueError('Unreviewed RpnOp variant count')
    sized = {}
    addresses_by_name = {}
    for line in symbols.splitlines():
        fields = line.split(maxsplit=3)
        if len(fields) == 4 and fields[2].lower() == 't':
            address, size = int(fields[0], 16), int(fields[1], 16)
            sized[fields[3]] = (address, size)
            addresses_by_name.setdefault(fields[3], set()).add(address)
    if HOT not in sized or 'bml_runtime::net::send_msg' not in sized:
        raise ValueError('Worker must retain both dispatcher and RPC sender symbols')
    if len(addresses_by_name[HOT]) != 1:
        raise ValueError('Ambiguous dispatcher symbol')
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
    instructions = function_instructions(assembly, HOT, start, size)
    addresses = {item[0] for item in instructions}
    calls, tables, local_helpers = [], [], {}
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
                if relocation is not None and relocation[0] == 'R_X86_64_GLOB_DAT' and relocation[1] in MATH:
                    target = relocation[1]
                elif relocation is not None and relocation[0] == 'R_X86_64_RELATIVE':
                    relative = re.fullmatch(r'\*ABS\*\+0x([0-9a-f]+)', relocation[1])
                    if relative is None or sized.get('trunc', (None,))[0] != int(relative[1], 16):
                        raise ValueError('Unapproved relative indirect call target')
                    target = 'trunc'
                    if target not in local_helpers:
                        local_helpers[target] = audit_local_trunc(assembly, *sized[target])
                else:
                    raise ValueError('Unapproved indirect call target')
            else:
                direct = re.match(r'0x([0-9a-f]+) <(.+)>$', args)
                if direct is None or direct[2] not in DIRECT or int(direct[1], 16) not in addresses_by_name.get(direct[2], set()):
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
            'jump_tables': tables, 'local_math_helpers': list(local_helpers.values()),
            'worker_network_imports': sorted(imported & {'socket', 'accept4', 'send', 'recv'})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    variants = rpn_variants((ROOT/'crates/compiler/src/rpn.rs').read_text())
    build = ['cargo', 'build', '-p', 'bml-worker', '--release', '--locked', '--config',
             'profile.release.strip=false', '--message-format=json']
    binary = worker_artifact(run(build))
    compiler = run(['rustc', '-vV'])
    host = next(line.split(': ', 1)[1] for line in compiler.splitlines() if line.startswith('host: '))
    llvm = Path(run(['rustc', '--print', 'sysroot']).strip()) / 'lib/rustlib' / host / 'bin'
    assembly = run([str(llvm/'llvm-objdump'), '--disassemble', '--demangle', '--no-show-raw-insn', str(binary)])
    symbols = run([str(llvm/'llvm-nm'), '--demangle', '--print-size', str(binary)])
    relocations = run([str(llvm/'llvm-objdump'), '-R', str(binary)])
    artifacts = ROOT/'target/quality'; artifacts.mkdir(parents=True, exist_ok=True)
    for name, data in [(args.output.stem+'.asm', assembly), (args.output.stem+'.nm', symbols),
                       (args.output.stem+'.reloc', relocations)]:
        (artifacts/name).write_text(data)
    result = audit(binary.read_bytes(), assembly, symbols, relocations, len(variants))
    files = ['crates/runtime/src/hot_loop.rs', 'crates/runtime/src/buffer.rs', 'crates/runtime/src/graph.rs',
             'crates/runtime/src/graph_rpc.rs', 'crates/runtime/src/net.rs', 'crates/worker/src/main.rs',
             'crates/runtime/src/runtime.rs', 'crates/runtime/src/operation_schedule.rs',
             'crates/compiler/src/rpn.rs', 'Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml',
             'tools/audit_hot_loop_binary.py', 'tools/test_audit_hot_loop_binary.py']
    report = {'schema': 1, 'date_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'compiler': compiler, 'build_command': build, 'binary_path': str(binary),
              'rpn_variants': variants, 'result': result,
              'flags': {key: os.environ.get(key) for key in ['RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS', 'CARGO_TARGET_DIR']},
              'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
              'source_sha256': {f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files},
              'disassembly_sha256': hashlib.sha256(assembly.encode()).hexdigest(),
              'scope': 'Retained x86-64 dispatcher inside the real worker, with network routines linked. '
                       'All calls resolve to recursion, exp2/log2, local trunc, Vec growth or bounds/ownership panic; all direct '
                       'branches and computed jump-table targets stay inside the dispatcher. No RPC/socket '
                       'calls or syscalls occur in it. The local trunc leaf also has no calls, traps or external branches. '
                       'Allocator/panic slow paths are identified, not '
                       'transitively verified; arbitrary panic hooks, allocator replacement or dynamic '
                       'library interposition are outside this audit. ValidatedGraph reserves the peak '
                       'stack and validates references before execution. No PMU/cache-residency claim.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(f"Audited {result['bytes']} dispatcher bytes, {len(result['calls'])} calls and {len(result['jump_tables'])} jump-table branches")


if __name__ == '__main__':
    main()
