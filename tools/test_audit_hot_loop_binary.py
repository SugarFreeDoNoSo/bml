"""Binary audit must reject network targets and unresolved control transfers."""
import struct
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from audit_hot_loop_binary import HOT, RPN_VARIANTS, audit, mapped_bytes, rpn_variants, worker_artifact


class DispatcherAudit(unittest.TestCase):
    def setUp(self):
        self.assembly = f'''0000000000000100 <{HOT}>:
 100: leaq 0xfc(%rip), %rdx # 0x200 <table>
 104: movslq (%rdx,%rcx,4), %rcx
 108: addq %rdx, %rcx
 10c: jmpq *%rcx
 110: callq *0x1ec(%rip) # 0x300 <misleading_symbol>
 114: callq 0x100 <{HOT}>
 118: retq

0000000000000120 <bml_runtime::net::send_msg>:
 120: retq
'''
        self.symbols = f'100 20 t {HOT}\n120 10 t bml_runtime::net::send_msg\n'
        self.relocations = '\n'.join(f'{address:x} R_X86_64_GLOB_DAT {name}' for address, name in
                                     [(0x300, 'log2'), (0x310, 'socket'), (0x320, 'accept4'), (0x330, 'send'), (0x340, 'recv')])

    def check(self, assembly=None, relocations=None, offsets=None, symbols=None):
        with patch('audit_hot_loop_binary.mapped_bytes', return_value=struct.pack('<16i', *(offsets or [-256]*16))):
            return audit(b'', assembly or self.assembly, symbols or self.symbols,
                         relocations or self.relocations, 16)

    def test_got_target_comes_from_relocation_not_disassembler_annotation(self):
        result = self.check()
        self.assertEqual(result['calls'][0]['target'], 'log2')
        self.assertEqual(result['jump_tables'][0]['targets'], [0x100]*16)

    def test_socket_call_and_direct_rpc_call_fail(self):
        with self.assertRaisesRegex(ValueError, 'indirect call'):
            self.check(relocations=self.relocations.replace('300 R_X86_64_GLOB_DAT log2', '300 R_X86_64_GLOB_DAT send'))
        with self.assertRaisesRegex(ValueError, 'direct call'):
            self.check(assembly=self.assembly.replace(f'callq 0x100 <{HOT}>', 'callq 0x120 <bml_runtime::net::send_msg>'))

    def test_unresolved_register_call_and_syscall_fail(self):
        for replacement in ['callq *%rax', 'callq *%rax # 0x300 <log2>', 'syscall']:
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                self.check(assembly=self.assembly.replace('callq *0x1ec(%rip) # 0x300 <misleading_symbol>', replacement))

    def test_jump_table_escape_or_changed_computation_fails(self):
        with self.assertRaisesRegex(ValueError, 'escapes'):
            self.check(offsets=[-256]*15 + [-224])  # one target is RPC, at 0x120
        with self.assertRaisesRegex(ValueError, 'computation'):
            self.check(assembly=self.assembly.replace('addq %rdx, %rcx', 'addq %rax, %rcx'))
        with self.assertRaisesRegex(ValueError, 'Branch escapes'):
            self.check(assembly=self.assembly.replace('retq', 'jmp 0x120 <bml_runtime::net::send_msg>', 1))

    def test_elf_virtual_mapping_respects_file_backed_bounds(self):
        ident = b'\x7fELF\x02\x01' + bytes(10)
        header = struct.pack('<16sHHIQQQIHHHHHH', ident, 3, 62, 1, 0, 64, 0, 0, 64, 56, 1, 0, 0, 0)
        segment = struct.pack('<IIQQQQQQ', 1, 4, 120, 0x200, 0, 4, 8, 1)
        binary = header + segment + b'abcd'
        self.assertEqual(mapped_bytes(binary, 0x201, 2), b'bc')
        with self.assertRaises(ValueError):
            mapped_bytes(binary, 0x202, 4)  # memory-backed tail is not file data

    def test_layout_checks_names_and_order_not_only_variant_count(self):
        source = 'pub enum RpnOp {\n' + ''.join(f'    {name},\n' for name in RPN_VARIANTS) + '}\n'
        self.assertEqual(rpn_variants(source), RPN_VARIANTS)
        for changed in [source.replace('ContextVarIndexed', 'Replacement'),
                        source.replace('    Zero,\n', ''),
                        source.replace('    One,\n    Zero,', '    Zero,\n    One,')]:
            with self.assertRaisesRegex(ValueError, 'layout changed'):
                rpn_variants(changed)
        with self.assertRaisesRegex(ValueError, 'variant count'):
            audit(b'', self.assembly, self.symbols, self.relocations, 14)

    def test_executable_comes_from_cargo_artifact_not_default_target_directory(self):
        artifact = {'reason': 'compiler-artifact', 'target': {'name': 'bml-worker', 'kind': ['bin']},
                    'executable': '/custom/build-target/release/bml-worker'}
        messages = '\n'.join(map(json.dumps, [{'reason': 'build-script-executed'}, artifact,
                                            {'reason': 'build-finished', 'success': True}]))
        self.assertEqual(worker_artifact(messages), Path(artifact['executable']))
        for invalid in ['', json.dumps(artifact) + '\n' + json.dumps(artifact),
                        json.dumps({**artifact, 'executable': 'target/release/bml-worker'}),
                        json.dumps({**artifact, 'target': {'name': 'other', 'kind': ['bin']}})]:
            with self.assertRaisesRegex(ValueError, 'executable path'):
                worker_artifact(invalid)

    def test_exact_ownership_panic_symbol_is_accepted_but_wrong_address_is_not(self):
        name = 'core::panicking::panic_fmt'
        symbols = self.symbols + f'400 10 t {name}\n'
        assembly = self.assembly.replace(f'callq 0x100 <{HOT}>', f'callq 0x400 <{name}>')
        self.assertEqual(self.check(assembly=assembly, symbols=symbols)['calls'][1]['target'], name)
        with self.assertRaisesRegex(ValueError, 'direct call'):
            self.check(assembly=assembly.replace(f'0x400 <{name}>', f'0x120 <{name}>'), symbols=symbols)
        with self.assertRaisesRegex(ValueError, 'direct call'):
            self.check(assembly=assembly.replace(name, 'core::panicking::other'), symbols=symbols)

    def test_relative_call_resolves_only_the_reviewed_local_trunc_leaf(self):
        helper = '''\n0000000000000400 <trunc>:
 400: movq %xmm0, %rax
 404: cmpl $0x432, %edx
 408: ja 0x40c <trunc+0xc>
 40c: retq
'''
        assembly = self.assembly + helper
        symbols = self.symbols + '400 10 t trunc\n'
        relocations = self.relocations.replace('300 R_X86_64_GLOB_DAT log2',
                                               '300 R_X86_64_RELATIVE *ABS*+0x400')
        result = self.check(assembly=assembly, symbols=symbols, relocations=relocations)
        self.assertEqual(result['calls'][0]['target'], 'trunc')
        self.assertEqual(result['local_math_helpers'][0]['bytes'], 16)
        for replacement in ['*ABS*+0x120', '*ABS*+0x404', 'trunc']:
            with self.assertRaisesRegex(ValueError, 'relative indirect call'):
                self.check(assembly=assembly, symbols=symbols,
                           relocations=relocations.replace('*ABS*+0x400', replacement))
        with self.assertRaisesRegex(ValueError, 'relative indirect call'):
            self.check(assembly=assembly, symbols=symbols.replace('t trunc', 't socket'),
                       relocations=relocations)
        for instruction in ['syscall', 'callq 0x120 <bml_runtime::net::send_msg>',
                            'callq *%rax', 'jmpq *%rax', 'int3']:
            with self.subTest(instruction=instruction), self.assertRaisesRegex(ValueError, 'trunc helper'):
                self.check(assembly=assembly.replace('400: movq %xmm0, %rax', '400: '+instruction),
                           symbols=symbols, relocations=relocations)
        with self.assertRaisesRegex(ValueError, 'Branch escapes local trunc'):
            self.check(assembly=assembly.replace('ja 0x40c', 'ja 0x120'),
                       symbols=symbols, relocations=relocations)


if __name__ == '__main__':
    unittest.main()
