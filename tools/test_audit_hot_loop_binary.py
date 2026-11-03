"""Binary audit must reject network targets and unresolved control transfers."""
import struct
import unittest
from unittest.mock import patch

from audit_hot_loop_binary import HOT, audit, mapped_bytes


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

    def check(self, assembly=None, relocations=None, offsets=None):
        with patch('audit_hot_loop_binary.mapped_bytes', return_value=struct.pack('<14i', *(offsets or [-256]*14))):
            return audit(b'', assembly or self.assembly, self.symbols,
                         relocations or self.relocations, 14)

    def test_got_target_comes_from_relocation_not_disassembler_annotation(self):
        result = self.check()
        self.assertEqual(result['calls'][0]['target'], 'log2')
        self.assertEqual(result['jump_tables'][0]['targets'], [0x100]*14)

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
            self.check(offsets=[-256]*13 + [-224])  # one target is RPC, at 0x120
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


if __name__ == '__main__':
    unittest.main()
