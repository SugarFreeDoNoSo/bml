"""The binary linkage scan must expose networking imports and Rust routines."""
import unittest

from measure_hot_loop import network_symbols


class NetworkSymbols(unittest.TestCase):
    def test_dynamic_versions_and_rust_network_paths_are_detected(self):
        lines = [
            '                 U connect@GLIBC_2.2.5',
            '                 U socket',
            '000123 T bml_runtime::net::NodeClient::health_check',
            '000456 t std::sys::net::connection::socket::TcpStream::connect',
            '                 U recvmsg@GLIBC_2.2.5',
        ]
        self.assertEqual(network_symbols('\n'.join(lines)),
                         [line.strip() for line in lines])

    def test_unrelated_names_and_math_imports_are_not_network_evidence(self):
        self.assertEqual(network_symbols('\n'.join([
            '                 U exp2@GLIBC_2.29',
            '000123 T bml_runtime::hot_loop::dispatch_ops',
            '000456 t allocator::rebind',
            '000789 t disconnected_counter',
        ])), [])


if __name__ == '__main__':
    unittest.main()
