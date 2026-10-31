"""Resource planning and code emission checks, independent of NumPy availability."""
import unittest
from synthetic_scale import kernel_source, workspace, MAX_KERNEL_BYTES, MAX_WEIGHT_BYTES
from gguf_fixture_reader import read_fixture
from synthetic_aot import fixture

class SyntheticScaleTests(unittest.TestCase):
    def test_operation_budgets_are_separate(self):
        self.assertLess(workspace(2560,1),MAX_KERNEL_BYTES)
        self.assertGreater(workspace(2560,2560),MAX_KERNEL_BYTES)
        self.assertLess(4*2560**2,MAX_WEIGHT_BYTES)
        self.assertGreater(4*5120**2,MAX_WEIGHT_BYTES)

    def test_kernel_has_external_resource_and_dynamic_input(self):
        src=kernel_source(160)
        self.assertIn('extern const float A[]',src)
        self.assertNotIn('static const float A[',src)
        self.assertIn('scanf',src)
        self.assertIn('CLOCK_MONOTONIC',src)
        self.assertIn('calloc',src)

    def test_independent_reader_corrupt_shape_and_file(self):
        import struct
        blob,_=fixture(10,17)
        info=read_fixture(blob)
        broken=bytearray(blob)
        name_size=8+len('matrix.weight')
        struct.pack_into('<Q',broken,info['descriptor_offset']+name_size+4,0)
        for candidate in [bytes(broken),blob[:-4],b'bad!'+blob[4:]]:
            with self.assertRaises(ValueError): read_fixture(candidate)
