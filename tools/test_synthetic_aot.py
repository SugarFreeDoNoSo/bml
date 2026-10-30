"""Boundaries, operation contracts and analytical scaling for synthetic AOT."""
import unittest
from synthetic_aot import fixture, reference, estimate, planned_sizes

class SyntheticAotTests(unittest.TestCase):
    def test_reproducible_signed_f32_fixture(self):
        a, weights = fixture(10, 17)
        b, other = fixture(10, 17)
        self.assertEqual(a, b)
        self.assertEqual(weights, other)
        self.assertNotEqual(a, fixture(10, 18)[0])
        self.assertTrue(any(v < 0 for v in weights))
        self.assertTrue(any(v > 0 for v in weights))
        self.assertTrue(a.startswith(b'GGUF\x03\x00\x00\x00'))

    def test_independent_gguf_roundtrip_and_rejection(self):
        from gguf_fixture_reader import read_fixture
        data, weights=fixture(10,17)
        self.assertEqual(read_fixture(data)['values'],weights)
        for bad in [b'RAW!'+data[4:],data[:-1],data+b'\x00']:
            with self.assertRaises(ValueError): read_fixture(bad)

    def test_huge_materialization_rejected(self):
        for n in [0, 129, 40960]:
            with self.assertRaises(ValueError):
                fixture(n, 17)

    def test_independent_signed_operations(self):
        self.assertEqual(reference([1,-2,3,4],[0.5,-1],2,1),[2.5,-2.5])
        self.assertEqual(reference([1,-2,3,4],[-1,2,0.5,-3],2,2),[-2,8,-1,-6])

    def test_storage_budget_doubling_plan(self):
        e = estimate(20480)
        self.assertEqual(e['f32_A_bytes'],1677721600)
        self.assertEqual(planned_sizes(),[10*2**i for i in range(12)])
        self.assertGreater(4*40960**2,3_000_000_000)
        self.assertEqual(e['matmul_flops_conventional'],17179869184000)
        self.assertEqual(e['matvec_flops_conventional'],838860800)
        self.assertFalse(e['assumptions']['hardware_calibrated'])
        self.assertEqual(e['kind'],'analytical_estimate_not_measurement')
