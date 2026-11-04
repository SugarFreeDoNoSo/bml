import copy
import os
import unittest
from unittest.mock import patch

from native_kernel_campaign import validate_benchmark, validate_host


def valid_report():
    sample = {"ns": 1, "allocation_calls": 0, "requested_bytes": 0,
              "peak_additional_live_bytes": 0}
    return {"iterations": 100, "pairs": 512, "repetitions": 7, "neon_available": True,
            "workloads": [{"name": name, "backends": [
                {"backend": b, "samples": [copy.deepcopy(sample) for _ in range(7)]}
                for b in ["Scalar", "Neon"]]}
                for name in ["equal-long", "late-difference", "short-mixed"]]}


class NativeCampaignContracts(unittest.TestCase):
    def test_complete_native_measurements_are_accepted(self):
        validate_benchmark(valid_report(), "Neon", 100)

    def test_availability_flag_cannot_substitute_for_measured_backend(self):
        report = valid_report()
        report["workloads"][0]["backends"].pop()
        with self.assertRaises(ValueError):
            validate_benchmark(report, "Neon", 100)

    def test_unavailable_backend_or_truncated_repetitions_are_rejected(self):
        report = valid_report()
        report["neon_available"] = False
        with self.assertRaises(ValueError):
            validate_benchmark(report, "Neon", 100)
        report = valid_report()
        report["workloads"][1]["backends"][1]["samples"].pop()
        with self.assertRaises(ValueError):
            validate_benchmark(report, "Neon", 100)

    def test_nonpositive_or_nonfinite_measurements_are_rejected(self):
        for value in [0, -1, float("nan")]:
            report = valid_report()
            report["workloads"][0]["backends"][0]["samples"][0]["ns"] = value
            with self.assertRaises(ValueError):
                validate_benchmark(report, "Neon", 100)

    def test_different_workload_sizes_or_duplicate_workloads_are_rejected(self):
        report = valid_report()
        report["iterations"] = 1
        with self.assertRaises(ValueError):
            validate_benchmark(report, "Neon", 100)
        report = valid_report()
        report["workloads"][1]["name"] = "equal-long"
        with self.assertRaises(ValueError):
            validate_benchmark(report, "Neon", 100)

    def test_wrong_host_and_cross_runner_are_not_native_evidence(self):
        with patch.dict(os.environ, {}, clear=True), patch("platform.system", return_value="Linux"), \
                patch("platform.machine", return_value="x86_64"):
            with self.assertRaises(ValueError):
                validate_host("aarch64")
            with patch.dict(os.environ, {"CARGO_TARGET_AARCH64_UNKNOWN_LINUX_MUSL_RUNNER": "qemu-aarch64"}):
                with self.assertRaises(ValueError):
                    validate_host("x86_64")


if __name__ == "__main__":
    unittest.main()
