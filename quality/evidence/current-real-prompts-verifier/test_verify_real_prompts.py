"""Small saved-byte fixtures only; never Cargo, a model or the measured binary."""
import copy
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest

import verify_real_prompts as v


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class NumericalBytes(unittest.TestCase):
    def test_argmax_matches_total_cmp_signed_zero_and_last_ties(self):
        self.assertEqual(v.last_argmax([1.0, 2.0, 2.0]), 2)
        self.assertEqual(v.last_argmax([0.0, -0.0]), 0)
        self.assertEqual(v.last_argmax([-0.0, 0.0]), 1)

    def test_scaled_rms_is_finite_for_large_finite_differences(self):
        stats = v.error_statistics([1e200, 0.0], [0.0, 0.0])
        self.assertEqual(stats["max_absolute_error"], 1e200)
        self.assertEqual(stats["mean_absolute_error"], 5e199)
        self.assertEqual(stats["rms_error"], 1e200 * math.sqrt(0.5))

    def test_vector_bytes_are_little_endian_exact_length_and_finite(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "vector"
            path.write_bytes(struct.pack("<3d", 1.25, -2.5, 3.0))
            values, receipt = v.read_vector(path, 3)
            self.assertEqual(values, (1.25, -2.5, 3.0))
            self.assertEqual(receipt["argmax"], 2)
            for data in [path.read_bytes()[:-1], struct.pack("<3d", 0.0, math.inf, 1.0),
                         struct.pack("<3d", 0.0, math.nan, 1.0)]:
                path.write_bytes(data)
                with self.assertRaises(ValueError):
                    v.read_vector(path, 3)


class CompletedCases(unittest.TestCase):
    def test_json_boolean_cannot_impersonate_a_numeric_protocol_field(self):
        self.assertFalse(v.same_json({"max_nodes": False}, {"max_nodes": 0}))
        self.assertFalse(v.same_json([True, 15043], [1, 15043]))

    def test_both_cases_are_mandatory_and_native_only_or_pending_never_passes(self):
        report = {"status": "all_cases_completed_unverified", "source_revision": v.REVISION,
                  "binary_sha256": v.BINARY_SHA,
                  "cases": [{"prompt": prompt, "status": "completed_unverified", "exit_code": 0}
                            for prompt in v.PROMPTS]}
        self.assertEqual(set(v.full_cases(report)), {"Hello", "Hi"})
        for change in ["pending", "missing", "duplicate", "native-only", "exit"]:
            changed = copy.deepcopy(report)
            if change == "pending":
                changed["status"] = "running_serial_full_case"
            elif change == "missing":
                changed["cases"].pop()
            elif change == "duplicate":
                changed["cases"][1]["prompt"] = "Hello"
            elif change == "native-only":
                changed["cases"][0]["status"] = "native_only_preflight_pass"
            else:
                changed["cases"][1]["exit_code"] = 1
            with self.subTest(change=change), self.assertRaises(ValueError):
                v.full_cases(changed)


class PositionFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.folder = self.root / "full-hello"
        self.folder.mkdir()
        self.binary = "/original/capture/real-prompts-measured"
        self.archived = {"mode": "native_only_profile_preflight", "source_revision_declared": v.REVISION,
                         "binary_sha256": v.BINARY_SHA, "selected_prompt": "Hello", "rust_encoded_input_ids": [1, 15043],
                         "predeclared_cases": [{"prompt": p, "input_ids": ids} for p, ids in v.PROMPTS.items()],
                         "context": 2, "cache_capacity_per_backend": 2, "symbolic_limits": {"max_nodes": 0}}
        save(self.folder / "protocol.json", {**self.archived, "mode": v.MODE})
        self.protocol_sha = v.sha(self.folder / "protocol.json")
        self.positions = []
        for index, token in enumerate(v.PROMPTS["Hello"]):
            native = [0.0] * v.VOCAB
            symbolic = [0.0] * v.VOCAB
            native[0] = native[-1] = 1.0
            symbolic[0] = symbolic[-1] = 1.25  # Deliberately large error; no global tolerance.
            for kind, values in [("native", native), ("symbolic", symbolic)]:
                (self.folder / f"{kind}-{index}.f64le").write_bytes(struct.pack(f"<{v.VOCAB}d", *values))
            self.positions.append({"status": "full_position_completed_diagnostic", "prompt": "Hello",
                "input_token": token, "position": index, "logits": v.VOCAB,
                "native_vector": f"native-{index}.f64le", "symbolic_vector": f"symbolic-{index}.f64le",
                "native_argmax": v.VOCAB - 1, "symbolic_argmax": v.VOCAB - 1,
                "native_cache_position": index + 1, "symbolic_cache_position": index + 1,
                "symbolic_elapsed_seconds": 1.0, "protocol_sha256": self.protocol_sha,
                **v.error_statistics(native, symbolic)})
        self.case = {"prompt": "Hello", "input_ids": [1, 15043],
                     "command": [self.binary, v.TEST, "--ignored", "--exact", "--nocapture"],
                     "output_directory": "/original/capture/full-hello", "log": "/original/capture/full-hello.log",
                     "started_at_utc": "2026-11-05T08:00:00Z", "ended_at_utc": "2026-11-05T08:01:00Z"}
        (self.root / "full-hello.exit").write_text("0\n")
        self.persist()

    def persist(self):
        for index, report in enumerate(self.positions):
            save(self.folder / f"position-{index}.json", report)
        save(self.folder / "summary.json", {"status": "full_completed_diagnostic", "prompt": "Hello",
            "input_ids": [1, 15043], "positions": self.positions, "executed_operations": v.FAMILIES,
            "protocol_sha256": self.protocol_sha})
        log = "\n".join(json.dumps(p) for p in self.positions)
        log += f"\ntest {v.TEST} ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored;\n"
        (self.root / "full-hello.log").write_text(log)
        self.case.update(summary_sha256=v.sha(self.folder / "summary.json"),
                         log_sha256=v.sha(self.root / "full-hello.log"))

    def verify(self):
        return v.verify_case(self.root, "Hello", self.case, self.archived, self.binary)

    def test_four_complete_vectors_and_large_diagnostic_error_are_accepted(self):
        result = self.verify()
        self.assertEqual(len(result["vectors"]), 4)
        self.assertEqual(result["positions"][0]["recomputed"]["max_absolute_error"], 0.25)

    def test_self_consistent_json_and_log_cannot_hide_wrong_raw_statistics(self):
        self.positions[0]["rms_error"] = math.nextafter(self.positions[0]["rms_error"], math.inf)
        self.persist()  # Rehash all envelopes, leaving only the raw-vector oracle decisive.
        with self.assertRaisesRegex(ValueError, "raw vectors"):
            self.verify()

    def test_self_consistent_metadata_cannot_hide_wrong_cache_position(self):
        self.positions[1]["symbolic_cache_position"] = 1
        self.persist()
        with self.assertRaisesRegex(ValueError, "position/tokens/cache"):
            self.verify()

    def test_native_only_protocol_cannot_count_as_full_execution(self):
        save(self.folder / "protocol.json", self.archived)
        with self.assertRaisesRegex(ValueError, "full protocol"):
            self.verify()

    def test_partial_failure_or_missing_vector_cannot_count_as_completed(self):
        failure = self.folder / "failure-1.json"
        failure.write_text("{}")
        with self.assertRaisesRegex(ValueError, "incomplete or unexpected"):
            self.verify()
        failure.unlink()
        (self.folder / "symbolic-1.f64le").unlink()
        with self.assertRaisesRegex(ValueError, "incomplete or unexpected"):
            self.verify()

    def test_other_prompt_or_historical_driver_command_is_rejected(self):
        self.case["command"][1] = "real_model_two_position_symbolic_logit_diagnostic"
        with self.assertRaisesRegex(ValueError, "command or predeclared"):
            self.verify()


if __name__ == "__main__":
    unittest.main()
