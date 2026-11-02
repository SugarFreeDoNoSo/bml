"""Reject incomparable work and malformed metrics before reporting performance."""

from copy import deepcopy
import contextlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import benchmark_inference as benchmark


def outputs():
    artifact = {
        "prompt_tokens": [1, 7], "tokens": [8] * 16, "eos_id": 2,
        "text": "a" * 16, "bytes": [97] * 16,
        "prefill_ms": 2.0, "decode_ms": 4.0,
    }
    reference = deepcopy(artifact)
    reference["timing"] = {key: reference.pop(key) for key in ("prefill_ms", "decode_ms")}
    reference["reference"] = {"threads": 1, "kv_dtype": "f32", "flash_attention": False, "weight_repacking": False}
    return reference, artifact


class BenchmarkContracts(unittest.TestCase):
    def run_protocol(self, directory, diverge=False, change_binary=False):
        directory = Path(directory)
        model, artifact, reference = [directory / name for name in ("model", "artifact", "reference")]
        for path in (model, artifact, reference):
            path.write_bytes(path.name.encode())
        reference.with_suffix(".build.json").write_text("{}")
        build = directory / "build.json"
        build.write_text(json.dumps({
            "sha256": benchmark.validation.sha256(artifact),
            "source_sha256": benchmark.validation.sha256(model),
            "target": "x86_64-unknown-linux-gnu", "rustc_version": "recorded rustc", "elapsed_ms": 200.0,
        }))
        calls = []

        def measure(command, prefix, timeout, time_program):
            self.assertNotIn("--logits", command)
            calls.append(command)
            engine = Path(command[0]).name
            round_index = int(prefix.name.split("-")[3])
            value = 1000.0 if round_index == 0 else float(round_index)
            ref, actual = outputs()
            output = ref if engine == "reference" else actual
            timing = output["timing"] if engine == "reference" else output
            timing["prefill_ms"] = timing["decode_ms"] = value
            if diverge and engine == "artifact":
                output["text"] = "different"
            if change_binary:
                artifact.write_bytes(b"changed during measurement")
            return output, {
                "command": command, "measurement_command": command,
                "wall_ms": value, "monotonic_wall_ms": value + 0.1, "max_rss_kib": 1000,
            }

        with patch.object(benchmark.validation, "revision", return_value=benchmark.validation.REFERENCE_COMMIT), \
                patch.object(benchmark.validation, "working_tree_dirty", return_value=False), \
                patch.object(benchmark.validation, "reference_manifest", return_value={"libraries_sha256": {}}), \
                patch.object(benchmark, "measure", side_effect=measure), \
                contextlib.redirect_stdout(io.StringIO()):
            code = benchmark.main([
                "--model", str(model), "--artifact", str(artifact),
                "--artifact-build-report", str(build), "--reference", str(reference),
                "--llama-source", str(directory), "--output", str(directory / "results"),
            ])
        report = json.loads((directory / "results/report.json").read_text())
        return code, report, calls

    def test_equal_outputs_validate_the_actual_timing_layouts(self):
        reference, artifact = outputs()
        self.assertEqual(benchmark.compare_outputs(reference, artifact), {
            "reference": {"prefill_ms": 2.0, "decode_ms": 4.0},
            "artifact": {"prefill_ms": 2.0, "decode_ms": 4.0},
        })

    def test_each_output_component_must_match_independently(self):
        changes = {
            "prompt_tokens": [1, 9], "tokens": [9] * 16, "eos_id": 3,
            "text": "different text", "bytes": [98] * 16,
        }
        for field, changed in changes.items():
            with self.subTest(field=field):
                reference, artifact = outputs()
                artifact[field] = changed
                with self.assertRaisesRegex(ValueError, "differ"):
                    benchmark.compare_outputs(reference, artifact)

    def test_elapsed_times_reject_missing_nonfinite_boolean_and_nonpositive_values(self):
        for engine in ("artifact", "reference"):
            for field in ("prefill_ms", "decode_ms"):
                for invalid in (None, True, "1.0", 0.0, -1.0, math.nan, math.inf, -math.inf):
                    with self.subTest(engine=engine, field=field, invalid=invalid):
                        reference, artifact = outputs()
                        times = reference["timing"] if engine == "reference" else artifact
                        times[field] = invalid
                        with self.assertRaisesRegex(ValueError, "finite positive"):
                            benchmark.compare_outputs(reference, artifact)

    def test_early_eos_is_valid_but_incomplete_or_post_eos_work_is_rejected(self):
        reference, artifact = outputs()
        for value in (reference, artifact):
            value["tokens"] = [8, 2]
        benchmark.compare_outputs(reference, artifact)
        for tokens in ([8], [2, 8], [], [8] * 17):
            with self.subTest(tokens=tokens):
                reference, artifact = outputs()
                for value in (reference, artifact):
                    value["tokens"] = tokens
                with self.assertRaises(ValueError):
                    benchmark.compare_outputs(reference, artifact)

    def test_equal_but_invalid_outputs_cannot_pass(self):
        for field, value in (("prompt_tokens", []), ("prompt_tokens", [True]), ("bytes", [256]), ("bytes", [False]), ("text", None), ("eos_id", True), ("logits", [[1.0]])):
            with self.subTest(field=field, value=value):
                reference, artifact = outputs()
                reference[field] = artifact[field] = value
                with self.assertRaises(ValueError):
                    benchmark.compare_outputs(reference, artifact)

    def test_reference_precision_and_threads_are_part_of_the_protocol(self):
        for field, value in (("threads", 2), ("threads", True), ("kv_dtype", "f16"), ("flash_attention", True), ("flash_attention", 0), ("weight_repacking", True)):
            with self.subTest(field=field):
                reference, artifact = outputs()
                reference["reference"][field] = value
                with self.assertRaisesRegex(ValueError, "reference"):
                    benchmark.compare_outputs(reference, artifact)

    def test_statistics_use_the_declared_population_and_reject_nan(self):
        result = benchmark.summarize([2.0, 4.0, 6.0])
        self.assertEqual({key: result[key] for key in ("count", "mean", "min", "max")}, {"count": 3, "mean": 4.0, "min": 2.0, "max": 6.0})
        self.assertAlmostEqual(result["stddev_population"], math.sqrt(8.0 / 3.0))
        for invalid in ([], [math.nan], [True]):
            with self.assertRaises(ValueError):
                benchmark.summarize(invalid)

    def test_build_provenance_requires_matching_model_artifact_and_recorded_toolchain(self):
        valid = {"sha256": "artifact", "source_sha256": "model", "target": "x86_64-unknown-linux-gnu", "rustc_version": "rustc pinned", "elapsed_ms": 123.0}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "build.json"
            path.write_text(json.dumps(valid))
            self.assertEqual(benchmark.artifact_build_report(path, "artifact", "model"), valid)
            for field, value in (("sha256", "stale"), ("source_sha256", "other"), ("target", "other"), ("rustc_version", ""), ("elapsed_ms", 0)):
                with self.subTest(field=field):
                    path.write_text(json.dumps({**valid, field: value}))
                    with self.assertRaises(ValueError):
                        benchmark.artifact_build_report(path, "artifact", "model")

    def test_missing_model_is_not_a_successful_measurement(self):
        with patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(benchmark.main([]), 77)
        self.assertEqual(json.loads(output.getvalue())["status"], "not_run")

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            report = path / "report.json"
            report.write_text("previous measurement")
            with contextlib.redirect_stdout(io.StringIO()):
                code = benchmark.main([
                    "--model", "unused.gguf", "--artifact", "unused.bin",
                    "--artifact-build-report", "unused.json", "--output", str(path),
                ])
            self.assertEqual(code, 1)
            self.assertEqual(report.read_text(), "previous measurement")

    def test_protocol_launches_each_sample_and_excludes_warmups_from_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            code, report, calls = self.run_protocol(directory)
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(len(calls), 16)
        self.assertEqual(len(report["samples"]), 8)
        self.assertEqual(sum(sample["warmup"] for sample in report["samples"]), 2)
        self.assertEqual(len(report["summary"]), 4)
        for summary in report["summary"]:
            self.assertEqual(summary["metrics"]["prefill_ms"]["count"], 3)
            self.assertEqual(summary["metrics"]["prefill_ms"]["mean"], 2.0)
            self.assertEqual(summary["metrics"]["prefill_ms"]["max"], 3.0)
        self.assertEqual(report["samples"][0]["engine_order"], ["artifact", "reference"])
        self.assertEqual(report["samples"][1]["engine_order"], ["reference", "artifact"])

    def test_divergence_makes_the_report_fail_before_accepting_any_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            code, report, calls = self.run_protocol(directory, diverge=True)
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "failed")
        self.assertIn("text differ", report["error"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(report["samples"], [])
        self.assertNotIn("summary", report)

    def test_changed_artifact_cannot_produce_a_passed_report(self):
        with tempfile.TemporaryDirectory() as directory:
            code, report, _ = self.run_protocol(directory, change_binary=True)
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "failed")
        self.assertTrue(report["provenance_errors"])

    @unittest.skipUnless(hasattr(os, "wait4"), "individual child resource accounting requires wait4")
    def test_peak_rss_belongs_to_each_process_not_earlier_children(self):
        with tempfile.TemporaryDirectory() as directory:
            measurements = []
            for index, size in enumerate((32 * 1024 * 1024, 1024 * 1024)):
                command = [sys.executable, "-c", f"import json; data=bytearray({size}); print(json.dumps({{'bytes':len(data)}}))"]
                output, measured = benchmark.measure(command, Path(directory) / f"child-{index}", 10, None)
                self.assertEqual(output, {"bytes": size})
                self.assertGreater(measured["wall_ms"], 0)
                self.assertGreater(measured["max_rss_kib"], 0)
                measurements.append(measured)
            self.assertGreater(measurements[0]["max_rss_kib"], measurements[1]["max_rss_kib"])


if __name__ == "__main__":
    unittest.main()
