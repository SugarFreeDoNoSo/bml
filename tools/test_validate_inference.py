"""Fail-closed numerical and execution contracts for the optional model gate."""

from copy import deepcopy
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import validate_inference as validation


def output():
    return {
        "prompt_tokens": [1, 0], "tokens": [2],
        "logits": [[0.0, -0.25, 1.5], [1.4, 0.3, 0.2]],
        "text": "A", "bytes": [65],
    }


class InferenceComparisonTests(unittest.TestCase):
    def test_equivalent_outputs_meet_the_complete_greedy_contract(self):
        expected = output()
        result = validation.compare(expected, deepcopy(expected), predict=1, eos_id=2, context=3)
        self.assertTrue(result["passed"])
        self.assertEqual(result["reference_shape"], [2, 3])
        self.assertEqual(result["violations"], 0)
        self.assertEqual(result["maximum_absolute_error"], 0.0)

    def test_error_budget_is_enforced_at_zero_and_at_nonzero_values(self):
        expected = output()
        actual = deepcopy(expected)
        actual["logits"][0][0] = validation.ABSOLUTE_TOLERANCE
        self.assertTrue(validation.compare(expected, actual)["passed"])
        actual["logits"][0][0] = math.nextafter(validation.ABSOLUTE_TOLERANCE, math.inf)
        result = validation.compare(expected, actual)
        self.assertFalse(result["passed"])
        self.assertEqual(result["violations"], 1)
        self.assertEqual(result["first_violation"]["step"], 0)
        self.assertEqual(result["first_violation"]["token"], 0)
        actual = deepcopy(expected)
        actual["logits"][1][0] += 0.01
        self.assertFalse(validation.compare(expected, actual)["passed"])

    def test_every_logits_row_and_vocabulary_column_is_checked(self):
        expected = output()
        for step in range(2):
            for token in range(3):
                with self.subTest(step=step, token=token):
                    actual = deepcopy(expected)
                    actual["logits"][step][token] += 0.01
                    self.assertFalse(validation.compare(expected, actual)["passed"])

    def test_text_and_bytes_are_independent_required_comparisons(self):
        expected = output()
        for name, value in [("text", "WRONG"), ("bytes", [66])]:
            with self.subTest(name=name):
                actual = deepcopy(expected)
                actual[name] = value
                self.assertFalse(validation.compare(expected, actual)["passed"])
        for name, value in [("text", None), ("bytes", None), ("bytes", [256]), ("bytes", [True])]:
            with self.subTest(name=name, value=value):
                actual = deepcopy(expected)
                actual[name] = value
                with self.assertRaises(ValueError):
                    validation.compare(expected, actual)

    def test_nonfinite_and_nonnumeric_logits_are_rejected_on_either_side(self):
        for value in [math.nan, math.inf, -math.inf, "0.0", None, True]:
            for side in ["reference", "actual"]:
                with self.subTest(value=value, side=side):
                    expected, actual = output(), output()
                    target = expected if side == "reference" else actual
                    target["logits"][1][2] = value
                    with self.assertRaises(ValueError):
                        validation.compare(expected, actual)

    def test_missing_rows_empty_rows_and_changing_vocabulary_are_rejected(self):
        for bad_rows in [[], [[]], [[1.0]], [[1.0], [1.0, 2.0]]]:
            with self.subTest(rows=bad_rows):
                actual = output()
                actual["logits"] = bad_rows
                with self.assertRaises(ValueError):
                    validation.compare(output(), actual)
        actual = output()
        actual["logits"] = [row + [-3.0] for row in actual["logits"]]
        self.assertFalse(validation.compare(output(), actual)["passed"])

    def test_prompt_ids_must_be_nonempty_nonnegative_integers_and_match(self):
        for prompt in [[], [True], [-1], ["1"], None]:
            actual = output()
            actual["prompt_tokens"] = prompt
            with self.subTest(prompt=prompt), self.assertRaises(ValueError):
                validation.compare(output(), actual)
        actual = output()
        actual["prompt_tokens"] = [0, 1]
        self.assertFalse(validation.compare(output(), actual)["passed"])

    def test_greedy_selection_and_lowest_id_ties_are_checked(self):
        for logits, tokens in [([[0.0, 2.0, 1.5], [1.4, 0.3, 0.2]], [2]),
                               ([[2.0, 2.0, 1.5], [1.4, 0.3, 0.2]], [1])]:
            actual = output()
            actual["logits"], actual["tokens"] = logits, tokens
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                validation.compare(output(), actual)

    def test_generation_cannot_stop_early_without_eos_or_continue_after_it(self):
        no_generation = output()
        no_generation.update(tokens=[], logits=[no_generation["logits"][0]], text="", bytes=[])
        with self.assertRaisesRegex(ValueError, "without EOS"):
            validation.compare(no_generation, no_generation, predict=4, eos_id=2)
        with self.assertRaisesRegex(ValueError, "without EOS"):
            validation.compare(output(), output(), predict=4, eos_id=0)
        self.assertTrue(validation.compare(output(), output(), predict=4, eos_id=2)["passed"])
        continued = output()
        continued["tokens"].append(0)
        continued["logits"].append([0.5, 1.0, 0.0])
        with self.assertRaisesRegex(ValueError, "continued after EOS"):
            validation.compare(continued, continued, predict=2, eos_id=2)

    def test_zero_prediction_and_context_limits_are_enforced(self):
        with self.assertRaisesRegex(ValueError, "more tokens"):
            validation.compare(output(), output(), predict=0, eos_id=2)
        with self.assertRaisesRegex(ValueError, "context"):
            validation.compare(output(), output(), predict=1, eos_id=2, context=2)


class InferenceGateExecutionTests(unittest.TestCase):
    def test_a_binary_or_model_replaced_during_validation_invalidates_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model-or-binary"
            path.write_bytes(b"original")
            snapshot = {path: validation.sha256(path)}
            self.assertEqual(validation.changed_files(snapshot), [])
            path.write_bytes(b"replacement")
            self.assertIn("changed during", validation.changed_files(snapshot)[0])
            path.unlink()
            self.assertIn("became unavailable", validation.changed_files(snapshot)[0])

    def test_no_model_is_not_a_successful_validation(self):
        stream = io.StringIO()
        with patch.dict(os.environ, {}, clear=True), patch("sys.stdout", stream):
            self.assertEqual(validation.main([]), 77)
        self.assertEqual(json.loads(stream.getvalue())["status"], "not_run")

    def test_process_failure_and_invalid_json_never_count_as_results(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "case"
            with self.assertRaisesRegex(RuntimeError, "exit status 7"):
                validation.invoke([sys.executable, "-c", "raise SystemExit(7)"], prefix, 3)
            self.assertTrue(prefix.with_suffix(".stderr.log").is_file())
            for literal in ['{"logits": NaN}', '[]', 'not JSON']:
                with self.subTest(literal=literal), self.assertRaises(ValueError):
                    validation.invoke([sys.executable, "-c", "print(" + repr(literal) + ")"], prefix, 3)

    def test_timeout_is_a_failed_run(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(validation.subprocess, "run", side_effect=subprocess.TimeoutExpired("oracle", 1)):
                with self.assertRaises(subprocess.TimeoutExpired):
                    validation.invoke(["oracle"], Path(directory) / "case", 1)

    def test_build_manifest_must_match_source_binary_and_libraries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "llama.cpp"
            binary = root / "llama-reference"
            binary.write_bytes(b"binary")
            libraries = {"build/src/libllama.a", "build/ggml/src/libggml.a",
                         "build/ggml/src/libggml-cpu.a", "build/ggml/src/libggml-base.a"}
            for name in libraries:
                path = source / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(name.encode())
            manifest = {
                "reference_revision": validation.REFERENCE_COMMIT,
                "binary_sha256": validation.sha256(binary),
                "source_sha256": validation.sha256(validation.ROOT / "tools/llama_reference.cpp"),
                "libraries_sha256": {name: validation.sha256(source / name) for name in libraries},
            }
            with self.assertRaisesRegex(ValueError, "missing"):
                validation.reference_manifest(source, binary)
            validation.write_json(binary.with_suffix(".build.json"), manifest)
            self.assertEqual(validation.reference_manifest(source, binary), manifest)
            for field in ["reference_revision", "binary_sha256", "source_sha256"]:
                invalid = deepcopy(manifest)
                invalid[field] = "wrong"
                validation.write_json(binary.with_suffix(".build.json"), invalid)
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "stale"):
                    validation.reference_manifest(source, binary)
            validation.write_json(binary.with_suffix(".build.json"), manifest)
            (source / "build/src/libllama.a").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "libraries changed"):
                validation.reference_manifest(source, binary)


if __name__ == "__main__":
    unittest.main()
