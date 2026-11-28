"""Fixtures never execute fake ELF bytes; ordinary tests need no model or Docker."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import validate_fragment_elf as validator


SOURCE = "a" * 64


def fake_elf(machine=62):
    header = bytearray(64)
    header[:7] = b"\x7fELF\x02\x01\x01"
    struct.pack_into("<HHI", header, 16, 3, machine, 1)
    struct.pack_into("<H", header, 52, 64)
    return bytes(header)


def bundle(root):
    (root / "run").mkdir()
    records = []
    for name in ["coordinator", "fragment_0", "fragment_1"]:
        path = root / "run" / name
        path.write_bytes(fake_elf() + name.encode())
        path.chmod(0o700)
        records.append({"file_name": name, "sha256": validator.sha256(path), "bytes": path.stat().st_size})
    fragments = []
    for index, record in enumerate(records[1:]):
        fragments.append({**record, "weight_bytes": 16,
                          "descriptor": {"version": 1, "source_sha256": SOURCE,
                                         "tensor_name": "output.weight", "row_start": index * 2,
                                         "row_end": (index + 1) * 2, "total_rows": 4, "cols": 2,
                                         "weights_sha256": "b" * 64}})
    report = {"format": "bml-native-head-fragments-v1", "target": "x86_64-unknown-linux-gnu",
              "cpu": "x86-64", "source_sha256": SOURCE, "normalized_source_sha256": SOURCE,
              "head_rows": 4, "head_cols": 2, "workers": 2, "fragments": fragments,
              "coordinator": records[0]}
    save_report(root, report)
    return report


def save_report(root, report):
    (root / "build-report.json").write_text(json.dumps(report))
    return validator.sha256(root / "build-report.json")


def outputs(name="case"):
    case = {"name": name, "tokens": [1, 3], "predict": 1, "context": 4}
    first = [0.5, -0.0, 1.0, -0.25]
    next_row = [0.25, 1.0, 0.0, -0.5]
    reference = {"prompt_tokens": [1, 3], "tokens": [2], "eos_id": 2,
                 "bytes": [], "text": "", "context": 4, "logits": [first, next_row]}
    all_bits = [[0, 0, 0, 0], [validator.float_bits(x) for x in first],
                [validator.float_bits(x) for x in next_row]]
    mono = {"mode": "monolithic", "run": deepcopy(reference), "all_logits_bits": all_bits,
            "forwarded_token_ids": [1, 3, 2], "source_sha256": SOURCE}
    fragmented = deepcopy(mono)
    fragmented["mode"] = "fragments"
    return case, reference, mono, fragmented


class IdentityTests(unittest.TestCase):
    def test_exact_amd64_bundle_and_partition(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = bundle(root)
            report, identities = validator.verify_bundle(root, validator.sha256(root / "build-report.json"))
            self.assertEqual(report, expected)
            self.assertEqual(set(identities), {"coordinator", "fragment_0", "fragment_1"})

    def test_bad_hash_isa_symlink_or_size_never_reaches_execution(self):
        for mutation in ["hash", "isa", "symlink", "size"]:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                report = bundle(root)
                target = root / "run/fragment_0"
                if mutation == "hash":
                    report["fragments"][0]["sha256"] = "0" * 64
                elif mutation == "isa":
                    target.write_bytes(fake_elf(machine=183))
                    report["fragments"][0].update(sha256=validator.sha256(target), bytes=64)
                elif mutation == "symlink":
                    target.unlink()
                    target.symlink_to("fragment_1")
                else:
                    report["fragments"][0]["bytes"] += 1
                with patch.object(validator, "invoke") as execute, self.assertRaises(ValueError):
                    validator.verify_bundle(root, save_report(root, report))
                execute.assert_not_called()

    def test_incompatible_manifest_partitions_paths_sources_and_budgets(self):
        mutations = [
            lambda r: r.update(format="unknown"),
            lambda r: r.update(workers=3),
            lambda r: r.update(head_cols=4097),
            lambda r: r["fragments"][0].update(file_name="../fragment_0"),
            lambda r: r["fragments"][1]["descriptor"].update(row_start=1),
            lambda r: r["fragments"][1]["descriptor"].update(row_start=3),
            lambda r: r["fragments"][1]["descriptor"].update(row_end=3),
            lambda r: r["fragments"][0]["descriptor"].update(source_sha256="c" * 64),
            lambda r: r["fragments"][0]["descriptor"].update(version=True),
            lambda r: r["fragments"][1]["descriptor"].update(tensor_name="token_embd.weight"),
            lambda r: r["fragments"][0].update(weight_bytes=15),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                report = bundle(root)
                mutation(report)
                with self.assertRaises(ValueError):
                    validator.verify_bundle(root, save_report(root, report))

    def test_manifest_digest_and_duplicate_json_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle(root)
            with self.assertRaisesRegex(ValueError, "JSON identity"):
                validator.verify_bundle(root, "0" * 64)
            path = root / "duplicate.json"
            path.write_text('{"x":1,"x":2}')
            with self.assertRaisesRegex(ValueError, "duplicate"):
                validator.read_json(path)


class ParityTests(unittest.TestCase):
    def test_full_prefill_decode_bits_and_token_ids_match(self):
        case, reference, mono, fragmented = outputs()
        result = validator.compare_outputs(reference, mono, fragmented, case, 4)
        self.assertEqual(result["logits_compared"], 12)
        self.assertEqual(result["forward_positions"], 3)
        self.assertTrue(result["bit_identical"])

    def test_first_prefill_error_signed_zero_token_and_missing_position_are_caught(self):
        changes = [
            lambda x: x["all_logits_bits"][0].__setitem__(0, 1),
            lambda x: x["all_logits_bits"][1].__setitem__(1, 0),
            lambda x: x["run"].update(tokens=[1]),
            lambda x: x["all_logits_bits"].pop(0),
            lambda x: x["forwarded_token_ids"].reverse(),
            lambda x: x["all_logits_bits"][0].__setitem__(0, 0x7fc00000),
        ]
        for change in changes:
            case, reference, mono, fragmented = outputs()
            change(fragmented)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validator.compare_outputs(reference, mono, fragmented, case, 4)

    def test_canonical_reference_mismatch_is_not_hidden_by_agreeing_coordinators(self):
        case, reference, mono, fragmented = outputs()
        reference["logits"][0][0] = 0.75
        with self.assertRaisesRegex(ValueError, "canonical logits"):
            validator.compare_outputs(reference, mono, fragmented, case, 4)

    def test_case_budgets_and_disjoint_input_modes(self):
        first = outputs()[0]
        second = {**first, "name": "second"}
        self.assertEqual(len(validator.cases_from([first, second])), 2)
        for changed in [{**second, "context": 257}, {**second, "prompt": "Hi"},
                        {**second, "tokens": [True]}, {**second, "name": "../escape"}]:
            with self.subTest(case=changed), self.assertRaises(ValueError):
                validator.cases_from([first, changed])


class ExecutionTests(unittest.TestCase):
    def test_success_and_failure_controls_only_use_owned_staged_copies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = bundle(root)
            case, reference, mono, fragmented = outputs()
            second = {**case, "name": "other"}
            refs = root / "refs.json"
            refs.write_text(json.dumps({"format_version": 1, "source_sha256": SOURCE,
                                       "cases": [{"request": case, "result": reference},
                                                 {"request": second, "result": reference}]}))
            args = argparse.Namespace(bundle=root, build_report_sha256=validator.sha256(root / "build-report.json"),
                                      reference_json=refs, reference_sha256=validator.sha256(refs),
                                      output=root / "results", timeout=10)
            original_hash = validator.sha256(root / "run/fragment_0")
            def fake_invoke(command, cwd, timeout):
                if "--unknown-option" in command:
                    return 2, b"", b"unknown or repeated argument"
                if "257" in command:
                    return 2, b"", b"fragment context must be in 1..256"
                if Path(cwd).name == "missing-worker":
                    self.assertFalse((Path(cwd) / "fragment_0").exists())
                    return 2, b"", b"fragment executable missing or unreadable"
                if Path(cwd).name == "worker-hash":
                    return 2, b"", b"fragment executable SHA-256 mismatch"
                value = mono if "monolithic" in command else fragmented
                return 0, json.dumps(value).encode(), b""
            with patch.object(validator, "invoke", side_effect=fake_invoke) as execute:
                result = validator.validate(args)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(execute.call_count, 8)
            self.assertEqual(len(result["failure_checks"]), 4)
            self.assertFalse(result["distributed_evidence"])
            self.assertFalse(result["filesystem_isolation_verified"])
            self.assertEqual(validator.sha256(root / "run/fragment_0"), original_hash)

    def test_process_error_and_wrong_identity_never_become_success(self):
        for result in [(1, b"", b"worker error"), (0, b"{}", b"unexpected stderr")]:
            with self.subTest(result=result), self.assertRaises(ValueError):
                validator.success_json(result)


if __name__ == "__main__":
    unittest.main()
