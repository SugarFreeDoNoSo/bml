import contextlib
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import ci_mutations
import quality
from test_merge_mutation_shards import candidate, report


class OrdinaryMutationScope(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.reports = self.root / "target/quality"
        self.diff = b"diff --git a/crates/example/src/lib.rs b/crates/example/src/lib.rs\n"
        self.candidates = [candidate(0)]
        self.statuses = ["CaughtMutant"]
        self.completed = True
        self.baseline = "Success"
        self.exit_code = 0
        self.list_exit = 0
        self.list_json = None
        self.list_exception = None
        self.list_stderr = None
        self.omit_outcomes = False
        self.outcome_candidates = None
        self.commands = []

    def command(self, argv, **kwargs):
        self.commands.append(argv)
        if argv[0] == ci_mutations.sys.executable:
            Path(argv[-1]).write_bytes(self.diff)
            return subprocess.CompletedProcess(argv, 0)
        if "--list" in argv:
            if self.list_exception:
                raise self.list_exception
            return subprocess.CompletedProcess(
                argv, self.list_exit,
                json.dumps(self.candidates) if self.list_json is None else self.list_json,
                self.list_stderr if self.list_stderr is not None
                else "listing error" if self.list_exit else "")
        if not self.omit_outcomes:
            outcomes = self.root / "mutants.out/outcomes.json"
            outcomes.parent.mkdir()
            data = report(self.candidates if self.outcome_candidates is None
                          else self.outcome_candidates, self.statuses)
            if not self.completed:
                data["end_time"] = None
            data["outcomes"][0]["summary"] = self.baseline
            outcomes.write_text(json.dumps(data))
        return subprocess.CompletedProcess(argv, self.exit_code)

    def execute(self, full=False):
        stdout = io.StringIO()
        with patch.object(ci_mutations, "revision", side_effect=lambda root, ref:
                          "1" * 40 if ref == "HEAD" else "2" * 40), \
                patch.object(ci_mutations.subprocess, "check_output",
                             return_value="cargo-mutants 27.1.0\n"), \
                patch.object(ci_mutations.subprocess, "run", side_effect=self.command), \
                patch.object(quality, "REPORTS", self.reports), \
                contextlib.redirect_stdout(stdout):
            ci_mutations.run_scope(self.root, "base-ref", full)
        return stdout.getvalue()

    def assert_na(self, full=False):
        self.candidates = []
        output = self.execute(full)
        self.assertIn("Mutation score: N/A", output)
        self.assertNotIn("100", output)
        evidence = json.loads((self.reports / "ci-mutation-selection.json").read_text())
        self.assertEqual(evidence["status"], "not_applicable")
        self.assertEqual(evidence["candidate_count"], 0)
        self.assertEqual(evidence["baseline_runs"], 0)
        self.assertEqual(evidence["base_revision"], "2" * 40)
        self.assertEqual(evidence["source_revision"], "1" * 40)
        self.assertEqual(evidence["diff_sha256"], hashlib.sha256(self.diff).hexdigest())
        self.assertEqual(evidence["candidates_sha256"], hashlib.sha256(
            (self.reports / "ci-mutation-candidates.json").read_bytes()).hexdigest())
        empty = json.loads((self.reports / "ci-mutation-empty-outcomes.json").read_text())
        self.assertEqual(empty["kind"], "verified_empty_selection")
        self.assertEqual(empty["outcomes"], [])
        self.assertFalse((self.root / "mutants.out").exists())
        self.assertEqual(sum("--list" in c for c in self.commands), 1)
        self.assertFalse(any("--jobs" in c for c in self.commands))

    def test_empty_diff_is_verified_by_listing_and_original_gate_reports_na(self):
        self.diff = b""
        self.assert_na()

    def test_nonempty_test_only_diff_with_zero_candidates_is_na(self):
        self.diff = b"diff --git a/crates/example/tests/contracts.rs b/crates/example/tests/contracts.rs\n"
        self.assert_na()

    def test_successful_cli_diff_filter_zero_result_has_explicit_proof(self):
        self.diff = b""
        self.list_json = ""
        self.list_stderr = "\x1b[32m INFO\x1b[0m Diff file is empty\n"
        self.assert_na()
        selection = json.loads((self.reports / "ci-mutation-selection.json").read_text())
        self.assertEqual(selection["selection_proof"], {
            "kind": "cargo_mutants_27_1_diff_filter", "message": "Diff file is empty"})

    def test_full_scope_does_not_accept_empty_diff_filter_shortcut(self):
        self.list_json = ""
        self.list_stderr = "INFO No mutants to filter\n"
        with self.assertRaisesRegex(ValueError, "missing JSON"):
            self.execute(full=True)

    def test_full_scope_ignores_empty_diff_and_still_executes_candidates(self):
        self.diff = b""
        self.assertIn("100.00%", self.execute(full=True))
        cargo = [c for c in self.commands if c[0] == "cargo"]
        self.assertEqual(len(cargo), 2)
        self.assertTrue(all("--workspace" in c and "--in-diff" not in c for c in cargo))
        self.assertTrue(all("--sharding" not in c for c in cargo))

    def test_nonempty_selection_preserves_diff_and_original_run_arguments(self):
        self.execute()
        cargo = [c for c in self.commands if c[0] == "cargo"]
        self.assertEqual(len(cargo), 2)
        self.assertTrue(all(c[-2:] == ["--in-diff", str(self.reports / "changes.diff")]
                            for c in cargo))
        self.assertIn("--test-workspace", cargo[1])
        self.assertIn("--cargo-arg=--locked", cargo[1])

    def test_listing_error_cannot_become_na(self):
        self.candidates = []
        self.list_exit = 2
        with self.assertRaisesRegex(ValueError, "candidate listing failed"):
            self.execute()
        self.assertFalse((self.reports / "ci-mutation-selection.json").exists())

    def test_malformed_listing_cannot_become_na(self):
        for value in ["", "{}", "null"]:
            with self.subTest(value=value):
                self.list_json = value
                with tempfile.TemporaryDirectory() as directory:
                    self.root = Path(directory)
                    self.reports = self.root / "target/quality"
                    with self.assertRaises(ValueError):
                        self.execute()
                    self.assertFalse((self.reports / "ci-mutation-empty-outcomes.json").exists())

    def test_listing_timeout_is_a_failure(self):
        self.list_exception = subprocess.TimeoutExpired("cargo mutants --list", 30)
        with self.assertRaises(subprocess.TimeoutExpired):
            self.execute()

    def test_zero_message_cannot_hide_a_failed_listing(self):
        self.list_json = ""
        self.list_stderr = "INFO No mutants to filter\n"
        self.list_exit = 2
        with self.assertRaisesRegex(ValueError, "candidate listing failed"):
            self.execute()

    def test_only_complete_info_records_can_prove_empty_selection(self):
        for text in ["ERROR INFO No mutants to filter\n",
                     "INFO No mutants to filter despite an error\n"]:
            with self.subTest(text=text):
                with self.assertRaisesRegex(ValueError, "missing JSON"):
                    ci_mutations.listed_candidates("", text, False)

    def test_successful_exit_with_empty_info_and_additional_error_cannot_be_na(self):
        self.diff = b""
        self.list_json = ""
        self.list_stderr = "INFO Diff file is empty\nERROR fallo\n"
        self.list_exit = 0
        with self.assertRaisesRegex(ValueError, "missing JSON"):
            self.execute()
        self.assertFalse((self.reports / "ci-mutation-empty-outcomes.json").exists())
        self.assertFalse((self.reports / "ci-mutation-selection.json").exists())

    def test_survivor_timeout_failed_baseline_and_incomplete_outcomes_still_fail(self):
        for statuses, baseline, completed in [
            (["MissedMutant"], "Success", True),
            (["Timeout"], "Success", True),
            (["CaughtMutant"], "Failure", True),
            (["CaughtMutant"], "Success", False),
        ]:
            with self.subTest(statuses=statuses, baseline=baseline, completed=completed):
                with tempfile.TemporaryDirectory() as directory:
                    self.root = Path(directory)
                    self.reports = self.root / "target/quality"
                    self.statuses, self.baseline, self.completed = statuses, baseline, completed
                    self.exit_code = 3
                    with self.assertRaises(SystemExit):
                        self.execute()

    def test_nonempty_selection_without_outcomes_never_becomes_na(self):
        self.omit_outcomes = True
        with self.assertRaises(FileNotFoundError):
            self.execute()
        self.assertFalse((self.reports / "ci-mutation-empty-outcomes.json").exists())

    def test_nonempty_selection_cannot_reuse_an_empty_campaign_report(self):
        self.outcome_candidates = []
        with self.assertRaisesRegex(ValueError, "nonempty listed selection"):
            self.execute()

    def test_infrastructure_exit_fails_even_if_report_passes_original_gate(self):
        self.exit_code = 124
        with self.assertRaisesRegex(ValueError, "exited with 124"):
            self.execute()

    def test_old_outcomes_are_not_reused_for_an_empty_selection(self):
        (self.root / "mutants.out").mkdir()
        self.candidates = []
        with self.assertRaisesRegex(ValueError, "stale mutation output"):
            self.execute()
        self.assertEqual(self.commands, [])


@unittest.skipUnless(shutil.which("cargo-mutants"), "requires installed cargo-mutants 27.1.0")
class ActualEmptySelectionParser(unittest.TestCase):
    def test_real_cli_empty_test_only_and_comment_only_diffs_without_builds(self):
        original_run = subprocess.run
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            (root / "src/lib.rs").write_text("// harmless comment\npub fn answer() -> bool { true }\n")
            (root / "tests").mkdir()
            (root / "tests/check.rs").write_text("// test only\n")
            (root / "Cargo.toml").write_text(
                '[package]\nname = "empty-scope-parser"\nversion = "0.1.0"\n'
                'edition = "2021"\n[workspace]\n')
            (root / "Cargo.lock").write_text(
                'version = 4\n[[package]]\nname = "empty-scope-parser"\nversion = "0.1.0"\n')
            diffs = [b"", b"diff --git a/tests/check.rs b/tests/check.rs\n"
                     b"--- /dev/null\n+++ b/tests/check.rs\n@@ -0,0 +1 @@\n+// test only\n",
                     b"diff --git a/src/lib.rs b/src/lib.rs\n--- a/src/lib.rs\n"
                     b"+++ b/src/lib.rs\n@@ -1 +1,2 @@\n+// harmless comment\n"
                     b" pub fn answer() -> bool { true }\n"]
            for index, diff in enumerate(diffs):
                with self.subTest(index=index):
                    reports = root / "target/quality"
                    if reports.exists():
                        shutil.rmtree(reports)
                    def commands(argv, **kwargs):
                        if argv[0] == ci_mutations.sys.executable:
                            Path(argv[-1]).write_bytes(diff)
                            return subprocess.CompletedProcess(argv, 0)
                        if "--version" in argv:
                            return original_run(argv, **kwargs)
                        self.assertIn("--list", argv, "must never build or run mutations")
                        return original_run(argv, **kwargs)
                    output = io.StringIO()
                    with patch.object(ci_mutations, "revision", return_value="1" * 40), \
                            patch.object(ci_mutations.subprocess, "run", side_effect=commands), \
                            patch.object(quality, "REPORTS", reports), \
                            contextlib.redirect_stdout(output):
                        ci_mutations.run_scope(root, "HEAD")
                    self.assertIn("Mutation score: N/A", output.getvalue())
                    self.assertFalse((root / "mutants.out").exists())
                    self.assertFalse((root / "target/debug").exists())


if __name__ == "__main__":
    unittest.main()
