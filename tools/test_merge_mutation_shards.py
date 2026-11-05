import contextlib
import copy
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from merge_mutation_shards import (
    BASE_REVISION, TOOL_VERSION, digest, make_manifest, merge_shards, seal_shard,
    validate_manifest, source_snapshot, verify_worktree, file_digest, write_new,
    workspace_packages,
    main as shard_main,
)
import quality


def candidate(index):
    return {"name": f"crates/example/src/lib.rs:{index + 1}:1: replace true with false",
            "package": "example", "file": "crates/example/src/lib.rs", "function": None,
            "span": {"start": {"line": index + 1, "column": 1},
                     "end": {"line": index + 1, "column": 5}},
            "replacement": "false", "genre": "FnValue", "diff": f"diff {index}"}


def phase(name, status, packages=None):
    argv = ["cargo", "test"] + (["--no-run"] if name == "Build" else []) + ["--verbose"]
    argv += ([f"--package={package}" for package in packages]
             if packages is not None else ["--workspace"])
    return {"phase": name, "duration": 1.0, "process_status": status,
            "argv": [*argv, "--locked"]}


def outcome(value, summary="CaughtMutant", packages=None):
    if summary in {"Success", "MissedMutant"}:
        phases = [phase("Build", "Success", packages), phase("Test", "Success", packages)]
    elif summary == "CaughtMutant":
        phases = [phase("Build", "Success"), phase("Test", {"Failure": 101})]
    elif summary == "Unviable":
        phases = [phase("Build", {"Failure": 101})]
    else:
        phases = [phase("Build", "Success"), phase("Test", "Timeout")]
    scenario = "Baseline" if value is None else {
        "Mutant": {key: item for key, item in value.items() if key != "diff"}}
    return {"scenario": scenario, "summary": summary, "phase_results": phases,
            "log_path": "log/baseline.log" if value is None else "log/mutant.log",
            "diff_path": None if value is None else "diff/mutant.diff"}


def report(candidates, summaries=None):
    summaries = summaries or ["CaughtMutant"] * len(candidates)
    counts = {name: summaries.count(summary) for name, summary in [
        ("caught", "CaughtMutant"), ("missed", "MissedMutant"),
        ("timeout", "Timeout"), ("unviable", "Unviable")]}
    packages = [f"{name}@0.1.0" for name in sorted({value["package"] for value in candidates})]
    return {"outcomes": [outcome(None, "Success", packages),
                         *[outcome(value, summary) for value, summary in zip(candidates, summaries)]],
            "total_mutants": len(candidates), **counts, "success": 0,
            "start_time": "2026-11-05T01:00:00Z", "end_time": "2026-11-05T01:10:00Z",
            "cargo_mutants_version": TOOL_VERSION}


def campaign():
    source = {"revision": "1" * 40, "base_revision": BASE_REVISION, "tree": "2" * 40,
              "files": {"crates/example/src/lib.rs": "3" * 64, "Cargo.lock": "4" * 64},
              "workspace_packages": {"example": "0.1.0"}}
    candidates = [candidate(index) for index in range(8)]
    listed = [candidates[index::3] for index in range(3)]
    manifest = make_manifest(source, "5" * 64, candidates, listed)
    results = [seal_shard(manifest, f"{index}/3", source, values, report(values))
               for index, values in enumerate(listed)]
    return manifest, results


def rehash(result):
    result["outcomes_sha256"] = digest(result["outcomes"])


class MutationShardContracts(unittest.TestCase):
    def runner_command(self, root, diff):
        stdout = io.StringIO()
        argv = ["merge_mutation_shards.py", "commands", "--manifest", str(root / "campaign.json"),
                "--root", str(root), "--diff", str(diff), "--shard", "0/3"]
        with patch.object(sys, "argv", argv), \
             patch("merge_mutation_shards.read_json", return_value=campaign()[0]), \
             patch("merge_mutation_shards.verify_worktree"), contextlib.redirect_stdout(stdout):
            shard_main()
        line = next(line for line in stdout.getvalue().splitlines() if line.startswith("if "))
        suffix = "; then shard_exit=0; else shard_exit=$?; fi"
        self.assertTrue(line.endswith(suffix))
        return shlex.split(line[len("if "):-len(suffix)])

    def test_in_place_runner_omits_jobs_and_clears_inherited_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = self.runner_command(root, root / "scope.diff")
        self.assertEqual(command[:3], ["env", "-u", "CARGO_MUTANTS_JOBS"])
        self.assertIn("--in-place", command)
        self.assertNotIn("--jobs", command)
        self.assertFalse(any(arg.startswith("--jobs=") for arg in command))

    def test_generated_runner_parses_with_real_pinned_cargo_mutants_list_only(self):
        if not shutil.which("cargo") or not shutil.which("cargo-mutants"):
            self.skipTest("cargo-mutants 27.1.0 is optional for the parser-only integration check")
        version = subprocess.check_output(["cargo", "mutants", "--version"], text=True).strip()
        if version != f"cargo-mutants {TOOL_VERSION}":
            self.skipTest(f"parser-only integration check needs cargo-mutants {TOOL_VERSION}")
        with tempfile.TemporaryDirectory(prefix="bml-mutants-cli-") as directory:
            root = Path(directory)
            (root / "Cargo.toml").write_text(
                '[package]\nname = "merge-shard-cli-probe"\nversion = "0.1.0"\n'
                'edition = "2021"\n[workspace]\n')
            (root / "Cargo.lock").write_text(
                'version = 4\n\n[[package]]\nname = "merge-shard-cli-probe"\nversion = "0.1.0"\n')
            (root / "src").mkdir()
            (root / "src/lib.rs").write_text("pub fn flag() -> bool { true }\n")
            diff = root / "scope.diff"
            diff.write_text(
                'diff --git a/src/lib.rs b/src/lib.rs\n--- a/src/lib.rs\n+++ b/src/lib.rs\n'
                '@@ -1 +1 @@\n-pub fn flag() -> bool { false }\n+pub fn flag() -> bool { true }\n')
            command = self.runner_command(root, diff) + ["--list", "--json"]
            inherited = {**os.environ, "CARGO_MUTANTS_JOBS": "7"}
            result = subprocess.run(command, cwd=root, env=inherited, text=True,
                                    capture_output=True, check=True, timeout=60)
            candidates = json.loads(result.stdout)
            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0]["package"], "merge-shard-cli-probe")
            self.assertEqual(candidates[0]["replacement"], "false")
            self.assertFalse((root / "target/final-mutation-build-0").exists())
            self.assertFalse((root / "target/final-mutation-shard-0").exists())

    def test_zero_based_round_robin_lists_and_complete_merge(self):
        manifest, results = campaign()
        self.assertEqual([row["shard"] for row in manifest["shards"]], ["0/3", "1/3", "2/3"])
        self.assertEqual(manifest["shards"][0]["candidates"], manifest["candidates"][::3])
        merged = merge_shards(manifest, list(reversed(results)))
        self.assertEqual(merged["total_mutants"], 8)
        self.assertEqual(merged["caught"], 8)
        self.assertEqual(len(merged["outcomes"]), 9)
        self.assertEqual(sum(row["scenario"] == "Baseline" for row in merged["outcomes"]), 1)
        self.assertEqual(len(merged["shard_baselines"]), 3)
        self.assertEqual([row["source_shard"] for row in merged["shard_baselines"]],
                         ["0/3", "1/3", "2/3"])

    def test_missing_extra_or_duplicate_shards_fail(self):
        manifest, results = campaign()
        for values in [results[:-1], results + [results[0]], [results[0], results[0], results[2]]]:
            with self.subTest(count=len(values)), self.assertRaises(ValueError):
                merge_shards(manifest, values)

    def test_missing_or_duplicated_candidate_outcomes_fail(self):
        for duplicate in [False, True]:
            manifest, results = campaign()
            outcomes = results[0]["outcomes"]
            if duplicate:
                outcomes["outcomes"].append(copy.deepcopy(outcomes["outcomes"][1]))
            else:
                outcomes["outcomes"].pop()
            rehash(results[0])
            with self.subTest(duplicate=duplicate), self.assertRaises(ValueError):
                merge_shards(manifest, results)

    def test_assignment_lists_cannot_drop_duplicate_or_move_candidates(self):
        manifest, _ = campaign()
        lists = [row["candidates"] for row in manifest["shards"]]
        for mode in ["missing", "duplicate", "wrong-shard"]:
            changed = copy.deepcopy(lists)
            if mode == "missing":
                changed[0].pop()
            elif mode == "duplicate":
                changed[1].append(copy.deepcopy(changed[0][0]))
            else:
                changed[0][0], changed[1][0] = changed[1][0], changed[0][0]
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                make_manifest(manifest["source"], manifest["diff_sha256"],
                              manifest["candidates"], changed)

    def test_failed_missing_or_duplicate_baseline_fails(self):
        for mode in ["failed", "missing", "duplicate"]:
            manifest, results = campaign()
            rows = results[1]["outcomes"]["outcomes"]
            if mode == "failed":
                rows[0]["summary"] = "Failure"
            elif mode == "missing":
                rows.pop(0)
            else:
                rows.append(copy.deepcopy(rows[0]))
            rehash(results[1])
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                merge_shards(manifest, results)

    def test_incomplete_or_reversed_time_interval_fails(self):
        for value in [None, "", "2026-11-04T00:00:00Z", "not a timestamp"]:
            manifest, results = campaign()
            results[2]["outcomes"]["end_time"] = value
            rehash(results[2])
            with self.subTest(end_time=value), self.assertRaises(ValueError):
                merge_shards(manifest, results)

    def test_unexpected_identity_or_outcome_from_another_shard_fails(self):
        for mode in ["unknown", "wrong-shard", "changed-package"]:
            manifest, results = campaign()
            mutant = results[0]["outcomes"]["outcomes"][1]["scenario"]["Mutant"]
            if mode == "unknown":
                mutant["name"] = "unexpected id"
            elif mode == "wrong-shard":
                other = results[1]["outcomes"]["outcomes"][1]["scenario"]["Mutant"]
                results[0]["outcomes"]["outcomes"][1]["scenario"]["Mutant"] = copy.deepcopy(other)
            else:
                mutant["package"] = "another-source-package"
            rehash(results[0])
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                merge_shards(manifest, results)

    def test_sources_revision_base_tree_and_file_hashes_must_match(self):
        for field in ["revision", "base_revision", "tree", "files", "workspace_packages"]:
            manifest, results = campaign()
            results[0]["source"] = copy.deepcopy(results[0]["source"])
            results[0]["source"][field] = {} if field == "files" else "f" * 40
            with self.subTest(field=field), self.assertRaises(ValueError):
                merge_shards(manifest, results)
        manifest, _ = campaign()
        manifest["source"]["base_revision"] = "f" * 40
        with self.assertRaises(ValueError):
            make_manifest(manifest["source"], manifest["diff_sha256"], manifest["candidates"],
                          [row["candidates"] for row in manifest["shards"]])

    def test_aggregate_counts_must_agree_with_every_terminal_outcome(self):
        for field in ["total_mutants", "caught", "missed", "timeout", "unviable", "success"]:
            manifest, results = campaign()
            results[0]["outcomes"][field] += 1
            rehash(results[0])
            with self.subTest(field=field), self.assertRaises(ValueError):
                merge_shards(manifest, results)

    def test_manifest_and_result_checksums_are_required(self):
        manifest, results = campaign()
        changed = copy.deepcopy(manifest)
        changed["candidates"][0]["diff"] = "different source diff"
        with self.assertRaises(ValueError):
            validate_manifest(changed)
        results[0]["outcomes"]["caught"] += 1
        with self.assertRaisesRegex(ValueError, "checksum"):
            merge_shards(manifest, results)
        results[0]["outcomes"]["caught"] -= 1
        results[0]["manifest_sha256"] = "f" * 64
        with self.assertRaises(ValueError):
            merge_shards(manifest, results)

    def test_caught_cannot_be_claimed_for_successful_tests_or_compile_failure(self):
        for phases in [[phase("Build", "Success"), phase("Test", "Success")],
                       [phase("Build", {"Failure": 101})],
                       [phase("Build", "Success"), phase("Test", {"Failure": 0})]]:
            manifest, results = campaign()
            results[0]["outcomes"]["outcomes"][1]["phase_results"] = phases
            rehash(results[0])
            with self.subTest(phases=phases), self.assertRaises(ValueError):
                merge_shards(manifest, results)

    def test_unknown_summary_check_only_or_wrong_tool_version_fails(self):
        for mode in ["summary", "check-only", "version"]:
            manifest, results = campaign()
            data = results[0]["outcomes"]
            if mode == "summary":
                data["outcomes"][1]["summary"] = "Success"
            elif mode == "check-only":
                data["outcomes"][0]["phase_results"] = [phase("Check", "Success")]
            else:
                data["cargo_mutants_version"] = "another-version"
            rehash(results[0])
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                merge_shards(manifest, results)

    def test_missed_timeout_and_unviable_are_preserved_for_quality_gate(self):
        manifest, results = campaign()
        assigned = results[0]["candidates"]
        data = report(assigned, ["MissedMutant", "Timeout", "Unviable"])
        results[0] = seal_shard(manifest, "0/3", manifest["source"], assigned, data)
        merged = merge_shards(manifest, results)
        self.assertEqual((merged["caught"], merged["missed"], merged["timeout"], merged["unviable"]),
                         (5, 1, 1, 1))
        self.assertEqual([row["summary"] for row in merged["outcomes"][1:]],
                         ["MissedMutant", "CaughtMutant", "CaughtMutant", "Timeout",
                          "CaughtMutant", "CaughtMutant", "Unviable", "CaughtMutant"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aggregate.json"
            path.write_text(json.dumps(merged))
            with patch.object(quality, "REPORTS", Path(directory)), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(SystemExit, "Mutation gate failed"):
                    quality.mutations(path)

    def test_complete_aggregate_is_compatible_with_unmodified_quality_gate(self):
        manifest, results = campaign()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aggregate.json"
            write_new(path, merge_shards(manifest, results))
            with patch.object(quality, "REPORTS", Path(directory)), contextlib.redirect_stdout(io.StringIO()):
                quality.mutations(path)
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                write_new(path, {})
            self.assertEqual(path.read_bytes(), before)

    def test_phase_argv_rejects_filters_features_unlocked_or_reduced_scope(self):
        changes = [
            lambda argv: argv + ["--", "--skip", "some_test"],
            lambda argv: argv + ["--test", "one_target"],
            lambda argv: argv + ["--no-default-features"],
            lambda argv: argv + ["--all-features"],
            lambda argv: argv + ["--features=extra"],
            lambda argv: argv + ["--exclude=example"],
            lambda argv: [arg for arg in argv if arg != "--locked"],
            lambda argv: ["--package=example@0.1.0" if arg == "--workspace" else arg
                          for arg in argv] + ["--", "only_this_test"],
        ]
        for outcome_index in [0, 1]:
            for phase_index in [0, 1]:
                for change in changes:
                    manifest, results = campaign()
                    row = results[0]["outcomes"]["outcomes"][outcome_index]["phase_results"][phase_index]
                    row["argv"] = change(row["argv"])
                    rehash(results[0])
                    with self.subTest(outcome=outcome_index, phase=phase_index, argv=row["argv"]), \
                         self.assertRaisesRegex(ValueError, "cargo argv differs"):
                        merge_shards(manifest, results)

    def test_baseline_package_scope_and_versions_match_shard_but_mutants_use_workspace(self):
        source = campaign()[0]["source"]
        source["workspace_packages"] = {
            "example": "0.1.0", "bml": "1.0.0", "bml-tests": "0.1.0", "untouched": "9.0.0"}
        candidates = [candidate(index) for index in range(3)]
        for row, package in zip(candidates, ["example", "bml", "bml-tests"]):
            row["package"] = package
        manifest = make_manifest(source, "5" * 64, candidates, [candidates])
        data = report(candidates)
        packages = ["bml@1.0.0", "bml-tests@0.1.0", "example@0.1.0"]
        data["outcomes"][0] = outcome(None, "Success", packages)
        result = seal_shard(manifest, "0/1", source, candidates, data)
        self.assertEqual(merge_shards(manifest, [result])["caught"], 3)
        for wrong_packages in [None, packages[:-1], packages + ["untouched@9.0.0"],
                               ["bml@0.1.0", *packages[1:]]]:
            changed = copy.deepcopy(data)
            changed["outcomes"][0] = outcome(None, "Success", wrong_packages)
            with self.subTest(packages=wrong_packages), self.assertRaisesRegex(ValueError, "cargo argv"):
                seal_shard(manifest, "0/1", source, candidates, changed)
        changed = copy.deepcopy(data)
        changed["outcomes"][1]["phase_results"][1] = phase("Test", {"Failure": 101}, packages)
        with self.assertRaisesRegex(ValueError, "cargo argv"):
            seal_shard(manifest, "0/1", source, candidates, changed)

    def test_workspace_metadata_includes_members_and_ignores_external_dependencies(self):
        metadata = {
            "workspace_members": ["bml-id", "tests-id"],
            "packages": [
                {"id": "bml-id", "name": "bml", "version": "1.0.0"},
                {"id": "tests-id", "name": "bml-tests", "version": "0.1.0"},
                {"id": "dependency-id", "name": "external", "version": "9.0.0"},
            ],
        }
        with patch("merge_mutation_shards.subprocess.check_output",
                   return_value=json.dumps(metadata).encode()) as cargo:
            self.assertEqual(workspace_packages(Path("/frozen")), {"bml": "1.0.0", "bml-tests": "0.1.0"})
        cargo.assert_called_once_with(
            ["cargo", "metadata", "--no-deps", "--format-version", "1", "--locked"], cwd=Path("/frozen"))
        metadata["workspace_members"].append("missing-member")
        with patch("merge_mutation_shards.subprocess.check_output", return_value=json.dumps(metadata).encode()), \
             self.assertRaisesRegex(ValueError, "metadata is incomplete"):
            workspace_packages(Path("/frozen"))

    def test_invalid_shard_counts_empty_lists_and_duplicate_candidate_ids_fail(self):
        manifest, _ = campaign()
        for candidates, lists in [([], []), (manifest["candidates"], []),
                                  (manifest["candidates"], [manifest["candidates"], []])]:
            with self.subTest(lists=len(lists)), self.assertRaises(ValueError):
                make_manifest(manifest["source"], manifest["diff_sha256"], candidates, lists)
        values = [candidate(0), candidate(0)]
        with self.assertRaises(ValueError):
            make_manifest(manifest["source"], manifest["diff_sha256"], values, [values])

    def test_source_verification_reads_file_hashes_even_if_git_diff_is_empty(self):
        revision, tree = "1" * 40, "2" * 40

        def git_answer(root, *args):
            return {("rev-parse", "HEAD"): revision, ("rev-parse", BASE_REVISION): BASE_REVISION,
                    ("rev-parse", "HEAD^{tree}"): tree, ("diff", "--name-only", "HEAD"): "",
                    ("ls-files", "--others", "--exclude-standard"): ""}[args]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "code.rs").write_text("original source")
            diff = root / "scope.diff"
            diff.write_text("frozen scope")
            with patch("merge_mutation_shards.git", side_effect=git_answer), \
                 patch("merge_mutation_shards.workspace_packages", return_value={"example": "0.1.0"}) as metadata, \
                    patch("merge_mutation_shards.subprocess.check_output", return_value=b"code.rs\0"):
                source = source_snapshot(root, revision, BASE_REVISION)
                candidates = [candidate(0)]
                manifest = make_manifest(source, file_digest(diff), candidates, [candidates])
                verify_worktree(manifest, root, diff)
                metadata.return_value = {"example": "0.2.0"}
                with self.assertRaisesRegex(ValueError, "sources differ"):
                    verify_worktree(manifest, root, diff)
                metadata.return_value = {"example": "0.1.0"}
                (root / "code.rs").write_text("different source")
                with self.assertRaisesRegex(ValueError, "sources differ"):
                    verify_worktree(manifest, root, diff)


if __name__ == "__main__":
    unittest.main()
