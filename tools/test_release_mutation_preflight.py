import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import merge_mutation_shards as historical
import release_mutation_preflight as pre


REVISION = "1" * 40


def candidate(index):
    return {"name": f"code.rs:{index + 1}:1: replace true with false", "package": "example",
            "file": "code.rs", "function": None, "genre": "FnValue", "replacement": "false",
            "span": {"start": {"line": index + 1, "column": 1},
                     "end": {"line": index + 1, "column": 5}}, "diff": f"diff {index}"}


def fixture(diff_hash="4" * 64):
    source = {"revision": REVISION, "base_revision": pre.RELEASE_BASE, "tree": "2" * 40,
              "files": {"code.rs": "3" * 64}, "workspace_packages": {"example": "0.1.0"}}
    candidates = [candidate(index) for index in range(129)]
    partitions = [[row for position, row in enumerate(candidates) if position % 64 == index]
                  for index in range(64)]
    producer = {name: "5" * 64 for name in pre.PRODUCER_FILES}
    return source, diff_hash, candidates, partitions, producer


def reseal(plan):
    plan["plan_sha256"] = pre.digest({key: value for key, value in plan.items()
                                     if key != "plan_sha256"})


class ReleasePreflightContracts(unittest.TestCase):
    def test_release_intent_is_explicit_and_only_accepts_the_documented_base(self):
        pre.check_intent("release-1.0", REVISION, pre.RELEASE_BASE)
        for profile, revision, base in [
                ("development", REVISION, pre.RELEASE_BASE),
                (None, REVISION, pre.RELEASE_BASE),
                (pre.PROFILE, REVISION, historical.BASE_REVISION),
                (pre.PROFILE, REVISION, "f" * 40),
                (pre.PROFILE, REVISION[:7], pre.RELEASE_BASE),
                (pre.PROFILE, "G" * 40, pre.RELEASE_BASE)]:
            with self.subTest(profile=profile, revision=revision, base=base), \
                    self.assertRaises(ValueError):
                pre.check_intent(profile, revision, base)

    def test_snapshot_checks_head_base_ancestry_and_cleanliness_before_metadata(self):
        answers = {("rev-parse", "HEAD"): REVISION,
                   ("rev-parse", "--verify", f"{pre.RELEASE_BASE}^{{commit}}"): pre.RELEASE_BASE,
                   ("merge-base", pre.RELEASE_BASE, REVISION): pre.RELEASE_BASE,
                   ("status", "--porcelain"): ""}
        for key, value, message in [
                (("rev-parse", "HEAD"), "a" * 40, "revision differs"),
                (("rev-parse", "--verify", f"{pre.RELEASE_BASE}^{{commit}}"), "a" * 40,
                 "base unavailable"),
                (("merge-base", pre.RELEASE_BASE, REVISION), "a" * 40, "not an ancestor"),
                (("status", "--porcelain"), " M code.rs", "must be clean"),
                (("status", "--porcelain"), "?? extra.rs", "must be clean")]:
            changed = {**answers, key: value}
            with self.subTest(condition=key, value=value), \
                    patch.object(pre, "git", side_effect=lambda root, *args: changed[args]), \
                    patch.object(pre, "workspace_packages") as metadata, \
                    patch.object(pre.subprocess, "check_output") as command, \
                    self.assertRaisesRegex(ValueError, message):
                pre.release_source_snapshot(Path("/frozen"), REVISION, pre.RELEASE_BASE)
            metadata.assert_not_called()
            command.assert_not_called()

    def test_snapshot_hashes_actual_files_and_not_only_git_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_file = root / "code.rs"
            source_file.write_bytes(b"original")
            answers = {("rev-parse", "HEAD"): REVISION,
                       ("rev-parse", "--verify", f"{pre.RELEASE_BASE}^{{commit}}"): pre.RELEASE_BASE,
                       ("merge-base", pre.RELEASE_BASE, REVISION): pre.RELEASE_BASE,
                       ("status", "--porcelain"): "", ("rev-parse", "HEAD^{tree}"): "2" * 40}
            with patch.object(pre, "git", side_effect=lambda root, *args: answers[args]), \
                    patch.object(pre, "workspace_packages", return_value={"example": "0.1.0"}), \
                    patch.object(pre.subprocess, "check_output", return_value=b"code.rs\0") as command:
                first = pre.release_source_snapshot(root, REVISION, pre.RELEASE_BASE)
                source_file.write_bytes(b"changed")
                second = pre.release_source_snapshot(root, REVISION, pre.RELEASE_BASE)
            self.assertEqual(first["files"], {"code.rs": hashlib.sha256(b"original").hexdigest()})
            self.assertNotEqual(first["files"], second["files"])
            self.assertTrue(all(call.args[0] == ["git", "ls-files", "-z"]
                                for call in command.call_args_list))

    def test_plan_contains_all_candidates_once_in_64_nonempty_partitions(self):
        plan = pre.make_plan(*fixture())
        pre.validate_plan(plan)
        self.assertEqual(plan["profile"], "release-1.0")
        self.assertEqual(plan["purpose"], "list-only")
        self.assertEqual([row["shard"] for row in plan["shards"]], [f"{i}/64" for i in range(64)])
        self.assertEqual([len(row["candidates"]) for row in plan["shards"]], [3] + [2] * 63)
        names = [row["name"] for shard in plan["shards"] for row in shard["candidates"]]
        self.assertEqual(len(names), 129)
        self.assertEqual(len(set(names)), 129)
        self.assertEqual(set(names), {row["name"] for row in plan["candidates"]})
        self.assertNotIn("format_version", plan)
        self.assertNotIn("manifest_sha256", plan)

    def test_partition_missing_overlap_wrong_order_or_wrong_count_is_rejected(self):
        for mode in ["missing-shard", "extra-shard", "omission", "overlap", "wrong-shard", "order"]:
            source, diff_hash, candidates, partitions, producer = fixture()
            if mode == "missing-shard":
                partitions.pop()
            elif mode == "extra-shard":
                partitions.append([candidates[0]])
            elif mode == "omission":
                partitions[0].pop()
            elif mode == "overlap":
                partitions[1][0] = partitions[0][0]
            elif mode == "wrong-shard":
                partitions[0], partitions[1] = partitions[1], partitions[0]
            else:
                partitions[0].reverse()
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "64 listed|assignment differs"):
                pre.make_plan(source, diff_hash, candidates, partitions, producer)

    def test_invalid_source_producer_and_candidate_identity_cannot_form_a_plan(self):
        for mode in ["base", "revision", "tree", "path", "file-hash", "producer-hash",
                     "producer-missing", "foreign-package", "package-type", "duplicate", "too-few"]:
            source, diff_hash, candidates, partitions, producer = fixture()
            if mode in {"base", "revision", "tree"}:
                field = "base_revision" if mode == "base" else mode
                source[field] = historical.BASE_REVISION if mode == "base" else "short"
            elif mode == "path":
                source["files"] = {"../code.rs": "3" * 64}
            elif mode == "file-hash":
                source["files"]["code.rs"] = "bad"
            elif mode == "producer-hash":
                producer[pre.PRODUCER_FILES[0]] = "bad"
            elif mode == "producer-missing":
                producer.pop(pre.PRODUCER_FILES[0])
            elif mode == "foreign-package":
                candidates[0]["package"] = "outside"
            elif mode == "package-type":
                candidates[0]["package"] = ["example"]
            elif mode == "duplicate":
                candidates[1] = candidates[0]
            else:
                candidates = candidates[:63]
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                pre.make_plan(source, diff_hash, candidates, partitions, producer)

    def test_plan_checksums_and_resealed_assignment_metadata_are_checked(self):
        for mode in ["checksum", "shard-id", "shard-hash", "candidate-hash", "profile", "extra"]:
            plan = pre.make_plan(*fixture())
            if mode == "checksum":
                plan["plan_sha256"] = "0" * 64
            elif mode == "shard-id":
                plan["shards"][0]["shard"] = "1/64"
            elif mode == "shard-hash":
                plan["shards"][0]["candidates_sha256"] = "0" * 64
            elif mode == "candidate-hash":
                plan["candidates_sha256"] = "0" * 64
            elif mode == "profile":
                plan["profile"] = "development"
            else:
                plan["timeout"] = 1
            if mode != "checksum":
                reseal(plan)
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                pre.validate_plan(plan)

    def test_release_plan_and_release_base_remain_rejected_by_historical_protocol(self):
        source, diff_hash, candidates, partitions, producer = fixture()
        with self.assertRaises(ValueError):
            historical.validate_manifest(pre.make_plan(source, diff_hash, candidates, partitions, producer))
        with self.assertRaisesRegex(ValueError, "unexpected development base"):
            historical.make_manifest(source, diff_hash, candidates, partitions)
        historical_source = {**source, "base_revision": historical.BASE_REVISION}
        manifest = historical.make_manifest(historical_source, diff_hash, candidates, partitions)
        self.assertEqual(len(historical.validate_manifest(manifest)), 64)

    @contextlib.contextmanager
    def verification_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            diff = root / "scope.diff"
            diff.write_text("complete formatted scope")
            plan = pre.make_plan(*fixture(pre.file_digest(diff)))
            plan_path = root / "plan.json"
            pre.write_new(plan_path, plan)
            with patch.object(pre, "producer_identity", return_value=plan["producer"]), \
                    patch.object(pre, "release_source_snapshot", return_value=plan["source"]) as source, \
                    patch.object(pre, "verify_full_scope") as full_scope:
                args = (root, pre.PROFILE, REVISION, pre.RELEASE_BASE, diff, plan_path,
                        pre.file_digest(plan_path))
                yield plan, args, source, full_scope

    def test_verify_checks_expected_intent_and_the_complete_scope(self):
        with self.verification_fixture() as (plan, args, source, full_scope):
            result = pre.verify(*args)
            self.assertEqual(result["verified"], "release preflight only")
            self.assertEqual(result["candidate_count"], 129)
            source.assert_called_once_with(args[0], REVISION, pre.RELEASE_BASE)
            full_scope.assert_called_once_with(args[0], args[4], pre.RELEASE_BASE)
        for changed_index, value, message in [
                (1, "development", "explicit release"),
                (2, "a" * 40, "intended revision"),
                (3, historical.BASE_REVISION, "unexpected Release")]:
            with self.subTest(index=changed_index), self.verification_fixture() as (_, args, source, _):
                changed = list(args)
                changed[changed_index] = value
                with self.assertRaisesRegex(ValueError, message):
                    pre.verify(*changed)
                source.assert_not_called()

    def test_verify_rejects_file_changes_and_wrong_independent_hash(self):
        for mode in ["altered-file", "wrong-hash", "short-hash"]:
            with self.subTest(mode=mode), self.verification_fixture() as (_, args, source, _):
                changed = list(args)
                if mode == "altered-file":
                    args[5].write_text(args[5].read_text() + " ")
                else:
                    changed[-1] = "0" * 64 if mode == "wrong-hash" else "short"
                with self.assertRaisesRegex(ValueError, "independent file digest|expected plan file digest"):
                    pre.verify(*changed)
                source.assert_not_called()

    def test_verify_rejects_source_producer_and_scope_drift(self):
        for mode in ["source", "producer", "diff", "reduced-scope", "scope-swap"]:
            with self.subTest(mode=mode), self.verification_fixture() as (plan, args, source, full_scope):
                if mode == "source":
                    source.return_value = copy.deepcopy(plan["source"])
                    source.return_value["files"]["code.rs"] = "a" * 64
                    message = "sources differ"
                elif mode == "producer":
                    changed = copy.deepcopy(plan)
                    changed["producer"][pre.PRODUCER_FILES[0]] = "a" * 64
                    reseal(changed)
                    args[5].write_text(json.dumps(changed))
                    args = (*args[:-1], pre.file_digest(args[5]))
                    message = "producer differs"
                elif mode == "diff":
                    args[4].write_text("reduced diff")
                    message = "diff changed"
                elif mode == "reduced-scope":
                    full_scope.side_effect = ValueError("scope diff is not the complete formatting-aware development diff")
                    message = "not the complete"
                else:
                    full_scope.side_effect = lambda *unused: args[4].write_text("swapped during verification")
                    message = "diff changed"
                with self.assertRaisesRegex(ValueError, message):
                    pre.verify(*args)

    def test_coordinated_omission_with_recomputed_hashes_still_fails_external_digest(self):
        with self.verification_fixture() as (plan, args, source, _):
            candidates = plan["candidates"][:-1]
            partitions = [[row for position, row in enumerate(candidates) if position % 64 == index]
                          for index in range(64)]
            changed = pre.make_plan(plan["source"], plan["diff_sha256"], candidates,
                                    partitions, plan["producer"])
            pre.validate_plan(changed)
            args[5].write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "independent file digest"):
                pre.verify(*args)
            source.assert_not_called()

    def prepared_environment(self, stack, root):
        diff = root / "scope.diff"
        diff.write_text("complete scope")
        source, _, candidates, partitions, producer = fixture(pre.file_digest(diff))
        snapshots = stack.enter_context(patch.object(pre, "release_source_snapshot", return_value=source))
        full_scope = stack.enter_context(patch.object(pre, "verify_full_scope"))
        producer_call = stack.enter_context(patch.object(pre, "producer_identity", return_value=producer))

        def only_listing(argv, cwd):
            self.assertEqual(cwd, root)
            if argv == ["cargo", "mutants", "--version"]:
                return b"cargo-mutants 27.1.0\n"
            self.assertEqual(argv[:8], ["cargo", "mutants", "--workspace", "--cargo-arg=--locked",
                                       "--list", "--json", "--in-diff", str(diff.resolve())])
            if len(argv) == 8:
                return json.dumps(candidates).encode()
            self.assertEqual(argv[8:11], ["--sharding", "round-robin", "--shard"])
            self.assertEqual(len(argv), 12)
            index, denominator = map(int, argv[-1].split("/"))
            self.assertEqual(denominator, 64)
            return json.dumps(partitions[index]).encode()

        command = stack.enter_context(patch.object(pre.subprocess, "check_output", side_effect=only_listing))
        return diff, source, snapshots, full_scope, producer_call, command

    def test_prepare_issues_only_one_global_and_64_independent_listings(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            root = Path(directory)
            diff, source, snapshots, full_scope, _, command = self.prepared_environment(stack, root)
            output = root / "new-plan.json"
            result = pre.prepare(root, pre.PROFILE, REVISION, pre.RELEASE_BASE, diff, output)
            commands = [call.args[0] for call in command.call_args_list]
            self.assertEqual(len(commands), 66)
            self.assertEqual(commands[0], ["cargo", "mutants", "--version"])
            self.assertEqual([argv[-1] for argv in commands[2:]], [f"{i}/64" for i in range(64)])
            self.assertEqual(result["plan_file_sha256"], pre.file_digest(output))
            self.assertEqual((result["candidate_count"], result["listed_shards"]), (129, 64))
            self.assertEqual((result["baseline_runs"], result["mutants_evaluated"]), (0, 0))
            pre.validate_plan(json.loads(output.read_text()))
            self.assertEqual(snapshots.call_count, 2)
            full_scope.assert_called_once_with(root, diff, pre.RELEASE_BASE)

    def test_prepare_rejects_drift_without_writing_a_plan(self):
        for mode in ["source", "producer", "diff"]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory, \
                    contextlib.ExitStack() as stack:
                root = Path(directory)
                diff, source, snapshots, _, producer_call, command = self.prepared_environment(stack, root)
                if mode == "source":
                    changed = copy.deepcopy(source)
                    changed["files"]["code.rs"] = "a" * 64
                    snapshots.side_effect = [source, changed]
                elif mode == "producer":
                    producer_call.side_effect = [producer_call.return_value,
                                                 {name: "a" * 64 for name in pre.PRODUCER_FILES}]
                else:
                    original = command.side_effect

                    def list_then_change(argv, cwd):
                        result = original(argv, cwd)
                        if argv[-1] == "63/64":
                            diff.write_text("changed after listing")
                        return result

                    command.side_effect = list_then_change
                output = root / "new-plan.json"
                with self.assertRaisesRegex(ValueError, "changed during listing"):
                    pre.prepare(root, pre.PROFILE, REVISION, pre.RELEASE_BASE, diff, output)
                self.assertFalse(output.exists())

    def test_prepare_rejects_existing_output_wrong_version_and_reduced_scope(self):
        for mode in ["existing", "version", "scope", "scope-swap"]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory, \
                    contextlib.ExitStack() as stack:
                root = Path(directory)
                diff, _, snapshots, full_scope, _, command = self.prepared_environment(stack, root)
                output = root / "new-plan.json"
                if mode == "existing":
                    output.write_text("preserve existing")
                    message = "already exists"
                elif mode == "version":
                    command.side_effect = None
                    command.return_value = b"cargo-mutants 27.0.0\n"
                    message = "version differs"
                elif mode == "scope":
                    full_scope.side_effect = ValueError("scope diff is not the complete formatting-aware development diff")
                    message = "not the complete"
                else:
                    full_scope.side_effect = lambda *unused: diff.write_text("swapped during verification")
                    message = "diff changed"
                with self.assertRaisesRegex(ValueError, message):
                    pre.prepare(root, pre.PROFILE, REVISION, pre.RELEASE_BASE, diff, output)
                if mode == "existing":
                    snapshots.assert_not_called()
                    self.assertEqual(output.read_text(), "preserve existing")
                else:
                    self.assertFalse(output.exists())
                self.assertFalse(any("--list" in call.args[0] for call in command.call_args_list))

    def test_cli_requires_profile_external_digest_and_has_no_runner_mode(self):
        shared = ["--revision", REVISION, "--base", pre.RELEASE_BASE, "--diff", "scope.diff"]
        for arguments in [["prepare", *shared, "--output", "plan.json"],
                          ["verify", *shared, "--profile", pre.PROFILE, "--plan", "plan.json"],
                          ["commands"], ["run"]]:
            with self.subTest(arguments=arguments), patch.object(sys, "argv", ["preflight", *arguments]), \
                    patch.object(pre, "prepare") as prepare, patch.object(pre, "verify") as verify, \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as stopped:
                pre.main()
            self.assertEqual(stopped.exception.code, 2)
            prepare.assert_not_called()
            verify.assert_not_called()


if __name__ == "__main__":
    unittest.main()
