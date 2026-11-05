import base64
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch
import zipfile

import mutation_subshards as sub
import mutation_recovery as recovery
import quality
from merge_mutation_shards import (
    digest, file_digest, make_manifest, merge_shards, seal_shard, validate_v1_result, write_new,
)
from test_merge_mutation_shards import candidate, report
from test_mutation_recovery import origin, LOG


def origins():
    second = {
        "run": {"id": sub.SECOND_RUN, "run_attempt": 1, "head_sha": sub.SECOND_CONTROLLER,
                "event": "workflow_dispatch", "path": ".github/workflows/mutation-recovery.yml",
                "repository": {"full_name": recovery.REPOSITORY, "id": recovery.REPOSITORY_ID}},
        "job": {"id": sub.SECOND_JOB, "run_id": sub.SECOND_RUN, "run_attempt": 1,
                "head_sha": sub.SECOND_CONTROLLER, "name": "recover", "status": "completed",
                "conclusion": "failure", "started_at": "2026-11-05T04:27:31Z",
                "completed_at": "2026-11-05T04:54:51Z", "steps": [
                    {"name": name, "conclusion": result} for name, result in [
                        ("Verify independent manifest SHA and original partition", "success"),
                        ("Run only the original frozen shard and its baseline", "failure"),
                        ("Attest the live source and identify the actual sealer", "skipped"),
                        ("Record recovery provenance and verify raw reports against the seal", "skipped"),
                        ("Preserve recovery evidence without build outputs", "skipped")]]},
        "artifacts": {"total_count": 0, "artifacts": []}}
    result = {"cause": "unknown; shutdown143 is observed, OOM is not established"}
    for key, api in [("original", origin()), ("second", second)]:
        result[key] = {"api": api, "api_sha256": digest(api),
                       "log_base64": base64.b64encode(LOG.encode()).decode(),
                       "log_sha256": hashlib.sha256(LOG.encode()).hexdigest()}
    return result


def artifact(identity, name, identifier):
    return {"id": identifier, "name": name, "expired": False, "digest": "sha256:" + "a" * 64,
            "workflow_run": {"id": identity["run_id"], "head_sha": identity["controller_revision"],
                             "repository_id": recovery.REPOSITORY_ID,
                             "head_repository_id": recovery.REPOSITORY_ID}}


def fixture():
    source = {"revision": recovery.SOURCE, "base_revision": sub.BASE_REVISION,
              "tree": "2" * 40, "files": {"tools/merge_mutation_shards.py": "3" * 64,
                                          "crates/example/tests/smoke.rs": "4" * 64},
              "workspace_packages": {f"p{i}": "0.1.0" for i in range(4)}}
    candidates = []
    for index in range(256):
        value = candidate(index)
        value["package"] = f"p{index // 64}"
        candidates.append(value)
    parent = make_manifest(source, "5" * 64, candidates, [candidates[i::64] for i in range(64)])
    nested = make_manifest(source, "5" * 64, candidates, [candidates[i::256] for i in range(256)])
    identity = {"repository": recovery.REPOSITORY, "run_id": 999999, "attempt": 1,
                "controller_revision": "6" * 40}
    baseline = {"producer": sub.PRODUCER, "identity": identity, "source": source,
                "source_before_sha256": digest(source), "source_after_sha256": digest(source),
                "environment": sub.ENVIRONMENT, "unmutated_workspace": True,
                "started_at": "2026-11-05T06:01:00Z", "ended_at": "2026-11-05T06:05:00Z",
                "generated_regressions_sha256": {}, "phases": []}
    for name, argv, minute in [("Build", sub.BUILD, 2), ("Test", sub.TEST, 3)]:
        baseline["phases"].append({"phase": name, "argv": argv, "exit_code": 0,
                                   "started_at": f"2026-11-05T06:0{minute}:00Z",
                                   "ended_at": f"2026-11-05T06:0{minute}:59Z",
                                   "log_file": f"workspace-{name.lower()}.log", "log_sha256": "7" * 64})
    plan = {"format_version": 1, "identity": identity, "source": source,
            "original_manifest_file_sha256": recovery.MANIFEST_SHA256,
            "original_manifest_sha256": parent["manifest_sha256"],
            "nested_manifest_sha256": nested["manifest_sha256"],
            "nested_manifest_file_sha256": sub.json_file_sha(nested),
            "listing_producer": "frozen-9a-prepare-256-independent-cli-listings",
            "listing_argv": ["python3", "/source/tools/merge_mutation_shards.py", "prepare",
                             "--root", "/source", "--revision", recovery.SOURCE, "--base", sub.BASE_REVISION,
                             "--shards", "256", "--diff", "/plan/final.diff", "--output", "/plan/nested-campaign.json"],
            "listing_log_sha256": "9" * 64,
            "controller_tools_sha256": {name: "8" * 64 for name in sub.CONTROLLER_TOOLS},
            "frozen_prepare_sha256": "3" * 64, "listed_at": "2026-11-05T06:00:00Z",
            "prepared_at": "2026-11-05T06:06:00Z", "workspace_preflight": baseline,
            "origins": origins()}
    plan_sha = sub.json_file_sha(plan)
    leaves, jobs, artifacts = [], [], []
    for number, shard in enumerate(sub.LEAVES):
        values = nested["shards"][int(shard.split("/")[0])]["candidates"]
        raw = report(values, [["MissedMutant"], ["Timeout"], ["Unviable"], ["CaughtMutant"]][number])
        raw.update(start_time="2026-11-05T06:10:00Z", end_time="2026-11-05T06:11:00Z")
        seal = seal_shard(nested, shard, source, values, raw)
        code = 2 if number == 0 else 3 if number == 1 else 0
        info = {"method": sub.LEAF_METHOD, "identity": identity, "shard": shard,
                "parent_shard": sub.PARENT, "plan_file_sha256": plan_sha,
                "manifest_file_sha256": plan["nested_manifest_file_sha256"],
                "source_revision": recovery.SOURCE, "runner_exit_code": code,
                "cargo_mutants_exit_code": code, "exit1_reason": None,
                "sealer_sha256": "8" * 64, "attested_at": "2026-11-05T06:12:00Z",
                "evidence_files_sha256": {name: "b" * 64 for name in sub.LEAF_FILES},
                "seal_sha256": digest(seal)}
        leaves.append({"format_version": 1, "seal": seal, "attestation": info})
        index = shard.split("/")[0]
        jobs.append({"id": number + 10, "name": f"leaf ({index})", "run_id": identity["run_id"],
                     "run_attempt": 1, "head_sha": identity["controller_revision"],
                     "status": "completed", "conclusion": "success",
                     "started_at": "2026-11-05T06:08:00Z", "completed_at": "2026-11-05T06:13:00Z",
                     "steps": [{"name": name, "conclusion": "success"} for name in (
                         "Execute frozen leaf with diagnostic telemetry", "Attest complete live-source leaf",
                         "Upload complete or partial leaf evidence")]})
        artifacts.append(artifact(identity, f"mutation-subleaf-999999-1-{index}", number + 20))
    jobs.append({"id": 9, "name": "prepare", "run_id": identity["run_id"], "run_attempt": 1,
                 "head_sha": identity["controller_revision"], "status": "completed", "conclusion": "success",
                 "started_at": "2026-11-05T05:58:00Z", "completed_at": "2026-11-05T06:07:00Z",
                 "steps": [{"name": name, "conclusion": "success"} for name in (
                     "List all 256 frozen assignments and run real workspace preflight",
                     "Transmit independent plan and manifest file digests",
                     "Upload immutable nested plan and real workspace baseline")]})
    artifacts.append(artifact(identity, "mutation-subplan-999999-1", 30))
    api = {"run": {"id": identity["run_id"], "run_attempt": 1, "head_sha": identity["controller_revision"],
                   "event": "workflow_dispatch", "path": sub.WORKFLOW,
                   "repository": {"full_name": recovery.REPOSITORY, "id": recovery.REPOSITORY_ID}},
           "jobs": {"total_count": len(jobs), "jobs": jobs},
           "artifacts": {"total_count": len(artifacts), "artifacts": artifacts}}
    return parent, nested, plan, leaves, api


class NestedRecovery(unittest.TestCase):
    def setUp(self):
        self.parent, self.nested, self.plan, self.leaves, self.api = fixture()
        # Small fixture keeps all real 64/256 labels, partition rules and
        # independent baselines. Only the pinned universe fingerprint/count vary.
        for name, value in [("ORIGINAL_MANIFEST_ID", self.parent["manifest_sha256"]), ("PARENT_COUNT", 4)]:
            context = patch.object(sub, name, value)
            context.start()
            self.addCleanup(context.stop)

    def envelope(self):
        return sub.make_envelope(self.parent, self.nested, self.plan, self.leaves, self.api,
                                 sub.json_file_sha(self.plan))

    def reject_changed(self, mutate, pattern):
        mutate()
        with self.assertRaisesRegex(ValueError, pattern):
            self.envelope()

    def test_complete_envelope_retains_literal_baselines_and_statuses(self):
        envelope = self.envelope()
        baselines, mutations, _, _ = sub.validate_envelope(self.parent, envelope)
        self.assertEqual(len(baselines), 4)
        self.assertEqual([b["phase_results"][0]["argv"] for b in baselines],
                         [leaf["seal"]["outcomes"]["outcomes"][0]["phase_results"][0]["argv"]
                          for leaf in self.leaves])
        self.assertEqual([m["summary"] for m in mutations],
                         ["MissedMutant", "Timeout", "Unviable", "CaughtMutant"])
        self.assertEqual(envelope["plan"]["workspace_preflight"]["phases"][0]["argv"], sub.BUILD)
        # A nested aggregate cannot pass as an original format-1 parent seal.
        with self.assertRaisesRegex(ValueError, "format"):
            validate_v1_result(self.parent, envelope)

    def test_full_parent_merge_accepts_exactly_one_v2_and_gate_still_rejects_missed(self):
        results = [seal_shard(self.parent, row["shard"], self.parent["source"],
                              row["candidates"], report(row["candidates"]))
                   for row in self.parent["shards"] if row["shard"] != sub.PARENT]
        results.append(self.envelope())
        merged = merge_shards(self.parent, list(reversed(results)))
        self.assertEqual(merged["total_mutants"], 256)
        self.assertEqual((merged["missed"], merged["timeout"]), (1, 1))
        self.assertEqual(len(merged["shard_baselines"]), 67)
        self.assertEqual(merged["nested_recoveries"][sub.PARENT]["leaves"], self.leaves)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            write_new(path, merged)
            with self.assertRaises(SystemExit):
                quality.mutations(path)

    def test_missing_duplicate_or_foreign_leaf_fails(self):
        for values in [self.leaves[:3], [self.leaves[0]] * 4]:
            with self.subTest(length=len(values)):
                with self.assertRaises(ValueError):
                    sub.make_envelope(self.parent, self.nested, self.plan, values, self.api,
                                      sub.json_file_sha(self.plan))
        self.reject_changed(lambda: self.leaves[0]["seal"].update(shard="21/256"), "assignment")

    def test_leaf_baseline_cannot_claim_parent_or_workspace_scope(self):
        raw = self.leaves[0]["seal"]["outcomes"]
        raw["outcomes"][0]["phase_results"][0]["argv"] = sub.BUILD
        self.leaves[0]["seal"]["outcomes_sha256"] = digest(raw)
        self.reject_changed(lambda: None, "argv")

    def test_workspace_preflight_requires_real_success_scope_source_and_order(self):
        for mutate in [
            lambda row: row["phases"][0].update(exit_code=101),
            lambda row: row["phases"][0].update(argv=["cargo", "test", "--locked"]),
            lambda row: row.update(producer="cargo-mutants-original-parent-baseline"),
            lambda row: row.update(source_after_sha256="0" * 64),
            lambda row: row["phases"][1].update(started_at="2026-11-05T05:00:00Z"),
            lambda row: row.update(unmutated_workspace=False),
        ]:
            with self.subTest(mutate=mutate):
                row = copy.deepcopy(self.plan["workspace_preflight"])
                mutate(row)
                with self.assertRaises(ValueError):
                    sub.validate_preflight(row, self.parent, self.plan["identity"])

    def test_source_diff_candidate_or_manifest_digest_change_is_rejected(self):
        for key, value in [("source", {}), ("diff_sha256", "0" * 64), ("candidates", [])]:
            with self.subTest(key=key):
                nested = copy.deepcopy(self.nested)
                nested[key] = value
                with self.assertRaises(ValueError):
                    sub.check_plans(self.parent, nested)
        self.reject_changed(lambda: self.plan.update(nested_manifest_file_sha256="0" * 64), "digest")

    def test_partial_raw_even_with_rehashed_seal_is_rejected(self):
        seal = self.leaves[0]["seal"]
        seal["outcomes"]["outcomes"] = seal["outcomes"]["outcomes"][:1]
        seal["outcomes_sha256"] = digest(seal["outcomes"])
        self.reject_changed(lambda: None, "incomplete")

    def test_wrong_current_run_attempt_controller_or_artifact_fails(self):
        mutations = [
            lambda api: api["run"].update(head_sha="0" * 40),
            lambda api: api["run"].update(run_attempt=2),
            lambda api: api["run"].update(path="other.yml"),
            lambda api: api["jobs"]["jobs"][0].update(conclusion="failure"),
            lambda api: api["jobs"]["jobs"][0].update(completed_at="2026-11-05T06:09:00Z"),
            lambda api: api["artifacts"]["artifacts"][0].update(expired=True),
            lambda api: api["artifacts"]["artifacts"][0]["workflow_run"].update(head_sha="0" * 40),
            lambda api: api["artifacts"].update(total_count=999),
            lambda api: api["jobs"]["jobs"][-1]["steps"][0].update(conclusion="failure"),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                api = copy.deepcopy(self.api)
                mutate(api)
                with self.assertRaises(ValueError):
                    sub.make_envelope(self.parent, self.nested, self.plan, self.leaves, api,
                                      sub.json_file_sha(self.plan))

    def test_exit1_is_only_narrow_guard_not_fabricated_cargo_status(self):
        self.leaves[0]["attestation"].update(runner_exit_code=1, cargo_mutants_exit_code=None,
                                             exit1_reason="frozen-final-untracked-seed-guard")
        with self.assertRaisesRegex(ValueError, "fallback seed"):
            self.envelope()
        self.leaves[0]["seal"]["artifacts"] = {"generated_regressions_sha256": {
            "crates/example/tests/smoke.proptest-regressions": "c" * 64}}
        self.leaves[0]["attestation"]["seal_sha256"] = digest(self.leaves[0]["seal"])
        self.envelope()
        self.reject_changed(lambda: self.leaves[0]["attestation"].update(cargo_mutants_exit_code=2),
                            "completion status")

    def test_leaf_chronology_plan_digest_and_envelope_integrity(self):
        self.reject_changed(lambda: self.leaves[0]["attestation"].update(plan_file_sha256="0" * 64), "chronology")
        self.leaves[0]["attestation"]["plan_file_sha256"] = sub.json_file_sha(self.plan)
        envelope = self.envelope()
        envelope["leaves"][0]["seal"]["outcomes"]["missed"] = 0
        with self.assertRaisesRegex(ValueError, "checksum"):
            sub.validate_envelope(self.parent, envelope)

    def test_leaf_api_completion_uses_only_the_reported_precision(self):
        leaves = {leaf["seal"]["shard"]: leaf for leaf in self.leaves}
        info = self.leaves[0]["attestation"]
        job = self.api["jobs"]["jobs"][0]
        plan_sha = sub.json_file_sha(self.plan)
        info["attested_at"] = "2026-11-05T06:12:00.500Z"
        job["completed_at"] = "2026-11-05T06:12:00Z"
        sub.validate_execution(self.api, self.plan, leaves, plan_sha)
        info["attested_at"] = "2026-11-05T06:12:01Z"
        with self.assertRaisesRegex(ValueError, "leaf run/plan chronology"):
            sub.validate_execution(self.api, self.plan, leaves, plan_sha)
        info["attested_at"] = "2026-11-05T06:12:00.500Z"
        job["completed_at"] = "2026-11-05T06:12:00.750Z"
        sub.validate_execution(self.api, self.plan, leaves, plan_sha)
        job["completed_at"] = "2026-11-05T06:12:00.000Z"
        with self.assertRaisesRegex(ValueError, "leaf run/plan chronology"):
            sub.validate_execution(self.api, self.plan, leaves, plan_sha)

    def test_prepare_api_completion_uses_only_the_reported_precision(self):
        leaves = {leaf["seal"]["shard"]: leaf for leaf in self.leaves}
        job = self.api["jobs"]["jobs"][-1]
        def validate():
            plan_sha = sub.json_file_sha(self.plan)
            for leaf in self.leaves:
                leaf["attestation"]["plan_file_sha256"] = plan_sha
            sub.validate_execution(self.api, self.plan, leaves, plan_sha)
        self.plan["prepared_at"] = "2026-11-05T06:07:00.500Z"
        job["completed_at"] = "2026-11-05T06:07:00Z"
        validate()
        self.plan["prepared_at"] = "2026-11-05T06:07:01Z"
        with self.assertRaisesRegex(ValueError, "prepare API/preflight chronology"):
            validate()
        self.plan["prepared_at"] = "2026-11-05T06:07:00.500Z"
        job["completed_at"] = "2026-11-05T06:07:00.750Z"
        validate()
        job["completed_at"] = "2026-11-05T06:07:00.000Z"
        with self.assertRaisesRegex(ValueError, "prepare API/preflight chronology"):
            validate()

    def test_normal_cli_status_must_match_raw_missed_and_timeout_counts(self):
        self.reject_changed(lambda: self.leaves[0]["attestation"].update(
            runner_exit_code=0, cargo_mutants_exit_code=0), "contradicts")

    def test_attest_reads_real_raw_bytes_rechecks_source_and_emits_v1_leaf(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, directory = root / "source", root / "plan"
            directory.mkdir()
            self.plan["controller_tools_sha256"] = {
                name: file_digest(Path(sub.__file__).with_name(name)) for name in sub.CONTROLLER_TOOLS}
            write_new(directory / "nested-plan.json", self.plan)
            write_new(directory / "nested-campaign.json", self.nested)
            raw = source / "target/final-mutation-shard-20/mutants.out"
            raw.mkdir(parents=True)
            values = self.leaves[0]["seal"]["candidates"]
            data = self.leaves[0]["seal"]["outcomes"]
            write_new(raw / "mutants.json", values)
            write_new(raw / "outcomes.json", data)
            artifacts = {"directory": str(raw.resolve()), "mutants_json_sha256": file_digest(raw / "mutants.json"),
                         "outcomes_json_sha256": file_digest(raw / "outcomes.json"),
                         "generated_regressions_sha256": {}}
            seal = seal_shard(self.nested, "20/256", self.parent["source"], values, data, artifacts)
            evidence = source / "target/quality/shard-20"
            evidence.mkdir(parents=True)
            (evidence / "exit-code.txt").write_text("2\n")
            (evidence / "tee-exit-code.txt").write_text("0\n")
            for name in sub.LEAF_FILES:
                if not (evidence / name).exists():
                    (evidence / name).write_text("retained real runner bytes\r\n")
            write_new(source / "target/quality/shard-result-20.json", seal)
            output = root / "leaf.json"
            with patch.object(sub, "checked_inputs", return_value=(self.parent, self.nested, self.plan)), \
                    patch.object(sub, "verify_worktree") as guard, \
                    patch.object(sub, "seed_inventory", return_value={}), \
                    patch.object(sub, "utc", return_value="2026-11-05T06:12:00Z"):
                sub.attest_leaf(source, directory, "20/256", output, "plan-external", "manifest-external")
                self.assertEqual(guard.call_count, 2)
                self.assertTrue(all(call.kwargs["allow_regressions"] for call in guard.call_args_list))
            result = json.loads(output.read_bytes())
            self.assertEqual(result["seal"], seal)
            self.assertEqual(result["attestation"]["runner_exit_code"], 2)
            self.assertEqual(result["attestation"]["cargo_mutants_exit_code"], 2)
            retained = source / "target/quality/leaf-evidence.json"
            write_new(retained, result)
            sub.verify_downloaded_leaf(retained, result)
            (evidence / "exit-code.txt").write_text("0\n")
            with self.assertRaisesRegex(ValueError, "runner evidence changed"):
                sub.verify_downloaded_leaf(retained, result)
            result["attestation"]["evidence_files_sha256"]["exit-code.txt"] = file_digest(evidence / "exit-code.txt")
            with self.assertRaisesRegex(ValueError, "status contradicts"):
                sub.verify_downloaded_leaf(retained, result)
            (evidence / "exit-code.txt").write_text("2\n")
            # A rehashed partial raw report cannot obtain a new attestation.
            data["outcomes"] = data["outcomes"][:1]
            (raw / "outcomes.json").write_text(json.dumps(data))
            output.unlink()
            with patch.object(sub, "checked_inputs", return_value=(self.parent, self.nested, self.plan)), \
                    patch.object(sub, "verify_worktree"), patch.object(sub, "seed_inventory", return_value={}):
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    sub.attest_leaf(source, directory, "20/256", output, "plan-external", "manifest-external")
            self.assertFalse(output.exists())

    def test_preflight_records_only_actual_process_argv_logs_and_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calls = []
            def process(argv, **kwargs):
                calls.append(argv)
                kwargs["stdout"].write(("actual " + " ".join(argv) + "\r\n").encode())
                class Status:
                    returncode = 0
                return Status()
            with patch.dict(sub.os.environ, sub.ENVIRONMENT), \
                    patch.object(sub, "verify_worktree") as guard, \
                    patch.object(sub, "seed_inventory", return_value={}), \
                    patch.object(sub.subprocess, "run", side_effect=process):
                result = sub.preflight(root, self.parent, root / "final.diff", self.plan["identity"], root)
            self.assertEqual(calls, [sub.BUILD, sub.TEST])
            self.assertEqual(result["producer"], sub.PRODUCER)
            self.assertEqual(guard.call_count, 2)
            for row in result["phases"]:
                self.assertEqual(file_digest(root / row["log_file"]), row["log_sha256"])
                self.assertEqual(row["argv"], calls[0 if row["phase"] == "Build" else 1])
                self.assertEqual(row["exit_code"], 0)

    def test_failed_workspace_build_never_emits_successful_preflight_or_runs_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            class Failed:
                returncode = 101
            with patch.dict(sub.os.environ, sub.ENVIRONMENT), patch.object(sub, "verify_worktree"), \
                    patch.object(sub.subprocess, "run", return_value=Failed()) as process:
                with self.assertRaisesRegex(ValueError, "Build failed"):
                    sub.preflight(root, self.parent, root / "final.diff", self.plan["identity"], root)
            process.assert_called_once()
            failed = json.loads((root / "workspace-preflight-failed.json").read_bytes())
            self.assertEqual(failed["phases"][0]["exit_code"], 101)
            self.assertNotIn("source_after_sha256", failed)

    def test_origins_require_both_actual_shutdowns_and_preserve_log_bytes(self):
        records = origins()
        blob = LOG.replace("\n", "\r\n").encode()
        records["second"]["log_base64"] = base64.b64encode(blob).decode()
        records["second"]["log_sha256"] = hashlib.sha256(blob).hexdigest()
        sub.validate_origins(records)
        for mutate in [
            lambda row: row["api"]["run"].update(head_sha="0" * 40),
            lambda row: row["api"]["job"]["steps"][2].update(conclusion="success"),
            lambda row: row["api"]["artifacts"].update(total_count=1),
            lambda row: row.update(log_sha256="0" * 64),
        ]:
            bad = copy.deepcopy(records)
            mutate(bad["second"])
            bad["second"]["api_sha256"] = digest(bad["second"]["api"])
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                sub.validate_origins(bad)


def zipped(entries):
    blob = io.BytesIO()
    with zipfile.ZipFile(blob, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return blob.getvalue()


class ArtifactByteBoundary(unittest.TestCase):
    def blob(self, extra=None):
        leaf = {"seal": {"shard": "20/256", "artifacts": {"generated_regressions_sha256": {}}}}
        return zipped({"target/quality/leaf-evidence.json": json.dumps(leaf),
                       "target/final-mutation-shard-20/mutants.out/outcomes.json": "{}", **(extra or {})})

    def test_verified_zip_retains_original_bytes_and_only_evidence_paths(self):
        blob = self.blob()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "20"
            sub.retain_leaf_zip(blob, {"digest": "sha256:" + hashlib.sha256(blob).hexdigest()}, "20", root)
            self.assertEqual((root / "original-artifact.zip").read_bytes(), blob)
            self.assertTrue((root / "target/quality/leaf-evidence.json").is_file())
            with self.assertRaisesRegex(ValueError, "already exists"):
                sub.retain_leaf_zip(blob, {"digest": "sha256:" + hashlib.sha256(blob).hexdigest()}, "20", root)

    def test_zip_checksum_traversal_and_unlisted_source_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "20"
            with self.assertRaisesRegex(ValueError, "digest"):
                sub.retain_leaf_zip(self.blob(), {"digest": "sha256:" + "0" * 64}, "20", root)
            for path in ["../escape", "crates/runtime/src/lib.rs", "/absolute"]:
                blob = self.blob({path: "untrusted"})
                with self.subTest(path=path), self.assertRaises(ValueError):
                    sub.retain_leaf_zip(blob, {"digest": "sha256:" + hashlib.sha256(blob).hexdigest()}, "20", root)
            self.assertFalse(root.exists())

    def test_json_file_digest_is_exact_writer_bytes(self):
        value = {"unicode": "á", "nested": {"value": 1}}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "value.json"
            write_new(path, value)
            self.assertEqual(sub.json_file_sha(value), file_digest(path))

    def test_partition_arithmetic_is_exhaustive_without_authorizing_other_parent(self):
        self.assertEqual(sub.partition_labels(20, 64, 4), ("20/256", "84/256", "148/256", "212/256"))
        self.assertEqual(sub.partition_labels(27, 64, 4), ("27/256", "91/256", "155/256", "219/256"))
        for args in [(64, 64, 4), (-1, 64, 4), (20, 64, 0), (True, 64, 4)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                sub.partition_labels(*args)

    def test_every_consumer_cli_requires_both_independent_hashes(self):
        for argv in [
            ["verify", "--plan", "target/plan"],
            ["attest", "--plan", "target/plan", "--source", "source", "--shard", "20/256", "--output", "target/leaf"],
            ["assemble", "--plan", "target/plan", "--leaves", "target/leaves", "--output", "target/envelope"],
        ]:
            for missing in ("--plan-sha", "--manifest-sha"):
                present = "--manifest-sha" if missing == "--plan-sha" else "--plan-sha"
                stderr = io.StringIO()
                with self.subTest(command=argv[0], missing=missing), \
                        patch.object(sys, "argv", ["mutation_subshards.py", *argv, present, "a" * 64]), \
                        contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as stopped:
                    sub.main()
                self.assertEqual(stopped.exception.code, 2)
                self.assertIn(missing, stderr.getvalue())


class DiagnosticModes(unittest.TestCase):
    def setUp(self):
        self.parent, self.nested, self.plan, self.leaves, self.api = fixture()
        for name, value in [("ORIGINAL_MANIFEST_ID", self.parent["manifest_sha256"]), ("PARENT_COUNT", 4)]:
            context = patch.object(sub, name, value)
            context.start()
            self.addCleanup(context.stop)

    def diagnostic_plan(self):
        self.plan.update(mode="diagnostic84", scheduled_shards=["84/256"])
        return self.plan

    def test_legacy_none_and_explicit_full_preserve_all_four_leaves(self):
        for extra in [{}, {"mode": None}, {"mode": "full", "scheduled_shards": list(sub.LEAVES)}]:
            record = {**self.plan, **extra}
            before = sub.json_file_sha(record)
            sub.check_plan_record(record, self.parent, self.nested)
            self.assertEqual(sub.scheduled_leaves(record), sub.LEAVES)
            self.assertEqual(sub.json_file_sha(record), before)

    def test_invalid_or_partial_schedules_cannot_claim_full_or_legacy_mode(self):
        for extra in [
            {"mode": "unknown", "scheduled_shards": list(sub.LEAVES)},
            {"scheduled_shards": ["84/256"]},
            {"mode": "full", "scheduled_shards": ["84/256"]},
            {"mode": "diagnostic84", "scheduled_shards": list(sub.LEAVES)},
            {"mode": "diagnostic84", "scheduled_shards": ["84/256", "84/256"]},
            {"mode": "diagnostic84", "scheduled_shards": ["148/256"]},
        ]:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                sub.check_plan_record({**self.plan, **extra}, self.parent, self.nested)

    def test_diagnostic_leaf_may_be_attested_but_other_leaves_are_rejected(self):
        self.diagnostic_plan()
        sub.check_plan_record(self.plan, self.parent, self.nested)
        self.assertEqual(sub.scheduled_leaves(self.plan), ("84/256",))
        sub.validate_leaf(self.plan, self.nested, self.leaves[1])
        for leaf in [self.leaves[0], self.leaves[2], self.leaves[3]]:
            with self.subTest(shard=leaf["seal"]["shard"]), self.assertRaisesRegex(ValueError, "not scheduled"):
                sub.validate_leaf(self.plan, self.nested, leaf)
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(sub, "checked_inputs", return_value=(self.parent, self.nested, self.plan)), \
                patch.object(sub, "verify_worktree") as source_check:
            path = Path(temporary)
            with self.assertRaisesRegex(ValueError, "not scheduled"):
                sub.attest_leaf(path, path, "20/256", path / "leaf.json", "plan", "manifest")
            source_check.assert_not_called()
            self.assertFalse((path / "leaf.json").exists())

    def test_diagnostic_mode_never_assembles_or_validates_a_parent_envelope(self):
        self.diagnostic_plan()
        with self.assertRaisesRegex(ValueError, "diagnostic plan"):
            sub.make_envelope(self.parent, self.nested, self.plan, self.leaves, self.api,
                              sub.json_file_sha(self.plan))
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(sub, "checked_inputs", return_value=(self.parent, self.nested, self.plan)), \
                patch.object(sub.recovery, "api") as api:
            path = Path(temporary)
            with self.assertRaisesRegex(ValueError, "diagnostic plan"):
                sub.assemble(path, path, path / "parent.json", "plan", "manifest")
            api.assert_not_called()
            self.assertFalse((path / "parent.json").exists())

    def test_prepare_cli_defaults_full_and_forwards_only_the_selected_mode(self):
        arguments = ["prepare", "--source", "source", "--plan", "target/plan", "--run", "999999",
                     "--attempt", "1", "--controller", "6" * 40]
        for flags, expected in [([], "full"), (["--mode", "full"], "full"),
                                (["--mode", "diagnostic84"], "diagnostic84")]:
            with self.subTest(mode=expected), patch.object(sys, "argv", ["mutation_subshards.py", *arguments, *flags]), \
                    patch.object(sub, "prepare") as prepare:
                sub.main()
                self.assertEqual(prepare.call_args.args[-1], expected)

    def test_prepare_persists_mode_without_skipping_workspace_or_changing_frozen_argv(self):
        for mode, expected in [("full", list(sub.LEAVES)), ("diagnostic84", ["84/256"])]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source, directory = root / "source", root / "plan"
                source.mkdir()
                directory.mkdir()
                frozen = source / "tools/merge_mutation_shards.py"
                real_digest = sub.file_digest

                def checksum(path):
                    return self.parent["source"]["files"]["tools/merge_mutation_shards.py"] \
                        if Path(path) == frozen else real_digest(path)

                def listing(argv, **kwargs):
                    self.assertIn("256", argv)
                    self.assertNotIn("--mode", argv)
                    write_new(directory / "nested-campaign.json", self.nested)
                    kwargs["stdout"].write(b"frozen listing fixture\n")

                with patch.object(sub.recovery, "checked_plan", return_value=self.parent), \
                        patch.object(sub, "git", return_value=self.plan["identity"]["controller_revision"]), \
                        patch.object(sub, "file_digest", side_effect=checksum), \
                        patch.object(sub, "verify_worktree"), \
                        patch.object(sub, "collect_origins", return_value=origins()), \
                        patch.object(sub.subprocess, "run", side_effect=listing), \
                        patch.object(sub, "preflight", return_value=self.plan["workspace_preflight"]) as baseline, \
                        patch.object(sub, "utc", side_effect=["2026-11-05T06:00:00Z", "2026-11-05T06:06:00Z"]):
                    result = sub.prepare(source, directory, self.plan["identity"], mode)
                self.assertEqual(result["mode"], mode)
                self.assertEqual(result["scheduled_shards"], expected)
                self.assertEqual(json.loads((directory / "nested-plan.json").read_bytes()), result)
                baseline.assert_called_once()


class DiagnosticTelemetry(unittest.TestCase):
    def test_stdout_and_file_receive_the_same_flushed_diagnostic_json(self):
        class Output(io.StringIO):
            flushes = 0

            def flush(self):
                self.flushes += 1
                super().flush()

        output = Output()
        sample = {"at": "2026-11-05T07:00:00Z", "root_pid": 123, "processes": [],
                  "cgroup": {}, "disk_available_bytes": 10, "mutation_progress": None,
                  "diagnostic_only": True, "environ": "SECRET_DO_NOT_PRINT"}
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(Path, "exists", side_effect=[True, False]), \
                patch.object(sub, "diagnostic_sample", return_value=sample), \
                patch.object(sub.time, "sleep") as sleep, contextlib.redirect_stdout(output):
            path = Path(temporary) / "telemetry.jsonl"
            sub.telemetry(123, Path(temporary), path)
            self.assertEqual(path.read_text(), output.getvalue())
            self.assertTrue(json.loads(path.read_text())["diagnostic_only"])
            self.assertNotIn("SECRET", path.read_text())
            self.assertGreater(output.flushes, 0)
            sleep.assert_called_once_with(30)

    def test_subtree_includes_nonleader_thread_children_and_missing_metrics_are_null(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proc, cgroup, source = root / "proc", root / "cgroup", root / "source"
            source.mkdir()
            for pid in [123, 200]:
                (proc / str(pid)).mkdir(parents=True)
                (proc / str(pid) / "status").write_text("Name:\tSECRET_ARG\nState:\tS (sleeping)\nVmRSS:\t12 kB\n")
                (proc / str(pid) / "environ").write_text("SECRET_ENV")
                (proc / str(pid) / "cmdline").write_text("SECRET_CMD")
            for tid, children in [(123, ""), (124, "200")]:
                task = proc / "123/task" / str(tid)
                task.mkdir(parents=True)
                (task / "children").write_text(children)
            (proc / "123/cgroup").write_text("0::/job\n")
            (cgroup / "job").mkdir(parents=True)
            (cgroup / "job/memory.current").write_text("12345\n")
            sample = sub.diagnostic_sample(123, source, proc, cgroup)
            self.assertEqual({row["pid"] for row in sample["processes"]}, {123, 200})
            self.assertTrue(all(row["VmHWM"] is None for row in sample["processes"]))
            self.assertEqual(sample["cgroup"]["memory.current"], "12345")
            self.assertIsNone(sample["cgroup"]["memory.peak"])
            self.assertNotIn("SECRET", sub.diagnostic_json(sample))
            with patch.object(sub.os, "statvfs", side_effect=OSError("unavailable")):
                missing = sub.diagnostic_sample(999, source, proc, cgroup)
            self.assertIsNone(missing["disk_available_bytes"])
            self.assertTrue(all(value is None for value in missing["cgroup"].values()))

    def test_progress_is_observed_without_claiming_newest_log_is_active(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "target/final-mutation-shard-84/mutants.out"
            logs = raw / "log"
            logs.mkdir(parents=True)
            for name, data, timestamp in [("old.log", b"a", 1), ("new.log", b"abc", 2)]:
                path = logs / name
                path.write_bytes(data)
                os.utime(path, ns=(timestamp, timestamp))
            outcomes = raw / "outcomes.json"
            outcomes.write_text(json.dumps({"total_mutants": 1, "caught": 1, "missed": 0,
                                            "timeout": 0, "unviable": 0, "outcomes": [{}],
                                            "end_time": None, "env": "SECRET"}))
            progress = sub.mutation_progress(root)
            self.assertIsNone(progress["active_mutant"])
            self.assertTrue(progress["newest_log"]["path"].endswith("/new.log"))
            self.assertEqual(progress["newest_log"]["bytes"], 3)
            self.assertEqual(progress["raw_progress"]["caught"], 1)
            self.assertFalse(progress["raw_progress"]["raw_end_time_present"])
            self.assertNotIn("SECRET", json.dumps(progress))
            outcomes.write_text("partial JSON {")
            self.assertIsNone(sub.mutation_progress(root)["raw_progress"])
            outcomes.unlink()
            outside = root / "outside.json"
            outside.write_text('{"caught":999}')
            outcomes.symlink_to(outside)
            self.assertIsNone(sub.mutation_progress(root)["raw_progress"])
            (root / "target/final-mutation-shard-20/mutants.out").mkdir(parents=True)
            self.assertIsNone(sub.mutation_progress(root))

    def test_stdout_sample_is_bounded_and_unavailable_source_is_null(self):
        sample = {"at": "2026-11-05T07:00:00Z", "root_pid": 123, "cgroup": {},
                  "processes": [{"pid": index, "VmRSS": "x" * 1000} for index in range(256)]}
        line = sub.diagnostic_json(sample)
        self.assertLessEqual(len(line.encode()), 32768)
        self.assertGreater(json.loads(line)["processes_truncated"], 0)
        self.assertEqual(len(sample["processes"]), 256)
        with tempfile.TemporaryDirectory() as temporary, patch.object(Path, "glob", side_effect=OSError("lost")):
            self.assertIsNone(sub.mutation_progress(Path(temporary)))


if __name__ == "__main__":
    unittest.main()
