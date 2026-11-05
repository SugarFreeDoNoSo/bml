import copy
import hashlib
import io
import json
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import mutation_recovery as recovery
from merge_mutation_shards import file_digest, make_manifest, seal_shard
from test_merge_mutation_shards import campaign, candidate, report


JOB = 126992338567
LOG = ("2026-11-05T03:37:30Z ##[error]Process completed with exit code 143.\n"
       "2026-11-05T03:37:30Z ##[error]The runner has received a shutdown signal.\n")


def origin():
    return {
        "run": {"id": recovery.ORIGIN_RUN, "head_sha": recovery.SOURCE, "run_attempt": 1,
                "event": "workflow_dispatch", "path": ".github/workflows/mutation-campaign.yml",
                "repository": {"full_name": recovery.REPOSITORY, "id": recovery.REPOSITORY_ID}},
        "artifact": {"id": recovery.ARTIFACT_ID, "expired": False,
                     "name": f"mutation-plan-{recovery.ORIGIN_RUN}-1",
                     "workflow_run": {"id": recovery.ORIGIN_RUN, "head_sha": recovery.SOURCE,
                                      "repository_id": recovery.REPOSITORY_ID,
                                      "head_repository_id": recovery.REPOSITORY_ID}},
        "job": {"id": JOB, "run_id": recovery.ORIGIN_RUN, "run_attempt": 1,
                "head_sha": recovery.SOURCE, "name": "shard (20, 20/64)",
                "status": "completed", "conclusion": "failure", "steps": [
                    {"name": "Check independent prepare digest and frozen revision",
                     "conclusion": "success"},
                    {"name": "Run baseline and workspace mutation tests; seal completed outcomes",
                     "conclusion": "failure"},
                    {"name": "Preserve raw reports, candidates, logs, seals and generated regressions",
                     "conclusion": "skipped"}]},
        "artifacts": {"total_count": 1, "artifacts": [{"id": recovery.ARTIFACT_ID}]},
    }


class InterruptedShardOrigin(unittest.TestCase):
    def test_terminal_shutdown_of_matching_original_shard_is_eligible(self):
        recovery.validate_origin(recovery.REPOSITORY, JOB, 20, origin(), LOG)

    def test_wrong_source_repo_attempt_artifact_or_nonterminal_job_fails_closed(self):
        for section, key, value in [
            ("run", "head_sha", "f" * 40), ("run", "run_attempt", 2),
            ("run", "event", "pull_request"), ("artifact", "id", 1),
            ("artifact", "expired", True), ("artifact", "name", "other"),
            ("job", "run_id", 1), ("job", "head_sha", "f" * 40),
            ("job", "status", "in_progress"), ("job", "conclusion", "success"),
            ("job", "name", "shard (21, 21/64)"),
        ]:
            with self.subTest(section=section, key=key):
                bundle = origin()
                bundle[section][key] = value
                with self.assertRaises(ValueError):
                    recovery.validate_origin(recovery.REPOSITORY, JOB, 20, bundle, LOG)
        with self.assertRaises(ValueError):
            recovery.validate_origin("other/repo", JOB, 20, origin(), LOG)

    def test_exit_143_without_runner_shutdown_is_not_an_eligible_code_failure(self):
        with self.assertRaisesRegex(ValueError, "shutdown"):
            recovery.validate_origin(recovery.REPOSITORY, JOB, 20, origin(), LOG.splitlines()[0])

    def test_existing_shard_evidence_or_partial_inventory_requires_review(self):
        for partial in [False, True]:
            bundle = origin()
            bundle["artifacts"]["total_count"] = 2
            if not partial:
                bundle["artifacts"]["artifacts"].append({
                    "name": f"mutation-shard-{recovery.ORIGIN_RUN}-1-20"})
            with self.assertRaises(ValueError):
                recovery.validate_origin(recovery.REPOSITORY, JOB, 20, bundle, LOG)


class ActionApiLogCompatibility(unittest.TestCase):
    def test_recent_gh_opts_in_only_for_captured_raw_logs_and_preserves_bytes(self):
        payload = b"\x1b[36mlog\x1b[0m\r\n"
        with patch.object(recovery.subprocess, "check_output", side_effect=[
            "--allow-escape-sequences Allow terminal escape sequences", payload
        ]) as check:
            actual = recovery.api("repos/owner/repo/actions/jobs/1/logs", raw=True)
        self.assertEqual(actual.encode("utf-8"), payload)
        self.assertEqual(check.call_args_list[0].args[0], ["gh", "api", "--help"])
        self.assertEqual(check.call_args_list[1].args[0], [
            "gh", "api", "--allow-escape-sequences", "repos/owner/repo/actions/jobs/1/logs"])
        self.assertNotIn("text", check.call_args_list[1].kwargs)

    def test_older_gh_reads_captured_logs_without_an_unsupported_flag(self):
        with patch.object(recovery.subprocess, "check_output", side_effect=[
            "gh api <endpoint> [flags]", b"original log\n"
        ]) as check:
            self.assertEqual(recovery.api("logs", raw=True), "original log\n")
        self.assertEqual(check.call_args_list[1].args[0], ["gh", "api", "logs"])

    def test_json_api_does_not_relax_escape_handling_or_probe_help(self):
        with patch.object(recovery.subprocess, "check_output", return_value='{"id": 1}') as check:
            self.assertEqual(recovery.api("metadata"), {"id": 1})
        check.assert_called_once_with(["gh", "api", "metadata"], text=True)

    def test_binary_archives_remain_bytes_and_are_not_decoded_or_rendered(self):
        payload = b"PK\x03\x04\xff\x1b\r\n"
        with patch.object(recovery.subprocess, "check_output", side_effect=[
            "--allow-escape-sequences", payload
        ]) as check:
            self.assertEqual(recovery.api_bytes("artifact/zip"), payload)
        self.assertNotIn("text", check.call_args_list[-1].kwargs)


def inspection_zip(entries):
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in entries.items():
            archive.writestr(name, value)
    return data.getvalue()


class CollectionFailureEligibility(unittest.TestCase):
    index = 21

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        source = copy.deepcopy(campaign()[0]["source"])
        source["revision"] = recovery.SOURCE
        self.diff = b"original scope\n"
        candidates = [candidate(i) for i in range(64)]
        self.manifest = make_manifest(source, hashlib.sha256(self.diff).hexdigest(), candidates,
                                      [[value] for value in candidates])
        self.plan = json.dumps(self.manifest).encode()
        self.plan_zip = inspection_zip({"campaign.json": self.plan, "final.diff": self.diff})
        self.job_id, self.artifact_id, _ = recovery.collection_failure(self.index)
        self.zip_constant = "COLLECTION_ZIP_SHA256" if self.index == 21 else "COLLECTION_27_ZIP_SHA256"
        self.base = f"target/final-mutation-shard-{self.index}/mutants.out/"
        self.evidence = f"target/quality/shard-{self.index}/"
        values = self.manifest["shards"][self.index]["candidates"]
        self.outcomes = report(values, ["MissedMutant"])
        self.entries = {
            self.base + "mutants.json": json.dumps(values),
            self.base + "outcomes.json": json.dumps(self.outcomes),
            self.evidence + "runner.log": recovery.UNTRACKED_ERROR + "\n",
            self.evidence + "exit-code.txt": "1\n"}
        self.archive = inspection_zip(self.entries)
        for name, value in [("MANIFEST_SHA256", hashlib.sha256(self.plan).hexdigest()),
                            (self.zip_constant, hashlib.sha256(self.archive).hexdigest())]:
            patcher = patch.object(recovery, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def bundle(self):
        bundle = origin()
        bundle["job"].update(id=self.job_id, name=f"shard ({self.index}, {self.index}/64)")
        bundle["job"]["steps"][-1]["conclusion"] = "success"
        bundle["collection_artifact"] = {
            "id": self.artifact_id, "name": f"mutation-shard-{recovery.ORIGIN_RUN}-1-{self.index}",
            "expired": False, "digest": "sha256:" + recovery.collection_failure(self.index)[2],
            "size_in_bytes": len(self.archive),
            "workflow_run": copy.deepcopy(bundle["artifact"]["workflow_run"])}
        bundle["artifacts"]["artifacts"].append(copy.deepcopy(bundle["collection_artifact"]))
        bundle["artifacts"]["total_count"] = 2
        return bundle

    def test_complete_original_collection_is_only_retry_eligibility(self):
        manifest = recovery.inspection_plan(self.plan_zip, self.index)
        inspected = recovery.inspect_collection_zip(self.archive, manifest, self.index)
        eligibility = recovery.validate_origin(recovery.REPOSITORY, self.job_id, self.index,
            self.bundle(), "2026-11-05T03:51:55Z ##[error]Process completed with exit code 1.\n",
            inspected)
        self.assertEqual(eligibility["state"], "retry_eligible_from_collection_failure")
        self.assertTrue(inspected["fresh_execution_required"])
        self.assertIn("unattested", inspected["source_state"])
        self.assertEqual(inspected["raw_counts"]["missed"], 1)

    def test_zip_and_plan_digest_must_match_before_inspection(self):
        with self.assertRaisesRegex(ValueError, "ZIP digest"):
            recovery.inspect_collection_zip(self.archive + b"tampered", self.manifest, self.index)
        changed_plan = inspection_zip({"campaign.json": self.plan + b" ", "final.diff": self.diff})
        with self.assertRaisesRegex(ValueError, "manifest file digest"):
            recovery.inspection_plan(changed_plan, self.index)

    def test_reauthenticated_wrong_guard_incomplete_raw_wrong_candidates_or_seal_fail(self):
        for name in ["guard", "incomplete", "candidate", "seal", "controller-seal", "baseline"]:
            entries = dict(self.entries)
            if name == "guard":
                entries[self.evidence + "runner.log"] = "another failure\n"
            elif name in {"incomplete", "baseline"}:
                outcomes = copy.deepcopy(self.outcomes)
                if name == "incomplete":
                    outcomes["end_time"] = None
                else:
                    outcomes["outcomes"][0]["summary"] = "Failure"
                entries[self.base + "outcomes.json"] = json.dumps(outcomes)
            elif name == "candidate":
                entries[self.base + "mutants.json"] = json.dumps([candidate(22)])
            elif name == "controller-seal":
                entries[f"target/quality/controller-shard-result-{self.index}.json"] = "{}"
            else:
                entries[f"target/quality/shard-result-{self.index}.json"] = "{}"
            blob = inspection_zip(entries)
            with self.subTest(name=name), patch.object(recovery, self.zip_constant,
                    hashlib.sha256(blob).hexdigest()), self.assertRaises(ValueError):
                recovery.inspect_collection_zip(blob, self.manifest, self.index)

    def test_collection_metadata_cannot_select_another_source_or_artifact(self):
        inspection = recovery.inspect_collection_zip(self.archive, self.manifest, self.index)
        for key, value in [("id", 1), ("digest", "sha256:" + "0" * 64),
                           ("expired", True), ("name", "other")]:
            bundle = self.bundle()
            bundle["collection_artifact"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                recovery.validate_origin(recovery.REPOSITORY, self.job_id, self.index, bundle,
                    "2026-11-05T03:51:55Z ##[error]Process completed with exit code 1.\n", inspection)
        bundle = self.bundle()
        bundle["collection_artifact"]["workflow_run"]["head_sha"] = "f" * 40
        with self.assertRaises(ValueError):
            recovery.validate_origin(recovery.REPOSITORY, self.job_id, self.index, bundle,
                "2026-11-05T03:51:55Z ##[error]Process completed with exit code 1.\n", inspection)

    def test_reviewed_collection_pairs_and_inspections_cannot_be_interchanged(self):
        self.assertTrue(recovery.collection_requested(self.job_id, self.index))
        other = 27 if self.index == 21 else 21
        other_job, other_artifact, other_sha = recovery.collection_failure(other)
        for job, index in [(self.job_id, other), (other_job, self.index),
                           (self.job_id, 26), (self.job_id, float(self.index))]:
            with self.subTest(job=job, index=index):
                self.assertFalse(recovery.collection_requested(job, index))
        inspected = recovery.inspect_collection_zip(self.archive, self.manifest, self.index)
        for key, value in [("artifact_id", other_artifact), ("zip_sha256", other_sha)]:
            wrong = {**inspected, key: value}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "inspection"):
                recovery.validate_origin(recovery.REPOSITORY, self.job_id, self.index,
                    self.bundle(), "2026-11-05T03:51:55Z ##[error]Process completed with exit code 1.\n",
                    wrong)
        other_constant = "COLLECTION_ZIP_SHA256" if other == 21 else "COLLECTION_27_ZIP_SHA256"
        with patch.object(recovery, other_constant, hashlib.sha256(self.archive).hexdigest()), \
                self.assertRaisesRegex(ValueError, "lacks required evidence"):
            recovery.inspect_collection_zip(self.archive, self.manifest, other)
        with self.assertRaisesRegex(ValueError, "not independently reviewed"):
            recovery.inspection_plan(self.plan_zip, 26)

    def test_default_inspection_api_remains_shard21_only(self):
        self.assertEqual(recovery.inspection_plan(self.plan_zip), self.manifest)
        if self.index == 21:
            self.assertEqual(recovery.inspect_collection_zip(self.archive, self.manifest),
                             recovery.inspect_collection_zip(self.archive, self.manifest, 21))
        else:
            with self.assertRaisesRegex(ValueError, "ZIP digest"):
                recovery.inspect_collection_zip(self.archive, self.manifest)

    def test_authorization_keeps_archives_in_control_without_extracting_or_sealing(self):
        bundle = self.bundle()
        def api(path, raw=False):
            if raw:
                return "2026-11-05T03:51:55Z ##[error]Process completed with exit code 1.\n"
            if path.endswith(f"artifacts/{self.artifact_id}"):
                return bundle["collection_artifact"]
            if "/jobs/" in path:
                return bundle["job"]
            if path.endswith("artifacts?per_page=100"):
                return bundle["artifacts"]
            if path.endswith(f"artifacts/{recovery.ARTIFACT_ID}"):
                return bundle["artifact"]
            return bundle["run"]
        output = self.root / "control/target/recovery"
        with patch.object(recovery, "api", side_effect=api), \
                patch.object(recovery, "api_bytes", side_effect=[self.plan_zip, self.archive]):
            recovery.authorize(recovery.REPOSITORY, self.job_id, self.index, 1234, 1,
                               "a" * 40, output)
        record = recovery.authorization(output)
        self.assertEqual(record["eligibility_state"], "retry_eligible_from_collection_failure")
        self.assertEqual(record["origin_exit_code"], 1)
        self.assertEqual(record["index"], self.index)
        self.assertEqual(record["collection_inspection"]["artifact_id"], self.artifact_id)
        self.assertEqual((output / "original-collection.zip").read_bytes(), self.archive)
        self.assertFalse((self.root / "source").exists())
        self.assertEqual(list(self.root.rglob("*shard-result*.json")), [])
        self.assertEqual(list(self.root.rglob("outcomes.json")), [])
        (output / "original-collection.zip").write_bytes(self.archive + b"changed")
        with self.assertRaisesRegex(ValueError, "archives changed"):
            recovery.authorization(output)

    def test_zip_inspection_rejects_traversal_absolute_and_symlink_entries(self):
        for name in ["../escape.rs", "/absolute.rs", "dir\\escape.rs"]:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "unsafe"):
                recovery.zip_members(inspection_zip({name: "data"}), set())
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w") as archive:
            link = zipfile.ZipInfo("link")
            link.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(link, "../../outside")
        with self.assertRaisesRegex(ValueError, "unsafe"):
            recovery.zip_members(data.getvalue(), set())


class CollectionFailure27Eligibility(CollectionFailureEligibility):
    index = 27


class FreshFailureEligibility(unittest.TestCase):
    index = 46
    job = 126992343007
    artifact_id = 13392074343

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        source = copy.deepcopy(campaign()[0]["source"])
        source["revision"] = recovery.SOURCE
        self.diff = b"original scope\n"
        candidates = [candidate(i) for i in range(64)]
        self.manifest = make_manifest(source, hashlib.sha256(self.diff).hexdigest(), candidates,
                                      [[value] for value in candidates])
        self.plan = json.dumps(self.manifest).encode()
        self.plan_zip = inspection_zip({"campaign.json": self.plan, "final.diff": self.diff})
        self.base = f"target/final-mutation-shard-{self.index}/mutants.out/"
        self.evidence = f"target/quality/shard-{self.index}/"
        values = self.manifest["shards"][self.index]["candidates"]
        self.outcomes = report(values, ["MissedMutant"])
        self.entries = {self.base + "mutants.json": json.dumps(values),
                        self.base + "outcomes.json": json.dumps(self.outcomes),
                        self.evidence + "runner.log": recovery.UNTRACKED_ERROR + "\n",
                        self.evidence + "exit-code.txt": "1\n"}
        self.archive = inspection_zip(self.entries)
        patcher = patch.object(recovery, "MANIFEST_SHA256", hashlib.sha256(self.plan).hexdigest())
        patcher.start()
        self.addCleanup(patcher.stop)

    def bundle(self, collection=False):
        value = origin()
        value["job"].update(id=self.job, name=f"shard ({self.index}, {self.index}/64)")
        if collection:
            value["job"]["steps"][-1]["conclusion"] = "success"
            value["collection_artifact"] = {
                "id": self.artifact_id, "name": f"mutation-shard-{recovery.ORIGIN_RUN}-1-{self.index}",
                "expired": False, "digest": "sha256:" + hashlib.sha256(self.archive).hexdigest(),
                "size_in_bytes": len(self.archive), "workflow_run": copy.deepcopy(value["artifact"]["workflow_run"])}
            value["artifacts"]["artifacts"].append(copy.deepcopy(value["collection_artifact"]))
            value["artifacts"]["total_count"] = 2
        return value

    def log(self):
        return {"state": "present", "bytes": b"2026-11-05T06:00:00Z ##[error]Process completed with exit code 1.\r\n"}

    def missing_log(self):
        return {"state": "absent_http404", "returncode": 1,
                "stdout": b'{"message":"Not Found","status":"404"}',
                "stderr": b"gh: Not Found (HTTP 404)\n"}

    def authorize(self, bundle, log):
        def api(path, raw=False):
            self.assertFalse(raw)
            if path.endswith(f"artifacts/{self.artifact_id}"):
                return bundle["collection_artifact"]
            if "/jobs/" in path:
                return bundle["job"]
            if path.endswith("artifacts?per_page=100"):
                return bundle["artifacts"]
            return bundle["artifact"] if path.endswith(f"artifacts/{recovery.ARTIFACT_ID}") else bundle["run"]
        output = self.root / "control/target/recovery"
        with patch.object(recovery, "api", side_effect=api), \
                patch.object(recovery, "fetch_original_log", return_value=log), \
                patch.object(recovery, "api_bytes", side_effect=[self.plan_zip, self.archive]):
            recovery.authorize(recovery.REPOSITORY, self.job, self.index, 1234, 1, "a" * 40, output)
        return output

    def test_terminal_no_artifact_does_not_fabricate_finished_steps_or_exit(self):
        bundle = self.bundle()
        bundle["job"]["steps"][-2].update(status="in_progress", conclusion=None)
        bundle["job"]["steps"][-1]["conclusion"] = None
        before = copy.deepcopy(bundle)
        result = recovery.validate_fresh_origin(recovery.REPOSITORY, self.job, self.index, bundle, self.missing_log())
        self.assertEqual(result, {"state": "retry_eligible_from_terminal_failure_without_artifact", "exit_code": None})
        self.assertEqual(bundle, before)

    def test_success_wrong_source_repo_attempt_job_and_legacy_index_fail_closed(self):
        for section, key, value in [
            ("run", "id", 2), ("run", "head_sha", "f" * 40), ("run", "run_attempt", 2),
            ("run", "run_attempt", True),
            ("run", "event", "push"), ("run", "path", "other.yml"),
            ("artifact", "expired", True), ("artifact", "id", 2), ("artifact", "name", "other"),
            ("job", "id", 2), ("job", "run_id", 2), ("job", "head_sha", "f" * 40),
            ("job", "run_attempt", 2), ("job", "name", "shard (45, 45/64)"),
            ("job", "run_attempt", True),
            ("job", "status", "in_progress"), ("job", "conclusion", "success")]:
            bundle = self.bundle()
            bundle[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                recovery.validate_fresh_origin(recovery.REPOSITORY, self.job, self.index, bundle, self.log())
        for index in [20, 21, 27, -1, 64, True, 46.0]:
            with self.subTest(index=index), self.assertRaises(ValueError):
                recovery.validate_fresh_identity(recovery.REPOSITORY, self.job, index, self.bundle())
        for repository in ["other/repo", "sugarfreedonoso/bml"]:
            with self.assertRaises(ValueError):
                recovery.validate_fresh_identity(repository, self.job, self.index, self.bundle())
        for section in ["run", "artifact"]:
            bundle = self.bundle()
            parent = bundle[section]["repository"] if section == "run" else bundle[section]["workflow_run"]
            parent["id" if section == "run" else "repository_id"] = 2
            with self.assertRaises(ValueError):
                recovery.validate_fresh_identity(recovery.REPOSITORY, self.job, self.index, bundle)

    def test_incomplete_duplicate_ambiguous_inventory_or_existing_uninspected_artifact_fails(self):
        cases = []
        partial = self.bundle()
        partial["artifacts"]["total_count"] = 2
        cases.append(partial)
        duplicate = self.bundle()
        duplicate["artifacts"]["artifacts"] *= 2
        duplicate["artifacts"]["total_count"] = 2
        cases.append(duplicate)
        ambiguous = self.bundle(True)
        other = {**ambiguous["collection_artifact"], "id": self.artifact_id + 1}
        ambiguous["artifacts"]["artifacts"].append(other)
        ambiguous["artifacts"]["total_count"] = 3
        cases.extend([ambiguous, self.bundle(True)])
        for bundle in cases:
            with self.subTest(bundle=bundle), self.assertRaises(ValueError):
                recovery.validate_fresh_origin(recovery.REPOSITORY, self.job, self.index, bundle, self.log())

    def test_missing_log_roundtrip_retains_real_404_bytes_and_no_fake_log(self):
        output = self.authorize(self.bundle(), self.missing_log())
        record = recovery.authorization(output)
        self.assertEqual(record["format_version"], 2)
        self.assertEqual(record["origin_log_state"], "absent_http404")
        self.assertIsNone(record["origin_exit_code"])
        self.assertIsNone(record["origin_log_sha256"])
        self.assertFalse((output / "origin-job.log").exists())
        self.assertEqual((output / "origin-job-log-fetch.stdout").read_bytes(), self.missing_log()["stdout"])
        self.assertEqual((output / "origin-job-log-fetch.stderr").read_bytes(), self.missing_log()["stderr"])
        (output / "origin-job-log-fetch.stderr").write_bytes(b"gh: HTTP 403\n")
        with self.assertRaisesRegex(ValueError, "bytes changed"):
            recovery.authorization(output)

    def test_present_log_roundtrip_keeps_crlf_and_requires_actual_logged_exit(self):
        log = {"state": "present", "bytes": LOG.replace("\n", "\r\n").encode()}
        output = self.authorize(self.bundle(), log)
        record = recovery.authorization(output)
        self.assertEqual(record["origin_exit_code"], 143)
        self.assertEqual((output / "origin-job.log").read_bytes(), log["bytes"])
        self.assertEqual(record["origin_log_state"], "present")
        self.assertFalse((output / "origin-job-log-fetch.json").exists())

    def test_rehashed_404_proof_cannot_hide_403_or_fabricate_an_origin_log(self):
        output = self.authorize(self.bundle(), self.missing_log())
        stderr = output / "origin-job-log-fetch.stderr"
        stderr.write_bytes(b"gh: HTTP 403\n")
        proof = json.loads((output / "origin-job-log-fetch.json").read_text())
        proof["stderr_sha256"] = file_digest(stderr)
        (output / "origin-job-log-fetch.json").write_text(json.dumps(proof))
        record = json.loads((output / "authorization.json").read_text())
        record["origin_log_fetch_sha256"] = file_digest(output / "origin-job-log-fetch.json")
        (output / "authorization.json").write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "authenticated HTTP 404"):
            recovery.authorization(output)

    def test_api_failure_never_creates_generic_authorization(self):
        output = self.root / "failed-authorization"
        with patch.object(recovery, "fetch_original_log", side_effect=subprocess.CalledProcessError(
                1, ["gh", "api", "logs"], output=b"error", stderr=b"gh: HTTP 401\n")), \
                self.assertRaises(subprocess.CalledProcessError):
            recovery.authorize_fresh(recovery.REPOSITORY, self.job, self.index, 1234, 1,
                                     "a" * 40, output, self.bundle())
        self.assertFalse(output.exists())

    def test_api_bound_collection_roundtrip_is_eligibility_only_and_preserves_missed(self):
        output = self.authorize(self.bundle(True), self.log())
        record = recovery.authorization(output)
        self.assertEqual(record["eligibility_state"], "retry_eligible_from_api_bound_collection_failure")
        inspection = record["collection_inspection"]
        self.assertEqual(inspection["raw_counts"]["missed"], 1)
        self.assertIn("unattested", inspection["source_state"])
        self.assertTrue(record["fresh_execution_required"])
        self.assertEqual((output / "original-collection.zip").read_bytes(), self.archive)
        self.assertFalse((self.root / "source").exists())
        self.assertEqual(list(self.root.rglob("outcomes.json")), [])
        self.assertEqual(list(self.root.rglob("*shard-result*.json")), [])
        (output / "original-collection.zip").write_bytes(self.archive + b"tampered")
        with self.assertRaisesRegex(ValueError, "archives changed"):
            recovery.authorization(output)

    def test_reauthenticated_partial_failed_baseline_wrong_candidates_or_any_seal_fails(self):
        for case in ["guard", "incomplete", "baseline", "candidate", "seal", "root-seal", "controller-seal", "exit"]:
            entries = dict(self.entries)
            if case == "guard":
                entries[self.evidence + "runner.log"] += "another error\n"
            elif case in {"incomplete", "baseline"}:
                outcomes = copy.deepcopy(self.outcomes)
                if case == "incomplete":
                    outcomes["end_time"] = None
                else:
                    outcomes["outcomes"][0]["summary"] = "Failure"
                entries[self.base + "outcomes.json"] = json.dumps(outcomes)
            elif case == "candidate":
                entries[self.base + "mutants.json"] = json.dumps([candidate(3)])
            elif case == "exit":
                entries[self.evidence + "exit-code.txt"] = "2\n"
            else:
                prefix = "" if case == "root-seal" else "target/quality/"
                stem = "controller-shard-result" if case == "controller-seal" else "shard-result"
                entries[f"{prefix}{stem}-{self.index}.json"] = "{}"
            blob = inspection_zip(entries)
            artifact = {**self.bundle(True)["collection_artifact"], "digest": "sha256:" + hashlib.sha256(blob).hexdigest(),
                        "size_in_bytes": len(blob)}
            with self.subTest(case=case), self.assertRaises(ValueError):
                recovery.inspect_fresh_collection(blob, self.manifest, self.index, artifact)

    def test_collection_requires_exact_api_inventory_digest_source_upload_and_present_log(self):
        bundle = self.bundle(True)
        artifact = bundle["collection_artifact"]
        inspected = recovery.inspect_fresh_collection(self.archive, self.manifest, self.index, artifact)
        for key, value in [("id", 1), ("expired", True), ("digest", "sha256:" + "0" * 64),
                           ("name", "other"), ("size_in_bytes", len(self.archive) + 1)]:
            changed = copy.deepcopy(bundle)
            changed["collection_artifact"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                recovery.validate_fresh_origin(recovery.REPOSITORY, self.job, self.index, changed, self.log(), inspected)
        for case in ["wrong-source", "upload", "duplicate-step", "missing-log"]:
            changed = copy.deepcopy(bundle)
            log = self.log()
            if case == "wrong-source":
                for item in [changed["collection_artifact"], changed["artifacts"]["artifacts"][-1]]:
                    item["workflow_run"]["head_sha"] = "f" * 40
            elif case == "upload":
                changed["job"]["steps"][-1]["conclusion"] = "failure"
            elif case == "duplicate-step":
                changed["job"]["steps"].append(copy.deepcopy(changed["job"]["steps"][-1]))
            else:
                log = self.missing_log()
            with self.subTest(case=case), self.assertRaises(ValueError):
                recovery.validate_fresh_origin(recovery.REPOSITORY, self.job, self.index, changed, log, inspected)
        for blob in [self.archive + b"changed", self.archive[:-1]]:
            with self.assertRaisesRegex(ValueError, "size or digest"):
                recovery.inspect_fresh_collection(blob, self.manifest, self.index, artifact)

    def test_tampered_v2_identity_or_downgrade_cannot_authorize(self):
        output = self.authorize(self.bundle(), self.missing_log())
        original = json.loads((output / "authorization.json").read_text())
        for key, value in [("format_version", 1), ("eligibility_state", "retry_eligible_from_runner_shutdown"),
                           ("origin_exit_code", 143), ("fresh_execution_required", False),
                           ("origin_attempt", 2), ("index", 21), ("recovery_run_id", recovery.ORIGIN_RUN)]:
            record = {**original, key: value}
            (output / "authorization.json").write_text(json.dumps(record))
            with self.subTest(key=key), self.assertRaises(ValueError):
                recovery.authorization(output)


class MissingLogResponses(unittest.TestCase):
    def test_both_real_404_shapes_are_preserved_and_no_other_error_is_hidden(self):
        responses = [(b'{"message":"Not Found","status":"404"}', b"gh: Not Found (HTTP 404)\n"),
                     (b'\xef\xbb\xbf<?xml version="1.0" encoding="utf-8"?><Error><Code>BlobNotFound</Code>'
                      b'<Message>The specified blob does not exist.\nRequestId:actual\nTime:actual</Message></Error>',
                      b"gh: HTTP 404\n")]
        for stdout, stderr in responses:
            error = subprocess.CalledProcessError(1, ["gh", "api", "logs"], output=stdout, stderr=stderr)
            with self.subTest(stderr=stderr), patch.object(recovery.subprocess, "check_output", side_effect=[
                    "--allow-escape-sequences", error]) as check:
                result = recovery.fetch_original_log("logs")
            self.assertEqual(result, {"state": "absent_http404", "stdout": stdout, "stderr": stderr, "returncode": 1})
            self.assertEqual(check.call_args.args[0], ["gh", "api", "--allow-escape-sequences", "logs"])
            self.assertEqual(check.call_args.kwargs, {"stderr": subprocess.PIPE})
        for code, stdout, stderr in [
            (1, b'{"message":"Not Found","status":"404"}', b"gh: HTTP 403\n"),
            (1, b'{"message":"Not Found","status":"401"}', b"gh: Not Found (HTTP 404)\n"),
            (1, b"", b"gh: HTTP 404\n"), (1, b"garbage", b"gh: HTTP 404\n"),
            (1, responses[1][0], b"gh: HTTP 500\n"), (1, responses[0][0], b"gh: HTTP 401\n"),
            (1, responses[1][0].replace(b"BlobNotFound", b"AuthenticationFailed"), b"gh: HTTP 404\n"),
            (1, b'<!DOCTYPE Error>' + responses[1][0], b"gh: HTTP 404\n"),
            (1, responses[0][0], b"gh: Not Found (HTTP 404)\nERROR other\n"),
            (2, responses[0][0], responses[0][1]),
        ]:
            error = subprocess.CalledProcessError(code, ["gh", "api", "logs"], output=stdout, stderr=stderr)
            with self.subTest(code=code, stderr=stderr), patch.object(recovery.subprocess, "check_output", side_effect=["help", error]), \
                    self.assertRaises(subprocess.CalledProcessError):
                recovery.fetch_original_log("logs")


class RecoveredShardEvidence(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.evidence = self.root / "evidence"
        self.source = self.root / "source"
        self.plan = self.root / "campaign.json"
        self.diff = self.root / "final.diff"
        self.diff.write_text("complete original scope\n")
        frozen = copy.deepcopy(campaign()[0]["source"])
        frozen["revision"] = recovery.SOURCE
        frozen["files"]["tools/merge_mutation_shards.py"] = "6" * 64
        candidates = [candidate(i) for i in range(64)]
        self.manifest = make_manifest(frozen, file_digest(self.diff), candidates,
                                      [[value] for value in candidates])
        self.plan.write_text(json.dumps(self.manifest))
        self.patch = patch.object(recovery, "MANIFEST_SHA256", file_digest(self.plan))
        self.patch.start()
        self.addCleanup(self.patch.stop)
        bundle = origin()

        def api(path, raw=False):
            if raw:
                return LOG
            if "/jobs/" in path:
                return bundle["job"]
            if path.endswith("artifacts?per_page=100"):
                return bundle["artifacts"]
            if f"/artifacts/{recovery.ARTIFACT_ID}" in path:
                return bundle["artifact"]
            return bundle["run"]

        with patch.object(recovery, "api", side_effect=api):
            recovery.authorize(recovery.REPOSITORY, JOB, 20, 999, 1, "a" * 40, self.evidence)

    def test_crlf_origin_log_keeps_exact_saved_bytes_and_digest(self):
        bundle = origin()
        output = self.root / "crlf-evidence"
        text = LOG.replace("\n", "\r\n")
        payload = text.encode("utf-8")
        with patch.object(recovery, "api", side_effect=[
            bundle["run"], bundle["artifact"], bundle["job"], bundle["artifacts"], text
        ]):
            recovery.authorize(recovery.REPOSITORY, JOB, 20, 1000, 1, "a" * 40, output)
        self.assertEqual((output / "origin-job.log").read_bytes(), payload)
        record = recovery.authorization(output)
        self.assertEqual(record["origin_log_sha256"], hashlib.sha256(payload).hexdigest())

    def sealed(self, summary="CaughtMutant", code=0):
        values = self.manifest["shards"][20]["candidates"]
        outcomes = report(values, [summary])
        raw = self.source / "target/final-mutation-shard-20/mutants.out"
        raw.mkdir(parents=True)
        (raw / "mutants.json").write_text(json.dumps(values))
        (raw / "outcomes.json").write_text(json.dumps(outcomes))
        artifacts = {"mutants_json_sha256": file_digest(raw / "mutants.json"),
                     "outcomes_json_sha256": file_digest(raw / "outcomes.json")}
        result = seal_shard(self.manifest, "20/64", self.manifest["source"], values,
                            outcomes, artifacts)
        quality = self.source / "target/quality"
        (quality / "shard-20").mkdir(parents=True)
        (quality / "shard-20/exit-code.txt").write_text(str(code))
        (quality / "shard-result-20.json").write_text(json.dumps(result))
        (quality / "shard-20/final-source-seal.json").write_text(json.dumps(result))
        recovery.attest(self.evidence, self.plan, self.diff, self.source)
        return raw

    def test_external_manifest_and_diff_hashes_are_required(self):
        recovery.verify_plan(self.evidence, self.plan, self.diff)
        self.plan.write_text(json.dumps(self.manifest, indent=2))
        with self.assertRaisesRegex(ValueError, "manifest file digest"):
            recovery.verify_plan(self.evidence, self.plan, self.diff)

    def test_changed_scope_or_origin_proof_is_rejected(self):
        self.diff.write_text("smaller scope")
        with self.assertRaisesRegex(ValueError, "scope diff"):
            recovery.verify_plan(self.evidence, self.plan, self.diff)
        (self.evidence / "origin-job.log").write_text("replacement")
        with self.assertRaisesRegex(ValueError, "evidence changed"):
            recovery.authorization(self.evidence)

    def test_original_run_cannot_be_reused_as_the_recovery_run(self):
        with self.assertRaisesRegex(ValueError, "separate workflow"):
            recovery.authorize(recovery.REPOSITORY, JOB, 20, recovery.ORIGIN_RUN, 2,
                               "a" * 40, self.root / "bad")

    def test_raw_survivor_and_exit_2_are_preserved_without_a_quality_approval(self):
        raw = self.sealed("MissedMutant", 2)
        recovery.finish(self.evidence, self.plan, self.diff, self.source)
        metadata = json.loads((self.evidence / "recovery-metadata.json").read_text())
        self.assertEqual(metadata["status"], "complete_sealed_shard")
        self.assertEqual(metadata["cargo_mutants_exit_code"], 2)
        self.assertEqual(json.loads((raw / "outcomes.json").read_text())["missed"], 1)
        self.assertIn("not evaluated", metadata["quality_gate"])
        self.assertEqual(metadata["authorization"]["origin_job_id"], JOB)

    def test_timeout_and_exit_3_remain_timeout_without_a_quality_approval(self):
        raw = self.sealed("Timeout", 3)
        recovery.finish(self.evidence, self.plan, self.diff, self.source)
        metadata = json.loads((self.evidence / "recovery-metadata.json").read_text())
        self.assertEqual(metadata["status"], "complete_sealed_shard")
        self.assertEqual(metadata["cargo_mutants_exit_code"], 3)
        self.assertEqual(json.loads((raw / "outcomes.json").read_text())["timeout"], 1)

    def test_final_source_recheck_disagreement_cannot_be_complete(self):
        self.sealed()
        (self.source / "target/quality/shard-20/final-source-seal.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "final frozen source recheck"):
            recovery.finish(self.evidence, self.plan, self.diff, self.source)
        self.assertEqual(json.loads((self.evidence / "recovery-metadata.json").read_text())[
            "status"], "incomplete")

    def test_api_permission_failure_cannot_create_authorization(self):
        output = self.root / "unauthorized"
        with patch.object(recovery, "api", side_effect=subprocess.CalledProcessError(1, "gh")):
            with self.assertRaises(subprocess.CalledProcessError):
                recovery.authorize(recovery.REPOSITORY, JOB, 20, 999, 1, "a" * 40, output)
        self.assertFalse((output / "authorization.json").exists())

    def test_tampered_raw_reports_cannot_be_called_complete(self):
        raw = self.sealed()
        (raw / "outcomes.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "seal differs"):
            recovery.finish(self.evidence, self.plan, self.diff, self.source)
        self.assertEqual(json.loads((self.evidence / "recovery-metadata.json").read_text())[
            "status"], "incomplete")

    def test_missing_seal_stays_incomplete(self):
        with self.assertRaises(OSError):
            recovery.finish(self.evidence, self.plan, self.diff, self.source)
        self.assertEqual(json.loads((self.evidence / "recovery-metadata.json").read_text())[
            "status"], "incomplete")

    def guard_run(self, complete=True):
        values = self.manifest["shards"][20]["candidates"]
        outcomes = report(values, ["MissedMutant"])
        if not complete:
            outcomes["end_time"] = None
        raw = self.source / "target/final-mutation-shard-20/mutants.out"
        raw.mkdir(parents=True)
        (raw / "mutants.json").write_text(json.dumps(values))
        (raw / "outcomes.json").write_text(json.dumps(outcomes))
        evidence = self.source / "target/quality/shard-20"
        evidence.mkdir(parents=True)
        (evidence / "exit-code.txt").write_text("1")
        (evidence / "runner.log").write_text("complete mutant summary\n" + recovery.UNTRACKED_ERROR + "\n")
        seed = self.source / "crates/example/tests/case.proptest-regressions"
        seed.parent.mkdir(parents=True)
        seed.write_text("cc " + "0" * 64 + " # shrinks to 0\n")
        return str(seed.relative_to(self.source))

    def test_controller_attestation_is_explicit_and_preserves_exit_1_and_raw_missed(self):
        seed = self.guard_run()
        with patch.object(recovery, "git", return_value=seed), \
                patch.object(recovery, "verify_worktree") as checked:
            recovery.attest(self.evidence, self.plan, self.diff, self.source)
            recovery.finish(self.evidence, self.plan, self.diff, self.source)
        self.assertEqual(checked.call_count, 3)
        for call in checked.call_args_list:
            self.assertTrue(call.kwargs["allow_regressions"])
        self.assertFalse((self.source / "target/quality/shard-result-20.json").exists())
        metadata = json.loads((self.evidence / "recovery-metadata.json").read_text())
        self.assertEqual(metadata["status"], "complete_sealed_shard")
        self.assertEqual(metadata["runner_exit_code"], 1)
        self.assertIsNone(metadata["cargo_mutants_exit_code"])
        self.assertEqual(metadata["sealer"]["method"], "controller-live-source-sealer")
        self.assertEqual(metadata["seal_relative_path"], "target/quality/controller-shard-result-20.json")
        result = json.loads((self.source / metadata["seal_relative_path"]).read_text())
        self.assertEqual(result["outcomes"]["missed"], 1)
        self.assertIn(seed, result["artifacts"]["generated_regressions_sha256"])

    def test_exit_1_from_any_other_error_cannot_get_a_controller_seal(self):
        self.guard_run()
        (self.source / "target/quality/shard-20/runner.log").write_text("build or test infrastructure failed\n")
        with self.assertRaisesRegex(ValueError, "not solely"):
            recovery.attest(self.evidence, self.plan, self.diff, self.source)
        self.assertFalse((self.source / "target/quality/controller-shard-result-20.json").exists())

    def test_guard_error_without_live_seed_cannot_get_a_controller_seal(self):
        self.guard_run()
        with patch.object(recovery, "git", return_value=""), \
                self.assertRaisesRegex(ValueError, "no generated"):
            recovery.attest(self.evidence, self.plan, self.diff, self.source)

    def test_incomplete_raw_or_failed_live_source_check_cannot_get_a_controller_seal(self):
        seed = self.guard_run(complete=False)
        with patch.object(recovery, "git", return_value=seed), \
                patch.object(recovery, "verify_worktree"), self.assertRaises(ValueError):
            recovery.attest(self.evidence, self.plan, self.diff, self.source)
        self.assertFalse((self.source / "target/quality/controller-shard-result-20.json").exists())
        with patch.object(recovery, "git", return_value=seed), \
                patch.object(recovery, "verify_worktree", side_effect=ValueError("source differs")), \
                self.assertRaisesRegex(ValueError, "source differs"):
            recovery.attest(self.evidence, self.plan, self.diff, self.source)
        self.assertFalse((self.source / "target/quality/controller-shard-result-20.json").exists())


if __name__ == "__main__":
    unittest.main()
