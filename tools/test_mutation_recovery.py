import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

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
