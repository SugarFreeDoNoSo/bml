"""Synthetic fixture contracts only; no mutation, baseline, runner or network execution."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import merge_mutation_shards as historical
import mutation_actions as actions
import quality
import release_mutation_dry_run as dry
import release_mutation_preflight as pre
from test_merge_mutation_shards import campaign, report
from test_release_mutation_preflight import REVISION, fixture


PLAN_FILE_HASH = "a" * 64
BUDGET = "documented-30-180"


def synthetic_plan():
    source, diff_hash, candidates, partitions, _ = fixture()
    return pre.make_plan(source, diff_hash, candidates, partitions, pre.producer_identity())


def synthetic_outcomes(plan, ticket, summaries=None):
    candidates = dry.assignment(plan, ticket["shard"])["candidates"]
    return {"schema": dry.FIXTURE_SCHEMA, "ticket_sha256": ticket["ticket_sha256"],
            "outcomes": report(candidates, summaries)}


def synthetic_run(budget=BUDGET, adverse=False):
    plan = synthetic_plan()
    tickets = dry.describe(plan, PLAN_FILE_HASH, budget)["tickets"]
    results = []
    for index, ticket in enumerate(tickets):
        statuses = ["MissedMutant", "Timeout", "Unviable"] if adverse and index == 0 else None
        data = synthetic_outcomes(plan, ticket, statuses)
        results.append(dry.seal_fixture(plan, PLAN_FILE_HASH, budget, ticket["shard"], ticket, data))
    return plan, tickets, results


def reseal_ticket(ticket):
    ticket["ticket_sha256"] = pre.digest({key: value for key, value in ticket.items()
                                         if key != "ticket_sha256"})


def reseal_result(result):
    result["fixture_sha256"] = pre.digest(result["fixture"])
    result["result_sha256"] = pre.digest({key: value for key, value in result.items()
                                         if key != "result_sha256"})


@contextlib.contextmanager
def execution_forbidden():
    """Make all reachable execution/dispatch boundaries fail instead of simulating success."""
    with contextlib.ExitStack() as stack:
        guards = []
        for module, name in [(subprocess, "Popen"), (pre, "prepare"), (pre, "verify"),
                             (historical, "main"), (historical, "seal_shard"),
                             (historical, "merge_shards"), (actions, "main"),
                             (actions, "plan_outputs"), (actions, "merge_directory"),
                             (quality, "mutations"), (socket, "create_connection")]:
            guards.append(stack.enter_context(patch.object(module, name,
                          side_effect=AssertionError(f"dry path reached {name}"))))
        for name in ["system", "popen", "execl", "execle", "execlp", "execlpe", "execv", "execve",
                     "execvp", "execvpe", "posix_spawn", "posix_spawnp", "spawnl", "spawnle",
                     "spawnlp", "spawnlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe", "fork", "forkpty"]:
            if hasattr(os, name):
                guards.append(stack.enter_context(patch.object(os, name,
                              side_effect=AssertionError(f"dry path reached os.{name}"))))
        guards.append(stack.enter_context(patch.object(socket.socket, "connect",
                      side_effect=AssertionError("dry path reached socket.connect"))))
        yield
        for guard in guards:
            guard.assert_not_called()


class ReleaseDryContracts(unittest.TestCase):
    @contextlib.contextmanager
    def plan_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = synthetic_plan()
            path = root / "plan.json"
            pre.write_new(path, plan)
            yield root, plan, path, pre.file_digest(path)

    def test_checked_plan_requires_expected_profile_base_head_and_external_digest(self):
        with self.plan_file() as (_, plan, path, digest):
            args = [path, pre.PROFILE, REVISION, pre.RELEASE_BASE, digest]
            self.assertEqual(dry.checked_plan(*args), plan)
            for index, value in [(1, "historical"), (2, "b" * 40), (2, REVISION[:7]),
                                 (3, historical.BASE_REVISION), (4, "0" * 64), (4, "short")]:
                changed = args.copy()
                changed[index] = value
                with self.subTest(index=index, value=value), self.assertRaises(ValueError):
                    dry.checked_plan(*changed)

    def test_checked_plan_rejects_changed_bytes_producer_and_historical_manifest(self):
        for mode in ["bytes", "producer", "historical"]:
            with self.subTest(mode=mode), self.plan_file() as (_, plan, path, digest):
                if mode == "bytes":
                    path.write_text(path.read_text() + " ")
                elif mode == "producer":
                    plan["producer"][pre.PRODUCER_FILES[0]] = "f" * 64
                    plan["plan_sha256"] = pre.digest({k: v for k, v in plan.items() if k != "plan_sha256"})
                    path.write_text(json.dumps(plan))
                    digest = pre.file_digest(path)
                else:
                    path.write_text(json.dumps(campaign()[0]))
                    digest = pre.file_digest(path)
                with self.assertRaises(ValueError):
                    dry.checked_plan(path, pre.PROFILE, REVISION, pre.RELEASE_BASE, digest)

    def test_descriptors_bind_64_partitions_without_aliasing_budget_defaults(self):
        plan = synthetic_plan()
        original_budgets = copy.deepcopy(dry.BUDGETS)
        for name, values in original_budgets.items():
            description = dry.describe(plan, PLAN_FILE_HASH, name)
            tickets = description["tickets"]
            self.assertEqual(description["execution"], "disabled")
            self.assertEqual([t["shard"] for t in tickets], [f"{i}/64" for i in range(64)])
            self.assertEqual(sum(t["candidate_count"] for t in tickets), 129)
            for ticket in tickets:
                self.assertEqual(ticket["binding"]["budget"], values)
                self.assertEqual(ticket["binding"]["plan_file_sha256"], PLAN_FILE_HASH)
                self.assertEqual(ticket["binding"]["source_sha256"], pre.digest(plan["source"]))
                self.assertEqual(ticket["baseline_packages"], ["example@0.1.0"])
                self.assertEqual(ticket["mutant_test_scope"], "workspace")
                self.assertTrue(ticket["cargo_locked"])
            tickets[0]["binding"]["budget"]["test_seconds"] = -1
            self.assertEqual(tickets[1]["binding"]["budget"], values)
        self.assertEqual(dry.BUDGETS, original_budgets)

    def test_no_budget_is_selected_implicitly(self):
        plan = synthetic_plan()
        for budget in [None, "", "default", "30/180"]:
            with self.subTest(budget=budget), self.assertRaisesRegex(ValueError, "explicit"):
                dry.describe(plan, PLAN_FILE_HASH, budget)

    def test_resealed_tickets_cannot_change_any_intended_binding_or_executor_scope(self):
        plan = synthetic_plan()
        ticket = dry.describe(plan, PLAN_FILE_HASH, BUDGET)["tickets"][0]
        changes = [("binding", name) for name in ticket["binding"]]
        changes += [(None, name) for name in ["schema", "mode", "execution", "shard", "candidates_sha256",
                                             "candidate_count", "baseline_packages", "mutant_test_scope", "cargo_locked"]]
        for section, name in changes:
            changed = copy.deepcopy(ticket)
            target = changed[section] if section else changed
            target[name] = {"test_seconds": 1, "build_seconds": 1} if name == "budget" else "changed"
            reseal_ticket(changed)
            data = synthetic_outcomes(plan, ticket)
            data["ticket_sha256"] = changed["ticket_sha256"]
            with self.subTest(section=section, field=name), self.assertRaisesRegex(ValueError, "ticket differs"):
                dry.seal_fixture(plan, PLAN_FILE_HASH, BUDGET, "0/64", changed, data)

    def test_only_an_explicit_fixture_bound_to_the_ticket_is_accepted(self):
        plan = synthetic_plan()
        ticket = dry.describe(plan, PLAN_FILE_HASH, BUDGET)["tickets"][0]
        original = synthetic_outcomes(plan, ticket)
        for mode in ["raw-report", "schema", "ticket", "extra"]:
            data = copy.deepcopy(original)
            if mode == "raw-report":
                data = data["outcomes"]
            elif mode == "schema":
                data["schema"] = "historical"
            elif mode == "ticket":
                data["ticket_sha256"] = "0" * 64
            else:
                data["trusted"] = True
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                dry.seal_fixture(plan, PLAN_FILE_HASH, BUDGET, "0/64", ticket, data)

    def test_historical_identity_phase_baseline_and_count_guards_remain_enforced(self):
        plan = synthetic_plan()
        ticket = dry.describe(plan, PLAN_FILE_HASH, BUDGET)["tickets"][0]
        for mode in ["identity", "test-filter", "count", "missing-baseline", "failed-baseline", "omission", "duplicate"]:
            data = synthetic_outcomes(plan, ticket)
            outcomes = data["outcomes"]
            if mode == "identity":
                outcomes["outcomes"][1]["scenario"]["Mutant"]["replacement"] = "different"
            elif mode == "test-filter":
                outcomes["outcomes"][1]["phase_results"][1]["argv"].append("--test=smoke")
            elif mode == "count":
                outcomes["caught"] += 1
            elif mode == "missing-baseline":
                outcomes["outcomes"].pop(0)
            elif mode == "failed-baseline":
                outcomes["outcomes"][0]["summary"] = "MissedMutant"
            elif mode == "omission":
                outcomes["outcomes"].pop()
            else:
                outcomes["outcomes"].append(copy.deepcopy(outcomes["outcomes"][1]))
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                dry.seal_fixture(plan, PLAN_FILE_HASH, BUDGET, "0/64", ticket, data)

    def test_complete_aggregate_preserves_all_adverse_fixture_outcomes(self):
        plan, _, results = synthetic_run(adverse=True)
        aggregate = dry.merge_fixtures(plan, PLAN_FILE_HASH, BUDGET, list(reversed(results)))
        self.assertEqual(aggregate["fixture_summary"], {"candidate_count": 129, "shard_count": 64,
                         "counts": {"caught": 126, "missed": 1, "timeout": 1, "unviable": 1}})
        self.assertEqual([row["shard"] for row in aggregate["fixture_shards"]], [f"{i}/64" for i in range(64)])
        self.assertEqual(aggregate["execution"], "not-run")
        self.assertFalse(aggregate["campaign_evidence"])

    def test_missing_extra_or_duplicate_fixture_shards_are_rejected(self):
        plan, _, results = synthetic_run()
        for changed in [results[:-1], [*results, results[0]], [*results[:-1], results[0]]]:
            with self.subTest(length=len(changed)), self.assertRaisesRegex(ValueError, "64 fixture|duplicate"):
                dry.merge_fixtures(plan, PLAN_FILE_HASH, BUDGET, changed)

    def test_resealed_results_cannot_change_checksums_protocol_or_ticket_metadata(self):
        plan, _, original = synthetic_run()
        for mode in ["checksum", "fixture-checksum", "execution", "extra", "ticket-source", "shard"]:
            results = copy.deepcopy(original)
            row = results[0]
            if mode == "checksum":
                row["result_sha256"] = "0" * 64
            elif mode == "fixture-checksum":
                row["fixture_sha256"] = "0" * 64
                row["result_sha256"] = pre.digest({k: v for k, v in row.items() if k != "result_sha256"})
            elif mode == "execution":
                row["execution"] = "run"
                reseal_result(row)
            elif mode == "extra":
                row["approved"] = True
                reseal_result(row)
            else:
                if mode == "ticket-source":
                    row["ticket"]["binding"]["source_sha256"] = "0" * 64
                else:
                    row["ticket"]["shard"] = "64/64"
                reseal_ticket(row["ticket"])
                row["fixture"]["ticket_sha256"] = row["ticket"]["ticket_sha256"]
                reseal_result(row)
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                dry.merge_fixtures(plan, PLAN_FILE_HASH, BUDGET, results)

    def test_different_plan_budget_or_historical_result_cannot_be_mixed(self):
        plan, _, original = synthetic_run()
        _, _, other_budget = synthetic_run("operational-360-600")
        alternate = copy.deepcopy(plan)
        alternate["source"]["files"]["code.rs"] = "f" * 64
        alternate["plan_sha256"] = pre.digest({k: v for k, v in alternate.items() if k != "plan_sha256"})
        other_ticket = dry.describe(alternate, "b" * 64, BUDGET)["tickets"][0]
        other_result = dry.seal_fixture(alternate, "b" * 64, BUDGET, "0/64", other_ticket,
                                        synthetic_outcomes(alternate, other_ticket))
        for replacement in [other_budget[0], other_result, campaign()[1][0]]:
            with self.subTest(schema=replacement.get("schema")), self.assertRaises(ValueError):
                dry.merge_fixtures(plan, PLAN_FILE_HASH, BUDGET, [replacement, *original[1:]])

    def test_overlapping_identity_is_rejected_even_after_rehashing_fixture_and_result(self):
        plan, _, results = synthetic_run()
        results[1]["fixture"]["outcomes"]["outcomes"][1] = copy.deepcopy(
            results[0]["fixture"]["outcomes"]["outcomes"][1])
        reseal_result(results[1])
        with self.assertRaisesRegex(ValueError, "unexpected candidate"):
            dry.merge_fixtures(plan, PLAN_FILE_HASH, BUDGET, results)

    def test_even_all_caught_synthetic_aggregate_cannot_be_accepted_by_real_consumers(self):
        plan, _, results = synthetic_run()
        aggregate = dry.merge_fixtures(plan, PLAN_FILE_HASH, BUDGET, results)
        self.assertTrue({"outcomes", "total_mutants", "end_time"}.isdisjoint(aggregate))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic-aggregate.json"
            pre.write_new(path, aggregate)
            with self.assertRaises(KeyError):
                quality.mutations(path)
            with self.assertRaises(ValueError):
                historical.validate_manifest(aggregate)
            with self.assertRaises(ValueError):
                actions.checked_manifest(path, REVISION, pre.file_digest(path))

    def invoke(self, arguments):
        stdout = io.StringIO()
        with patch.object(sys, "argv", ["release_mutation_dry_run.py", *arguments]), \
                contextlib.redirect_stdout(stdout):
            dry.main()
        return json.loads(stdout.getvalue())

    def test_all_three_cli_stages_work_with_every_execution_boundary_forbidden(self):
        with self.plan_file() as (root, plan, path, file_hash), execution_forbidden():
            shared = ["--profile", pre.PROFILE, "--revision", REVISION, "--base", pre.RELEASE_BASE,
                      "--plan", str(path), "--plan-file-sha256", file_hash, "--budget-profile", BUDGET]
            description_path = root / "description.json"
            self.assertEqual(self.invoke(["describe", *shared, "--output", str(description_path)])["execution"], "not-run")
            tickets = json.loads(description_path.read_text())["tickets"]
            result_paths = []
            for index, ticket in enumerate(tickets):
                data = synthetic_outcomes(plan, ticket)
                output = root / f"result-{index}.json"
                if index == 0:
                    ticket_path, fixture_path = root / "ticket.json", root / "fixture.json"
                    pre.write_new(ticket_path, ticket)
                    pre.write_new(fixture_path, data)
                    self.invoke(["seal-fixture", *shared, "--shard", "0/64", "--ticket", str(ticket_path),
                                 "--fixture", str(fixture_path), "--output", str(output)])
                else:
                    pre.write_new(output, dry.seal_fixture(plan, file_hash, BUDGET, ticket["shard"], ticket, data))
                result_paths.extend(["--result", str(output)])
            output = root / "aggregate.json"
            self.invoke(["merge-fixtures", *shared, *result_paths, "--output", str(output)])
            aggregate = json.loads(output.read_text())
            self.assertEqual(aggregate["fixture_summary"]["candidate_count"], 129)
            self.assertFalse(aggregate["campaign_evidence"])

    def test_cli_has_no_execute_dispatch_runner_or_default_budget_mode(self):
        with self.plan_file() as (root, _, path, file_hash):
            shared = ["--profile", pre.PROFILE, "--revision", REVISION, "--base", pre.RELEASE_BASE,
                      "--plan", str(path), "--plan-file-sha256", file_hash, "--output", str(root / "output.json")]
            for arguments in [[name] for name in ["run", "execute", "dispatch", "commands"]] + [["describe", *shared]]:
                with self.subTest(arguments=arguments), execution_forbidden(), \
                        contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as stopped:
                    self.invoke(arguments)
                self.assertEqual(stopped.exception.code, 2)

    def test_cli_rejects_wrong_intent_and_preserves_existing_output(self):
        with self.plan_file() as (root, _, path, file_hash):
            output = root / "keep.json"
            shared = ["--profile", pre.PROFILE, "--revision", REVISION, "--base", pre.RELEASE_BASE,
                      "--plan", str(path), "--plan-file-sha256", file_hash, "--budget-profile", BUDGET]
            wrong = shared.copy()
            wrong[wrong.index("--base") + 1] = historical.BASE_REVISION
            with execution_forbidden(), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as stopped:
                self.invoke(["describe", *wrong, "--output", str(output)])
            self.assertEqual(stopped.exception.code, 1)
            self.assertFalse(output.exists())
            output.write_text("preserve")
            with execution_forbidden(), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as stopped:
                self.invoke(["describe", *shared, "--output", str(output)])
            self.assertEqual(stopped.exception.code, 1)
            self.assertEqual(output.read_text(), "preserve")


if __name__ == "__main__":
    unittest.main()
