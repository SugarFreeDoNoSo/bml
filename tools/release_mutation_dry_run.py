#!/usr/bin/env python3
"""Exercise the Release plan/executor/aggregator contract with fixtures only.

No runner, shell, subprocess or Actions dispatch is exposed. Descriptors are
data, not commands; outputs deliberately cannot be mutation gate reports.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

import release_mutation_preflight as pre
from merge_mutation_shards import COUNTS, digest, require, sha, validate_outcomes, write_new

BUDGETS = {"documented-30-180": {"test_seconds": 30, "build_seconds": 180},
           "operational-360-600": {"test_seconds": 360, "build_seconds": 600}}
FIXTURE_SCHEMA = "bml-release-outcomes-fixture-v1"
RESULT_SCHEMA = "bml-release-dry-shard-v1"


def checked_plan(path, profile, revision, base, expected_file_sha256):
    pre.check_intent(profile, revision, base)
    sha(expected_file_sha256, "independent plan digest", 64)
    blob = path.read_bytes()
    require(hashlib.sha256(blob).hexdigest() == expected_file_sha256,
            "plan differs from independent digest")
    plan = json.loads(blob)
    pre.validate_plan(plan)
    require(plan["source"]["revision"] == revision and plan["source"]["base_revision"] == base,
            "plan differs from intended source")
    require(plan["producer"] == pre.producer_identity(), "preflight producer differs")
    return plan


def binding(plan, expected_file_sha256, budget):
    pre.validate_plan(plan)
    sha(expected_file_sha256, "independent plan digest", 64)
    require(budget in BUDGETS, "explicit documented or operational budget required")
    return {"profile": pre.PROFILE, "revision": plan["source"]["revision"],
            "base": plan["source"]["base_revision"], "plan_file_sha256": expected_file_sha256,
            "plan_payload_sha256": plan["plan_sha256"], "source_sha256": digest(plan["source"]),
            "diff_sha256": plan["diff_sha256"], "budget_profile": budget, "budget": dict(BUDGETS[budget]),
            "adapter_sha256": pre.file_digest(Path(__file__))}


def assignment(plan, shard):
    rows = {row["shard"]: row for row in plan["shards"]}
    require(isinstance(shard, str) and shard in rows, "unexpected release shard")
    return rows[shard]


def ticket(plan, common, shard):
    row = assignment(plan, shard)
    packages = plan["source"]["workspace_packages"]
    value = {"schema": "bml-release-dry-ticket-v1", "mode": "dry-run", "execution": "disabled",
             "binding": copy.deepcopy(common), "shard": shard, "candidates_sha256": row["candidates_sha256"],
             "candidate_count": len(row["candidates"]),
             "baseline_packages": [f"{name}@{packages[name]}" for name in
                                   sorted({candidate["package"] for candidate in row["candidates"]})],
             "mutant_test_scope": "workspace", "cargo_locked": True}
    return {**value, "ticket_sha256": digest(value)}


def describe(plan, expected_file_sha256, budget):
    common = binding(plan, expected_file_sha256, budget)
    return {"schema": "bml-release-dry-description-v1", "mode": "dry-run", "execution": "disabled",
            "tickets": [ticket(plan, common, row["shard"]) for row in plan["shards"]]}


def validate_fixture(plan, expected_ticket, fixture):
    require(isinstance(fixture, dict) and set(fixture) == {"schema", "ticket_sha256", "outcomes"}
            and fixture["schema"] == FIXTURE_SCHEMA, "explicit fixture envelope required")
    require(fixture["ticket_sha256"] == expected_ticket["ticket_sha256"],
            "fixture belongs to another ticket")
    return validate_outcomes(fixture["outcomes"], assignment(plan, expected_ticket["shard"])["candidates"],
                             plan["source"]["workspace_packages"])


def seal_fixture(plan, expected_file_sha256, budget, shard, supplied_ticket, fixture):
    expected = ticket(plan, binding(plan, expected_file_sha256, budget), shard)
    require(supplied_ticket == expected, "ticket differs from intended plan/shard/budget")
    validate_fixture(plan, expected, fixture)
    result = {"schema": RESULT_SCHEMA, "mode": "dry-run", "execution": "not-run",
              "ticket": expected, "fixture": fixture, "fixture_sha256": digest(fixture)}
    return {**result, "result_sha256": digest(result)}


def merge_fixtures(plan, expected_file_sha256, budget, results):
    common = binding(plan, expected_file_sha256, budget)
    require(isinstance(results, list) and len(results) == pre.SHARDS, "exactly 64 fixture shards required")
    seen_shards, seen_candidates, counts, summaries = set(), set(), Counter(), []
    for result in results:
        require(isinstance(result, dict) and set(result) == {
            "schema", "mode", "execution", "ticket", "fixture", "fixture_sha256", "result_sha256"}
            and result["schema"] == RESULT_SCHEMA and result["mode"] == "dry-run"
            and result["execution"] == "not-run", "unsupported dry result protocol")
        require(result["result_sha256"] == digest({key: value for key, value in result.items()
                                                  if key != "result_sha256"}), "dry result checksum mismatch")
        require(isinstance(result["ticket"], dict), "invalid dry ticket")
        shard = result["ticket"].get("shard")
        expected = ticket(plan, common, shard)
        require(shard not in seen_shards, "duplicate fixture shard")
        require(result["ticket"] == expected, "fixture result belongs to another plan/shard/budget")
        require(result["fixture_sha256"] == digest(result["fixture"]), "fixture checksum mismatch")
        _, mutations = validate_fixture(plan, expected, result["fixture"])
        names = {row["scenario"]["Mutant"]["name"] for row in mutations}
        require(not (names & seen_candidates), "fixture candidate appears in multiple shards")
        seen_candidates.update(names)
        seen_shards.add(shard)
        local = Counter(row["summary"] for row in mutations)
        counts.update(local)
        summaries.append({"shard": shard, "fixture_counts": {key: local[name] for name, key in COUNTS.items()}})
    require(seen_shards == {row["shard"] for row in plan["shards"]}
            and seen_candidates == set(pre.candidate_map(plan["candidates"])), "fixture union has holes")
    # No top-level outcomes/end_time/total_mutants: quality.py must never accept
    # a synthetic rehearsal as a real campaign, even when all fixtures are caught.
    return {"schema": "bml-release-dry-aggregate-v1", "mode": "dry-run", "execution": "not-run",
            "binding": common, "fixture_summary": {"candidate_count": len(seen_candidates),
                "shard_count": len(seen_shards), "counts": {key: counts[name] for name, key in COUNTS.items()}},
            "fixture_shards": sorted(summaries, key=lambda row: int(row["shard"].split("/")[0])),
            "campaign_evidence": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("describe", "seal-fixture", "merge-fixtures"):
        command = commands.add_parser(name)
        command.add_argument("--profile", choices=[pre.PROFILE], required=True)
        command.add_argument("--revision", required=True)
        command.add_argument("--base", required=True)
        command.add_argument("--plan", type=Path, required=True)
        command.add_argument("--plan-file-sha256", required=True)
        command.add_argument("--budget-profile", choices=sorted(BUDGETS), required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "seal-fixture":
            command.add_argument("--shard", required=True)
            command.add_argument("--ticket", type=Path, required=True)
            command.add_argument("--fixture", type=Path, required=True)
        elif name == "merge-fixtures":
            command.add_argument("--result", type=Path, action="append", required=True)
    args = parser.parse_args()
    try:
        require(not args.output.exists(), "dry output already exists")
        plan = checked_plan(args.plan, args.profile, args.revision, args.base, args.plan_file_sha256)
        common = (plan, args.plan_file_sha256, args.budget_profile)
        if args.command == "describe":
            result = describe(*common)
        elif args.command == "seal-fixture":
            result = seal_fixture(*common, args.shard, json.loads(args.ticket.read_bytes()),
                                  json.loads(args.fixture.read_bytes()))
        else:
            result = merge_fixtures(*common, [json.loads(path.read_bytes()) for path in args.result])
        write_new(args.output, result)
        print(json.dumps({"mode": "dry-run", "execution": "not-run", "output": str(args.output),
                          "output_sha256": pre.file_digest(args.output)}, sort_keys=True))
    except (ValueError, OSError) as error:
        parser.exit(1, f"Release dry contract rejected: {error}\n")


if __name__ == "__main__":
    main()
