#!/usr/bin/env python3
"""List and verify Release 1.0 mutation plans; never execute a campaign.

This explicit release-1.0 profile does not extend the historical campaign
protocol. Plans have their own schema and cannot be passed to its runner.
The analyzed worktree must stay clean; an external proposal tool is identified
separately by its own hashes, never attributed to the analyzed source SHA.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from merge_mutation_shards import (
    TOOL_VERSION, candidate_map, digest, file_digest, git, listing_argv,
    require, sha, verify_full_scope, workspace_packages, write_new,
)

PROFILE = "release-1.0"
RELEASE_BASE = "deec3aa8d2516ae19e335a5af77e6c38f8a854d8"
SHARDS = 64
SCHEMA = "bml-release-mutation-preflight-v1"
PRODUCER_FILES = ("release_mutation_preflight.py", "merge_mutation_shards.py")


def producer_identity():
    return {name: file_digest(Path(__file__).with_name(name)) for name in PRODUCER_FILES}


def check_intent(profile, revision, base):
    require(profile == PROFILE, "explicit release-1.0 profile required")
    sha(revision, "revision", 40)
    require(base == RELEASE_BASE, "unexpected Release 1.0 base")


def check_source(source):
    require(isinstance(source, dict) and set(source) == {
        "revision", "base_revision", "tree", "files", "workspace_packages"},
        "invalid release source snapshot")
    check_intent(PROFILE, source["revision"], source["base_revision"])
    sha(source["tree"], "tree", 40)
    require(isinstance(source["files"], dict) and source["files"], "missing source hashes")
    for name, value in source["files"].items():
        require(isinstance(name, str) and name and not Path(name).is_absolute()
                and ".." not in Path(name).parts, "invalid source path")
        sha(value, "source file digest", 64)
    packages = source["workspace_packages"]
    require(isinstance(packages, dict) and packages and all(
        isinstance(name, str) and name and isinstance(version, str) and version
        for name, version in packages.items()), "invalid workspace packages")


def release_source_snapshot(root, revision, base):
    check_intent(PROFILE, revision, base)
    require(git(root, "rev-parse", "HEAD") == revision, "worktree revision differs")
    require(git(root, "rev-parse", "--verify", f"{base}^{{commit}}") == base,
            "release base unavailable")
    require(git(root, "merge-base", base, revision) == base, "release base is not an ancestor")
    require(not git(root, "status", "--porcelain"), "release worktree must be clean")
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    files = {}
    for name in filter(None, paths):
        path = root / name
        content = os.readlink(path).encode() if path.is_symlink() else path.read_bytes()
        files[name] = hashlib.sha256(content).hexdigest()
    source = {"revision": revision, "base_revision": base,
              "tree": git(root, "rev-parse", "HEAD^{tree}"), "files": files,
              "workspace_packages": workspace_packages(root)}
    check_source(source)
    return source


def make_plan(source, diff_sha256, candidates, listed_shards, producer):
    check_source(source)
    sha(diff_sha256, "diff digest", 64)
    require(isinstance(producer, dict) and set(producer) == set(PRODUCER_FILES),
            "invalid preflight producer")
    for value in producer.values():
        sha(value, "producer digest", 64)
    candidate_map(candidates)
    require(len(candidates) >= SHARDS, "release plan needs 64 nonempty shards")
    require(all(isinstance(row["package"], str) and row["package"] in source["workspace_packages"]
                for row in candidates),
            "candidate package is outside frozen workspace")
    require(isinstance(listed_shards, list) and len(listed_shards) == SHARDS,
            "release plan requires exactly 64 listed shards")
    shards = []
    for index, actual in enumerate(listed_shards):
        candidate_map(actual)
        require(actual == candidates[index::SHARDS],
                f"release assignment differs from round-robin shard {index}/64")
        shards.append({"shard": f"{index}/64", "candidates": actual,
                       "candidates_sha256": digest(actual)})
    plan = {"schema": SCHEMA, "profile": PROFILE, "purpose": "list-only",
            "cargo_mutants_version": TOOL_VERSION, "producer": producer,
            "source": source, "diff_sha256": diff_sha256,
            "candidates": candidates, "candidates_sha256": digest(candidates), "shards": shards}
    plan["plan_sha256"] = digest(plan)
    return plan


def validate_plan(plan):
    require(isinstance(plan, dict), "invalid release plan")
    payload = {key: value for key, value in plan.items() if key != "plan_sha256"}
    require(plan.get("plan_sha256") == digest(payload), "release plan checksum mismatch")
    require(plan.get("schema") == SCHEMA and plan.get("profile") == PROFILE
            and plan.get("purpose") == "list-only"
            and plan.get("cargo_mutants_version") == TOOL_VERSION, "unsupported preflight protocol")
    shards = plan.get("shards")
    require(isinstance(shards, list) and all(isinstance(row, dict) for row in shards),
            "invalid release shard assignments")
    rebuilt = make_plan(plan.get("source"), plan.get("diff_sha256"), plan.get("candidates"),
                        [row.get("candidates") for row in shards], plan.get("producer"))
    require(rebuilt == plan, "release candidate or assignment metadata mismatch")


def prepare(root, profile, revision, base, diff, output):
    check_intent(profile, revision, base)
    require(not output.exists(), "preflight output already exists")
    producer = producer_identity()
    diff_sha256 = file_digest(diff)
    source = release_source_snapshot(root, revision, base)
    verify_full_scope(root, diff, base)
    require(file_digest(diff) == diff_sha256, "diff changed during scope verification")
    version = subprocess.check_output(["cargo", "mutants", "--version"], cwd=root).decode().strip()
    require(version == f"cargo-mutants {TOOL_VERSION}", "cargo-mutants version differs")
    candidates = json.loads(subprocess.check_output(listing_argv(diff), cwd=root))
    shards = [json.loads(subprocess.check_output(listing_argv(diff, f"{index}/64"), cwd=root))
              for index in range(SHARDS)]
    plan = make_plan(source, diff_sha256, candidates, shards, producer)
    require(release_source_snapshot(root, revision, base) == source, "source changed during listing")
    require(file_digest(diff) == diff_sha256, "diff changed during listing")
    require(producer_identity() == producer, "preflight producer changed during listing")
    write_new(output, plan)
    return {"plan_file_sha256": file_digest(output), "plan_sha256": plan["plan_sha256"],
            "source_revision": revision, "base_revision": base,
            "candidate_count": len(candidates), "listed_shards": SHARDS,
            "baseline_runs": 0, "mutants_evaluated": 0}


def verify(root, profile, revision, base, diff, plan_path, expected_file_sha256):
    check_intent(profile, revision, base)
    sha(expected_file_sha256, "expected plan file digest", 64)
    blob = plan_path.read_bytes()
    require(hashlib.sha256(blob).hexdigest() == expected_file_sha256,
            "release plan differs from independent file digest")
    plan = json.loads(blob)
    validate_plan(plan)
    require(plan["source"]["revision"] == revision, "plan revision differs from intended revision")
    require(plan["source"]["base_revision"] == base, "plan base differs from intended base")
    require(plan["producer"] == producer_identity(), "preflight producer differs")
    require(release_source_snapshot(root, revision, base) == plan["source"],
            "worktree sources differ from release plan")
    require(file_digest(diff) == plan["diff_sha256"], "release diff changed")
    verify_full_scope(root, diff, base)
    require(file_digest(diff) == plan["diff_sha256"], "diff changed during scope verification")
    return {"verified": "release preflight only", "source_revision": revision,
            "candidate_count": len(plan["candidates"]), "listed_shards": SHARDS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--profile", choices=[PROFILE], required=True)
        command.add_argument("--root", type=Path, default=Path.cwd())
        command.add_argument("--revision", required=True)
        command.add_argument("--base", required=True)
        command.add_argument("--diff", type=Path, required=True)
        if name == "prepare":
            command.add_argument("--output", type=Path, required=True)
        else:
            command.add_argument("--plan", type=Path, required=True)
            command.add_argument("--plan-file-sha256", required=True)
    args = parser.parse_args()
    shared = (args.root.resolve(), args.profile, args.revision, args.base, args.diff.resolve())
    try:
        result = (prepare(*shared, args.output.resolve()) if args.command == "prepare"
                  else verify(*shared, args.plan.resolve(), args.plan_file_sha256))
        print(json.dumps(result, sort_keys=True))
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Release preflight rejected: {error}\n")


if __name__ == "__main__":
    main()
