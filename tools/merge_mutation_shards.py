#!/usr/bin/env python3
"""Prepare and merge complete cargo-mutants 27.1.0 round-robin campaigns.

Shard indices are ZERO-based: 0/N through (N-1)/N. prepare invokes only
cargo-mutants --list --json, including a separate --list for each shard; it does
not run builds/tests or mutations. All output files are created exclusively.

On a clean frozen worktree, using the pinned Rust toolchain:
  python3 tools/mutation_diff.py 40b3f1be2d3f97028b741750ed61ab72f501a555 target/quality/final.diff
  python3 tools/merge_mutation_shards.py prepare --revision SHA --base 40b3f1be2d3f97028b741750ed61ab72f501a555 --diff target/quality/final.diff --shards N --output target/quality/campaign.json
  python3 tools/merge_mutation_shards.py commands --manifest target/quality/campaign.json --diff target/quality/final.diff --root . --shard 0/N

commands prints, without running, a runner sequence with source verification,
an exclusive CARGO_TARGET_DIR, a baseline, --locked and explicit timeouts.
Each runner needs its own clean worktree at the same SHA and the same diff.
The frozen snapshot records workspace package versions from cargo metadata.
As cargo-mutants 27.1.0 does, each shard baseline tests its mutated packages;
each mutant tests the entire workspace. Build/test argv must match this exact
default-feature protocol, without target/test filters or additional arguments.
After collecting every sealed shard result:
  python3 tools/merge_mutation_shards.py merge --manifest campaign.json --shard-result result-0.json --shard-result result-1.json --output aggregate.json
  python3 tools/quality.py mutations --report aggregate.json

A successful merge means complete consistent evidence, not a passing mutation
score. Missed mutants and timeouts remain unchanged for quality.py to reject.
The manifest hash must be retained independently of the shard artifacts.
"""
import argparse
from collections import Counter
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile

BASE_REVISION = "40b3f1be2d3f97028b741750ed61ab72f501a555"
TOOL_VERSION = "27.1.0"
IDENTITY_FIELDS = {"name", "package", "file", "function", "span", "replacement", "genre"}
COUNTS = {"CaughtMutant": "caught", "MissedMutant": "missed", "Timeout": "timeout",
          "Unviable": "unviable"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha(value, label, length):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{%d}" % length, value),
            f"invalid {label}")


def candidate_map(candidates):
    require(isinstance(candidates, list), "candidate list must be an array")
    result = {}
    for candidate in candidates:
        require(isinstance(candidate, dict) and set(candidate) == IDENTITY_FIELDS | {"diff"},
                "unexpected cargo-mutants candidate format")
        name = candidate["name"]
        require(isinstance(name, str) and name and isinstance(candidate["diff"], str),
                "invalid candidate identity or diff")
        require(name not in result, f"duplicate candidate: {name}")
        result[name] = candidate
    return result


def check_source(source):
    require(isinstance(source, dict) and set(source) == {
                "revision", "base_revision", "tree", "files", "workspace_packages"},
            "invalid source snapshot")
    for field in ["revision", "base_revision", "tree"]:
        sha(source[field], field, 40)
    require(source["base_revision"] == BASE_REVISION, "unexpected development base")
    require(isinstance(source["files"], dict) and source["files"], "missing source file hashes")
    for path, value in source["files"].items():
        require(isinstance(path, str) and path and not Path(path).is_absolute()
                and ".." not in Path(path).parts, "invalid source path")
        sha(value, "source file digest", 64)
    packages = source["workspace_packages"]
    require(isinstance(packages, dict) and packages, "missing workspace package versions")
    require(all(isinstance(name, str) and name and isinstance(version, str) and version
                for name, version in packages.items()), "invalid workspace package versions")


def make_manifest(source, diff_sha256, candidates, listed_shards):
    check_source(source)
    sha(diff_sha256, "diff digest", 64)
    expected = candidate_map(candidates)
    require(expected, "final campaign must have candidates")
    require(all(isinstance(candidate["package"], str)
                and candidate["package"] in source["workspace_packages"] for candidate in candidates),
            "candidate package is outside frozen workspace")
    require(isinstance(listed_shards, list) and 1 <= len(listed_shards) <= len(candidates),
            "invalid shard count or empty shard")
    count = len(listed_shards)
    assignments = []
    for index, actual in enumerate(listed_shards):
        candidate_map(actual)
        require(actual == candidates[index::count],
                f"cargo-mutants assignment differs from round-robin shard {index}/{count}")
        assignments.append({"shard": f"{index}/{count}", "candidates": actual,
                            "candidates_sha256": digest(actual)})
    manifest = {"format_version": 1, "cargo_mutants_version": TOOL_VERSION,
                "sharding": "round-robin", "source": source, "diff_sha256": diff_sha256,
                "candidates": candidates, "candidates_sha256": digest(candidates),
                "shards": assignments}
    manifest["manifest_sha256"] = digest(manifest)
    return manifest


def validate_manifest(manifest):
    require(isinstance(manifest, dict), "invalid campaign manifest")
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    require(manifest.get("manifest_sha256") == digest(payload), "manifest checksum mismatch")
    require(type(manifest.get("format_version")) is int and manifest["format_version"] == 1
            and manifest.get("cargo_mutants_version") == TOOL_VERSION
            and manifest.get("sharding") == "round-robin", "unsupported campaign protocol")
    shards = manifest.get("shards")
    require(isinstance(shards, list) and shards, "missing shard assignments")
    require(all(isinstance(row, dict) for row in shards), "invalid shard assignment")
    rebuilt = make_manifest(manifest.get("source"), manifest.get("diff_sha256"),
                            manifest.get("candidates"), [row.get("candidates") for row in shards])
    require(rebuilt == manifest, "campaign candidate or assignment metadata mismatch")
    return {row["shard"]: row for row in shards}


def timestamp(value):
    require(isinstance(value, str) and value, "missing campaign timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("invalid campaign timestamp") from error
    require(parsed.tzinfo is not None, "campaign timestamp needs a timezone")
    return parsed


def check_phases(outcome, summary, baseline_packages=None):
    phases = outcome.get("phase_results")
    require(isinstance(phases, list) and all(isinstance(phase, dict) for phase in phases),
            "missing or invalid phase evidence")
    statuses = [(phase.get("phase"), phase.get("process_status")) for phase in phases]
    for phase in phases:
        name = phase.get("phase")
        require(name in {"Build", "Test"}, "unexpected cargo phase")
        expected = ["test"] + (["--no-run"] if name == "Build" else []) + ["--verbose"]
        expected += ([f"--package={package}" for package in baseline_packages]
                     if baseline_packages is not None else ["--workspace"])
        expected += ["--locked"]
        argv = phase.get("argv")
        require(isinstance(argv, list) and argv and all(isinstance(arg, str) for arg in argv)
                and Path(argv[0]).name in {"cargo", "cargo.exe"} and argv[1:] == expected,
                "cargo argv differs from frozen baseline/workspace test scope")
    build_ok = ("Build", "Success")
    test_ok = ("Test", "Success")
    def failed(status):
        return isinstance(status, dict) and set(status) == {"Failure"} \
            and type(status["Failure"]) is int and status["Failure"] != 0

    if summary in {"Success", "MissedMutant"}:
        valid = statuses == [build_ok, test_ok]
    elif summary == "CaughtMutant":
        valid = len(statuses) == 2 and statuses[0] == build_ok and statuses[1][0] == "Test" \
            and failed(statuses[1][1])
    elif summary == "Unviable":
        valid = len(statuses) == 1 and statuses[0][0] == "Build" \
            and failed(statuses[0][1])
    else:
        valid = statuses in [[("Build", "Timeout")], [build_ok, ("Test", "Timeout")]]
    require(valid, f"{summary} contradicts build/test phase evidence")


def validate_outcomes(data, candidates, workspace_packages):
    expected = candidate_map(candidates)
    baseline_packages = [f"{name}@{workspace_packages[name]}"
                         for name in sorted({candidate["package"] for candidate in candidates})]
    require(isinstance(data, dict) and data.get("cargo_mutants_version") == TOOL_VERSION,
            "unexpected outcomes version")
    require(timestamp(data.get("end_time")) >= timestamp(data.get("start_time")),
            "campaign ends before it starts")
    outcomes = data.get("outcomes")
    require(isinstance(outcomes, list), "missing outcomes")
    baselines, mutations, seen = [], [], set()
    for outcome in outcomes:
        require(isinstance(outcome, dict), "invalid outcome record")
        scenario = outcome.get("scenario")
        if scenario == "Baseline":
            require(outcome.get("summary") == "Success", "shard baseline failed")
            check_phases(outcome, "Success", baseline_packages)
            baselines.append(outcome)
            continue
        require(isinstance(scenario, dict) and set(scenario) == {"Mutant"}
                and isinstance(scenario["Mutant"], dict), "unexpected scenario")
        mutant = scenario["Mutant"]
        name = mutant.get("name")
        require(isinstance(name, str) and name in expected, f"unexpected candidate: {name}")
        require(name not in seen, f"duplicate mutant outcome: {name}")
        require(mutant == {key: value for key, value in expected[name].items() if key != "diff"},
                f"candidate identity/source mismatch: {name}")
        summary = outcome.get("summary")
        require(summary in COUNTS, f"unknown mutant summary: {summary}")
        check_phases(outcome, summary)
        seen.add(name)
        mutations.append(outcome)
    require(len(baselines) == 1, "exactly one successful baseline required per shard")
    require(seen == set(expected), "incomplete candidate outcomes")
    counts = Counter(row["summary"] for row in mutations)
    require(type(data.get("total_mutants")) is int and data["total_mutants"] == len(expected),
            "total_mutants count mismatch")
    for summary, key in COUNTS.items():
        require(type(data.get(key)) is int and data[key] == counts[summary],
                f"{key} count mismatch")
    require(type(data.get("success")) is int and data["success"] == 0,
            "unexpected successful mutant count")
    return baselines[0], mutations


def seal_shard(manifest, shard, source, candidates, outcomes, artifacts=None):
    assignments = validate_manifest(manifest)
    require(shard in assignments, "unexpected shard index")
    require(source == manifest["source"], "shard sources differ from frozen snapshot")
    require(candidates == assignments[shard]["candidates"], "shard candidate list mismatch")
    validate_outcomes(outcomes, candidates, manifest["source"]["workspace_packages"])
    return {"format_version": 1, "manifest_sha256": manifest["manifest_sha256"],
            "shard": shard, "source": source, "candidates": candidates,
            "outcomes": outcomes, "outcomes_sha256": digest(outcomes),
            "artifacts": artifacts or {}}


def validate_v1_result(manifest, result):
    """Validate one original format-1 seal; the acceptance rules are unchanged."""
    assignments = validate_manifest(manifest)
    require(isinstance(result, dict) and type(result.get("format_version")) is int
            and result["format_version"] == 1, "invalid shard result format")
    shard = result.get("shard")
    require(isinstance(shard, str) and shard in assignments, "unexpected shard index")
    require(result.get("manifest_sha256") == manifest["manifest_sha256"],
            "shard belongs to another candidate manifest")
    require(result.get("source") == manifest["source"], "shard sources differ")
    require(result.get("candidates") == assignments[shard]["candidates"],
            "shard assignment mismatch")
    require(result.get("outcomes_sha256") == digest(result.get("outcomes")),
            "shard outcome checksum mismatch")
    return validate_outcomes(result.get("outcomes"), result["candidates"],
                             manifest["source"]["workspace_packages"])


def merge_shards(manifest, results):
    assignments = validate_manifest(manifest)
    require(isinstance(results, list) and len(results) == len(assignments), "missing or extra shard")
    by_shard, nested = {}, {}
    for result in results:
        require(isinstance(result, dict), "invalid shard result format")
        shard = result.get("shard")
        require(isinstance(shard, str) and shard in assignments, "unexpected shard index")
        require(shard not in by_shard, "duplicate shard result")
        if type(result.get("format_version")) is int and result["format_version"] == 2:
            # Explicit parent envelopes preserve their real leaf baselines. They
            # never masquerade as a single original package-scope baseline.
            from mutation_subshards import validate_envelope
            baselines, mutations, start, end = validate_envelope(manifest, result)
            nested[shard] = copy.deepcopy(result)
        else:
            baseline, mutations = validate_v1_result(manifest, result)
            baselines = [copy.deepcopy(baseline)]
            baselines[0]["source_shard"] = shard
            start, end = result["outcomes"]["start_time"], result["outcomes"]["end_time"]
        by_shard[shard] = (result, baselines, mutations, start, end)
    baselines, mutations, seen = [], [], set()
    artifacts = {}
    for shard in assignments:
        result, originals, outcomes, _, _ = by_shard[shard]
        baselines.extend(copy.deepcopy(originals))
        artifacts[shard] = result.get("artifacts", {})
        for outcome in outcomes:
            name = outcome["scenario"]["Mutant"]["name"]
            require(name not in seen, "candidate appears in more than one shard")
            seen.add(name)
            row = copy.deepcopy(outcome)
            row["source_shard"] = shard
            mutations.append(row)
    require(seen == set(candidate_map(manifest["candidates"])), "candidate union has holes")
    order = {row["name"]: index for index, row in enumerate(manifest["candidates"])}
    mutations.sort(key=lambda row: order[row["scenario"]["Mutant"]["name"]])
    counts = Counter(row["summary"] for row in mutations)
    starts = [row[3] for row in by_shard.values()]
    ends = [row[4] for row in by_shard.values()]
    merged = {"outcomes": [copy.deepcopy(baselines[0]), *mutations],
              "total_mutants": len(mutations),
              **{key: counts[summary] for summary, key in COUNTS.items()}, "success": 0,
              "start_time": min(starts, key=timestamp), "end_time": max(ends, key=timestamp),
              "cargo_mutants_version": TOOL_VERSION, "campaign_manifest": manifest,
              "shard_baselines": baselines, "shard_artifacts": artifacts,
              "baseline_scope": "Every real shard/leaf baseline passed; first original baseline retained above."}
    if nested:
        merged["nested_recoveries"] = nested
    return merged


def read_json(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open("x") as output:
        json.dump(value, output, indent=2)
        output.write("\n")


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root).decode().strip()


def workspace_packages(root):
    metadata = json.loads(subprocess.check_output(
        ["cargo", "metadata", "--no-deps", "--format-version", "1", "--locked"], cwd=root))
    members = set(metadata["workspace_members"])
    packages = {package["name"]: package["version"] for package in metadata["packages"]
                if package["id"] in members}
    require(len(packages) == len(members), "workspace package metadata is incomplete")
    return packages


def proptest_sourceparallel(path, tracked):
    """Match proptest 1.11 SourceParallel's path or its WithSource fallback.

    A regression must map to a tracked Rust source, not merely have a familiar
    suffix. SourceParallel searches ancestors for lib.rs/main.rs, writes in a
    sibling proptest-regressions tree, or replaces the extension if none exists.
    """
    target = Path(path)
    if target.is_absolute() or ".." in target.parts:
        return False
    tracked = set(tracked)
    for name in tracked:
        source = Path(name)
        if source.suffix != ".rs":
            continue
        parent = source.parent
        while True:
            if str(parent / "lib.rs") in tracked or str(parent / "main.rs") in tracked:
                expected = (parent.parent / "proptest-regressions"
                            / source.relative_to(parent)).with_suffix(".txt")
                break
            if parent == parent.parent:
                expected = source.with_suffix(".proptest-regressions")
                break
            parent = parent.parent
        if target == expected:
            return True
    return False


def proptest_seed_file(path):
    """Admit only bounded regular files containing proptest persistence seeds."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 * 1024:
        return False
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return False
    found = False
    for line in lines:
        seed = line.split("#", 1)[0].strip()
        if not seed:
            continue
        parts = seed.split()
        valid = ((parts[0] in {"cc", "rc"} and len(parts) == 2
                  and re.fullmatch(r"[0-9a-fA-F]{64}", parts[1]))
                 or (parts[0] == "xs" and len(parts) == 5
                     and all(re.fullmatch(r"[0-9]{1,10}", p) and int(p) <= 0xffffffff
                             for p in parts[1:]))
                 or (parts[0] == "pt" and (len(parts) == 1 or
                     (len(parts) == 2 and re.fullmatch(r"(?:[0-9a-fA-F]{2})*", parts[1])))))
        if not valid:
            return False
        found = True
    return found


def source_snapshot(root, revision, base, allow_regressions=False):
    sha(revision, "revision", 40)
    require(base == BASE_REVISION, "unexpected development base")
    require(git(root, "rev-parse", "HEAD") == revision, "worktree revision differs")
    require(git(root, "rev-parse", base) == base, "development base unavailable")
    require(not git(root, "diff", "--name-only", "HEAD"), "tracked source files changed")
    untracked = git(root, "ls-files", "--others", "--exclude-standard").splitlines()
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    require(not untracked or (allow_regressions and all(
        proptest_sourceparallel(path, filter(None, paths)) and proptest_seed_file(root / path)
        for path in untracked)), "untracked source/input files present")
    files = {}
    for name in filter(None, paths):
        path = root / name
        content = os.readlink(path).encode() if path.is_symlink() else path.read_bytes()
        files[name] = hashlib.sha256(content).hexdigest()
    return {"revision": revision, "base_revision": base,
            "tree": git(root, "rev-parse", "HEAD^{tree}"), "files": files,
            "workspace_packages": workspace_packages(root)}


def verify_worktree(manifest, root, diff, allow_regressions=False):
    validate_manifest(manifest)
    source = manifest["source"]
    require(source_snapshot(root, source["revision"], source["base_revision"], allow_regressions) == source,
            "worktree sources differ from frozen manifest")
    require(file_digest(diff) == manifest["diff_sha256"], "mutation scope diff changed")


def verify_full_scope(root, diff, base):
    # Recompute the repository's formatting-aware diff, rather than accepting
    # a conveniently smaller mutation scope against the same named base.
    with tempfile.TemporaryDirectory(prefix="bml-mutation-scope-") as directory:
        expected = Path(directory) / "scope.diff"
        subprocess.run([sys.executable, str(root / "tools/mutation_diff.py"), base, str(expected)],
                       cwd=root, check=True)
        require(file_digest(diff) == file_digest(expected),
                "scope diff is not the complete formatting-aware development diff")


def listing_argv(diff, shard=None):
    argv = ["cargo", "mutants", "--workspace", "--cargo-arg=--locked", "--list", "--json",
            "--in-diff", str(Path(diff).resolve())]
    # In 27.1.0, --sharding requires --shard even for a list-only invocation.
    if shard is not None:
        argv += ["--sharding", "round-robin", "--shard", shard]
    return argv


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="list candidates and freeze assignments; no campaign")
    prepare.add_argument("--revision", required=True)
    prepare.add_argument("--base", required=True)
    prepare.add_argument("--shards", type=int, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    for command in [prepare, commands.add_parser("verify"), commands.add_parser("seal"),
                    commands.add_parser("commands", help="print a runner sequence; do not run it")]:
        command.add_argument("--root", type=Path, default=Path.cwd())
        command.add_argument("--diff", type=Path, required=True)
        if command is not prepare:
            command.add_argument("--manifest", type=Path, required=True)
    seal = commands.choices["seal"]
    seal.add_argument("--shard", required=True)
    seal.add_argument("--mutants-out", type=Path, required=True)
    seal.add_argument("--output", type=Path, required=True)
    runner = commands.choices["commands"]
    runner.add_argument("--shard", required=True)
    runner.add_argument("--timeout", type=int, default=360)
    runner.add_argument("--build-timeout", type=int, default=600)
    merge = commands.add_parser("merge")
    merge.add_argument("--manifest", type=Path, required=True)
    merge.add_argument("--shard-result", type=Path, action="append", required=True)
    merge.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            root = args.root.resolve()
            require(args.shards > 0, "shard count must be positive")
            source = source_snapshot(root, args.revision, args.base)
            verify_full_scope(root, args.diff, args.base)
            version = subprocess.check_output(["cargo", "mutants", "--version"], cwd=root).decode()
            require(version.strip() == f"cargo-mutants {TOOL_VERSION}", "cargo-mutants version differs")
            candidates = json.loads(subprocess.check_output(listing_argv(args.diff), cwd=root))
            require(args.shards <= len(candidates), "shard count creates empty shards")
            shards = [json.loads(subprocess.check_output(listing_argv(args.diff, f"{i}/{args.shards}"),
                                                        cwd=root)) for i in range(args.shards)]
            manifest = make_manifest(source, file_digest(args.diff), candidates, shards)
            verify_worktree(manifest, root, args.diff)
            write_new(args.output, manifest)
        elif args.command == "merge":
            manifest = read_json(args.manifest)
            result = merge_shards(manifest, [read_json(path) for path in args.shard_result])
            write_new(args.output, result)
        else:
            manifest = read_json(args.manifest)
            verify_worktree(manifest, args.root.resolve(), args.diff,
                            allow_regressions=args.command == "seal")
            if args.command == "seal":
                directory = args.mutants_out
                candidates = read_json(directory / "mutants.json")
                outcomes = read_json(directory / "outcomes.json")
                artifacts = {"directory": str(directory.resolve()),
                             "mutants_json_sha256": file_digest(directory / "mutants.json"),
                             "outcomes_json_sha256": file_digest(directory / "outcomes.json"),
                             "generated_regressions_sha256": {
                                 path: file_digest(args.root.resolve() / path)
                                 for path in git(args.root.resolve(), "ls-files", "--others",
                                                 "--exclude-standard").splitlines()}}
                result = seal_shard(manifest, args.shard, manifest["source"],
                                    candidates, outcomes, artifacts)
                write_new(args.output, result)
            elif args.command == "commands":
                require(args.shard in validate_manifest(manifest), "unexpected shard index")
                require(args.timeout > 0 and args.build_timeout > 0, "timeouts must be positive")
                index = args.shard.split("/")[0]
                root = args.root.resolve()
                output = root / f"target/final-mutation-shard-{index}"
                script = root / "tools/merge_mutation_shards.py"
                shared = ["--manifest", str(args.manifest.resolve()), "--diff", str(args.diff.resolve()),
                          "--root", str(root)]
                # --in-place is serial and conflicts with --jobs, including its
                # inherited CARGO_MUTANTS_JOBS environment value in 27.1.0.
                run = ["env", "-u", "CARGO_MUTANTS_JOBS",
                       f"CARGO_TARGET_DIR={root / ('target/final-mutation-build-' + index)}",
                       "CARGO_INCREMENTAL=0", "cargo", "mutants", "--workspace",
                       "--cargo-arg=--locked", "--test-workspace", "true", "--in-place",
                       "--timeout", str(args.timeout), "--build-timeout",
                       str(args.build_timeout), "--in-diff", str(args.diff.resolve()),
                       "--sharding", "round-robin", "--shard", args.shard, "--output", str(output)]
                print("set -eu")
                print(shlex.join(["cd", str(root)]))
                print(shlex.join(["test", "!", "-e", str(output)]))
                print(shlex.join(["python3", str(script), "verify", *shared]))
                print("if " + shlex.join(run) + "; then shard_exit=0; else shard_exit=$?; fi")
                print(shlex.join(["python3", str(script), "seal", *shared, "--shard", args.shard,
                                  "--mutants-out", str(output / "mutants.out"), "--output",
                                  str(root / f"target/quality/shard-result-{index}.json")]) + " || exit $?")
                print('exit "$shard_exit"')
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Mutation campaign evidence rejected: {error}\n")


if __name__ == "__main__":
    main()
