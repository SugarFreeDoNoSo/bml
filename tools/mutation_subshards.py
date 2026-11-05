#!/usr/bin/env python3
"""Fresh nested recovery of original 20/64; complete evidence is not quality approval.

The frozen 9a helper prepares all 256 independently listed assignments. Four
unmodified format-1 leaves execute 20,84,148,212/256. Their package baselines
remain literal cargo-mutants records. A separate controller-emitted, successful
unmutated WORKSPACE Build/Test preflight is mandatory: it is not an original
9-package cargo-mutants baseline. Format 2 preserves those five real baselines.

No raw prefix from either interrupted run is reused. Telemetry is diagnostic,
never a source attestation or a replacement for a complete raw report. The
unchanged quality.py gate still rejects missed/timeouts after the full union.
Collection must retain the plan's independently transmitted file SHA and verify
the actual run/controller/artifact provenance; an envelope is not a signature.
"""
import argparse
import base64
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import mutation_recovery as recovery
from merge_mutation_shards import (
    BASE_REVISION, TOOL_VERSION, candidate_map, digest, file_digest, git, read_json,
    require, seal_shard, sha, timestamp, validate_manifest, validate_v1_result,
    verify_worktree, write_new, proptest_sourceparallel, proptest_seed_file,
)

PARENT = "20/64"


def partition_labels(parent_index, parent_count, fanout):
    """Round-robin arithmetic only; this grants no eligibility to another shard."""
    require(type(parent_index) is int and type(parent_count) is int and type(fanout) is int
            and parent_count > 0 and 0 <= parent_index < parent_count and fanout > 0,
            "invalid parent partition")
    return tuple(f"{parent_index + parent_count * part}/{parent_count * fanout}"
                 for part in range(fanout))


LEAVES = partition_labels(20, 64, 4)
PARENT_COUNT = 81
ORIGINAL_JOB = 126992338567
SECOND_RUN = 41967922830
SECOND_JOB = 127049145325
SECOND_CONTROLLER = "17769835b84ca36d49439116dcedace561ae5b2c"
ORIGINAL_MANIFEST_ID = "cfee7050785ae54c46f84826c3c9e155540d4b22a4dab4793adfb14471d9322e"
WORKFLOW = ".github/workflows/mutation-subshards.yml"
PRODUCER = "mutation-subshards-controller-workspace-preflight-v1"
LEAF_METHOD = "mutation-subshards-controller-live-source-v1"
BUILD = ["cargo", "test", "--no-run", "--verbose", "--workspace", "--locked"]
TEST = ["cargo", "test", "--verbose", "--workspace", "--locked"]
ENVIRONMENT = {"CARGO_INCREMENTAL": "0", "CARGO_PROFILE_DEV_DEBUG": "0",
               "CARGO_PROFILE_TEST_DEBUG": "0", "CARGO_BUILD_JOBS": "2",
               "PROPTEST_RNG_SEED": "20260923", "CARGO_TERM_COLOR": "never"}
CONTROLLER_TOOLS = ("mutation_subshards.py", "mutation_recovery.py", "merge_mutation_shards.py")
LEAF_FILES = ("runner.log", "exit-code.txt", "tee-exit-code.txt", "run-leaf.sh",
              "toolchain.txt", "telemetry.jsonl")


def utc():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def json_file_sha(value):
    """The exact bytes emitted by the shared exclusive write_new function."""
    return hashlib.sha256((json.dumps(value, indent=2) + "\n").encode()).hexdigest()


def checked_original(manifest):
    assignments = validate_manifest(manifest)
    require(len(assignments) == 64 and manifest["source"]["revision"] == recovery.SOURCE
            and manifest["manifest_sha256"] == ORIGINAL_MANIFEST_ID
            and len(assignments[PARENT]["candidates"]) == PARENT_COUNT,
            "not the pinned original 20/64 campaign")
    return assignments


def check_plans(original, nested):
    parent = checked_original(original)[PARENT]["candidates"]
    children = validate_manifest(nested)
    require(len(children) == 256 and nested["source"] == original["source"]
            and nested["diff_sha256"] == original["diff_sha256"]
            and nested["candidates"] == original["candidates"],
            "nested manifest changed the frozen source, diff or candidate universe")
    seen = set()
    for index, shard in enumerate(LEAVES):
        values = children[shard]["candidates"]
        require(values == parent[index::4], "nested CLI assignment differs from parent partition")
        names = set(candidate_map(values))
        require(not seen & names, "nested leaves overlap")
        seen.update(names)
    require(seen == set(candidate_map(parent)), "nested union has holes")


def check_second(bundle, log):
    run, job, inventory = (bundle[key] for key in ("run", "job", "artifacts"))
    require(run.get("id") == SECOND_RUN and run.get("run_attempt") == 1
            and run.get("head_sha") == SECOND_CONTROLLER and run.get("event") == "workflow_dispatch"
            and run.get("path") == ".github/workflows/mutation-recovery.yml"
            and run.get("repository", {}).get("full_name") == recovery.REPOSITORY
            and run["repository"].get("id") == recovery.REPOSITORY_ID,
            "second interrupted recovery identity differs")
    require(job.get("id") == SECOND_JOB and job.get("run_id") == SECOND_RUN
            and job.get("run_attempt") == 1 and job.get("head_sha") == SECOND_CONTROLLER
            and job.get("name") == "recover" and job.get("status") == "completed"
            and job.get("conclusion") == "failure", "second recovery is not its terminal failure")
    steps = job.get("steps", [])
    for name, conclusion in [
        ("Verify independent manifest SHA and original partition", "success"),
        ("Run only the original frozen shard and its baseline", "failure"),
        ("Attest the live source and identify the actual sealer", "skipped"),
        ("Record recovery provenance and verify raw reports against the seal", "skipped"),
        ("Preserve recovery evidence without build outputs", "skipped"),
    ]:
        require(len([s for s in steps if s.get("name") == name
                     and s.get("conclusion") == conclusion]) == 1,
                "second recovery phase evidence differs")
    require(inventory.get("total_count") == len(inventory.get("artifacts", [])) == 0
            and re.search(r"Z ##\[error\]Process completed with exit code 143\.\r?$", log, re.M)
            and re.search(r"Z ##\[error\]The runner has received a shutdown signal\.", log),
            "second recovery does not prove shutdown143 without an artifact")
    require(timestamp(job.get("completed_at")) >= timestamp(job.get("started_at")),
            "second recovery timestamps differ")


def validate_origins(origins):
    require(isinstance(origins, dict) and set(origins) == {"original", "second", "cause"}
            and origins["cause"] == "unknown; shutdown143 is observed, OOM is not established",
            "missing interrupted-run provenance")
    for key in ("original", "second"):
        row = origins[key]
        require(isinstance(row, dict) and set(row) == {"api", "api_sha256", "log_base64", "log_sha256"}
                and row["api_sha256"] == digest(row["api"]), "origin API evidence changed")
        try:
            blob = base64.b64decode(row["log_base64"], validate=True)
            log = blob.decode("utf-8")
        except (ValueError, UnicodeError) as error:
            raise ValueError("invalid origin log bytes") from error
        require(hashlib.sha256(blob).hexdigest() == row["log_sha256"], "origin log bytes changed")
        if key == "original":
            recovery.validate_origin(recovery.REPOSITORY, ORIGINAL_JOB, 20, row["api"], log)
        else:
            check_second(row["api"], log)


def collect_origins():
    prefix = f"repos/{recovery.REPOSITORY}/actions"
    original = {"run": recovery.api(f"{prefix}/runs/{recovery.ORIGIN_RUN}"),
                "job": recovery.api(f"{prefix}/jobs/{ORIGINAL_JOB}"),
                "artifact": recovery.api(f"{prefix}/artifacts/{recovery.ARTIFACT_ID}"),
                "artifacts": recovery.api(f"{prefix}/runs/{recovery.ORIGIN_RUN}/artifacts?per_page=100")}
    second = {"run": recovery.api(f"{prefix}/runs/{SECOND_RUN}"),
              "job": recovery.api(f"{prefix}/jobs/{SECOND_JOB}"),
              "artifacts": recovery.api(f"{prefix}/runs/{SECOND_RUN}/artifacts?per_page=100")}
    result = {"cause": "unknown; shutdown143 is observed, OOM is not established"}
    for key, api, job in [("original", original, ORIGINAL_JOB), ("second", second, SECOND_JOB)]:
        blob = recovery.api_bytes(f"{prefix}/jobs/{job}/logs")
        result[key] = {"api": api, "api_sha256": digest(api),
                       "log_base64": base64.b64encode(blob).decode("ascii"),
                       "log_sha256": hashlib.sha256(blob).hexdigest()}
    validate_origins(result)
    return result


def validate_identity(identity):
    require(isinstance(identity, dict) and set(identity) == {
        "repository", "run_id", "attempt", "controller_revision"}
        and identity["repository"] == recovery.REPOSITORY
        and type(identity["run_id"]) is int and identity["run_id"] > 0
        and identity["run_id"] not in {recovery.ORIGIN_RUN, SECOND_RUN}
        and type(identity["attempt"]) is int and identity["attempt"] > 0,
        "invalid nested recovery run identity")
    sha(identity["controller_revision"], "controller revision", 40)


def validate_seeds(generated, source):
    require(isinstance(generated, dict), "missing generated seed inventory")
    for path, checksum in generated.items():
        require(isinstance(path, str) and proptest_sourceparallel(path, source["files"]),
                "generated seed does not map to tracked Rust SourceParallel input")
        sha(checksum, "generated seed hash", 64)


def validate_preflight(record, manifest, identity):
    require(isinstance(record, dict) and record.get("producer") == PRODUCER
            and record.get("identity") == identity and record.get("source") == manifest["source"]
            and record.get("source_before_sha256") == digest(manifest["source"])
            and record.get("source_after_sha256") == digest(manifest["source"])
            and record.get("environment") == ENVIRONMENT
            and record.get("unmutated_workspace") is True,
            "workspace preflight identity/source/scope differs")
    phases = record.get("phases")
    require(isinstance(phases, list) and len(phases) == 2, "workspace preflight is incomplete")
    previous = timestamp(record.get("started_at"))
    for phase, name, argv in zip(phases, ("Build", "Test"), (BUILD, TEST)):
        require(phase.get("phase") == name and phase.get("argv") == argv
                and type(phase.get("exit_code")) is int and phase["exit_code"] == 0,
                "workspace preflight requires actual successful Build/Test argv")
        start, end = timestamp(phase.get("started_at")), timestamp(phase.get("ended_at"))
        require(previous <= start <= end, "workspace preflight phase timestamps differ")
        sha(phase.get("log_sha256"), "preflight log hash", 64)
        require(phase.get("log_file") == f"workspace-{name.lower()}.log",
                "workspace preflight log identity differs")
        previous = end
    require(previous <= timestamp(record.get("ended_at")), "workspace preflight ends too early")
    validate_seeds(record.get("generated_regressions_sha256"), manifest["source"])


def check_plan_record(record, original, nested):
    check_plans(original, nested)
    identity = record.get("identity")
    validate_identity(identity)
    require(record.get("format_version") == 1
            and record.get("original_manifest_file_sha256") == recovery.MANIFEST_SHA256
            and record.get("original_manifest_sha256") == original["manifest_sha256"]
            and record.get("nested_manifest_sha256") == nested["manifest_sha256"]
            and record.get("source") == original["source"]
            and record.get("listing_producer") == "frozen-9a-prepare-256-independent-cli-listings"
            and record.get("frozen_prepare_sha256") == original["source"]["files"]["tools/merge_mutation_shards.py"],
            "nested plan provenance differs")
    sha(record.get("nested_manifest_file_sha256"), "independent nested manifest hash", 64)
    require(record["nested_manifest_file_sha256"] == json_file_sha(nested),
            "nested manifest file digest differs from its retained contents")
    argv = record.get("listing_argv")
    require(isinstance(argv, list) and len(argv) == 15 and all(isinstance(x, str) for x in argv),
            "missing literal frozen prepare argv")
    root, scope, output = Path(argv[4]), Path(argv[12]), Path(argv[14])
    require(root.is_absolute() and scope.is_absolute() and output.is_absolute()
            and argv == ["python3", str(root / "tools/merge_mutation_shards.py"), "prepare",
                         "--root", str(root), "--revision", recovery.SOURCE, "--base", BASE_REVISION,
                         "--shards", "256", "--diff", str(scope), "--output", str(output)]
            and scope.name == "final.diff" and output.name == "nested-campaign.json"
            and scope.parent == output.parent, "frozen prepare argv differs from exact 256-list scope")
    sha(record.get("listing_log_sha256"), "listing log hash", 64)
    tools = record.get("controller_tools_sha256")
    require(isinstance(tools, dict) and set(tools) == set(CONTROLLER_TOOLS),
            "missing controller tool inventory")
    for checksum in tools.values():
        sha(checksum, "controller tool hash", 64)
    require(timestamp(record.get("listed_at")) <= timestamp(record.get("prepared_at")),
            "nested prepare timestamps differ")
    validate_origins(record.get("origins"))
    validate_preflight(record.get("workspace_preflight"), original, identity)
    require(timestamp(record["listed_at"]) <= timestamp(record["workspace_preflight"]["started_at"])
            <= timestamp(record["workspace_preflight"]["ended_at"])
            <= timestamp(record["prepared_at"]), "preflight is outside prepare interval")


def seed_inventory(source):
    return {path: file_digest(source / path)
            for path in git(source, "ls-files", "--others", "--exclude-standard").splitlines()}


def preflight(source, original, diff, identity, directory):
    """Execute real unmutated workspace commands; only workflow calls this."""
    verify_worktree(original, source, diff)
    require(all(os.environ.get(key) == value for key, value in ENVIRONMENT.items()),
            "preflight environment differs from frozen execution environment")
    record = {"producer": PRODUCER, "identity": identity, "source": original["source"],
              "source_before_sha256": digest(original["source"]), "environment": ENVIRONMENT,
              "unmutated_workspace": True, "started_at": utc(), "phases": []}
    env = {**os.environ, "CARGO_TARGET_DIR": str(source / "target/nested-workspace-preflight")}
    for name, argv in [("Build", BUILD), ("Test", TEST)]:
        log = directory / f"workspace-{name.lower()}.log"
        row = {"phase": name, "argv": argv, "started_at": utc(), "log_file": log.name}
        with log.open("xb") as stream:
            status = subprocess.run(argv, cwd=source, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT, check=False).returncode
        row.update(exit_code=status, ended_at=utc(), log_sha256=file_digest(log))
        record["phases"].append(row)
        if status:
            record.update(ended_at=utc(), failure="real workspace preflight process failed")
            write_new(directory / "workspace-preflight-failed.json", record)
            raise ValueError(f"real workspace {name} failed with exit {status}")
    verify_worktree(original, source, diff, allow_regressions=True)
    record.update(ended_at=utc(), source_after_sha256=digest(original["source"]),
                  generated_regressions_sha256=seed_inventory(source))
    for relative in record["generated_regressions_sha256"]:
        destination = directory / "preflight-seeds" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, destination)
    validate_preflight(record, original, identity)
    return record


def checked_inputs(directory, expected_plan_sha=None, expected_manifest_sha=None):
    if expected_plan_sha is not None:
        require(file_digest(directory / "nested-plan.json") == expected_plan_sha,
                "independently transmitted plan file SHA differs")
    if expected_manifest_sha is not None:
        require(file_digest(directory / "nested-campaign.json") == expected_manifest_sha,
                "independently transmitted 256 manifest file SHA differs")
    original = recovery.checked_plan(directory / "campaign.json", directory / "final.diff", PARENT)
    nested, record = read_json(directory / "nested-campaign.json"), read_json(directory / "nested-plan.json")
    require(file_digest(directory / "nested-campaign.json") == record["nested_manifest_file_sha256"],
            "nested manifest bytes changed")
    check_plan_record(record, original, nested)
    require(record["controller_tools_sha256"] == {
        name: file_digest(Path(__file__).with_name(name)) for name in CONTROLLER_TOOLS},
        "controller tool bytes differ between prepare and consumer")
    for phase in record["workspace_preflight"]["phases"]:
        require(file_digest(directory / phase["log_file"]) == phase["log_sha256"],
                "preflight raw log differs from recorded bytes")
    require(file_digest(directory / "prepare-listing.log") == record["listing_log_sha256"],
            "frozen prepare listing log bytes changed")
    for relative, checksum in record["workspace_preflight"]["generated_regressions_sha256"].items():
        require(file_digest(directory / "preflight-seeds" / relative) == checksum
                and proptest_seed_file(directory / "preflight-seeds" / relative),
                "workspace preflight seed bytes are missing or changed")
    return original, nested, record


def prepare(source, directory, identity):
    validate_identity(identity)
    require(git(Path(__file__).resolve().parent.parent, "rev-parse", "HEAD") == identity["controller_revision"],
            "controller checkout differs from current workflow SHA")
    original = recovery.checked_plan(directory / "campaign.json", directory / "final.diff", PARENT)
    verify_worktree(original, source, directory / "final.diff")
    origins = collect_origins()
    frozen = source / "tools/merge_mutation_shards.py"
    require(file_digest(frozen) == original["source"]["files"]["tools/merge_mutation_shards.py"],
            "frozen prepare helper differs")
    argv = ["python3", str(frozen), "prepare", "--root", str(source),
            "--revision", recovery.SOURCE, "--base", BASE_REVISION, "--shards", "256",
            "--diff", str(directory / "final.diff"), "--output", str(directory / "nested-campaign.json")]
    with (directory / "prepare-listing.log").open("xb") as log:
        subprocess.run(argv, cwd=source, stdout=log, stderr=subprocess.STDOUT, check=True)
    nested = read_json(directory / "nested-campaign.json")
    check_plans(original, nested)
    verify_worktree(original, source, directory / "final.diff")
    listed_at = utc()
    baseline = preflight(source, original, directory / "final.diff", identity, directory)
    record = {"format_version": 1, "identity": identity, "source": original["source"],
              "original_manifest_file_sha256": recovery.MANIFEST_SHA256,
              "original_manifest_sha256": original["manifest_sha256"],
              "nested_manifest_sha256": nested["manifest_sha256"],
              "nested_manifest_file_sha256": file_digest(directory / "nested-campaign.json"),
              "listing_producer": "frozen-9a-prepare-256-independent-cli-listings",
              "listing_argv": argv, "listing_log_sha256": file_digest(directory / "prepare-listing.log"),
              "controller_tools_sha256": {name: file_digest(Path(__file__).with_name(name))
                                           for name in CONTROLLER_TOOLS},
              "frozen_prepare_sha256": file_digest(frozen), "listed_at": listed_at,
              "workspace_preflight": baseline, "prepared_at": utc(), "origins": origins}
    check_plan_record(record, original, nested)
    write_new(directory / "nested-plan.json", record)
    return record


def attest_leaf(source, directory, shard, output, expected_plan_sha, expected_manifest_sha):
    original, nested, plan = checked_inputs(directory, expected_plan_sha, expected_manifest_sha)
    require(shard in LEAVES, "unexpected nested leaf")
    index = shard.split("/")[0]
    evidence = source / f"target/quality/shard-{index}"
    code = int((evidence / "exit-code.txt").read_text())
    require(code in {0, 1, 2, 3}, "leaf execution is incomplete or interrupted")
    require(int((evidence / "tee-exit-code.txt").read_text()) == 0, "runner log tee failed")
    if code == 1:
        recovery.controller_guard_recovery(source, nested, directory / "final.diff", index)
    verify_worktree(nested, source, directory / "final.diff", allow_regressions=True)
    raw = source / f"target/final-mutation-shard-{index}/mutants.out"
    candidates, outcomes = read_json(raw / "mutants.json"), read_json(raw / "outcomes.json")
    artifacts = {"directory": str(raw.resolve()),
                 "mutants_json_sha256": file_digest(raw / "mutants.json"),
                 "outcomes_json_sha256": file_digest(raw / "outcomes.json"),
                 "generated_regressions_sha256": seed_inventory(source)}
    seal = seal_shard(nested, shard, nested["source"], candidates, outcomes, artifacts)
    if code in {0, 2, 3}:
        require(read_json(source / f"target/quality/shard-result-{index}.json") == seal,
                "controller attestation differs from frozen original leaf seal")
    verify_worktree(nested, source, directory / "final.diff", allow_regressions=True)
    require(seed_inventory(source) == artifacts["generated_regressions_sha256"],
            "generated seeds changed during leaf attestation")
    info = {"method": LEAF_METHOD, "identity": plan["identity"], "shard": shard,
            "parent_shard": PARENT, "plan_file_sha256": file_digest(directory / "nested-plan.json"),
            "manifest_file_sha256": file_digest(directory / "nested-campaign.json"),
            "source_revision": recovery.SOURCE, "runner_exit_code": code,
            "cargo_mutants_exit_code": code if code in {0, 2, 3} else None,
            "exit1_reason": "frozen-final-untracked-seed-guard" if code == 1 else None,
            "sealer_sha256": file_digest(Path(__file__).with_name("merge_mutation_shards.py")),
            "evidence_files_sha256": {name: file_digest(evidence / name) for name in LEAF_FILES},
            "attested_at": utc(), "seal_sha256": digest(seal)}
    require(timestamp(plan["prepared_at"]) <= timestamp(outcomes["start_time"])
            <= timestamp(outcomes["end_time"]) <= timestamp(info["attested_at"]),
            "leaf execution timestamps precede prepare or exceed attestation")
    leaf = {"format_version": 1, "seal": seal, "attestation": info}
    validate_leaf(plan, nested, leaf)
    write_new(output, leaf)


def validate_leaf(plan, nested, leaf):
    require(isinstance(leaf, dict) and set(leaf) == {"format_version", "seal", "attestation"}
            and type(leaf["format_version"]) is int and leaf["format_version"] == 1,
            "invalid leaf evidence wrapper")
    seal, info = leaf["seal"], leaf["attestation"]
    baseline, mutations = validate_v1_result(nested, seal)
    require(seal["shard"] in LEAVES and info.get("method") == LEAF_METHOD
            and info.get("identity") == plan["identity"] and info.get("shard") == seal["shard"]
            and info.get("parent_shard") == PARENT and info.get("source_revision") == recovery.SOURCE
            and info.get("manifest_file_sha256") == plan["nested_manifest_file_sha256"]
            and info.get("seal_sha256") == digest(seal), "leaf attestation identity/checksum differs")
    for key in ("plan_file_sha256", "sealer_sha256"):
        sha(info.get(key), key, 64)
    require(info["sealer_sha256"] == plan["controller_tools_sha256"]["merge_mutation_shards.py"],
            "leaf sealer differs from prepared controller")
    files = info.get("evidence_files_sha256")
    require(isinstance(files, dict) and set(files) == set(LEAF_FILES), "leaf execution evidence is incomplete")
    for checksum in files.values():
        sha(checksum, "leaf execution file hash", 64)
    code = info.get("runner_exit_code")
    require(type(code) is int and code in {0, 1, 2, 3}
            and info.get("cargo_mutants_exit_code") == (None if code == 1 else code)
            and info.get("exit1_reason") == ("frozen-final-untracked-seed-guard" if code == 1 else None),
            "leaf runner completion status differs")
    raw = seal["outcomes"]
    expected_code = 3 if raw["timeout"] else 2 if raw["missed"] else 0
    require(code == 1 or code == expected_code, "runner exit code contradicts raw cargo-mutants outcomes")
    seeds = seal.get("artifacts", {}).get("generated_regressions_sha256", {})
    validate_seeds(seeds, nested["source"])
    require(code != 1 or any(path.endswith(".proptest-regressions") for path in seeds),
            "exit1 guard lacks a generated fallback seed inventory")
    start, end = seal["outcomes"]["start_time"], seal["outcomes"]["end_time"]
    require(timestamp(plan["prepared_at"]) <= timestamp(start) <= timestamp(end)
            <= timestamp(info.get("attested_at")), "leaf timestamps differ")
    return baseline, mutations


def within_api_completion(precise, completed_at):
    """GitHub's second-precision bound covers [t,t+1s); fractions stay exact."""
    event, completion = timestamp(precise), timestamp(completed_at)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:\d{2})", completed_at):
        return event < completion + timedelta(seconds=1)
    return event <= completion


def validate_execution(api, plan, leaves, plan_sha):
    identity = plan["identity"]
    require(isinstance(api, dict) and set(api) == {"run", "jobs", "artifacts"}, "missing execution API")
    run, jobs, inventory = api["run"], api["jobs"], api["artifacts"]
    require(run.get("id") == identity["run_id"] and run.get("run_attempt") == identity["attempt"]
            and run.get("head_sha") == identity["controller_revision"]
            and run.get("event") == "workflow_dispatch" and run.get("path") == WORKFLOW
            and run.get("repository", {}).get("full_name") == recovery.REPOSITORY
            and run["repository"].get("id") == recovery.REPOSITORY_ID,
            "actual nested workflow run differs")
    rows, artifacts = jobs.get("jobs", []), inventory.get("artifacts", [])
    require(jobs.get("total_count") == len(rows) <= 100
            and inventory.get("total_count") == len(artifacts) <= 100,
            "incomplete execution job/artifact inventory")
    prepares = [job for job in rows if job.get("name") == "prepare"]
    require(len(prepares) == 1 and prepares[0].get("run_id") == identity["run_id"]
            and prepares[0].get("run_attempt") == identity["attempt"]
            and prepares[0].get("head_sha") == identity["controller_revision"]
            and prepares[0].get("status") == "completed"
            and prepares[0].get("conclusion") == "success", "prepare API job is not successful")
    require(timestamp(prepares[0].get("started_at")) <= timestamp(plan["listed_at"])
            <= timestamp(plan["workspace_preflight"]["started_at"])
            <= timestamp(plan["prepared_at"])
            and within_api_completion(plan["prepared_at"], prepares[0].get("completed_at")),
            "prepare API/preflight chronology differs")
    for name in ("List all 256 frozen assignments and run real workspace preflight",
                 "Transmit independent plan and manifest file digests",
                 "Upload immutable nested plan and real workspace baseline"):
        require(len([step for step in prepares[0].get("steps", [])
                     if step.get("name") == name and step.get("conclusion") == "success"]) == 1,
                "prepare listing/preflight/digest/upload phase failed")
    plans = [a for a in artifacts if a.get("name") ==
             f"mutation-subplan-{identity['run_id']}-{identity['attempt']}"]
    require(len(plans) == 1, "missing or duplicate prepare artifact")
    artifact_ids = set()
    for shard, leaf in leaves.items():
        index = shard.split("/")[0]
        matching = [j for j in rows if j.get("name") == f"leaf ({index})"]
        require(len(matching) == 1, "missing or duplicate leaf API job")
        job = matching[0]
        require(job.get("run_id") == identity["run_id"] and job.get("run_attempt") == identity["attempt"]
                and job.get("head_sha") == identity["controller_revision"]
                and job.get("status") == "completed" and job.get("conclusion") == "success",
                "leaf API job did not complete successfully")
        steps = job.get("steps", [])
        for name in ("Execute frozen leaf with diagnostic telemetry", "Attest complete live-source leaf",
                     "Upload complete or partial leaf evidence"):
            require(len([s for s in steps if s.get("name") == name
                         and s.get("conclusion") == "success"]) == 1,
                    "leaf execution, attestation or upload phase failed")
        seal, info = leaf["seal"], leaf["attestation"]
        require(info["plan_file_sha256"] == plan_sha
                and timestamp(job.get("started_at")) <= timestamp(seal["outcomes"]["start_time"])
                <= timestamp(seal["outcomes"]["end_time"]) <= timestamp(info["attested_at"])
                and within_api_completion(info["attested_at"], job.get("completed_at")),
                "leaf run/plan chronology differs")
        name = f"mutation-subleaf-{identity['run_id']}-{identity['attempt']}-{index}"
        matches = [a for a in artifacts if a.get("name") == name]
        require(len(matches) == 1, "missing or duplicate leaf artifact")
        artifact = matches[0]
        require(type(artifact.get("id")) is int and artifact["id"] > 0
                and artifact["id"] not in artifact_ids and artifact.get("expired") is False
                and artifact.get("workflow_run", {}).get("id") == identity["run_id"]
                and artifact["workflow_run"].get("head_sha") == identity["controller_revision"]
                and artifact["workflow_run"].get("repository_id") == recovery.REPOSITORY_ID
                and artifact["workflow_run"].get("head_repository_id") == recovery.REPOSITORY_ID,
                "leaf artifact belongs to another run/source")
        require(str(artifact.get("digest", "")).startswith("sha256:"), "missing leaf API artifact digest")
        sha(artifact["digest"][7:], "API artifact digest", 64)
        artifact_ids.add(artifact["id"])
    artifact = plans[0]
    require(type(artifact.get("id")) is int and artifact["id"] > 0
            and artifact["id"] not in artifact_ids and artifact.get("expired") is False
            and artifact.get("workflow_run", {}).get("id") == identity["run_id"]
            and artifact["workflow_run"].get("head_sha") == identity["controller_revision"]
            and artifact["workflow_run"].get("repository_id") == recovery.REPOSITORY_ID
            and artifact["workflow_run"].get("head_repository_id") == recovery.REPOSITORY_ID,
            "prepare artifact belongs to another run/source")
    require(str(artifact.get("digest", "")).startswith("sha256:"), "missing plan API artifact digest")
    sha(artifact["digest"][7:], "API plan artifact digest", 64)


def make_envelope(original, nested, plan, leaves, execution, plan_sha):
    result = {"format_version": 2, "protocol": "bml-nested-mutation-recovery-v2",
              "shard": PARENT, "manifest_sha256": original["manifest_sha256"],
              "original_manifest_file_sha256": recovery.MANIFEST_SHA256,
              "source": original["source"], "candidates": original["shards"][20]["candidates"],
              "nested_manifest": nested, "plan": plan, "plan_file_sha256": plan_sha,
              "leaves": leaves, "execution_api": execution, "execution_api_sha256": digest(execution),
              "baseline_scope": "Four literal cargo-mutants leaf package baselines plus one real controller workspace preflight; no original parent baseline is asserted.",
              "quality_gate": "not approved; full original union and unchanged quality.py required"}
    result["envelope_sha256"] = digest(result)
    validate_envelope(original, result)
    return result


def validate_envelope(original, result):
    """Dispatch target for merge_shards. No format-1 rule or score is relaxed."""
    require(type(result.get("format_version")) is int and result["format_version"] == 2
            and result.get("protocol") == "bml-nested-mutation-recovery-v2"
            and result.get("shard") == PARENT
            and result.get("manifest_sha256") == original["manifest_sha256"]
            and result.get("original_manifest_file_sha256") == recovery.MANIFEST_SHA256
            and result.get("source") == original["source"]
            and result.get("candidates") == original["shards"][20]["candidates"],
            "nested envelope parent/source differs")
    require(result.get("envelope_sha256") == digest({key: value for key, value in result.items()
                                                     if key != "envelope_sha256"}),
            "nested envelope checksum differs")
    nested, plan = result.get("nested_manifest"), result.get("plan")
    check_plan_record(plan, original, nested)
    sha(result.get("plan_file_sha256"), "plan file hash", 64)
    require(result["plan_file_sha256"] == json_file_sha(plan), "plan file digest differs from contents")
    require(result.get("execution_api_sha256") == digest(result.get("execution_api")),
            "nested execution API checksum differs")
    leaves = result.get("leaves")
    require(isinstance(leaves, list) and len(leaves) == 4, "four complete leaves are required")
    by_shard, baselines, outcomes = {}, [], []
    for leaf in leaves:
        baseline, mutations = validate_leaf(plan, nested, leaf)
        shard = leaf["seal"]["shard"]
        require(shard not in by_shard, "duplicate nested leaf")
        by_shard[shard] = leaf
        row = copy.deepcopy(baseline)
        row["source_shard"] = f"{PARENT}::{shard}"
        baselines.append(row)
        outcomes.extend(mutations)
    require(set(by_shard) == set(LEAVES), "nested leaf union is incomplete")
    validate_execution(result["execution_api"], plan, by_shard, result["plan_file_sha256"])
    seen = [row["scenario"]["Mutant"]["name"] for row in outcomes]
    require(len(seen) == len(set(seen)) == PARENT_COUNT
            and set(seen) == set(candidate_map(result["candidates"])), "nested outcome union differs")
    order = {row["name"]: i for i, row in enumerate(result["candidates"])}
    outcomes.sort(key=lambda row: order[row["scenario"]["Mutant"]["name"]])
    return baselines, outcomes, min((leaf["seal"]["outcomes"]["start_time"] for leaf in leaves), key=timestamp), max((leaf["seal"]["outcomes"]["end_time"] for leaf in leaves), key=timestamp)


def retain_leaf_zip(blob, artifact, index, destination):
    """Bounded inspection and whitelisted extraction into a new target directory."""
    require(artifact.get("digest") == "sha256:" + hashlib.sha256(blob).hexdigest(),
            "downloaded leaf ZIP differs from immutable API artifact digest")
    values, names = recovery.zip_members(blob, {"target/quality/leaf-evidence.json"})
    leaf = json.loads(values["target/quality/leaf-evidence.json"])
    require(leaf.get("seal", {}).get("shard") == f"{index}/256", "downloaded ZIP has another leaf")
    generated = leaf["seal"].get("artifacts", {}).get("generated_regressions_sha256", {})
    require(isinstance(generated, dict), "leaf ZIP seed inventory missing")
    for name in names:
        if name.endswith("/"):
            continue
        require(name.startswith("target/quality/")
                or name.startswith(f"target/final-mutation-shard-{index}/mutants.out/")
                or name in generated, "leaf ZIP contains a path outside evidence/declared seeds")
    values, _ = recovery.zip_members(blob, {name for name in names if not name.endswith("/")})
    require(not destination.exists(), "leaf download destination already exists")
    destination.mkdir(parents=True)
    for name, content in values.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(content)
    with (destination / "original-artifact.zip").open("xb") as stream:
        stream.write(blob)
    write_new(destination / "artifact-api.json", artifact)


def verify_downloaded_leaf(path, leaf):
    """Compare retained raw/runner/seed bytes with this live-source attestation."""
    seal, info = leaf["seal"], leaf["attestation"]
    index = seal["shard"].split("/")[0]
    artifact_root = path.parents[2]
    roots = list(artifact_root.rglob(f"final-mutation-shard-{index}/mutants.out"))
    require(len(roots) == 1, "missing or duplicate downloaded raw directory")
    raw = roots[0]
    require(read_json(raw / "outcomes.json") == seal["outcomes"]
            and read_json(raw / "mutants.json") == seal["candidates"]
            and file_digest(raw / "outcomes.json") == seal["artifacts"]["outcomes_json_sha256"]
            and file_digest(raw / "mutants.json") == seal["artifacts"]["mutants_json_sha256"],
            "downloaded raw differs from leaf seal")
    evidence = path.parent / f"shard-{index}"
    require(set(info.get("evidence_files_sha256", {})) == set(LEAF_FILES),
            "missing leaf execution evidence hashes")
    for name, checksum in info["evidence_files_sha256"].items():
        require(file_digest(evidence / name) == checksum, "downloaded runner evidence changed")
    code = int((evidence / "exit-code.txt").read_text())
    require(code == info["runner_exit_code"] and (evidence / "tee-exit-code.txt").read_text().strip() == "0",
            "downloaded runner/tee status contradicts attestation")
    frozen = path.parent / f"shard-result-{index}.json"
    if code == 1:
        lines = (evidence / "runner.log").read_text().rstrip().splitlines()
        require(not frozen.exists() and lines and lines[-1] == recovery.UNTRACKED_ERROR,
                "downloaded exit1 is not the final untracked-seed guard without a frozen seal")
    else:
        require(read_json(frozen) == seal, "downloaded frozen v1 seal differs from controller attestation")
    for relative, checksum in seal["artifacts"].get("generated_regressions_sha256", {}).items():
        matches = [p for p in artifact_root.rglob(Path(relative).name) if str(p).endswith("/" + relative)]
        require(len(matches) == 1 and file_digest(matches[0]) == checksum and proptest_seed_file(matches[0]),
                "downloaded generated seed is missing or changed")


def assemble(directory, leaf_root, output, expected_plan_sha, expected_manifest_sha):
    original, nested, plan = checked_inputs(directory, expected_plan_sha, expected_manifest_sha)
    identity = plan["identity"]
    require(git(Path(__file__).resolve().parent.parent, "rev-parse", "HEAD") == identity["controller_revision"],
            "aggregation controller differs from prepare")
    prefix = f"repos/{recovery.REPOSITORY}/actions/runs/{identity['run_id']}"
    api = {"run": recovery.api(prefix),
           "jobs": recovery.api(prefix + f"/attempts/{identity['attempt']}/jobs?per_page=100"),
           "artifacts": recovery.api(prefix + "/artifacts?per_page=100")}
    inventory = api["artifacts"]
    require(inventory.get("total_count") == len(inventory.get("artifacts", [])) <= 100,
            "incomplete leaf artifact API inventory")
    plans = [a for a in inventory["artifacts"] if a.get("name") ==
             f"mutation-subplan-{identity['run_id']}-{identity['attempt']}"]
    require(len(plans) == 1 and plans[0].get("expired") is False,
            "missing/duplicate/expired current-attempt prepare artifact")
    plan_artifact = plans[0]
    require(type(plan_artifact.get("id")) is int and plan_artifact["id"] > 0, "invalid plan artifact ID")
    blob = recovery.api_bytes(f"repos/{recovery.REPOSITORY}/actions/artifacts/{plan_artifact['id']}/zip")
    require(plan_artifact.get("digest") == "sha256:" + hashlib.sha256(blob).hexdigest(),
            "downloaded plan ZIP differs from immutable API digest")
    required = {"nested-plan.json", "nested-campaign.json", "campaign.json", "final.diff",
                "workspace-build.log", "workspace-test.log"}
    members, _ = recovery.zip_members(blob, required)
    for name, content in members.items():
        require(content == (directory / name).read_bytes(), "API plan ZIP differs from independently verified download")
    with (leaf_root / "prepare-artifact.zip").open("xb") as stream:
        stream.write(blob)
    write_new(leaf_root / "prepare-artifact-api.json", plan_artifact)
    for shard in LEAVES:
        index = shard.split("/")[0]
        name = f"mutation-subleaf-{identity['run_id']}-{identity['attempt']}-{index}"
        artifacts = [a for a in inventory["artifacts"] if a.get("name") == name]
        require(len(artifacts) == 1 and artifacts[0].get("expired") is False,
                "missing/duplicate/expired current-attempt leaf artifact")
        artifact = artifacts[0]
        require(type(artifact.get("id")) is int and artifact["id"] > 0,
                "invalid leaf artifact ID")
        blob = recovery.api_bytes(f"repos/{recovery.REPOSITORY}/actions/artifacts/{artifact['id']}/zip")
        retain_leaf_zip(blob, artifact, index, leaf_root / index)
    paths = list(leaf_root.rglob("leaf-evidence.json"))
    require(len(paths) == 4, "missing or duplicate downloaded leaf evidence")
    leaves = [read_json(path) for path in paths]
    # Re-read retained raw bytes; a seal checksum alone does not attest downloads.
    for path, leaf in zip(paths, leaves):
        verify_downloaded_leaf(path, leaf)
    result = make_envelope(original, nested, plan, leaves, api, file_digest(directory / "nested-plan.json"))
    write_new(output, result)


def diagnostic_sample(pid, root):
    """Only this PID subtree, cgroup counters and disk; never argv/env/secrets."""
    processes, pending, seen = [], [pid], set()
    while pending and len(seen) < 256:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        try:
            status = Path(f"/proc/{current}/status").read_text()
            values = {key: value.strip() for key, value in (line.split(":", 1)
                      for line in status.splitlines() if ":" in line)
                      if key in {"State", "VmRSS", "VmHWM", "Threads"}}
            processes.append({"pid": current, **values})
            for children in Path(f"/proc/{current}/task").glob("*/children"):
                pending.extend(int(x) for x in children.read_text().split())
        except (OSError, ValueError):
            pass
    cgroup = {}
    for name in ("memory.current", "memory.peak", "memory.max", "memory.events", "pids.current"):
        try:
            # Resolve the runner's actual unified cgroup; never traverse outside it.
            line = next(line for line in Path(f"/proc/{pid}/cgroup").read_text().splitlines()
                        if line.startswith("0::"))
            relative = Path(line[3:].lstrip("/"))
            require(".." not in relative.parts, "invalid cgroup path")
            cgroup[name] = (Path("/sys/fs/cgroup") / relative / name).read_text().strip()
        except (OSError, StopIteration, ValueError):
            cgroup[name] = "unavailable"
    disk = os.statvfs(root)
    return {"at": utc(), "root_pid": pid, "processes": processes, "cgroup": cgroup,
            "disk_available_bytes": disk.f_bavail * disk.f_frsize,
            "diagnostic_only": True}


def telemetry(pid, root, output):
    with output.open("x") as log:
        while Path(f"/proc/{pid}").exists():
            log.write(json.dumps(diagnostic_sample(pid, root)) + "\n")
            log.flush()
            time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--run", type=int, required=True)
    p.add_argument("--attempt", type=int, required=True)
    p.add_argument("--controller", required=True)
    p = commands.add_parser("verify")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--plan-sha", required=True)
    p.add_argument("--manifest-sha", required=True)
    p = commands.add_parser("attest")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--shard", choices=LEAVES, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--plan-sha", required=True)
    p.add_argument("--manifest-sha", required=True)
    p = commands.add_parser("assemble")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--leaves", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--plan-sha", required=True)
    p.add_argument("--manifest-sha", required=True)
    p = commands.add_parser("telemetry")
    p.add_argument("--pid", type=int, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            identity = {"repository": recovery.REPOSITORY, "run_id": args.run,
                        "attempt": args.attempt, "controller_revision": args.controller}
            prepare(args.source.resolve(), args.plan.resolve(), identity)
        elif args.command == "verify":
            checked_inputs(args.plan.resolve(), args.plan_sha, args.manifest_sha)
        elif args.command == "attest":
            attest_leaf(args.source.resolve(), args.plan.resolve(), args.shard, args.output,
                        args.plan_sha, args.manifest_sha)
        elif args.command == "assemble":
            assemble(args.plan.resolve(), args.leaves.resolve(), args.output, args.plan_sha, args.manifest_sha)
        else:
            require(args.pid > 0, "invalid diagnostic root PID")
            telemetry(args.pid, args.source.resolve(), args.output)
    except (ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Nested mutation evidence rejected: {error}\n")


if __name__ == "__main__":
    main()
