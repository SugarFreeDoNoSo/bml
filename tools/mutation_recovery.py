#!/usr/bin/env python3
"""Recover one reviewed failed shard of the immutable 9a campaign with a fresh run.

This controller never starts jobs or runs cargo. The workflow executes the
original snapshot's runner/sealer; this adapter authenticates its inputs and
records a separate recovery provenance, without changing original outcomes.
If only the frozen sealer's final untracked-file guard fails on a real generated
proptest seed, the versioned controller can attest the still-live source and
complete raw reports itself. That distinct seal never claims to come from 9a.

The pinned original shard21 and shard27 ZIPs may establish retry eligibility only.
Their complete raw reports lack an attested source/seed inventory, so they are never
copied into the fresh source checkout or converted into a source seal. Archives
are checksum-verified and inspected in memory, then retained under control's
target directory solely as provenance. A fresh baseline and all assigned
mutants must execute before the existing live-source attestation can succeed.

The original aggregate cannot use the missing shard20 seal or unattested
shard21/shard27 reports, and does not discover these separate recovery runs. Collection
is an explicit later operation: retain exactly one attested seal for each of
the 64 indices, replacing only reviewed failures with fresh recoveries. Before selecting one,
validate its recovery metadata against the API's actual recovery run/attempt,
reviewed controller revision, artifact identity and original manifest digest.
Never select a stale metadata file or two results for the same shard. Only
then use the original merge_shards protocol for the 64-way candidate union and
the unchanged quality gate. merge_shards alone does not check recovery-run
provenance. A complete seal preserves missed/timeout outcomes and is not a
quality approval. No rerun or aggregation is automatic here.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import stat
import subprocess
import zipfile

from merge_mutation_shards import (
    digest, file_digest, read_json, require, sha, validate_manifest,
    validate_outcomes, write_new, verify_worktree, seal_shard, git,
)

REPOSITORY = "SugarFreeDoNoSo/bml"
REPOSITORY_ID = 1291412160
ORIGIN_RUN = 41950170549
ARTIFACT_ID = 13379899471
ORIGIN_ATTEMPT = 1
SOURCE = "9a6bc9680d5915576ca92528f60956efe9796cef"
MANIFEST_SHA256 = "719be79f7dfba0746291522fa312b395a9caf857abf16e71f61af70b8a5e7c63"
SHARDS = 64
UNTRACKED_ERROR = "Mutation campaign evidence rejected: untracked source/input files present"
COLLECTION_INDEX = 21
COLLECTION_JOB = 126992338544
COLLECTION_ARTIFACT = 13386138847
COLLECTION_ZIP_SHA256 = "0ecabdc7b8758dcf7b8686f0e145de9c4e9f5cdfad54b085fc77ed6af52f5b50"
COLLECTION_27_JOB = 126992339813
COLLECTION_27_ARTIFACT = 13389245750
COLLECTION_27_ZIP_SHA256 = "faa89ea81b3b6624da0563d752b46ea0cda78d712b4edb5413c8df97dbb1009f"
MAX_INSPECTION_ZIP = 32 * 1024 * 1024


def collection_failure(index):
    """Return only the two independently reviewed collection-failure identities."""
    if type(index) is not int:
        return None
    if index == COLLECTION_INDEX:
        return COLLECTION_JOB, COLLECTION_ARTIFACT, COLLECTION_ZIP_SHA256
    if index == 27:
        return COLLECTION_27_JOB, COLLECTION_27_ARTIFACT, COLLECTION_27_ZIP_SHA256
    return None


def collection_requested(job_id, index):
    pinned = collection_failure(index)
    return pinned is not None and job_id == pinned[0]


def zip_members(blob, required):
    """Read only named bounded members in memory; never extract archive paths."""
    require(len(blob) <= MAX_INSPECTION_ZIP, "inspection ZIP exceeds its byte budget")
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            entries = archive.infolist()
            require(len(entries) <= 1024 and sum(e.file_size for e in entries) <= 128 * 1024 * 1024,
                    "inspection ZIP expands beyond its budget")
            names = []
            for entry in entries:
                name = entry.filename.rstrip("/")
                path = Path(name)
                require(name and not path.is_absolute() and ".." not in path.parts
                        and "\\" not in name and ":" not in name
                        and entry.orig_filename == entry.filename
                        and not stat.S_ISLNK(entry.external_attr >> 16)
                        and not entry.flag_bits & 1 and 0 <= entry.file_size <= MAX_INSPECTION_ZIP,
                        "unsafe or oversized inspection ZIP entry")
                require(entry.compress_type in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED},
                        "unsupported inspection ZIP compression")
                names.append(entry.filename)
            require(len(set(names)) == len(names), "duplicate inspection ZIP entry")
            require(required <= set(names), "inspection ZIP lacks required evidence")
            return {name: archive.read(name) for name in required}, names
    except (zipfile.BadZipFile, RuntimeError) as error:
        raise ValueError(f"invalid inspection ZIP: {error}") from error


def inspection_plan(blob, index=COLLECTION_INDEX):
    require(collection_failure(index) is not None, "collection shard is not independently reviewed")
    values, _ = zip_members(blob, {"campaign.json", "final.diff"})
    manifest = checked_plan_bytes(values["campaign.json"], values["final.diff"], f"{index}/{SHARDS}")
    return manifest


def inspect_collection_zip(blob, manifest, index=COLLECTION_INDEX):
    pinned = collection_failure(index)
    require(pinned is not None, "collection shard is not independently reviewed")
    _, artifact_id, zip_sha256 = pinned
    require(hashlib.sha256(blob).hexdigest() == zip_sha256,
            "original collection ZIP digest differs")
    base = f"target/final-mutation-shard-{index}/mutants.out/"
    evidence = f"target/quality/shard-{index}/"
    required = {base + "mutants.json", base + "outcomes.json",
                evidence + "runner.log", evidence + "exit-code.txt"}
    values, names = zip_members(blob, required)
    require(not any(name.endswith((f"/shard-result-{index}.json", f"/controller-shard-result-{index}.json"))
                    for name in names), "collection failure unexpectedly contains a source seal")
    lines = values[evidence + "runner.log"].decode("utf-8").rstrip().splitlines()
    require(lines and lines[-1] == UNTRACKED_ERROR
            and values[evidence + "exit-code.txt"].decode("ascii").strip() == "1",
            "original collection did not end at the untracked-file guard with exit 1")
    candidates = json.loads(values[base + "mutants.json"])
    assigned = manifest["shards"][index]["candidates"]
    require(manifest["source"]["revision"] == SOURCE
            and manifest["shards"][index]["shard"] == f"{index}/{SHARDS}"
            and candidates == assigned, "original collection candidates differ from pinned manifest")
    outcomes = json.loads(values[base + "outcomes.json"])
    validate_outcomes(outcomes, assigned, manifest["source"]["workspace_packages"])
    return {"state": "retry_eligible_from_collection_failure",
            "artifact_id": artifact_id, "zip_sha256": zip_sha256,
            "candidate_count": len(candidates),
            "raw_counts": {key: outcomes[key] for key in ["caught", "missed", "unviable", "timeout"]},
            "members_sha256": {name: hashlib.sha256(value).hexdigest() for name, value in values.items()},
            "source_state": "unattested; original live source and seed inventory are unavailable",
            "fresh_execution_required": True}


def validate_origin(repository, job_id, index, bundle, log, collection=None):
    require(repository == REPOSITORY, "recovery is restricted to the original repository")
    require(type(job_id) is int and job_id > 0 and type(index) is int
            and 0 <= index < SHARDS, "invalid original job or shard index")
    run, artifact, job, artifacts = (bundle[k] for k in ["run", "artifact", "job", "artifacts"])
    require(run.get("id") == ORIGIN_RUN and run.get("head_sha") == SOURCE
            and run.get("run_attempt") == ORIGIN_ATTEMPT
            and run.get("event") == "workflow_dispatch"
            and run.get("path") == ".github/workflows/mutation-campaign.yml"
            and run.get("repository", {}).get("full_name") == repository
            and run["repository"].get("id") == REPOSITORY_ID,
            "original run repository, source or attempt differs")
    require(artifact.get("id") == ARTIFACT_ID and artifact.get("expired") is False
            and artifact.get("name") == f"mutation-plan-{ORIGIN_RUN}-{ORIGIN_ATTEMPT}"
            and artifact.get("workflow_run", {}).get("id") == ORIGIN_RUN
            and artifact["workflow_run"].get("head_sha") == SOURCE
            and artifact["workflow_run"].get("repository_id") == REPOSITORY_ID
            and artifact["workflow_run"].get("head_repository_id") == REPOSITORY_ID,
            "original plan artifact is expired or belongs to another source")
    require(job.get("id") == job_id and job.get("run_id") == ORIGIN_RUN
            and job.get("run_attempt") == ORIGIN_ATTEMPT and job.get("head_sha") == SOURCE
            and job.get("name") == f"shard ({index}, {index}/{SHARDS})"
            and job.get("status") == "completed" and job.get("conclusion") == "failure",
            "original shard job is not the matching terminal failure")
    steps = job.get("steps", [])
    execution = [s for s in steps if s.get("name") ==
                 "Run baseline and workspace mutation tests; seal completed outcomes"]
    uploads = [s for s in steps if s.get("name") ==
               "Preserve raw reports, candidates, logs, seals and generated regressions"]
    require(len(execution) == len(uploads) == 1
            and execution[0].get("conclusion") == "failure"
            and any(s.get("name") == "Check independent prepare digest and frozen revision"
                    and s.get("conclusion") == "success" for s in steps),
            "original job did not fail after admission and before evidence upload")
    rows = artifacts.get("artifacts", [])
    require(artifacts.get("total_count") == len(rows) and len(rows) <= 100,
            "incomplete original artifact inventory")
    require(any(a.get("id") == ARTIFACT_ID for a in rows), "original plan is not in run inventory")
    name = f"mutation-shard-{ORIGIN_RUN}-{ORIGIN_ATTEMPT}-{index}"
    if collection_requested(job_id, index):
        _, collection_artifact, collection_sha256 = collection_failure(index)
        original = bundle.get("collection_artifact", {})
        require(uploads[0].get("conclusion") == "success"
                and re.search(r"Z ##\[error\]Process completed with exit code 1\.\r?$", log, re.M)
                and original.get("id") == collection_artifact and original.get("name") == name
                and original.get("expired") is False
                and original.get("digest") == "sha256:" + collection_sha256
                and 0 < original.get("size_in_bytes", 0) <= MAX_INSPECTION_ZIP
                and original.get("workflow_run", {}).get("id") == ORIGIN_RUN
                and original["workflow_run"].get("head_sha") == SOURCE
                and original["workflow_run"].get("repository_id") == REPOSITORY_ID
                and original["workflow_run"].get("head_repository_id") == REPOSITORY_ID
                and len([a for a in rows if a.get("name") == name]) == 1
                and any(a.get("name") == name and a.get("id") == collection_artifact for a in rows),
                "original collection artifact, exit or upload does not match the pinned failure")
        require(collection and collection.get("state") == "retry_eligible_from_collection_failure"
                and collection.get("artifact_id") == collection_artifact
                and collection.get("zip_sha256") == collection_sha256
                and collection.get("fresh_execution_required") is True,
                "original collection inspection is missing or invalid")
        return {"state": "retry_eligible_from_collection_failure", "exit_code": 1}
    require(uploads[0].get("conclusion") == "skipped"
            and re.search(r"Z ##\[error\]Process completed with exit code 143\.\r?$", log, re.M)
            and re.search(r"Z ##\[error\]The runner has received a shutdown signal\.", log),
            "original job does not prove a runner shutdown with exit 143")
    require(not any(a.get("name") == name for a in rows),
            "original shard already has evidence; recovery requires separate review")
    return {"state": "retry_eligible_from_runner_shutdown", "exit_code": 143}


def api_bytes(path):
    command = ["gh", "api"]
    # Capture opaque logs/ZIPs without terminal rendering or newline conversion.
    # Feature detection retains compatibility with older installed gh versions.
    help_text = subprocess.check_output(["gh", "api", "--help"], text=True)
    if "--allow-escape-sequences" in help_text:
        command.append("--allow-escape-sequences")
    return subprocess.check_output([*command, path])


def api(path, raw=False):
    if raw:
        return api_bytes(path).decode("utf-8")
    return json.loads(subprocess.check_output(["gh", "api", path], text=True))


def authorize(repository, job_id, index, run_id, attempt, controller, output):
    sha(controller, "controller revision", 40)
    require(run_id > 0 and run_id != ORIGIN_RUN and attempt > 0,
            "recovery must be a separate workflow run")
    require(repository == REPOSITORY, "unexpected recovery repository")
    prefix = f"repos/{repository}/actions"
    bundle = {"run": api(f"{prefix}/runs/{ORIGIN_RUN}"),
              "artifact": api(f"{prefix}/artifacts/{ARTIFACT_ID}"),
              "job": api(f"{prefix}/jobs/{job_id}"),
              "artifacts": api(f"{prefix}/runs/{ORIGIN_RUN}/artifacts?per_page=100")}
    log = api(f"{prefix}/jobs/{job_id}/logs", raw=True)
    collection, collection_zip, plan_zip = None, None, None
    if collection_requested(job_id, index):
        _, collection_artifact, _ = collection_failure(index)
        bundle["collection_artifact"] = api(f"{prefix}/artifacts/{collection_artifact}")
        plan_zip = api_bytes(f"{prefix}/artifacts/{ARTIFACT_ID}/zip")
        manifest = inspection_plan(plan_zip, index)
        collection_zip = api_bytes(f"{prefix}/artifacts/{collection_artifact}/zip")
        collection = inspect_collection_zip(collection_zip, manifest, index)
    eligibility = validate_origin(repository, job_id, index, bundle, log, collection)
    output.mkdir(parents=True, exist_ok=True)
    write_new(output / "origin-api.json", bundle)
    with (output / "origin-job.log").open("x") as stream:
        stream.write(log)
    record = {"format_version": 1, "repository": repository,
              "origin_run_id": ORIGIN_RUN, "origin_attempt": ORIGIN_ATTEMPT,
              "origin_job_id": job_id, "origin_exit_code": eligibility["exit_code"],
              "eligibility_state": eligibility["state"],
              "artifact_id": ARTIFACT_ID, "source_revision": SOURCE,
              "manifest_file_sha256": MANIFEST_SHA256,
              "shard": f"{index}/{SHARDS}", "index": index,
              "recovery_run_id": run_id, "recovery_attempt": attempt,
              "controller_revision": controller,
              "origin_api_sha256": file_digest(output / "origin-api.json"),
              "origin_log_sha256": file_digest(output / "origin-job.log")}
    if collection is not None:
        for name, blob in [("inspection-plan.zip", plan_zip), ("original-collection.zip", collection_zip)]:
            with (output / name).open("xb") as stream:
                stream.write(blob)
        record.update(collection_inspection=collection,
                      inspection_plan_zip_sha256=file_digest(output / "inspection-plan.zip"),
                      original_collection_zip_sha256=file_digest(output / "original-collection.zip"))
    write_new(output / "authorization.json", record)


def authorization(directory):
    record = read_json(directory / "authorization.json")
    require(record.get("format_version") == 1 and record.get("origin_run_id") == ORIGIN_RUN
            and record.get("origin_attempt") == ORIGIN_ATTEMPT
            and record.get("origin_exit_code") ==
            (1 if collection_requested(record.get("origin_job_id"), record.get("index")) else 143)
            and record.get("artifact_id") == ARTIFACT_ID
            and record.get("source_revision") == SOURCE
            and record.get("manifest_file_sha256") == MANIFEST_SHA256
            and record.get("shard") == f"{record.get('index')}/{SHARDS}"
            and type(record.get("recovery_run_id")) is int
            and record["recovery_run_id"] > 0 and record["recovery_run_id"] != ORIGIN_RUN
            and type(record.get("recovery_attempt")) is int and record["recovery_attempt"] > 0,
            "recovery authorization identity changed")
    sha(record["controller_revision"], "controller revision", 40)
    require(file_digest(directory / "origin-api.json") == record["origin_api_sha256"]
            and file_digest(directory / "origin-job.log") == record["origin_log_sha256"],
            "original API or log evidence changed")
    collection = None
    if collection_requested(record["origin_job_id"], record["index"]):
        _, _, collection_sha256 = collection_failure(record["index"])
        require(file_digest(directory / "inspection-plan.zip") == record.get("inspection_plan_zip_sha256")
                and file_digest(directory / "original-collection.zip") ==
                record.get("original_collection_zip_sha256") == collection_sha256,
                "original inspection archives changed")
        manifest = inspection_plan((directory / "inspection-plan.zip").read_bytes(), record["index"])
        collection = inspect_collection_zip((directory / "original-collection.zip").read_bytes(),
                                            manifest, record["index"])
        require(collection == record.get("collection_inspection"), "collection inspection record changed")
    eligibility = validate_origin(record["repository"], record["origin_job_id"], record["index"],
                    read_json(directory / "origin-api.json"),
                    (directory / "origin-job.log").read_text(), collection)
    require(record.get("eligibility_state", eligibility["state"]) == eligibility["state"],
            "recovery eligibility state changed")
    return record


def checked_plan_bytes(manifest_bytes, diff_bytes, shard):
    require(hashlib.sha256(manifest_bytes).hexdigest() == MANIFEST_SHA256,
            "original manifest file digest differs")
    manifest = json.loads(manifest_bytes)
    assignments = validate_manifest(manifest)
    require(manifest["source"]["revision"] == SOURCE and len(assignments) == SHARDS
            and shard in assignments, "original manifest source or partition differs")
    require(hashlib.sha256(diff_bytes).hexdigest() == manifest["diff_sha256"], "original scope diff differs")
    return manifest


def checked_plan(manifest_path, diff_path, shard):
    return checked_plan_bytes(manifest_path.read_bytes(), diff_path.read_bytes(), shard)


def verify_plan(directory, manifest_path, diff_path):
    record = authorization(directory)
    manifest = checked_plan(manifest_path, diff_path, record["shard"])
    return record, manifest


def controller_guard_recovery(source, manifest, diff_path, index):
    """Attest only a complete run whose frozen final guard rejected a real seed."""
    original = source / f"target/quality/shard-result-{index}.json"
    require(not original.exists(), "an original seal exists; controller fallback is forbidden")
    log = source / f"target/quality/shard-{index}/runner.log"
    lines = log.read_text().rstrip().splitlines()
    require(lines and lines[-1] == UNTRACKED_ERROR,
            "exit 1 is not solely the frozen final untracked-file guard")
    untracked = git(source.resolve(), "ls-files", "--others", "--exclude-standard").splitlines()
    require(any(path.endswith(".proptest-regressions") for path in untracked),
            "the live source has no generated fallback proptest seed to explain the guard")
    # The corrected controller checks every tracked byte against the 9a
    # manifest and admits only exact SourceParallel paths with valid seeds.
    verify_worktree(manifest, source.resolve(), diff_path, allow_regressions=True)
    return {path: file_digest(source / path) for path in untracked}


def attest(directory, manifest_path, diff_path, source):
    record, manifest = verify_plan(directory, manifest_path, diff_path)
    index = record["index"]
    evidence = source / f"target/quality/shard-{index}"
    code = int((evidence / "exit-code.txt").read_text())
    if code in {0, 2, 3}:
        result_path = source / f"target/quality/shard-result-{index}.json"
        require((evidence / "final-source-seal.json").read_bytes() == result_path.read_bytes(),
                "final frozen source recheck differs from original recovery seal")
        method = "frozen-9a-sealer"
        tool_hash = manifest["source"]["files"]["tools/merge_mutation_shards.py"]
    else:
        require(code == 1, "only complete original statuses or the narrow exit-1 guard are eligible")
        generated = controller_guard_recovery(source, manifest, diff_path, index)
        raw = source / f"target/final-mutation-shard-{index}/mutants.out"
        candidates, outcomes = read_json(raw / "mutants.json"), read_json(raw / "outcomes.json")
        artifacts = {"directory": str(raw.resolve()),
                     "mutants_json_sha256": file_digest(raw / "mutants.json"),
                     "outcomes_json_sha256": file_digest(raw / "outcomes.json"),
                     "generated_regressions_sha256": generated}
        result = seal_shard(manifest, record["shard"], manifest["source"], candidates,
                            outcomes, artifacts)
        require(controller_guard_recovery(source, manifest, diff_path, index) == generated,
                "live source or generated seeds changed during controller attestation")
        result_path = source / f"target/quality/controller-shard-result-{index}.json"
        write_new(result_path, result)
        method = "controller-live-source-sealer"
        tool_hash = file_digest(Path(__file__).with_name("merge_mutation_shards.py"))
    write_new(evidence / "sealer-info.json", {
        "method": method, "runner_exit_code": code,
        "source_revision": SOURCE, "controller_revision": record["controller_revision"],
        "sealer_tool_sha256": tool_hash, "seal_sha256": file_digest(result_path)})


def finish(directory, manifest_path, diff_path, source):
    record, manifest = verify_plan(directory, manifest_path, diff_path)
    index = record["index"]
    output = {"format_version": 1, "authorization": record,
              "campaign_manifest_sha256": manifest["manifest_sha256"],
              "status": "incomplete", "quality_gate": "not evaluated; complete union required"}
    try:
        code = int((source / f"target/quality/shard-{index}/exit-code.txt").read_text())
        sealer = read_json(source / f"target/quality/shard-{index}/sealer-info.json")
        require(sealer.get("runner_exit_code") == code and sealer.get("source_revision") == SOURCE
                and sealer.get("controller_revision") == record["controller_revision"],
                "sealer provenance differs from this recovery")
        if code in {0, 2, 3}:
            require(sealer.get("method") == "frozen-9a-sealer"
                    and sealer.get("sealer_tool_sha256") ==
                    manifest["source"]["files"]["tools/merge_mutation_shards.py"],
                    "original completion must retain the frozen sealer identity")
            result_path = source / f"target/quality/shard-result-{index}.json"
            final_seal = source / f"target/quality/shard-{index}/final-source-seal.json"
            require(final_seal.read_bytes() == result_path.read_bytes(),
                    "final frozen source recheck differs from original recovery seal")
        else:
            require(code == 1 and sealer.get("method") == "controller-live-source-sealer"
                    and sealer.get("sealer_tool_sha256") ==
                    file_digest(Path(__file__).with_name("merge_mutation_shards.py")),
                    "only the reviewed controller may attest the narrow exit-1 guard")
            generated = controller_guard_recovery(source, manifest, diff_path, index)
            result_path = source / f"target/quality/controller-shard-result-{index}.json"
            require(read_json(result_path).get("artifacts", {}).get("generated_regressions_sha256")
                    == generated, "live generated seeds differ from controller seal")
        require(sealer.get("seal_sha256") == file_digest(result_path), "seal file changed after attestation")
        result = read_json(result_path)
        raw = source / f"target/final-mutation-shard-{index}/mutants.out"
        candidates, outcomes = read_json(raw / "mutants.json"), read_json(raw / "outcomes.json")
        assigned = next(s["candidates"] for s in manifest["shards"] if s["shard"] == record["shard"])
        require(result.get("shard") == record["shard"] and result.get("format_version") == 1
                and result.get("manifest_sha256") == manifest["manifest_sha256"]
                and result.get("source") == manifest["source"]
                and result.get("candidates") == candidates == assigned
                and result.get("outcomes") == outcomes
                and result.get("outcomes_sha256") == digest(outcomes),
                "recovery seal differs from original campaign or raw reports")
        validate_outcomes(outcomes, assigned, manifest["source"]["workspace_packages"])
        artifacts = result.get("artifacts", {})
        require(artifacts.get("mutants_json_sha256") == file_digest(raw / "mutants.json")
                and artifacts.get("outcomes_json_sha256") == file_digest(raw / "outcomes.json"),
                "recovery raw file hashes differ from seal")
        output.update(status="complete_sealed_shard", runner_exit_code=code,
                      cargo_mutants_exit_code=code if code in {0, 2, 3} else None,
                      sealer=sealer, seal_relative_path=str(result_path.relative_to(source)),
                      shard_result_sha256=file_digest(result_path),
                      raw_outcomes_sha256=file_digest(raw / "outcomes.json"),
                      candidates_sha256=file_digest(raw / "mutants.json"))
    except (ValueError, OSError, KeyError, TypeError) as error:
        output["failure"] = str(error)
        write_new(directory / "recovery-metadata.json", output)
        raise
    write_new(directory / "recovery-metadata.json", output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    auth = sub.add_parser("authorize")
    auth.add_argument("--repository", required=True)
    auth.add_argument("--job", type=int, required=True)
    auth.add_argument("--index", type=int, required=True)
    auth.add_argument("--run", type=int, required=True)
    auth.add_argument("--attempt", type=int, required=True)
    auth.add_argument("--controller", required=True)
    for name in ["verify", "attest", "finish"]:
        command = sub.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--diff", type=Path, required=True)
        if name in {"attest", "finish"}:
            command.add_argument("--source", type=Path, required=True)
    for command in sub.choices.values():
        command.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "authorize":
            authorize(args.repository, args.job, args.index, args.run, args.attempt,
                      args.controller, args.evidence)
        elif args.command == "verify":
            verify_plan(args.evidence, args.manifest, args.diff)
        elif args.command == "attest":
            attest(args.evidence, args.manifest, args.diff, args.source)
        else:
            finish(args.evidence, args.manifest, args.diff, args.source)
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Mutation recovery rejected: {error}\n")


if __name__ == "__main__":
    main()
