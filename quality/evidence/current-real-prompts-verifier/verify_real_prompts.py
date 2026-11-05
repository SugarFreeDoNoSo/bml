#!/usr/bin/env python3
"""Verify completed Hello/Hi bytes and original build receipts; sets no parity floor.

Requires the repository Git object bbb897..., the published build/native archive,
the immutable binary, pinned GGUF/AOT files and both completed full-case folders.
Never executes the binary, Cargo, models or an external network request.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import struct
import subprocess
import sys
import tarfile

REVISION = "bbb89735100e2e4ec6486f3b7029e48977944f3c"
TREE = "81da51727c6598fc70a863f4177b71eb2baf352f"
BINARY_SHA = "f65902f207d85ff0cb0a3ccb148604d1f177199c7de8fdaead0bb0ffbd931c2f"
MODEL_SHA = "da3087fb14aede55fde6eb81a0e55e886810e43509ec82ecdc7aa5d62a03b556"
MANIFEST_SHA = "0b6e9bc33693e9581676fad805fec2cf2173410c1ba112ddce99f490709bbd97"
ARCHIVE_SHA = "b841fbe317c8dd96e37f6975f606adfac1adf365bb38d5cb625cd27a478f8528"
BUNDLE_SHA = "2e9a252c202bd5fa03c2f89444d7dd564871ba7ae5c5ac36eaa5dd9cc6a74628"
PROMPTS = {"Hello": [1, 15043], "Hi": [1, 6324]}
VOCAB = 32000
TEST = "real_model_two_position_prompt_symbolic_logit_diagnostic"
MODE = "full_symbolic_prompt_diagnostic"
FAMILIES = ["attention_dot", "dot", "down_dot", "residual_add", "rms_norm",
            "rope", "scale", "softmax", "swiglu", "weighted_sum"]
BUILD = ["cargo", "test", "--release", "--verbose", "-p", "bml-runtime",
         "--test", "real_model_two_position_diagnostic", "--locked", "--no-run",
         "--message-format=json"]
NUMERIC = {"bml_gguf", "bml_domain", "bml_model_weights", "bml_inference", "bml_compiler", "bml_runtime"}


def require(value, reason):
    if not value:
        raise ValueError(reason)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(data)
    return result.hexdigest()


def read_json(path):
    return json.loads(path.read_bytes())


def same_json(left, right):
    # Keep JSON types distinct: Python otherwise equates true,1 and1.0.
    return json.dumps(left, sort_keys=True, separators=(",", ":"), allow_nan=False) == json.dumps(
        right, sort_keys=True, separators=(",", ":"), allow_nan=False)


def last_argmax(values):
    # Rust total_cmp orders +0 above -0; max_by retains the last equal value.
    return max(range(len(values)), key=lambda i: (values[i], math.copysign(1.0, values[i]), i))


def error_statistics(native, symbolic):
    require(len(native) == len(symbolic) > 0, "different vector lengths")
    differences = [abs(a - b) for a, b in zip(native, symbolic)]
    require(all(math.isfinite(x) for x in differences), "nonfinite differences")
    maximum, mean, squares = max(differences), 0.0, 0.0
    for difference in differences:
        mean += difference / len(differences)
        if maximum:
            scaled = difference / maximum
            squares += scaled * scaled
    rms = maximum * math.sqrt(squares / len(differences)) if maximum else 0.0
    require(math.isfinite(mean) and math.isfinite(rms), "nonfinite aggregate statistics")
    return {"max_absolute_error": maximum, "mean_absolute_error": mean, "rms_error": rms}


def read_vector(path, count=VOCAB):
    data = path.read_bytes()
    require(len(data) == count * 8, "vector must contain exactly finite32000 f64LE values: " + str(path))
    values = struct.unpack(f"<{count}d", data)
    require(all(math.isfinite(value) for value in values), "nonfinite logits: " + str(path))
    return values, {"file": path.name, "bytes": len(data), "f64le_values": count,
                    "all_finite": True, "sha256": digest(data), "argmax": last_argmax(values)}


def read_build_archive(repository):
    bundle_path = repository / "quality/evidence/current-real-prompts-build-native.bundle.json"
    require(sha(bundle_path) == BUNDLE_SHA, "published build bundle digest differs")
    bundle = read_json(bundle_path)
    require(bundle["status"] == "verified_build_and_native_only_capture"
            and bundle["source_revision"] == REVISION and bundle["sha256"] == ARCHIVE_SHA,
            "build bundle identity differs")
    archive = repository / bundle["archive"]
    require(sha(archive) == ARCHIVE_SHA and archive.stat().st_size == bundle["bytes"], "build archive differs")
    expected = {row["path"]: row for row in bundle["members"]}
    require(len(expected) == len(bundle["members"]) == 27, "build archive inventory differs")
    values = {}
    with tarfile.open(archive, "r:gz") as stream:
        for entry in stream:
            prefix = "current-real-prompts-build-native/"
            require(entry.isfile() and entry.name.startswith(prefix), "unsafe build archive member")
            name = entry.name[len(prefix):]
            path = PurePosixPath(name)
            require(name and not path.is_absolute() and ".." not in path.parts and "\\" not in name
                    and name not in values and name in expected and 0 <= entry.size <= 2 * 1024 * 1024,
                    "unsafe or unexpected build archive path")
            data = stream.extractfile(entry).read()
            require(len(data) == expected[name]["bytes"] and digest(data) == expected[name]["sha256"],
                    "build archive member differs: " + name)
            values[name] = data
    require(set(values) == set(expected), "missing build archive member")
    return values, {"bundle_sha256": BUNDLE_SHA, "archive_sha256": ARCHIVE_SHA,
                    "archive_bytes": archive.stat().st_size, "members": len(values)}


def verify_git_inputs(repository, captured):
    tree = subprocess.check_output(["git", "rev-parse", REVISION + "^{tree}"], cwd=repository).decode().strip()
    require(tree == TREE, "source tree differs")
    rows = subprocess.check_output(["git", "ls-tree", "-r", "-z", REVISION], cwd=repository).split(b"\0")
    entries = {}
    for row in filter(None, rows):
        meta, path = row.split(b"\t", 1)
        mode, kind, oid = meta.decode().split()
        require(kind == "blob" and mode in {"100644", "100755", "120000"}, "unsupported source tree entry")
        entries[path.decode()] = oid
    require(set(entries) == set(captured) and len(entries) == 796, "captured tracked inputs differ from Git tree")
    # Blob IDs avoid path quoting and permit one read-only batch operation.
    request = "\n".join(entries.values()).encode() + b"\n"
    stream = io.BytesIO(subprocess.check_output(["git", "cat-file", "--batch"], input=request, cwd=repository))
    for name, oid in entries.items():
        header = stream.readline().decode().strip().split()
        require(len(header) == 3 and header[:2] == [oid, "blob"], "missing source blob")
        data = stream.read(int(header[2]))
        saved = captured[name]
        require(stream.read(1) == b"\n" and saved["kind"] == "file"
                and len(data) == saved["bytes"] and digest(data) == saved["sha256"],
                "captured source input differs: " + name)
    require(stream.read() == b"", "extra source blob bytes")
    return {"revision": REVISION, "tree": tree, "tracked_inputs": len(entries), "all_captured_hashes_match_git": True}


def verify_build(repository, archived, binary):
    report = json.loads(archived["build-report.json"])
    require(report["source_revision"] == REVISION and report["source_tree"] == TREE
            and report["cargo_exit_code"] == 0 and report["command"] == BUILD
            and report["source_inputs_unchanged"] is True, "original build receipt differs")
    require(digest(archived["source-before.json"]) == report["source_before_sha256"]
            == digest(archived["source-after.json"]) == report["source_after_sha256"], "build source snapshots differ")
    source = verify_git_inputs(repository, json.loads(archived["source-before.json"]))
    require(digest(archived["cargo-build.jsonl"]) == report["stdout_sha256"]
            and digest(archived["cargo-build.stderr"]) == report["stderr_sha256"], "original compiler capture differs")
    messages = [json.loads(line) for line in archived["cargo-build.jsonl"].splitlines() if line]
    require(messages[-1] == {"reason": "build-finished", "success": True}, "compiler did not complete successfully")
    driver = report["driver_artifact"]
    require(driver in messages and driver["fresh"] is False and driver["target"]["name"] == "real_model_two_position_diagnostic"
            and driver["target"]["kind"] == ["test"] and driver["profile"]["opt_level"] == "3", "driver compilation differs")
    numeric = report["numeric_artifacts"]
    require(len(numeric) == 6 and {row["target"]["name"] for row in numeric} == NUMERIC
            and all(row in messages and row["fresh"] is False for row in numeric), "numeric library compilation was not fresh")
    require(report["immutable_binary_sha256"] == BINARY_SHA and sha(binary) == BINARY_SHA
            and binary.stat().st_size == report["immutable_binary_bytes"], "immutable measured binary differs")
    require("1.98.1" in report["toolchain"]["rustc"] and report["explicit_environment"]["RUSTFLAGS"] == ""
            and report["explicit_environment"]["CARGO_ENCODED_RUSTFLAGS"] == "", "build toolchain/flags differ")
    return report, {"source": source, "binary_sha256": BINARY_SHA,
                    "build_report_sha256": digest(archived["build-report.json"]),
                    "original_compiler_stdout_sha256": report["stdout_sha256"],
                    "original_compiler_stderr_sha256": report["stderr_sha256"], "fresh_numeric_libraries": sorted(NUMERIC)}


def verify_external_inputs(protocol, model, artifact):
    require(protocol["source_sha256"] == MODEL_SHA and sha(model) == MODEL_SHA, "pinned GGUF differs")
    expected = protocol["artifact_files"]
    require(len(expected) == 13 and len({row["name"] for row in expected}) == 13, "AOT inventory differs")
    require({path.name for path in artifact.iterdir()} == {row["name"] for row in expected}, "AOT directory files differ")
    for row in expected:
        path = artifact / row["name"]
        require(path.name == row["name"] and path.is_file() and not path.is_symlink()
                and path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "pinned AOT input differs")
    manifest = read_json(artifact / "symbolic.bmlgraph.json")
    require(sha(artifact / "symbolic.bmlgraph.json") == MANIFEST_SHA
            and manifest["version"] == 4 and manifest["context_length"] == 2
            and manifest["source_sha256"] == MODEL_SHA and manifest["bounded_profile"] == protocol["profile"],
            "AOT manifest contract differs")
    return {"model_sha256": MODEL_SHA, "aot_manifest_sha256": MANIFEST_SHA, "aot_files": expected}


def full_cases(report):
    require(report.get("status") == "all_cases_completed_unverified" and report.get("source_revision") == REVISION
            and report.get("binary_sha256") == BINARY_SHA, "both full cases must finish before verification")
    rows = report.get("cases", [])
    require(len(rows) == 2 and {row.get("prompt") for row in rows} == set(PROMPTS), "missing or duplicate full prompt")
    require(all(row.get("status") == "completed_unverified" and row.get("exit_code") == 0 for row in rows),
            "partial or unsuccessful full prompt")
    return {row["prompt"]: row for row in rows}


def verify_case(directory, prompt, case, archived_protocol, expected_binary_path):
    ids = PROMPTS[prompt]
    require(case["prompt"] == prompt and same_json(case["input_ids"], ids)
            and case["command"] == [expected_binary_path, TEST, "--ignored", "--exact", "--nocapture"],
            "case command or predeclared tokens differ")
    origin = Path(expected_binary_path).parent
    require(case["output_directory"] == str(origin / ("full-" + prompt.lower()))
            and case["log"] == str(origin / ("full-" + prompt.lower() + ".log")), "original case output paths differ")
    started = datetime.fromisoformat(case["started_at_utc"].replace("Z", "+00:00"))
    ended = datetime.fromisoformat(case["ended_at_utc"].replace("Z", "+00:00"))
    require(started.tzinfo is not None and ended.tzinfo is not None and started < ended, "case chronology differs")
    folder, log_path, exit_path = directory / ("full-" + prompt.lower()), directory / ("full-" + prompt.lower() + ".log"), directory / ("full-" + prompt.lower() + ".exit")
    require({p.name for p in folder.iterdir()} == {"protocol.json", "summary.json", "position-0.json", "position-1.json",
            "native-0.f64le", "native-1.f64le", "symbolic-0.f64le", "symbolic-1.f64le"}, "incomplete or unexpected full-case files")
    protocol_path = folder / "protocol.json"
    protocol, summary = read_json(protocol_path), read_json(folder / "summary.json")
    require(same_json(protocol, {**archived_protocol, "mode": MODE}), "full protocol differs from predeclared authenticated input contract")
    require(protocol["source_revision_declared"] == REVISION and protocol["binary_sha256"] == BINARY_SHA
            and protocol["selected_prompt"] == prompt and protocol["rust_encoded_input_ids"] == ids,
            "full protocol source/binary/tokens differ")
    require(protocol["predeclared_cases"] == [{"input_ids": value, "prompt": key} for key, value in PROMPTS.items()]
            and protocol["context"] == protocol["cache_capacity_per_backend"] == 2
            and protocol["symbolic_limits"]["max_nodes"] == 0, "full protocol is not exact context2")
    protocol_sha = sha(protocol_path)
    require(summary["status"] == "full_completed_diagnostic" and summary["prompt"] == prompt
            and same_json(summary["input_ids"], ids) and summary["protocol_sha256"] == protocol_sha
            and summary["executed_operations"] == FAMILIES and len(summary["positions"]) == 2,
            "full summary contract differs")
    require(sha(folder / "summary.json") == case["summary_sha256"] and int(exit_path.read_text()) == 0,
            "summary digest or saved exit differs")
    require(sha(log_path) == case["log_sha256"], "full log digest differs")
    log = log_path.read_text()
    require(re.search(r"^test " + re.escape(TEST) + r" \.\.\. ok$", log, re.M)
            and re.search(r"^test result: ok\. 1 passed; 0 failed;", log, re.M), "exact full test did not pass")
    logged = []
    for line in log.splitlines():
        start = line.find("{")
        if start >= 0:
            try:
                value = json.loads(line[start:])
            except ValueError:
                continue
            if isinstance(value, dict) and value.get("status") == "full_position_completed_diagnostic":
                logged.append(value)
    require(same_json(logged, summary["positions"]), "printed positions differ from saved summary")
    vectors, reports = [], []
    for position, token in enumerate(ids):
        saved = read_json(folder / f"position-{position}.json")
        require(same_json(saved, summary["positions"][position]) and saved["status"] == "full_position_completed_diagnostic"
                and saved["prompt"] == prompt and type(saved["position"]) is int and saved["position"] == position
                and type(saved["input_token"]) is int and saved["input_token"] == token
                and type(saved["logits"]) is int and saved["logits"] == VOCAB
                and type(saved["native_cache_position"]) is int and type(saved["symbolic_cache_position"]) is int
                and saved["native_cache_position"] == saved["symbolic_cache_position"] == position + 1
                and saved["protocol_sha256"] == protocol_sha
                and math.isfinite(saved["symbolic_elapsed_seconds"]) and saved["symbolic_elapsed_seconds"] > 0,
                "position/tokens/cache/elapsed metadata differs")
        values = []
        for kind in ("native", "symbolic"):
            name = f"{kind}-{position}.f64le"
            require(saved[kind + "_vector"] == name, "unexpected vector path")
            data, receipt = read_vector(folder / name)
            require(type(saved[kind + "_argmax"]) is int and saved[kind + "_argmax"] == receipt["argmax"],
                    "saved argmax differs from bytes")
            vectors.append(receipt)
            values.append(data)
        stats = error_statistics(*values)
        require(all(type(saved[key]) in (int, float) and math.isfinite(saved[key]) and saved[key] == value
                    for key, value in stats.items()), "saved error statistics differ from raw vectors")
        reports.append({**saved, "recomputed": stats, "statistics_match_exactly": True})
    return {"prompt": prompt, "input_ids": ids, "protocol_sha256": protocol_sha,
            "summary_sha256": case["summary_sha256"], "log_sha256": case["log_sha256"],
            "positions": reports, "vectors": vectors, "executed_operations": FAMILIES,
            "saved_files": [{"file": path.name, "bytes": path.stat().st_size, "sha256": sha(path)}
                            for path in sorted(folder.iterdir())], "exit_file_sha256": sha(exit_path)}


def verify(repository, directory, binary=None, model=None, artifact=None):
    controller_sha = sha(directory / "full-cases.json")
    current = read_json(directory / "full-cases.json")
    cases = full_cases(current)  # Fail before expensive hashing if either prompt is still pending.
    archived, archive_receipt = read_build_archive(repository)
    build = json.loads(archived["build-report.json"])
    binary = binary or Path(build["immutable_binary"])
    build, build_receipt = verify_build(repository, archived, binary)
    require(current["build_report_sha256"] == digest(archived["build-report.json"])
            and current["native_preflights_sha256"] == digest(archived["native-preflights.json"])
            and current["binary"] == build["immutable_binary"], "full controller build linkage differs")
    native = json.loads(archived["native-preflights.json"])
    require(native["status"] == "native_preflights_passed" and len(native["cases"]) == 2
            and all(row["passed"] for row in native["cases"]), "necessary native preflight missing")
    protocols = {prompt: json.loads(archived[f"native-{prompt.lower()}/protocol.json"]) for prompt in PROMPTS}
    for prompt, ids in PROMPTS.items():
        preflight = json.loads(archived[f"native-{prompt.lower()}/native-preflight-summary.json"])
        require(preflight["status"] == "native_only_preflight_pass" and preflight["bml_executed"] is False
                and preflight["prompt"] == prompt and preflight["input_ids"] == ids
                and [row["native_cache_position"] for row in preflight["positions"]] == [1, 2],
                "native-only preflight contract differs; it cannot count as BML execution")
    shared = protocols["Hello"]
    external = verify_external_inputs(shared, model or Path(shared["source"]), artifact or Path(shared["artifact"]))
    results = [verify_case(directory, prompt, cases[prompt], protocols[prompt], build["immutable_binary"])
               for prompt in PROMPTS]
    require(sha(directory / "full-cases.json") == controller_sha, "full controller report changed during verification")
    return {"status": "verified_completed_Hello_and_Hi_diagnostics", "verified_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "Both predeclared prompts; two positions and four complete vectors per prompt. Byte/report/source-build integrity; no preset numerical tolerance, global parity, throughput or performance claim.",
            "source_revision": REVISION, "build_archive": archive_receipt, "original_build": build_receipt,
            "external_inputs": external, "cases": results, "full_controller_report_sha256": controller_sha,
            "verifier_sha256": sha(Path(__file__))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("--binary", type=Path, help="Relocated immutable binary; pinned hash still required")
    parser.add_argument("--model", type=Path, help="Relocated GGUF; pinned hash still required")
    parser.add_argument("--artifact", type=Path, help="Relocated13-file AOT; pinned hashes still required")
    args = parser.parse_args()
    try:
        result = verify(args.repository.resolve(), args.results.resolve(), args.binary, args.model, args.artifact)
    except (ValueError, KeyError, TypeError, OSError, subprocess.SubprocessError, tarfile.TarError) as error:
        print("Real-prompt evidence rejected: " + str(error), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
