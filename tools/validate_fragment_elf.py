#!/usr/bin/env python3
"""Bounded validation of trusted, generated native head-fragment ELF bundles.

This checks local process composition, not two hosts, Lambda, a symbolic DAG,
or filesystem isolation. The Docker recipe separately removes model/source
mounts from the execution stage. No ldd, download, build or deployment occurs.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import struct
import subprocess
import tempfile


MAX_JSON_BYTES = 256 * 1024 * 1024
MAX_ELF_BYTES = 1024 * 1024 * 1024
SHA = re.compile(r"[0-9a-f]{64}\Z")
NAME = re.compile(r"[A-Za-z0-9_-]{1,48}\Z")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, minimum, maximum, name):
    require(type(value) is int and minimum <= value <= maximum, f"invalid {name}")
    return value


def digest(value):
    require(isinstance(value, str) and SHA.fullmatch(value), "invalid SHA-256")
    return value


def pairs_unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path, expected_sha256=None):
    path = Path(path)
    require(path.stat().st_size <= MAX_JSON_BYTES, "JSON exceeds byte budget")
    if expected_sha256 is not None:
        require(sha256(path) == digest(expected_sha256), "JSON identity mismatch")
    def reject_constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    return json.loads(path.read_text(), object_pairs_hook=pairs_unique,
                      parse_constant=reject_constant)


def elf_identity(path, expected_hash, expected_bytes=None):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "ELF must be a regular non-symlink file")
    size = path.stat().st_size
    require(64 <= size <= MAX_ELF_BYTES, "ELF byte budget or header size invalid")
    if expected_bytes is not None:
        require(size == integer(expected_bytes, 64, MAX_ELF_BYTES, "ELF bytes"), "ELF size mismatch")
    require(sha256(path) == digest(expected_hash), "ELF identity mismatch")
    with path.open("rb") as handle:
        header = handle.read(64)
    require(header[:7] == b"\x7fELF\x02\x01\x01", "expected 64-bit little-endian ELF")
    kind, machine, version = struct.unpack_from("<HHI", header, 16)
    require(kind in (2, 3) and machine == 62 and version == 1,
            "expected native AMD64 executable ELF")
    require(struct.unpack_from("<H", header, 52)[0] == 64, "invalid ELF header length")
    require(os.access(path, os.X_OK), "ELF is not executable")
    return {"sha256": expected_hash, "bytes": size, "machine": "AMD64"}


def verify_bundle(bundle, report_hash):
    bundle = Path(bundle)
    report = read_json(bundle / "build-report.json", report_hash)
    require(isinstance(report, dict), "build report must be an object")
    require(not bundle.is_symlink() and not (bundle / "run").is_symlink(), "bundle directories must not be symlinks")
    require(report.get("format") == "bml-native-head-fragments-v1", "unsupported bundle format")
    require(report.get("target") == "x86_64-unknown-linux-gnu" and report.get("cpu") == "x86-64",
            "unsupported target or CPU profile")
    source = digest(report.get("source_sha256"))
    require(report.get("normalized_source_sha256") == source, "normalized source identity mismatch")
    rows = integer(report.get("head_rows"), 1, 32768, "head rows")
    cols = integer(report.get("head_cols"), 1, 4096, "head columns")
    count = integer(report.get("workers"), 1, 8, "worker count")
    fragments = report.get("fragments")
    require(isinstance(fragments, list) and len(fragments) == count, "worker count mismatch")
    coordinator = report.get("coordinator", {})
    require(isinstance(coordinator, dict), "invalid coordinator record")
    require(coordinator.get("file_name") == "coordinator", "invalid coordinator filename")
    artifacts = {"coordinator": elf_identity(bundle / "run/coordinator", coordinator.get("sha256"),
                                            coordinator.get("bytes"))}
    next_row = 0
    tensor_name = None
    for index, fragment in enumerate(fragments):
        require(isinstance(fragment, dict), "invalid fragment record")
        name = fragment.get("file_name")
        require(name == f"fragment_{index}", "invalid fragment filename or order")
        d = fragment.get("descriptor", {})
        require(isinstance(d, dict), "invalid fragment descriptor")
        require(type(d.get("version")) is int and d.get("version") == 1 and d.get("source_sha256") == source,
                "fragment version or model identity mismatch")
        require(integer(d.get("total_rows"), 1, 32768, "fragment total rows") == rows
                and integer(d.get("cols"), 1, 4096, "fragment columns") == cols,
                "fragment dimensions mismatch")
        start = integer(d.get("row_start"), 0, rows - 1, "row start")
        end = integer(d.get("row_end"), 1, rows, "row end")
        require(start == next_row and end > start, "fragment gap, overlap or empty range")
        name_in_model = d.get("tensor_name")
        require(name_in_model in ("output.weight", "token_embd.weight"), "unsupported head tensor")
        tensor_name = name_in_model if tensor_name is None else tensor_name
        require(name_in_model == tensor_name, "mixed head tensors")
        weight_bytes = (end - start) * cols * 4
        require(weight_bytes <= 64 * 1024 * 1024 and (end - start) * cols <= 32 * 1024 * 1024,
                "fragment weight/work budget exceeded")
        require(fragment.get("weight_bytes") == weight_bytes, "fragment weight byte mismatch")
        digest(d.get("weights_sha256"))
        artifacts[name] = elf_identity(bundle / "run" / name, fragment.get("sha256"), fragment.get("bytes"))
        next_row = end
    require(next_row == rows, "fragment coverage incomplete")
    return report, artifacts


def cases_from(value):
    require(isinstance(value, list) and 2 <= len(value) <= 8, "need two to eight cases")
    names = set()
    for case in value:
        require(isinstance(case, dict) and set(case) <= {"name", "prompt", "tokens", "predict", "context"},
                "invalid case fields")
        name = case.get("name")
        require(isinstance(name, str) and NAME.fullmatch(name) and name not in names, "invalid or duplicate case name")
        names.add(name)
        require(("prompt" in case) != ("tokens" in case), "case needs prompt or tokens")
        if "prompt" in case:
            require(isinstance(case["prompt"], str) and 0 < len(case["prompt"].encode()) <= 8192,
                    "invalid prompt byte length")
        else:
            require(isinstance(case["tokens"], list) and 1 <= len(case["tokens"]) <= 256, "invalid input tokens")
            for token in case["tokens"]:
                integer(token, 0, 2**32 - 1, "token ID")
        integer(case.get("predict"), 1, 64, "prediction count")
        integer(case.get("context"), 1, 256, "context")
    return value


def case_args(case):
    prefix = ["--prompt", case["prompt"]] if "prompt" in case else ["--tokens", ",".join(map(str, case["tokens"]))]
    return prefix + ["--predict", str(case["predict"]), "--context", str(case["context"]), "--logits"]


def invoke(command, cwd, timeout):
    """Limit elapsed time/output, killing only the new owned process group on timeout."""
    require(1 <= timeout <= 120, "timeout must be 1..120 seconds")
    def limits():
        resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_JSON_BYTES, MAX_JSON_BYTES))
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(list(map(str, command)), cwd=cwd, stdin=subprocess.DEVNULL,
                                   stdout=stdout, stderr=stderr, start_new_session=True,
                                   preexec_fn=limits)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise RuntimeError("generated-artifact process exceeded timeout") from None
        stdout.seek(0, 2)
        stderr.seek(0, 2)
        require(stdout.tell() <= MAX_JSON_BYTES and stderr.tell() <= MAX_JSON_BYTES, "subprocess output exceeds budget")
        stdout.seek(0)
        stderr.seek(0)
        return process.returncode, stdout.read(), stderr.read()


def success_json(result):
    code, stdout, stderr = result
    require(code == 0, f"artifact failed ({code}): {stderr[:2000].decode(errors='replace')}")
    require(not stderr, "successful artifact wrote unexpected stderr")
    return json.loads(stdout, object_pairs_hook=pairs_unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"nonfinite JSON: {value}")))


def float_bits(value):
    require(type(value) in (int, float) and math.isfinite(value), "invalid logit")
    try:
        return struct.unpack("<I", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError("logit not representable as F32") from error


def checked_run(run, case, rows):
    require(isinstance(run, dict), "missing canonical run")
    prompt = run.get("prompt_tokens")
    tokens = run.get("tokens")
    require(isinstance(prompt, list) and prompt and isinstance(tokens, list) and tokens,
            "empty prompt or generated tokens")
    for token in prompt + tokens:
        integer(token, 0, rows - 1, "output token ID")
    require(len(tokens) <= case["predict"] and len(prompt) + len(tokens) <= case["context"],
            "token/context budget mismatch")
    if len(tokens) < case["predict"]:
        require(tokens[-1] == run.get("eos_id"), "early generation without EOS")
    require(run.get("context") == case["context"], "reported context mismatch")
    if "tokens" in case:
        require(prompt == case["tokens"], "input token IDs changed")
    logits = run.get("logits")
    require(isinstance(logits, list) and len(logits) == len(tokens) + 1, "missing canonical logits")
    require(all(isinstance(row, list) and len(row) == rows for row in logits), "logit width mismatch")
    return [[float_bits(x) for x in row] for row in logits]


def require_bits_equal(expected, actual, label):
    if expected != actual:
        for position, (left, right) in enumerate(zip(expected, actual)):
            for column, (a, b) in enumerate(zip(left, right)):
                if a != b:
                    raise ValueError(f"{label}: first divergence position={position} logit={column} "
                                     f"expected_bits={a:#010x} actual_bits={b:#010x}")
        raise ValueError(f"{label}: vector shape differs")


def compare_outputs(reference, mono, fragmented, case, rows):
    ref_bits = checked_run(reference, case, rows)
    for label, value in [("monolithic", mono), ("fragments", fragmented)]:
        require(value.get("mode") == label, "coordinator mode mismatch")
        run = value.get("run")
        bits = checked_run(run, case, rows)
        require_bits_equal(ref_bits, bits, f"{label}: canonical logits")
        for key in ["prompt_tokens", "tokens", "eos_id", "bytes", "text", "context"]:
            require(run.get(key) == reference.get(key), f"{label}: canonical {key} differs")
        forwarded = value.get("forwarded_token_ids")
        require(forwarded == run["prompt_tokens"] + run["tokens"], "missing or reordered forward tokens")
        all_bits = value.get("all_logits_bits")
        require(isinstance(all_bits, list) and len(all_bits) == len(forwarded), "missing prefill/decode positions")
        for row in all_bits:
            require(isinstance(row, list) and len(row) == rows, "all-logits width mismatch")
            for bit in row:
                integer(bit, 0, 2**32 - 1, "logit bits")
                require(math.isfinite(struct.unpack("<f", struct.pack("<I", bit))[0]), "nonfinite logit bits")
        require_bits_equal(bits, all_bits[len(run["prompt_tokens"]) - 1:], "bit log and canonical outputs")
    require_bits_equal(mono["all_logits_bits"], fragmented["all_logits_bits"], "full prefill/decode bit parity")
    return {"name": case["name"], "prompt_tokens": reference["prompt_tokens"],
            "generated_tokens": reference["tokens"], "forward_positions": len(mono["all_logits_bits"]),
            "logits_compared": len(mono["all_logits_bits"]) * rows,
            "max_absolute_error": 0.0, "bit_identical": True, "first_divergence": None}


def save_json(path, value):
    path = Path(path)
    with path.open("x") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def prepare_reference(args):
    model = args.model.resolve(strict=True)
    require(sha256(model) == digest(args.model_sha256), "GGUF identity mismatch")
    binary = args.bml_model.resolve(strict=True)
    elf_identity(binary, args.bml_model_sha256)
    cases = cases_from(read_json(args.cases))
    results = []
    for case in cases:
        value = success_json(invoke([binary, "run", "--model", model, *case_args(case)], model.parent, args.timeout))
        results.append({"request": case, "result": value})
    result = {"format_version": 1, "source_sha256": args.model_sha256,
              "bml_model_sha256": args.bml_model_sha256, "cases": results}
    save_json(args.output, result)
    return {"reference_sha256": sha256(args.output), "cases": len(results)}


def failure_checks(stage, case, timeout, output):
    checks = []
    for name, extra, marker in [("unknown-option", ["--unknown-option"], "unknown"),
                                ("context-limit", ["--tokens", "1", "--predict", "1", "--context", "257"], "context")]:
        command = [stage / "coordinator", "--mode", "fragments", *extra]
        code, stdout, stderr = invoke(command, stage, timeout)
        require(code != 0 and not stdout and marker in stderr.decode(errors="replace").lower(),
                f"{name} did not fail with its diagnostic")
        checks.append({"name": name, "command": list(map(str, command)), "exit_code": code,
                       "stdout_bytes": len(stdout), "stderr": stderr[:2000].decode(errors="replace")})
    for name in ["missing-worker", "worker-hash"]:
        copy = output / name
        shutil.copytree(stage, copy)
        worker = copy / "fragment_0"
        if name == "missing-worker":
            worker.unlink()
        else:
            with worker.open("ab") as handle:
                handle.write(b"invalid-trailing-byte")
        command = [copy / "coordinator", "--mode", "fragments", *case_args(case)]
        code, stdout, stderr = invoke(command, copy, timeout)
        message = stderr.decode(errors="replace").lower()
        require(code != 0 and not stdout and "fragment" in message,
                f"{name} did not reject the changed worker")
        if name == "worker-hash":
            require("hash" in message or "identity" in message or "sha-256" in message,
                    "changed worker lacked identity diagnostic")
        checks.append({"name": name, "command": list(map(str, command)), "exit_code": code,
                       "stdout_bytes": len(stdout), "stderr": stderr[:2000].decode(errors="replace")})
    return checks


def validate(args):
    args.bundle = args.bundle.resolve(strict=True)
    args.reference_json = args.reference_json.resolve(strict=True)
    args.output = args.output.resolve()
    report, identities = verify_bundle(args.bundle, args.build_report_sha256)
    references = read_json(args.reference_json, args.reference_sha256)
    require(isinstance(references, dict), "reference must be an object")
    require(references.get("format_version") == 1 and references.get("source_sha256") == report["source_sha256"],
            "reference model or version mismatch")
    entries = references.get("cases")
    require(isinstance(entries, list) and all(isinstance(x, dict) for x in entries), "missing reference cases")
    cases_from([x.get("request") for x in entries])
    args.output.mkdir(parents=True, exist_ok=False)
    stage = args.output / "run"
    stage.mkdir()
    for name in identities:
        shutil.copy2(args.bundle / "run" / name, stage / name)
        elf_identity(stage / name, identities[name]["sha256"], identities[name]["bytes"])
    summary = {"status": "failed", "scope": "local-single-host-native-head-fragments",
               "source_sha256": report["source_sha256"], "build_report_sha256": args.build_report_sha256,
               "reference_sha256": args.reference_sha256, "artifacts": identities,
               "runtime_model_argument": False, "filesystem_isolation_verified": False,
               "distributed_evidence": False, "symbolic_bmlx_evidence": False,
               "coordinator_retains_full_model_including_head": True, "cases": [], "failure_checks": [],
               "commands": []}
    try:
        for entry in entries:
            case = entry["request"]
            outputs = {}
            for mode in ["monolithic", "fragments"]:
                command = [stage / "coordinator", "--mode", mode, *case_args(case)]
                result = invoke(command, stage, args.timeout)
                summary["commands"].append({"case": case["name"], "mode": mode,
                                            "argv": list(map(str, command)), "exit_code": result[0]})
                value = success_json(result)
                require(isinstance(value, dict) and value.get("source_sha256") == report["source_sha256"],
                        "coordinator source identity mismatch")
                save_json(args.output / f"{case['name']}-{mode}.json", value)
                outputs[mode] = value
            summary["cases"].append(compare_outputs(entry["result"], outputs["monolithic"], outputs["fragments"], case, report["head_rows"]))
        summary["failure_checks"] = failure_checks(stage, entries[0]["request"], args.timeout, args.output)
        summary["status"] = "passed"
    except (OSError, ValueError, RuntimeError) as error:
        summary["error"] = str(error)
    save_json(args.output / "report.json", summary)
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    reference = sub.add_parser("prepare-reference")
    for flag in ["--bml-model", "--model", "--cases", "--output"]:
        reference.add_argument(flag, type=Path, required=True)
    for flag in ["--bml-model-sha256", "--model-sha256"]:
        reference.add_argument(flag, required=True)
    reference.add_argument("--timeout", type=int, default=60)
    check = sub.add_parser("validate")
    for flag in ["--bundle", "--reference-json", "--output"]:
        check.add_argument(flag, type=Path, required=True)
    for flag in ["--build-report-sha256", "--reference-sha256"]:
        check.add_argument(flag, required=True)
    check.add_argument("--trusted-generated-artifacts", action="store_true", required=True)
    check.add_argument("--timeout", type=int, default=60)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result = prepare_reference(args) if args.command == "prepare-reference" else validate(args)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0 if result.get("status", "passed") == "passed" else 2
    except (OSError, ValueError, RuntimeError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
