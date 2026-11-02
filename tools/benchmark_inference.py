#!/usr/bin/env python3
"""Measure matching native BML and llama.cpp runs without downloading models."""

import argparse
import json
import math
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import time

import validate_inference as validation


PROMPTS = ("Once upon a time", "Hello")
PREDICT = 16
CONTEXT = 128
WARMUPS = 1
REPETITIONS = 3
ROOT = Path(__file__).resolve().parents[1]


def positive_number(value, name):
    """Reject booleans, non-finite metrics and nonpositive elapsed durations."""
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def output_contract(value, engine, predict=PREDICT, context=CONTEXT):
    """Validate enough output to prevent accepting unrelated or incomplete work."""
    if not isinstance(value, dict):
        raise ValueError(f"{engine} output must be a JSON object")
    prompt = validation.ids(value.get("prompt_tokens"), "prompt_tokens")
    tokens = validation.ids(value.get("tokens"), "tokens")
    eos = value.get("eos_id")
    if type(eos) is not int or eos < 0:
        raise ValueError("eos_id must be a nonnegative integer")
    if not prompt or len(prompt) + predict > context:
        raise ValueError("prompt is empty or exceeds the context budget")
    if not tokens or len(tokens) > predict:
        raise ValueError("generated token count is invalid")
    if len(tokens) != predict and tokens[-1] != eos:
        raise ValueError("generation stopped before the requested count without EOS")
    if eos in tokens[:-1]:
        raise ValueError("generation continued after EOS")
    if not isinstance(value.get("text"), str):
        raise ValueError("text must be a string")
    data = value.get("bytes")
    if not isinstance(data, list) or any(type(byte) is not int or not 0 <= byte <= 255 for byte in data):
        raise ValueError("bytes must contain only byte-valued integers")
    if value.get("logits") not in (None, []):
        raise ValueError("benchmark output unexpectedly contains diagnostic logits")
    timing = value.get("timing") if engine == "reference" else value
    if not isinstance(timing, dict):
        raise ValueError(f"{engine} timing is missing")
    times = {name: positive_number(timing.get(name), name) for name in ("prefill_ms", "decode_ms")}
    if engine == "reference":
        reference = value.get("reference")
        if not isinstance(reference, dict) or type(reference.get("threads")) is not int or reference["threads"] != 1:
            raise ValueError("reference must declare exactly one CPU thread")
        for name, expected in (("kv_dtype", "f32"), ("flash_attention", False), ("weight_repacking", False)):
            if type(reference.get(name)) is not type(expected) or reference[name] != expected:
                raise ValueError(f"reference has an unexpected {name}")
    return times


def compare_outputs(reference, artifact, predict=PREDICT, context=CONTEXT):
    """Require identical prompt, generation and output before accepting timings."""
    timings = {
        "reference": output_contract(reference, "reference", predict, context),
        "artifact": output_contract(artifact, "artifact", predict, context),
    }
    for field in ("prompt_tokens", "tokens", "eos_id", "text", "bytes"):
        if reference[field] != artifact[field]:
            raise ValueError(f"{field} differ between artifact and reference")
    return timings


def summarize(values):
    """Describe the measured population; warmups never enter this function."""
    if not values or any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
        raise ValueError("statistics require nonempty finite numeric samples")
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
        "stddev_population": statistics.pstdev(values),
    }


def system_description():
    cpu = None
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "cpu_model": cpu,
        "logical_cpus": os.cpu_count(),
        "allowed_cpus": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "python": platform.python_version(),
    }


def wait_for_named_child(process, timeout):
    """Collect Linux peak RSS for this PID, never cumulative RUSAGE_CHILDREN."""
    deadline = time.perf_counter() + timeout
    while True:
        pid, status, usage = os.wait4(process.pid, os.WNOHANG)
        if pid:
            process.returncode = os.waitstatus_to_exitcode(status)
            return process.returncode, usage.ru_maxrss
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            os.killpg(process.pid, signal.SIGKILL)
            _, status, _ = os.wait4(process.pid, 0)
            process.returncode = os.waitstatus_to_exitcode(status)
            raise subprocess.TimeoutExpired(process.args, timeout)
        time.sleep(min(0.001, remaining))


def measure(command, prefix, timeout, time_program):
    """Launch one process, using its own GNU time record for peak RSS."""
    prefix = Path(prefix)
    measured_command = list(command)
    time_file = prefix.with_suffix(".time.json")
    if time_program:
        measured_command = [
            str(time_program), "-f", '{"wall_seconds":%e,"max_rss_kib":%M}',
            "-o", str(time_file), "--", *command,
        ]
    started = time.perf_counter()
    rss = None
    with prefix.with_suffix(".json").open("w") as output, prefix.with_suffix(".stderr.log").open("w") as errors:
        process = subprocess.Popen(measured_command, stdout=output, stderr=errors, start_new_session=True)
        if not time_program and hasattr(os, "wait4"):
            returncode, rss = wait_for_named_child(process, timeout)
        else:
            try:
                returncode = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise
    monotonic_wall_ms = (time.perf_counter() - started) * 1000.0
    if returncode:
        raise RuntimeError(f"exit status {returncode}; see {prefix.with_suffix('.stderr.log')}")
    value = json.loads(prefix.with_suffix(".json").read_text(), parse_constant=validation.reject_constant)
    wall_ms = monotonic_wall_ms
    if time_program:
        measured = json.loads(time_file.read_text(), parse_constant=validation.reject_constant)
        seconds, rss = measured.get("wall_seconds"), measured.get("max_rss_kib")
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("GNU time wall duration is invalid")
        if type(rss) is not int or rss <= 0:
            raise ValueError("GNU time maximum RSS is invalid")
        wall_ms = seconds * 1000.0
    return value, {
        "command": command,
        "measurement_command": measured_command,
        "wall_ms": wall_ms,
        "monotonic_wall_ms": monotonic_wall_ms,
        "max_rss_kib": rss,
    }


def artifact_build_report(path, artifact_hash, model_hash):
    """Bind compiler provenance and build duration to the supplied artifact."""
    report = json.loads(path.read_text(), parse_constant=validation.reject_constant)
    if not isinstance(report, dict) or report.get("sha256") != artifact_hash:
        raise ValueError("artifact build report does not identify this executable")
    if report.get("source_sha256") != model_hash:
        raise ValueError("artifact build report does not identify this GGUF model")
    if report.get("target") != "x86_64-unknown-linux-gnu":
        raise ValueError("artifact build target is not Linux x86_64 GNU")
    if not isinstance(report.get("rustc_version"), str) or not report["rustc_version"].strip():
        raise ValueError("artifact build report must record rustc_version")
    positive_number(report.get("elapsed_ms"), "artifact build elapsed_ms")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=os.environ.get("BML_TEST_GGUF"))
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--artifact-build-report", type=Path)
    parser.add_argument("--reference", type=Path, default=ROOT / "target/bml-1.0/llama-reference")
    parser.add_argument("--llama-source", type=Path, default=ROOT / "target/bml-1.0/llama.cpp")
    parser.add_argument("--expected-model-sha256")
    parser.add_argument("--output", type=Path, default=ROOT / "target/bml-1.0/benchmark")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args(argv)
    if args.model is None:
        print(json.dumps({"status": "not_run", "reason": "provide --model or BML_TEST_GGUF; no model is downloaded"}))
        return 77
    report = {"schema_version": 1, "status": "running", "samples": []}
    output_created = False
    try:
        positive_number(args.timeout, "timeout")
        if args.artifact is None or args.artifact_build_report is None:
            raise ValueError("--artifact and --artifact-build-report are required")
        args.output.mkdir(parents=True, exist_ok=False)
        output_created = True
        model, artifact, reference = args.model.resolve(), args.artifact.resolve(), args.reference.resolve()
        build_path, source = args.artifact_build_report.resolve(), args.llama_source.resolve()
        if validation.revision(source) != validation.REFERENCE_COMMIT or validation.working_tree_dirty(source):
            raise ValueError("reference checkout must be clean and pinned to the expected commit")
        reference_build = validation.reference_manifest(source, reference)
        snapshot = {
            path: validation.sha256(path)
            for path in (
                model, artifact, reference, build_path, reference.with_suffix(".build.json"),
                Path(__file__).resolve(), Path(validation.__file__).resolve(),
                Path(__file__).with_name("llama_reference.cpp").resolve(),
            )
        }
        for name in reference_build["libraries_sha256"]:
            path = source / name
            snapshot[path] = validation.sha256(path)
        if args.expected_model_sha256 and snapshot[model] != args.expected_model_sha256:
            raise ValueError("model SHA-256 differs from the expected value")
        build = artifact_build_report(build_path, snapshot[artifact], snapshot[model])
        time_program = Path("/usr/bin/time") if Path("/usr/bin/time").is_file() else None
        report.update({
            "created_unix": time.time(),
            "protocol": {
                "prompts": list(PROMPTS), "predict": PREDICT, "context": CONTEXT,
                "warmups_per_prompt_per_engine": WARMUPS,
                "measured_repetitions_per_prompt_per_engine": REPETITIONS,
                "cpu_threads_per_engine": 1, "sampling": "greedy", "logits": False,
                "new_process_per_sample": True,
                "cache_policy": "OS page cache is not evicted or controlled; provenance hashing and warmups precede measurements. These are not cold-cache runs.",
                "engine_order": "Alternates each paired round; exact order is recorded per sample.",
                "wall_source": "/usr/bin/time %e (10 ms resolution)" if time_program else "Python perf_counter around the child process; wait4 exit polling interval at most 1 ms plus scheduling delay",
                "rss_source": "/usr/bin/time %M for each individual process" if time_program else "os.wait4(child_pid).ru_maxrss (Linux KiB)" if hasattr(os, "wait4") else None,
                "statistics": "mean/min/max/population standard deviation; warmups excluded",
            },
            "system": system_description(),
            "model": {"path": str(model), "sha256": snapshot[model], "bytes": model.stat().st_size},
            "artifact": {"path": str(artifact), "sha256": snapshot[artifact], "bytes": artifact.stat().st_size, "build_report": build},
            "reference": {"path": str(reference), "sha256": snapshot[reference], "bytes": reference.stat().st_size, "build_manifest": reference_build},
            "input_sha256": {str(path): digest for path, digest in snapshot.items()},
            "limitations": [
                "Only the supplied model, two fixed prompts and this host are measured.",
                "This benchmark checks identical generated outputs; the separate parity gate checks full logits.",
                "Internal prefill/decode timings exclude startup, model loading and resource integrity checks; process wall time includes them.",
                "Maximum RSS is a per-process high-water mark, not allocated-byte accounting or a sum across children.",
                "No general performance superiority is inferred from these measurements.",
            ],
        })
        if not time_program:
            report["limitations"].append("/usr/bin/time is unavailable; wall timing uses perf_counter and includes subprocess launch and exit observation overhead.")
            if not hasattr(os, "wait4"):
                report["limitations"].append("os.wait4 is also unavailable, so maximum RSS is not measured.")
        validation.write_json(args.output / "report.json", report)
        for prompt_index, prompt in enumerate(PROMPTS):
            for round_index in range(WARMUPS + REPETITIONS):
                order = ["artifact", "reference"] if round_index % 2 == 0 else ["reference", "artifact"]
                outputs, runs = {}, {}
                for engine in order:
                    command = [str(artifact)] if engine == "artifact" else [str(reference), "--model", str(model)]
                    command += ["--prompt", prompt, "--predict", str(PREDICT), "--context", str(CONTEXT)]
                    prefix = args.output / f"prompt-{prompt_index}-round-{round_index}-{engine}"
                    outputs[engine], runs[engine] = measure(command, prefix, args.timeout, time_program)
                timing = compare_outputs(outputs["reference"], outputs["artifact"])
                for engine in order:
                    run = runs[engine]
                    run.update(timing[engine])
                    run["prefill_tokens_per_second"] = len(outputs[engine]["prompt_tokens"]) * 1000.0 / run["prefill_ms"]
                    run["decode_tokens_per_second"] = len(outputs[engine]["tokens"]) * 1000.0 / run["decode_ms"]
                report["samples"].append({
                    "prompt": prompt, "prompt_index": prompt_index, "round": round_index,
                    "warmup": round_index < WARMUPS, "engine_order": order,
                    "outputs_equal": True,
                    "output": {field: outputs["reference"][field] for field in ("prompt_tokens", "tokens", "eos_id", "text", "bytes")},
                    "runs": runs,
                })
                validation.write_json(args.output / "report.json", report)
        summary = []
        for prompt in PROMPTS:
            measured = [sample for sample in report["samples"] if sample["prompt"] == prompt and not sample["warmup"]]
            for engine in ("artifact", "reference"):
                metrics = {}
                for metric in ("prefill_ms", "decode_ms", "wall_ms", "monotonic_wall_ms", "max_rss_kib", "prefill_tokens_per_second", "decode_tokens_per_second"):
                    values = [sample["runs"][engine][metric] for sample in measured]
                    metrics[metric] = None if any(value is None for value in values) else summarize(values)
                summary.append({"prompt": prompt, "engine": engine, "metrics": metrics})
        report["summary"] = summary
        report["provenance_errors"] = validation.changed_files(snapshot)
        if validation.revision(source) != validation.REFERENCE_COMMIT or validation.working_tree_dirty(source):
            report["provenance_errors"].append("reference checkout changed during measurement")
        if report["provenance_errors"]:
            raise ValueError("benchmark inputs changed during measurement")
        report["status"] = "passed"
        report["finished_unix"] = time.time()
        validation.write_json(args.output / "report.json", report)
        print(json.dumps({"status": "passed", "report": str(args.output / "report.json"), "measured_pairs": len(PROMPTS) * REPETITIONS}))
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        report["status"] = "failed"
        report["error"] = str(error)
        if output_created:
            validation.write_json(args.output / "report.json", report)
        print(json.dumps({"status": "failed", "error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
