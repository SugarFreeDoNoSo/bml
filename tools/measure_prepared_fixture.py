#!/usr/bin/env python3
"""Measure independent fixture conversion/load/execution, excluding build time.

Linux wait4 ru_maxrss measures each complete child lifetime, including setup.
Internal Rust timers separate load/tokenization/execution. Never claim execution
RSS excludes loading. Outputs are retained in a newly-created campaign directory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def measure(command, directory, label):
    log = directory / (label + ".stdout")
    errors = directory / (label + ".stderr")
    started = time.monotonic_ns()
    with log.open("wb") as out, errors.open("wb") as err:
        child = subprocess.Popen(command, stdout=out, stderr=err)
        _, status, usage = os.wait4(child.pid, 0)
        child.returncode = os.waitstatus_to_exitcode(status)
    result = {
        "command": command, "exit_code": child.returncode,
        "wall_ns": time.monotonic_ns() - started,
        "user_seconds": usage.ru_utime, "system_seconds": usage.ru_stime,
        "peak_process_rss_kib": usage.ru_maxrss,
        "stdout": str(log), "stderr": str(errors),
        "stdout_sha256": sha256(log), "stderr_sha256": sha256(errors),
    }
    if label != "conversion" and child.returncode == 0:
        result["phase_timers"] = json.loads(log.read_text())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--cli", type=Path, default=Path("target/release/bml-cli"))
    parser.add_argument("--driver", type=Path, default=Path("target/release/examples/measure_prepared_fixture"))
    args = parser.parse_args()
    cli, driver = args.cli.resolve(), args.driver.resolve()
    # Validate binaries before creating the campaign directory.
    binaries = {str(path): sha256(path) for path in (cli, driver)}
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    source = directory / "fixture.gguf"
    output = directory / "fixture.bmlgguf"
    subprocess.run([str(driver), "create", str(source)], check=True, capture_output=True)
    report = {"schema": 1, "binary_sha256": binaries,
              "source_sha256": sha256(source), "source_bytes": source.stat().st_size,
              "rss_scope": "per-child full process lifetime, including load/setup; Linux KiB",
              "timing_scope": "wall includes process startup; Rust phase timers exclude startup",
              "samples_per_phase": 1, "runs": []}
    conversion = [str(cli), "convert-weights", "-m", str(source), "-o", str(output),
                  "--max-synthesis-steps-total", "1000000", "--max-block-terms", "1024",
                  "--max-output-bytes", "1073741824", "--max-resident-bytes", "268435456",
                  "--max-cache-bytes", "1048576"]
    report["runs"].append(measure(conversion, directory, "conversion"))
    report["source_unchanged"] = sha256(source) == report["source_sha256"]
    if report["runs"][-1]["exit_code"] == 0:
        report.update(output_bytes=output.stat().st_size, output_sha256=sha256(output))
        source.unlink()
        report["source_removed_before_loading"] = not source.exists()
        for phase in ("load", "execute"):
            report["runs"].append(measure([str(driver), phase, str(output)], directory, phase))
        report["output_unchanged"] = sha256(output) == report["output_sha256"]
    report["temporary_files_remaining"] = [p.name for p in directory.glob(".bmlgguf-*.tmp")]
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if any(run["exit_code"] != 0 for run in report["runs"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
