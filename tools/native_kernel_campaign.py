#!/usr/bin/env python3
"""Measure structural-code kernels on a native Linux host, never an ARM emulator."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = {"aarch64": "Neon", "x86_64": "Avx2"}
WORKLOADS = {"equal-long", "late-difference", "short-mixed"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_host(expected_arch):
    require(platform.system() == "Linux", "native campaign requires Linux")
    require(platform.machine() == expected_arch, "native architecture mismatch")
    runner_arch = os.environ.get("RUNNER_ARCH")
    if runner_arch:
        require(runner_arch == {"aarch64": "ARM64", "x86_64": "X64"}[expected_arch],
                "GitHub runner architecture mismatch")
    require(not os.environ.get("CARGO_BUILD_TARGET"), "cross compilation is not native evidence")
    require(not any(k.startswith("CARGO_TARGET_") and k.endswith("_RUNNER") for k in os.environ),
            "Cargo runner overrides are not native evidence")


def validate_benchmark(data, backend, iterations):
    require(data["iterations"] == iterations and data["pairs"] == 512,
            "unexpected comparison workload size")
    require(data["repetitions"] == 7, "seven measured repetitions required")
    require(data[{"Neon": "neon_available", "Avx2": "avx2_available"}[backend]] is True,
            "expected backend unavailable")
    workloads = data["workloads"]
    require(len(workloads) == 3 and {w["name"] for w in workloads} == WORKLOADS,
            "missing or duplicate workloads")
    for workload in workloads:
        rows = workload["backends"]
        require(len(rows) == 2 and {r["backend"] for r in rows} == {"Scalar", backend},
                "missing or duplicate backend measurements")
        for row in rows:
            require(len(row["samples"]) == 7, "missing measured samples")
            for sample in row["samples"]:
                for field in ["ns", "allocation_calls", "requested_bytes", "peak_additional_live_bytes"]:
                    value = sample[field]
                    require(type(value) is int and value >= (1 if field == "ns" else 0),
                            f"invalid measurement: {field}")


def capture(args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-arch", choices=BACKENDS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    validate_host(args.expected_arch)
    require(1 <= args.iterations <= 10000, "iteration budget outside benchmark range")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    revision = capture(["git", "rev-parse", "HEAD"])
    dirty = bool(capture(["git", "status", "--porcelain"]))
    if os.environ.get("GITHUB_ACTIONS") == "true":
        require(revision == os.environ["GITHUB_SHA"] and not dirty,
                "campaign must execute the exact clean workflow revision")
    rustc = capture(["rustc", "-Vv"])
    channel_match = re.search(r'^\s*channel\s*=\s*"([^"]+)"',
                              (ROOT / "rust-toolchain.toml").read_text(), re.MULTILINE)
    require(channel_match is not None, "missing pinned Rust channel")
    channel = channel_match.group(1)
    require(rustc.startswith(f"rustc {channel} "), "pinned Rust toolchain required")
    sources = ["Cargo.toml", "rust-toolchain.toml", "Cargo.lock", "crates/domain/Cargo.toml",
               "crates/kernels/Cargo.toml", "crates/kernels/tests/code_comparison.rs",
               "crates/kernels/tests/tensor_simd.rs", "crates/kernels/examples/code_compare_bench.rs",
               "tools/native_kernel_campaign.py", ".github/workflows/native-kernels.yml"]
    sources += sorted(str(p.relative_to(ROOT)) for crate in ["domain", "kernels"]
                      for p in (ROOT / "crates" / crate / "src").rglob("*.rs"))
    source_hashes = {s: sha256(ROOT / s) for s in sources}
    with (output / "tests.log").open("w") as log:
        subprocess.run(["cargo", "test", "-p", "bml-kernels", "--all-features", "--locked",
                        "--", "--nocapture"], cwd=ROOT, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    with (output / "build.log").open("w") as log:
        subprocess.run(["cargo", "build", "-p", "bml-kernels", "--example", "code_compare_bench",
                        "--release", "--locked"], cwd=ROOT, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    target = Path(json.loads(capture(["cargo", "metadata", "--no-deps", "--format-version=1",
                                    "--locked"]))["target_directory"])
    binary = target / "release/examples/code_compare_bench"
    header = binary.read_bytes()[:20]
    require(header[:5] == b"\x7fELF\x02" and header[5] == 1, "expected little-endian ELF64")
    machine = struct.unpack_from("<H", header, 18)[0]
    require(machine == {"aarch64": 183, "x86_64": 62}[args.expected_arch],
            "benchmark binary architecture mismatch")
    allowed_cpus = sorted(os.sched_getaffinity(0))
    cpu = allowed_cpus[0]
    with (output / "benchmark.json").open("w") as report, (output / "benchmark.stderr").open("w") as err:
        subprocess.run(["taskset", "--cpu-list", str(cpu), str(binary), str(args.iterations)],
                       cwd=ROOT, stdout=report, stderr=err, check=True)
    data = json.loads((output / "benchmark.json").read_text())
    validate_benchmark(data, BACKENDS[args.expected_arch], args.iterations)
    shutil.copyfile(binary, output / "benchmark-executable")
    require(all(sha256(ROOT / s) == digest for s, digest in source_hashes.items()),
            "source changed during campaign")
    manifest = {
        "scope": "Native structural-code comparison experiment; no BML numerical or inference throughput claim.",
        "execution": "native ELF executed directly with taskset; no Cargo runner override",
        "measured_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "revision": revision, "worktree_dirty": dirty, "architecture": platform.machine(),
        "expected_backend": BACKENDS[args.expected_arch], "rustc": rustc,
        "runner_arch": os.environ.get("RUNNER_ARCH"), "runner_os": os.environ.get("RUNNER_OS"),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "cpu": json.loads(capture(["lscpu", "--json"])),
        "uname": capture(["uname", "-a"]), "allowed_cpus": allowed_cpus, "benchmark_cpu": cpu,
        "source_sha256": source_hashes,
        "source_unchanged_at_completion": True,
        "files_sha256": {p.name: sha256(p) for p in output.iterdir() if p.is_file()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"native {args.expected_arch}: validated {BACKENDS[args.expected_arch]} and Scalar measurements")


if __name__ == "__main__":
    main()
