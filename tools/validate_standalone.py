#!/usr/bin/env python3
"""Validate a trusted generated Linux ELF in a minimal, network-isolated root.

Only use BML-generated artifacts whose origin is already trusted. Dependency
inspection uses ldd, which is not a safe operation on arbitrary executable files.
The explicit --trusted-generated-artifact flag acknowledges that prerequisite.
No namespace, tracing or comparison failure is converted into a passing result.
"""

import argparse
import ast
import json
import os
from pathlib import Path, PurePosixPath
import re
import resource
import shutil
import signal
import subprocess
import tempfile
import time

from validate_inference import compare, reject_constant, sha256


NETWORK_CALLS = {
    "socket", "socketpair", "connect", "accept", "accept4", "bind", "listen",
    "sendto", "recvfrom", "sendmsg", "recvmsg", "recvmmsg", "sendmmsg",
    "shutdown", "getpeername", "getsockname", "setsockopt", "getsockopt",
}
PROCESS_CALLS = {"clone", "clone3", "fork", "vfork"}
PROCFS_FILES = {"/proc/self/exe", "/proc/self/maps"}
MUTATING_FILE_CALLS = {"mkdir", "mkdirat", "unlink", "unlinkat", "rename", "renameat",
                       "renameat2", "creat", "chmod", "fchmodat", "truncate", "link",
                       "linkat", "symlink", "symlinkat"}
FILE_CALLS = {
    "open", "openat", "openat2", "access", "faccessat", "faccessat2", "stat",
    "lstat", "statx", "newfstatat", "readlink", "readlinkat", "chdir", "execve",
    "execveat", "mkdir", "mkdirat", "unlink", "unlinkat", "rename", "renameat",
    "renameat2", "creat", "chmod", "fchmodat", "truncate", "link", "linkat",
    "symlink", "symlinkat",
}


def absolute_runtime_path(value):
    """Accept an absolute, non-escaping path to a runtime file."""
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts or value == "/":
        raise ValueError(f"invalid absolute runtime path: {value}")
    return str(path)


def dependencies(program_headers, dynamic_section, ldd_output):
    """Resolve only interpreter/NEEDED libraries declared by readelf and ldd."""
    interpreters = re.findall(r"Requesting program interpreter:\s*([^\]]+)\]", program_headers)
    if len(interpreters) != 1:
        raise ValueError("expected exactly one ELF dynamic interpreter")
    interpreter = absolute_runtime_path(interpreters[0])
    needed = re.findall(r"\(NEEDED\).*Shared library: \[([^\]]+)\]", dynamic_section)
    if not needed or len(needed) != len(set(needed)):
        raise ValueError("missing or duplicate ELF NEEDED declarations")
    libraries = {}
    for raw in ldd_output.splitlines():
        line = raw.strip()
        if not line:
            continue
        if re.fullmatch(r"linux-vdso\.so\.\d+ \(0x[0-9a-fA-F]+\)", line):
            continue
        mapped = re.fullmatch(r"([^\s]+) => (/[^\n]+?) \(0x[0-9a-fA-F]+\)", line)
        direct = re.fullmatch(r"(/[^\n]+?) \(0x[0-9a-fA-F]+\)", line)
        if mapped:
            name, location = mapped.groups()
        elif direct:
            location = direct[1]
            name = PurePosixPath(location).name
        else:
            raise ValueError(f"unresolved or unrecognized ldd dependency: {line}")
        location = absolute_runtime_path(location)
        if name in libraries and libraries[name] != location:
            raise ValueError(f"ambiguous library resolution: {name}")
        libraries[name] = location
    if len(libraries) > 64:
        raise ValueError("too many runtime library dependencies")
    missing = set(needed) - libraries.keys()
    if missing:
        raise ValueError(f"ELF dependencies were not resolved: {sorted(missing)}")
    locations = set(libraries.values())
    locations.add(interpreter)
    return {"interpreter": interpreter, "needed": needed, "paths": sorted(locations)}


def first_quoted_string(arguments):
    match = re.search(r'"(?:\\.|[^"\\])*"', arguments)
    if not match:
        return None
    try:
        value = ast.literal_eval(match[0])
    except (SyntaxError, ValueError) as error:
        raise ValueError("unreadable quoted path in syscall trace") from error
    return value if isinstance(value, str) else None


def analyze_trace(trace, runtime_files):
    """Require a traced /model exec and inspect its file/network/process calls."""
    processes = set()
    started = False
    accesses = []
    network = []
    spawned = []
    unknown = []
    mutations = []
    normal_exits = set()
    model_pid = None
    # The resource loader maps its own ELF. Rust's stack/runtime initialization
    # also inspects its own mappings; neither path exposes another process.
    allowed = {"/", "/proc", "/proc/self", *PROCFS_FILES}
    for name in runtime_files:
        path = PurePosixPath(name)
        allowed.add(str(path))
        allowed.update(str(parent) for parent in path.parents)
    for line in trace.splitlines():
        entry = re.match(r"^(\d+)\s+(\w+)\((.*)", line)
        if not entry:
            resumed = re.match(r"^(\d+)\s+<\.\.\.\s+(\w+)\s+resumed>", line)
            if resumed and int(resumed[1]) in processes:
                unknown.append(line)
            continue
        pid, call, arguments = int(entry[1]), entry[2], entry[3]
        path = first_quoted_string(arguments)
        success = re.search(r"\)\s+=\s+([0-9]+)(?:\s|$)", line)
        if call == "execve" and path == "/model" and success and not started:
            processes.add(pid)
            model_pid = pid
            started = True
            continue
        if pid not in processes:
            continue
        if "<unfinished ...>" in line:
            unknown.append(line)
        if call in PROCESS_CALLS and success:
            processes.add(int(success[1]))
            spawned.append(line)
        if call in NETWORK_CALLS:
            network.append(line)
        if call in MUTATING_FILE_CALLS or (call in {"open", "openat", "openat2"}
                                          and re.search(r"\bO_(WRONLY|RDWR|CREAT|TRUNC)\b", arguments)):
            mutations.append(line)
        if call in {"exit", "exit_group"} and re.match(r"0\)", arguments):
            normal_exits.add(pid)
        if call in {"execve", "execveat"}:
            spawned.append(line)
        if call in FILE_CALLS and path:
            normalized = str(PurePosixPath("/") / path)
            accesses.append({"syscall": call, "path": normalized,
                             "succeeded": bool(success), "allowed": normalized in allowed})
    if not started:
        raise ValueError("trace does not prove successful execution of /model")
    if model_pid not in normal_exits:
        raise ValueError("trace does not include the model's successful process exit")
    unexpected = [item for item in accesses if item["succeeded"] and not item["allowed"]]
    return {"model_pids": sorted(processes), "file_accesses": accesses,
            "network_calls": network, "unexpected_successful_file_accesses": unexpected,
            "subprocess_calls": spawned, "unresolved_trace_lines": unknown,
            "filesystem_mutations": mutations,
            "passed": not network and not unexpected and not spawned and not unknown and not mutations}


def tool(name):
    path = shutil.which(name)
    if path is None and name == "chroot" and os.access("/usr/sbin/chroot", os.X_OK):
        path = "/usr/sbin/chroot"
    if path is None:
        raise RuntimeError(f"required tool unavailable: {name}")
    return path


def runtime_environment():
    """Keep the session environment while disabling host library injection."""
    environment = os.environ.copy()
    for name in ("LD_PRELOAD", "LD_LIBRARY_PATH", "LD_AUDIT"):
        environment.pop(name, None)
    return environment


def metadata_command(command, output, stem, records):
    completed = subprocess.run(command, text=True, capture_output=True, timeout=10,
                               check=False, env=runtime_environment())
    (output / f"{stem}.stdout.log").write_text(completed.stdout)
    (output / f"{stem}.stderr.log").write_text(completed.stderr)
    records.append({"argv": command, "exit_code": completed.returncode,
                    "stdout": f"{stem}.stdout.log", "stderr": f"{stem}.stderr.log"})
    if completed.returncode:
        raise RuntimeError(f"{stem} failed with exit code {completed.returncode}")
    return completed.stdout


def child_limits(memory_mib, timeout):
    resource.setrlimit(resource.RLIMIT_AS, (memory_mib * 1024**2,) * 2)
    resource.setrlimit(resource.RLIMIT_CPU, (timeout,) * 2)
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024**2,) * 2)


def invoke_isolated(command, output, timeout, memory_mib):
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.monotonic()
    timed_out = False
    with (output / "model.stdout.json").open("wb") as stdout, (output / "model.stderr.log").open("wb") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True,
                                   preexec_fn=lambda: child_limits(memory_mib, timeout),
                                   env=runtime_environment())
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            code = process.wait()
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {"argv": command, "exit_code": code, "timed_out": timed_out,
            "wall_seconds": time.monotonic() - started,
            "user_cpu_seconds": after.ru_utime - before.ru_utime,
            "system_cpu_seconds": after.ru_stime - before.ru_stime,
            "children_peak_rss_kib": after.ru_maxrss,
            "rss_scope": "Linux high-water mark of waited child processes, including inspection/tracing tools",
            "per_process_address_space_limit_bytes": memory_mib * 1024**2,
            "timeout_seconds": timeout,
            "stdout": "model.stdout.json", "stderr": "model.stderr.log"}


def validate(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "status": "failed", "created_unix": time.time(),
              "commands": [], "errors": [], "trusted_generated_artifact": True,
              "isolation": {"user_namespace": True, "mount_namespace": True,
                            "network_namespace": True, "pid_namespace": True,
                            "procfs_required": True, "procfs_files": sorted(PROCFS_FILES),
                            "rootfs_contains_build_tools": False, "verified": False}}
    try:
        artifact = args.artifact.resolve(strict=True)
        reference_path = args.reference_json.resolve(strict=True)
        report["artifact"] = {"path": str(artifact), "sha256": sha256(artifact),
                              "bytes": artifact.stat().st_size}
        report["reference_json"] = {"path": str(reference_path), "sha256": sha256(reference_path)}
        report["validator_source_sha256"] = sha256(Path(__file__))
        readelf = tool("readelf")
        headers = metadata_command([readelf, "-lW", str(artifact)], output, "readelf-program", report["commands"])
        dynamic = metadata_command([readelf, "-dW", str(artifact)], output, "readelf-dynamic", report["commands"])
        linked = metadata_command([tool("ldd"), str(artifact)], output, "ldd", report["commands"])
        declared = dependencies(headers, dynamic, linked)
        report["dependencies"] = declared
        # Model artifacts can exceed a small /tmp tmpfs; keep this disposable
        # root beside its evidence on the workspace filesystem instead.
        with tempfile.TemporaryDirectory(prefix=".rootfs-", dir=output) as directory:
            root = Path(directory)
            installed = []
            for source, target in [(artifact, "/model"), *((Path(path), path) for path in declared["paths"])]:
                destination = root / target.lstrip("/")
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                installed.append({"path": target, "sha256": sha256(destination),
                                  "bytes": destination.stat().st_size})
            (root / "proc").mkdir()
            report["rootfs_files"] = installed
            trace_path = output / "syscalls.log"
            command = [tool("strace"), "--kill-on-exit", "-f", "-qq", "-s", "4096",
                       "-e", "trace=%file,%network,%process", "-o", str(trace_path),
                       tool("unshare"), "--user", "--map-root-user", "--mount", "--net",
                       "--pid", "--fork", f"--mount-proc={root / 'proc'}",
                       tool("chroot"), str(root), "/model", "--prompt", args.prompt,
                       "--predict", str(args.predict), "--context", str(args.context), "--logits"]
            execution = invoke_isolated(command, output, args.timeout, args.memory_mib)
            report["commands"].append(execution)
            if execution["exit_code"] != 0 or execution["timed_out"]:
                raise RuntimeError("isolated execution/tracing failed; inspect model.stderr.log")
            report["trace"] = analyze_trace(trace_path.read_text(), [item["path"] for item in installed])
            report["trace"]["sha256"] = sha256(trace_path)
            if not report["trace"]["passed"]:
                raise RuntimeError("syscall trace contains unexpected access or incomplete evidence")
            report["isolation"]["verified"] = True
            expected = json.loads(reference_path.read_text(), parse_constant=reject_constant)
            actual = json.loads((output / "model.stdout.json").read_text(), parse_constant=reject_constant)
            if not isinstance(expected, dict) or type(expected.get("eos_id")) is not int:
                raise ValueError("reference JSON must declare its integer eos_id")
            if not isinstance(actual, dict) or actual.get("eos_id") != expected["eos_id"]:
                raise ValueError("standalone EOS token ID differs from reference")
            comparison = compare(expected, actual, predict=args.predict,
                                 eos_id=expected["eos_id"], context=args.context)
            report["comparison"] = comparison
            if not comparison["passed"]:
                raise RuntimeError("standalone output differs from the reference")
            report["status"] = "passed"
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        report["errors"].append(str(error))
    finally:
        report["finished_unix"] = time.time()
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--artifact", type=Path, required=True)
    result.add_argument("--reference-json", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--trusted-generated-artifact", action="store_true", required=True)
    result.add_argument("--prompt", default="Hello")
    result.add_argument("--predict", type=int, default=4)
    result.add_argument("--context", type=int, default=128)
    result.add_argument("--timeout", type=int, default=60)
    result.add_argument("--memory-mib", type=int, default=1024)
    return result


def main(argv=None):
    arguments = parser()
    args = arguments.parse_args(argv)
    if not 1 <= args.timeout <= 60 or not 64 <= args.memory_mib <= 16384:
        arguments.error("timeout must be 1..60 seconds and memory-mib 64..16384")
    if args.predict < 0 or args.context < 1:
        arguments.error("predict must be nonnegative and context positive")
    report = validate(args)
    print(json.dumps({"status": report["status"], "report": str(args.output / "report.json"),
                      "errors": report["errors"]}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
