#!/usr/bin/env python3
"""Validate trusted native head-fragment ELF roles over local loopback TCP.

No build, download, Docker invocation or external host access occurs. The
transport is deliberately test-only: descriptors are checked, but there is no
TLS or cryptographic peer authentication. Separate directories are not a
filesystem sandbox and loopback processes are not evidence of distinct hosts.
"""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import resource
import selectors
import shutil
import socket
import subprocess
import tempfile
import threading
import time

import validate_fragment_elf as local


MAX_LISTENER_BYTES = 16 * 1024


class Deadline:
    """One shared elapsed-time budget, never renewed per case or process."""
    def __init__(self, seconds):
        local.require(type(seconds) is int and 1 <= seconds <= 120, "timeout must be 1..120 seconds")
        self.started = time.monotonic()
        self.cleanup_end = self.started + seconds
        self.end = self.cleanup_end - min(5, seconds / 2)

    def remaining(self, minimum=0):
        seconds = self.end - time.monotonic()
        if seconds <= minimum:
            raise RuntimeError("TCP validation exhausted its shared deadline")
        return seconds

    def cleanup_remaining(self):
        return max(0, self.cleanup_end - time.monotonic())


class WorkerFailure(ValueError):
    def __init__(self, message, record):
        super().__init__(message)
        self.record = record


def available_stdout(stream):
    """Drain only currently available bytes; inherited pipes must not block cleanup."""
    try:
        descriptor = stream.fileno()
    except (AttributeError, OSError):
        return stream.read(MAX_LISTENER_BYTES + 1)  # In-memory test stream.
    os.set_blocking(descriptor, False)
    try:
        return os.read(descriptor, MAX_LISTENER_BYTES + 1)
    except BlockingIOError as error:
        raise ValueError("worker stdout remained open after its exit") from error


def parse_listener(raw, expected_descriptor):
    local.require(isinstance(raw, bytes) and 1 <= len(raw) <= MAX_LISTENER_BYTES,
                  "listener announcement exceeds byte budget")
    local.require(raw.endswith(b"\n") and raw.count(b"\n") == 1,
                  "listener must announce exactly one JSON line")
    value = json.loads(raw, object_pairs_hook=local.pairs_unique)
    local.require(isinstance(value, dict) and set(value) == {"format", "address", "descriptor"},
                  "invalid listener announcement fields")
    local.require(value["format"] == "bml-native-fragment-tcp-listener-v1",
                  "unsupported listener announcement version")
    local.require(json.dumps(value["descriptor"], sort_keys=True, allow_nan=False)
                  == json.dumps(expected_descriptor, sort_keys=True, allow_nan=False),
                  "listener descriptor mismatch")
    address = value["address"]
    local.require(isinstance(address, str), "listener address must be text")
    host, separator, port = address.partition(":")
    local.require(separator == ":" and host == "127.0.0.1" and port.isascii()
                  and port.isdecimal() and 1 <= int(port) <= 65535 and port == str(int(port)),
                  "listener address is not an exact IPv4 loopback endpoint")
    return value


def read_listener_line(stream, deadline):
    """Read one bounded pipe line without a blocking readline or retry loop."""
    data = bytearray()
    with selectors.DefaultSelector() as selector:
        selector.register(stream, selectors.EVENT_READ)
        while b"\n" not in data:
            if not selector.select(deadline.remaining()):
                raise RuntimeError("listener announcement deadline expired")
            chunk = os.read(stream.fileno(), min(4096, MAX_LISTENER_BYTES + 1 - len(data)))
            if not chunk:
                raise ValueError("listener exited before a complete announcement")
            data.extend(chunk)
            local.require(len(data) <= MAX_LISTENER_BYTES, "listener announcement exceeds byte budget")
    # Any bytes following the first newline in this read violate the contract.
    local.require(data.count(b"\n") == 1 and data.endswith(b"\n"),
                  "listener emitted data after its announcement")
    return bytes(data)


class OwnedWorker:
    """A single verifier-owned process, cleaned up only through its handle."""
    def __init__(self, process, stderr, command):
        self.process = process
        self.stderr = stderr
        self.command = list(map(str, command))
        self.listener = None
        self.closed = False

    @classmethod
    def start(cls, executable, expected_identity, descriptor, deadline):
        local.elf_identity(executable, expected_identity["sha256"], expected_identity["bytes"])
        deadline.remaining()
        command = [str(executable), "--tcp-test-listen", "127.0.0.1:0"]
        stderr = tempfile.TemporaryFile()
        def limit_output():
            resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_LISTENER_BYTES, MAX_LISTENER_BYTES))
        try:
            process = subprocess.Popen(command, cwd=executable.parent, stdin=subprocess.DEVNULL,
                                       stdout=subprocess.PIPE, stderr=stderr, bufsize=0,
                                       preexec_fn=limit_output)
        except Exception:
            stderr.close()
            raise
        worker = cls(process, stderr, command)
        try:
            worker.listener = parse_listener(read_listener_line(process.stdout, deadline), descriptor)
        except Exception as error:
            record = worker.close(False, deadline)
            raise WorkerFailure(str(error), record) from error
        return worker

    def close(self, success, deadline):
        if self.closed:
            return None
        killed = False
        error = None
        result = {"pid": self.process.pid, "command": self.command,
                  "listener": self.listener, "exit_code": None,
                  "killed_by_verifier": False, "cleanup_confirmed": False}
        try:
            if success:
                try:
                    self.process.wait(timeout=deadline.remaining())
                except (subprocess.TimeoutExpired, RuntimeError):
                    error = "worker did not finish inside the shared deadline"
            if self.process.poll() is None:
                self.process.kill()
                killed = True
                result["killed_by_verifier"] = True
                try:
                    self.process.wait(timeout=min(1, deadline.cleanup_remaining()))
                except subprocess.TimeoutExpired as failure:
                    raise RuntimeError(f"owned worker cleanup unconfirmed: pid={self.process.pid}") from failure
            extra_stdout = available_stdout(self.process.stdout)
            self.stderr.seek(0)
            stderr = self.stderr.read(MAX_LISTENER_BYTES + 1)
            result.update(exit_code=self.process.returncode, cleanup_confirmed=True,
                          killed_by_verifier=killed, extra_stdout_bytes=len(extra_stdout),
                          stderr=stderr.decode(errors="replace"))
            if success:
                local.require(error is None, error)
                local.require(self.process.returncode == 0 and not killed,
                              "positive TCP worker did not exit successfully")
                local.require(not extra_stdout and not stderr,
                              "positive TCP worker emitted unexpected output")
            return result
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as failure:
            result["error"] = str(failure)
            raise WorkerFailure(str(failure), result) from failure
        finally:
            self.closed = True
            self.process.stdout.close()
            self.stderr.close()


def start_workers(report, identities, roles, deadline, evidence=None):
    workers = []
    try:
        for fragment in report["fragments"]:
            name = fragment["file_name"]
            workers.append(OwnedWorker.start(roles[name] / name, identities[name],
                                             fragment["descriptor"], deadline))
        addresses = [worker.listener["address"] for worker in workers]
        local.require(len(addresses) == len(set(addresses)), "duplicate listener endpoints")
        return workers
    except Exception as failure:
        if evidence is not None and isinstance(failure, WorkerFailure):
            evidence.append(failure.record)
        stop_workers(workers, False, deadline, evidence)
        raise


def stop_workers(workers, success, deadline, evidence=None):
    results = []
    error = None
    for worker in workers:
        try:
            result = worker.close(success, deadline)
            if result is not None:
                results.append(result)
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as failure:
            if isinstance(failure, WorkerFailure):
                results.append(failure.record)
            error = failure if error is None else error
    if evidence is not None:
        evidence.extend(results)
    if error is not None:
        raise error
    return results


def stage_roles(bundle, output, identities):
    roles = {}
    for name, identity in identities.items():
        directory = output / ("coordinator-only" if name == "coordinator" else f"worker-{name.removeprefix('fragment_')}-only")
        directory.mkdir()
        shutil.copy2(bundle / "run" / name, directory / name)
        local.elf_identity(directory / name, identity["sha256"], identity["bytes"])
        local.require([p.name for p in directory.iterdir()] == [name], "role directory has extra files")
        roles[name] = directory
    return roles


def invoke_coordinator(roles, arguments, deadline, summary, label):
    command = [roles["coordinator"] / "coordinator", *arguments]
    result = local.invoke(command, roles["coordinator"], deadline.remaining(minimum=1))
    summary["commands"].append({"label": label, "argv": list(map(str, command)), "exit_code": result[0]})
    return result


def checked_tcp_parity(reference, mono, tcp, case, rows, source):
    local.require(isinstance(tcp, dict) and tcp.get("mode") == "tcp-test",
                  "expected real tcp-test mode")
    local.require(isinstance(mono, dict) and mono.get("source_sha256") == source
                  and tcp.get("source_sha256") == source, "coordinator source identity mismatch")
    # Only adapt a copy for the established numeric comparator; persist/report
    # the original tcp-test output unchanged.
    comparison_copy = deepcopy(tcp)
    comparison_copy["mode"] = "fragments"
    result = local.compare_outputs(reference, mono, comparison_copy, case, rows)
    result["mode"] = "tcp-test"
    return result


def rejected(result, name, markers):
    code, stdout, stderr = result
    text = stderr.decode(errors="replace")
    local.require(code != 0 and not stdout and any(marker in text.lower() for marker in markers),
                  f"{name} did not fail with the expected diagnostic")
    return {"name": name, "exit_code": code, "stdout_bytes": len(stdout), "stderr": text[:16384]}


def failure_checks(report, identities, roles, deadline, summary, case):
    checks = summary.setdefault("failure_checks", [])
    fixed = [
        ("unknown-argument", ["--mode", "tcp-test", "--not-a-mode-argument"], ["unknown"]),
        ("unknown-mode", ["--mode", "not-a-mode", *local.case_args(case)], ["invalid fragment execution mode"]),
    ]
    for name, arguments, markers in fixed:
        checks.append(rejected(invoke_coordinator(roles, arguments, deadline, summary, name), name, markers))
    # Listen only on verifier-owned loopback sockets, but do not accept until
    # the bad-input coordinator exits. Any queued connection violates preflight.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as first, \
            socket.socket(socket.AF_INET, socket.SOCK_STREAM) as second:
        endpoints = []
        for listener in (first, second):
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            listener.setblocking(False)
            endpoints.append(f"127.0.0.1:{listener.getsockname()[1]}")
        bad_case = {"name": "invalid-token-preflight", "tokens": [4294967295],
                    "predict": 1, "context": case["context"]}
        arguments = ["--mode", "tcp-test", "--fragment-endpoints", ",".join(endpoints),
                     *local.case_args(bad_case)]
        result = invoke_coordinator(roles, arguments, deadline, summary, "invalid-token-preflight")
        connections = 0
        for listener in (first, second):
            try:
                connection, _ = listener.accept()
            except BlockingIOError:
                continue
            connection.close()
            connections += 1
        local.require(connections == 0, "invalid token preflight connected to a worker")
        check = rejected(result, "invalid-token-preflight", ["token"])
        check["connections_accepted"] = connections
        checks.append(check)
    # A bound, non-listening socket reserves this port throughout the check:
    # there is no free-port race or connection to another process.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reserved, \
            socket.socket(socket.AF_INET, socket.SOCK_STREAM) as second:
        reserved.bind(("127.0.0.1", 0))
        second.bind(("127.0.0.1", 0))
        address = f"127.0.0.1:{reserved.getsockname()[1]}"
        second_address = f"127.0.0.1:{second.getsockname()[1]}"
        nonloopback = f"0.0.0.0:{reserved.getsockname()[1]}"
        arguments = ["--mode", "tcp-test", "--fragment-endpoints", nonloopback + "," + second_address, *local.case_args(case)]
        checks.append(rejected(invoke_coordinator(roles, arguments, deadline, summary, "nonloopback"),
                               "nonloopback", ["loopback"]))
        arguments = ["--mode", "tcp-test", "--fragment-endpoints", address + "," + second_address, *local.case_args(case)]
        checks.append(rejected(invoke_coordinator(roles, arguments, deadline, summary, "missing-listener"),
                               "missing-listener", ["connect", "refused"]))
    for name in ["swapped-workers", "duplicate-endpoint"]:
        workers = start_workers(report, identities, roles, deadline, summary["negative_workers"])
        try:
            addresses = [worker.listener["address"] for worker in workers]
            endpoints = list(reversed(addresses)) if name == "swapped-workers" else [addresses[0], addresses[0]]
            arguments = ["--mode", "tcp-test", "--fragment-endpoints", ",".join(endpoints), *local.case_args(case)]
            result = invoke_coordinator(roles, arguments, deadline, summary, name)
            markers = ["descriptor", "identity", "handshake"] if name == "swapped-workers" else ["duplicate"]
            checks.append(rejected(result, name, markers))
        finally:
            stop_workers(workers, False, deadline, summary["negative_workers"])
    # The verifier owns this one-shot loopback socket and immediately closes an
    # accepted connection. It never fabricates a successful worker response.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener, \
            socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reserved:
        listener.bind(("127.0.0.1", 0))
        reserved.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.setblocking(False)
        address = f"127.0.0.1:{listener.getsockname()[1]}"
        failures = []
        cancelled = threading.Event()
        accepted = threading.Event()
        def accept_and_close():
            try:
                with selectors.DefaultSelector() as selector:
                    selector.register(listener, selectors.EVENT_READ)
                    while not cancelled.is_set():
                        if not selector.select(min(0.05, deadline.remaining())):
                            continue
                        connection, _ = listener.accept()
                        connection.close()
                        accepted.set()
                        return
            except (OSError, RuntimeError) as error:
                failures.append(str(error))
        thread = threading.Thread(target=accept_and_close, daemon=True)
        thread.start()
        try:
            second_address = f"127.0.0.1:{reserved.getsockname()[1]}"
            arguments = ["--mode", "tcp-test", "--fragment-endpoints", address + "," + second_address, *local.case_args(case)]
            checks.append(rejected(invoke_coordinator(roles, arguments, deadline, summary, "disconnect"),
                                   "disconnect", ["eof", "closed", "failed to fill", "reset", "handshake"]))
        finally:
            cancelled.set()
            thread.join(timeout=min(1, deadline.cleanup_remaining()))
            listener.close()
        local.require(not thread.is_alive() and accepted.is_set() and not failures,
                      "disconnect fixture did not finish its owned connection")
    return checks


def validate(args):
    deadline = Deadline(args.timeout)
    bundle = args.bundle.resolve(strict=True)
    report, identities = local.verify_bundle(bundle, args.build_report_sha256)
    local.require(report["workers"] == 2, "this TCP verifier requires exactly two worker roles")
    references = local.read_json(args.reference_json, args.reference_sha256)
    local.require(isinstance(references, dict) and references.get("format_version") == 1
                  and references.get("source_sha256") == report["source_sha256"],
                  "reference source or version mismatch")
    entries = references.get("cases")
    local.require(isinstance(entries, list) and all(isinstance(x, dict) for x in entries), "invalid reference cases")
    local.cases_from([entry.get("request") for entry in entries])
    for entry in entries:
        local.checked_run(entry.get("result"), entry["request"], report["head_rows"])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    roles = stage_roles(bundle, output, identities)
    summary = {"status": "failed", "scope": "local-single-host-native-head-fragments-loopback-tcp",
               "transport": "tcp-test", "tls": False, "cryptographic_peer_authentication": False,
               "distinct_hosts_verified": False, "filesystem_isolation_verified": False,
               "symbolic_bmlx_evidence": False, "runtime_model_argument": False,
               "coordinator_retains_full_model_including_head": True,
               "build_report_sha256": args.build_report_sha256, "reference_sha256": args.reference_sha256,
               "source_sha256": report["source_sha256"], "artifacts": identities,
               "role_directories": {name: str(path) for name, path in roles.items()},
               "cases": [], "workers": [], "negative_workers": [], "failure_checks": [], "commands": [],
               "deadline_seconds": args.timeout}
    try:
        for entry in entries:
            case = entry["request"]
            mono = local.success_json(invoke_coordinator(roles, ["--mode", "monolithic", *local.case_args(case)],
                                                         deadline, summary, case["name"] + "-monolithic"))
            local.save_json(output / (case["name"] + "-monolithic.json"), mono)
            workers = start_workers(report, identities, roles, deadline, summary["workers"])
            succeeded = False
            try:
                endpoints = ",".join(worker.listener["address"] for worker in workers)
                tcp = local.success_json(invoke_coordinator(roles, ["--mode", "tcp-test", "--fragment-endpoints", endpoints,
                                                                   *local.case_args(case)], deadline, summary, case["name"] + "-tcp-test"))
                local.save_json(output / (case["name"] + "-tcp-test.json"), tcp)
                summary["cases"].append(checked_tcp_parity(entry["result"], mono, tcp, case, report["head_rows"], report["source_sha256"]))
                succeeded = True
            finally:
                stop_workers(workers, succeeded, deadline, summary["workers"])
        summary["failure_checks"] = failure_checks(report, identities, roles, deadline, summary, entries[0]["request"])
        deadline.remaining()
        summary["status"] = "passed"
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        summary["error"] = str(error)
    summary["elapsed_seconds"] = time.monotonic() - deadline.started
    local.save_json(output / "report.json", summary)
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ["--bundle", "--reference-json", "--output"]:
        p.add_argument(flag, type=Path, required=True)
    for flag in ["--build-report-sha256", "--reference-sha256"]:
        p.add_argument(flag, required=True)
    p.add_argument("--trusted-generated-artifacts", action="store_true", required=True)
    p.add_argument("--timeout", type=int, default=110)
    return p


def main(argv=None):
    try:
        result = validate(parser().parse_args(argv))
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0 if result["status"] == "passed" else 2
    except (OSError, ValueError, RuntimeError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
