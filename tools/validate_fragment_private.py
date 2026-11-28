#!/usr/bin/env python3
"""Bounded same-host RFC1918/mTLS diagnostic for trusted native fragment ELFs.

Plan A uses two private TLS workers and two loopback client proxies. Plan B
uses native worker zero on loopback and worker one with certificate-first peer
bootstrap over TLS. Both execute every reference case and compare every F32
prefill/decode bit through the existing v2 comparator. Python bridges remain
required; this is not two-host, namespace-isolation, native-TLS or Lambda proof.
Only role credentials are copied. Issuer material is never opened or staged.
"""

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import ssl
import subprocess
import sys
import tempfile
import time

import fragment_private_transport as transport
import validate_fragment_elf as local
import validate_fragment_tcp as tcp


MAX_CONTROL_BYTES = 16 * 1024
ROLE_NAMES = ("driver", "worker-0", "worker-1")
TLS_NAMES = {"ca_certificate": "ca.crt", "certificate": "cert.crt", "private_key": "key.key"}


class Deadline:
    def __init__(self, seconds):
        local.require(type(seconds) is int and 1 <= seconds <= 360, "timeout must be 1..360 seconds")
        self.started = time.monotonic()
        self.cleanup_end = self.started + seconds
        self.end = self.cleanup_end - min(10, seconds / 2)

    def remaining(self, minimum=0):
        value = self.end - time.monotonic()
        if value <= minimum:
            raise RuntimeError("private validator exhausted its shared deadline")
        return value

    def cleanup_remaining(self):
        return max(0, self.cleanup_end - time.monotonic())


def certificate_digest(path):
    with transport.checked_file(path, 64 * 1024) as fd:
        pem = os.read(fd, 64 * 1024 + 1)
    try:
        return hashlib.sha256(ssl.PEM_cert_to_DER_cert(pem.decode("ascii"))).hexdigest()
    except (UnicodeError, ValueError):
        raise ValueError("invalid role certificate encoding") from None


def reject_symlink_parents(path):
    for component in [path, *path.parents]:
        local.require(not component.is_symlink(), "input/output path contains a symlink")


def verify_credentials(directory):
    directory = Path(directory).absolute()
    reject_symlink_parents(directory)
    manifest = local.read_json(directory / "manifest.json")
    local.require(isinstance(manifest, dict) and set(manifest) == {
        "format", "run_id", "openssl_version", "ca_certificate_sha256", "roles"},
        "invalid credential manifest fields")
    local.require(manifest["format"] == "bml-fragment-run-credentials-v1", "credential format mismatch")
    run_id = manifest["run_id"]
    local.require(isinstance(run_id, str) and re.fullmatch(r"[0-9a-f]{32}", run_id), "invalid credential run ID")
    local.digest(manifest["ca_certificate_sha256"])
    local.require(isinstance(manifest["roles"], dict) and set(manifest["roles"]) == set(ROLE_NAMES),
                  "credential roles missing or duplicated")
    paths, pins = {}, set()
    for role in ROLE_NAMES:
        record = manifest["roles"][role]
        local.require(isinstance(record, dict) and set(record) == {"files", "certificate_sha256", "server_name"},
                      "invalid credential role fields")
        pin = local.digest(record["certificate_sha256"])
        local.require(pin not in pins, "role certificates must be distinct")
        pins.add(pin)
        local.require(record["server_name"] == f"{role}.{run_id}.bml.invalid", "role name/run identity mismatch")
        expected = {key: f"runtime/{role}/{name}" for key, name in TLS_NAMES.items()}
        local.require(record["files"] == expected, "credential paths do not belong to the expected role")
        paths[role] = {key: directory / relative for key, relative in expected.items()}
        for key, path in paths[role].items():
            reject_symlink_parents(path)
            with transport.checked_file(path, 64 * 1024, private_key=key == "private_key"):
                pass
        local.require(certificate_digest(paths[role]["certificate"]) == pin, "role certificate pin mismatch")
        local.require(certificate_digest(paths[role]["ca_certificate"]) == manifest["ca_certificate_sha256"],
                      "role CA identity mismatch")
    return manifest, paths


def private_copy(source, destination):
    with transport.checked_file(source, 64 * 1024, private_key=destination.name == "key.key") as source_fd:
        data = os.read(source_fd, 64 * 1024 + 1)
    local.require(len(data) <= 64 * 1024, "credential copy exceeds quota")
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as file:
        os.fchmod(file.fileno(), 0o600)
        file.write(data)


def stage_roles(bundle, output, identities, credential_paths, manifest):
    roles = {}
    for role, executable in (("driver", "coordinator"), ("worker-0", "fragment_0"), ("worker-1", "fragment_1")):
        directory = output / role
        directory.mkdir(mode=0o700)
        target = directory / executable
        shutil.copy2(bundle / "run" / executable, target)
        local.elf_identity(target, identities[executable]["sha256"], identities[executable]["bytes"])
        tls = directory / "tls"
        tls.mkdir(mode=0o700)
        for key, filename in TLS_NAMES.items():
            private_copy(credential_paths[role][key], tls / filename)
        local.require(certificate_digest(tls / "cert.crt") == manifest["roles"][role]["certificate_sha256"],
                      "copied certificate identity mismatch")
        local.require(certificate_digest(tls / "ca.crt") == manifest["ca_certificate_sha256"],
                      "copied CA identity mismatch")
        local.require({p.name for p in directory.iterdir()} == {executable, "tls"}, "extra runtime role files")
        local.require({p.name for p in tls.iterdir()} == set(TLS_NAMES.values()), "extra runtime credential files")
        roles[role] = directory
    return roles


def configuration(role, index, report, identities, roles, credentials, worker_ip, driver_ip,
                  port=0, bootstrap=False):
    own = f"worker-{index}" if role == "worker" else "driver"
    peer = "driver" if role == "worker" else f"worker-{index}"
    tls_dir = roles[own] / "tls"
    value = {"version": 1, "run_id": credentials["run_id"],
             "descriptor": deepcopy(report["fragments"][index]["descriptor"]),
             "tls": {"ca_file": str(tls_dir / "ca.crt"), "cert_file": str(tls_dir / "cert.crt"),
                     "key_file": str(tls_dir / "key.key"),
                     "peer_cert_sha256": credentials["roles"][peer]["certificate_sha256"],
                     "server_name": credentials["roles"][peer]["server_name"]},
             "port": port, "startup_seconds": 15}
    if role == "worker":
        name = f"fragment_{index}"
        value.update(elf_file=str(roles[own] / name), elf_sha256=identities[name]["sha256"],
                     bind_ip=worker_ip, allowed_peer_ips=[] if bootstrap else [driver_ip],
                     bootstrap_peer_from_certificate=bootstrap)
    else:
        value.update(peer_ip=worker_ip, source_ip=driver_ip, listen_ip="127.0.0.1", listen_port=0)
    return transport.validate_config(value, role, test_ephemeral_port=role == "worker")


def verify_bound_config(actual, expected, role):
    actual = transport.validate_config(actual, role, test_ephemeral_port=role == "worker")
    local.require(json.dumps(actual, sort_keys=True, allow_nan=False)
                  == json.dumps(expected, sort_keys=True, allow_nan=False),
                  "configuration differs from pinned role/fragment binding")
    return actual


def parse_ready(raw, role, run_id, expected_ip):
    local.require(isinstance(raw, bytes) and 0 < len(raw) <= MAX_CONTROL_BYTES
                  and raw.endswith(b"\n") and raw.count(b"\n") == 1,
                  "private readiness must be exactly one bounded line")
    value = transport.parse_json(raw)
    local.require(isinstance(value, dict) and set(value) == {"format", "address", "run_id"},
                  "invalid private readiness fields")
    local.require(value["format"] == f"bml-private-fragment-{role}-v1" and value["run_id"] == run_id,
                  "private readiness role/run mismatch")
    address = value["address"]
    local.require(isinstance(address, str), "private readiness address must be text")
    host, separator, port = address.partition(":")
    local.require(separator == ":" and host == expected_ip and port.isascii() and port.isdecimal()
                  and port == str(int(port)) and 1 <= int(port) <= 65535,
                  "private readiness endpoint mismatch")
    return value


def inspect_stderr(raw, bootstrap, run_id, driver_ip):
    if not bootstrap:
        local.require(not raw, "positive process emitted unexpected stderr")
        return None
    local.require(0 < len(raw) <= MAX_CONTROL_BYTES and raw.endswith(b"\n") and raw.count(b"\n") == 1,
                  "missing or invalid authenticated-peer bootstrap event")
    event = transport.parse_json(raw)
    expected = {"format": "bml-private-fragment-authenticated-peer-v1", "run_id": run_id, "peer_ip": driver_ip}
    local.require(event == expected, "authenticated-peer bootstrap event mismatch")
    return expected


class OwnedProcess:
    """One new owned session; group cleanup includes the core's ELF child."""
    def __init__(self, process, stderr, command, label, bootstrap=False):
        self.process, self.stderr, self.command = process, stderr, list(map(str, command))
        self.label, self.bootstrap, self.ready, self.closed = label, bootstrap, None, False

    @classmethod
    def start(cls, command, cwd, label, deadline, parse, processes, bootstrap=False):
        deadline.remaining()
        stderr = tempfile.TemporaryFile()
        def limits():
            resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_CONTROL_BYTES, MAX_CONTROL_BYTES))
        try:
            process = subprocess.Popen(list(map(str, command)), cwd=cwd, stdin=subprocess.DEVNULL,
                                       stdout=subprocess.PIPE, stderr=stderr, bufsize=0,
                                       start_new_session=True, preexec_fn=limits)
        except BaseException:
            stderr.close()
            raise
        owned = cls(process, stderr, command, label, bootstrap)
        processes.append(owned)  # Register before any fallible readiness read.
        owned.ready = parse(tcp.read_listener_line(process.stdout, deadline))
        return owned

    def close(self, success, deadline, credentials, driver_ip):
        if self.closed:
            return None
        record = {"label": self.label, "pid": self.process.pid, "command": self.command,
                  "ready": self.ready, "exit_code": None, "cleanup_confirmed": False,
                  "killed_by_verifier": False, "bootstrap_event": None}
        failure = None
        try:
            if success and self.process.poll() is None:
                try:
                    self.process.wait(timeout=min(15, deadline.remaining()))
                except (subprocess.TimeoutExpired, RuntimeError):
                    failure = "positive process exceeded the shared deadline"
            if self.process.poll() is None:
                os.killpg(self.process.pid, signal.SIGKILL)
                record["killed_by_verifier"] = True
                self.process.wait(timeout=min(1, deadline.cleanup_remaining()))
            # A lingering group is ours, never a process selected by name/IP.
            try:
                os.killpg(self.process.pid, 0)
            except ProcessLookupError:
                pass
            else:
                os.killpg(self.process.pid, signal.SIGKILL)
                record["killed_by_verifier"] = True
            extra = tcp.available_stdout(self.process.stdout)
            self.stderr.seek(0)
            stderr = self.stderr.read(MAX_CONTROL_BYTES + 1)
            record.update(exit_code=self.process.returncode, cleanup_confirmed=True,
                          extra_stdout_bytes=len(extra), stderr_bytes=len(stderr))
            if success:
                local.require(failure is None, failure)
                local.require(self.process.returncode == 0 and not record["killed_by_verifier"],
                              "positive owned process did not finish cleanly")
                local.require(not extra, "positive process wrote stdout after readiness")
                record["bootstrap_event"] = inspect_stderr(stderr, self.bootstrap, credentials["run_id"], driver_ip)
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
            record["error"] = str(error)
        finally:
            self.closed = True
            self.process.stdout.close()
            self.stderr.close()
        return record


def close_processes(processes, success, deadline, credentials, driver_ip, summary):
    records = []
    for process in reversed(processes):
        record = process.close(success, deadline, credentials, driver_ip)
        if record is not None:
            records.append(record)
    summary["processes"].extend(records)
    local.require(all(not r.get("error") and r["cleanup_confirmed"] for r in records),
                  "owned process exit or cleanup verification failed")


def checkpoint(output, summary):
    # Output is newly and exclusively created by this validator.
    with (output / "report.json").open("w") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")


def coordinator(roles, identities, arguments, deadline, summary, label):
    executable = roles["driver"] / "coordinator"
    local.elf_identity(executable, identities["coordinator"]["sha256"], identities["coordinator"]["bytes"])
    command = [executable, *arguments]
    result = local.invoke(command, roles["driver"], min(120, deadline.remaining(minimum=1)))
    summary["commands"].append({"label": label, "argv": list(map(str, command)), "exit_code": result[0],
                                "stdout_bytes": len(result[1]), "stderr_bytes": len(result[2])})
    local.require(result[0] == 0 and not result[2], "coordinator failed or emitted unexpected stderr")
    return local.success_json(result)


def launch_core(role, config, label, output, roles, index, deadline, processes, worker_ip, driver_ip):
    path = output / "configs" / (label + ".json")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as file:
        json.dump(config, file, sort_keys=True, allow_nan=False)
        file.write("\n")
    loaded = transport.load_config(path, role, test_ephemeral_port=role == "worker")
    verify_bound_config(loaded, config, role)
    if role == "worker":
        local.elf_identity(config["elf_file"], config["elf_sha256"])
    command = [sys.executable, Path(transport.__file__).resolve(), role, "--private-ipv4", "--config", path]
    if role == "worker":
        command.append("--test-ephemeral-port")
    expected_ip = worker_ip if role == "worker" else "127.0.0.1"
    cwd = roles[f"worker-{index}" if role == "worker" else "driver"]
    return OwnedProcess.start(command, cwd, label, deadline,
                              lambda raw: parse_ready(raw, role, config["run_id"], expected_ip),
                              processes, bootstrap=config.get("bootstrap_peer_from_certificate", False))


def run_plan(plan, entry, mono, report, identities, roles, credentials, output, args, deadline, summary):
    processes, endpoints = [], []
    case = entry["request"]
    prefix = plan + "-" + case["name"]
    succeeded = False
    try:
        for index, fragment in enumerate(report["fragments"]):
            if plan == "B" and index == 0:
                executable = roles["worker-0"] / "fragment_0"
                local.elf_identity(executable, identities["fragment_0"]["sha256"], identities["fragment_0"]["bytes"])
                worker = OwnedProcess.start([executable, "--tcp-test-listen", "127.0.0.1:0"], executable.parent,
                                           prefix + "-local-worker-0", deadline,
                                           lambda raw: tcp.parse_listener(raw, fragment["descriptor"]), processes)
                endpoints.append(worker.ready["address"])
                continue
            config = configuration("worker", index, report, identities, roles, credentials,
                                   args.worker_ip, args.driver_ip, bootstrap=plan == "B")
            worker = launch_core("worker", config, prefix + f"-tls-worker-{index}", output, roles, index,
                                 deadline, processes, args.worker_ip, args.driver_ip)
            port = int(worker.ready["address"].rsplit(":", 1)[1])
            config = configuration("client", index, report, identities, roles, credentials,
                                   args.worker_ip, args.driver_ip, port=port)
            proxy = launch_core("client", config, prefix + f"-client-{index}", output, roles, index,
                                deadline, processes, args.worker_ip, args.driver_ip)
            endpoints.append(proxy.ready["address"])
        local.require(len(set(endpoints)) == 2, "duplicate coordinator endpoints")
        result = coordinator(roles, identities, ["--mode", "tcp-test", "--fragment-endpoints", ",".join(endpoints),
                                                *local.case_args(case)], deadline, summary, prefix)
        local.save_json(output / (prefix + "-tcp-test.json"), result)
        comparison = tcp.checked_tcp_parity(entry["result"], mono, result, case,
                                            report["head_rows"], report["source_sha256"])
        comparison.update(plan=plan, tls_worker_indices=[0, 1] if plan == "A" else [1],
                          local_worker_indices=[] if plan == "A" else [0],
                          certificate_peer_bootstrap=plan == "B")
        summary["cases"].append(comparison)
        succeeded = True
    finally:
        close_processes(processes, succeeded, deadline, credentials, args.driver_ip, summary)


def validate(args):
    output = Path(args.output).absolute()
    reject_symlink_parents(output)
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    started = time.monotonic()
    summary = {"status": "failed", "scope": "same-host-private-ipv4-mtls-native-head-fragments",
               "tls": "TLS1.3/mTLS Python bridge", "native_elf_tls": False,
               "distinct_hosts_verified": False, "network_namespaces_verified": False,
               "filesystem_isolation_verified": False, "real_model_validation_claimed": False,
               "symbolic_bmlx_evidence": False, "runtime_external_gguf": False,
               "coordinator_retains_full_model_including_head": True,
               "trusted_loopback_hop": True, "cases": [], "processes": [], "commands": [],
               "stage": "preflight", "deadline_seconds": args.timeout}
    checkpoint(output, summary)
    try:
        local.require(args.trusted_generated_artifacts is True, "trusted generated artifacts acknowledgment required")
        local.require(args.test_ephemeral_ports is True, "diagnostic ephemeral-port acknowledgment required")
        deadline = Deadline(args.timeout)
        transport.private_ipv4(args.worker_ip)
        transport.private_ipv4(args.driver_ip)
        local.require(args.worker_ip != args.driver_ip, "diagnostic needs two distinct existing private IPv4 addresses")
        bundle = Path(args.bundle).absolute()
        report, identities = local.verify_bundle(bundle, args.build_report_sha256)
        local.require(report["workers"] == 2, "private validator requires exactly two fragment workers")
        references = local.read_json(args.reference_json, args.reference_sha256)
        local.require(isinstance(references, dict) and references.get("format_version") == 1
                      and references.get("source_sha256") == report["source_sha256"], "reference source/version mismatch")
        entries = references.get("cases")
        local.require(isinstance(entries, list) and len(entries) == 2
                      and all(isinstance(x, dict) for x in entries), "private diagnostic requires exactly two reference cases")
        local.cases_from([entry.get("request") for entry in entries])
        for entry in entries:
            local.checked_run(entry.get("result"), entry["request"], report["head_rows"])
        credentials, paths = verify_credentials(args.credentials_dir)
        roles = stage_roles(bundle, output, identities, paths, credentials)
        (output / "configs").mkdir(mode=0o700)
        summary.update(build_report_sha256=args.build_report_sha256, reference_sha256=args.reference_sha256,
                       source_sha256=report["source_sha256"], artifacts=identities,
                       private_addresses={"worker": args.worker_ip, "driver": args.driver_ip},
                       credential_run_id=credentials["run_id"],
                       certificates={role: {key: credentials["roles"][role][key]
                                            for key in ("certificate_sha256", "server_name")} for role in ROLE_NAMES},
                       transport_source_sha256=local.sha256(transport.__file__))
        for entry in entries:
            case = entry["request"]
            summary["stage"] = case["name"] + "-monolithic"
            checkpoint(output, summary)
            mono = coordinator(roles, identities, ["--mode", "monolithic", *local.case_args(case)],
                               deadline, summary, summary["stage"])
            local.save_json(output / (case["name"] + "-monolithic.json"), mono)
            for plan in ("A", "B"):
                summary["stage"] = plan + "-" + case["name"]
                checkpoint(output, summary)
                run_plan(plan, entry, mono, report, identities, roles, credentials, output, args, deadline, summary)
        deadline.remaining()
        summary.update(status="passed", stage="complete")
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        summary["error"] = str(error)
    finally:
        summary["elapsed_seconds"] = time.monotonic() - started
        checkpoint(output, summary)
    return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for flag in ("--bundle", "--reference-json", "--credentials-dir", "--output"):
        result.add_argument(flag, type=Path, required=True)
    for flag in ("--build-report-sha256", "--reference-sha256", "--worker-ip", "--driver-ip"):
        result.add_argument(flag, required=True)
    result.add_argument("--timeout", type=int, default=240)
    result.add_argument("--trusted-generated-artifacts", action="store_true", required=True)
    result.add_argument("--test-ephemeral-ports", action="store_true", required=True)
    return result


def main(argv=None):
    try:
        summary = validate(parser().parse_args(argv))
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
        return 0 if summary["status"] == "passed" else 2
    except (OSError, ValueError, RuntimeError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
