#!/usr/bin/env python3
"""Opt-in, one-session TLS 1.3/mTLS bridge for native fragment wire v1.

Only literal RFC1918 IPv4 endpoints are admitted; local coordinator listeners
remain 127.0.0.1. Peer chain, purpose, exact run-specific DNS SAN, DER pin and
ALPN are checked before a worker executable is started. DNS is never resolved.
Frames are validated and forwarded verbatim: no arithmetic or backend changes.

Limits: 256 KiB/frame body, 256 ordered requests, 20 s absolute handshake/frame
I/O deadlines, 120 s session, and at most 120 s startup accept. Deadlines are
checked on every nonblocking I/O retry, not renewed by partial reads/writes.
The owner must still provide an outer process deadline and aggregate model
budgets. This bridge authenticates a pinned peer; it does not sandbox that peer
or the trusted executable. There is no plaintext fallback or reconnect.
"""

import argparse
import contextlib
import errno
import hashlib
import hmac
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import select
import socket
import ssl
import stat
import struct
import subprocess
import sys
import time

from validate_fragment_elf import elf_identity


MAX_FRAME_BYTES = 256 * 1024
MAX_REQUESTS = 256
MAX_CONFIG_BYTES = 64 * 1024
IO_SECONDS = 20
SESSION_SECONDS = 120
ALPN = "bml-native-head-v1"
PRIVATE_NETWORKS = tuple(ipaddress.IPv4Network(value) for value in
                         ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"))
DESCRIPTOR_FIELDS = {"version", "source_sha256", "tensor_name", "row_start",
                     "row_end", "total_rows", "cols", "weights_sha256"}


class TransportError(ValueError):
    """A fail-closed transport/configuration/protocol error without secret data."""


def require(condition, message):
    if not condition:
        raise TransportError(message)


def integer(value, minimum, maximum, name):
    require(type(value) is int and minimum <= value <= maximum, f"invalid {name}")
    return value


def sha256_text(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value),
            "invalid lowercase SHA-256")
    return value


def exact_fields(value, required, optional=()):
    require(type(value) is dict and set(required) <= set(value)
            and set(value) <= set(required) | set(optional), "invalid configuration fields")


def private_ipv4(value):
    require(isinstance(value, str), "expected literal RFC1918 IPv4")
    try:
        address = ipaddress.IPv4Address(value)
    except ipaddress.AddressValueError:
        raise TransportError("expected literal RFC1918 IPv4") from None
    require(str(address) == value and any(address in network for network in PRIVATE_NETWORKS),
            "expected literal RFC1918 IPv4")
    return value


def validate_descriptor(value):
    exact_fields(value, DESCRIPTOR_FIELDS)
    integer(value["version"], 1, 1, "descriptor version")
    sha256_text(value["source_sha256"])
    sha256_text(value["weights_sha256"])
    require(isinstance(value["tensor_name"], str)
            and re.fullmatch(r"[A-Za-z0-9_.]{1,128}", value["tensor_name"]),
            "invalid descriptor tensor name")
    rows = integer(value["total_rows"], 1, 32768, "total rows")
    cols = integer(value["cols"], 1, 4096, "columns")
    start = integer(value["row_start"], 0, rows - 1, "row start")
    end = integer(value["row_end"], start + 1, rows, "row end")
    cells = (end - start) * cols
    require(cells <= 32 * 1024 * 1024, "fragment logical work exceeds budget")
    require(cells * 4 <= 64 * 1024 * 1024, "fragment weight bytes exceed budget")
    return value


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON field")
        result[key] = value
    return result


def parse_json(data):
    def reject_constant(_value):
        raise TransportError("nonfinite JSON constant")
    try:
        return json.loads(data, object_pairs_hook=unique_pairs, parse_constant=reject_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError):
        raise TransportError("invalid JSON encoding") from None


def validate_config(value, role, test_ephemeral_port=False):
    """Validate exact configuration fields; return a copy with explicit defaults."""
    require(role in ("worker", "client"), "invalid transport role")
    require(not test_ephemeral_port or role == "worker", "ephemeral private port is worker-test only")
    common = {"version", "run_id", "descriptor", "tls"}
    if role == "worker":
        required = common | {"elf_file", "elf_sha256", "bind_ip", "allowed_peer_ips",
                             "bootstrap_peer_from_certificate"}
        optional = {"port", "startup_seconds"}
    else:
        required = common | {"peer_ip", "listen_ip", "listen_port"}
        optional = {"port", "startup_seconds", "source_ip"}
    exact_fields(value, required, optional)
    result = dict(value)
    integer(result["version"], 1, 1, "configuration version")
    run_id = result["run_id"]
    require(isinstance(run_id, str) and re.fullmatch(r"[0-9a-f]{32}", run_id), "invalid run ID")
    validate_descriptor(result["descriptor"])
    result["port"] = integer(result.get("port", 44761),
                             0 if test_ephemeral_port else 1, 65535, "private port")
    result["startup_seconds"] = integer(result.get("startup_seconds", 120), 1, 120, "startup seconds")
    tls = result["tls"]
    exact_fields(tls, {"ca_file", "cert_file", "key_file", "peer_cert_sha256", "server_name"})
    for name in ("ca_file", "cert_file", "key_file"):
        require(isinstance(tls[name], str) and tls[name] and "\0" not in tls[name], "invalid TLS file path")
    sha256_text(tls["peer_cert_sha256"])
    name = tls["server_name"]
    pattern = (rf"driver\.{run_id}\.bml\.invalid" if role == "worker"
               else rf"worker-[0-7]\.{run_id}\.bml\.invalid")
    require(isinstance(name, str) and re.fullmatch(pattern, name), "peer DNS identity does not match run/role")
    if role == "worker":
        private_ipv4(result["bind_ip"])
        require(isinstance(result["elf_file"], str) and result["elf_file"] and "\0" not in result["elf_file"],
                "invalid ELF file path")
        sha256_text(result["elf_sha256"])
        allowed = result["allowed_peer_ips"]
        require(type(allowed) is list and len(allowed) <= 8, "invalid peer allowlist")
        for address in allowed:
            private_ipv4(address)
        require(len(set(allowed)) == len(allowed), "duplicate allowed peer IP")
        bootstrap = result["bootstrap_peer_from_certificate"]
        require(type(bootstrap) is bool, "invalid bootstrap flag")
        require(not allowed if bootstrap else bool(allowed), "bootstrap and peer allowlist conflict")
    else:
        private_ipv4(result["peer_ip"])
        source = result.setdefault("source_ip", None)
        if source is not None:
            private_ipv4(source)
        require(result["listen_ip"] == "127.0.0.1", "coordinator listener must be 127.0.0.1")
        integer(result["listen_port"], 0, 65535, "local listen port")
    return result


def load_config(path, role, test_ephemeral_port=False):
    path = Path(path)
    with path.open("rb") as handle:
        raw = handle.read(MAX_CONFIG_BYTES + 1)
    require(len(raw) <= MAX_CONFIG_BYTES, "configuration exceeds byte budget")
    result = validate_config(parse_json(raw), role, test_ephemeral_port)
    # Relative paths have a single stable meaning independent of the launch cwd.
    result["tls"] = dict(result["tls"])
    for name in ("ca_file", "cert_file", "key_file"):
        result["tls"][name] = str(path.absolute().parent / result["tls"][name])
    if role == "worker":
        result["elf_file"] = str(path.absolute().parent / result["elf_file"])
    return result


@contextlib.contextmanager
def checked_file(path, maximum, private_key=False):
    """Hold a no-follow regular-file descriptor for SSL loading and ELF exec."""
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        metadata = os.fstat(descriptor)
        require(stat.S_ISREG(metadata.st_mode) and 0 < metadata.st_size <= maximum,
                "credential/file type or byte budget invalid")
        if private_key:
            require(metadata.st_uid == os.getuid()
                    and stat.S_IMODE(metadata.st_mode) in (0o400, 0o600),
                    "private key must belong to current user with mode 0400 or 0600")
        yield descriptor
    finally:
        os.close(descriptor)


def tls_context(config, role):
    require(ssl.HAS_TLSv1_3, "TLS 1.3 unavailable")
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER if role == "worker" else ssl.PROTOCOL_TLS_CLIENT)
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    context.maximum_version = ssl.TLSVersion.TLSv1_3
    context.verify_mode = ssl.CERT_REQUIRED
    context.check_hostname = role == "client"
    context.set_alpn_protocols([ALPN])
    tls = config["tls"]
    # Linux /proc/self/fd retains the exact checked file instead of reopening a
    # replaceable key path. No key bytes are copied to argv, output or temp files.
    with checked_file(tls["ca_file"], 1024 * 1024) as ca_fd, \
            checked_file(tls["cert_file"], 1024 * 1024) as cert_fd, \
            checked_file(tls["key_file"], 16 * 1024, private_key=True) as key_fd:
        context.load_verify_locations(cafile=f"/proc/self/fd/{ca_fd}")
        # An explicit rejecting password callback prevents interactive prompts.
        def reject_password():
            raise TransportError("encrypted private keys are unsupported for unattended startup")
        context.load_cert_chain(f"/proc/self/fd/{cert_fd}", f"/proc/self/fd/{key_fd}",
                                password=reject_password)
    return context


def remaining(deadline):
    value = deadline - time.monotonic()
    require(value > 0, "transport deadline exceeded")
    return value


def phase_deadline(session_end):
    return min(session_end, time.monotonic() + IO_SECONDS)


def wait_io(handle, reading, deadline):
    while True:
        try:
            ready_r, ready_w, _ = select.select([handle] if reading else [],
                                                [] if reading else [handle], [], remaining(deadline))
            require(ready_r or ready_w, "transport deadline exceeded")
            remaining(deadline)
            return
        except InterruptedError:
            continue


def receive(handle, count, deadline):
    while True:
        remaining(deadline)
        try:
            result = os.read(handle, count) if isinstance(handle, int) else handle.recv(count)
            remaining(deadline)
            return result
        except ssl.SSLWantWriteError:
            wait_io(handle, False, deadline)
        except (ssl.SSLWantReadError, BlockingIOError):
            wait_io(handle, True, deadline)
        except ssl.SSLZeroReturnError:
            remaining(deadline)
            return b""
        except InterruptedError:
            continue


def read_exact(handle, count, deadline, allow_eof=False):
    result = bytearray()
    while len(result) < count:
        chunk = receive(handle, count - len(result), deadline)
        if not chunk:
            require(not result and allow_eof, "truncated fragment frame")
            return None
        result.extend(chunk)
    return bytes(result)


def write_all(handle, data, deadline):
    view = memoryview(data)
    while view:
        remaining(deadline)
        try:
            size = os.write(handle, view) if isinstance(handle, int) else handle.send(view)
            remaining(deadline)
            require(size > 0, "fragment transport closed during write")
            view = view[size:]
        except ssl.SSLWantReadError:
            wait_io(handle, True, deadline)
        except (ssl.SSLWantWriteError, BlockingIOError):
            wait_io(handle, False, deadline)
        except InterruptedError:
            continue


def read_frame(handle, deadline, allow_eof=False):
    prefix = read_exact(handle, 4, deadline, allow_eof)
    if prefix is None:
        return None
    length, = struct.unpack("<I", prefix)
    require(12 <= length <= MAX_FRAME_BYTES, "fragment frame length exceeds bounds")
    return prefix + read_exact(handle, length, deadline)


def validate_frame(frame, kind, descriptor, sequence=0):
    require(isinstance(frame, bytes) and 16 <= len(frame) <= MAX_FRAME_BYTES + 4,
            "fragment frame length exceeds bounds")
    length, = struct.unpack_from("<I", frame)
    require(length == len(frame) - 4, "fragment frame length mismatch")
    require(frame[4:8] == b"BMLF" and struct.unpack_from("<II", frame, 8) == (1, kind),
            "fragment envelope mismatch")
    payload = frame[16:]
    if kind == 1:
        require(len(payload) <= 4096, "Ready descriptor exceeds budget")
        actual = validate_descriptor(parse_json(payload))
        require(actual == descriptor, "fragment Ready identity mismatch")
        return
    integer(sequence, 0, MAX_REQUESTS - 1, "fragment request sequence/quota")
    if kind == 2:
        require(len(payload) >= 8, "truncated projection request")
        seq, count = struct.unpack_from("<II", payload)
        require(seq == sequence and count == descriptor["cols"], "fragment request sequence/shape mismatch")
        values = payload[8:]
    elif kind == 3:
        require(len(payload) >= 16, "truncated projection output")
        seq, start, end, count = struct.unpack_from("<IIII", payload)
        require(seq == sequence and start == descriptor["row_start"] and end == descriptor["row_end"]
                and count == end - start, "fragment output sequence/coverage mismatch")
        values = payload[16:]
    else:
        raise TransportError("unsupported fragment frame kind")
    require(len(values) == count * 4, "fragment vector length mismatch")
    require(all(math.isfinite(value[0]) for value in struct.iter_unpack("<f", values)),
            "fragment vector is not finite")


def tls_handshake(connection, context, config, role, deadline):
    tls = context.wrap_socket(connection, server_side=role == "worker",
                              server_hostname=config["tls"]["server_name"] if role == "client" else None,
                              do_handshake_on_connect=False, suppress_ragged_eofs=False)
    tls.setblocking(False)
    try:
        while True:
            remaining(deadline)
            try:
                tls.do_handshake()
                remaining(deadline)
                break
            except ssl.SSLWantReadError:
                wait_io(tls, True, deadline)
            except ssl.SSLWantWriteError:
                wait_io(tls, False, deadline)
        require(tls.version() == "TLSv1.3" and tls.selected_alpn_protocol() == ALPN,
                "TLS version/ALPN mismatch")
        peer = tls.getpeercert(binary_form=True)
        require(peer and hmac.compare_digest(hashlib.sha256(peer).hexdigest(),
                                            config["tls"]["peer_cert_sha256"]), "TLS peer certificate pin mismatch")
        # Standard ssl already verifies the chain and role purpose. The exact
        # sole SAN binds the accepted identity to this run, including server side.
        require(tls.getpeercert().get("subjectAltName") == (("DNS", config["tls"]["server_name"]),),
                "TLS peer DNS identity mismatch")
        remaining(deadline)
        return tls
    except BaseException:
        tls.close()
        raise


def tls_close(connection, deadline):
    """Exchange close_notify; an abrupt TCP EOF never counts as clean success."""
    while True:
        remaining(deadline)
        try:
            raw = connection.unwrap()
            raw.close()
            remaining(deadline)
            return
        except ssl.SSLWantReadError:
            wait_io(connection, True, deadline)
        except ssl.SSLWantWriteError:
            wait_io(connection, False, deadline)


def accept_one(listener, deadline):
    while True:
        remaining(deadline)
        try:
            connection, peer = listener.accept()
            try:
                connection.setblocking(False)
                remaining(deadline)
                return connection, peer
            except BaseException:
                connection.close()
                raise
        except BlockingIOError:
            wait_io(listener, True, deadline)
        except InterruptedError:
            continue


def connect_private(config, deadline):
    connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connection.setblocking(False)
    try:
        if config["source_ip"] is not None:
            connection.bind((config["source_ip"], 0))
        status = connection.connect_ex((config["peer_ip"], config["port"]))
        if status != 0:
            require(status in (errno.EINPROGRESS, errno.EWOULDBLOCK, errno.EINTR), "private TCP connect failed")
            wait_io(connection, False, deadline)
            require(connection.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR) == 0, "private TCP connect failed")
        private_ipv4(connection.getsockname()[0])
        require(connection.getpeername() == (config["peer_ip"], config["port"]), "private peer endpoint mismatch")
        remaining(deadline)
        return connection
    except BaseException:
        connection.close()
        raise


def announce(role, listener, config):
    address, port = listener.getsockname()
    print(json.dumps({"format": f"bml-private-fragment-{role}-v1", "address": f"{address}:{port}",
                      "run_id": config["run_id"]}, separators=(",", ":")), flush=True)


def relay(descriptor, request_source, request_target, response_source, response_target, session_end):
    """Relay one Ready and at most 256 sequential exchanges; clean EOF only at a boundary."""
    deadline = phase_deadline(session_end)
    ready = read_frame(response_source, deadline)
    validate_frame(ready, 1, descriptor)
    write_all(response_target, ready, deadline)
    for sequence in range(MAX_REQUESTS + 1):
        deadline = phase_deadline(session_end)
        request = read_frame(request_source, deadline, allow_eof=True)
        if request is None:
            return
        validate_frame(request, 2, descriptor, sequence)
        write_all(request_target, request, deadline)
        deadline = phase_deadline(session_end)
        response = read_frame(response_source, deadline)
        validate_frame(response, 3, descriptor, sequence)
        write_all(response_target, response, deadline)
    raise TransportError("fragment request quota exhausted")


def serve_child(config, connection, session_end):
    # Use the existing v1 AMD64/hash validator, then pin the actual opened ELF
    # bytes and execute through the held FD so path replacement cannot swap it.
    # The owner must prevent writes to the trusted inode; this is not a snapshot.
    elf_identity(config["elf_file"], config["elf_sha256"])
    with checked_file(config["elf_file"], 1024 * 1024 * 1024) as elf_fd:
        digest = hashlib.sha256()
        while True:
            remaining(session_end)
            chunk = os.read(elf_fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        require(hmac.compare_digest(digest.hexdigest(), config["elf_sha256"]), "held ELF identity mismatch")
        remaining(session_end)
        child = subprocess.Popen([f"/proc/self/fd/{elf_fd}"], stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, pass_fds=(elf_fd,), bufsize=0)
    try:
        os.set_blocking(child.stdin.fileno(), False)
        os.set_blocking(child.stdout.fileno(), False)
        relay(config["descriptor"], connection, child.stdin.fileno(), child.stdout.fileno(),
              connection, session_end)
        child.stdin.close()
        deadline = phase_deadline(session_end)
        require(read_exact(child.stdout.fileno(), 1, deadline, allow_eof=True) is None,
                "worker emitted trailing output after clean EOF")
        require(child.wait(timeout=remaining(deadline)) == 0, "worker failed after clean EOF")
    finally:
        if child.poll() is None:
            child.kill()  # Only this Popen child is owned; never a peer/other PID.
        child.wait()
        child.stdin.close()
        child.stdout.close()


def run_worker(config):
    context = tls_context(config, "worker")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((config["bind_ip"], config["port"]))
        listener.listen(1)
        listener.setblocking(False)
        announce("worker", listener, config)
        connection, peer = accept_one(listener, time.monotonic() + config["startup_seconds"])
        with connection:
            session_end = time.monotonic() + SESSION_SECONDS
            private_ipv4(peer[0])
            require(config["bootstrap_peer_from_certificate"] or peer[0] in config["allowed_peer_ips"],
                    "peer IP is not allowed")
            with tls_handshake(connection, context, config, "worker", phase_deadline(session_end)) as tls:
                if config["bootstrap_peer_from_certificate"]:
                    print(json.dumps({"format": "bml-private-fragment-authenticated-peer-v1",
                                      "run_id": config["run_id"], "peer_ip": peer[0]},
                                     separators=(",", ":")), file=sys.stderr, flush=True)
                serve_child(config, tls, session_end)
                tls_close(tls, phase_deadline(session_end))


def run_client(config):
    context = tls_context(config, "client")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((config["listen_ip"], config["listen_port"]))
        listener.listen(1)
        listener.setblocking(False)
        announce("client", listener, config)
        local, peer = accept_one(listener, time.monotonic() + config["startup_seconds"])
        with local:
            require(peer[0] == "127.0.0.1", "coordinator peer must be 127.0.0.1")
            session_end = time.monotonic() + SESSION_SECONDS
            deadline = phase_deadline(session_end)
            with connect_private(config, deadline) as connection:
                with tls_handshake(connection, context, config, "client", deadline) as tls:
                    relay(config["descriptor"], local, tls, tls, local, session_end)
                    tls_close(tls, phase_deadline(session_end))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("worker", "client"))
    parser.add_argument("--private-ipv4", action="store_true", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--test-ephemeral-port", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, args.role, args.test_ephemeral_port)
        (run_worker if args.role == "worker" else run_client)(config)
        return 0
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        # Never serialize configuration, credentials, certificate subjects or
        # request data. strerror/reason omit path-bearing exception reprs.
        detail = (getattr(error, "reason", None) or error.strerror or "I/O failure") \
            if isinstance(error, OSError) else str(error)
        print(f"private transport error: {detail}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
