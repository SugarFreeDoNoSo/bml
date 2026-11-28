import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import socket
import ssl
import struct
import tempfile
import threading
import time
import unittest
from unittest import mock

import fragment_private_transport as transport
import fragment_run_credentials as credentials


def descriptor():
    return {"version": 1, "source_sha256": "a" * 64, "tensor_name": "output.weight",
            "row_start": 0, "row_end": 2, "total_rows": 4, "cols": 3, "weights_sha256": "b" * 64}


def config(role="worker"):
    run_id = "0123456789abcdef" * 2
    peer = "driver" if role == "worker" else "worker-1"
    value = {"version": 1, "run_id": run_id, "descriptor": descriptor(),
             "tls": {"ca_file": "ca.crt", "cert_file": "cert.crt", "key_file": "key.key",
                     "peer_cert_sha256": hashlib.sha256(b"certificate").hexdigest(),
                     "server_name": f"{peer}.{run_id}.bml.invalid"}}
    if role == "worker":
        value.update(elf_file="fragment_0", elf_sha256="c" * 64, bind_ip="10.0.0.1",
                     allowed_peer_ips=["10.0.0.2"], bootstrap_peer_from_certificate=False)
    else:
        value.update(peer_ip="10.0.0.1", listen_ip="127.0.0.1", listen_port=0)
    return value


def frame(kind, payload):
    body = b"BMLF" + struct.pack("<II", 1, kind) + payload
    return struct.pack("<I", len(body)) + body


def ready(value=None):
    return frame(1, json.dumps(value or descriptor()).encode())


def request(sequence=0, values=(-0.0, 2.0, -3.0)):
    return frame(2, struct.pack("<II", sequence, len(values)) + struct.pack(f"<{len(values)}f", *values))


def response(sequence=0, values=(-0.0, 1.0), start=0, end=2):
    return frame(3, struct.pack("<IIII", sequence, start, end, len(values))
                 + struct.pack(f"<{len(values)}f", *values))


class MemoryIO:
    def __init__(self, incoming=b""):
        self.incoming = incoming
        self.outgoing = bytearray()

    def recv(self, count):
        result = self.incoming[:min(count, 3)]
        self.incoming = self.incoming[len(result):]
        return result

    def send(self, data):
        count = min(len(data), 3)
        self.outgoing.extend(data[:count])
        return count


class ConfigurationTests(unittest.TestCase):
    def test_exact_rfc1918_boundaries_not_generic_is_private(self):
        for value in ("10.0.0.0", "10.255.255.255", "172.16.0.0", "172.31.255.255",
                      "192.168.0.0", "192.168.255.255"):
            self.assertEqual(transport.private_ipv4(value), value)
        for value in ("172.15.255.255", "172.32.0.0", "192.167.255.255", "192.169.0.0",
                      "127.0.0.1", "0.0.0.0", "169.254.1.1", "100.64.1.1", "8.8.8.8",
                      "192.0.2.1", "::1", "::ffff:10.0.0.1", "localhost", "10.0.0.01", 10):
            with self.subTest(value=value), self.assertRaisesRegex(transport.TransportError, "RFC1918"):
                transport.private_ipv4(value)

    def test_exact_fields_versions_shapes_and_defaults(self):
        for role in ("worker", "client"):
            value = transport.validate_config(config(role), role)
            self.assertEqual(value["port"], 44761)
            self.assertEqual(value["startup_seconds"], 120)
        bad_values = [dict(config(), unexpected=True), dict(config(), version=True),
                      dict(config(), run_id="A" * 32), dict(config(), startup_seconds=121),
                      dict(config(), startup_seconds=0), dict(config(), port=True),
                      dict(config(), port=65536)]
        for field, value in (("version", 2), ("cols", 0), ("cols", 4097), ("cols", True),
                             ("row_start", 2), ("row_start", -1), ("row_end", 5),
                             ("total_rows", 32769), ("tensor_name", "../bad"),
                             ("source_sha256", "a" * 63), ("weights_sha256", "C" * 64),
                             ("other", 1)):
            bad = config()
            bad["descriptor"][field] = value
            bad_values.append(bad)
        for bad in bad_values:
            with self.subTest(bad=bad), self.assertRaises(transport.TransportError):
                transport.validate_config(bad, "worker")

    def test_descriptor_work_and_bytes_caps_are_independent(self):
        for rows, cols, message in ((32768, 4096, "work"), (8192, 4096, "bytes")):
            value = descriptor()
            value.update(row_end=rows, total_rows=rows, cols=cols)
            with self.assertRaisesRegex(transport.TransportError, message):
                transport.validate_descriptor(value)
        value = descriptor()
        value.update(row_end=4096, total_rows=4096, cols=4096)
        self.assertEqual(transport.validate_descriptor(value), value)

    def test_bootstrap_allowlist_and_explicit_ephemeral_worker_only(self):
        value = config()
        value.update(port=0, allowed_peer_ips=[], bootstrap_peer_from_certificate=True)
        self.assertEqual(transport.validate_config(value, "worker", True)["port"], 0)
        with self.assertRaises(transport.TransportError):
            transport.validate_config(value, "worker")
        for bootstrap, allowed in ((False, []), (True, ["10.0.0.2"]),
                                   (False, ["10.0.0.2"] * 2), (1, ["10.0.0.2"]),
                                   (False, ["127.0.0.1"])):
            value = config()
            value.update(bootstrap_peer_from_certificate=bootstrap, allowed_peer_ips=allowed)
            with self.assertRaises(transport.TransportError):
                transport.validate_config(value, "worker")
        with self.assertRaises(transport.TransportError):
            transport.validate_config(config("client"), "client", True)
        value = config("client")
        value["port"] = 0
        with self.assertRaises(transport.TransportError):
            transport.validate_config(value, "client")

    def test_role_and_run_dns_binding_and_local_listener(self):
        for role in ("worker", "client"):
            for name in ("worker-0.other.bml.invalid", "driver.example.com", "10.0.0.1",
                         f"worker-8.{config()[ 'run_id']}.bml.invalid"):
                value = config(role)
                value["tls"]["server_name"] = name
                with self.assertRaises(transport.TransportError):
                    transport.validate_config(value, role)
        for field, bad in (("listen_ip", "0.0.0.0"), ("listen_port", -1),
                           ("source_ip", "127.0.0.1"), ("peer_ip", "hostname.invalid")):
            value = config("client")
            value[field] = bad
            with self.assertRaises(transport.TransportError):
                transport.validate_config(value, "client")

    def test_json_duplicates_nonfinite_budget_and_relative_paths(self):
        for value in (b'{"version":1,"version":1}', b'{"value":NaN}', b'\xff', b'{'):
            with self.assertRaises(transport.TransportError):
                transport.parse_json(value)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config()))
            result = transport.load_config(path, "worker")
            self.assertEqual(result["tls"]["key_file"], str(Path(directory) / "key.key"))
            self.assertEqual(result["elf_file"], str(Path(directory) / "fragment_0"))
            path.write_bytes(b" " * (transport.MAX_CONFIG_BYTES + 1))
            with self.assertRaisesRegex(transport.TransportError, "budget"):
                transport.load_config(path, "worker")

    def test_keys_reject_permissions_symlinks_size_and_other_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "key"
            path.write_bytes(b"not a real private key")
            for mode in (0o400, 0o600):
                path.chmod(mode)
                with transport.checked_file(path, 16384, True) as fd:
                    self.assertEqual(os.read(fd, 100), path.read_bytes())
            path.chmod(0o640)
            with self.assertRaisesRegex(transport.TransportError, "mode"):
                with transport.checked_file(path, 16384, True):
                    self.fail("accepted permissive key")
            path.chmod(0o600)
            with mock.patch.object(transport.os, "getuid", return_value=os.getuid() + 1):
                with self.assertRaisesRegex(transport.TransportError, "current user"):
                    with transport.checked_file(path, 16384, True):
                        self.fail("accepted another owner")
            alias = Path(directory) / "alias"
            alias.symlink_to(path)
            with self.assertRaises(OSError):
                with transport.checked_file(alias, 16384, True):
                    self.fail("followed a key symlink")
            path.write_bytes(b"x" * 16385)
            with self.assertRaisesRegex(transport.TransportError, "budget"):
                with transport.checked_file(path, 16384, True):
                    self.fail("accepted oversized key")

    def test_explicit_cli_flag_required_without_opening_config(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            transport.main(["worker", "--config", "missing.json"])
        self.assertEqual(error.exception.code, 2)


class WireTests(unittest.TestCase):
    def test_ready_identity_unknown_fields_duplicates_and_version(self):
        transport.validate_frame(ready(), 1, descriptor())
        for value in (dict(descriptor(), source_sha256="d" * 64), dict(descriptor(), other=1),
                      dict(descriptor(), version=2), dict(descriptor(), cols=True)):
            with self.assertRaises(transport.TransportError):
                transport.validate_frame(ready(value), 1, descriptor())
        for payload in (b'{"version":1,"version":1}', b" " * 4097):
            with self.assertRaises(transport.TransportError):
                transport.validate_frame(frame(1, payload), 1, descriptor())

    def test_requests_and_outputs_check_sequence_shape_finite_and_trailing_bytes(self):
        transport.validate_frame(request(), 2, descriptor())
        transport.validate_frame(response(), 3, descriptor())
        cases = [(request(1), 2), (request(values=(1.0,)), 2),
                 (request(values=(float("nan"), 1.0, 2.0)), 2),
                 (response(1), 3), (response(start=1, end=3), 3),
                 (response(values=(float("inf"), 0.0)), 3), (response(values=(1.0,)), 3),
                 (request() + b"x", 2), (request()[:-1], 2),
                 (frame(2, b""), 2), (frame(3, b""), 3)]
        for data, kind in cases:
            with self.subTest(kind=kind, data=data), self.assertRaises(transport.TransportError):
                transport.validate_frame(data, kind, descriptor())
        for offset in (4, 8, 12):
            bad = bytearray(request())
            bad[offset] ^= 8
            with self.assertRaisesRegex(transport.TransportError, "envelope"):
                transport.validate_frame(bytes(bad), 2, descriptor())
        with self.assertRaisesRegex(transport.TransportError, "quota"):
            transport.validate_frame(request(256), 2, descriptor(), 256)

    def test_frame_bound_rejects_before_body_and_eof_only_between_frames(self):
        deadline = time.monotonic() + 1
        self.assertIsNone(transport.read_frame(MemoryIO(), deadline, True))
        for data in (b"\0", struct.pack("<I", 12), struct.pack("<I", 11),
                     struct.pack("<I", transport.MAX_FRAME_BYTES + 1)):
            with self.assertRaises(transport.TransportError):
                transport.read_frame(MemoryIO(data), deadline, True)
        self.assertEqual(transport.read_frame(MemoryIO(request()), deadline), request())

    def test_relay_keeps_signed_zero_and_all_bits_verbatim(self):
        requests = request() + request(1, (1e-40, -0.0, 1.0))
        responses = ready() + response() + response(1, (1e-40, -0.0))
        source, target = MemoryIO(requests), MemoryIO(responses)
        transport.relay(descriptor(), source, target, target, source, time.monotonic() + 1)
        self.assertEqual(bytes(target.outgoing), requests)
        self.assertEqual(bytes(source.outgoing), responses)
        self.assertIn(struct.pack("<I", 0x80000000), source.outgoing)

    def test_relay_stops_on_first_invalid_response_without_reassociation(self):
        source = MemoryIO(request() + request(1))
        target = MemoryIO(ready() + response(1))
        with self.assertRaisesRegex(transport.TransportError, "sequence/coverage"):
            transport.relay(descriptor(), source, target, target, source, time.monotonic() + 1)
        self.assertEqual(bytes(target.outgoing), request())
        self.assertEqual(bytes(source.outgoing), ready())

    def test_request_quota_256_then_clean_eof_or_fail(self):
        requests = b"".join(request(i) for i in range(256))
        responses = ready() + b"".join(response(i) for i in range(256))
        for extra in (b"", request(256)):
            source, target = MemoryIO(requests + extra), MemoryIO(responses)
            if extra:
                with self.assertRaisesRegex(transport.TransportError, "quota"):
                    transport.relay(descriptor(), source, target, target, source, time.monotonic() + 2)
            else:
                transport.relay(descriptor(), source, target, target, source, time.monotonic() + 2)
            self.assertEqual(bytes(target.outgoing), requests)
            self.assertEqual(bytes(source.outgoing), responses)


class DeadlineAndAuthenticationTests(unittest.TestCase):
    def test_partial_frame_drip_cannot_renew_absolute_deadline(self):
        left, right = socket.socketpair()
        left.setblocking(False)
        data = request()

        def drip():
            try:
                for index in range(6):
                    right.sendall(data[index:index + 1])
                    time.sleep(0.04)
                right.sendall(data[6:])
            except (BrokenPipeError, OSError):
                pass
            finally:
                right.close()

        sender = threading.Thread(target=drip)
        sender.start()
        try:
            with self.assertRaisesRegex(transport.TransportError, "deadline"):
                transport.read_frame(left, time.monotonic() + 0.1)
        finally:
            left.close()
            sender.join(1)
        self.assertFalse(sender.is_alive())

    def test_nonblocking_pipe_and_stalled_write_are_bounded(self):
        read_fd, write_fd = os.pipe()
        try:
            os.set_blocking(read_fd, False)
            os.set_blocking(write_fd, False)
            with self.assertRaisesRegex(transport.TransportError, "deadline"):
                transport.read_frame(read_fd, time.monotonic() + 0.02)
            with self.assertRaisesRegex(transport.TransportError, "deadline"):
                transport.write_all(write_fd, b"x" * transport.MAX_FRAME_BYTES, time.monotonic() + 0.02)
        finally:
            os.close(read_fd)
            os.close(write_fd)

    def test_last_successful_syscall_cannot_finish_after_deadline(self):
        clock = [0.0]

        def late(value):
            def result(*_args, **_kwargs):
                clock[0] = 2.0
                return value
            return result

        with mock.patch.object(transport.time, "monotonic", side_effect=lambda: clock[0]):
            for operation in (lambda io: transport.receive(io, 1, 1),
                              lambda io: transport.write_all(io, b"x", 1)):
                clock[0] = 0.0
                endpoint = mock.Mock()
                endpoint.recv.side_effect = late(b"x")
                endpoint.send.side_effect = late(1)
                with self.assertRaisesRegex(transport.TransportError, "deadline"):
                    operation(endpoint)
            clock[0] = 0.0
            listener, accepted = mock.Mock(), mock.Mock()
            listener.accept.side_effect = late((accepted, ("10.0.0.1", 1)))
            with self.assertRaisesRegex(transport.TransportError, "deadline"):
                transport.accept_one(listener, 1)
            accepted.close.assert_called_once()
            clock[0] = 0.0
            tls, raw = mock.Mock(), mock.Mock()
            tls.unwrap.side_effect = late(raw)
            with self.assertRaisesRegex(transport.TransportError, "deadline"):
                transport.tls_close(tls, 1)
            raw.close.assert_called_once()

    def fake_tls(self, value):
        tls = mock.Mock()
        tls.version.return_value = "TLSv1.3"
        tls.selected_alpn_protocol.return_value = transport.ALPN
        tls.getpeercert.side_effect = lambda binary_form=False: b"certificate" if binary_form else {
            "subjectAltName": (("DNS", value["tls"]["server_name"]),)}
        return tls

    def test_handshake_requires_alpn_der_pin_san_and_run_before_return(self):
        value = config()
        for failure in (None, "version", "alpn", "pin", "san"):
            tls = self.fake_tls(value)
            attempt = copy.deepcopy(value)
            if failure == "version":
                tls.version.return_value = "TLSv1.2"
            elif failure == "alpn":
                tls.selected_alpn_protocol.return_value = None
            elif failure == "pin":
                attempt["tls"]["peer_cert_sha256"] = "a" * 64
            elif failure == "san":
                tls.getpeercert.side_effect = lambda binary_form=False: b"certificate" if binary_form else {
                    "subjectAltName": (("DNS", "driver.other.bml.invalid"),)}
            context = mock.Mock()
            context.wrap_socket.return_value = tls
            if failure is None:
                self.assertIs(transport.tls_handshake(mock.Mock(), context, attempt, "worker",
                                                      time.monotonic() + 1), tls)
                tls.close.assert_not_called()
            else:
                with self.subTest(failure=failure), self.assertRaises(transport.TransportError):
                    transport.tls_handshake(mock.Mock(), context, attempt, "worker", time.monotonic() + 1)
                tls.close.assert_called_once()
            self.assertFalse(context.wrap_socket.call_args.kwargs["suppress_ragged_eofs"])

    def test_handshake_last_success_past_deadline_fails_closed(self):
        value = config()
        tls = self.fake_tls(value)
        context = mock.Mock()
        context.wrap_socket.return_value = tls
        clock = [0.0]
        tls.do_handshake.side_effect = lambda: clock.__setitem__(0, 2)
        with mock.patch.object(transport.time, "monotonic", side_effect=lambda: clock[0]):
            with self.assertRaisesRegex(transport.TransportError, "deadline"):
                transport.tls_handshake(mock.Mock(), context, value, "worker", 1)
        tls.close.assert_called_once()
        tls.getpeercert.assert_not_called()

    def test_worker_never_launches_elf_before_authentication(self):
        value = transport.validate_config(config(), "worker")
        listener = mock.MagicMock()
        listener.__enter__.return_value = listener
        listener.getsockname.return_value = ("10.0.0.1", 44761)
        connection = mock.MagicMock()
        with mock.patch.object(transport.socket, "socket", return_value=listener), \
                mock.patch.object(transport, "tls_context"), \
                mock.patch.object(transport, "accept_one", return_value=(connection, ("10.0.0.2", 1234))), \
                mock.patch.object(transport, "tls_handshake", side_effect=transport.TransportError("bad peer pin")), \
                mock.patch.object(transport, "serve_child") as child, contextlib.redirect_stdout(io.StringIO()) as output:
            with self.assertRaisesRegex(transport.TransportError, "pin"):
                transport.run_worker(value)
        child.assert_not_called()
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(set(json.loads(lines[0])), {"format", "address", "run_id"})

    def test_owned_child_cleanup_and_held_elf_hash_precede_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trusted-elf"
            path.write_bytes(b"synthetic file; Popen and ELF validator are mocked")
            value = transport.validate_config(config(), "worker")
            value.update(elf_file=str(path), elf_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            read_fd, write_fd = os.pipe()
            child = mock.Mock()
            child.stdin = os.fdopen(write_fd, "wb", buffering=0)
            child.stdout = os.fdopen(read_fd, "rb", buffering=0)
            child.poll.return_value = None
            with mock.patch.object(transport, "elf_identity") as identity, \
                    mock.patch.object(transport.subprocess, "Popen", return_value=child) as spawn, \
                    mock.patch.object(transport, "relay", side_effect=transport.TransportError("malformed frame")):
                with self.assertRaisesRegex(transport.TransportError, "malformed"):
                    transport.serve_child(value, mock.Mock(), time.monotonic() + 1)
                identity.assert_called_once_with(str(path), value["elf_sha256"])
                self.assertTrue(spawn.call_args.args[0][0].startswith("/proc/self/fd/"))
                self.assertNotIn("start_new_session", spawn.call_args.kwargs)
                child.kill.assert_called_once()
                child.wait.assert_called_once()
                value["elf_sha256"] = "f" * 64
                spawn.reset_mock()
                with self.assertRaisesRegex(transport.TransportError, "held ELF identity"):
                    transport.serve_child(value, mock.Mock(), time.monotonic() + 1)
                spawn.assert_not_called()


@unittest.skipUnless(shutil.which("openssl"), "requires installed OpenSSL; no installation or network")
class RealTlsTests(unittest.TestCase):
    """Real cryptographic handshakes over owned AF_UNIX socketpairs, not host interfaces."""

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.pki = cls.root / "run"
        cls.manifest = credentials.create(cls.pki)
        cls.other_pki = cls.root / "other-run"
        cls.other_manifest = credentials.create(cls.other_pki)

    def configured(self, role):
        value = config(role)
        value["run_id"] = self.manifest["run_id"]
        own = "driver" if role == "client" else "worker-0"
        peer = "worker-0" if role == "client" else "driver"
        files = self.manifest["roles"][own]["files"]
        value["tls"] = {
            "ca_file": str(self.pki / files["ca_certificate"]),
            "cert_file": str(self.pki / files["certificate"]),
            "key_file": str(self.pki / files["private_key"]),
            "peer_cert_sha256": self.manifest["roles"][peer]["certificate_sha256"],
            "server_name": self.manifest["roles"][peer]["server_name"],
        }
        return transport.validate_config(value, role)

    def exchange(self, client=None, worker=None, change_context=None):
        client = client or self.configured("client")
        worker = worker or self.configured("worker")
        client_context = transport.tls_context(client, "client")
        worker_context = transport.tls_context(worker, "worker")
        if change_context is not None:
            change_context(client_context, worker_context)
        self.assertEqual(worker_context.verify_mode, ssl.CERT_REQUIRED)
        self.assertEqual(client_context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(client_context.check_hostname)
        left, right = socket.socketpair()
        left.setblocking(False)
        right.setblocking(False)
        opened = []
        errors = {}
        completed = {}

        class TrackingContext:
            def __init__(self, context):
                self.context = context

            def wrap_socket(self, *args, **kwargs):
                result = self.context.wrap_socket(*args, **kwargs)
                opened.append(result)
                return result

        deadline = time.monotonic() + 2

        def server():
            try:
                with transport.tls_handshake(right, TrackingContext(worker_context), worker,
                                             "worker", deadline) as connection:
                    completed["worker_handshake"] = (connection.version(), connection.selected_alpn_protocol())
                    transport.write_all(connection, b"S", deadline)
                    if transport.read_exact(connection, 1, deadline) != b"C":
                        raise AssertionError("wrong authenticated application byte")
                    transport.tls_close(connection, deadline)
                    completed["worker_clean_close"] = True
            except Exception as error:
                errors["worker"] = error

        thread = threading.Thread(target=server)
        thread.start()
        try:
            try:
                with transport.tls_handshake(left, TrackingContext(client_context), client,
                                             "client", deadline) as connection:
                    completed["client_handshake"] = (connection.version(), connection.selected_alpn_protocol())
                    if transport.read_exact(connection, 1, deadline) != b"S":
                        raise AssertionError("wrong authenticated application byte")
                    transport.write_all(connection, b"C", deadline)
                    transport.tls_close(connection, deadline)
                    completed["client_clean_close"] = True
            except Exception as error:
                errors["client"] = error
            thread.join(3)
        finally:
            left.close()
            right.close()
            for connection in opened:
                connection.close()
            thread.join(3)
        self.assertFalse(thread.is_alive(), "owned TLS handshake thread did not terminate")
        return completed, errors

    def test_real_tls13_mutual_authentication_and_clean_close(self):
        completed, errors = self.exchange()
        self.assertEqual(errors, {})
        self.assertEqual(completed, {
            "worker_handshake": ("TLSv1.3", transport.ALPN),
            "client_handshake": ("TLSv1.3", transport.ALPN),
            "worker_clean_close": True, "client_clean_close": True,
        })

    def test_real_tls_rejects_untrusted_ca_in_both_directions(self):
        for role in ("client", "worker"):
            value = self.configured(role)
            value["tls"]["ca_file"] = str(self.other_pki / "runtime/driver/ca.crt")
            completed, errors = self.exchange(**{role: value})
            with self.subTest(role=role):
                self.assertIsInstance(errors.get(role), ssl.SSLCertVerificationError)
                self.assertNotIn(f"{role}_handshake", completed)
                self.assertNotIn("worker_clean_close", completed)

    def test_real_tls_rejects_wrong_der_pin_in_both_directions(self):
        for role in ("client", "worker"):
            value = self.configured(role)
            value["tls"]["peer_cert_sha256"] = "f" * 64
            completed, errors = self.exchange(**{role: value})
            with self.subTest(role=role):
                self.assertIsInstance(errors.get(role), transport.TransportError)
                self.assertIn("pin mismatch", str(errors[role]))
                self.assertNotIn(f"{role}_handshake", completed)

    def test_real_tls_rejects_wrong_certificate_purpose(self):
        for role, replacement, rejecting_role in (("client", "worker-1", "worker"),
                                                   ("worker", "driver", "client")):
            value = self.configured(role)
            files = self.manifest["roles"][replacement]["files"]
            value["tls"]["cert_file"] = str(self.pki / files["certificate"])
            value["tls"]["key_file"] = str(self.pki / files["private_key"])
            completed, errors = self.exchange(**{role: value})
            with self.subTest(role=role):
                self.assertIsInstance(errors.get(rejecting_role), ssl.SSLCertVerificationError)
                self.assertIn("purpose", str(errors[rejecting_role]).lower())
                self.assertNotIn(f"{rejecting_role}_handshake", completed)

    def test_real_tls_rejects_wrong_worker_hostname_and_driver_run_san(self):
        client = self.configured("client")
        client["tls"]["server_name"] = self.manifest["roles"]["worker-1"]["server_name"]
        completed, errors = self.exchange(client=client)
        self.assertIsInstance(errors.get("client"), ssl.SSLCertVerificationError)
        self.assertNotIn("client_handshake", completed)
        for role, peer in (("client", "worker-0"), ("worker", "driver")):
            value = self.configured(role)
            value["run_id"] = self.other_manifest["run_id"]
            value["tls"]["server_name"] = f"{peer}.{value['run_id']}.bml.invalid"
            value = transport.validate_config(value, role)
            completed, errors = self.exchange(**{role: value})
            with self.subTest(role=role):
                expected = ssl.SSLCertVerificationError if role == "client" else transport.TransportError
                self.assertIsInstance(errors.get(role), expected)
                self.assertNotIn(f"{role}_handshake", completed)

    def test_real_tls_rejects_missing_alpn_and_tls12(self):
        def no_matching_alpn(client, _worker):
            client.set_alpn_protocols(["another-protocol"])

        completed, errors = self.exchange(change_context=no_matching_alpn)
        self.assertTrue(any(isinstance(error, transport.TransportError) and "ALPN" in str(error)
                            for error in errors.values()))
        self.assertNotIn("worker_handshake", completed)
        self.assertNotIn("client_handshake", completed)

        def tls12_only(client, _worker):
            client.minimum_version = ssl.TLSVersion.TLSv1_2
            client.maximum_version = ssl.TLSVersion.TLSv1_2

        completed, errors = self.exchange(change_context=tls12_only)
        self.assertTrue(any(isinstance(error, ssl.SSLError) for error in errors.values()))
        self.assertEqual(completed, {})

    def test_real_tls_rejects_plaintext_without_fallback(self):
        value = self.configured("worker")
        context = transport.tls_context(value, "worker")
        with contextlib.ExitStack() as stack:
            left, right = socket.socketpair()
            stack.enter_context(left)
            stack.enter_context(right)
            left.sendall(b"GET /plaintext HTTP/1.0\r\n\r\n")
            left.shutdown(socket.SHUT_WR)
            right.setblocking(False)
            with self.assertRaises(ssl.SSLError):
                transport.tls_handshake(right, context, value, "worker", time.monotonic() + 2)


if __name__ == "__main__":
    unittest.main()
