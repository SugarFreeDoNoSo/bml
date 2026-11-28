"""Deterministic TCP-verifier tests; fake ELF bytes are never executed."""

import argparse
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from test_validate_fragment_elf import SOURCE, bundle, outputs
import validate_fragment_elf as local
import validate_fragment_tcp as tcp


def announcement(descriptor, address="127.0.0.1:32123"):
    return (json.dumps({"format": "bml-native-fragment-tcp-listener-v1", "address": address,
                       "descriptor": descriptor}) + "\n").encode()


class ListenerTests(unittest.TestCase):
    descriptor = {"version": 1, "source_sha256": SOURCE, "row_start": 0, "row_end": 2}

    def test_exact_descriptor_and_literal_loopback_address(self):
        self.assertEqual(tcp.parse_listener(announcement(self.descriptor), self.descriptor)["address"],
                         "127.0.0.1:32123")

    def test_incompatible_address_identity_shape_and_extra_output_rejected(self):
        records = [announcement(self.descriptor, x) for x in
                   ["0.0.0.0:32123", "192.0.2.1:32123", "localhost:32123", "127.0.0.1:0",
                    "127.0.0.1:65536", "127.0.0.1:032123", "[::1]:32123"]]
        records += [announcement({**self.descriptor, "version": True}),
                    announcement({**self.descriptor, "row_start": 1}),
                    announcement(self.descriptor) + b"another line\n",
                    b"x" * (tcp.MAX_LISTENER_BYTES + 1), b"{}\n", b"{}"]
        for raw in records:
            with self.subTest(raw=raw[:100]), self.assertRaises(ValueError):
                tcp.parse_listener(raw, self.descriptor)

    def test_pipe_read_eof_and_expired_deadline_fail_closed(self):
        read_fd, write_fd = os.pipe()
        with os.fdopen(read_fd, "rb", buffering=0) as stream:
            raw = announcement(self.descriptor)
            os.write(write_fd, raw)
            os.close(write_fd)
            self.assertEqual(tcp.read_listener_line(stream, tcp.Deadline(1)), raw)
        read_fd, write_fd = os.pipe()
        os.write(write_fd, b"partial")
        os.close(write_fd)
        with os.fdopen(read_fd, "rb", buffering=0) as stream, self.assertRaisesRegex(ValueError, "complete announcement"):
            tcp.read_listener_line(stream, tcp.Deadline(1))
        deadline = tcp.Deadline(1)
        deadline.end = 0
        with self.assertRaisesRegex(RuntimeError, "shared deadline"):
            deadline.remaining()
        for value in [0, 121, True, 1.5]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                tcp.Deadline(value)


class ParityTests(unittest.TestCase):
    def test_tcp_mode_preserved_while_comparing_all_prefill_and_decode(self):
        case, reference, mono, fragmented = outputs()
        fragmented["mode"] = "tcp-test"
        saved = deepcopy(fragmented)
        result = tcp.checked_tcp_parity(reference, mono, fragmented, case, 4, SOURCE)
        self.assertEqual(result["logits_compared"], 12)
        self.assertEqual(result["mode"], "tcp-test")
        self.assertEqual(fragmented, saved)
        fragmented["all_logits_bits"][0][0] = 1
        with self.assertRaisesRegex(ValueError, "position=0 logit=0"):
            tcp.checked_tcp_parity(reference, mono, fragmented, case, 4, SOURCE)

    def test_wrong_mode_or_model_cannot_pass_as_tcp(self):
        case, reference, mono, fragmented = outputs()
        with self.assertRaisesRegex(ValueError, "tcp-test"):
            tcp.checked_tcp_parity(reference, mono, fragmented, case, 4, SOURCE)
        fragmented.update(mode="tcp-test", source_sha256="b" * 64)
        with self.assertRaisesRegex(ValueError, "source identity"):
            tcp.checked_tcp_parity(reference, mono, fragmented, case, 4, SOURCE)


class OwnershipTests(unittest.TestCase):
    def test_cleanup_kills_only_its_owned_handle(self):
        process = MagicMock()
        process.pid = 12345
        process.poll.return_value = None
        process.returncode = -9
        process.stdout = io.BytesIO()
        worker = tcp.OwnedWorker(process, io.BytesIO(), ["owned-worker"])
        result = worker.close(False, tcp.Deadline(1))
        process.kill.assert_called_once_with()
        process.wait.assert_called_once()
        self.assertGreater(process.wait.call_args.kwargs["timeout"], 0)
        self.assertLessEqual(process.wait.call_args.kwargs["timeout"], 1)
        self.assertTrue(result["killed_by_verifier"])
        self.assertIsNone(worker.close(False, tcp.Deadline(1)))
        process.kill.assert_called_once()

    def test_positive_worker_requires_zero_exit_and_no_extra_output(self):
        for returncode, stdout, stderr in [(1, b"", b""), (0, b"extra", b""), (0, b"", b"error")]:
            with self.subTest(returncode=returncode, stdout=stdout, stderr=stderr):
                process = MagicMock()
                process.pid = 12345
                process.returncode = returncode
                process.poll.return_value = returncode
                process.stdout = io.BytesIO(stdout)
                worker = tcp.OwnedWorker(process, io.BytesIO(stderr), ["owned-worker"])
                evidence = []
                with self.assertRaises(ValueError):
                    tcp.stop_workers([worker], True, tcp.Deadline(1), evidence)
                self.assertEqual(evidence[0]["pid"], 12345)
                self.assertEqual(evidence[0]["exit_code"], returncode)
                self.assertEqual(evidence[0]["stderr"], stderr.decode())
                self.assertTrue(evidence[0]["cleanup_confirmed"])
                process.kill.assert_not_called()

    def test_inherited_open_stdout_is_nonblocking_failure(self):
        read_fd, write_fd = os.pipe()
        try:
            with os.fdopen(read_fd, "rb", buffering=0) as stream:
                with self.assertRaisesRegex(ValueError, "remained open"):
                    tcp.available_stdout(stream)
        finally:
            os.close(write_fd)

    def test_partial_start_failure_cleans_the_already_started_worker(self):
        report = {"fragments": [{"file_name": "fragment_0", "descriptor": {}},
                                {"file_name": "fragment_1", "descriptor": {}}]}
        identities = {"fragment_0": {}, "fragment_1": {}}
        roles = {"fragment_0": Path("first"), "fragment_1": Path("second")}
        worker = MagicMock()
        with patch.object(tcp.OwnedWorker, "start", side_effect=[worker, ValueError("bad announcement")]):
            with self.assertRaisesRegex(ValueError, "bad announcement"):
                tcp.start_workers(report, identities, roles, tcp.Deadline(1))
        worker.close.assert_called_once()
        self.assertFalse(worker.close.call_args.args[0])


class FullVerifierTests(unittest.TestCase):
    def test_missing_reference_result_never_executes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle(root)
            case = outputs()[0]
            refs = root / "refs.json"
            refs.write_text(json.dumps({"format_version": 1, "source_sha256": SOURCE,
                                       "cases": [{"request": case},
                                                 {"request": {**case, "name": "second"}}]}))
            args = argparse.Namespace(bundle=root, build_report_sha256=local.sha256(root / "build-report.json"),
                                      reference_json=refs, reference_sha256=local.sha256(refs),
                                      output=root / "result", timeout=10)
            with patch.object(local, "invoke") as invoke, patch.object(tcp.OwnedWorker, "start") as start:
                with self.assertRaisesRegex(ValueError, "missing canonical run"):
                    tcp.validate(args)
            invoke.assert_not_called()
            start.assert_not_called()
            self.assertFalse(args.output.exists())

    def test_separate_roles_two_workers_and_reported_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle(root)
            case, reference, mono, fragmented = outputs()
            fragmented["mode"] = "tcp-test"
            refs = root / "refs.json"
            refs.write_text(json.dumps({"format_version": 1, "source_sha256": SOURCE,
                                       "cases": [{"request": case, "result": reference},
                                                 {"request": {**case, "name": "second"}, "result": reference}]}))
            args = argparse.Namespace(bundle=root, build_report_sha256=local.sha256(root / "build-report.json"),
                                      reference_json=refs, reference_sha256=local.sha256(refs),
                                      output=root / "result", timeout=10)
            created = []
            class Worker:
                def __init__(self, descriptor, index):
                    self.listener = {"address": f"127.0.0.1:{32000 + index}", "descriptor": descriptor}
                    self.closed = False
                def close(self, success, deadline):
                    self.closed = True
                    return {"exit_code": 0 if success else -9, "killed_by_verifier": not success}
            def start(executable, identity, descriptor, deadline):
                self.assertEqual([p.name for p in executable.parent.iterdir()], [executable.name])
                worker = Worker(descriptor, len(created))
                created.append(worker)
                return worker
            def invoke(command, cwd, timeout):
                self.assertEqual(cwd.name, "coordinator-only")
                self.assertEqual([p.name for p in cwd.iterdir()], ["coordinator"])
                value = fragmented if "tcp-test" in command else mono
                return 0, json.dumps(value).encode(), b""
            with patch.object(tcp.OwnedWorker, "start", side_effect=start), \
                    patch.object(local, "invoke", side_effect=invoke), \
                    patch.object(tcp, "failure_checks", return_value=[{"name": "fixture-only"}]):
                result = tcp.validate(args)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(len(created), 4)
            self.assertTrue(all(worker.closed for worker in created))
            self.assertEqual(len(result["workers"]), 4)
            self.assertFalse(result["distinct_hosts_verified"])
            self.assertFalse(result["cryptographic_peer_authentication"])
            self.assertEqual(json.loads((args.output / "case-tcp-test.json").read_text())["mode"], "tcp-test")

    def test_invalid_token_must_fail_without_any_connection(self):
        case = outputs()[0]
        roles = {"coordinator": Path("/unused")}
        def premature_connect(roles, arguments, deadline, summary, label):
            if label == "invalid-token-preflight":
                address = arguments[arguments.index("--fragment-endpoints") + 1].split(",")[0]
                host, port = address.split(":")
                with socket.create_connection((host, int(port)), timeout=1):
                    pass
                return 2, b"", b"token outside model vocabulary"
            if label == "unknown-mode":
                return 2, b"", b"invalid fragment execution mode"
            return 2, b"", b"unknown argument"
        with patch.object(tcp, "invoke_coordinator", side_effect=premature_connect):
            with self.assertRaisesRegex(ValueError, "preflight connected"):
                tcp.failure_checks({}, {}, roles, tcp.Deadline(2), {}, case)

    def test_all_negative_checks_use_owned_roles_and_leave_no_workers(self):
        case = outputs()[0]
        workers = []
        class Worker:
            def __init__(self, port):
                self.listener = {"address": f"127.0.0.1:{port}"}
                self.closed = False
            def close(self, success, deadline):
                self.closed = True
                return {"exit_code": -9, "killed_by_verifier": True}
        def start(report, identities, roles, deadline, evidence):
            pair = [Worker(32001), Worker(32002)]
            workers.extend(pair)
            return pair
        messages = {"unknown-argument": "unknown argument",
                    "unknown-mode": "invalid fragment execution mode",
                    "invalid-token-preflight": "token outside vocabulary",
                    "nonloopback": "requires a literal loopback endpoint",
                    "missing-listener": "TCP connection failed",
                    "swapped-workers": "Ready identity mismatch",
                    "duplicate-endpoint": "duplicates rejected",
                    "disconnect": "closed before Ready/output"}
        def invoke(roles, arguments, deadline, summary, label):
            if label == "disconnect":
                address = arguments[arguments.index("--fragment-endpoints") + 1].split(",")[0]
                host, port = address.split(":")
                with socket.create_connection((host, int(port)), timeout=1) as connection:
                    # A real coordinator observes EOF before returning. Wait
                    # for that same event so cleanup cannot overtake accept.
                    self.assertEqual(connection.recv(1), b"")
            return 2, b"", messages[label].encode()
        summary = {"negative_workers": [], "failure_checks": []}
        with patch.object(tcp, "start_workers", side_effect=start), \
                patch.object(tcp, "invoke_coordinator", side_effect=invoke):
            checks = tcp.failure_checks({}, {}, {}, tcp.Deadline(4), summary, case)
        self.assertEqual({check["name"] for check in checks}, set(messages))
        self.assertEqual(len(workers), 4)
        self.assertTrue(all(worker.closed for worker in workers))
        self.assertEqual(len(summary["negative_workers"]), 4)
        self.assertEqual(checks[2]["connections_accepted"], 0)
        # If the coordinator aborts before opening a connection, cancellation
        # must stop the owned accept thread, not leave a daemon blocked in accept.
        threads = []
        thread_class = tcp.threading.Thread
        def owned_thread(*args, **kwargs):
            thread = thread_class(*args, **kwargs)
            threads.append(thread)
            return thread
        def abort_before_connect(roles, arguments, deadline, summary, label):
            if label == "disconnect":
                raise RuntimeError("coordinator aborted before connect")
            return invoke(roles, arguments, deadline, summary, label)
        with patch.object(tcp, "start_workers", side_effect=start), \
                patch.object(tcp, "invoke_coordinator", side_effect=abort_before_connect), \
                patch.object(tcp.threading, "Thread", side_effect=owned_thread):
            with self.assertRaisesRegex(RuntimeError, "coordinator aborted before connect"):
                tcp.failure_checks({}, {}, {}, tcp.Deadline(4), {"negative_workers": []}, case)
        self.assertEqual(len(threads), 1)
        self.assertTrue(all(not thread.is_alive() for thread in threads))


if __name__ == "__main__":
    unittest.main()
