"""Private validator controls: no network, real TLS session or fake ELF execution."""

import argparse
import contextlib
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import ssl
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from test_validate_fragment_elf import SOURCE, bundle, outputs
import fragment_private_transport as transport
import validate_fragment_elf as local
import validate_fragment_private as private


RUN = "1" * 32
WORKER_IP = "172.31.3.94"
DRIVER_IP = "172.26.91.250"


def credentials(root):
    """Identity-only dummy PEM files; never supplied to SSL or a real child."""
    root.mkdir(mode=0o700)
    ca = b"dummy certificate identity for unit tests"
    manifest = {"format": "bml-fragment-run-credentials-v1", "run_id": RUN,
                "openssl_version": "unit-only, not an issuer execution",
                "ca_certificate_sha256": hashlib.sha256(ca).hexdigest(), "roles": {}}
    for role in private.ROLE_NAMES:
        directory = root / "runtime" / role
        directory.mkdir(mode=0o700, parents=True)
        certificate = ("dummy unit certificate " + role).encode()
        for name, data in (("ca.crt", ssl.DER_cert_to_PEM_cert(ca).encode()),
                           ("cert.crt", ssl.DER_cert_to_PEM_cert(certificate).encode()),
                           ("key.key", b"unit dummy key, never loaded by SSL")):
            path = directory / name
            path.write_bytes(data)
            path.chmod(0o600)
        manifest["roles"][role] = {
            "certificate_sha256": hashlib.sha256(certificate).hexdigest(),
            "server_name": f"{role}.{RUN}.bml.invalid",
            "files": {key: f"runtime/{role}/{name}" for key, name in private.TLS_NAMES.items()}}
    (root / "manifest.json").write_text(json.dumps(manifest))
    # An issuer sentinel proves staging copies only the enumerated role files.
    (root / "issuer").mkdir(mode=0o700)
    (root / "issuer" / "ca.key").write_text("unit issuer must never be staged")
    return manifest


def prepare(root):
    source = root / "bundle"
    source.mkdir()
    report = bundle(source)
    credential_dir = root / "credentials"
    manifest = credentials(credential_dir)
    case, reference, mono, fragmented = outputs()
    fragmented["mode"] = "tcp-test"
    refs = root / "references.json"
    refs.write_text(json.dumps({"format_version": 1, "source_sha256": SOURCE,
                               "cases": [{"request": case, "result": reference},
                                         {"request": {**case, "name": "second"}, "result": reference}]}))
    args = argparse.Namespace(bundle=source, build_report_sha256=local.sha256(source / "build-report.json"),
                              reference_json=refs, reference_sha256=local.sha256(refs),
                              credentials_dir=credential_dir, worker_ip=WORKER_IP, driver_ip=DRIVER_IP,
                              output=root / "validation", timeout=240,
                              trusted_generated_artifacts=True, test_ephemeral_ports=True)
    return args, report, manifest, mono, fragmented


def ready(role, ip, port=12345, run_id=RUN):
    return (json.dumps({"format": f"bml-private-fragment-{role}-v1",
                       "address": f"{ip}:{port}", "run_id": run_id}) + "\n").encode()


class ConfigurationTests(unittest.TestCase):
    def test_readiness_exact_role_run_address_and_single_line(self):
        self.assertEqual(private.parse_ready(ready("worker", WORKER_IP), "worker", RUN, WORKER_IP)["address"],
                         f"{WORKER_IP}:12345")
        for raw in [ready("client", WORKER_IP), ready("worker", DRIVER_IP),
                    ready("worker", WORKER_IP, 0), ready("worker", WORKER_IP, 65536),
                    ready("worker", WORKER_IP, run_id="2" * 32), b"{}\n",
                    ready("worker", WORKER_IP) + b"extra\n", b"x" * 16385]:
            with self.subTest(raw=raw[:100]), self.assertRaises(ValueError):
                private.parse_ready(raw, "worker", RUN, WORKER_IP)

    def test_exact_bindings_and_public_endpoint_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            args, report, manifest, _, _ = prepare(Path(directory))
            _, identities = local.verify_bundle(args.bundle, args.build_report_sha256)
            roles = {name: Path(directory) / name for name in private.ROLE_NAMES}
            worker = private.configuration("worker", 1, report, identities, roles, manifest,
                                           WORKER_IP, DRIVER_IP, bootstrap=True)
            self.assertEqual(worker["allowed_peer_ips"], [])
            self.assertTrue(worker["bootstrap_peer_from_certificate"])
            self.assertEqual(worker["tls"]["peer_cert_sha256"], manifest["roles"]["driver"]["certificate_sha256"])
            client = private.configuration("client", 1, report, identities, roles, manifest,
                                           WORKER_IP, DRIVER_IP, port=12345)
            self.assertEqual(client["source_ip"], DRIVER_IP)
            self.assertEqual(client["tls"]["server_name"], f"worker-1.{RUN}.bml.invalid")
            self.assertEqual(client["descriptor"], report["fragments"][1]["descriptor"])
            for field, replacement in [("descriptor", report["fragments"][0]["descriptor"]),
                                       ("tls", {**client["tls"], "peer_cert_sha256": manifest["roles"]["worker-0"]["certificate_sha256"]})]:
                bad = deepcopy(client)
                bad[field] = replacement
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "pinned"):
                    private.verify_bound_config(bad, client, "client")
            for role, config, key in [("worker", worker, "bind_ip"), ("client", client, "peer_ip"),
                                      ("client", client, "source_ip")]:
                bad = {**config, key: "8.8.8.8"}
                with self.subTest(role=role, key=key), self.assertRaisesRegex(ValueError, "RFC1918"):
                    transport.validate_config(bad, role, test_ephemeral_port=role == "worker")
            with self.assertRaisesRegex(ValueError, "private port"):
                transport.validate_config(worker, "worker")
            self.assertEqual(private.verify_bound_config(client, client, "client"), client)

    def test_bootstrap_stderr_is_one_exact_public_audit_event(self):
        raw = (json.dumps({"format": "bml-private-fragment-authenticated-peer-v1",
                           "run_id": RUN, "peer_ip": DRIVER_IP}) + "\n").encode()
        self.assertEqual(private.inspect_stderr(raw, True, RUN, DRIVER_IP)["peer_ip"], DRIVER_IP)
        for data in (b"", raw + raw, b"private transport error: failure\n", raw.replace(DRIVER_IP.encode(), WORKER_IP.encode())):
            with self.subTest(data=data[:100]), self.assertRaises(ValueError):
                private.inspect_stderr(data, True, RUN, DRIVER_IP)
        with self.assertRaises(ValueError):
            private.inspect_stderr(raw, False, RUN, DRIVER_IP)


class CredentialTests(unittest.TestCase):
    def test_copy_only_role_credentials_with_private_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, _, _, _, _ = prepare(root)
            _, identities = local.verify_bundle(args.bundle, args.build_report_sha256)
            manifest, paths = private.verify_credentials(args.credentials_dir)
            args.output.mkdir()
            roles = private.stage_roles(args.bundle, args.output, identities, paths, manifest)
            self.assertEqual(set(roles), set(private.ROLE_NAMES))
            for role, executable in (("driver", "coordinator"), ("worker-0", "fragment_0"), ("worker-1", "fragment_1")):
                self.assertEqual({p.name for p in roles[role].iterdir()}, {executable, "tls"})
                self.assertEqual({p.name for p in (roles[role] / "tls").iterdir()}, {"ca.crt", "cert.crt", "key.key"})
                self.assertEqual((roles[role] / "tls/key.key").stat().st_mode & 0o777, 0o600)
                self.assertEqual(private.certificate_digest(roles[role] / "tls/cert.crt"), manifest["roles"][role]["certificate_sha256"])
            self.assertFalse(any(p.name == "issuer" for p in args.output.rglob("*")))

    def test_swapped_pin_foreign_role_path_and_public_key_mode_rejected(self):
        for mutation in ("pin", "path", "permissions", "symlink"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "credentials"
                manifest = credentials(path)
                if mutation == "pin":
                    manifest["roles"]["worker-0"]["certificate_sha256"] = "0" * 64
                elif mutation == "path":
                    manifest["roles"]["worker-0"]["files"]["private_key"] = "issuer/ca.key"
                elif mutation == "permissions":
                    (path / "runtime/driver/key.key").chmod(0o644)
                else:
                    key = path / "runtime/driver/key.key"
                    key.unlink()
                    key.symlink_to(path / "issuer/ca.key")
                (path / "manifest.json").write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    private.verify_credentials(path)


class OwnershipTests(unittest.TestCase):
    def test_cleanup_targets_only_the_owned_session(self):
        process = MagicMock()
        process.pid = 12345
        process.poll.return_value = None
        process.returncode = -9
        process.stdout = io.BytesIO()
        owned = private.OwnedProcess(process, io.BytesIO(), ["owned"], "test")
        def killpg(pid, sig):
            self.assertEqual(pid, 12345)
            if sig == 0:
                raise ProcessLookupError()
        with patch.object(os, "killpg", side_effect=killpg) as kill:
            result = owned.close(False, private.Deadline(2), {"run_id": RUN}, DRIVER_IP)
            self.assertEqual(kill.call_args_list[0].args, (12345, signal.SIGKILL))
            self.assertTrue(result["cleanup_confirmed"])
            self.assertTrue(result["killed_by_verifier"])
            self.assertIsNone(owned.close(False, private.Deadline(2), {"run_id": RUN}, DRIVER_IP))
        process.wait.assert_called_once()

    def test_positive_exit_requires_no_extra_output_and_no_lingering_group(self):
        for extra, lingering in [(b"extra", False), (b"", True)]:
            with self.subTest(extra=extra, lingering=lingering):
                process = MagicMock()
                process.pid = 12345
                process.poll.return_value = 0
                process.returncode = 0
                process.stdout = io.BytesIO(extra)
                def killpg(_pid, _sig):
                    if not lingering:
                        raise ProcessLookupError()
                owned = private.OwnedProcess(process, io.BytesIO(), ["owned"], "test")
                with patch.object(os, "killpg", side_effect=killpg):
                    result = owned.close(True, private.Deadline(2), {"run_id": RUN}, DRIVER_IP)
                self.assertIn("error", result)

    def test_shared_deadline_bounds(self):
        for seconds in (0, 361, True, 1.5):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                private.Deadline(seconds)
        deadline = private.Deadline(240)
        deadline.end = 0
        with self.assertRaisesRegex(RuntimeError, "shared deadline"):
            deadline.remaining()


class OrchestrationTests(unittest.TestCase):
    def test_two_plans_two_cases_preserve_tcp_outputs_and_all_bits(self):
        with tempfile.TemporaryDirectory() as directory:
            args, report, _, mono, fragmented = prepare(Path(directory))
            created = []
            class Process:
                def __init__(self, value, bootstrap):
                    self.ready, self.bootstrap, self.closed = value, bootstrap, False
                def close(self, success, _deadline, _credentials, _driver):
                    self.closed = True
                    return {"cleanup_confirmed": True, "exit_code": 0 if success else -9}
            def start(command, cwd, label, deadline, parse, processes, bootstrap=False):
                self.assertTrue(cwd.is_dir())
                if "--config" in command:
                    config_path = Path(command[command.index("--config") + 1])
                    config = local.read_json(config_path)
                    role = command[2]
                    ip = WORKER_IP if role == "worker" else "127.0.0.1"
                    value = parse(ready(role, ip, 32000 + len(created), config["run_id"]))
                else:
                    value = parse((json.dumps({"format": "bml-native-fragment-tcp-listener-v1",
                                               "address": f"127.0.0.1:{32000 + len(created)}",
                                               "descriptor": report["fragments"][0]["descriptor"]}) + "\n").encode())
                item = Process(value, bootstrap)
                processes.append(item)
                created.append(item)
                return item
            def invoke(command, cwd, timeout):
                self.assertGreater(timeout, 0)
                self.assertLessEqual(timeout, 120)
                value = fragmented if "tcp-test" in command else mono
                return 0, json.dumps(value).encode(), b""
            with patch.object(private.OwnedProcess, "start", side_effect=start), \
                    patch.object(local, "invoke", side_effect=invoke):
                summary = private.validate(args)
            self.assertEqual(summary["status"], "passed", summary)
            self.assertEqual(len(summary["cases"]), 4)
            self.assertEqual([c["plan"] for c in summary["cases"]], ["A", "B", "A", "B"])
            self.assertTrue(all(c["bit_identical"] and c["logits_compared"] == 12 for c in summary["cases"]))
            self.assertEqual(len(created), 14)
            self.assertTrue(all(p.closed for p in created))
            self.assertEqual(sum(p.bootstrap for p in created), 2)
            self.assertEqual(local.read_json(args.output / "A-case-tcp-test.json")["mode"], "tcp-test")
            self.assertFalse(summary["distinct_hosts_verified"])
            self.assertFalse(summary["network_namespaces_verified"])
            text = json.dumps(summary)
            for forbidden in ("PRIVATE KEY", "private_key", "key.key", "issuer"):
                self.assertNotIn(forbidden, text)

    def test_public_ip_missing_flags_or_wrong_input_hash_never_executes(self):
        for mutation in ("public_worker", "public_driver", "trust", "ephemeral", "hash"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                args, _, _, _, _ = prepare(Path(directory))
                if mutation == "public_worker":
                    args.worker_ip = "8.8.8.8"
                elif mutation == "public_driver":
                    args.driver_ip = "8.8.8.8"
                elif mutation == "trust":
                    args.trusted_generated_artifacts = False
                elif mutation == "ephemeral":
                    args.test_ephemeral_ports = False
                else:
                    args.reference_sha256 = "0" * 64
                with patch.object(subprocess, "Popen") as process:
                    summary = private.validate(args)
                process.assert_not_called()
                self.assertEqual(summary["status"], "failed")
                self.assertEqual(local.read_json(args.output / "report.json")["status"], "failed")

    def test_changed_prefill_logit_cannot_pass_and_outputs_remain_original(self):
        case, reference, mono, fragmented = outputs()
        fragmented["mode"] = "tcp-test"
        original = deepcopy(fragmented)
        fragmented["all_logits_bits"][0][0] = 1
        with self.assertRaisesRegex(ValueError, "position=0 logit=0"):
            private.tcp.checked_tcp_parity(reference, mono, fragmented, case, 4, SOURCE)
        self.assertEqual(fragmented["mode"], original["mode"])

    def test_command_line_requires_both_explicit_execution_acknowledgments(self):
        arguments = ["--bundle", "bundle", "--build-report-sha256", "a" * 64,
                     "--reference-json", "reference", "--reference-sha256", "b" * 64,
                     "--credentials-dir", "credentials", "--worker-ip", WORKER_IP,
                     "--driver-ip", DRIVER_IP, "--output", "new"]
        for flags in ([], ["--trusted-generated-artifacts"], ["--test-ephemeral-ports"]):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                private.parser().parse_args(arguments + flags)
        parsed = private.parser().parse_args(arguments + ["--trusted-generated-artifacts", "--test-ephemeral-ports"])
        self.assertEqual(parsed.timeout, 240)


if __name__ == "__main__":
    unittest.main()
