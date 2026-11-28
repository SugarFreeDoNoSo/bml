"""Offline contract tests for role-separated ephemeral fragment credentials."""

import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import ssl
import stat
import subprocess
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

import fragment_run_credentials as credentials


OPENSSL = shutil.which("openssl")


@unittest.skipUnless(OPENSSL, "requires already-installed OpenSSL; no installation or network")
class CredentialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.output = cls.root / "complete"
        cls.manifest = credentials.create(cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def openssl(self, *arguments, success=True, input_data=None):
        result = subprocess.run([OPENSSL, *map(str, arguments)], input=input_data,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                timeout=10, check=False, umask=0o077)
        if success:
            self.assertEqual(result.returncode, 0, "OpenSSL public-certificate inspection failed")
        else:
            self.assertNotEqual(result.returncode, 0)
        return result.stdout.decode("ascii")

    def test_manifest_is_public_and_has_exact_role_layout(self):
        self.assertEqual(self.manifest["format"], credentials.FORMAT)
        self.assertRegex(self.manifest["run_id"], r"^[0-9a-f]{32}$")
        self.assertTrue(self.manifest["openssl_version"].startswith("OpenSSL 3."))
        self.assertEqual(set(self.manifest["roles"]), set(credentials.ROLES))
        serialized = json.dumps(self.manifest)
        for forbidden in ("BEGIN", "private_key_sha256", "private_key_bytes", str(self.root)):
            self.assertNotIn(forbidden, serialized)
        self.assertEqual(json.loads((self.output / "manifest.json").read_text()), self.manifest)
        ca_bytes = (self.output / "issuer/ca.crt").read_bytes()
        for role, metadata in self.manifest["roles"].items():
            self.assertEqual(metadata["server_name"],
                             f"{role}.{self.manifest['run_id']}.bml.invalid")
            expected = {"ca_certificate": f"runtime/{role}/ca.crt",
                        "certificate": f"runtime/{role}/cert.crt",
                        "private_key": f"runtime/{role}/key.key"}
            self.assertEqual(metadata["files"], expected)
            self.assertEqual({p.name for p in (self.output / "runtime" / role).iterdir()},
                             {"ca.crt", "cert.crt", "key.key"})
            self.assertEqual((self.output / metadata["files"]["ca_certificate"]).read_bytes(), ca_bytes)
        self.assertTrue((self.output / "issuer/ca.key").is_file())
        self.assertEqual(len(list((self.output / "runtime").rglob("*.key"))), 3)
        self.assertFalse(any(p.name == "ca.key" for p in (self.output / "runtime").rglob("*")))

    def test_every_directory_and_file_is_owner_only(self):
        for path in [self.output, *self.output.rglob("*")]:
            mode = path.lstat().st_mode
            self.assertFalse(stat.S_ISLNK(mode))
            self.assertEqual(stat.S_IMODE(mode), 0o700 if path.is_dir() else 0o600)

    def test_der_hashes_and_distinct_public_keys_are_authentic(self):
        fingerprints = set()
        public_keys = set()
        for role, metadata in self.manifest["roles"].items():
            certificate = self.output / metadata["files"]["certificate"]
            der = ssl.PEM_cert_to_DER_cert(certificate.read_text())
            self.assertEqual(hashlib.sha256(der).hexdigest(), metadata["certificate_sha256"])
            fingerprints.add(metadata["certificate_sha256"])
            public_key = self.openssl("x509", "-in", certificate, "-pubkey", "-noout")
            public_keys.add(public_key)
            description = self.openssl("pkey", "-pubin", "-text", "-noout",
                                       input_data=public_key.encode())
            self.assertIn("ASN1 OID: prime256v1", description)
            # Loading the actual private key through TLS checks it matches the
            # certificate, without reading or fingerprinting private key bytes.
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            context.load_cert_chain(certificate, self.output / metadata["files"]["private_key"])
        self.assertEqual(len(fingerprints), 3)
        self.assertEqual(len(public_keys), 3)
        ca_der = ssl.PEM_cert_to_DER_cert((self.output / "issuer/ca.crt").read_text())
        self.assertEqual(hashlib.sha256(ca_der).hexdigest(), self.manifest["ca_certificate_sha256"])

    def test_ca_constraints_and_role_eku_are_strict(self):
        ca = self.output / "issuer/ca.crt"
        ca_text = self.openssl("x509", "-in", ca, "-text", "-noout")
        self.assertIn("X509v3 Basic Constraints: critical", ca_text)
        self.assertIn("CA:TRUE, pathlen:0", ca_text)
        self.assertIn("X509v3 Key Usage: critical", ca_text)
        self.assertIn("Certificate Sign, CRL Sign", ca_text)
        self.assertIn("ecdsa-with-SHA256", ca_text)
        for role, metadata in self.manifest["roles"].items():
            cert = self.output / metadata["files"]["certificate"]
            purpose = "sslclient" if role == "driver" else "sslserver"
            self.openssl("verify", "-CAfile", ca, "-purpose", purpose,
                         "-verify_hostname", metadata["server_name"], cert)
            wrong = "sslserver" if role == "driver" else "sslclient"
            self.openssl("verify", "-CAfile", ca, "-purpose", wrong, cert, success=False)
            self.openssl("verify", "-CAfile", ca, "-verify_hostname",
                         f"other.{self.manifest['run_id']}.bml.invalid", cert, success=False)
            text = self.openssl("x509", "-in", cert, "-text", "-noout")
            self.assertIn("X509v3 Basic Constraints: critical", text)
            self.assertIn("CA:FALSE", text)
            self.assertIn("X509v3 Key Usage: critical", text)
            self.assertIn("Digital Signature", text)
            self.assertIn("ecdsa-with-SHA256", text)
            san = self.openssl("x509", "-in", cert, "-ext", "subjectAltName", "-noout")
            self.assertEqual(re.findall(r"DNS:([^\s,]+)", san), [metadata["server_name"]])

    def test_all_certificates_expire_after_one_day(self):
        certs = [self.output / "issuer/ca.crt"]
        certs.extend(self.output / item["files"]["certificate"]
                     for item in self.manifest["roles"].values())
        for cert in certs:
            dates = self.openssl("x509", "-in", cert, "-dates", "-noout")
            fields = dict(line.split("=", 1) for line in dates.splitlines())
            start = ssl.cert_time_to_seconds(fields["notBefore"])
            end = ssl.cert_time_to_seconds(fields["notAfter"])
            self.assertEqual(end - start, 86400)
            self.assertLessEqual(abs(start - time.time()), 120)

    def test_existing_output_is_not_modified(self):
        before = (self.output / "manifest.json").read_bytes()
        with self.assertRaisesRegex(credentials.CredentialError, "already exists"):
            credentials.create(self.output)
        self.assertEqual((self.output / "manifest.json").read_bytes(), before)

    def test_permissive_parent_umask_does_not_expose_files_or_accept_another_ca(self):
        output = self.root / "permissive-umask"
        old_umask = os.umask(0)
        try:
            other = credentials.create(output)
        finally:
            os.umask(old_umask)
        credentials._check_private_tree(output)
        self.assertNotEqual(other["run_id"], self.manifest["run_id"])
        self.assertNotEqual(other["ca_certificate_sha256"], self.manifest["ca_certificate_sha256"])
        certificate = self.output / self.manifest["roles"]["worker-0"]["files"]["certificate"]
        self.openssl("verify", "-CAfile", output / "issuer/ca.crt", certificate, success=False)

    def test_openssl_failure_preserves_private_state_and_redacts_child_output(self):
        output = self.root / "failed"
        original = subprocess.run
        marker = b"PRIVATE-MATERIAL-MUST-NOT-ESCAPE"

        def fail_leaf_sign(command, **kwargs):
            if "-CAkey" in command:
                return subprocess.CompletedProcess(command, 7, stdout=marker, stderr=marker)
            return original(command, **kwargs)

        with mock.patch.object(credentials.subprocess, "run", side_effect=fail_leaf_sign):
            with self.assertRaisesRegex(credentials.CredentialError, r"sign driver certificate \(exit 7\)") as caught:
                credentials.create(output)
        self.assertNotIn(marker.decode(), str(caught.exception))
        self.assertTrue((output / "issuer/ca.key").is_file())
        self.assertTrue((output / "runtime/driver/key.key").is_file())
        self.assertFalse((output / "manifest.json").exists())
        failure = (output / "creation-failed.json").read_text()
        self.assertNotIn(marker.decode(), failure)
        self.assertEqual(json.loads(failure)["status"], "incomplete")
        credentials._check_private_tree(output)

    def test_cli_emits_only_public_manifest(self):
        output = self.root / "cli"
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = credentials.main(["--output", str(output)])
        self.assertEqual(status, 0)
        self.assertEqual(stderr.getvalue(), "")
        result = json.loads(stdout.getvalue())
        self.assertEqual(result, json.loads((output / "manifest.json").read_text()))
        self.assertNotEqual(result["run_id"], self.manifest["run_id"])
        self.assertNotEqual(result["ca_certificate_sha256"], self.manifest["ca_certificate_sha256"])
        self.assertNotIn("BEGIN", stdout.getvalue())
        self.assertNotIn(str(output), stdout.getvalue())


class AdmissionTests(unittest.TestCase):
    def test_relative_parent_traversal_and_symlink_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.mkdir()
            link = root / "link"
            link.symlink_to(target, target_is_directory=True)
            for output in [Path("relative"), root / ".." / "forbidden", link / "new"]:
                with self.subTest(output=str(output)), self.assertRaises(credentials.CredentialError):
                    credentials.create(output)
            self.assertEqual(list(target.iterdir()), [])

    def test_missing_openssl_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "missing"
            with mock.patch.object(credentials.shutil, "which", return_value=None):
                with self.assertRaisesRegex(credentials.CredentialError, "installed OpenSSL"):
                    credentials.create(output)
            self.assertFalse(output.exists())

    def test_timeout_does_not_expose_subprocess_output(self):
        with mock.patch.object(credentials.subprocess, "run",
                               side_effect=subprocess.TimeoutExpired("openssl", 10,
                                   output=b"PRIVATE", stderr=b"PRIVATE")):
            with self.assertRaisesRegex(credentials.CredentialError, "timed out") as caught:
                credentials._run("openssl", ["version"], "version check", time.monotonic() + 30)
        self.assertNotIn("PRIVATE", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
