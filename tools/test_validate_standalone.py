"""Pure parser/fixture tests; ordinary CI needs no namespaces, ptrace or model."""

import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import validate_standalone as standalone


HEADERS = "[Requesting program interpreter: /lib64/ld-linux-x86-64.so.2]"
DYNAMIC = """
  0x0000000000000001 (NEEDED) Shared library: [libm.so.6]
  0x0000000000000001 (NEEDED) Shared library: [libc.so.6]
"""
LDD = """
    linux-vdso.so.1 (0x00007fff12340000)
    libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007fff56780000)
    libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007fff9abc0000)
    /lib64/ld-linux-x86-64.so.2 (0x00007fffabcd0000)
"""
TRACE = """
123 execve("/usr/bin/unshare", ["unshare"], 0x0) = 0
123 openat(AT_FDCWD, "/etc/ld.so.cache", O_RDONLY) = 3
124 execve("/usr/sbin/chroot", ["chroot", "/temporary/root"], 0x0) = 0
124 execve("/model", ["/model", "--prompt", "Hello"], 0x0) = 0
124 access("/etc/ld.so.preload", R_OK) = -1 ENOENT (No such file or directory)
124 openat(AT_FDCWD, "/lib/x86_64-linux-gnu/libc.so.6", O_RDONLY|O_CLOEXEC) = 3
124 newfstatat(3, "", {st_mode=S_IFREG|0755}, AT_EMPTY_PATH) = 0
124 openat(AT_FDCWD, "/proc/self/maps", O_RDONLY|O_CLOEXEC) = 3
124 openat(AT_FDCWD, "/proc/self/exe", O_RDONLY|O_CLOEXEC) = 3
124 exit_group(0) = ?
123 <... wait4 resumed>NULL, 0, NULL) = 124
"""


class DependencyTests(unittest.TestCase):
    def test_declared_dynamic_libraries_and_interpreter_are_resolved(self):
        result = standalone.dependencies(HEADERS, DYNAMIC, LDD)
        self.assertEqual(result["interpreter"], "/lib64/ld-linux-x86-64.so.2")
        self.assertEqual(result["needed"], ["libm.so.6", "libc.so.6"])
        self.assertEqual(result["paths"], ["/lib/x86_64-linux-gnu/libc.so.6",
                                          "/lib/x86_64-linux-gnu/libm.so.6",
                                          "/lib64/ld-linux-x86-64.so.2"])

    def test_missing_ambiguous_or_relative_dependencies_are_rejected(self):
        for output in [
            LDD.replace("/lib/x86_64-linux-gnu/libm.so.6", "not found"),
            LDD + "libc.so.6 => /elsewhere/libc.so.6 (0x1234)\n",
            LDD.replace("/lib/x86_64-linux-gnu/libm.so.6", "relative.so"),
            LDD.replace("/lib/x86_64-linux-gnu/libm.so.6", "/lib/../outside.so"),
            "not a dynamic executable",
        ]:
            with self.subTest(output=output), self.assertRaises(ValueError):
                standalone.dependencies(HEADERS, DYNAMIC, output)
        with self.assertRaises(ValueError):
            standalone.dependencies("no interpreter", DYNAMIC, LDD)


class TraceTests(unittest.TestCase):
    files = ["/model", "/lib/x86_64-linux-gnu/libc.so.6"]

    def test_only_model_calls_count_and_missing_optional_loader_files_are_ok(self):
        report = standalone.analyze_trace(TRACE, self.files)
        self.assertTrue(report["passed"])
        self.assertEqual(report["model_pids"], [124])
        self.assertIn("/proc/self/exe", [entry["path"] for entry in report["file_accesses"]])
        self.assertNotIn("/etc/ld.so.cache", [entry["path"] for entry in report["file_accesses"]])

    def test_untraced_execution_network_access_and_extra_file_access_never_pass(self):
        with self.assertRaises(ValueError):
            standalone.analyze_trace("strace: ptrace denied", self.files)
        with self.assertRaises(ValueError):
            standalone.analyze_trace(TRACE.replace("124 exit_group(0) = ?", ""), self.files)
        for extra in [
            '124 socket(AF_INET, SOCK_STREAM, IPPROTO_TCP) = -1 EPERM\n',
            '124 openat(AT_FDCWD, "/source/model.gguf", O_RDONLY) = 4\n',
            '124 execve("/usr/bin/cargo", ["cargo"], 0x0) = -1 ENOENT\n',
            '124 openat(AT_FDCWD, "/model", O_WRONLY) = 3\n',
            '124 openat(AT_FDCWD, "/model", O_RDONLY <unfinished ...>\n',
            '124 <... openat resumed>) = 3\n',
        ]:
            with self.subTest(extra=extra):
                self.assertFalse(standalone.analyze_trace(TRACE + extra, self.files)["passed"])

    def test_model_child_process_calls_cannot_escape_trace_review(self):
        extra = ('124 clone(child_stack=NULL, flags=SIGCHLD) = 125\n'
                 '125 connect(3, {sa_family=AF_INET}, 16) = -1 ENETUNREACH\n')
        report = standalone.analyze_trace(TRACE + extra, self.files)
        self.assertFalse(report["passed"])
        self.assertEqual(report["model_pids"], [124, 125])
        self.assertEqual(len(report["network_calls"]), 1)


class RunnerContractTests(unittest.TestCase):
    def test_explicit_trust_acknowledgment_and_bounded_timeout_are_required(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            standalone.parser().parse_args(["--artifact", "model", "--reference-json", "ref",
                                           "--output", "report"])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            standalone.main(["--artifact", "model", "--reference-json", "ref", "--output", "report",
                             "--trusted-generated-artifact", "--timeout", "61"])

    def test_missing_tool_is_recorded_as_failure_without_running_an_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "model"
            artifact.write_bytes(b"trusted fixture, never executed")
            reference = root / "reference.json"
            reference.write_text("{}")
            args = argparse.Namespace(artifact=artifact, reference_json=reference, output=root / "report",
                                      prompt="Hello", predict=4, context=128, timeout=60, memory_mib=1024)
            with patch.object(standalone, "tool", side_effect=RuntimeError("strace unavailable")), \
                    patch.object(standalone, "invoke_isolated") as execute:
                report = standalone.validate(args)
            self.assertEqual(report["status"], "failed")
            self.assertIn("strace unavailable", report["errors"])
            execute.assert_not_called()
            saved = json.loads((args.output / "report.json").read_text())
            self.assertEqual(saved["status"], "failed")


if __name__ == "__main__":
    unittest.main()
