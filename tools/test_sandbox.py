"""Notebook persistence and execution boundaries, without a Rust toolchain."""

import http.client
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "sandbox_server", Path(__file__).resolve().parents[1] / "sandbox" / "server.py"
)
sandbox = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sandbox)


class NotebookTests(unittest.TestCase):
    def test_ide_requests_are_validated_and_do_not_save_source(self):
        self.start_server()
        source = 'fn main() { let emoji = "😀"; }'
        with patch.object(self.notebook.rust_analyzer, "query", return_value={"ready": True, "diagnostics": [], "result": None}) as query:
            status, _ = self.request('/api/cells/01-hello/ide', {
                "source": source, "operation": "hover", "position": {"line": 0, "character": 3}})
            self.assertEqual(status, 200)
            query.assert_called_once_with("01-hello", source, "hover", {"line": 0, "character": 3})
            self.assertEqual(self.notebook.cell("01-hello")["source"], self.original)
            for operation, position in (("workspace/executeCommand", {"line": 0, "character": 0}),
                                        ("hover", {"line": -1, "character": 0}),
                                        ("hover", {"line": 0, "character": True}),
                                        ("hover", {"line": 8, "character": 0}),
                                        ("hover", {"line": 0, "character": 100})):
                status, _ = self.request('/api/cells/01-hello/ide', {
                    "source": source, "operation": operation, "position": position})
                self.assertEqual(status, 400)
            self.assertEqual(query.call_count, 1)
            status, _ = self.request('/api/cells/01-hello/ide', {
                "source": source}, headers={"X-Sandbox-Token": "invalid"})
            self.assertEqual(status, 403)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "examples").mkdir()
        self.original = 'fn main() { println!("original"); }\n'
        (self.root / "examples" / "hello.rs").write_text(self.original, encoding="utf-8")
        (self.root / "notebook.json").write_text(json.dumps({
            "title": "Notebook de prueba",
            "cells": [{"id": "01-hello", "title": "Hola", "description": "Ejemplo",
                       "file": "examples/hello.rs"}],
        }), encoding="utf-8")
        self.notebook = sandbox.Notebook(self.root)

    def start_server(self):
        server = sandbox.NotebookServer(("127.0.0.1", 0), self.notebook)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def close():
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.addCleanup(close)
        self.server = server
        self.port = server.server_port

    def request(self, path, payload=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        request_headers = {"Content-Type": "application/json",
                           "X-Sandbox-Token": self.notebook.token}
        request_headers.update(headers or {})
        try:
            connection.request("GET" if payload is None else "POST", path,
                               body=None if payload is None else json.dumps(payload),
                               headers=request_headers)
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def test_edits_survive_reload_and_reset_preserves_original(self):
        edited = 'fn main() { println!("edición"); }\n'
        self.assertTrue(self.notebook.save("01-hello", edited)["modified"])
        reloaded = sandbox.Notebook(self.root)
        self.assertEqual(reloaded.snapshot()["cells"][0]["source"], edited)
        self.assertEqual((self.root / "examples/hello.rs").read_text(), self.original)
        self.assertEqual((self.root / ".local/cells/01-hello.rs").read_text(), edited)
        reset = reloaded.reset("01-hello")
        self.assertEqual(reset["source"], self.original)
        self.assertFalse(reset["modified"])
        self.assertFalse((self.root / ".local/cells/01-hello.rs").exists())

    def test_http_save_and_reload_keep_source_in_auxiliary_file(self):
        self.start_server()
        edited = 'fn main() { println!("edited"); }'
        status, saved = self.request("/api/cells/01-hello/save", {"source": edited})
        self.assertEqual(status, 200)
        self.assertEqual(saved["source"], edited)
        status, snapshot = self.request("/api/notebook")
        self.assertEqual(status, 200)
        self.assertEqual(snapshot["cells"][0]["source"], edited)
        self.assertEqual((self.root / "examples/hello.rs").read_text(), self.original)

    def test_untrusted_requests_never_execute_code(self):
        self.start_server()
        with patch.object(self.notebook, "run") as execute:
            for headers in (
                {"X-Sandbox-Token": ""},
                {"X-Sandbox-Token": "wrong"},
                {"X-Sandbox-Token": "inválido"},
                {"Origin": "https://untrusted.example"},
                {"Host": "untrusted.example"},
            ):
                with self.subTest(headers=headers):
                    status, body = self.request("/api/cells/01-hello/run",
                                                {"source": self.original}, headers)
                    self.assertEqual(status, 403)
                    self.assertIn("error", body)
            execute.assert_not_called()

    def test_http_rejects_unknown_cells_and_invalid_source(self):
        self.start_server()
        with patch.object(self.notebook, "run") as execute:
            for path, payload, expected in (
                ("/api/cells/../../outside/save", {"source": "text"}, 404),
                ("/api/cells/unknown/run", {"source": "text"}, 404),
                ("/api/cells/01-hello/run", {"source": 42}, 400),
                ("/api/cells/01-hello/run", [], 400),
                ("/api/cells/01-hello/run",
                 {"source": "x" * (sandbox.MAX_SOURCE_BYTES + 1)}, 413),
            ):
                with self.subTest(path=path, expected=expected):
                    status, _ = self.request(path, payload)
                    self.assertEqual(status, expected)
            execute.assert_not_called()
        for path in ("/../Cargo.toml", "/.local/cells/01-hello.rs", "/server.py"):
            with self.subTest(path=path):
                status, _ = self.request(path)
                self.assertEqual(status, 404)

    def test_busy_runner_does_not_overwrite_another_run(self):
        with self.notebook.execution_lock:
            self.assertIsNone(self.notebook.run("01-hello", "replacement"))
        self.assertFalse((self.root / ".local/runner.rs").exists())
        self.assertEqual(self.notebook.cell("01-hello")["source"], self.original)

    def test_missing_cargo_preserves_edit_and_releases_execution_lock(self):
        with patch.object(sandbox.shutil, "which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "cargo"):
                self.notebook.run("01-hello", "edited")
        self.assertFalse(self.notebook.execution_lock.locked())
        self.assertEqual(self.notebook.cell("01-hello")["source"], "edited")

    def test_server_close_stops_active_execution_and_releases_lock(self):
        self.start_server()
        real_run_process = sandbox.run_process
        marker = self.root / ".local" / "started"
        results = []

        def run_python_instead_of_cargo(command, **kwargs):
            return real_run_process([
                sys.executable, "-c",
                "from pathlib import Path; import time; "
                "Path('started').write_text('ready'); time.sleep(60)",
            ], **kwargs)

        with patch.object(sandbox.shutil, "which", return_value=sys.executable), \
                patch.object(sandbox, "run_process", side_effect=run_python_instead_of_cargo):
            thread = threading.Thread(
                target=lambda: results.append(self.notebook.run("01-hello", "edited")),
                daemon=True,
            )
            thread.start()
            try:
                deadline = time.monotonic() + 3
                while not marker.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(marker.exists(), "The child process did not start")
                self.server.server_close()
                thread.join(timeout=2)
                self.assertFalse(thread.is_alive())
                self.assertFalse(self.notebook.execution_lock.locked())
                self.assertNotEqual(results[0]["exit_code"], 0)
                self.assertFalse(results[0]["timed_out"])
                self.assertLess(results[0]["duration_ms"], 5000)
            finally:
                self.notebook.close()
                thread.join(timeout=2)


class ProcessTests(unittest.TestCase):
    def execute(self, source, timeout=3):
        with tempfile.TemporaryDirectory() as directory:
            return sandbox.run_process([sys.executable, "-c", source], cwd=directory,
                                       env=os.environ.copy(), timeout=timeout)

    def test_stdout_stderr_and_failure_status_are_reported(self):
        result = self.execute("import sys; print('result'); print('error', file=sys.stderr); sys.exit(7)")
        self.assertEqual(result["stdout"], "result\n")
        self.assertEqual(result["stderr"], "error\n")
        self.assertEqual(result["exit_code"], 7)
        self.assertFalse(result["timed_out"])
        self.assertFalse(result["output_truncated"])

    def test_nonterminating_script_is_stopped(self):
        result = self.execute("import time; print('started', flush=True); time.sleep(60)", timeout=0.2)
        self.assertTrue(result["timed_out"])
        self.assertNotEqual(result["exit_code"], 0)
        self.assertLess(result["duration_ms"], 5000)

    def test_large_output_is_drained_without_exceeding_retention_limit(self):
        size = sandbox.MAX_OUTPUT_BYTES * 3
        result = self.execute(f"import sys; sys.stdout.write('a' * {size}); sys.stderr.write('b' * {size})")
        self.assertEqual(result["exit_code"], 0)
        self.assertFalse(result["timed_out"])
        self.assertTrue(result["output_truncated"])
        self.assertEqual(result["stdout"], "a" * sandbox.MAX_OUTPUT_BYTES)
        self.assertEqual(result["stderr"], "b" * sandbox.MAX_OUTPUT_BYTES)

    @unittest.skipUnless(os.name == "posix", "Process groups require POSIX")
    def test_child_holding_output_pipe_does_not_escape_timeout(self):
        result = self.execute(
            "import subprocess, sys; "
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])",
            timeout=0.2,
        )
        self.assertTrue(result["timed_out"])
        self.assertLess(result["duration_ms"], 5000)


if __name__ == "__main__":
    unittest.main()
