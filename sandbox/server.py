#!/usr/bin/env python3
"""Local editable Rust notebook. Run with: python3 sandbox/server.py."""

import argparse
import hmac
import json
import os
from pathlib import Path
import queue
import re
import secrets
import shutil
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parent
MAX_SOURCE_BYTES = 128 * 1024
MAX_OUTPUT_BYTES = 128 * 1024  # Each stream; extra output is drained and discarded.


class RustAnalyzer:
    """Small, serialized LSP bridge. Only fixed Rust IDE operations are exposed."""

    def __init__(self, notebook):
        self.notebook = notebook
        self.lock = threading.Lock()
        self.condition = threading.Condition()
        self.process = None
        self.pending = {}
        self.diagnostics = {}
        self.versions = {}
        self.sources = {}
        self.sequence = 0
        self.write_lock = threading.Lock()

    def send(self, message):
        data = json.dumps(message).encode("utf-8")
        with self.write_lock:
            self.process.stdin.write(f"Content-Length: {len(data)}\r\n\r\n".encode() + data)
            self.process.stdin.flush()

    def read(self):
        try:
            while True:
                headers = {}
                while True:
                    line = self.process.stdout.readline()
                    if not line:
                        return
                    if line == b"\r\n":
                        break
                    key, value = line.decode().split(":", 1)
                    headers[key.lower()] = value.strip()
                length = int(headers["content-length"])
                if not 0 < length <= 16 * 1024 * 1024:
                    return
                message = json.loads(self.process.stdout.read(length))
                if "method" not in message:
                    with self.condition:
                        waiting = self.pending.get(message.get("id"))
                        if waiting:
                            waiting.put(message)
                elif "id" in message:
                    method = message["method"]
                    result = None
                    if method == "workspace/configuration":
                        result = [None for _ in message.get("params", {}).get("items", [])]
                    elif method == "workspace/applyEdit":
                        result = {"applied": False}
                    self.send({"jsonrpc": "2.0", "id": message["id"], "result": result})
                elif message["method"] == "textDocument/publishDiagnostics":
                    params = message["params"]
                    with self.condition:
                        self.diagnostics[params["uri"]] = params
                        self.condition.notify_all()
        except (OSError, ValueError, KeyError):
            pass
        finally:
            with self.condition:
                for waiting in self.pending.values():
                    waiting.put({"error": {"message": "rust-analyzer se ha detenido"}})
                self.condition.notify_all()

    def call(self, method, params, timeout=30):
        self.sequence += 1
        identifier = self.sequence
        waiting = queue.Queue()
        with self.condition:
            self.pending[identifier] = waiting
        try:
            self.send({"jsonrpc": "2.0", "id": identifier, "method": method, "params": params})
            try:
                response = waiting.get(timeout=timeout)
            except queue.Empty as error:
                self.send({"jsonrpc": "2.0", "method": "$/cancelRequest", "params": {"id": identifier}})
                raise RuntimeError("rust-analyzer sigue ocupado; vuelve a intentarlo") from error
            if "error" in response:
                raise RuntimeError(response["error"].get("message", "Error de rust-analyzer"))
            return response.get("result")
        finally:
            with self.condition:
                self.pending.pop(identifier, None)

    def start(self):
        if self.process is not None:
            if self.process.poll() is not None:
                raise RuntimeError("rust-analyzer se ha detenido; reinicia el servidor")
            return
        binary = shutil.which("rust-analyzer")
        if not binary:
            raise RuntimeError("Instala rust-analyzer y rust-src con rustup y reinicia el servidor")
        root = (self.notebook.local / "ide").resolve()
        root.mkdir(parents=True, exist_ok=True)
        manifest = (self.notebook.root / "Cargo.toml").read_text().split("[[bin]]")[0]
        manifest = re.sub(r'path\s*=\s*"([^"]+)"',
                          lambda m: "path = " + json.dumps(str((self.notebook.root / m[1]).resolve())), manifest)
        for cell_id in self.notebook.cells:
            path = root / f"{cell_id}.rs"
            path.write_text(self.notebook.cell(cell_id)["source"], encoding="utf-8")
            manifest += f'\n[[bin]]\nname = "cell-{cell_id}"\npath = "{cell_id}.rs"\n'
        (root / "Cargo.toml").write_text(manifest, encoding="utf-8")
        shutil.copyfile(self.notebook.root / "Cargo.lock", root / "Cargo.lock")
        self.process = subprocess.Popen([binary], cwd=self.notebook.root,
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL)
        threading.Thread(target=self.read, daemon=True).start()
        try:
            self.call("initialize", {
                "processId": os.getpid(), "rootUri": root.as_uri(),
                "capabilities": {"general": {"positionEncodings": ["utf-16"]},
                                 "textDocument": {"publishDiagnostics": {"versionSupport": True},
                                                  "completion": {"completionItem": {"snippetSupport": False}}}},
                "initializationOptions": {"checkOnSave": False, "check": {"enable": False},
                                          "cargo": {"buildScripts": {"enable": False}, "offline": True},
                                          "procMacro": {"enable": False}, "numThreads": 2,
                                          "cachePriming": {"enable": False}},
            })
            self.send({"jsonrpc": "2.0", "method": "initialized", "params": {}})
        except (OSError, RuntimeError):
            self.close()
            raise

    def query(self, cell_id, source, operation, position):
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("rust-analyzer está procesando otra consulta; vuelve a intentarlo")
        try:
            self.start()
            uri = (self.notebook.local / "ide" / f"{cell_id}.rs").resolve().as_uri()
            if uri not in self.versions:
                self.versions[uri] = 1
                self.sources[uri] = source
                self.send({"jsonrpc": "2.0", "method": "textDocument/didOpen", "params": {
                    "textDocument": {"uri": uri, "languageId": "rust", "version": 1, "text": source}}})
            elif self.sources[uri] != source:
                self.versions[uri] += 1
                self.sources[uri] = source
                self.send({"jsonrpc": "2.0", "method": "textDocument/didChange", "params": {
                    "textDocument": {"uri": uri, "version": self.versions[uri]},
                    "contentChanges": [{"text": source}]}})
            result = None
            if operation != "diagnostics":
                result = self.call(f"textDocument/{operation}", {"textDocument": {"uri": uri}, "position": position})
            with self.condition:
                self.condition.wait_for(lambda: self.diagnostics.get(uri, {}).get("version") == self.versions[uri], timeout=2)
                latest = self.diagnostics.get(uri, {})
                ready = latest.get("version") == self.versions[uri]
                diagnostics = latest.get("diagnostics", []) if ready else []
            return {"result": result, "diagnostics": diagnostics, "ready": ready}
        finally:
            self.lock.release()

    def close(self):
        process = self.process
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
            for stream in (process.stdin, process.stdout):
                stream.close()


class Notebook:
    def __init__(self, root=ROOT, timeout=120):
        self.root = Path(root)
        self.local = self.root / ".local"
        self.edits = self.local / "cells"
        self.edits.mkdir(parents=True, exist_ok=True)
        self.catalog = json.loads((self.root / "notebook.json").read_text(encoding="utf-8"))
        self.cells = {cell["id"]: cell for cell in self.catalog["cells"]}
        if len(self.cells) != len(self.catalog["cells"]):
            raise ValueError("El notebook tiene identificadores duplicados")
        for cell_id, cell in self.cells.items():
            if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", cell_id):
                raise ValueError("Identificador de celda inválido")
            source = (self.root / cell["file"]).resolve()
            if not source.is_relative_to(self.root.resolve()) or not source.is_file():
                raise ValueError("Ejemplo fuera del sandbox o inexistente")
        self.token = secrets.token_urlsafe(32)
        self.timeout = timeout
        self.execution_lock = threading.Lock()
        self.edit_lock = threading.Lock()
        self.cancel = threading.Event()
        self.rust_analyzer = RustAnalyzer(self)

    def close(self):
        self.cancel.set()
        self.rust_analyzer.close()
        if self.execution_lock.acquire(timeout=4):
            self.execution_lock.release()

    def original(self, cell_id):
        return (self.root / self.cells[cell_id]["file"]).read_text(encoding="utf-8")

    def cell(self, cell_id):
        original = self.original(cell_id)
        path = self.edits / f"{cell_id}.rs"
        source = path.read_text(encoding="utf-8") if path.exists() else original
        return {**self.cells[cell_id], "source": source, "modified": source != original}

    def snapshot(self):
        with self.edit_lock:
            return {"title": self.catalog["title"], "token": self.token,
                    "cells": [self.cell(cell_id) for cell_id in self.cells]}

    def save(self, cell_id, source):
        with self.edit_lock:
            path = self.edits / f"{cell_id}.rs"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(source, encoding="utf-8")
            temporary.replace(path)
            return self.cell(cell_id)

    def reset(self, cell_id):
        with self.edit_lock:
            (self.edits / f"{cell_id}.rs").unlink(missing_ok=True)
            return self.cell(cell_id)

    def run(self, cell_id, source):
        """Return None when another cell is executing; never invoke a shell."""
        if not self.execution_lock.acquire(blocking=False):
            return None
        try:
            saved = self.save(cell_id, source)
            (self.local / "runner.rs").write_text(source, encoding="utf-8")
            cargo = shutil.which("cargo")
            if cargo is None:
                raise RuntimeError(
                    "No se encontró cargo en PATH. Instala el toolchain de "
                    "rust-toolchain.toml y vuelve a iniciar el servidor."
                )
            environment = os.environ.copy()
            environment["CARGO_TERM_COLOR"] = "never"
            # Keep build artifacts with the other local notebook files.
            environment["CARGO_TARGET_DIR"] = str(self.local / "target")
            result = run_process(
                [cargo, "run", "--quiet", "--locked", "--manifest-path",
                 str(self.root / "Cargo.toml"), "--bin", "notebook-cell"],
                cwd=self.local, env=environment, timeout=self.timeout, cancel=self.cancel,
            )
            return {**result, "modified": saved["modified"]}
        finally:
            self.execution_lock.release()


def run_process(command, *, cwd, env, timeout, cancel=None):
    """Bound execution time and retained output, including Rust compiler errors."""
    start = time.monotonic()
    process = subprocess.Popen(
        command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=(os.name == "posix"),
    )
    outputs = [bytearray(), bytearray()]
    truncated = [False, False]

    def drain(stream, index):
        with stream:
            while chunk := stream.read(8192):
                available = MAX_OUTPUT_BYTES - len(outputs[index])
                outputs[index].extend(chunk[:available])
                if len(chunk) > available:
                    truncated[index] = True

    threads = [threading.Thread(target=drain, args=(stream, index), daemon=True)
               for index, stream in enumerate((process.stdout, process.stderr))]
    for thread in threads:
        thread.start()
    timed_out = False
    try:
        # Descendants may keep the pipes open after the direct child exits.
        while process.poll() is None or any(thread.is_alive() for thread in threads):
            if cancel is not None and cancel.is_set():
                break
            if time.monotonic() - start >= timeout:
                timed_out = True
                break
            time.sleep(0.02)
    finally:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        elif process.poll() is None:
            process.kill()
        process.wait()
        for thread in threads:
            thread.join(timeout=1)
    return {
        "stdout": outputs[0].decode("utf-8", errors="replace"),
        "stderr": outputs[1].decode("utf-8", errors="replace"),
        "exit_code": process.returncode,
        "duration_ms": round((time.monotonic() - start) * 1000),
        "timed_out": timed_out,
        "output_truncated": any(truncated),
    }


class NotebookServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, notebook):
        self.notebook = notebook
        super().__init__(address, Handler)

    def server_close(self):
        self.notebook.close()
        super().server_close()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format_string, *args):
        # Do not log submitted Rust source or authentication tokens.
        pass

    def respond(self, status, body, content_type="application/json; charset=utf-8"):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; script-src 'self'; style-src 'self'; "
                         "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; "
                         "form-action 'none'")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def local_request(self):
        port = self.server.server_port
        allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        host = self.headers.get("Host", "")
        origin = self.headers.get("Origin")
        if host not in allowed_hosts or (origin and origin != f"http://{host}"):
            self.respond(403, {"error": "Solicitud de un origen no permitido"})
            return False
        return True

    def do_GET(self):
        if not self.local_request():
            return
        path = urlsplit(self.path).path
        if path == "/api/notebook":
            self.respond(200, self.server.notebook.snapshot())
            return
        assets = {"/": ("index.html", "text/html; charset=utf-8"),
                  "/index.html": ("index.html", "text/html; charset=utf-8"),
                  "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                  "/rust-editor.js": ("rust-editor.js", "text/javascript; charset=utf-8"),
                  "/style.css": ("style.css", "text/css; charset=utf-8")}
        if path in assets:
            filename, content_type = assets[path]
            self.respond(200, (self.server.notebook.root / filename).read_bytes(), content_type)
        else:
            self.respond(404, {"error": "Recurso inexistente"})

    def do_POST(self):
        if not self.local_request():
            return
        notebook = self.server.notebook
        supplied_token = self.headers.get("X-Sandbox-Token", "").encode("utf-8")
        if not hmac.compare_digest(supplied_token, notebook.token.encode("ascii")):
            self.respond(403, {"error": "Recarga el notebook para renovar la sesión"})
            return
        match = re.fullmatch(r"/api/cells/([a-z0-9_-]+)/(save|run|reset|ide)", self.path)
        if not match or match[1] not in notebook.cells:
            self.respond(404, {"error": "Celda o acción inexistente"})
            return
        if self.headers.get_content_type() != "application/json":
            self.respond(415, {"error": "Se requiere application/json"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_SOURCE_BYTES * 6 + 100:
                self.respond(413, {"error": "Solicitud demasiado grande o vacía"})
                return
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("Objeto JSON requerido")
            source = payload.get("source")
            if match[2] != "reset":
                if not isinstance(source, str):
                    raise ValueError("El código debe ser texto")
                if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
                    self.respond(413, {"error": "El código supera el límite de 128 KiB"})
                    return
            if match[2] == "ide":
                operation = payload.get("operation", "diagnostics")
                position = payload.get("position", {"line": 0, "character": 0})
                if operation not in ("diagnostics", "completion", "hover"):
                    raise ValueError("Operación IDE no permitida")
                if (not isinstance(position, dict) or any(type(position.get(key)) is not int
                    or position[key] < 0 for key in ("line", "character"))):
                    raise ValueError("Posición IDE inválida")
                lines = source.split("\n")
                if (position["line"] >= len(lines) or position["character"] >
                        len(lines[position["line"]].encode("utf-16-le")) // 2):
                    raise ValueError("Posición IDE fuera del código")
                result = notebook.rust_analyzer.query(match[1], source, operation, position)
            elif match[2] == "reset":
                result = notebook.reset(match[1])
            elif match[2] == "save":
                result = notebook.save(match[1], source)
            else:
                result = notebook.run(match[1], source)
                if result is None:
                    self.respond(409, {"error": "Ya hay una celda en ejecución"})
                    return
            self.respond(200, result)
        except (ValueError, UnicodeError) as error:
            self.respond(400, {"error": str(error)})
        except (OSError, RuntimeError) as error:
            self.respond(500, {"error": str(error)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--timeout", type=float, default=120,
                        help="Límite por celda, incluida la compilación (segundos)")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout debe ser mayor que cero")
    if not 0 <= args.port <= 65535:
        parser.error("--port debe estar entre 0 y 65535")
    notebook = Notebook(timeout=args.timeout)
    with NotebookServer(("127.0.0.1", args.port), notebook) as server:
        print(f"Notebook BML: http://127.0.0.1:{server.server_port}", flush=True)
        print("Ctrl+C para cerrar. Las ediciones se guardan en sandbox/.local/cells/.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nServidor cerrado.")


if __name__ == "__main__":
    main()
