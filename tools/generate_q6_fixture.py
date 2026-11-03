#!/usr/bin/env python3
"""Regenerate finite Q6_K known answers with the pinned upstream decoder.

Usage: python3 tools/generate_q6_fixture.py
Requires the llama.cpp checkout and built ggml-base library under target/bml-1.0.
"""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REVISION = "785322d17f2d51e5185cf996a1f4e46d87640841"


def main():
    if sys.byteorder != "little":
        raise SystemExit("This C oracle emits native floats; generation requires little endian")
    checkout = ROOT / "target/bml-1.0/llama.cpp"
    revision = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise SystemExit(f"Expected upstream revision {REVISION}, got {revision}")
    binary = ROOT / "target/bml-1.0/q6-reference"
    command = ["cc", "-O2", "-I", str(checkout / "ggml/include"), "-I", str(checkout / "ggml/src"),
               str(ROOT / "tools/q6_reference.c"), str(checkout / "build/ggml/src/libggml-base.a"),
               "-lstdc++", "-lm", "-lpthread", "-o", str(binary)]
    subprocess.run(command, check=True)
    payload = bytearray()
    state = 0x12345678
    # Full byte range, independent packed bit planes; signed/subnormal/max scales.
    for scale in [0x3555, 0xB555, 0x0001, 0x8001, 0x7BFF, 0xFBFF, 0x0000, 0x8000]:
        for _ in range(208):
            state = (1664525 * state + 1013904223) & 0xFFFFFFFF
            payload.append(state >> 24)
        payload.extend(struct.pack("<H", scale))
    output = subprocess.check_output([str(binary)], input=bytes(payload))
    if len(output) != 8 * 256 * 4:
        raise SystemExit("Unexpected oracle output size")
    directory = ROOT / "crates/gguf/tests/fixtures"
    directory.mkdir(exist_ok=True)
    paths = {"q6-ggml.bin": bytes(payload), "q6-ggml.f32le": output}
    for name, data in paths.items():
        (directory / name).write_bytes(data)
    manifest = {
        "schema": 1, "oracle": "ggml dequantize_row_q6_K", "llama_cpp_revision": revision,
        "source": f"https://github.com/ggml-org/llama.cpp/blob/{revision}/ggml/src/ggml-quants.c",
        "regenerate": "python3 tools/generate_q6_fixture.py", "blocks": 8, "elements": 2048,
        "encoding": "Q6_K input; IEEE binary32 little-endian output",
        "sha256": {name: hashlib.sha256(data).hexdigest() for name, data in paths.items()},
        "oracle_source_sha256": hashlib.sha256((checkout / "ggml/src/ggml-quants.c").read_bytes()).hexdigest(),
    }
    (directory / "q6-ggml.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
