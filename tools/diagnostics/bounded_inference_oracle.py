#!/usr/bin/env python3
"""Independent F32 Llama diagnostic using checked BML formulas and a native oracle.

Requires Python 3.11+ and a C11 compiler (CC, default cc) on a Unix-like host. No
model is downloaded. This is not the production Rust runtime, a llama.cpp
comparison, or a throughput benchmark. Each BML intermediate is checked by the
separate bounded_basis.c interpreter; max selection, F32 KV conversion and
position/dimension constants are host responsibilities.

Examples:
  python3 tools/diagnostics/bounded_inference_oracle.py --self-check --output DIR
  python3 tools/diagnostics/bounded_inference_oracle.py model.gguf \
      --tokens 1,9038 --output DIR
  python3 tools/diagnostics/bounded_inference_oracle.py model.gguf \
      --tokens 1,9038 --native-only --output DIR

The fixed operand declarations are experimental and may reject other models or
prompts. A completed sample does not prove a uniform numerical error bound.
"""

import argparse
import ctypes
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import random
import struct
import subprocess
import sys
import time


class Model:
    """Minimal independent reader for F32 tensors in a local GGUF v2/v3."""

    def __init__(self, path):
        self.file = path.open("rb")
        self.data = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        self.position = 0
        magic, version, tensor_count, metadata_count = self.read("IIQQ")
        if magic != 0x46554747 or version not in (2, 3):
            raise ValueError("expected GGUF v2/v3")
        self.meta = {}
        for _ in range(metadata_count):
            key = self.string()
            self.meta[key] = self.value(self.read("I"))
        if self.meta.get("general.architecture") != "llama":
            raise ValueError("only dense Llama is supported by this diagnostic")
        self.tensors = {}
        for _ in range(tensor_count):
            name = self.string()
            dimensions = [self.read("Q") for _ in range(self.read("I"))]
            tensor_type, offset = self.read("IQ")
            if tensor_type != 0 or not dimensions or any(n == 0 for n in dimensions):
                raise ValueError(f"expected nonempty F32 tensor: {name}")
            self.tensors[name] = (dimensions, offset)
        alignment = self.meta.get("general.alignment", 32)
        if alignment <= 0:
            raise ValueError("invalid GGUF alignment")
        self.start = (self.position + alignment - 1) // alignment * alignment

    def read(self, fmt):
        parser = struct.Struct("<" + fmt)
        values = parser.unpack_from(self.data, self.position)
        self.position += parser.size
        return values[0] if len(values) == 1 else values

    def string(self):
        count = self.read("Q")
        end = self.position + count
        if end > len(self.data):
            raise ValueError("truncated GGUF string")
        value = self.data[self.position:end].decode()
        self.position = end
        return value

    def value(self, kind):
        if kind == 8:
            return self.string()
        if kind == 9:
            element_kind, count = self.read("IQ")
            return [self.value(element_kind) for _ in range(count)]
        formats = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
                   6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
        if kind not in formats:
            raise ValueError(f"unsupported GGUF metadata type {kind}")
        return self.read(formats[kind])

    def tensor(self, name):
        dimensions, offset = self.tensors[name]
        start = self.start + offset
        end = start + 4 * math.prod(dimensions)
        if end > len(self.data):
            raise ValueError(f"truncated GGUF tensor {name}")
        # The file is little-endian; ctypes/native memoryview conversion below
        # deliberately rejects hosts where this would reinterpret bytes wrongly.
        if sys.byteorder != "little":
            raise ValueError("diagnostic currently requires a little-endian host")
        return memoryview(self.data)[start:end].cast("f")

    def row(self, name, index):
        width, count = self.tensors[name][0]
        if not 0 <= index < count:
            raise ValueError("token index outside embedding tensor")
        return self.tensor(name)[index * width:(index + 1) * width]


D = ctypes.c_double
F = ctypes.c_float
DP = ctypes.POINTER(D)
FP = ctypes.POINTER(F)
Z = ctypes.c_size_t


class Stats(ctypes.Structure):
    _fields_ = [("nodes", ctypes.c_uint64), ("max_exponent", D),
                ("min_argument", D), ("max_abs_value", D), ("error", ctypes.c_int),
                ("bad_left", D), ("bad_right", D)]


def load_backend(output):
    source = Path(__file__).with_name("bounded_basis.c")
    library = output / "bounded_basis.so"
    command = [os.environ.get("CC", "cc"), "-O2", "-std=c11", "-shared", "-fPIC",
               "-fno-fast-math", "-ffp-contract=off", str(source), "-lm", "-o", str(library)]
    subprocess.run(command, check=True)
    lib = ctypes.CDLL(str(library.resolve()))
    lib.read_stats.restype = Stats
    for name, args in {"bounded_product": [D, D, D, D], "bounded_add": [D, D],
                       "bounded_scale": [D, D], "bounded_swiglu": [D, D]}.items():
        getattr(lib, name).argtypes = args
        getattr(lib, name).restype = D
    lib.bounded_rope.argtypes = [D, D, D, D, DP]
    lib.bounded_rms.argtypes = [DP, FP, Z, D, DP]
    lib.bounded_dot.argtypes = [DP, DP, Z, D, D, DP]
    lib.bounded_project.argtypes = [DP, FP, Z, Z, DP, DP]
    lib.native_project.argtypes = [DP, FP, Z, Z, DP, DP]
    lib.bounded_softmax.argtypes = [DP, Z, DP]
    return lib, command


def array(values):
    return (D * len(values))(*values)


def float32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def require(values, bound, label):
    for index, value in enumerate(values):
        if not math.isfinite(value) or abs(value) > bound:
            raise ValueError(f"{label}: operand {index}={value} exceeds bound {bound}")


class Oracle:
    def __init__(self, model, lib, output, progress):
        self.model, self.lib, self.output, self.progress = model, lib, output, progress
        self.weights, self.inventory, self.records = {}, [], []
        self.total_nodes = 0
        for name, (dimensions, _) in model.tensors.items():
            view = model.tensor(name)
            self.weights[name] = (F * len(view)).from_buffer_copy(view)
            self.inventory.append({"name": name, "shape": dimensions,
                                   "min": min(view), "max": max(view), "max_abs": max(map(abs, view))})
        self.heads = model.meta["llama.attention.head_count"]
        self.kv_heads = model.meta.get("llama.attention.head_count_kv", self.heads)
        self.width = model.meta["llama.embedding_length"]
        self.layers = model.meta["llama.block_count"]
        if self.heads <= 0 or self.kv_heads <= 0 or self.width % self.heads or self.heads % self.kv_heads:
            raise ValueError("unsupported attention dimensions")
        self.dim = self.width // self.heads
        if self.dim % 2:
            raise ValueError("RoPE requires even head width")
        self.epsilon = model.meta.get("llama.attention.layer_norm_rms_epsilon", 1e-5)
        self.rope_base = model.meta.get("llama.rope.freq_base", 10000.0)
        if not math.isfinite(self.rope_base) or self.rope_base <= 0:
            raise ValueError("invalid RoPE frequency base")
        if model.meta.get("llama.rope.dimension_count", self.dim) != self.dim:
            raise ValueError("partial rotary dimensions are unsupported")
        if model.meta.get("llama.rope.scaling.type", "none") != "none":
            raise ValueError("scaled RoPE is unsupported")
        if model.meta.get("llama.expert_count", 0) or any(name.endswith(".bias") for name in self.weights):
            raise ValueError("experts and bias tensors are unsupported")
        scaled = self.epsilon / 256.0
        if not (0 < self.width <= 255 * 255 and 2 ** -32 <= scaled <= 1):
            raise ValueError("model outside experimental RMS configuration limits")
        if (math.ceil(math.sqrt(self.width)) + 1) * 16 > 480:
            raise ValueError("model width exceeds the fixed RMS weight-bound product")

    def record(self, label, values, bounded, **extra):
        record = {"label": label, "bounded": bounded, "length": len(values),
                  "min": min(values), "max": max(values), "max_abs": max(map(abs, values)), **extra}
        if bounded:
            state = self.lib.read_stats()
            stats = {name: getattr(state, name) for name, _ in Stats._fields_}
            if state.error:
                raise ValueError(f"{label}: invalid BML intermediate {stats}")
            self.total_nodes += state.nodes
            record["stats"] = stats
        self.records.append(record)
        if self.progress:
            print(json.dumps({key: record[key] for key in ("label", "bounded", "length", "max_abs")}), file=sys.stderr, flush=True)
        write_json(self.output / "progress.json", {"records": self.records, "total_bml_nodes": self.total_nodes})

    def norm(self, values, name, bounded):
        weights = self.weights[name]
        if len(weights) != len(values):
            raise ValueError(f"normalization shape mismatch: {name}")
        if bounded:
            require(values, 16, name + " inputs")
            require(weights, 16, name + " weights")
            self.lib.reset_stats()
            output = (D * len(values))()
            self.lib.bounded_rms(array(values), weights, len(values), self.epsilon, output)
            result = list(output)
        else:
            root = math.sqrt(sum(value * value for value in values) / len(values) + self.epsilon)
            result = [value * weight / root for value, weight in zip(values, weights)]
        self.record(name, result, bounded, input_max=max(map(abs, values)), weight_max=max(map(abs, weights)))
        return result

    def project(self, values, name, bounded):
        width, rows = self.model.tensors[name][0]
        if len(values) != width:
            raise ValueError(f"projection shape mismatch: {name}")
        weights = self.weights[name]
        output, partial = (D * rows)(), D()
        if bounded:
            require(values, 128, name + " inputs")
            require(weights, 2, name + " weights")
            self.lib.reset_stats()
            self.lib.bounded_project(array(values), weights, width, rows, output, ctypes.byref(partial))
        else:
            self.lib.native_project(array(values), weights, width, rows, output, ctypes.byref(partial))
        result = list(output)
        self.record(name, result, bounded, input_max=max(map(abs, values)),
                    weight_max=max(map(abs, weights)), max_partial=partial.value)
        return result

    def rotate(self, values, position, label, bounded):
        output = []
        if bounded:
            require(values, 16, label)
            self.lib.reset_stats()
        for index in range(0, len(values), 2):
            angle = position / (self.rope_base ** ((index % self.dim) / self.dim))
            cosine, sine = math.cos(angle), math.sin(angle)
            if bounded:
                pair = (D * 2)()
                self.lib.bounded_rope(values[index], values[index + 1], cosine, sine, pair)
                output.extend(pair)
            else:
                output.extend([values[index] * cosine - values[index + 1] * sine,
                               values[index] * sine + values[index + 1] * cosine])
        self.record(label, output, bounded)
        return output

    def residual(self, left, right, label, bounded):
        if len(left) != len(right):
            raise ValueError("residual shape mismatch")
        if bounded:
            require(left, 32, label + " left")
            require(right, 32, label + " right")
            self.lib.reset_stats()
            result = [self.lib.bounded_add(x, y) for x, y in zip(left, right)]
        else:
            result = [x + y for x, y in zip(left, right)]
        self.record(label, result, bounded)
        return result

    def attention(self, query, keys, values, position, label, bounded):
        if bounded:
            self.lib.reset_stats()
        output, all_scores, all_probabilities = [], [], []
        for head in range(self.heads):
            kv_head = head // (self.heads // self.kv_heads)
            current = query[head * self.dim:(head + 1) * self.dim]
            scores = []
            for previous in range(position + 1):
                key = keys[previous][kv_head * self.dim:(kv_head + 1) * self.dim]
                if bounded:
                    require(current, 16, label + " query")
                    require(key, 16, label + " key")
                    dot = D()
                    self.lib.bounded_dot(array(current), array(key), self.dim, 16, 16, ctypes.byref(dot))
                    require([dot.value], 256, label + " score before scaling")
                    scores.append(self.lib.bounded_scale(dot.value, 1 / math.sqrt(self.dim)))
                else:
                    scores.append(sum(x * y for x, y in zip(current, key)) / math.sqrt(self.dim))
            if bounded:
                require(scores, 16, label + " scores")
                result = (D * len(scores))()
                self.lib.bounded_softmax(array(scores), len(scores), result)
                probabilities = list(result)
            else:
                exponentials = [math.exp(value - max(scores)) for value in scores]
                total = sum(exponentials)
                probabilities = [value / total for value in exponentials]
            all_scores.extend(scores)
            all_probabilities.extend(probabilities)
            for dimension in range(self.dim):
                history = [values[t][kv_head * self.dim + dimension] for t in range(position + 1)]
                if bounded:
                    require(probabilities, 1, label + " probabilities")
                    require(history, 16, label + " values")
                    dot = D()
                    self.lib.bounded_dot(array(probabilities), array(history), len(history), 1, 16, ctypes.byref(dot))
                    output.append(dot.value)
                else:
                    output.append(sum(x * y for x, y in zip(probabilities, history)))
        self.record(label, output, bounded, max_score=max(map(abs, all_scores)), max_probability=max(all_probabilities))
        return output

    def forward(self, token_ids, bounded):
        cache = [([], []) for _ in range(self.layers)]
        logits = []
        for position, token in enumerate(token_ids):
            hidden = list(self.model.row("token_embd.weight", token))
            for layer in range(self.layers):
                prefix = f"blk.{layer}"
                normalized = self.norm(hidden, prefix + ".attn_norm.weight", bounded)
                query = self.project(normalized, prefix + ".attn_q.weight", bounded)
                key = self.project(normalized, prefix + ".attn_k.weight", bounded)
                value = self.project(normalized, prefix + ".attn_v.weight", bounded)
                query = self.rotate(query, position, prefix + ".rope_q", bounded)
                key = self.rotate(key, position, prefix + ".rope_k", bounded)
                keys, values = cache[layer]
                keys.append(list(map(float32, key)))
                values.append(list(map(float32, value)))
                attention = self.attention(query, keys, values, position, prefix + ".attention", bounded)
                projected = self.project(attention, prefix + ".attn_output.weight", bounded)
                hidden = self.residual(hidden, projected, prefix + ".attention_residual", bounded)
                normalized = self.norm(hidden, prefix + ".ffn_norm.weight", bounded)
                gate = self.project(normalized, prefix + ".ffn_gate.weight", bounded)
                up = self.project(normalized, prefix + ".ffn_up.weight", bounded)
                if bounded:
                    require(gate, 16, prefix + ".gate")
                    require(up, 16, prefix + ".up")
                    self.lib.reset_stats()
                    activated = [self.lib.bounded_swiglu(x, y) for x, y in zip(gate, up)]
                else:
                    activated = [x / (1 + math.exp(-x)) * y for x, y in zip(gate, up)]
                self.record(prefix + ".swiglu", activated, bounded,
                            gate_max=max(map(abs, gate)), up_max=max(map(abs, up)))
                down = self.project(activated, prefix + ".ffn_down.weight", bounded)
                hidden = self.residual(hidden, down, prefix + ".ffn_residual", bounded)
            normalized = self.norm(hidden, "output_norm.weight", bounded)
            name = "output.weight" if "output.weight" in self.weights else "token_embd.weight"
            logits = self.project(normalized, name, bounded)
        return logits


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class PythonBasis:
    """Slow scalar crosscheck, independent of the C dispatch implementation."""

    def b(self, left, right):
        if not math.isfinite(left) or not math.isfinite(right) or right <= 0:
            raise ValueError("invalid BML inputs")
        exponential = math.exp2(left)
        if not math.isfinite(exponential) or exponential == 0:
            raise ValueError("BML exponential outside range")
        value = exponential - math.log2(right)
        if not math.isfinite(value):
            raise ValueError("BML output outside range")
        return value

    def exp(self, value):
        return self.b(value, 1)

    def log(self, value):
        return self.b(1, self.b(self.b(1, value), 1))

    def sub(self, left, right):
        return self.b(self.log(left), self.exp(right))

    def add(self, left, right, shift=64):
        return self.sub(shift, self.sub(self.sub(shift, left), right))

    def neg(self, value, shift=64):
        return self.sub(self.sub(shift, value), shift)

    def positive_add(self, left, right, shift=64):
        return self.sub(left, self.neg(right, shift))

    def mul(self, left, right):
        return self.exp(self.add(self.log(left), self.log(right), 64))

    def old_add(self, left, right):
        positive = self.exp(left)
        remainder = self.b(left, positive)
        return self.sub(positive, self.sub(remainder, right))

    def old_mul(self, left, right):
        return self.exp(self.old_add(self.log(left), self.log(right)))

    def old_signed(self, left, right):
        p, r = self.exp(left), self.exp(right)
        q, s = self.b(left, p), self.b(right, r)
        return self.sub(self.old_add(self.old_mul(p, r), self.old_mul(q, s)),
                        self.old_add(self.old_mul(p, s), self.old_mul(q, r)))

    def normalize(self, value, bound):
        return self.sub(self.mul(self.add(value, 2 * bound, 2 * bound), 1 / bound), 2)

    def signed(self, left, right, left_bound, right_bound):
        product = self.old_signed(self.normalize(left, left_bound), self.normalize(right, right_bound))
        bound = left_bound * right_bound
        return self.sub(self.mul(self.add(product, 2, 2), bound), 2 * bound)

    def reciprocal(self, value):
        return self.exp(self.neg(self.log(value), 64))

    def swiglu(self, gate, up):
        shifted = self.add(gate, -16, 32)
        numerator = self.exp(self.signed(shifted, math.log2(math.e), 32, 2))
        denominator = self.positive_add(math.exp(-16), numerator, 2)
        sigmoid = self.mul(numerator, self.reciprocal(denominator))
        return self.signed(self.signed(gate, sigmoid, 16, 1), up, 16, 16)

    def rope(self, left, right, cosine, sine):
        return [self.add(self.signed(left, cosine, 16, 1), self.neg(self.signed(right, sine, 16, 1), 32), 32),
                self.add(self.signed(left, sine, 16, 1), self.signed(right, cosine, 16, 1), 32)]

    def softmax(self, values):
        negative = self.neg(max(values), 32)
        exponentials = [self.exp(self.signed(self.add(value, negative, 32), math.log2(math.e), 32, 2)) for value in values]
        total = exponentials[0]
        for value in exponentials[1:]:
            total = self.positive_add(total, value, 2)
        inverse = self.reciprocal(total)
        return [self.mul(value, inverse) for value in exponentials]


def self_check(lib):
    reports = []

    def check(label, expected, actual):
        state = lib.read_stats()
        if state.error:
            raise AssertionError(f"{label}: checked C primitive rejected")
        identical = len(expected) == len(actual) and all(
            struct.pack("d", left) == struct.pack("d", right)
            for left, right in zip(expected, actual))
        reports.append({"label": label, "values": len(expected), "bitwise_equal": identical,
                        "max_difference": max(abs(left - right) for left, right in zip(expected, actual))})
        if not identical:
            raise AssertionError(f"{label}: Python/C transcription differs")

    generator = random.Random(614)
    for left_bound, right_bound in [(128, 2), (16, 16), (1, 16), (256, 1)]:
        pairs = [(-left_bound, -right_bound), (-left_bound, right_bound), (left_bound, right_bound), (0, 0)]
        pairs += [(generator.uniform(-left_bound, left_bound), generator.uniform(-right_bound, right_bound)) for _ in range(10)]
        for left, right in pairs:
            lib.reset_stats()
            expected = PythonBasis().signed(left, right, left_bound, right_bound)
            actual = lib.bounded_product(left, right, left_bound, right_bound)
            check("product", [expected], [actual])
    for gate in [-16, -8, 0, 8, 16]:
        for up in [-16, 0, 16]:
            lib.reset_stats()
            check("swiglu", [PythonBasis().swiglu(gate, up)], [lib.bounded_swiglu(gate, up)])
    for left, right, angle in [(2.3903015418765046, 1.1409290220926804, 0), (-16, 16, 0),
                               (-16, -16, 1), (16, 16, 2), (0, 0, 3)]:
        lib.reset_stats()
        actual = (D * 2)()
        cosine, sine = math.cos(angle), math.sin(angle)
        lib.bounded_rope(left, right, cosine, sine, actual)
        check("rope", PythonBasis().rope(left, right, cosine, sine), actual)
    vectors = [[-16], [-16, -16], [-16, 16], [0, 0], [-16] * 32,
               [generator.uniform(-16, 16) for _ in range(128)]]
    for values in vectors:
        lib.reset_stats()
        actual = (D * len(values))()
        lib.bounded_softmax(array(values), len(values), actual)
        check("softmax", PythonBasis().softmax(values), actual)
    return {"scope": "Scalar checked Python vs checked C formula transcription",
            "cases": len(reports), "all_bitwise_equal": True, "reports": reports}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tokens(text):
    try:
        result = [int(value) for value in text.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("tokens must be comma-separated integers") from error
    if not result or any(value < 0 for value in result):
        raise argparse.ArgumentTypeError("tokens must be nonempty and nonnegative")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path, nargs="?", help="local F32 Llama GGUF")
    parser.add_argument("--tokens", type=tokens, help="explicit token IDs, e.g. 1,9038; default is model BOS")
    parser.add_argument("--output", type=Path, required=True, help="directory for source hashes, logits and reports")
    parser.add_argument("--native-only", action="store_true", help="run only the independent scalar native reference")
    parser.add_argument("--self-check", action="store_true", help="crosscheck 82 scalar/vector formula cases without a model")
    parser.add_argument("--progress", action="store_true", help="emit operation progress as JSON to stderr")
    args = parser.parse_args()
    if not args.self_check and args.model is None:
        parser.error("a model path or --self-check is required")
    if args.self_check and (args.model is not None or args.tokens is not None or args.native_only):
        parser.error("--self-check does not accept model, tokens or --native-only")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "report.json").exists():
        parser.error("output already contains report.json; select a new directory to preserve evidence")
    lib, command = load_backend(args.output)
    source = Path(__file__).resolve()
    provenance = {"python": sys.version, "compiler_command": command,
                  "source_hashes": {str(source): digest(source), str(source.with_name("bounded_basis.c")): digest(source.with_name("bounded_basis.c"))}}
    if args.self_check:
        report = {**self_check(lib), "provenance": provenance}
        write_json(args.output / "report.json", report)
        print(json.dumps({key: value for key, value in report.items() if key not in ("reports", "provenance")}, indent=2))
        return 0
    model = Model(args.model)
    oracle = Oracle(model, lib, args.output, args.progress)
    token_ids = args.tokens or [model.meta.get("tokenizer.ggml.bos_token_id", 1)]
    if len(token_ids) > model.meta.get("llama.context_length", 0):
        raise ValueError("tokens exceed the model context")
    vocabulary = model.tensors["token_embd.weight"][0][1]
    if any(value >= vocabulary for value in token_ids):
        raise ValueError("token outside source vocabulary")
    write_json(args.output / "weights.json", oracle.inventory)
    start = time.monotonic()
    native = oracle.forward(token_ids, False)
    native_seconds = time.monotonic() - start
    write_json(args.output / "native-logits.json", native)
    native_argmax = max(range(len(native)), key=native.__getitem__)
    result = {"status": "native-only", "native_argmax": native_argmax}
    bounded_seconds = None
    if not args.native_only:
        start = time.monotonic()
        try:
            bounded = oracle.forward(token_ids, True)
            write_json(args.output / "bounded-logits.json", bounded)
            sorted_logits = sorted(native, reverse=True)
            result = {"status": "completed", "native_argmax": native_argmax,
                      "bounded_argmax": max(range(len(bounded)), key=bounded.__getitem__),
                      "native_argmax_margin": sorted_logits[0] - sorted_logits[1],
                      "max_logit_absolute_error": max(abs(left - right) for left, right in zip(native, bounded)),
                      "rms_logit_error": math.sqrt(sum((left - right) ** 2 for left, right in zip(native, bounded)) / len(native))}
        except (ArithmeticError, ValueError) as error:
            result = {"status": "rejected", "error": str(error)}
        bounded_seconds = time.monotonic() - start
    report = {"scope": "Independent Python/C diagnostic, not production Rust or llama.cpp validation",
              "source_model": str(args.model.resolve()), "source_sha256": digest(args.model),
              "operand_bounds": {"rms_input": 16, "rms_weight": 16,
                                 "projection_input": 128, "projection_weight": 2,
                                 "attention_query": 16, "attention_key": 16,
                                 "attention_probability": 1, "attention_value": 16,
                                 "rope_input": 16, "residual_input": 32,
                                 "score_before_scaling": 256, "softmax_score": 16,
                                 "swiglu_gate": 16, "swiglu_up": 16},
              "ids": token_ids, "native_seconds": native_seconds, "bounded_seconds": bounded_seconds,
              "result": result, "total_bml_nodes": oracle.total_nodes, "records": oracle.records,
              "provenance": provenance}
    write_json(args.output / "report.json", report)
    print(json.dumps({key: value for key, value in report.items() if key not in ("records", "provenance")}, indent=2))
    return 2 if result["status"] == "rejected" else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, ArithmeticError, subprocess.CalledProcessError) as error:
        print(f"diagnostic failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
