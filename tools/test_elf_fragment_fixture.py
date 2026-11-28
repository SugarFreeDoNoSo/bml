"""Independent format/numerical controls; no trained model or large GGUF build."""

import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import elf_fragment_fixture as fixture


def read_small(data):
    """Independent bounded reader for the fixture's GGUF v3/F32 subset."""
    position = 0

    def take(count):
        nonlocal position
        if count < 0 or position + count > len(data):
            raise ValueError("truncated GGUF")
        result = data[position:position + count]
        position += count
        return result

    def number(kind):
        return struct.unpack("<" + kind, take(struct.calcsize("<" + kind)))[0]

    def string():
        size = number("Q")
        if size > 4096:
            raise ValueError("string exceeds small-fixture reader quota")
        return take(size).decode("utf-8")

    def value(kind):
        if kind == 8:
            return string()
        if kind in (4, 5, 6):
            return number({4: "I", 5: "i", 6: "f"}[kind])
        if kind == 7:
            boolean = number("B")
            if boolean not in (0, 1):
                raise ValueError("noncanonical boolean")
            return bool(boolean)
        if kind == 9:
            subtype, count = number("I"), number("Q")
            if subtype not in (5, 6, 8) or count > 260:
                raise ValueError("array exceeds small-fixture reader contract")
            return [value(subtype) for _ in range(count)]
        raise ValueError("unsupported metadata type")

    if take(4) != b"GGUF" or number("I") != 3:
        raise ValueError("not GGUF v3")
    count, kv_count = number("Q"), number("Q")
    if count != 12 or kv_count > 32:
        raise ValueError("unexpected model size")
    metadata = {}
    for _ in range(kv_count):
        name = string()
        if name in metadata:
            raise ValueError("duplicate metadata")
        metadata[name] = value(number("I"))
    tensors = {}
    end = 0
    for _ in range(count):
        name, rank = string(), number("I")
        if name in tensors or rank not in (1, 2):
            raise ValueError("duplicate tensor or unsupported rank")
        dims = [number("Q") for _ in range(rank)]
        if not all(0 < n <= 260 for n in dims) or number("I") != 0:
            raise ValueError("unsupported shape or encoding")
        offset = number("Q")
        size = math.prod(dims) * 4
        if offset % 32 or offset < end:
            raise ValueError("misaligned or overlapping tensor")
        tensors[name] = {"dimensions": dims, "offset": offset, "size": size}
        end = offset + size
    start = (position + 31) // 32 * 32
    if start + end != len(data) or any(data[position:start]):
        raise ValueError("invalid padding or exact file size")
    for tensor in tensors.values():
        payload = data[start + tensor["offset"]:start + tensor["offset"] + tensor["size"]]
        tensor["payload"] = payload
        tensor["values"] = struct.unpack("<" + "f" * (len(payload) // 4), payload)
        if not all(math.isfinite(v) for v in tensor["values"]):
            raise ValueError("nonfinite tensor")
    return metadata, tensors, start


class FixtureTests(unittest.TestCase):
    def test_small_gguf_roles_shapes_and_sentencepiece_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "small.gguf"
            report = fixture.generate(path, "small")
            data = path.read_bytes()
        metadata, tensors, start = read_small(data)
        self.assertEqual(metadata["general.architecture"], "llama")
        self.assertEqual(metadata["tokenizer.ggml.model"], "llama")
        self.assertNotIn("tokenizer.ggml.pre", metadata)
        self.assertTrue(metadata["tokenizer.ggml.byte_fallback"])
        self.assertTrue(metadata["tokenizer.ggml.add_bos_token"])
        self.assertFalse(metadata["tokenizer.ggml.add_eos_token"])
        self.assertEqual(metadata["llama.context_length"], 8)
        tokens = metadata["tokenizer.ggml.tokens"]
        self.assertEqual(tokens[:4], ["<unk>", "<s>", "</s>", "▁"])
        self.assertEqual(tokens[4:], [f"<0x{b:02X}>" for b in range(256)])
        self.assertEqual(metadata["tokenizer.ggml.token_type"], [2, 3, 3, 1] + [6] * 256)
        expected_shapes = {
            "token_embd.weight": [32, 260], "output.weight": [32, 260],
            "output_norm.weight": [32], "blk.0.attn_norm.weight": [32],
            "blk.0.ffn_norm.weight": [32], "blk.0.attn_q.weight": [32, 32],
            "blk.0.attn_k.weight": [32, 32], "blk.0.attn_v.weight": [32, 32],
            "blk.0.attn_output.weight": [32, 32], "blk.0.ffn_gate.weight": [32, 48],
            "blk.0.ffn_up.weight": [32, 48], "blk.0.ffn_down.weight": [48, 32]}
        self.assertEqual({n: t["dimensions"] for n, t in tensors.items()}, expected_shapes)
        self.assertEqual(report["data_start"], start)
        self.assertEqual(report["size_bytes"], len(data))
        self.assertEqual(report["sha256"], hashlib.sha256(data).hexdigest())
        for role in report["tensor_roles"]:
            tensor = tensors[role["name"]]
            self.assertEqual(role["sha256"], hashlib.sha256(tensor["payload"]).hexdigest())
            self.assertEqual(role["size_bytes"], len(tensor["payload"]))
        for bad in (b"BAD!" + data[4:], data[:-1], data + b"\0"):
            with self.assertRaises(ValueError):
                read_small(bad)

    def test_head_has_distinct_row_and_column_halves_with_exact_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "small.gguf"
            fixture.generate(path, "small")
            _, tensors, _ = read_small(path.read_bytes())
        values = tensors["output.weight"]["values"]
        rows = [values[i * 32:(i + 1) * 32] for i in range(260)]
        self.assertEqual(rows[0][0], -0.171875)
        self.assertEqual(rows[0][16], -0.109375)
        self.assertEqual(rows[130][0], 23682 / 262144)
        self.assertTrue(all(abs(v) <= 0.5 for v in values))
        self.assertTrue(all(a != b for a, b in zip(rows[:130], rows[130:])))
        self.assertTrue(all(row[:16] != row[16:] for row in rows))
        # Dense positive activation: a half-range swap changes projected logits.
        logits = [sum(v / 32.0 for v in row) for row in rows]
        self.assertTrue(all(a != b for a, b in zip(logits[:130], logits[130:])))
        embeddings = tensors["token_embd.weight"]["values"]
        self.assertEqual(len({embeddings[i * 32:(i + 1) * 32] for i in range(260)}), 260)
        self.assertEqual(tensors["output_norm.weight"]["values"], (1.0,) * 32)

    def test_cli_reproducibility_identity_and_no_overwrite(self):
        script = Path(fixture.__file__)
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "a.gguf", Path(directory) / "b.gguf"
            cmd = [sys.executable, str(script), "--profile", "small", "--output"]
            a = subprocess.run(cmd + [str(first)], capture_output=True, text=True, check=True)
            b = subprocess.run(cmd + [str(second)], capture_output=True, text=True, check=True)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(json.loads(a.stdout), json.loads(b.stdout))
            report = json.loads(a.stdout)
            self.assertTrue(report["synthetic"])
            self.assertFalse(report["trained_model"])
            self.assertEqual(report["sha256"], hashlib.sha256(first.read_bytes()).hexdigest())
            before = first.read_bytes()
            rejected = subprocess.run(cmd + [str(first)], capture_output=True, text=True)
            self.assertEqual(rejected.returncode, 1)
            self.assertEqual(rejected.stdout, "")
            self.assertEqual(first.read_bytes(), before)

    def test_quota_is_inclusive_and_failure_precedes_file_creation(self):
        total = fixture.planned_layout("small")[-1]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "small.gguf"
            for limit in (0, -1, fixture.MAX_BYTES + 1, total - 1, True):
                with self.subTest(limit=limit), self.assertRaises(ValueError):
                    fixture.generate(path, "small", max_bytes=limit)
                self.assertFalse(path.exists())
            report = fixture.generate(path, "small", max_bytes=total)
            self.assertEqual(report["size_bytes"], total)
        with self.assertRaises(ValueError):
            fixture.planned_layout("unknown")

    def test_failed_payload_removes_partial_but_never_existing_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial.gguf"
            with patch.object(fixture, "write_payload", side_effect=OSError("injected write failure")):
                with self.assertRaisesRegex(OSError, "injected write failure"):
                    fixture.generate(path, "small")
            self.assertFalse(path.exists())
            target = Path(directory) / "original"
            target.write_bytes(b"keep")
            path.symlink_to(target)
            with self.assertRaises(FileExistsError):
                fixture.generate(path, "small")
            self.assertTrue(path.is_symlink())
            self.assertEqual(target.read_bytes(), b"keep")

    def test_representative_plan_and_unique_vocabulary_without_payload(self):
        profile, specs, offsets, start, total = fixture.planned_layout("representative")
        self.assertEqual((profile.width, profile.feed_forward, profile.vocabulary, profile.heads),
                         (288, 768, 32000, 6))
        self.assertEqual(sum(t.size_bytes for t in specs), 77712768)
        self.assertLess(total, 128 * 1024 * 1024)
        self.assertEqual(total, start + offsets[-1] + specs[-1].size_bytes)
        self.assertEqual(len({fixture.token(i) for i in range(32000)}), 32000)
        head = next(t for t in specs if t.name == "output.weight")
        for row in (0, 1, 15999):
            for col in (0, 143, 287):
                self.assertNotEqual(fixture.weight(head, row, col), fixture.weight(head, row + 16000, col))
        self.assertLessEqual(abs(fixture.weight(head, 31999, 287)), 0.5)

    def test_payload_serialization_is_block_bounded(self):
        class Counter:
            def __init__(self):
                self.blocks = []

            def write(self, block):
                self.blocks.append(len(block))
                return len(block)

        output = Counter()
        tensor = fixture.Tensor("output.weight", 32, 1025, "head")
        writer = fixture.Writer(output, tensor.size_bytes)
        fixture.write_payload(writer, tensor)
        self.assertEqual(output.blocks, [65536, 65536, 128])
        self.assertEqual(writer.count, tensor.size_bytes)


if __name__ == "__main__":
    unittest.main()
