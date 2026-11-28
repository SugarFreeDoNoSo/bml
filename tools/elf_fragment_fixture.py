#!/usr/bin/env python3
"""Stream a reproducible synthetic F32 Llama GGUF, never trained model data.

The 32/48/260 control and 288/768/32000 representative profile share the same
one-layer, context-eight contract. Output rows and column halves deliberately
differ, so swapping fragment ranges cannot hide behind repeated head weights.
Writes a new file only; its JSON identity/role manifest is emitted on stdout.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import struct
import sys


ALIGNMENT = 32
BLOCK_BYTES = 64 * 1024
MAX_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True)
class Profile:
    width: int
    feed_forward: int
    vocabulary: int
    heads: int


PROFILES = {
    "small": Profile(32, 48, 260, 4),
    "representative": Profile(288, 768, 32000, 6),
}


@dataclass(frozen=True)
class Tensor:
    name: str
    columns: int
    rows: int
    role: str
    salt: int = 0

    @property
    def size_bytes(self):
        return self.columns * self.rows * 4

    @property
    def dimensions(self):
        return [self.columns] if self.role == "norm" else [self.columns, self.rows]


def tensors(profile):
    width, ffn, vocab = profile.width, profile.feed_forward, profile.vocabulary
    result = [Tensor("token_embd.weight", width, vocab, "embedding"),
              Tensor("output.weight", width, vocab, "head")]
    result.extend(Tensor(name, width, 1, "norm") for name in (
        "output_norm.weight", "blk.0.attn_norm.weight", "blk.0.ffn_norm.weight"))
    result.extend(Tensor(name, columns, rows, "projection", salt)
                  for name, columns, rows, salt in (
                      ("blk.0.attn_q.weight", width, width, 1),
                      ("blk.0.attn_k.weight", width, width, 3),
                      ("blk.0.attn_v.weight", width, width, 5),
                      ("blk.0.attn_output.weight", width, width, 7),
                      ("blk.0.ffn_gate.weight", width, ffn, 9),
                      ("blk.0.ffn_up.weight", width, ffn, 11),
                      ("blk.0.ffn_down.weight", ffn, width, 13)))
    return result


def signed_pattern(index):
    value = index % 16
    return value - 8 if value < 8 else value - 7


def weight(tensor, row, column):
    if not 0 <= row < tensor.rows or not 0 <= column < tensor.columns:
        raise ValueError("weight coordinate outside tensor")
    if tensor.role == "norm":
        return 1.0
    if tensor.role == "embedding":
        # Unique token rows, including rows whose periodic pattern is equal.
        return (signed_pattern(row * 3 + column * 5) / 64.0
                + (2 * row - (tensor.rows - 1)) / 262144.0)
    if tensor.role == "head":
        # Every term is an exact binary fraction and the sum fits F32 exactly.
        # Row/column half offsets exceed any periodic-pattern cancellation.
        return (signed_pattern(row * 3 + column * 5) / 512.0
                + row / 262144.0
                + (0.125 if row >= tensor.rows // 2 else -0.125)
                + (0.03125 if column >= tensor.columns // 2 else -0.03125))
    if tensor.role == "projection":
        return signed_pattern(row * 5 + column * 3 + tensor.salt) / 1024.0
    raise ValueError("unknown tensor role")


def token(index):
    if index < 4:
        return ("<unk>", "<s>", "</s>", "▁")[index]
    return f"<0x{index - 4:02X}>" if index < 260 else f"▁synthetic_{index:05}"


def token_type(index):
    return 2 if index == 0 else 3 if index in (1, 2) else 6 if 4 <= index < 260 else 1


class Writer:
    """A counting preflight sink, or a quota-enforced hashing file writer."""

    def __init__(self, file=None, limit=MAX_BYTES):
        self.file, self.limit, self.count = file, limit, 0
        self.digest = hashlib.sha256() if file is not None else None

    def write(self, data):
        if self.count + len(data) > self.limit:
            raise ValueError("synthetic GGUF exceeds byte quota")
        if self.file is not None:
            if self.file.write(data) != len(data):
                raise OSError("short synthetic GGUF write")
            self.digest.update(data)
        self.count += len(data)

    def number(self, kind, value):
        self.write(struct.pack("<" + kind, value))

    def string(self, value):
        data = value.encode("utf-8")
        self.number("Q", len(data))
        self.write(data)

    def key(self, name, kind):
        self.string(name)
        self.number("I", kind)


def align(value):
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def write_header(writer, profile, specs):
    integers = (
        ("general.alignment", ALIGNMENT), ("llama.block_count", 1),
        ("llama.attention.head_count", profile.heads),
        ("llama.attention.head_count_kv", profile.heads),
        ("llama.embedding_length", profile.width),
        ("llama.feed_forward_length", profile.feed_forward),
        ("llama.context_length", 8),
        ("llama.rope.dimension_count", profile.width // profile.heads),
        ("llama.vocab_size", profile.vocabulary),
        ("tokenizer.ggml.bos_token_id", 1), ("tokenizer.ggml.eos_token_id", 2),
        ("tokenizer.ggml.unknown_token_id", 0))
    booleans = (("tokenizer.ggml.add_bos_token", True),
                ("tokenizer.ggml.add_eos_token", False),
                ("tokenizer.ggml.add_space_prefix", True),
                ("tokenizer.ggml.remove_extra_whitespaces", False),
                ("tokenizer.ggml.byte_fallback", True))
    writer.write(b"GGUF")
    writer.number("I", 3)
    writer.number("Q", len(specs))
    writer.number("Q", len(integers) + len(booleans) + 7)
    for name in ("general.architecture", "tokenizer.ggml.model"):
        writer.key(name, 8)
        writer.string("llama")
    for name, value in integers:
        writer.key(name, 4)
        writer.number("I", value)
    for name, value in booleans:
        writer.key(name, 7)
        writer.number("?", value)
    for name, value in (("llama.attention.layer_norm_rms_epsilon", 0.5),
                        ("llama.rope.freq_base", 16.0)):
        writer.key(name, 6)
        writer.number("f", value)
    for name, kind in (("tokenizer.ggml.tokens", 8),
                       ("tokenizer.ggml.scores", 6),
                       ("tokenizer.ggml.token_type", 5)):
        writer.key(name, 9)
        writer.number("I", kind)
        writer.number("Q", profile.vocabulary)
        for index in range(profile.vocabulary):
            if kind == 8:
                writer.string(token(index))
            else:
                writer.number("f" if kind == 6 else "i",
                              -1.0 if kind == 6 else token_type(index))
    offsets, end = [], 0
    for tensor in specs:
        offset = align(end)
        offsets.append(offset)
        end = offset + tensor.size_bytes
        writer.string(tensor.name)
        writer.number("I", len(tensor.dimensions))
        for dimension in tensor.dimensions:
            writer.number("Q", dimension)
        writer.number("I", 0)  # GGML F32, axis zero contiguous.
        writer.number("Q", offset)
    writer.write(b"\0" * (align(writer.count) - writer.count))
    return offsets, end


def planned_layout(profile_name, max_bytes=MAX_BYTES):
    if type(max_bytes) is not int or not 0 < max_bytes <= MAX_BYTES:
        raise ValueError("byte quota must be positive and at most 128 MiB")
    if profile_name not in PROFILES:
        raise ValueError("unknown synthetic profile")
    profile = PROFILES[profile_name]
    specs = tensors(profile)
    counter = Writer(limit=max_bytes)
    offsets, payload = write_header(counter, profile, specs)
    total = counter.count + payload
    if total > max_bytes:
        raise ValueError("synthetic GGUF exceeds byte quota")
    return profile, specs, offsets, counter.count, total


def write_payload(writer, tensor):
    block = bytearray(BLOCK_BYTES)
    digest = hashlib.sha256()
    elements = tensor.columns * tensor.rows
    for start in range(0, elements, BLOCK_BYTES // 4):
        count = min(elements - start, BLOCK_BYTES // 4)
        for index in range(count):
            row, column = divmod(start + index, tensor.columns)
            struct.pack_into("<f", block, 4 * index, weight(tensor, row, column))
        payload = memoryview(block)[:count * 4]
        writer.write(payload)
        digest.update(payload)
    return digest.hexdigest()


def generate(output, profile_name, max_bytes=MAX_BYTES):
    profile, specs, offsets, data_start, total = planned_layout(profile_name, max_bytes)
    output = Path(output)
    # Exclusive creation rejects existing files, directories and symlinks.
    with output.open("xb") as file:
        owned = os.fstat(file.fileno())
        try:
            writer = Writer(file, max_bytes)
            actual_offsets, _ = write_header(writer, profile, specs)
            if writer.count != data_start or actual_offsets != offsets:
                raise ValueError("GGUF header differs from preflight")
            roles = []
            for tensor, offset in zip(specs, offsets):
                writer.write(b"\0" * (data_start + offset - writer.count))
                sha = write_payload(writer, tensor)
                roles.append({"name": tensor.name, "role": tensor.role,
                              "dimensions": tensor.dimensions, "dtype": "F32",
                              "data_offset": offset, "size_bytes": tensor.size_bytes,
                              "sha256": sha})
            file.flush()
            if writer.count != total or os.fstat(file.fileno()).st_size != total:
                raise ValueError("GGUF size differs from preflight")
        except BaseException:
            # Remove only the file created here, never a replacement at its path.
            try:
                current = output.lstat()
                if (current.st_dev, current.st_ino) == (owned.st_dev, owned.st_ino):
                    output.unlink()
            except FileNotFoundError:
                pass
            raise
    return {"schema": "bml-elf-fragment-fixture-v1", "synthetic": True,
            "trained_model": False, "profile": profile_name,
            "gguf_version": 3, "architecture": "llama", "context_length": 8,
            "embedding_length": profile.width, "feed_forward_length": profile.feed_forward,
            "vocabulary": profile.vocabulary, "head_count": profile.heads,
            "head_count_kv": profile.heads, "alignment": ALIGNMENT,
            "data_start": data_start, "size_bytes": total,
            "sha256": writer.digest.hexdigest(), "tensor_roles": roles}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new GGUF path")
    parser.add_argument("--profile", choices=PROFILES, required=True)
    args = parser.parse_args(argv)
    try:
        result = generate(args.output, args.profile)
    except (OSError, ValueError) as error:
        print(f"synthetic fixture: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
