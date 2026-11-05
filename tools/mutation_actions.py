#!/usr/bin/env python3
"""Bind Actions shards to the prepare job's independent manifest file digest.

The digest must arrive through a trusted job output, never from a downloaded
shard artifact. Candidate, baseline and outcome validation remains in
merge_mutation_shards; this adapter only connects that protocol to Actions.
"""
import argparse
import json
from pathlib import Path

from merge_mutation_shards import (
    file_digest, merge_shards, read_json, require, sha, validate_manifest, write_new,
)


def checked_manifest(path, revision, expected_sha256=None):
    sha(revision, "workflow revision", 40)
    if expected_sha256 is not None:
        sha(expected_sha256, "independent manifest digest", 64)
        require(file_digest(path) == expected_sha256,
                "downloaded manifest differs from prepare job digest")
    manifest = read_json(path)
    validate_manifest(manifest)
    require(manifest["source"]["revision"] == revision,
            "manifest revision differs from workflow snapshot")
    return manifest


def plan_outputs(path, revision):
    manifest = checked_manifest(path, revision)
    matrix = {"include": [{"index": index, "shard": row["shard"]}
                          for index, row in enumerate(manifest["shards"])]}
    return {"manifest_sha256": file_digest(path),
            "matrix": json.dumps(matrix, separators=(",", ":"))}


def merge_directory(manifest, directory, output):
    paths = sorted(directory.rglob("shard-result-*.json"))
    # merge_shards rejects missing/extra/duplicate shards and incomplete reports.
    result = merge_shards(manifest, [read_json(path) for path in paths])
    write_new(output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["publish", "check", "merge"]:
        command = commands.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--revision", required=True)
        if name == "publish":
            command.add_argument("--github-output", type=Path, required=True)
        else:
            command.add_argument("--sha256", required=True)
    commands.choices["check"].add_argument("--shard")
    commands.choices["merge"].add_argument("--results", type=Path, required=True)
    commands.choices["merge"].add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "publish":
            outputs = plan_outputs(args.manifest, args.revision)
            with args.github_output.open("a") as stream:
                for key, value in outputs.items():
                    stream.write(f"{key}={value}\n")
            print(json.dumps(outputs, sort_keys=True))
        else:
            manifest = checked_manifest(args.manifest, args.revision, args.sha256)
            if args.command == "check" and args.shard is not None:
                require(args.shard in validate_manifest(manifest), "unexpected Actions shard")
            if args.command == "merge":
                merge_directory(manifest, args.results, args.output)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Actions mutation evidence rejected: {error}\n")


if __name__ == "__main__":
    main()
