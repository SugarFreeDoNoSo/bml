#!/usr/bin/env python3
"""List the ordinary CI mutation scope, then run and score it without exclusions.

N/A requires a successful JSON listing of zero candidates or an exact, successful
27.1.0 diff-filter zero result. Raw output and that derivation are retained; the
N/A report is not a cargo-mutants campaign or a successful baseline. Full mode
never applies the development diff and always requires a JSON listing.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys

from merge_mutation_shards import TOOL_VERSION, candidate_map, file_digest, require, write_new
import quality

ROOT = Path(__file__).resolve().parents[1]


def listed_candidates(stdout, stderr, full):
    if stdout.strip():
        candidates = json.loads(stdout)
        candidate_map(candidates)
        return candidates, {"kind": "json_listing"}
    # 27.1.0 returns before printing JSON for these three *successful* diff
    # filter results. Neither an arbitrary blank response nor an error is N/A.
    plain = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", stderr)
    message = re.fullmatch(
        r"INFO[ \t]+(Diff file is empty|Diff changes no Rust source files|"
        r"No mutants to filter)", plain.strip())
    require(not full and message is not None, "missing JSON candidate listing")
    return [], {"kind": "cargo_mutants_27_1_diff_filter", "message": message.group(1)}


def revision(root, reference):
    return subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{reference}^{{commit}}"], cwd=root,
        text=True).strip()


def run_scope(root, base, full=False):
    reports = root / "target/quality"
    reports.mkdir(parents=True, exist_ok=True)
    outcomes = root / "mutants.out/outcomes.json"
    require(not outcomes.parent.exists(), "stale mutation output directory exists")
    base = revision(root, base)
    head = revision(root, "HEAD")
    diff = reports / "changes.diff"
    subprocess.run([sys.executable, str(root / "tools/mutation_diff.py"), base, str(diff)],
                   cwd=root, check=True)
    diff_sha256 = file_digest(diff)
    version = subprocess.check_output(["cargo", "mutants", "--version"], cwd=root,
                                      text=True).strip()
    require(version == f"cargo-mutants {TOOL_VERSION}", "unexpected cargo-mutants version")
    scope = [] if full else ["--in-diff", str(diff)]
    listing = ["cargo", "mutants", "--workspace", "--cargo-arg=--locked",
               "--list", "--json", *scope]
    listed = subprocess.run(listing, cwd=root, capture_output=True, text=True)
    with (reports / "ci-mutation-list.stdout").open("x") as stream:
        stream.write(listed.stdout)
    with (reports / "ci-mutation-list.stderr").open("x") as stream:
        stream.write(listed.stderr)
    require(listed.returncode == 0,
            f"candidate listing failed ({listed.returncode}): {listed.stderr}")
    candidates, proof = listed_candidates(listed.stdout, listed.stderr, full)
    require(file_digest(diff) == diff_sha256, "mutation diff changed during listing")
    require(proof.get("message") != "Diff file is empty" or diff.stat().st_size == 0,
            "CLI empty-diff result contradicts the recorded diff")
    candidates_file = reports / "ci-mutation-candidates.json"
    write_new(candidates_file, candidates)
    require(full or diff.stat().st_size != 0 or not candidates,
            "empty development diff unexpectedly selected mutants")
    require(revision(root, "HEAD") == head, "revision changed during candidate listing")
    selection = {"schema_version": 1, "source_revision": head, "base_revision": base,
                 "scope": "workspace" if full else "development_diff",
                 "diff_applied": not full, "diff_sha256": diff_sha256,
                 "cargo_mutants_version": TOOL_VERSION, "listing_command": listing,
                 "listing_exit_code": listed.returncode, "candidate_count": len(candidates),
                 "selection_proof": proof,
                 "candidates_sha256": file_digest(candidates_file),
                 "listed_at": datetime.now(timezone.utc).isoformat()}
    selection_file = reports / "ci-mutation-selection.json"
    if not candidates:
        selection.update(status="not_applicable", baseline_runs=0,
                         reason="cargo-mutants successfully listed zero selected candidates")
        write_new(selection_file, selection)
        outcomes = reports / "ci-mutation-empty-outcomes.json"
        write_new(outcomes, {"kind": "verified_empty_selection", "selection": selection,
                             "total_mutants": 0, "outcomes": [],
                             "end_time": selection["listed_at"]})
    else:
        # Preserve a listing even if cargo is interrupted or returns an error.
        write_new(selection_file, selection)
        command = ["cargo", "mutants", "--workspace", "--cargo-arg=--locked",
                   "--test-workspace", "true", "--jobs", "2", "--timeout", "30",
                   "--build-timeout", "180", *scope]
        result = subprocess.run(command, cwd=root)
        write_new(reports / "ci-mutation-execution.json",
                  {"command": command, "exit_code": result.returncode})
        data = json.loads(outcomes.read_text())
        names = [row["scenario"]["Mutant"].get("name") for row in data.get("outcomes", [])
                 if isinstance(row.get("scenario"), dict) and "Mutant" in row["scenario"]]
        require(len(names) == len(candidates) and len(set(names)) == len(names)
                and set(names) == set(candidate_map(candidates)),
                "mutation outcomes do not match the nonempty listed selection")
        # The original gate must still reject missing/incomplete reports,
        # baseline failures, survivors and timeouts, including cargo exit 2/3.
        quality.mutations(outcomes)
        require(result.returncode == 0, f"cargo-mutants exited with {result.returncode}")
        return
    quality.mutations(outcomes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    try:
        run_scope(ROOT, args.base, args.full)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, f"CI mutation scope failed: {error}\n")


if __name__ == "__main__":
    main()
