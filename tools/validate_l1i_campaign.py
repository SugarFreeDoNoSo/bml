#!/usr/bin/env python3
"""Validate seven perf-stat JSON samples against the fixed Runtime L1i protocol.

Usage: python3 tools/validate_l1i_campaign.py campaign.json

The JSON envelope has format_version=1, route="Runtime::execute_sub_fragments",
hot_loop_machine_code_bytes, affinity_cpu, warmup_iterations,
measurement_iterations, and repetitions. Each repetition repeats the affinity
and iteration fields, has index=1..7, and embeds the two original perf-stat JSON
counter objects in perf_stat. Required counter keys are event, counter-value,
event-runtime and pcnt-running. Select instructions and L1-icache-load-misses,
either both unqualified or both qualified with :u.

Counters use LC_ALL=C numeric spelling. Missing/unsupported/not-counted events,
zero retired instructions, partial counting and a per-repetition rate above one
miss per 1,000 retired instructions fail. Zero measured misses are valid.

This validates reported counter data only: the real Runtime driver, warmup
interval, disassembly, event meaning on the target CPU and provenance still
require independent audit. No perf process or workload is launched here.
"""
import argparse
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path

REPETITIONS = 7
HOT_LOOP_LIMIT = 32 * 1024
ROUTE = "Runtime::execute_sub_fragments"
EVENTS = {"instructions", "L1-icache-load-misses"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive_integer(value, label, minimum=1):
    require(type(value) is int and value >= minimum, f"invalid {label}")
    return value


def number(value, label):
    require(isinstance(value, (str, int, float)) and not isinstance(value, bool),
            f"invalid {label}")
    try:
        result = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError(f"invalid {label}: unsupported or uncounted counter") from error
    require(result.is_finite() and result >= 0, f"invalid {label}")
    return result


def counters(perf_stat, label):
    require(isinstance(perf_stat, list) and len(perf_stat) == 2,
            f"{label}: exactly two counters required")
    result = {}
    scopes = set()
    for row in perf_stat:
        require(isinstance(row, dict), f"{label}: invalid counter record")
        selector = row.get("event")
        require(isinstance(selector, str), f"{label}: missing event name")
        event, separator, scope = selector.partition(":")
        require(event in EVENTS and scope in {"", "u"} and not (separator and not scope),
                f"{label}: unsupported event selector {selector!r}")
        require(event not in result, f"{label}: duplicate {event}")
        scopes.add(scope)
        value = number(row.get("counter-value"), f"{label}: {event} value")
        require(value == value.to_integral_value(), f"{label}: fractional event count")
        require(number(row.get("event-runtime"), f"{label}: {event} runtime") > 0,
                f"{label}: zero event runtime")
        require(number(row.get("pcnt-running"), f"{label}: {event} running percent") == 100,
                f"{label}: partial or multiplexed counter")
        result[event] = int(value)
    require(set(result) == EVENTS and len(scopes) == 1,
            f"{label}: missing counters or mismatched event scopes")
    require(result["instructions"] > 0, f"{label}: zero retired instructions")
    return result


def validate_campaign(report):
    """Return accepted per-repetition counts; raise ValueError on invalid data."""
    require(isinstance(report, dict), "campaign must be a JSON object")
    require(type(report.get("format_version")) is int and report["format_version"] == 1,
            "unsupported campaign format")
    require(report.get("route") == ROUTE, "real Runtime subfragment route required")
    size = positive_integer(report.get("hot_loop_machine_code_bytes"), "machine-code size")
    require(size < HOT_LOOP_LIMIT, "hot-loop machine code must be smaller than 32 KiB")
    cpu = positive_integer(report.get("affinity_cpu"), "affinity CPU", minimum=0)
    warmup = positive_integer(report.get("warmup_iterations"), "warmup iterations")
    iterations = positive_integer(report.get("measurement_iterations"), "measurement iterations")
    repetitions = report.get("repetitions")
    require(isinstance(repetitions, list) and len(repetitions) == REPETITIONS,
            "exactly seven measured repetitions required")
    accepted = []
    for index, sample in enumerate(repetitions, start=1):
        label = f"repetition {index}"
        require(isinstance(sample, dict), f"{label}: invalid sample")
        require(type(sample.get("index")) is int and sample["index"] == index,
                f"{label}: missing, duplicated or reordered repetition")
        for field, expected in [("affinity_cpu", cpu), ("warmup_iterations", warmup),
                                ("measurement_iterations", iterations)]:
            require(type(sample.get(field)) is int and sample[field] == expected,
                    f"{label}: changed {field}")
        values = counters(sample.get("perf_stat"), label)
        require(values["L1-icache-load-misses"] * 1000 <= values["instructions"],
                f"{label}: L1i miss threshold exceeded")
        accepted.append({"index": index, **values})
    return accepted


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    try:
        report = json.loads(args.report.read_text())
        accepted = validate_campaign(report)
    except (OSError, ValueError) as error:
        parser.exit(1, f"L1i counter protocol failed: {error}\n")
    print(json.dumps({"counter_protocol": "passed", "repetitions": accepted,
                      "qualification": "Counter data only; Runtime driver and provenance require audit."}))


if __name__ == "__main__":
    main()
