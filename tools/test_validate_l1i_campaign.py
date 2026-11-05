import copy
import unittest

from validate_l1i_campaign import validate_campaign


def valid_report():
    sample = {"affinity_cpu": 2, "warmup_iterations": 100,
              "measurement_iterations": 1000,
              "perf_stat": [
                  {"event": "instructions", "counter-value": "1000000",
                   "event-runtime": 1000, "pcnt-running": 100.00},
                  {"event": "L1-icache-load-misses", "counter-value": "1000",
                   "event-runtime": 1000, "pcnt-running": 100.00}]}
    return {"format_version": 1, "route": "Runtime::execute_sub_fragments",
            "hot_loop_machine_code_bytes": 2533, "affinity_cpu": 2,
            "warmup_iterations": 100, "measurement_iterations": 1000,
            "repetitions": [{"index": index, **copy.deepcopy(sample)}
                            for index in range(1, 8)]}


class L1iCampaignContracts(unittest.TestCase):
    def test_exact_threshold_is_inclusive_for_all_seven_repetitions(self):
        self.assertEqual(len(validate_campaign(valid_report())), 7)

    def test_one_bad_repetition_cannot_hide_in_a_passing_average(self):
        report = valid_report()
        for sample in report["repetitions"]:
            sample["perf_stat"][1]["counter-value"] = "0"
        report["repetitions"][-1]["perf_stat"][1]["counter-value"] = "1001"
        with self.assertRaisesRegex(ValueError, "repetition 7: L1i miss threshold"):
            validate_campaign(report)

    def test_zero_misses_are_valid_but_zero_instructions_are_not(self):
        report = valid_report()
        for sample in report["repetitions"]:
            sample["perf_stat"][1]["counter-value"] = "0"
        validate_campaign(report)
        report["repetitions"][0]["perf_stat"][0]["counter-value"] = "0"
        with self.assertRaisesRegex(ValueError, "zero retired instructions"):
            validate_campaign(report)

    def test_absent_unsupported_and_uncounted_counters_fail(self):
        for field in ["event", "counter-value", "event-runtime", "pcnt-running"]:
            report = valid_report()
            del report["repetitions"][0]["perf_stat"][1][field]
            with self.subTest(missing=field), self.assertRaises(ValueError):
                validate_campaign(report)
        for value in ["<not supported>", "<not counted>", "", "NaN", "Infinity",
                      "-1", "1.5", True]:
            report = valid_report()
            report["repetitions"][0]["perf_stat"][1]["counter-value"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_campaign(report)

    def test_missing_duplicate_or_generic_cache_events_fail(self):
        for events in [[], [{"event": "instructions"}],
                       ["instructions", "instructions"],
                       ["instructions", "cache-misses"]]:
            report = valid_report()
            if events and isinstance(events[0], str):
                for row, event in zip(report["repetitions"][0]["perf_stat"], events):
                    row["event"] = event
            else:
                report["repetitions"][0]["perf_stat"] = events
            with self.subTest(events=events), self.assertRaises(ValueError):
                validate_campaign(report)

    def test_partial_counting_and_zero_runtime_fail(self):
        for event in [0, 1]:
            for field, value in [("pcnt-running", 99.999), ("pcnt-running", 0),
                                 ("pcnt-running", 100.1), ("event-runtime", 0)]:
                report = valid_report()
                report["repetitions"][0]["perf_stat"][event][field] = value
                with self.subTest(event=event, field=field, value=value), self.assertRaises(ValueError):
                    validate_campaign(report)

    def test_scope_must_match_for_both_hardware_events(self):
        report = valid_report()
        for sample in report["repetitions"]:
            for row in sample["perf_stat"]:
                row["event"] += ":u"
        validate_campaign(report)
        report["repetitions"][0]["perf_stat"][1]["event"] = "L1-icache-load-misses"
        with self.assertRaises(ValueError):
            validate_campaign(report)

    def test_affinity_work_and_warmup_are_fixed(self):
        for field, value in [("affinity_cpu", 3), ("warmup_iterations", 0),
                             ("measurement_iterations", 1001), ("index", 1)]:
            report = valid_report()
            report["repetitions"][1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_campaign(report)

    def test_incomplete_or_extra_repetitions_and_unwarmed_campaign_fail(self):
        for count in [0, 6, 8]:
            report = valid_report()
            report["repetitions"] = (report["repetitions"] * 2)[:count]
            with self.subTest(count=count), self.assertRaises(ValueError):
                validate_campaign(report)
        for field in ["warmup_iterations", "measurement_iterations"]:
            report = valid_report()
            report[field] = 0
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_campaign(report)

    def test_bytecode_size_and_other_evaluators_are_not_runtime_machine_code(self):
        for field, value in [("hot_loop_machine_code_bytes", 32768),
                             ("hot_loop_machine_code_bytes", 0),
                             ("route", "RpnProgram::evaluate")]:
            report = valid_report()
            report[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_campaign(report)
        report = valid_report()
        report["bytecode_bytes"] = report.pop("hot_loop_machine_code_bytes")
        with self.assertRaises(ValueError):
            validate_campaign(report)


if __name__ == "__main__":
    unittest.main()
