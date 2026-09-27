"""Regression tests for quality gates that must fail closed."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import quality

class MutationGate(unittest.TestCase):
    def run_report(self,statuses,*,total=None,completed=True,baseline='Success'):
        with tempfile.TemporaryDirectory() as directory:
            original=quality.REPORTS; quality.REPORTS=Path(directory);path=Path(directory)/'outcomes.json'
            outcomes=[{'scenario':'Baseline','summary':baseline}]
            outcomes += [{'scenario':{'Mutant':{'name':str(i)}},'summary':s} for i,s in enumerate(statuses)]
            path.write_text(json.dumps({'outcomes':outcomes,'total_mutants':len(statuses) if total is None else total,'end_time':'done' if completed else None}))
            try:
                with contextlib.redirect_stdout(io.StringIO()): quality.mutations(path)
            finally: quality.REPORTS=original

    def test_detected_pass_but_survivors_and_timeouts_fail(self):
        self.run_report(['CaughtMutant','Unviable'])
        for outcome in ['MissedMutant','Timeout','Unknown']:
            with self.subTest(outcome=outcome),self.assertRaises(SystemExit):self.run_report([outcome])

    def test_incomplete_or_failed_baseline_cannot_pass(self):
        for args in [{'total':2},{'completed':False},{'baseline':'Failure'}]:
            with self.subTest(args=args),self.assertRaises(SystemExit):self.run_report(['CaughtMutant'],**args)

    def test_empty_campaign_is_not_a_detection_claim(self):self.run_report([])

class TraceabilityGate(unittest.TestCase):
    def run_mapping(self, mapping, feature, listed):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'quality').mkdir()
            (root / 'quality/contracts.feature').write_text(feature)
            (root / 'quality/traceability.json').write_text(json.dumps(mapping))
            def list_tests(args, **kwargs):
                suite = (args[3], args[5])
                return SimpleNamespace(stdout='\n'.join(name + ': test' for name in listed[suite]))
            with patch.object(quality, 'ROOT', root), patch.object(quality, 'REPORTS', root / 'reports'), patch.object(quality, 'command', side_effect=list_tests), contextlib.redirect_stdout(io.StringIO()):
                quality.trace()
                return json.loads((root / 'reports/traceability.json').read_text())

    def test_test_names_are_scoped_to_their_cargo_suite(self):
        mapping = {'C01': {'package': 'compiler', 'target': 'contracts', 'tests': ['same_name']},
                   'C07': {'package': 'domain', 'target': 'tensor', 'tests': ['same_name']}}
        listed = {('compiler', 'contracts'): ['same_name'], ('domain', 'tensor'): ['same_name']}
        report = self.run_mapping(mapping, '@C01\n@C07', listed)
        self.assertEqual(report['mapped'], 2)
        self.assertEqual(set(report['tests']), {'compiler::contracts', 'domain::tensor'})
        listed[('domain', 'tensor')] = ['different_test']
        with self.assertRaises(SystemExit):
            self.run_mapping(mapping, '@C01\n@C07', listed)

    def test_missing_empty_or_duplicate_scenarios_fail(self):
        mapping = {'C07': {'package': 'domain', 'target': 'tensor', 'tests': []}}
        for feature in ['@C07', '@C07\n@C07', '@C08']:
            with self.subTest(feature=feature), self.assertRaises(SystemExit):
                self.run_mapping(mapping, feature, {('domain', 'tensor'): ['some_test']})

if __name__=='__main__':unittest.main()
