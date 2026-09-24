"""Regression tests for quality gates that must fail closed."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
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

if __name__=='__main__':unittest.main()
