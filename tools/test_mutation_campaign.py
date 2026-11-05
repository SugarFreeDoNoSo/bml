"""Exercise the full campaign's real runner/telemetry shell lifecycle."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CampaignTelemetry(unittest.TestCase):
    def test_runner_subtree_status_and_log_separation_on_success_and_failure(self):
        workflow = (ROOT / '.github/workflows/mutation-campaign.yml').read_text()
        start = workflow.index('          timeout --signal=TERM')
        end = workflow.index('          df -h . > "$evidence/disk-after.txt"', start)
        shell = textwrap.dedent(workflow[start:end])
        for exit_code in (0, 2, 3, 124, 143):
            with self.subTest(exit_code=exit_code), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                evidence, bin_dir = root / 'evidence', root / 'bin'
                evidence.mkdir()
                bin_dir.mkdir()
                # Accelerate only the test monitor's interval; run the existing
                # generic telemetry CLI and its real /proc subtree sampler.
                monitor = bin_dir / 'python3'
                monitor.write_text(
                    f'#!{sys.executable}\n'
                    'import os, sys, time\n'
                    'from pathlib import Path\n'
                    f'sys.path.insert(0, {str(ROOT / "tools")!r})\n'
                    'import mutation_subshards as sub\n'
                    'sleep = time.sleep\n'
                    'sample = sub.diagnostic_sample\n'
                    'evidence = Path(os.environ["evidence"])\n'
                    'observed = False\n'
                    'def sampled(pid, root):\n'
                    '    global observed\n'
                    '    result = sample(pid, root)\n'
                    '    child = evidence / "child-pid.txt"\n'
                    '    if child.exists() and int(child.read_text()) in {p["pid"] for p in result["processes"]}:\n'
                    '        observed = True\n'
                    '    return result\n'
                    'sub.diagnostic_sample = sampled\n'
                    # telemetry sleeps after flushing the sample to file/stdout.
                    'def pause(_):\n'
                    '    if observed:\n'
                    '        (evidence / "release").touch()\n'
                    '    sleep(0.01)\n'
                    'sub.time.sleep = pause\n'
                    'sys.argv = sys.argv[1:]\n'
                    'sub.main()\n'
                )
                monitor.chmod(0o755)
                (evidence / 'run-shard.sh').write_text(
                    'printf "runner stdout\\n"\n'
                    'printf "runner stderr\\n" >&2\n'
                    'bash -c \'while ! test -f "$evidence/release"; do sleep 0.01; done\' &\n'
                    'printf "%s\\n" "$!" > "$evidence/child-pid.txt"\n'
                    'wait "$!"\n'
                    f'exit {exit_code}\n'
                )
                result = subprocess.run(
                    ['bash', '-e', '-o', 'pipefail', '-c', shell +
                     '\nprintf "%s %s\\n" "$runner_pid" "$telemetry_pid" > "$evidence/pids.txt"\n'],
                    cwd=ROOT, env={**os.environ, 'evidence': str(evidence),
                                   'PATH': str(bin_dir) + os.pathsep + os.environ['PATH']},
                    capture_output=True, text=True, timeout=10,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual((evidence / 'exit-code.txt').read_text(), f'{exit_code}\n')
                self.assertEqual((evidence / 'runner.log').read_text(), 'runner stdout\nrunner stderr\n')
                self.assertEqual(result.stderr, '')
                self.assertEqual(result.stdout, (evidence / 'telemetry.jsonl').read_text())
                lines = result.stdout.splitlines()
                self.assertTrue(lines)
                samples = [json.loads(line) for line in lines]
                runner_pid, monitor_pid = map(int, (evidence / 'pids.txt').read_text().split())
                child_pid = int((evidence / 'child-pid.txt').read_text())
                self.assertTrue(all(sample['root_pid'] == runner_pid for sample in samples))
                self.assertTrue(any(child_pid in {row['pid'] for row in sample['processes']}
                                    for sample in samples))
                self.assertTrue(all(len(line.encode()) <= 32768 for line in lines))
                for pid in (runner_pid, monitor_pid):
                    with self.assertRaises(ProcessLookupError):
                        os.kill(pid, 0)
