"""Container lifecycle regressions with no daemon or network requirement."""
import subprocess
import unittest
from unittest.mock import patch
from docker_runner import ContainerCleanupError, run_container

CMD=['docker','--host=unix:///var/run/docker.sock','run','--rm','image','sleep','30']

class ContainerLifecycleTests(unittest.TestCase):
    def test_timeout_cleanup_confirmed_before_next_run(self):
        live=set(); events=[]
        def fake(cmd,**kwargs):
            if 'run' in cmd:
                self.assertFalse(live,'previous timed-out container survived')
                name=cmd[cmd.index('--name')+1];live.add(name);events.append(('start',name))
                raise subprocess.TimeoutExpired(cmd,kwargs['timeout'])
            name=cmd[-1];self.assertIn(name,live);live.remove(name);events.append(('removed',name))
            return subprocess.CompletedProcess(cmd,0,'','')
        with patch('docker_runner.subprocess.run',side_effect=fake):
            for _ in range(2):
                with self.assertRaises(subprocess.TimeoutExpired):run_container(CMD,timeout=1)
        self.assertFalse(live);self.assertNotEqual(events[0][1],events[2][1])
        self.assertEqual([e[0] for e in events],['start','removed','start','removed'])

    def test_nonzero_execution_status_preserved(self):
        with patch('docker_runner.subprocess.run',side_effect=[subprocess.CompletedProcess(CMD,17,'','bad input'),subprocess.CompletedProcess([],0,'','')]):
            self.assertEqual(run_container(CMD,timeout=1).returncode,17)

    def test_cleanup_failure_does_not_hide_original_timeout(self):
        with patch('docker_runner.subprocess.run',side_effect=[subprocess.TimeoutExpired(CMD,1),subprocess.CompletedProcess([],125,'','daemon unavailable')]) as calls:
            with self.assertRaises(ContainerCleanupError) as error:run_container(CMD,timeout=1)
        self.assertIsInstance(error.exception.__context__,subprocess.TimeoutExpired)
        self.assertEqual(calls.call_count,2)

    def test_cleanup_failure_records_execution_status(self):
        with patch('docker_runner.subprocess.run',side_effect=[subprocess.CompletedProcess(CMD,23,'','failed'),subprocess.CompletedProcess([],125,'','permission denied')]):
            with self.assertRaises(ContainerCleanupError) as error:run_container(CMD,timeout=1)
        self.assertEqual(error.exception.execution_returncode,23)

    def test_already_auto_removed_is_confirmed_absent(self):
        def fake(cmd,**kwargs):
            if 'run' in cmd:return subprocess.CompletedProcess(cmd,0,'ok','')
            return subprocess.CompletedProcess(cmd,1,'','Error response from daemon: No such container: '+cmd[-1])
        with patch('docker_runner.subprocess.run',side_effect=fake):
            self.assertEqual(run_container(CMD,timeout=1).returncode,0)
