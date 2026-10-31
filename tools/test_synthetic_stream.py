"""AOT self-resource contract with tiny payload; no NumPy/model dependency."""
import json
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from synthetic_stream import kernel, preflight, JOB_MEMORY, embed_in_place, header

class StreamBudgetTests(unittest.TestCase):
    def test_refuse_unbounded_job_and_file_dimension(self):
        with tempfile.TemporaryDirectory() as d:
            with patch('synthetic_stream.snapshot',return_value={'memory.max':str(32*1024**3)}):
                with self.assertRaises(RuntimeError): preflight(5120,Path(d))
            with self.assertRaises(ValueError): preflight(40960,Path(d))

    def test_refuse_disk_before_generation(self):
        with tempfile.TemporaryDirectory() as d:
            with patch('synthetic_stream.snapshot',return_value={'memory.max':str(JOB_MEMORY)}):
                with patch('synthetic_stream.shutil.disk_usage',return_value=shutil._ntuple_diskusage(10,10,0)):
                    with self.assertRaises(RuntimeError): preflight(20480,Path(d))

    def test_inplace_overlap_preserves_multichunk_payload(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);model=root/'fixture';launcher=root/'launcher';dest=root/'kernel'
            n=1100;size=4*n*n;payload=(bytes(range(256))*((size+255)//256))[:size]
            model.write_bytes(bytes(288)+payload);launcher.write_bytes(b'ELF'+bytes(5000))
            offset=embed_in_place(model,launcher,dest,288,n,0)
            self.assertEqual(dest.read_bytes()[offset:offset+size],payload)
            self.assertFalse(model.exists())

    def test_controller_propagates_failed_job(self):
        import run_synthetic_stream as controller
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            args=['controller','--results',str(root/'results'),'--work',str(root/'work'),'--sizes','5120']
            with patch.object(controller,'np',object()), patch('sys.argv',args), patch.object(controller,'host_preflight',return_value={}):
                with patch.object(controller.subprocess,'run',return_value=subprocess.CompletedProcess([],2,'','failed job')):
                    self.assertEqual(controller.main(),2)
            report=json.loads((root/'results/summary.json').read_text())
            self.assertEqual(report[0]['container_exit'],2)

    def test_peak_disk_counts_all_prepared_inputs_references_and_output(self):
        from synthetic_stream import disk_budget
        for n in [5120,10240]:
            budget=disk_budget(n,0.001)
            # At matmul's first output: two F32 B + two F64 references + one F64 output.
            self.assertEqual(budget['peak_auxiliary_disk_bytes'],32*n*n)
            old_requirement=len(header(n,20261030))+4*n*n+512*1024**2
            with tempfile.TemporaryDirectory() as d:
                root=Path(d)
                with patch('synthetic_stream.snapshot',return_value={'memory.max':str(JOB_MEMORY)}):
                    with patch('synthetic_stream.shutil.disk_usage',return_value=shutil._ntuple_diskusage(10**12,0,old_requirement+1024)):
                        with self.assertRaises(RuntimeError):preflight(n,root,0.001)
                self.assertEqual(list(root.iterdir()),[],'disk refusal must precede fixture/input writes')

    def test_skipped_matmul_has_no_dense_input_disk_reservation(self):
        from synthetic_stream import disk_budget
        self.assertEqual(disk_budget(20480,1.0)['peak_auxiliary_disk_bytes'],32*20480)

    def test_calibration_invalid_values_rejected_before_output(self):
        import run_synthetic_stream as controller
        import synthetic_stream as worker
        for module in [controller,worker]:
            for value in ['0','-1','nan','inf','-inf']:
                with tempfile.TemporaryDirectory() as d:
                    root=Path(d)
                    args=['runner','--work',str(root/'work'),'--results',str(root/'results'),'--mm-seconds-at-1280='+value]
                    if module is worker:args+=['--n','5120']
                    with patch.object(module,'np',object()),patch('sys.argv',args),patch('sys.stderr'):
                        with self.assertRaises(SystemExit) as error:module.main()
                    self.assertEqual(error.exception.code,2)
                    self.assertEqual(list(root.iterdir()),[])

    def test_main_rejects_dense_disk_budget_before_fixture_write(self):
        import synthetic_stream as worker
        n=5120
        old_requirement=len(header(n,20261030))+4*n*n+512*1024**2
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);work=root/'work';results=root/'results'
            args=['worker','--n',str(n),'--work',str(work),'--results',str(results),'--mm-seconds-at-1280','0.001']
            with patch.object(worker,'np',SimpleNamespace(__version__='test')),patch('sys.argv',args),patch('sys.stdout'):
                with patch.object(worker,'snapshot',return_value={'memory.max':str(JOB_MEMORY)}):
                    with patch.object(worker.shutil,'disk_usage',return_value=shutil._ntuple_diskusage(10**12,0,old_requirement+1024)):
                        self.assertEqual(worker.main(),2)
            self.assertEqual(list(work.iterdir()),[])
            self.assertIn('disk',json.loads((results/'result.json').read_text())['error'])

@unittest.skipUnless(shutil.which('cc'),'C compiler required for tiny AOT regression')
class SelfResourceTests(unittest.TestCase):
    def test_dynamic_inputs_and_corrupt_or_truncated_resource(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);src=root/'kernel.c';exe=root/'kernel';src.write_text(kernel(2))
            subprocess.run(['cc','-std=c11','-O2',str(src),'-lm','-o',str(exe)],check=True,capture_output=True)
            weights=struct.pack('<4f',1,-2,3,4);checksum=14695981039346656037
            for b in weights:checksum=((checksum^b)*1099511628211)&((1<<64)-1)
            launcher=root/'launcher';exe.rename(launcher)
            model=root/'fixture.gguf';h=header(2,17);model.write_bytes(h+weights)
            offset=embed_in_place(model,launcher,exe,len(h),2,checksum)
            self.assertFalse(model.exists())
            self.assertEqual(exe.read_bytes()[offset:offset+len(weights)],weights)
            original=exe.read_bytes()
            def run(values,op='--matvec'):
                return subprocess.run([str(exe),op],input=struct.pack('<'+str(len(values))+'f',*values),capture_output=True,timeout=5)
            r=run([0.5,-1]);self.assertEqual(r.returncode,0);self.assertEqual(struct.unpack('<2d',r.stdout),(2.5,-2.5))
            self.assertIn('integrity_seconds',json.loads(r.stderr))
            other=run([1,0]);self.assertNotEqual(other.stdout,r.stdout)
            m=run([-1,2,0.5,-3],'--matmul');self.assertEqual(struct.unpack('<4d',m.stdout),(-2,8,-1,-6))
            for pos in [offset,len(original)-40]:
                broken=bytearray(original);broken[pos]^=1;exe.write_bytes(broken)
                self.assertEqual(run([0.5,-1]).returncode,2)
            exe.write_bytes(original[:-1]);self.assertEqual(run([0.5,-1]).returncode,2)
