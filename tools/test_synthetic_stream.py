"""AOT self-resource contract with tiny payload; no NumPy/model dependency."""
import json
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest
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
