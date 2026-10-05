"""CPU packaging checks; no CUDA library imports or model loading."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import zipfile

spec = importlib.util.spec_from_file_location('spark_candidate', Path(__file__).parents[2] / 'scripts/spark_candidate.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class WheelContract(unittest.TestCase):
    def wheel(self, extension='deep_gemm/_C.cpython-312-aarch64-linux-gnu.so', header=b'expected'):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / 'candidate.whl'
        with zipfile.ZipFile(path, 'w') as z:
            if extension:
                z.writestr(extension, b'native-fixture')
            z.writestr('deep_gemm/include/test.cuh', header)
        source = {'source_sha256': {'deep_gemm/include/test.cuh': m.sha(b'expected')}}
        return path, source

    def test_native_wheel_records_exact_bytes(self):
        path, source = self.wheel()
        result = m.check_wheel(path, source)
        self.assertEqual(result['sha256'], m.sha(path.read_bytes()))
        self.assertEqual(result['installed_files_sha256']['deep_gemm/include/test.cuh'], m.sha(b'expected'))

    def test_reject_cpu_only_wheel(self):
        with self.assertRaisesRegex(ValueError, 'native extension'):
            m.check_wheel(*self.wheel(extension=None))

    def test_reject_x86_wheel(self):
        with self.assertRaisesRegex(ValueError, 'native extension'):
            m.check_wheel(*self.wheel(extension='deep_gemm/_C.cpython-312-x86_64-linux-gnu.so'))

    def test_reject_stale_packaged_header(self):
        with self.assertRaisesRegex(ValueError, 'source mismatch'):
            m.check_wheel(*self.wheel(header=b'stale'))

    def test_inspection_requires_successful_build(self):
        with self.assertRaisesRegex(ValueError, 'completed native-build'):
            m.inspect_install({'validation': 'SOURCE_ONLY_NOT_CUDA_OR_MODEL'})


    def provider_case(self, wrong_provider=False, wrong_native=False):
        path, source = self.wheel()
        wheel = m.check_wheel(path, source)
        install = path.parent / 'installed'
        with zipfile.ZipFile(path) as z:
            z.extractall(install)
        init = install / 'deep_gemm/__init__.py'
        init.write_text('# fixture')
        wheel['installed_files_sha256']['deep_gemm/__init__.py'] = m.sha(init.read_bytes())
        dist = SimpleNamespace(version='test', locate_file=lambda n: install / n)
        native = install / 'deep_gemm/_C.cpython-312-aarch64-linux-gnu.so'
        provider = SimpleNamespace(__name__='deep_gemm',
            __file__=str(init) if not wrong_provider else '/other/__init__.py',
            _C=SimpleNamespace(__file__=str(native) if not wrong_native else '/other/_C.so'))
        wrapper = SimpleNamespace(_import_deep_gemm=lambda: provider)
        receipt = {'validation': 'NATIVE_BUILD_ONLY_NOT_JIT_OR_GPU_OR_MODEL',
                   'source_revision': 'fixture', 'wheel': wheel}
        with patch.object(m.importlib.metadata, 'distribution', return_value=dist), \
             patch.dict('sys.modules', {'vllm.utils.deep_gemm': wrapper}):
            return m.inspect_install(receipt, resolve_provider=True)

    def test_actual_resolver_path_is_checked(self):
        result = self.provider_case()
        self.assertEqual(result['validation'], 'PROVIDER_RESOLVED_NOT_JIT_OR_GPU_OR_MODEL')

    def test_reject_vendored_fallback(self):
        with self.assertRaisesRegex(ValueError, 'different DeepGEMM provider'):
            self.provider_case(wrong_provider=True)

    def test_reject_mixed_native_extension(self):
        with self.assertRaisesRegex(ValueError, 'different native extension'):
            self.provider_case(wrong_native=True)


if __name__ == '__main__':
    unittest.main()
