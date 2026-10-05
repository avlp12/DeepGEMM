#!/usr/bin/env python3
"""Build/audit an isolated Spark candidate. Inspection never imports CUDA code."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import zipfile

BASE = '5f33a18079e96d26d5869c9759657eb6150f31b1'
ROOT = Path(__file__).resolve().parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


def source_manifest(expected):
    if not re.fullmatch(r'[0-9a-f]{40}', expected):
        raise ValueError('expected revision must be a full commit ID')
    if git('rev-parse', 'HEAD') != expected:
        raise ValueError('source revision differs from requested revision')
    if git('status', '--porcelain', '--untracked-files=normal'):
        raise ValueError('candidate source or submodules are dirty')
    subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', BASE, expected], check=True)
    modules = git('submodule', 'status', '--recursive').splitlines()
    if not modules or any(line[0] in '-+U' for line in modules):
        raise ValueError('submodules must be initialized at their recorded revisions')
    files = {}
    for p in sorted((ROOT / 'deep_gemm').rglob('*')):
        if p.is_file() and (p.suffix in ('.cuh', '.hpp', '.py')) and not p.is_symlink():
            files[str(p.relative_to(ROOT))] = sha(p.read_bytes())
    return {'source_revision': expected, 'base_revision': BASE,
            'submodules': modules, 'source_sha256': files,
            'validation': 'SOURCE_ONLY_NOT_CUDA_OR_MODEL'}


def check_wheel(path, source):
    """Verify native wheel and source headers before it can shadow a vendored copy."""
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        native = [n for n in names if re.fullmatch(r'deep_gemm/_C\..*aarch64.*\.so', n)]
        if len(native) != 1:
            raise ValueError('wheel must contain exactly one aarch64 native extension')
        for name, expected in source['source_sha256'].items():
            if name not in names or sha(z.read(name)) != expected:
                raise ValueError('wheel source mismatch: ' + name)
        return {'filename': path.name, 'sha256': sha(path.read_bytes()),
                'installed_files_sha256': {n: sha(z.read(n)) for n in names
                    if n.startswith('deep_gemm/') and not n.endswith('/')}}


def build(expected, output):
    source = source_manifest(expected)
    if platform.system() != 'Linux' or platform.machine() != 'aarch64':
        raise ValueError('build inside the Linux/aarch64 serving image, not on the desktop')
    if output.is_relative_to(ROOT):
        raise ValueError('output directory must be outside the source checkout')
    output.mkdir(parents=True, exist_ok=True)
    import torch  # Host compiler ABI only; no device query or tensor allocation.
    if torch.version.cuda is None:
        raise ValueError('the serving image must provide a CUDA-enabled torch/toolkit')
    env = dict(os.environ, DG_FORCE_BUILD='1', DG_USE_LOCAL_VERSION='1',
               DG_SKIP_CUDA_BUILD='0', CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='void',
               MAX_JOBS='4')
    # Do not bake another installation's JIT cache path into the wheel.
    env.pop('DG_JIT_CACHE_DIR', None)
    with tempfile.TemporaryDirectory(dir=output) as td:
        subprocess.run([sys.executable, 'setup.py', 'bdist_wheel', '--dist-dir', td],
                       cwd=ROOT, env=env, check=True)
        wheels = list(Path(td).glob('*.whl'))
        if len(wheels) != 1:
            raise ValueError('expected exactly one wheel')
        wheel = check_wheel(wheels[0], source)
        destination = output / wheels[0].name
        if destination.exists():
            raise ValueError('refusing to overwrite an earlier wheel')
        wheels[0].rename(destination)
    receipt = dict(source, wheel=wheel, torch=torch.__version__, cuda=torch.version.cuda,
                   cxx11_abi=bool(torch.compiled_with_cxx11_abi()),
                   validation='NATIVE_BUILD_ONLY_NOT_JIT_OR_GPU_OR_MODEL')
    receipt_path = output / (destination.name + '.json')
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def inspect_install(receipt, resolve_provider=False):
    """Verify wheel bytes on disk; do not import torch, deep_gemm, or vllm."""
    if receipt.get('validation') != 'NATIVE_BUILD_ONLY_NOT_JIT_OR_GPU_OR_MODEL':
        raise ValueError('a completed native-build receipt is required')
    dist = importlib.metadata.distribution('deep-gemm')
    for name, expected in receipt['wheel']['installed_files_sha256'].items():
        p = Path(dist.locate_file(name))
        if not p.is_file() or sha(p.read_bytes()) != expected:
            raise ValueError('installed candidate mismatch: ' + name)
    result = {'installed_version': dist.version,
              'source_revision': receipt['source_revision'],
              'validation': 'INSTALLED_BYTES_ONLY_NOT_PROVIDER_RESOLUTION_OR_GPU'}
    if resolve_provider:
        # Opt-in only: use in a fresh candidate process. Imports the engine's
        # actual resolver rather than assuming an installed wheel is selected.
        from vllm.utils.deep_gemm import _import_deep_gemm
        provider = _import_deep_gemm()
        expected_init = Path(dist.locate_file('deep_gemm/__init__.py')).resolve()
        if provider is None or Path(provider.__file__).resolve() != expected_init:
            raise ValueError('vLLM resolved a different DeepGEMM provider')
        native_paths = [Path(dist.locate_file(n)).resolve()
                        for n in receipt['wheel']['installed_files_sha256']
                        if n.startswith('deep_gemm/_C.') and n.endswith('.so')]
        if Path(provider._C.__file__).resolve() not in native_paths:
            raise ValueError('provider resolved a different native extension')
        result.update(provider=provider.__name__,
                      validation='PROVIDER_RESOLVED_NOT_JIT_OR_GPU_OR_MODEL')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    audit = sub.add_parser('source')
    audit.add_argument('--revision', required=True)
    wheel = sub.add_parser('build')
    wheel.add_argument('--revision', required=True)
    wheel.add_argument('--output', type=Path, required=True)
    inspect = sub.add_parser('inspect')
    inspect.add_argument('--receipt', type=Path, required=True)
    inspect.add_argument('--resolve-provider', action='store_true')
    args = parser.parse_args()
    if args.action == 'source':
        result = source_manifest(args.revision)
    elif args.action == 'build':
        result = build(args.revision, args.output.resolve())
    else:
        result = inspect_install(json.loads(args.receipt.read_text()), args.resolve_provider)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
