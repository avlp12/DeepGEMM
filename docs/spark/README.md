# Spark integration candidate

I maintain this branch as a conservative SM12x integration base for an existing
EXL3 serving stack. It is **not a measured speedup or a production release**.
I retain the original kernel implementations and license notices.

## Source choice

| Source | Pin | Decision |
|---|---|---|
| DeepSeek upstream main | `057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7` | Reference. This release targets SM90/SM100; not a Spark drop-in. |
| vLLM `nv_dev+situ` | `5f33a18079e96d26d5869c9759657eb6150f31b1` | Base of this branch; existing SM12x FP8 indexer API and cleaning kernel retained. |
| vLLM build's earlier dev pin | `e1f418c2a4f20818221f6b0e578b4c2f634d4c3f` | Newer API/JIT implementation; SM12x unpaged `clean_logits=True` is explicitly refused. |
| vLLM dev inspected on 2026-10-06 | `550036ad85e8c9af43d6da983af0a68e0d417888` | Same cleaning incompatibility. Includes FP8 paged KV32 support, but requires separate integration. |

References: [upstream](https://github.com/deepseek-ai/DeepGEMM),
[vLLM base](https://github.com/vllm-project/DeepGEMM/tree/5f33a18079e96d26d5869c9759657eb6150f31b1),
[latest inspected dev](https://github.com/vllm-project/DeepGEMM/tree/550036ad85e8c9af43d6da983af0a68e0d417888).

GB10 is compute capability 12.1. Blackwell datacenter kernels do not automatically
support it. EXL3 weights are not interchangeable with MXFP4/FP8 operands, and
NVLink-oriented MegaMoE does not replace inter-node RoCE collectives.

## Actual engine boundary

The existing vLLM `vllm.utils.deep_gemm._import_deep_gemm` resolver prefers the
external `deep_gemm` package, then falls back to `vllm.third_party.deep_gemm`.
An isolated install therefore reaches both the vLLM indexer and a TensorFold
adapter calling that same low-level wrapper. It does not route TensorFold model
inference through a vLLM HTTP server. It does not add DeepGEMM to llama.cpp.

The currently required unpaged interface is:

```python
fp8_fp4_mqa_logits(
    (q_fp8, None), (k_fp8, k_scale_fp32), weights_fp32,
    start_int32, end_int32, clean_logits=True,
)
```

The target indexer uses 32 heads of width 128. Scales folded into head weights
remain FP32. The result uses absolute key columns in a float32 `[M,N]` tensor;
compressed-output mode (`max_seqlen_k > 0`) is not enabled. The paged path uses
`clean_logits=False`, 2D context lengths and the existing schedule metadata.
This base admits **FP8 paged block size 64**; do not select block size 32 merely
because a later fork supports it. Main MLA's 656-byte cache and indexer's
132-byte pool row are different formats.

## Build without touching a service

Use the exact Linux/aarch64 serving image, with its existing CUDA/PyTorch ABI.
Initialize the pinned submodules before the offline build:

```sh
git clone --recurse-submodules --branch codex/spark-sm12x-b0 https://github.com/avlp12/DeepGEMM.git
cd DeepGEMM
DG_REV=$(git rev-parse HEAD)
python3 scripts/spark_candidate.py source --revision "$DG_REV" > /tmp/deepgemm-source.json
```

Choose an existing immutable local image ID as `B0_IMAGE` and a new, empty output
directory outside the checkout as `DG_OUT`. The following creates only a build
container: no model, GPU exposure, service restart or dependency download.

```sh
docker run --rm --runtime=runc --network=none --cpus=4 --memory=8g \
  -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES= \
  -e DG_REV="$DG_REV" \
  -v "$PWD:/source:ro" -v "$DG_OUT:/out" \
  --entrypoint bash "$B0_IMAGE" -euc '
    cp -a /source /tmp/deepgemm-source
    cd /tmp/deepgemm-source
    git config --global --add safe.directory /tmp/deepgemm-source
    python3 scripts/spark_candidate.py build --revision "$DG_REV" --output /out
  ' > "$DG_OUT/build.log" 2>&1
```

The helper rejects dirty sources, wrong revisions/submodules, CPU-only or x86
wheels, and stale packaged headers. It builds with `DG_FORCE_BUILD=1` and records
wheel/native/header hashes plus the torch/CUDA ABI. A native build does not
compile every lazily generated CUDA kernel.

## Candidate install and rollback

In a **new candidate container**, mount the output directory read-only and install
its exact wheel with `python3 -m pip install --no-deps --no-index /out/<wheel>`.
Do not pip-install it into an already-running baseline. Check installed bytes:

```sh
python3 /source/scripts/spark_candidate.py inspect --receipt /out/<wheel>.json
```

At the next authorized serving attempt, add `--resolve-provider` in a fresh
process to verify the real vLLM resolver selects this package and native library.
Use the existing model smoke/benchmark controller and original model, tokenizer,
FP8 KV, draft and TP4 settings. Reuse completed unchanged-component evidence;
measure end-to-end TTFT/prefill/decode/concurrency for an actual changed candidate.
Pin the same wheel on all ranks. Preserve JIT caches and distinguish cold-JIT
startup from warm serving. Do not set a shared mutable cache path in the wheel.

Rollback means stopping only the candidate with SIGINT and returning to the
unchanged baseline image/launch configuration. No baseline package was replaced.

## Validation status

Local CPU checks cover artifact integrity and provider selection; their fixtures
are not GPU tests. Native ARM64 compilation, JIT, candidate import resolution and
real-model performance remain unexecuted for this branch. I make no speed claim.
The existing SM12x kernels are retained, so installing this base alone is not
expected to create a new performance gain. It provides an owned, reproducible
base for targeted indexer changes rather than an unqualified latest-version swap.
