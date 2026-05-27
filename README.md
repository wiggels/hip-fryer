# hip-fryer

An AMD GPU stress testing tool for MI300X / MI325X / MI355X and other
ROCm-compatible accelerators. Built against ROCm 7.

## Quickstart

```bash
# 60-second BF16 burn on every visible AMD GPU
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  ghcr.io/wiggels/hip-fryer:latest 60
```

If you'd rather grant full device access (matches the upstream Docker examples):

```bash
docker run --rm --privileged ghcr.io/wiggels/hip-fryer:latest 60
```

## Usage

```
hip-fryer [OPTIONS] [DURATION_SECS]

Arguments:
  [DURATION_SECS]                       Seconds of steady-state burn (default 60, min 10)

Options:
      --tflops-tolerance <PERCENT>      Pass threshold vs. best GPU (default 10)
      --use-fp32                        Force FP32 GEMM
      --use-bf16                        Force BF16 GEMM (fails if any GPU lacks support)
```

When neither `--use-fp32` nor `--use-bf16` is set, hip-fryer picks BF16 if every
detected GPU supports it, otherwise FP32.

## Features

- Concurrent rocBLAS GEMM burn across every visible GPU
- FP32 (`rocblas_sgemm`) and BF16 (`rocblas_gemm_ex`, f32 accumulate) paths
- Per-GPU warmup detection — the duration timer only starts once every GPU is
  at steady state, so allocation latency doesn't eat into the measurement window
- Per-second throughput + temperature reporting (junction temp where available,
  edge / memory as fallback) via ROCm SMI
- Pass/fail vs. configurable TFLOPS tolerance

## Building from source

Needs Rust stable (2024 edition) and ROCm 7 installed at `/opt/rocm` (override
with `ROCM_PATH`).

```bash
cargo build --release
./target/release/hip-fryer 60
```

## Attribution

Derived from [gpu-fryer](https://github.com/huggingface/gpu-fryer) by Hugging Face,
originally licensed under Apache License 2.0. The CUDA / cuBLAS path was ported
to HIP / rocBLAS for the ROCm ecosystem.
