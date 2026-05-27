# hip-fryer

An AMD GPU stress testing tool for MI300X / MI325X / MI355X and other
ROCm-compatible accelerators. Built against ROCm 7.

## Quickstart

```bash
# 60-second FP8 burn on every visible AMD GPU
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  ghcr.io/wiggels/hip-fryer:latest --precision fp8 60
```

If you'd rather grant full device access (matches the upstream Docker examples):

```bash
docker run --rm --privileged ghcr.io/wiggels/hip-fryer:latest --precision fp8 60
```

## Usage

```
hip-fryer [OPTIONS] [DURATION_SECS]

Arguments:
  [DURATION_SECS]                  Seconds of steady-state burn (default 60, min 10)

Options:
      --tflops-tolerance <PERCENT> Pass threshold: each GPU must be within this %
                                   of the best GPU's average (default 10)
      --precision <PRECISION>      Numeric format for the burn GEMM: fp32, bf16,
                                   or fp8 (OCP E4M3). Defaults to the highest-
                                   throughput format every visible GPU supports
      --size <N>                   Square GEMM dimension N (burns an N×N×N matmul);
                                   must be a multiple of 64 (default 16384)
```

When `--precision` is omitted, hip-fryer picks the fastest format every detected
GPU supports — FP8 if available, else BF16, else FP32.

```bash
# Force FP8 and sweep the matrix size to chase peak throughput
./hip-fryer --precision fp8 --size 12288 60
```

## Features

- Concurrent hipBLASLt GEMM burn across every visible GPU
- FP32, BF16, and FP8 (E4M3) paths, all f32-accumulate; the heuristic's kernel is
  warmed once on GPU 0 to prime the shared comgr cache before the parallel burn
- GPUs are identified by ISA arch (e.g. `gfx950`) as well as marketing name, so
  detection and the peak-throughput table work even in containers that report a
  generic "AMD Radeon Graphics"
- Per-GPU warmup detection — the duration timer only starts once every GPU is
  at steady state, so allocation latency doesn't eat into the measurement window
- Per-second throughput + temperature reporting (junction temp where available,
  edge / memory as fallback) via ROCm SMI
- Pass/fail vs. configurable TFLOPS tolerance, reported as % of published peak

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
to HIP / hipBLASLt for the ROCm ecosystem.
