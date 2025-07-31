# hip-fryer

An AMD GPU stress testing tool for MI300x, MI325x and other ROCm-compatible GPUs.

## Quickstart

Use Docker:

```bash
# You need a system with AMD GPUs
docker run --privileged ghcr.io/wiggels/hip-fryer:0.1.1 60
```

## Features

- Matrix multiplication stress testing using ROCBlas
- Multi-GPU support 
- Performance monitoring and health checks
- Configurable duration and tolerance settings
- FP32 and BF16 precision support
- Real-time GFLOPS reporting

## Current Limitations

- Throttling detection not yet implemented

## Attribution

This project is derived from gpu-fryer by Hugging Face, originally licensed under Apache License 2.0. The original work has been adapted from NVIDIA CUDA/cuBLAS to AMD HIP/ROCBlas for use with AMD GPUs and the ROCm ecosystem.
