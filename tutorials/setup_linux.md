# Linux Development Environment Setup

This guide walks you through setting up a GPU-accelerated environment for vision benchmarks on Linux (tested on Pop!_OS 24.04 with RTX 3090).

## Prerequisites

- **NVIDIA Driver:** 580.82+ (Supports CUDA 13.0+)
- **Python:** 3.11 or 3.12
- **Hardware:** NVIDIA GPU (Ampere or newer recommended)
- **Compiler Cache:** `ccache` (Recommended for PaddlePaddle performance)

## 1. System Dependencies

Before creating the environment, install `ccache` to speed up compilation of deep learning extensions (like PaddlePaddle custom ops).

```bash
sudo apt update && sudo apt install -y ccache
```

## 2. Create Virtual Environment

We use a standard Python virtual environment for isolation.

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

## 2. Install Core AI Stack

The installation requires specific index URLs to ensure CUDA 12.x compatibility across Torch, Paddle, and ONNX Runtime.

```bash
# Install Torch and Torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install PaddlePaddle GPU
pip install paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/stable/cu123/ --extra-index-url https://pypi.org/simple

# Install remaining dependencies from requirements.txt
pip install -r requirements.txt
```

### Why two index URLs?

- **Torch** uses the pytorch.org index for `+cu128` wheels.
- **Paddle** uses its own stable index for `cu123` compatible wheels.
- We then align the runtime libraries during the requirements install.

## 3. Verification

Run the verification script to ensure all libraries (Torch, Paddle, ONNX, TensorRT) detect the GPU correctly.

```bash
python scripts/verify_env.py
```

Expected output should show `True` for both PyTorch and Paddle GPU availability.

## Troubleshooting

### Undefined Symbol Error

If you see `undefined symbol: cudaGetDriverEntryPointByVersion`, it means your `nvidia-cuda-runtime-cu12` was downgraded by a package (likely Paddle). Run:

```bash
pip install --upgrade nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cudnn-cu12
```

### Missing libnvToolsExt.so.1

If Paddle fails with a missing `libnvToolsExt.so.1`, install the legacy NVTX wrapper:

```bash
pip install nvidia-nvtx-cu11
```
