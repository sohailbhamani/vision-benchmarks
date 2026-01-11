# Issue #10: Set Up Linux Dev Environment (Revised)

> **Issue:** [locatelogic#10](https://github.com/sohailbhamani/locatelogic/issues/10)  
> **Repo:** `sohailbhamani/vision-benchmarks`  
> **Priority:** High | **Reasoning:** High (CUDA version alignment)

---

## Current Status & Problem

The initial environment setup failed due to a CUDA symbol mismatch:

- `torch-2.9.1+cu128` requires CUDA 12.8 runtime symbols.
- `paddlepaddle-gpu-3.1.0` (from cu123 index) forced a downgrade of `nvidia-cuda-*-cu12` packages to `12.3.x`.
- Error: `undefined symbol: cudaGetDriverEntryPointByVersion`.

## Proposed Solution: Environment Alignment

We will force the alignment of NVIDIA CUDA runtime packages to match the requirements of the highest-tier dependency (Torch 2.9/cu128), while maintaining PaddlePaddle compatibility through backward-compatible runtime libraries.

### 1. Updated Installation Strategy

- **Step 1: Install Torch (cu128)** to establish the baseline for high-performance inference.
- **Step 2: Install PaddlePaddle GPU** using a version-flexible approach or and addressing its strict dependency pins.
- **Step 3: Force Sync Runtime Libs** to `12.8+` to resolve the Symbol error.

### 2. Required Packages Sync

Force upgrade the following to satisfy Torch 2.9:

- `nvidia-cuda-runtime-cu12 >= 12.8.90`
- `nvidia-cuda-nvrtc-cu12 >= 12.8.93`
- `nvidia-cudnn-cu12 >= 9.10`

## Execution Plan

1. **Environment Wipe & Restart (Optional but recommended)**
    - If current venv is too messy, recreate it.
    - *Decision:* Try fixing current venv first to save time.
2. **Force Alignment of CUDA Runtimes**
    - Run `pip install --upgrade nvidia-cuda-*` to pull the latest 12.x runtimes.
3. **Verify Library Linking**
    - Use `ldd` and `scripts/verify_env.py` to confirm Torch and Paddle can both load.
4. **Finalize Dependencies**
    - Update `requirements.txt` with these known-good aligned versions.
5. **Document Setup**
    - Ensure `tutorials/setup_linux.md` mentions the specific index URLs for CUDA-specific wheels.

## Acceptance Criteria

- [ ] `import torch; torch.cuda.is_available()` returns `True`.
- [ ] `import paddle; paddle.is_compiled_with_cuda()` returns `True`.
- [ ] `import onnxruntime` detects the GPU provider.
- [ ] `verify_env.py` runs to completion without `ImportError`.
