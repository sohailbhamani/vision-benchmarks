# Issue #10: Set Up Linux Dev Environment

> **Issue:** [locatelogic#10](https://github.com/sohailbhamani/locatelogic/issues/10)  
> **Repo:** `sohailbhamani/vision-benchmarks`  
> **Priority:** High | **Estimate:** 2 hours

---

## Summary

Establish a reproducible, GPU-accelerated Python environment for running vision benchmarks on Linux (RTX 3090). We will match the reference environment versions where possible but use standard tools (`pip`/`venv`) for public accessibility.

## Proposed Changes

### [NEW] `requirements.txt` (Pinned Versions)

Core stack based on reference environment:

- `torch>=2.7.1` (CUDA 12.8 compatible)
- `torchvision>=0.22.1`
- `onnxruntime-gpu==1.23.2`
- `ultralytics` (Latest)
- `tensorrt` (Python bindings)

**PaddlePaddle:**
We will check for a CUDA 12.x compatible build.

### [NEW] Documentation: `tutorials/setup_linux.md`

Standard setup guide:

1. **Prerequisites:** NVIDIA Drivers, CUDA 12.x.
2. **Installation:**

    ```bash
    # Create virtual environment
    python3 -m venv .venv
    source .venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install Torch stack (matching reference env)
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    
    # Install remaining requirements
    pip install -r requirements.txt
    ```

3. **Verification:** How to run the verification script.

### [NEW] Verification Script: `scripts/verify_env.py`

A script to assert:

- `torch.cuda.is_available()` == True
- `onnxruntime.get_available_providers()` includes 'CUDAExecutionProvider'
- `tensorrt` import works
- `ultralytics` YOLO model load test

---

## Execution Plan

1. **Environment Check**:
    - Verify host CUDA (`nvcc`) and Driver (`nvidia-smi`).
2. **Define Dependencies**:
    - Create `requirements.txt`.
3. **Install**:
    - Create standard venv and install.
4. **Verify**:
    - Write and run `scripts/verify_env.py`.
5. **Document**:
    - Create `tutorials/setup_linux.md`.
6. **Commit**:
    - Push to `vision-benchmarks/main`.

## Acceptance Criteria

- [ ] `pip install` succeeds without conflicts.
- [ ] `verify_env.py` passes for Torch, ONNX, and TensorRT.
- [ ] Documentation uses standard commands accessible to all users.

---

## Next Steps

- Update `arch.json` -> Issue #12 (YOLO Benchmarks).
