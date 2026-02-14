# Vision Benchmarks

[![CI](https://github.com/sohailbhamani/vision-benchmarks/actions/workflows/ci.yml/badge.svg)](https://github.com/sohailbhamani/vision-benchmarks/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Benchmarks and tutorials for vision AI models.

## Purpose

This repository provides **practical analysis and comparison of available tools** for vision AI. Our goal is to build a growing resource that helps developers:

- Evaluate models and frameworks before committing to a stack
- Understand real-world performance characteristics
- Find tested configurations and best practices
- Contribute their own benchmarks and findings

We believe the AI/ML community benefits from open, reproducible benchmarks. This is one of several domain-specific benchmark repos we maintain (vision, audio, etc.) to help developers make informed decisions.

## What We Benchmark

- **Object Detection:** YOLO variants (v11, v12, v26, YOLO-World, RF-DETR)
- **OCR:** PaddleOCR, EasyOCR — accuracy on curved/rotated text
- **Embeddings:** DINOv2, CLIP for similarity search
- **Segmentation:** SAM, YOLO-Seg variants

## Structure

- `benchmarks/` - Speed and accuracy results
- `configs/` - Model configuration files
- `tutorials/` - Setup and usage guides

## Getting Started

### Prerequisites

- **GPU:** NVIDIA GPU with CUDA drivers (Recommended: RTX 30 series or newer)
- **OS:** Linux (Ubuntu 22.04+ recommended)
- **Python:** 3.10+

### Installation

```bash
# Clone
git clone https://github.com/sohailbhamani/vision-benchmarks.git
cd vision-benchmarks

# Activate shared LocateLogic environment
source /mnt/devdisk/locatelogic_env/bin/activate

# Install Dependencies
pip install -r requirements.txt
```

## Running Benchmarks

### 1. Speed Benchmark (FPS)

Measures inference speed (Pre-process + Inference + Post-process).

```bash
# Run all standard models (v11n, v11s, v11m, v8s-world, v8m-world)
python3 benchmarks/benchmark_yolo.py --runs 100 --fp16

# Run specific model
python3 benchmarks/benchmark_yolo.py --model yolo11n.pt
```

### 2. Accuracy Benchmark (mAP)

Measures mAP@50 and mAP@50-95 on COCO dataset.

```bash
# Fast check (COCO128 - downloads automatically)
python3 benchmarks/benchmark_accuracy.py --data coco128.yaml

# Full validation (Requires COCO dataset)
python3 benchmarks/benchmark_accuracy.py --data coco.yaml
```

### 3. Webcam Latency Benchmark (Glass-to-Glass)

Measures end-to-end latency from "photon to pixel". Requires a connected webcam.

```bash
# Default: Source 0
python3 benchmarks/benchmark_webcam.py

# Specify source
python3 benchmarks/benchmark_webcam.py --source 1
```

## Results

Benchmark results are automatically saved to the `results/` directory:
- `results/yolo_speed_results.md`
- `results/yolo_accuracy_results.md`
- `results/webcam_latency_results.md`

## Contributing

We welcome contributions! If you've benchmarked a model or framework not covered here, please open a PR.

## License

MIT License - See [LICENSE](LICENSE)
