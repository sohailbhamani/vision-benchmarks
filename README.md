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

## Contributing

We welcome contributions! If you've benchmarked a model or framework not covered here, please open a PR.

## License

MIT License - See [LICENSE](LICENSE)
