# YOLO Speed Benchmarks

**GPU:** NVIDIA GeForce RTX 3090
**Ultralytics:** 8.4.14 | **TensorRT:** 10.15.1 | **PyTorch:** 2.10.0+cu128
**Input:** 640x640 RGB | **Runs:** 100 per model | **Warmup:** 10

## Results

| Model | PyTorch FP32 | PyTorch FP16 | TensorRT FP16 |
|-------|-------------:|-------------:|--------------:|
| yolo11n | 85.70 FPS (11.67ms) | 69.22 FPS (14.45ms) | 352.18 FPS (2.84ms) |
| yolo11s | 82.42 FPS (12.13ms) | 77.83 FPS (12.85ms) | 326.15 FPS (3.07ms) |
| yolo11m | 74.23 FPS (13.47ms) | 66.38 FPS (15.06ms) | 247.64 FPS (4.04ms) |
| yolo11l | 46.43 FPS (21.54ms) | 43.15 FPS (23.18ms) | 216.51 FPS (4.62ms) |
| yolo12n | 63.00 FPS (15.87ms) | 54.85 FPS (18.23ms) | 327.20 FPS (3.06ms) |
| yolo12s | 63.68 FPS (15.70ms) | 55.79 FPS (17.93ms) | 282.19 FPS (3.54ms) |
| yolo26n | 83.10 FPS (12.03ms) | 69.08 FPS (14.48ms) | 416.59 FPS (2.40ms) |
| yolo26s | 81.36 FPS (12.29ms) | 65.93 FPS (15.17ms) | 368.64 FPS (2.71ms) |
| yolov8s-world | 75.80 FPS (13.19ms) | 57.18 FPS (17.49ms) | N/A* |
| yolov8m-world | 67.01 FPS (14.92ms) | 55.84 FPS (17.91ms) | N/A* |

*YOLO-World models use `adaptive_max_pool2d` which is not supported in ONNX/TensorRT export.

## Key Findings

- **TensorRT FP16 delivers 3-5x speedup** over PyTorch FP32 across all compatible models
- **YOLO26n is the fastest model** at 417 FPS (TensorRT) — ideal for real-time applications
- **PyTorch FP16 shows no speedup** on RTX 3090 with ultralytics inference pipeline (overhead-dominated)
- **YOLOv12 is slower than v11** in PyTorch but competitive in TensorRT due to attention-based architecture
- **YOLO-World** cannot export to TensorRT due to `adaptive_max_pool2d` limitation
