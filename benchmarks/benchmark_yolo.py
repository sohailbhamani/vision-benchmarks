import argparse
import time
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

# Benchmark Configuration
MODELS = {
    "yolo11n": "yolo11n.pt",
    "yolo11s": "yolo11s.pt",
    "yolo11m": "yolo11m.pt",
    "yolo11l": "yolo11l.pt",
    "yolo12n": "yolo12n.pt",
    "yolo12s": "yolo12s.pt",
    "yolo26n": "yolo26n.pt",
    "yolo26s": "yolo26s.pt",
    "yolov8s-world": "yolov8s-world.pt",
    "yolov8m-world": "yolov8m-world.pt",
}


def download_if_missing(model_name):
    """Ensure model weights exist locally."""
    path = Path(f"{model_name}.pt")
    if not path.exists():
        print(f"Downloading {model_name}...")
        YOLO(f"{model_name}.pt")
    return str(path)


def export_tensorrt(model_path, fp16=True):
    """Export a .pt model to TensorRT engine and return the engine path."""
    engine_path = Path(model_path).with_suffix(".engine")
    if engine_path.exists():
        print(f"  TensorRT engine already exists: {engine_path}")
        return str(engine_path)

    print(f"  Exporting {model_path} to TensorRT (FP16={fp16})...")
    model = YOLO(model_path)
    path = model.export(format="engine", half=fp16, device=0)
    print(f"  Export complete: {path}")
    return str(path)


def benchmark_model(model_path, device="cuda", warmup=10, runs=100, fp16=False):
    """Run inference benchmark loop."""
    print(f"\nBenchmarking {model_path} on {device} (FP16={fp16})...")

    model = YOLO(model_path)

    # Dummy input (640x640 RGB)
    img = np.zeros((640, 640, 3), dtype=np.uint8)

    # Warmup
    print("  Warming up...")
    for _ in range(warmup):
        model(img, verbose=False, half=fp16)

    # Benchmark loop
    print(f"  Running {runs} inferences...")
    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(img, verbose=False, half=fp16)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)  # ms

    avg_latency = np.mean(latencies)
    fps = 1000.0 / avg_latency

    print(f"  Result: {fps:.2f} FPS | Avg Latency: {avg_latency:.2f}ms")
    return fps, avg_latency


def main():
    parser = argparse.ArgumentParser(description="YOLO Speed Benchmark Suite")
    parser.add_argument(
        "--model",
        type=str,
        help="Specific model to benchmark (e.g. yolo11n). Omit to run all.",
    )
    parser.add_argument("--runs", type=int, default=100, help="Number of inference runs")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 precision (PyTorch)")
    parser.add_argument(
        "--export", action="store_true", help="Export to TensorRT FP16 and benchmark"
    )

    args = parser.parse_args()

    results = []
    target_models = [args.model] if args.model else list(MODELS.keys())

    precision = "FP16" if args.fp16 else "FP32"
    print("=== Vision Benchmarks: Speed Benchmark ===")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Precision: {precision} | TensorRT export: {args.export}")

    for name in target_models:
        model_file = MODELS.get(name, name)

        # Download if needed
        try:
            path = download_if_missing(name) if name in MODELS else model_file
        except Exception:
            path = model_file

        # PyTorch benchmark
        fps, latency = benchmark_model(path, runs=args.runs, fp16=args.fp16)
        results.append(
            {"model": name, "format": f"PyTorch {precision}", "fps": fps, "latency": latency}
        )

        # TensorRT benchmark
        if args.export:
            try:
                engine_path = export_tensorrt(path, fp16=True)
                trt_fps, trt_latency = benchmark_model(engine_path, runs=args.runs)
                results.append(
                    {
                        "model": name,
                        "format": "TensorRT FP16",
                        "fps": trt_fps,
                        "latency": trt_latency,
                    }
                )
            except Exception as e:
                print(f"  TensorRT export failed for {name}: {e}")

    # Save results
    Path("results").mkdir(exist_ok=True)
    with open("results/yolo_speed_results.md", "w") as f:
        f.write("# YOLO Speed Benchmarks\n\n")
        f.write(f"**GPU:** {torch.cuda.get_device_name(0)}\n\n")
        f.write("| Model | Format | FPS | Latency (ms) |\n")
        f.write("|-------|--------|----:|--------------:|\n")
        for r in results:
            f.write(f"| {r['model']} | {r['format']} | {r['fps']:.2f} | {r['latency']:.2f} |\n")

    print("\nResults saved to results/yolo_speed_results.md")


if __name__ == "__main__":
    main()
