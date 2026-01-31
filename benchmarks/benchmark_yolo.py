import argparse
import time
import torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# Benchmark Configuration
MODELS = {
    "yolo11n": "yolo11n.pt",
    "yolo11s": "yolo11s.pt",
    "yolo11m": "yolo11m.pt",
    "yolov8s-world": "yolov8s-world.pt",
    "yolov8m-world": "yolov8m-world.pt",
}


def download_if_missing(model_name):
    """Ensure model weights exist locally."""
    path = Path(f"{model_name}.pt")
    if not path.exists():
        print(f"⬇️ Downloading {model_name}...")
        # Ultralytics auto-downloads on load, but we explicit load to trigger it
        YOLO(f"{model_name}.pt")
    return str(path)


def benchmark_model(model_path, device="cuda", warmup=10, runs=100, fp16=False):
    """Run inference benchmark loop."""
    print(f"\nBenchmarking {model_path} on {device} (FP16={fp16})...")

    # Load Model
    model = YOLO(model_path)

    # Dummy Input (1, 3, 640, 640)
    img = np.zeros((640, 640, 3), dtype=np.uint8)

    # Warmup
    print("  🔥 Warming up...")
    for _ in range(warmup):
        model(img, verbose=False, half=fp16)

    # Benchmark Loop
    print(f"  🚀  Running {runs} inferences...")
    latencies = []

    # Start Timer
    start_global = time.perf_counter()

    for _ in range(runs):
        t0 = time.perf_counter()
        results = model(img, verbose=False, half=fp16)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)  # ms

    end_global = time.perf_counter()

    avg_latency = np.mean(latencies)
    fps = 1.0 / (avg_latency / 1000.0)

    print(f"  ✅ Result: {fps:.2f} FPS | Avg Latency: {avg_latency:.2f}ms")
    return fps, avg_latency


def main():
    parser = argparse.ArgumentParser(description="YOLO Speed Benchmark Suite")
    parser.add_argument(
        "--model",
        type=str,
        help="Specific model to buffer (e.g. yolo11n.pt). Unset runs all standard.",
    )
    parser.add_argument("--runs", type=int, default=100, help="Number of inference runs")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 precision")
    parser.add_argument("--export", action="store_true", help="Also benchmark TensorRT export")

    args = parser.parse_args()

    results = []

    target_models = [args.model] if args.model else MODELS.keys()

    print("=== Vision Benchmarks: Speed Benchmark ===")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    for name in target_models:
        model_file = MODELS.get(name, name)  # Handle custom path if passed manually

        # Download/Load
        try:
            path = download_if_missing(name) if name in MODELS else model_file
        except:
            path = model_file  # file path passed directly

        # Run PyTorch Benchmark
        fps, latency = benchmark_model(path, runs=args.runs, fp16=args.fp16)
        results.append(f"| {name} | PyTorch | {fps:.2f} | {latency:.2f} |")

        # Run TensorRT Benchmark (Optional)
        if args.export:
            # TODO: Logic to invoke export_trt.py and benchmark engine
            pass

    # Save Results
    with open("results/yolo_speed_results.md", "w") as f:
        f.write("# YOLO Speed Benchmarks\n\n")
        f.write("| Model | Format | FPS | Latency (ms) |\n")
        f.write("|-------|--------|-----|--------------|\n")
        for line in results:
            f.write(line + "\n")

    print("\n📄 Results saved to results/yolo_speed_results.md")


if __name__ == "__main__":
    main()
