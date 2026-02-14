import argparse
from pathlib import Path

from ultralytics import YOLO

MODELS = [
    "yolo11n.pt",
    "yolo11s.pt",
    "yolo11m.pt",
    "yolo11l.pt",
    "yolo12n.pt",
    "yolo12s.pt",
    "yolo26n.pt",
    "yolo26s.pt",
    "yolov8s-world.pt",
    "yolov8m-world.pt",
]


def benchmark_accuracy(model_path, data="coco128.yaml"):
    """Run validation on COCO dataset to measure mAP."""
    print(f"\nBenchmarking accuracy for {model_path} on {data}...")

    model = YOLO(model_path)
    metrics = model.val(data=data, split="val", verbose=True)

    map50 = metrics.box.map50
    map5095 = metrics.box.map
    precision = metrics.box.mp
    recall = metrics.box.mr

    print(f"  Result: mAP@50={map50:.3f}, mAP@50-95={map5095:.3f}")
    return {
        "model": model_path,
        "data": data,
        "map50": map50,
        "map5095": map5095,
        "precision": precision,
        "recall": recall,
    }


def main():
    parser = argparse.ArgumentParser(description="YOLO Accuracy Benchmark Suite")
    parser.add_argument("--model", type=str, help="Specific model (e.g. yolo11n.pt). Omit for all.")
    parser.add_argument("--data", type=str, default="coco128.yaml", help="Dataset yaml")
    args = parser.parse_args()

    target_models = [args.model] if args.model else MODELS
    results = []

    for model_path in target_models:
        try:
            r = benchmark_accuracy(model_path, args.data)
            results.append(r)
        except Exception as e:
            print(f"  Failed for {model_path}: {e}")

    # Save results
    Path("results").mkdir(exist_ok=True)
    with open("results/yolo_accuracy_results.md", "w") as f:
        f.write("# YOLO Accuracy Benchmarks\n\n")
        f.write(f"**Dataset:** {args.data}\n\n")
        f.write("| Model | Dataset | mAP@50 | mAP@50-95 | Precision | Recall |\n")
        f.write("|-------|---------|-------:|----------:|----------:|-------:|\n")
        for r in results:
            f.write(
                f"| {r['model']} | {r['data']} "
                f"| {r['map50']:.3f} | {r['map5095']:.3f} "
                f"| {r['precision']:.3f} | {r['recall']:.3f} |\n"
            )

    print("\nResults saved to results/yolo_accuracy_results.md")


if __name__ == "__main__":
    main()
