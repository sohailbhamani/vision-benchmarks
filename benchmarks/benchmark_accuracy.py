import argparse

from ultralytics import YOLO


def benchmark_accuracy(model_path="yolo11n.pt", data="coco128.yaml"):
    """
    Run validation on COCO dataset to measure mAP.
    data: 'coco128.yaml' (small subset) or 'coco.yaml' (full).
    """
    print(f"Benchmarking Accuracy for {model_path} on {data}...")

    model = YOLO(model_path)

    # Run validation
    metrics = model.val(data=data, split="val", verbose=True)

    map50 = metrics.box.map50
    map5095 = metrics.box.map
    precision = metrics.box.mp
    recall = metrics.box.mr

    print(f"\n✅ Results: mAP@50={map50:.3f}, mAP@50-95={map5095:.3f}")

    # Save
    with open("results/yolo_accuracy_results.md", "a") as f:
        f.write(
            f"| {model_path} | {data} | {map50:.3f} | {map5095:.3f} | {precision:.3f} | {recall:.3f} |\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolo11n.pt")
    parser.add_argument(
        "--data", type=str, default="coco128.yaml", help="Dataset yaml (coco128.yaml or coco.yaml)"
    )
    args = parser.parse_args()

    benchmark_accuracy(args.model, args.data)
