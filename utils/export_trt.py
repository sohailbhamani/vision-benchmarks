import argparse

from ultralytics import YOLO


def export_to_tensorrt(model_path, fp16=True):
    """
    Export a YOLO model to TensorRT engine format.
    """
    print(f"Loading model: {model_path}")
    try:
        model = YOLO(model_path)

        print(f"Starting TensorRT export (FP16={fp16})...")
        # Export the model
        # dynamic=True often helps with varying batch sizes, but for fixed benchmarks static is fine.
        # simple=True is default for benchmarks.
        path = model.export(format="engine", half=fp16, device=0, verbose=True)

        print(f"Export successful: {path}")
        return path
    except Exception as e:
        print(f"Error exporting model: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLO model to TensorRT")
    parser.add_argument("--model", type=str, required=True, help="Path to .pt model file")
    parser.add_argument("--fp16", action="store_true", help="Enable half-precision (FP16) export")

    args = parser.parse_args()

    export_to_tensorrt(args.model, args.fp16)
