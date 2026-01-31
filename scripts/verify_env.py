import sys

import onnxruntime as ort
import torch
import ultralytics
from ultralytics import YOLO


def verify():
    print(f"Python Version: {sys.version}")
    print("-" * 20)

    # PyTorch
    torch_available = torch.cuda.is_available()
    print(f"PyTorch Version: {torch.__version__}")
    print(f"PyTorch CUDA Available: {torch_available}")
    if torch_available:
        print(f"PyTorch Device Name: {torch.cuda.get_device_name(0)}")

    # ONNX Runtime
    print("-" * 20)
    available_providers = ort.get_available_providers()
    onnx_gpu = "CUDAExecutionProvider" in available_providers
    print(f"ONNX Runtime Version: {ort.__version__}")
    print(f"ONNX Runtime Providers: {available_providers}")
    print(f"ONNX Runtime GPU Available: {onnx_gpu}")

    # TensorRT
    print("-" * 20)
    try:
        import tensorrt as trt

        print(f"TensorRT Version: {trt.__version__}")
    except ImportError as e:
        print(f"TensorRT Import Failed: {e}")

    # PaddlePaddle
    print("-" * 20)
    try:
        import paddle

        paddle_gpu = paddle.is_compiled_with_cuda()
        print(f"Paddle Version: {paddle.__version__}")
        print(f"Paddle GPU Available: {paddle_gpu}")
        if paddle_gpu:
            print(f"Paddle CUDA Version: {paddle.version.cuda()}")
    except Exception as e:
        print(f"Paddle Verification Failed: {e}")

    # Ultralytics
    print("-" * 20)
    print(f"Ultralytics Version: {ultralytics.__version__}")
    try:
        # Simple test load
        _ = YOLO("yolov8n.pt")
        print("Ultralytics YOLO model load: SUCCESS")
    except Exception as e:
        print(f"Ultralytics YOLO model load: FAILED: {e}")


if __name__ == "__main__":
    verify()
