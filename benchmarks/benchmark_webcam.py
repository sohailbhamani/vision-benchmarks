import argparse
import time

import cv2
import numpy as np
from ultralytics import YOLO


def benchmark_latency(source=0, model_path="yolo11n.pt", frames=200):
    print(f"Opening camera source {source}...")
    cap = cv2.VideoCapture(source)

    # Optimize for latency: 720p, MJPG, minimal buffer
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # cap.set(cv2.CAP_PROP_FPS, 60) # Optional: Enable if supported
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Critical for latency

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Diagnostics
    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"📷 Camera Config: {int(actual_w)}x{int(actual_h)} @ {actual_fps} FPS (Buffer=1)")

    print(f"Loading model {model_path}...")
    model = YOLO(model_path)

    print("Warming up...")
    for _ in range(30):
        ret, frame = cap.read()
        if ret:
            model(frame, verbose=False)

    print(f"Starting Glass-to-Glass Latency Test ({frames} frames)...")

    latencies = []

    for _ in range(frames):
        # 1. Capture Start
        t0 = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            break

        # 2. Inference
        t1 = time.perf_counter()
        _ = model(frame, verbose=False)

        # 3. Simulate Render/Display (draw boxes)
        t2 = time.perf_counter()
        # _ = results[0].plot() # Unused, removed for linting

        # 4. Total Time
        t3 = time.perf_counter()

        total_latency = (t3 - t0) * 1000
        # inference_time = (t2 - t1) * 1000 # Unused

        latencies.append(total_latency)

        # GUI Calls removed for headless support
        # cv2.imshow('Benchmark', res_plotted)
        # if cv2.waitKey(1) == ord('q'):
        #    break

    cap.release()
    # cv2.destroyAllWindows()

    avg_lat = np.mean(latencies)
    min_lat = np.min(latencies)
    max_lat = np.max(latencies)

    print(f"\n📊 Results for {model_path} @ 720p")
    print(f"Average Latency: {avg_lat:.2f} ms")
    print(f"Min: {min_lat:.2f} ms | Max: {max_lat:.2f} ms")
    print(f"Theoretical Max FPS: {1000 / avg_lat:.2f}")

    # Save
    with open("results/webcam_latency_results.md", "a") as f:
        f.write(f"| {model_path} | 720p | {avg_lat:.2f} | {min_lat:.2f} | {max_lat:.2f} |\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=int, default=0, help="Camera index")
    parser.add_argument("--model", type=str, default="yolo11n.pt")
    args = parser.parse_args()

    benchmark_latency(args.source, args.model)
