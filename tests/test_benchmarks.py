import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from benchmarks.benchmark_accuracy import benchmark_accuracy
from benchmarks.benchmark_webcam import benchmark_latency
from benchmarks.benchmark_yolo import benchmark_model

# benchmark_webcam requires hardware, so we might skip it or mock it


def test_benchmark_webcam_import():
    """Verify Webcam Benchmark function is importable."""
    assert benchmark_latency is not None


def test_benchmark_yolo_import():
    """Verify Yolo Speed Benchmark function is importable."""
    assert benchmark_model is not None


def test_benchmark_accuracy_import():
    """Verify Accuracy Benchmark function is importable."""
    assert benchmark_accuracy is not None


@pytest.mark.skipif(not Path("yolo11n.pt").exists(), reason="Requires yolo11n.pt model")
def test_benchmark_yolo_dry_run():
    """Smoke test: Run 1 inference on Nano model."""
    fps, latency = benchmark_model("yolo11n.pt", warmup=1, runs=1)
    assert fps > 0
    assert latency > 0


@pytest.mark.skip(reason="Requires COCO dataset")
def test_benchmark_accuracy_dry_run():
    """Smoke test: Accuracy check on small data."""
    # We skip this in CI usually unless we mock ultralytics
    pass


def test_requirements_file():
    """Verify requirements.txt exists."""
    req = Path(__file__).parent.parent / "requirements.txt"
    assert req.exists()
