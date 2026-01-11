# Test Infrastructure

This document describes the testing setup for the vision-benchmarks repository.

## Overview

The test infrastructure is designed to support benchmarking of vision AI models with a focus on:

- Image generation for testing object detection, OCR, and segmentation
- Temporary file/directory management
- Fixture reusability across benchmark tests

## Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures
└── test_infrastructure.py   # Infrastructure validation tests
```

## Available Fixtures

### Image Generation

#### `sample_image()`

Factory fixture to create solid color images.

```python
def test_example(sample_image):
    img = sample_image(width=640, height=480, color=(255, 0, 0))
    # Returns PIL.Image.Image
```

#### `sample_image_with_objects()`

Factory fixture to create images with geometric shapes for object detection testing.

```python
def test_detection(sample_image_with_objects):
    img = sample_image_with_objects(width=640, height=480, num_objects=3)
    # Returns PIL.Image.Image with colored rectangles
```

#### `sample_image_with_text()`

Factory fixture to create images with text for OCR testing.

```python
def test_ocr(sample_image_with_text):
    img = sample_image_with_text(text="HELLO WORLD", font_size=48)
    # Returns PIL.Image.Image with centered text
```

#### `numpy_image()`

Factory fixture to create numpy array images for frameworks that use numpy.

```python
def test_numpy(numpy_image):
    img = numpy_image(width=640, height=480, channels=3)
    # Returns np.ndarray (height, width, channels), dtype uint8
```

### File Management

#### `temp_image_path`

Provides a temporary file path for saving test images. Automatically cleaned up after test.

```python
def test_save(temp_image_path, sample_image):
    img = sample_image()
    img.save(temp_image_path)
    # Path is cleaned up automatically
```

#### `temp_dir`

Provides a temporary directory for test outputs. Automatically cleaned up after test.

```python
def test_outputs(temp_dir):
    output_file = temp_dir / "result.txt"
    output_file.write_text("test")
    # Directory is cleaned up automatically
```

#### `save_test_image()`

Factory fixture to save PIL images to the temp directory.

```python
def test_save_multiple(save_test_image, sample_image):
    img = sample_image()
    path = save_test_image(img, "output.png")
    # Returns Path to saved file
```

## Running Tests

### Basic Test Run

```bash
source venv/bin/activate
pytest tests/ -v
```

### With Coverage

```bash
source venv/bin/activate
pytest tests/ --cov=. --cov-branch --cov-report=term-missing
```

### Lint Check

```bash
source venv/bin/activate
ruff check .
```

## Coverage Targets

- **Minimum**: 80% (blocks release if below)
- **Target**: 90%+
- **Current**: 98%

Coverage is configured in `pyproject.toml` with branch coverage enabled.

## Adding New Benchmark Tests

When adding new benchmark scripts:

1. Create test file: `tests/test_<benchmark_name>.py`
2. Use existing fixtures from `conftest.py`
3. Add new fixtures to `conftest.py` if needed
4. Run tests with coverage to ensure >80%
5. Ensure linting passes with `ruff check .`

## CI Integration

Tests run automatically on:

- Push to `main`
- Pull requests to `main`

CI workflow includes:

- Linting with ruff
- Testing with pytest
- Coverage reporting to Codecov
- Branch coverage analysis

See `.github/workflows/ci.yml` for details.
