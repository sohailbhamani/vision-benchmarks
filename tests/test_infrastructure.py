"""Basic tests for vision-benchmarks infrastructure.

These tests verify that the test infrastructure is working correctly.
Actual benchmark tests will be added as benchmark scripts are developed.
"""

import numpy as np
from PIL import Image


def test_sample_image_fixture(sample_image):
    """Test that sample_image fixture creates valid images."""
    img = sample_image(width=320, height=240, color=(255, 0, 0))

    assert isinstance(img, Image.Image)
    assert img.size == (320, 240)
    assert img.mode == "RGB"

    # Check that the image is the correct color
    pixels = np.array(img)
    assert pixels.shape == (240, 320, 3)
    assert np.all(pixels == [255, 0, 0])


def test_sample_image_with_objects_fixture(sample_image_with_objects):
    """Test that sample_image_with_objects fixture creates images with shapes."""
    img = sample_image_with_objects(width=640, height=480, num_objects=3)

    assert isinstance(img, Image.Image)
    assert img.size == (640, 480)
    assert img.mode == "RGB"

    # Verify image is not blank (has some variation)
    pixels = np.array(img)
    assert pixels.std() > 0, "Image should have variation (not blank)"


def test_sample_image_with_text_fixture(sample_image_with_text):
    """Test that sample_image_with_text fixture creates images with text."""
    img = sample_image_with_text(text="TEST", width=400, height=300)

    assert isinstance(img, Image.Image)
    assert img.size == (400, 300)
    assert img.mode == "RGB"

    # Verify image is not blank
    pixels = np.array(img)
    assert pixels.std() > 0, "Image should have text (not blank)"


def test_numpy_image_fixture(numpy_image):
    """Test that numpy_image fixture creates valid numpy arrays."""
    img = numpy_image(width=320, height=240, channels=3)

    assert isinstance(img, np.ndarray)
    assert img.shape == (240, 320, 3)
    assert img.dtype == np.uint8
    assert img.min() >= 0
    assert img.max() <= 255


def test_temp_image_path_fixture(temp_image_path, sample_image):
    """Test that temp_image_path fixture provides a valid temporary path."""
    import os

    # Path should exist (created by fixture)
    assert isinstance(temp_image_path, str)
    assert temp_image_path.endswith(".jpg")

    # Should be able to save an image to it
    img = sample_image()
    img.save(temp_image_path)

    assert os.path.exists(temp_image_path)
    assert os.path.getsize(temp_image_path) > 0


def test_temp_dir_fixture(temp_dir):
    """Test that temp_dir fixture provides a valid temporary directory."""
    from pathlib import Path

    assert isinstance(temp_dir, Path)
    assert temp_dir.exists()
    assert temp_dir.is_dir()

    # Should be able to create files in it
    test_file = temp_dir / "test.txt"
    test_file.write_text("test content")
    assert test_file.exists()


def test_save_test_image_fixture(save_test_image, sample_image):
    """Test that save_test_image fixture saves images correctly."""
    img = sample_image(width=100, height=100, color=(0, 255, 0))

    saved_path = save_test_image(img, "test_output.png")

    assert saved_path.exists()
    assert saved_path.name == "test_output.png"

    # Verify saved image can be loaded
    loaded_img = Image.open(saved_path)
    assert loaded_img.size == (100, 100)
