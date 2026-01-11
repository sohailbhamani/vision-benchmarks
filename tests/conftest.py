"""Shared test fixtures for vision-benchmarks tests."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont


@pytest.fixture
def temp_image_path():
    """Create a temporary file path for test images."""
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    import shutil
    import tempfile

    dirpath = tempfile.mkdtemp()
    yield Path(dirpath)
    shutil.rmtree(dirpath, ignore_errors=True)


@pytest.fixture
def sample_image():
    """Generate a simple test image (RGB, 640x480)."""

    def _create(width=640, height=480, color=(128, 128, 128)):
        """Create a solid color image.

        Args:
            width: Image width in pixels
            height: Image height in pixels
            color: RGB tuple (0-255)

        Returns:
            PIL.Image.Image
        """
        return Image.new("RGB", (width, height), color)

    return _create


@pytest.fixture
def sample_image_with_objects():
    """Generate test images with simple geometric objects for detection testing."""

    def _create(width=640, height=480, num_objects=3):
        """Create an image with simple geometric shapes.

        Args:
            width: Image width
            height: Image height
            num_objects: Number of shapes to draw

        Returns:
            PIL.Image.Image
        """
        img = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Draw simple rectangles as "objects"
        for i in range(num_objects):
            x1 = (i * width // num_objects) + 20
            y1 = height // 4
            x2 = x1 + (width // num_objects) - 40
            y2 = 3 * height // 4

            # Different colors for each object
            color = (
                (255, 0, 0)
                if i % 3 == 0
                else (0, 255, 0)
                if i % 3 == 1
                else (0, 0, 255)
            )
            draw.rectangle([x1, y1, x2, y2], fill=color, outline=(0, 0, 0), width=2)

        return img

    return _create


@pytest.fixture
def sample_image_with_text():
    """Generate test images with text for OCR testing."""

    def _create(text="HELLO WORLD", width=640, height=480, font_size=48):
        """Create an image with text.

        Args:
            text: Text to render
            width: Image width
            height: Image height
            font_size: Font size in points

        Returns:
            PIL.Image.Image
        """
        img = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Use default font (PIL's built-in)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size
            )
        except OSError:
            # Fallback to default font if DejaVu not available
            font = ImageFont.load_default()

        # Center the text
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (width - text_width) // 2
        y = (height - text_height) // 2

        draw.text((x, y), text, fill=(0, 0, 0), font=font)

        return img

    return _create


@pytest.fixture
def numpy_image():
    """Generate a numpy array image (for frameworks that use numpy)."""

    def _create(width=640, height=480, channels=3):
        """Create a random numpy image.

        Args:
            width: Image width
            height: Image height
            channels: Number of color channels (3 for RGB)

        Returns:
            np.ndarray with shape (height, width, channels), dtype uint8
        """
        return np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)

    return _create


@pytest.fixture
def save_test_image(temp_dir):
    """Factory fixture to save PIL images to temp directory."""
    saved_files = []

    def _save(image: Image.Image, filename: str) -> Path:
        """Save an image to the temp directory.

        Args:
            image: PIL Image to save
            filename: Filename (will be saved in temp_dir)

        Returns:
            Path to saved file
        """
        filepath = temp_dir / filename
        image.save(filepath)
        saved_files.append(filepath)
        return filepath

    yield _save

    # Cleanup handled by temp_dir fixture
