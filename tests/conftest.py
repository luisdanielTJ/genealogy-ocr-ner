import pytest
from PIL import Image, ImageDraw, ImageFont


@pytest.fixture
def simple_text_image():
    """White 400x100 image with black 'John Smith 1850' text — readable by Tesseract."""
    img = Image.new("RGB", (400, 100), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 35), "John Smith 1850", fill="black")
    return img


@pytest.fixture
def noisy_image():
    """Slightly rotated and noisy version of the simple image."""
    import numpy as np
    img = Image.new("RGB", (400, 100), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 35), "John Smith 1850", fill="black")
    arr = np.array(img, dtype=np.float32)
    arr += np.random.normal(0, 15, arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).rotate(3)
