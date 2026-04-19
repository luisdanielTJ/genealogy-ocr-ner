from PIL import Image
from src.data.preprocessing import preprocess


def test_preprocess_returns_pil_image(simple_text_image):
    result = preprocess(simple_text_image)
    assert isinstance(result, Image.Image)


def test_preprocess_converts_to_grayscale_then_back(simple_text_image):
    result = preprocess(simple_text_image)
    # Output is grayscale (single channel) converted back to RGB-compatible
    assert result.mode in ("L", "RGB")


def test_preprocess_noisy_image_returns_image(noisy_image):
    result = preprocess(noisy_image)
    assert isinstance(result, Image.Image)
    assert result.size == noisy_image.size
