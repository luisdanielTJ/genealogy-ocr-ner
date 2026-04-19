import pytest
from src.pipeline.config import PipelineConfig
from src.pipeline.pipeline import run
from src.data.schemas import PipelineResult


@pytest.fixture
def tesseract_config():
    return PipelineConfig(
        ocr_model="tesseract",
        confidence_threshold=0.0,
        max_sequence_length=128,
    )


def test_run_returns_pipeline_result(simple_text_image, tesseract_config):
    result = run(simple_text_image, tesseract_config)
    assert isinstance(result, PipelineResult)


def test_run_populates_raw_text(simple_text_image, tesseract_config):
    result = run(simple_text_image, tesseract_config)
    assert isinstance(result.raw_text, str)


def test_run_model_used_matches_config(simple_text_image, tesseract_config):
    result = run(simple_text_image, tesseract_config)
    assert result.model_used == "tesseract"


def test_run_entities_is_list(simple_text_image, tesseract_config):
    result = run(simple_text_image, tesseract_config)
    assert isinstance(result.entities, list)
