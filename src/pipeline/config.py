from dataclasses import dataclass
import yaml


@dataclass
class PipelineConfig:
    ocr_model: str
    confidence_threshold: float
    max_sequence_length: int


def load_config(path: str) -> PipelineConfig:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return PipelineConfig(**cfg)
