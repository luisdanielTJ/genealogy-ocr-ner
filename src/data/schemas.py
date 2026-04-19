from dataclasses import dataclass, field
from PIL import Image


@dataclass
class Entity:
    label: str
    value: str
    confidence: float
    span: tuple[int, int]


@dataclass
class Document:
    image: Image.Image
    text: str
    entities: list[Entity]
    source: str


@dataclass
class OCRResult:
    text: str
    confidence: float
    model: str


@dataclass
class PipelineResult:
    raw_text: str
    entities: list[Entity]
    ocr_confidence: float
    model_used: str
