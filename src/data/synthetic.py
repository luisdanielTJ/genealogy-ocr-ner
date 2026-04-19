import random
import numpy as np
from PIL import Image, ImageDraw
from faker import Faker
from src.data.schemas import Document, Entity

fake = Faker()
_RELATIONSHIPS = ["Father", "Mother", "Son", "Daughter", "Spouse", "Sibling", "Grandparent", "Grandchild"]
_FIELD_ORDER = ["NAME", "DATE", "LOCATION", "AGE", "RELATIONSHIP"]


def generate_document() -> Document:
    fields = _generate_fields()
    image, entities, full_text = _render(fields)
    return Document(image=image, text=full_text, entities=entities, source="synthetic")


def generate_dataset(n: int = 1500, seed: int | None = None) -> list[Document]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        Faker.seed(seed)
    return [generate_document() for _ in range(n)]


def _generate_fields() -> list[tuple[str, str]]:
    generators = {
        "NAME": lambda: fake.name(),
        "DATE": lambda: fake.date_of_birth(minimum_age=0, maximum_age=100).strftime("%B %d, %Y"),
        "LOCATION": lambda: f"{fake.city()}, {fake.state()}",
        "AGE": lambda: str(random.randint(1, 95)),
        "RELATIONSHIP": lambda: random.choice(_RELATIONSHIPS),
    }
    return [(label, generators[label]()) for label in _FIELD_ORDER]


def _render(fields: list[tuple[str, str]]) -> tuple[Image.Image, list[Entity], str]:
    img = Image.new("RGB", (700, 350), color=(255, 250, 235))
    draw = ImageDraw.Draw(img)

    entities: list[Entity] = []
    full_text = ""
    y = 40

    for label, value in fields:
        prefix = f"{label.title()}: "
        line = prefix + value
        draw.text((50, y), line, fill=(20, 20, 20))

        start = len(full_text) + len(prefix)
        end = start + len(value)
        full_text += line + " "

        entities.append(Entity(label=label, value=value, confidence=1.0, span=(start, end)))
        y += 50

    img = _age(img)
    return img, entities, full_text.rstrip()


def _age(img: Image.Image) -> Image.Image:
    arr = np.array(img, dtype=np.float32)
    arr += np.random.normal(0, 6, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
