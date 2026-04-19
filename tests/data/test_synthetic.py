from src.data.synthetic import generate_document, generate_dataset
from src.data.schemas import Document, Entity
from PIL import Image


def test_generate_document_returns_document():
    doc = generate_document()
    assert isinstance(doc, Document)
    assert isinstance(doc.image, Image.Image)
    assert doc.source == "synthetic"


def test_generate_document_has_five_entities():
    doc = generate_document()
    assert len(doc.entities) == 5
    labels = {e.label for e in doc.entities}
    assert labels == {"NAME", "DATE", "LOCATION", "AGE", "RELATIONSHIP"}


def test_entity_values_appear_in_text():
    doc = generate_document()
    for entity in doc.entities:
        assert entity.value in doc.text


def test_entity_spans_are_correct():
    doc = generate_document()
    for entity in doc.entities:
        start, end = entity.span
        assert doc.text[start:end] == entity.value


def test_generate_dataset_returns_n_documents():
    docs = generate_dataset(n=5)
    assert len(docs) == 5
    for doc in docs:
        assert isinstance(doc, Document)
