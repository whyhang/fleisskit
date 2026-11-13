from dataclasses import FrozenInstanceError
from datetime import datetime

import pytest

from fleisskit.models import Annotation, AnnotationCollection


def test_valid_annotation_created():
    """A valid annotation is created without errors."""
    ann = Annotation(
        item_id="item_001",
        annotator_id="user_A",
        is_gold=False,
        label="positive",
        confidence=0.85,
        duration_in_seconds=12.5,
        annotated_at=datetime(2023, 10, 1, 12, 0, 0),
        batch_id="batch_1",
    )
    assert ann.item_id == "item_001"
    assert ann.label == "positive"
    assert ann.confidence == 0.85


def test_valid_gold_annotation_created():
    """A valid gold annotation is created without errors."""
    ann = Annotation(
        item_id="item_002",
        annotator_id="user_B",
        is_gold=True,
        gold_label="negative",
        label="negative",
    )
    assert ann.is_gold is True
    assert ann.gold_label == "negative"


def test_confidence_outside_range_raises_error():
    """Each bad case raises ValueError: confidence outside 0-1."""
    with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
        Annotation(item_id="1", annotator_id="A", is_gold=False, confidence=1.5)

    with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
        Annotation(item_id="1", annotator_id="A", is_gold=False, confidence=-0.1)


def test_negative_duration_raises_error():
    """Each bad case raises ValueError: negative duration."""
    with pytest.raises(ValueError, match="Duration cannot be negative"):
        Annotation(
            item_id="1", annotator_id="A", is_gold=False, duration_in_seconds=-5.0
        )


def test_missing_gold_label_for_gold_item_raises_error():
    """Each bad case raises ValueError: gold item missing gold label."""
    with pytest.raises(ValueError, match="gold_label is required when is_gold is True"):
        Annotation(item_id="1", annotator_id="A", is_gold=True, gold_label=None)


def test_empty_string_raises_error():
    """Missing means None, never an empty string ''."""
    with pytest.raises(ValueError, match="label cannot be an empty string"):
        Annotation(item_id="1", annotator_id="A", is_gold=False, label="")


def test_frozen_dataclass_prevents_changes():
    """Loaded data shouldn’t be edited by accident."""
    ann = Annotation(item_id="1", annotator_id="A", is_gold=False)
    with pytest.raises(FrozenInstanceError):
        ann.label = "new_label"


def test_collection_lookups():
    """The collection’s lookups return the right results, including for an annotator who labeled only one item."""
    annotations = [
        Annotation(item_id="1", annotator_id="Alice", is_gold=False, label="positive"),
        Annotation(item_id="2", annotator_id="Alice", is_gold=False, label="neutral"),
        Annotation(item_id="3", annotator_id="Bob", is_gold=False, label="negative"),
    ]

    collection = AnnotationCollection(annotations)

    alice_results = collection.get_by_annotator("Alice")
    assert len(alice_results) == 2
    assert [a.item_id for a in alice_results] == ["1", "2"]

    # Testing for an annotator who labeled only one item
    bob_results = collection.get_by_annotator("Bob")
    assert len(bob_results) == 1
    assert bob_results[0].item_id == "3"

    # Testing for an annotator with no items
    charlie_results = collection.get_by_annotator("Charlie")
    assert len(charlie_results) == 0
