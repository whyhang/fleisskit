import csv

import pytest

from fleisskit.generator import generate_synthetic_data
from fleisskit.io import load_csv


def test_generator_determinism():
    """Same seed must always yield the exact same labels and annotations."""
    col1, true1 = generate_synthetic_data(20, 5, ["A", "B", "C"], seed=42)
    col2, true2 = generate_synthetic_data(20, 5, ["A", "B", "C"], seed=42)

    assert true1 == true2
    assert len(col1.annotations) == len(col2.annotations)
    for a1, a2 in zip(col1.annotations, col2.annotations):
        assert a1.label == a2.label
        assert a1.duration_sec == a2.duration_sec


def test_generator_accuracy_applied_correctly():
    """At 100% accuracy with no spammers, every label must be true label."""
    col, true_labels = generate_synthetic_data(
        num_items=50,
        num_annotators=3,
        labels=["Pos", "Neg"],
        accuracy=1.0,
        spammer_share=0.0,
        seed=100,
    )

    for ann in col.annotations:
        assert ann.label == true_labels[ann.item_id]


def test_generator_skip_rate():
    """A 50% skip rate should exactly halve the total expected annotations."""
    col, _ = generate_synthetic_data(
        num_items=100, num_annotators=10, labels=["A", "B"], skip_rate=0.5, seed=123
    )

    # 100 items * 10 annotators = 1000 max annotations
    # 50% skip rate must leave exactly 500 annotations
    assert len(col.annotations) == 500


def test_generator_spammer_behavior():
    """Spammers complete tasks suspiciously fast and stick to one label."""
    col, _ = generate_synthetic_data(
        num_items=20,
        num_annotators=2,
        labels=["A", "B", "C"],
        spammer_share=1.0,  # 100% of annotators are spammers
        seed=77,
    )

    # Extract the labels chosen by each annotator
    annotator_labels = {"annotator_0": set(), "annotator_1": set()}

    for ann in col.annotations:
        # Spammers are fast
        assert ann.duration_sec is not None
        assert ann.duration_sec < 2.5

        annotator_labels[ann.annotator_id].add(ann.label)

    # should only ever use a single label across all their tasks
    assert len(annotator_labels["annotator_0"]) == 1
    assert len(annotator_labels["annotator_1"]) == 1


def test_generator_csv_export_comprehensive(tmp_path):
    """The generator should successfully export a readable CSV that matches the generated collection exactly."""
    csv_file = tmp_path / "synthetic.csv"

    # Generate data
    orig_col, _ = generate_synthetic_data(
        num_items=5, num_annotators=2, labels=["A", "B"], output_csv=csv_file, seed=1
    )

    assert csv_file.exists()

    # Check raw CSV structure and explicit string formatting rules
    with csv_file.open("r", encoding="utf-8") as f:
        raw_rows = list(csv.DictReader(f))

    assert len(raw_rows) == 10  # 5 items * 2 annotators (0 skip rate)

    first_row = raw_rows[0]
    assert first_row["is_gold"] == "false"  # Must be strictly lowercase
    assert first_row["batch_id"] == ""  # None values must be empty strings, not "None"
    assert first_row["gold_label"] == ""

    # Verify decimal rounding (generator rounds confidence to 4, duration to 2)
    assert len(first_row["confidence"].split(".")[-1]) <= 4
    assert len(first_row["duration_sec"].split(".")[-1]) <= 2

    # Test round-trip integrity using our own loader
    loaded_col = load_csv(csv_file)

    assert len(loaded_col.annotations) == len(orig_col.annotations)

    # comparing each loaded annotation back to the original objects
    for orig, loaded in zip(orig_col.annotations, loaded_col.annotations):
        assert loaded.item_id == orig.item_id
        assert loaded.annotator_id == orig.annotator_id
        assert loaded.is_gold == orig.is_gold
        assert loaded.label == orig.label

        # To cover roudnding we use pytest.approx to compare them safely
        assert loaded.confidence == pytest.approx(orig.confidence, abs=1e-4)
        assert loaded.duration_sec == pytest.approx(orig.duration_sec, abs=1e-2)

        assert loaded.batch_id is None
        assert loaded.gold_label is None
