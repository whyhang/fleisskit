import json

import pytest

from fleisskit.io import DataValidationError, load_csv, load_json


def test_clean_file_loads_correctly(tmp_path):
    """A clean file loads with the right number of annotations."""
    csv_file = tmp_path / "clean.csv"
    csv_file.write_text(
        "item_id,annotator_id,is_gold,label,confidence\n"
        "item_1,user_1,false,positive,0.9\n"
        "item_2,user_2,false,negative,0.8\n",
        encoding="utf-8",
    )
    collection = load_csv(csv_file)
    assert len(collection.annotations) == 2
    assert collection.annotations[0].item_id == "item_1"
    assert collection.annotations[1].item_id == "item_2"


def test_text_with_comma_inside_quotes(tmp_path):
    """Text with a comma inside quotes loads correctly (testing the csv module)."""
    csv_file = tmp_path / "comma.csv"
    csv_file.write_text(
        "item_id,annotator_id,is_gold,label\n"
        'item_1,user_1,false,"Great price, but slow"\n',
        encoding="utf-8",
    )
    collection = load_csv(csv_file)
    assert len(collection.annotations) == 1
    assert collection.annotations[0].label == "Great price, but slow"


def test_empty_label_loads_as_none(tmp_path):
    """An empty label loads as None."""
    csv_file = tmp_path / "empty.csv"
    # Row has trailing comma => 'label' is empty

    csv_file.write_text(
        "item_id,annotator_id,is_gold,label\nitem_1,user_1,false,\n", encoding="utf-8"
    )

    collection = load_csv(csv_file)
    assert len(collection.annotations) == 1
    assert collection.annotations[0].label is None


def test_missing_column_gives_clear_error(tmp_path):
    """A missing column gives a clear error naming it."""
    csv_file = tmp_path / "missing.csv"
    # Missing must have 'is_gold' and 'annotator_id'
    csv_file.write_text("item_id,label\nitem_1,positive\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required columns in header:"):
        load_csv(csv_file)


def test_file_with_3_bad_rows_reports_all(tmp_path):
    """A file with 3 bad rows reports all 3, with row numbers."""
    csv_file = tmp_path / "bad_rows.csv"
    csv_file.write_text(
        "item_id,annotator_id,is_gold,confidence,duration_sec,gold_label\n"
        "1,A,false,1.5,10,\n"  # Row 2: confidence > 1
        "2,B,false,0.5,-5,\n"  # Row 3: negative duration
        "3,C,true,0.5,10,\n",  # Row 4: is_gold is true but gold_label is empty
        encoding="utf-8",
    )

    with pytest.raises(DataValidationError) as exc_info:
        load_csv(csv_file)

    error_msg = str(exc_info.value)

    # Assert all 3 row numbers are mentioned in the aggregated error message
    assert "Row 2" in error_msg
    assert "Row 3" in error_msg
    assert "Row 4" in error_msg
    # Validate specific error triggers are mentioned
    assert "Confidence must be between 0 and 1" in error_msg
    assert "Duration cannot be negative" in error_msg
    assert "gold_label is required when is_gold is True" in error_msg


def test_duplicate_annotation_is_caught(tmp_path):
    """A duplicate annotation is caught."""
    csv_file = tmp_path / "duplicate.csv"
    csv_file.write_text(
        "item_id,annotator_id,is_gold\n"
        "1,A,false\n"
        "1,A,false\n",  # Duplicate annotator for the same item
        encoding="utf-8",
    )

    with pytest.raises(DataValidationError, match="Row 3: Duplicate annotation"):
        load_csv(csv_file)


def test_csv_and_json_parity(tmp_path):
    """The same data as CSV and as JSON gives identical results."""
    # 1. Setup the identical data
    data = [
        {
            "item_id": "1",
            "annotator_id": "A",
            "is_gold": False,
            "label": "positive",
            "confidence": 0.9,
        },
        {
            "item_id": "2",
            "annotator_id": "B",
            "is_gold": True,
            "gold_label": "neutral",
            "label": "neutral",
        },
    ]

    # 2. Write CSV
    csv_file = tmp_path / "data.csv"
    csv_file.write_text(
        "item_id,annotator_id,is_gold,label,confidence,gold_label\n"
        "1,A,false,positive,0.9,\n"
        "2,B,true,neutral,,neutral\n",
        encoding="utf-8",
    )

    # 3. Write JSON
    json_file = tmp_path / "data.json"
    json_file.write_text(json.dumps(data), encoding="utf-8")

    # 4. Load both
    csv_collection = load_csv(csv_file)
    json_collection = load_json(json_file)

    # 5. Compare lists (requires identical structure and properties)
    assert len(csv_collection.annotations) == len(json_collection.annotations)
    for csv_ann, json_ann in zip(
        csv_collection.annotations, json_collection.annotations
    ):
        assert csv_ann == json_ann
