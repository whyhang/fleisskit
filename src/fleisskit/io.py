import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from fleisskit.models import Annotation, AnnotationCollection


class DataValidationError(Exception):
    """Raised when invalid data."""


def _parse_bool(val: str) -> bool:
    val_lower = val.strip().lower()
    if val_lower == "true":
        return True
    if val_lower == "false":
        return False
    raise ValueError(f"Expected true/false, got '{val}'")


def _convert_types(row: dict[str, Any]) -> dict[str, Any]:
    """Raw strings get converted to right datatypes"""
    clean_row: dict[str, Any] = {}
    for key, val in row.items():
        # Empty is None
        if val == "" or val is None:
            clean_row[key] = None
            continue

        if isinstance(val, str):
            val = val.strip()
            if val == "":
                # Empty is None
                clean_row[key] = None
                continue

        try:
            if key == "is_gold":
                clean_row[key] = _parse_bool(val) if isinstance(val, str) else bool(val)
            elif key in ("confidence", "duration_sec"):
                clean_row[key] = float(val)
            elif key == "annotated_at":
                clean_row[key] = datetime.fromisoformat(val)
            else:
                clean_row[key] = val
        except ValueError as e:
            raise ValueError(f"Invalid format for {key}: {e}")

    return clean_row


def _process_rows(
    rows: list[dict[str, Any]], start_row: int = 1
) -> AnnotationCollection:
    """validates the rows and creates the Annotation object"""
    errors = []
    annotations = []
    seen_keys: dict[
        tuple[str, str], int
    ] = {}  # (item_id, annotator_id) -> row_number, For tracking duplicates

    for idx, row in enumerate(rows, start=start_row):
        try:
            clean_row = _convert_types(row)
            item_id = clean_row.get("item_id")
            annotator_id = clean_row.get("annotator_id")

            # Check for duplicates
            if item_id and annotator_id:
                key = (item_id, annotator_id)
                if key in seen_keys:
                    prev_row = seen_keys[key]
                    errors.append(
                        f"Row {idx}: Duplicate annotation for item '{item_id}' by '{annotator_id}' (first seen on row {prev_row})"
                    )
                    continue
                seen_keys[key] = idx

            annotations.append(Annotation(**clean_row))

        except (ValueError, TypeError) as e:
            errors.append(f"Row {idx}: {e!s}")

    if errors:
        # Report every bad row at once
        raise DataValidationError("Errors found in data:\n" + "\n".join(errors))

    return AnnotationCollection(annotations)


def _check_headers(headers: list[str]):
    """First row header check"""
    required = {"item_id", "annotator_id", "is_gold"}
    missing = required - set(headers)
    if missing:
        raise ValueError(f"Missing required columns in header: {', '.join(missing)}")


def load_csv(file_path: str | Path) -> AnnotationCollection:
    """Loads annotations from a CSV file."""
    path = Path(file_path)

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV file is empty or missing headers.")

        _check_headers(list(reader.fieldnames))

        # start=2 because row 1 is the header
        return _process_rows(list(reader), start_row=2)


def load_json(file_path: str | Path) -> AnnotationCollection:
    """Loads annotations from a JSON file (list of objects)."""
    path = Path(file_path)

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise TypeError("JSON file must contain a list of annotation objects.")

    if len(data) > 0:
        _check_headers(list(data[0].keys()))

    # start=1 , 0 is headers
    return _process_rows(data, start_row=1)
