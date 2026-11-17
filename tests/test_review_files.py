import csv
import json
from collections import Counter
from pathlib import Path

import pytest

from fleisskit.bad_actors import find_bad_actors
from fleisskit.disputes import find_disputes
from fleisskit.io import load_csv, load_json, load_with_rows
from fleisskit.models import Annotation
from fleisskit.review_files import (
    AGREED_FILE,
    DISPUTED_FILE,
    TO_REVIEW_COLUMNS,
    TO_REVIEW_FILE,
    TOO_FEW_VOTES,
    TOO_FEW_VOTES_SEVERITY,
    ReviewFilesExistError,
    split_rows,
    to_review_rows,
    write_review_files,
)

DATA = Path(__file__).parent / "data"
FIXTURES = [DATA / "sample.csv", DATA / "outliers.csv", DATA / "spam_known.csv"]


def _write(path: Path, files_dir: Path, use_bad_actors: bool = False):
    """Runs the whole pipeline on `path` and writes the files to files_dir."""
    collection, source = load_with_rows(path)
    bad = (
        {b.annotator_id for b in find_bad_actors(collection)}
        if use_bad_actors
        else None
    )
    report = find_disputes(collection, bad_actor_ids=bad)
    files = write_review_files(collection, source, report, files_dir)
    return collection, report, files


def _lines(path: Path) -> list[bytes]:
    return path.read_bytes().splitlines(keepends=True)


def _merge_in_input_order(
    source: list[bytes], agreed: list[bytes], disputed: list[bytes]
) -> list[bytes]:
    """
    Rebuilds the input from the two files: walks the input and takes each line
    from whichever file has it next. Fails if a line is missing, duplicated or
    out of order in either file.
    """
    merged, a, d = [], 0, 0
    for line in source:
        if a < len(agreed) and agreed[a] == line:
            merged.append(agreed[a])
            a += 1
        elif d < len(disputed) and disputed[d] == line:
            merged.append(disputed[d])
            d += 1
        else:
            raise AssertionError(f"input line not found in order: {line!r}")
    assert a == len(agreed) and d == len(disputed), "extra lines in the output"
    return merged


def _annotation_counter(annotations: list[Annotation]) -> Counter:
    return Counter(annotations)


# --- load_with_rows ---


def test_load_with_rows_matches_load_csv_row_for_row():
    collection, source = load_with_rows(DATA / "sample.csv")
    assert collection.annotations == load_csv(DATA / "sample.csv").annotations
    assert source.columns == (
        "item_id",
        "annotator_id",
        "is_gold",
        "label",
        "confidence",
        "duration_sec",
        "annotated_at",
        "batch_id",
        "gold_label",
    )
    assert len(source.rows) == len(collection.annotations)
    for ann, row in zip(collection.annotations, source.rows, strict=True):
        assert row["item_id"] == ann.item_id
        assert row["annotator_id"] == ann.annotator_id
    # Raw text, not converted values.
    assert source.rows[0]["confidence"] == "0.7217"
    assert source.rows[0]["is_gold"] == "false"


def test_load_with_rows_json_keeps_every_key_in_first_seen_order(tmp_path):
    path = tmp_path / "data.json"
    path.write_text(
        json.dumps(
            [
                {"item_id": "i1", "annotator_id": "a", "is_gold": False},
                {"annotator_id": "b", "item_id": "i1", "is_gold": False, "label": "x"},
            ]
        ),
        encoding="utf-8",
    )
    collection, source = load_with_rows(path)
    assert collection.annotations == load_json(path).annotations
    assert source.columns == ("item_id", "annotator_id", "is_gold", "label")
    assert source.rows[1]["label"] == "x"


def test_load_with_rows_routes_by_extension(tmp_path):
    path = tmp_path / "DATA.JSON"
    path.write_text(
        '[{"item_id": "i", "annotator_id": "a", "is_gold": false}]', encoding="utf-8"
    )
    collection, _ = load_with_rows(path)
    assert len(collection.annotations) == 1


# --- agreed.csv + disputed.csv are the input ---


@pytest.mark.parametrize("use_bad_actors", [False, True])
@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_agreed_plus_disputed_is_the_input_byte_for_byte(
    path, use_bad_actors, tmp_path
):
    _, report, files = _write(path, tmp_path, use_bad_actors)

    source = _lines(path)
    agreed = _lines(tmp_path / AGREED_FILE)
    disputed = _lines(tmp_path / DISPUTED_FILE)

    # Same header as the input, in both files.
    assert agreed[0] == disputed[0] == source[0]
    # Every input row in exactly one file, unchanged, in input order.
    merged = _merge_in_input_order(source[1:], agreed[1:], disputed[1:])
    assert [source[0], *merged] == source

    # Split by item: disputed.csv holds exactly the items in to_review.csv.
    with (tmp_path / DISPUTED_FILE).open(encoding="utf-8", newline="") as f:
        disputed_ids = {row["item_id"] for row in csv.DictReader(f)}
    with (tmp_path / AGREED_FILE).open(encoding="utf-8", newline="") as f:
        agreed_ids = {row["item_id"] for row in csv.DictReader(f)}
    _, to_review = _read_to_review(tmp_path / TO_REVIEW_FILE)
    assert disputed_ids == {r["Item ID"] for r in to_review}
    assert disputed_ids == {item.item_id for item in report.items}  # all judged
    assert not agreed_ids & disputed_ids

    assert files.agreed.rows == len(agreed) - 1
    assert files.disputed.rows == len(disputed) - 1
    assert files.agreed.items == len(agreed_ids)
    assert files.disputed.items == len(disputed_ids)


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_agreed_plus_disputed_load_back_to_the_input_annotations(path, tmp_path):
    collection, _, _ = _write(path, tmp_path, use_bad_actors=True)
    back = (
        load_csv(tmp_path / AGREED_FILE).annotations
        + load_csv(tmp_path / DISPUTED_FILE).annotations
    )
    assert _annotation_counter(back) == _annotation_counter(collection.annotations)


def test_structure_is_inherited_from_an_unusual_csv(tmp_path):
    """Column order, missing optional columns and raw cell text all survive."""
    path = tmp_path / "odd.csv"
    path.write_bytes(
        b"label,is_gold,annotator_id,item_id,confidence\r\n"
        # The loader strips " pos " and reads False; the raw text is kept.
        b" pos ,False,a,i1,0.50\r\n"
        b"neg,false,b,i1,1\r\n"
        b"pos,false,a,i2,\r\n"
        b"pos,false,b,i2,0.9\r\n"
    )

    _, report, _ = _write(path, tmp_path / "out")
    assert [i.item_id for i in report.items] == ["i1"]

    assert _lines(tmp_path / "out" / DISPUTED_FILE) == [
        b"label,is_gold,annotator_id,item_id,confidence\r\n",
        b" pos ,False,a,i1,0.50\r\n",
        b"neg,false,b,i1,1\r\n",
    ]
    assert _lines(tmp_path / "out" / AGREED_FILE) == [
        b"label,is_gold,annotator_id,item_id,confidence\r\n",
        b"pos,false,a,i2,\r\n",
        b"pos,false,b,i2,0.9\r\n",
    ]


def test_cells_that_need_quoting_are_quoted_and_load_back(tmp_path):
    path = tmp_path / "quoted.csv"
    path.write_text(
        'item_id,annotator_id,is_gold,label\n"i,1",a,false,"x, y"\n"i,1",b,false,z\n',
        encoding="utf-8",
    )
    collection, _, _ = _write(path, tmp_path / "out")
    assert (tmp_path / "out" / DISPUTED_FILE).read_bytes() == (
        b'item_id,annotator_id,is_gold,label\r\n"i,1",a,false,"x, y"\r\n'
        b'"i,1",b,false,z\r\n'
    )
    back = load_csv(tmp_path / "out" / DISPUTED_FILE).annotations
    assert back == collection.annotations


def test_json_input_becomes_csv_with_its_keys_as_columns(tmp_path):
    path = tmp_path / "data.json"
    objects = [
        {
            "item_id": "i1",
            "annotator_id": "a",
            "is_gold": True,
            "gold_label": "x",
            "label": "x",
            "confidence": 0.75,
            "duration_sec": 12,
        },
        {
            "item_id": "i1",
            "annotator_id": "b",
            "is_gold": True,
            "gold_label": "x",
            "label": "y",
            "confidence": None,
        },
        {"item_id": "i2", "annotator_id": "a", "is_gold": False, "label": "x"},
        {
            "item_id": "i2",
            "annotator_id": "b",
            "is_gold": False,
            "label": "x",
            "annotated_at": "2024-01-02T03:04:05",
        },
    ]
    path.write_text(json.dumps(objects), encoding="utf-8")
    collection, report, _ = _write(path, tmp_path / "out")
    assert [i.item_id for i in report.items] == ["i1"]

    header = (
        b"item_id,annotator_id,is_gold,gold_label,label,confidence,duration_sec,"
        b"annotated_at\r\n"
    )
    assert _lines(tmp_path / "out" / DISPUTED_FILE) == [
        header,
        b"i1,a,true,x,x,0.75,12,\r\n",
        b"i1,b,true,x,y,,,\r\n",
    ]
    assert _lines(tmp_path / "out" / AGREED_FILE) == [
        header,
        b"i2,a,false,,x,,,\r\n",
        b"i2,b,false,,x,,,2024-01-02T03:04:05\r\n",
    ]
    back = (
        load_csv(tmp_path / "out" / AGREED_FILE).annotations
        + load_csv(tmp_path / "out" / DISPUTED_FILE).annotations
    )
    assert _annotation_counter(back) == _annotation_counter(collection.annotations)


# --- to_review.csv ---


def _read_to_review(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return tuple(reader.fieldnames or ()), list(reader)


def test_to_review_headers_are_capitalised():
    assert TO_REVIEW_COLUMNS == (
        "Item ID",
        "Severity",
        "Tag",
        "Reason",
        "Label Counts",
        "Winners",
        "Current Label",
        "Gold Labels",
        "Decision",
        "Reviewer",
        "Notes",
    )


def test_to_review_has_one_row_per_disputed_item_worst_first(tmp_path):
    _, report, files = _write(DATA / "spam_known.csv", tmp_path, use_bad_actors=True)
    columns, rows = _read_to_review(tmp_path / TO_REVIEW_FILE)

    assert columns == TO_REVIEW_COLUMNS
    assert (
        (tmp_path / TO_REVIEW_FILE)
        .read_bytes()
        .startswith(
            b"Item ID,Severity,Tag,Reason,Label Counts,Winners,Current Label,"
            b"Gold Labels,Decision,Reviewer,Notes\r\n"
        )
    )
    assert [r["Item ID"] for r in rows] == [item.item_id for item in report.items]
    assert len({r["Item ID"] for r in rows}) == len(rows) == 63
    assert rows == to_review_rows(report)
    assert files.to_review.rows == files.to_review.items == 63
    assert (files.disputed_items, files.too_few_votes_items) == (63, 0)
    severities = [int(r["Severity"]) for r in rows]
    assert severities == sorted(severities)


def test_to_review_gives_the_most_severe_tag_and_its_reason(tmp_path):
    _, report, _ = _write(DATA / "spam_known.csv", tmp_path, use_bad_actors=True)
    _, rows = _read_to_review(tmp_path / TO_REVIEW_FILE)
    by_id = {r["Item ID"]: r for r in rows}

    for item in report.items:
        row = by_id[item.item_id]
        assert row["Tag"] == item.tags[0]
        assert row["Reason"] == item.reasons[0]
        assert row["Severity"] == str(item.severity)

    # item_0001 is BAD_ACTOR_FLIP and WEAK_MAJORITY: the flip is worse.
    assert by_id["item_0001"] == {
        "Item ID": "item_0001",
        "Severity": "4",
        "Tag": "BAD_ACTOR_FLIP",
        "Reason": "'negative' with bad actors, 'neutral' without them",
        "Label Counts": "neutral 5, negative 4",
        "Winners": "neutral",
        "Current Label": "negative",
        "Gold Labels": "",
        "Decision": "",
        "Reviewer": "",
        "Notes": "",
    }


def test_to_review_label_counts_are_the_votes_the_reason_uses(tmp_path):
    """Without bad actors when they are left out, so the numbers match the reason."""
    _write(DATA / "spam_known.csv", tmp_path / "a", use_bad_actors=True)
    _write(DATA / "spam_known.csv", tmp_path / "b", use_bad_actors=False)
    _, with_bad = _read_to_review(tmp_path / "a" / TO_REVIEW_FILE)
    _, plain = _read_to_review(tmp_path / "b" / TO_REVIEW_FILE)
    row_with_bad = next(r for r in with_bad if r["Item ID"] == "item_0134")
    row_plain = next(r for r in plain if r["Item ID"] == "item_0134")

    assert row_with_bad["Label Counts"] == "neutral 5, negative 4"
    assert "(4 of 9 votes)" in row_with_bad["Reason"]
    assert row_plain["Label Counts"] == "neutral 7, negative 4, positive 1"
    assert "(4 of 12 votes)" in row_plain["Reason"]


def test_to_review_reviewer_columns_are_empty(tmp_path):
    _write(DATA / "outliers.csv", tmp_path, use_bad_actors=True)
    _, rows = _read_to_review(tmp_path / TO_REVIEW_FILE)
    assert rows
    assert all(r["Decision"] == r["Reviewer"] == r["Notes"] == "" for r in rows)
    # Each line ends in the three empty cells.
    lines = (tmp_path / TO_REVIEW_FILE).read_bytes().splitlines()
    assert all(line.endswith(b",,,") for line in lines[1:])


# --- Items with too few votes ---


def _few_votes_csv(tmp_path: Path) -> Path:
    path = tmp_path / "in.csv"
    path.write_text(
        "item_id,annotator_id,is_gold,label,gold_label\n"
        "single,a,false,x,\n"
        "ok,a,false,x,\n"
        "blank,a,false,,\n"
        "split,a,false,x,\n"
        "split,b,false,y,\n"
        "blank,b,false,,\n"
        "ok,b,false,x,\n"
        "gold,a,true,y,z\n"  # one vote AND a gold key conflict
        "gold,b,true,,q\n",
        encoding="utf-8",
    )
    return path


def test_items_with_too_few_votes_are_listed_for_review(tmp_path):
    path = _few_votes_csv(tmp_path)
    _, report, files = _write(path, tmp_path / "out")
    assert report.not_assessable == ("blank", "gold", "single")
    _, rows = _read_to_review(tmp_path / "out" / TO_REVIEW_FILE)

    # Disputed items first (worst first), then too few votes, by item id.
    assert [(r["Item ID"], r["Severity"], r["Tag"]) for r in rows] == [
        ("gold", "1", "GOLD_KEY_CONFLICT"),  # listed once, by its worse tag
        ("split", "3", "TIE"),
        ("blank", "9", TOO_FEW_VOTES),
        ("single", "9", TOO_FEW_VOTES),
    ]
    assert TOO_FEW_VOTES_SEVERITY == 9
    blank, single = rows[2], rows[3]
    assert single == {
        "Item ID": "single",
        "Severity": "9",
        "Tag": "TOO_FEW_VOTES",
        "Reason": "only 1 vote; at least 2 are needed to check for a dispute",
        "Label Counts": "x 1",
        "Winners": "x",
        "Current Label": "x",
        "Gold Labels": "",
        "Decision": "",
        "Reviewer": "",
        "Notes": "",
    }
    assert blank["Reason"] == "no votes; at least 2 are needed to check for a dispute"
    assert blank["Label Counts"] == "-"
    assert blank["Winners"] == blank["Current Label"] == ""

    assert (files.disputed_items, files.too_few_votes_items) == (2, 2)
    assert files.to_review.rows == files.to_review.items == 4


def test_items_with_too_few_votes_go_to_disputed_csv(tmp_path):
    """Every item in to_review.csv has its rows in disputed.csv, none in agreed."""
    path = _few_votes_csv(tmp_path)
    _, _, files = _write(path, tmp_path / "out")

    assert _lines(tmp_path / "out" / AGREED_FILE) == [
        b"item_id,annotator_id,is_gold,label,gold_label\r\n",
        b"ok,a,false,x,\r\n",
        b"ok,b,false,x,\r\n",
    ]
    assert _lines(tmp_path / "out" / DISPUTED_FILE) == [
        b"item_id,annotator_id,is_gold,label,gold_label\r\n",
        b"single,a,false,x,\r\n",
        b"blank,a,false,,\r\n",
        b"split,a,false,x,\r\n",
        b"split,b,false,y,\r\n",
        b"blank,b,false,,\r\n",
        b"gold,a,true,y,z\r\n",
        b"gold,b,true,,q\r\n",
    ]
    assert (files.agreed.items, files.disputed.items) == (1, 4)


def test_too_few_votes_reason_counts_bad_actors(tmp_path):
    """With bad actors left out, an item can drop below 2 votes."""
    path = tmp_path / "in.csv"
    path.write_text(
        "item_id,annotator_id,is_gold,label\n"
        "i1,good,false,x\n"
        "i1,bad1,false,y\n"
        "i1,bad2,false,y\n"
        "i2,bad1,false,y\n",
        encoding="utf-8",
    )
    collection, _ = load_with_rows(path)
    report = find_disputes(collection, bad_actor_ids={"bad1", "bad2"})
    rows = to_review_rows(report)

    assert [(r["Item ID"], r["Tag"], r["Reason"]) for r in rows] == [
        (
            "i1",
            TOO_FEW_VOTES,
            (
                "only 1 vote without bad actors (3 with them); at least 2 are "
                "needed to check for a dispute"
            ),
        ),
        (
            "i2",
            TOO_FEW_VOTES,
            (
                "no votes without bad actors (1 with them); at least 2 are "
                "needed to check for a dispute"
            ),
        ),
    ]
    assert rows[0]["Label Counts"] == "x 1"
    assert rows[0]["Current Label"] == "y"  # what the majority vote gives today


# --- Safety ---


def test_existing_files_are_not_overwritten_and_nothing_is_written(tmp_path):
    collection, source = load_with_rows(DATA / "sample.csv")
    report = find_disputes(collection)
    (tmp_path / DISPUTED_FILE).write_text("keep", encoding="utf-8")

    with pytest.raises(ReviewFilesExistError, match="disputed.csv"):
        write_review_files(collection, source, report, tmp_path)
    assert (tmp_path / DISPUTED_FILE).read_text(encoding="utf-8") == "keep"
    assert not (tmp_path / AGREED_FILE).exists()
    assert not (tmp_path / TO_REVIEW_FILE).exists()

    write_review_files(collection, source, report, tmp_path, force=True)
    assert (tmp_path / DISPUTED_FILE).read_bytes().startswith(b"item_id,")


def test_the_input_is_never_overwritten(tmp_path):
    path = tmp_path / AGREED_FILE
    path.write_bytes((DATA / "sample.csv").read_bytes())
    collection, source = load_with_rows(path)
    report = find_disputes(collection)

    with pytest.raises(ValueError, match="Refusing to overwrite the input file"):
        write_review_files(
            collection, source, report, tmp_path, force=True, input_path=path
        )
    assert path.read_bytes() == (DATA / "sample.csv").read_bytes()


def test_source_rows_must_match_the_collection():
    _, source = load_with_rows(DATA / "sample.csv")
    other, _ = load_with_rows(DATA / "outliers.csv")
    with pytest.raises(ValueError, match="must match one to one"):
        split_rows(other, source, find_disputes(other))
