"""
Writes a dispute report out as three CSV files for a reviewer:

- to_review.csv one row per item the reviewer should look at, most severe
                first. That is every disputed item, plus every item with too
                few votes to be judged (tag TOO_FEW_VOTES, after all the
                dispute tags). Each row has the item's most severe tag and
                that tag's reason, the label counts in one cell, and three
                empty columns for the reviewer: Decision, Reviewer, Notes.
- disputed.csv  the input rows of every item in to_review.csv (all of its
                rows, bad actors' and blank labels included),
- agreed.csv    the input rows of every other item.

agreed.csv and disputed.csv keep the input's structure: the same columns in
the same order, and each row exactly as it was read. Every input row goes to
exactly one of them, in its original order, so together they are the input.
For CSV input that holds byte for byte when the input uses the csv module's
defaults (minimal quoting, CRLF line endings, as the fixtures do). JSON input
becomes CSV with the JSON keys as columns: null is written as an empty cell
and true/false as "true"/"false", which load back to the same annotations.
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fleisskit.disputes import (
    MIN_VOTES,
    TAG_ORDER,
    DisputeReport,
    ItemVotes,
    format_votes,
)
from fleisskit.io import SourceRows
from fleisskit.models import AnnotationCollection

AGREED_FILE = "agreed.csv"
DISPUTED_FILE = "disputed.csv"
TO_REVIEW_FILE = "to_review.csv"

# Items with fewer than MIN_VOTES votes: not a dispute, but the reviewer should
# still decide on them. Ranked after every dispute tag.
TOO_FEW_VOTES = "TOO_FEW_VOTES"
TOO_FEW_VOTES_SEVERITY = len(TAG_ORDER) + 1

ITEM_ID = "Item ID"
# 1-based position of the tag in fleisskit.disputes.TAG_ORDER, or
# TOO_FEW_VOTES_SEVERITY.
SEVERITY = "Severity"
# The item's most severe tag, and the reason for it.
TAG = "Tag"
REASON = "Reason"
# Each label's count, most votes first: "neutral 5, negative 4". These are the
# votes the rules used (without bad actors when they were left out), so they
# match the numbers in the reason.
LABEL_COUNTS = "Label Counts"
WINNERS = "Winners"
CURRENT_LABEL = "Current Label"
GOLD_LABELS = "Gold Labels"
# Left empty, for the reviewer.
REVIEWER_COLUMNS = ("Decision", "Reviewer", "Notes")

TO_REVIEW_COLUMNS = (
    ITEM_ID,
    SEVERITY,
    TAG,
    REASON,
    LABEL_COUNTS,
    WINNERS,
    CURRENT_LABEL,
    GOLD_LABELS,
    *REVIEWER_COLUMNS,
)


class ReviewFilesExistError(Exception):
    """One of the three files already exists and force was not given."""


@dataclass(frozen=True)
class WrittenFile:
    path: Path
    # Data rows written (the header not included).
    rows: int
    # Distinct items in those rows.
    items: int


@dataclass(frozen=True)
class ReviewFiles:
    agreed: WrittenFile
    disputed: WrittenFile
    to_review: WrittenFile
    # How the items in to_review.csv (and disputed.csv) break down.
    disputed_items: int
    too_few_votes_items: int


def _cell(value: Any) -> str:
    """
    One raw value as CSV text. CSV input is already text and is kept as is.
    JSON values: null -> "", booleans -> "true"/"false" (what the CSV loader
    reads), numbers via str(), lists and objects as JSON.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    if isinstance(value, int | float):
        return str(value)
    return json.dumps(value)


def too_few_votes_items(report: DisputeReport) -> list[ItemVotes]:
    """
    The items the rules could not judge that are not already disputed (an
    item with too few votes can still have GOLD_KEY_CONFLICT), by item id.
    """
    disputed = {item.item_id for item in report.items}
    return [v for v in report.not_assessable_votes if v.item_id not in disputed]


def review_item_ids(report: DisputeReport) -> set[str]:
    """Every item in to_review.csv: disputed, or with too few votes."""
    return {item.item_id for item in report.items} | {
        v.item_id for v in report.not_assessable_votes
    }


def split_rows(
    collection: AnnotationCollection, source: SourceRows, report: DisputeReport
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    The input rows split into (agreed, disputed) by their item, each in input
    order: disputed gets every item in to_review.csv. source must be the rows
    `collection` was loaded from (as returned by fleisskit.io.load_with_rows),
    and `report` must be for `collection`.
    """
    if len(source.rows) != len(collection.annotations):
        raise ValueError(
            f"Got {len(source.rows)} source rows for "
            f"{len(collection.annotations)} annotations; they must match one to one."
        )
    to_review = review_item_ids(report)
    agreed: list[dict[str, Any]] = []
    disputed: list[dict[str, Any]] = []
    for ann, row in zip(collection.annotations, source.rows, strict=True):
        (disputed if ann.item_id in to_review else agreed).append(row)
    return agreed, disputed


def _too_few_votes_reason(item: ItemVotes, used_bad_actors: bool) -> str:
    """'only 1 vote; at least 2 are needed to check for a dispute', etc."""
    n = sum(item.votes.values())
    reason = "no votes" if n == 0 else f"only {n} vote"
    total = sum(item.all_votes.values())
    if used_bad_actors and total != n:
        reason += f" without bad actors ({total} with them)"
    return f"{reason}; at least {MIN_VOTES} are needed to check for a dispute"


def to_review_rows(report: DisputeReport) -> list[dict[str, str]]:
    """
    The to_review.csv rows, keyed by column header: the disputed items in the
    report's order (most severe first), each with its most severe tag and that
    tag's reason, then the items with too few votes, by item id. The reviewer
    columns are empty.
    """
    empty = dict.fromkeys(REVIEWER_COLUMNS, "")
    rows = [
        {
            ITEM_ID: item.item_id,
            SEVERITY: str(item.severity),
            TAG: item.tags[0],
            REASON: item.reasons[0],
            LABEL_COUNTS: format_votes(item.votes),
            WINNERS: ", ".join(item.winners),
            CURRENT_LABEL: item.current_label or "",
            GOLD_LABELS: ", ".join(item.gold_labels),
            **empty,
        }
        for item in report.items
    ]
    rows += [
        {
            ITEM_ID: item.item_id,
            SEVERITY: str(TOO_FEW_VOTES_SEVERITY),
            TAG: TOO_FEW_VOTES,
            REASON: _too_few_votes_reason(item, report.used_bad_actors),
            LABEL_COUNTS: format_votes(item.votes),
            WINNERS: ", ".join(item.winners),
            CURRENT_LABEL: item.current_label or "",
            GOLD_LABELS: ", ".join(item.gold_labels),
            **empty,
        }
        for item in too_few_votes_items(report)
    ]
    return rows


def _write_csv(
    path: Path, columns: tuple[str, ...], rows: list[dict[str, Any]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_cell(row.get(col)) for col in columns])


def write_review_files(
    collection: AnnotationCollection,
    source: SourceRows,
    report: DisputeReport,
    output_dir: str | Path,
    force: bool = False,
    input_path: str | Path | None = None,
) -> ReviewFiles:
    """
    Writes agreed.csv, disputed.csv and to_review.csv into output_dir (created
    if missing) and says what went where.

    Raises ReviewFilesExistError, before writing anything, if any of the three
    exists and force is False, and OSError if they can't be written. Raises
    ValueError if one of them is input_path, even with force: the input is
    never overwritten.
    """
    out = Path(output_dir)
    paths = {name: out / name for name in (AGREED_FILE, DISPUTED_FILE, TO_REVIEW_FILE)}

    if input_path is not None:
        source_file = Path(input_path).resolve()
        for path in paths.values():
            if path.resolve() == source_file:
                raise ValueError(f"Refusing to overwrite the input file: {path}")
    if not force:
        existing = [str(p) for p in paths.values() if p.exists()]
        if existing:
            raise ReviewFilesExistError(
                f"Already exists: {', '.join(existing)} (use --force to overwrite)"
            )

    agreed, disputed = split_rows(collection, source, report)
    to_review = to_review_rows(report)
    all_items = {ann.item_id for ann in collection.annotations}
    review_items = review_item_ids(report) & all_items

    out.mkdir(parents=True, exist_ok=True)
    _write_csv(paths[AGREED_FILE], source.columns, agreed)
    _write_csv(paths[DISPUTED_FILE], source.columns, disputed)
    _write_csv(paths[TO_REVIEW_FILE], TO_REVIEW_COLUMNS, to_review)

    return ReviewFiles(
        agreed=WrittenFile(
            paths[AGREED_FILE], len(agreed), len(all_items - review_items)
        ),
        disputed=WrittenFile(paths[DISPUTED_FILE], len(disputed), len(review_items)),
        to_review=WrittenFile(paths[TO_REVIEW_FILE], len(to_review), len(to_review)),
        disputed_items=len(report.items),
        too_few_votes_items=len(too_few_votes_items(report)),
    )
