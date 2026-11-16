from pathlib import Path

import pytest

from fleisskit.bad_actors import (
    LEAVE_ONE_OUT,
    SPAM_CHECK,
    exclude_annotators,
    find_bad_actors,
)
from fleisskit.fleiss import fleiss_kappa
from fleisskit.io import load_csv
from fleisskit.models import Annotation, AnnotationCollection

DATA = Path(__file__).parent / "data"
OUTLIERS_CSV = DATA / "outliers.csv"
SPAM_KNOWN_CSV = DATA / "spam_known.csv"
SAMPLE_CSV = DATA / "sample.csv"


def _ann(item: str, annotator: str, label: str | None, seconds: float = 10.0):
    return Annotation(
        item_id=item,
        annotator_id=annotator,
        is_gold=False,
        label=label,
        duration_sec=seconds,
    )


def test_both_checks_report_the_same_two_outliers():
    """On outliers.csv the spam check and the leave-one-out score agree."""
    bad_actors = find_bad_actors(load_csv(OUTLIERS_CSV))

    assert [b.annotator_id for b in bad_actors] == ["spammer", "random_clicker"]
    for bad in bad_actors:
        assert bad.sources == (SPAM_CHECK, LEAVE_ONE_OUT)
        assert len(bad.reasons) == 2
        assert bad.spam is not None and bad.spam.is_spammer
        assert bad.influence is not None and bad.influence.is_outlier

    spammer = bad_actors[0]
    assert spammer.reasons[0] == (
        "spam check (99% confident): agreement, variety, speed"
    )
    assert spammer.reasons[1] == (
        "leave-one-out outlier (score 16.2): kappa +0.0994 without them"
    )


def test_spam_check_alone_can_make_a_bad_actor():
    """
    On spam_known.csv the three spammers are caught by the spam check, but
    none of them stands out enough for the leave-one-out score.
    """
    bad_actors = find_bad_actors(load_csv(SPAM_KNOWN_CSV))

    assert [b.annotator_id for b in bad_actors] == [
        "constant_spammer",
        "biased_clicker",
        "random_clicker_fast",
    ]
    for bad in bad_actors:
        assert bad.sources == (SPAM_CHECK,)
        assert bad.influence is None


def test_clean_data_has_no_bad_actors():
    assert find_bad_actors(load_csv(SAMPLE_CSV)) == []


def test_leave_one_out_is_skipped_when_kappa_is_undefined():
    """Varying ratings per item: the spam check still runs on its own."""
    rows = []
    for i in range(40):
        truth = "x" if i % 2 else "y"
        for a in ("h1", "h2", "h3"):
            rows.append(_ann(f"i{i}", a, truth))
        if i % 3:  # h4 skips some items, so items have 3 or 4 ratings
            rows.append(_ann(f"i{i}", "h4", "x"))
    collection = AnnotationCollection(rows)

    with pytest.raises(ValueError, match="Varying number of ratings"):
        fleiss_kappa(collection)

    bad_actors = find_bad_actors(collection)
    assert [b.annotator_id for b in bad_actors] == ["h4"]
    assert bad_actors[0].sources == (SPAM_CHECK,)


def test_base_rate_changes_the_confidence_in_the_reason():
    """Expecting spam to be rare lowers the confidence quoted in the reason."""
    usual = find_bad_actors(load_csv(OUTLIERS_CSV))
    cautious = find_bad_actors(load_csv(OUTLIERS_CSV), base_rate=0.02)

    assert usual[1].reasons[0] == "spam check (89% confident): agreement"
    # Below 50%: still flagged, but more likely honest than not. The flag is a
    # prompt to review, and the confidence says how much to trust it.
    assert cautious[1].reasons[0] == "spam check (49% confident): agreement"
    assert [b.annotator_id for b in cautious] == [b.annotator_id for b in usual]


def test_outlier_threshold_is_passed_through():
    """A very low bar makes the leave-one-out check flag more annotators."""
    strict = find_bad_actors(load_csv(OUTLIERS_CSV), outlier_threshold=0.1)
    by_id = {b.annotator_id: b for b in strict}

    assert len(strict) > 2
    # Flagged only by the leave-one-out score, not the spam check.
    assert by_id["good_1"].sources == (LEAVE_ONE_OUT,)
    assert by_id["good_1"].spam is None
    assert by_id["good_1"].reasons[0].startswith("leave-one-out outlier (score 0.2)")
    assert by_id["spammer"].sources == (SPAM_CHECK, LEAVE_ONE_OUT)


def test_bad_actors_are_sorted_worst_first():
    bad_actors = find_bad_actors(load_csv(OUTLIERS_CSV), outlier_threshold=0.1)
    scores = [b.spam.score if b.spam else 0.0 for b in bad_actors]
    assert scores == sorted(scores, reverse=True)


def test_exclude_annotators_removes_only_their_rows():
    collection = load_csv(OUTLIERS_CSV)
    remaining = exclude_annotators(collection, {"spammer", "random_clicker"})

    assert {a.annotator_id for a in remaining.annotations} == {
        f"good_{i}" for i in range(1, 9)
    }
    assert len(remaining.annotations) == len(collection.annotations) - 240
    # The original is untouched.
    assert len(collection.annotations) == 1200


def test_excluding_bad_actors_raises_kappa():
    collection = load_csv(OUTLIERS_CSV)
    bad_actors = find_bad_actors(collection)
    remaining = exclude_annotators(collection, {b.annotator_id for b in bad_actors})

    assert fleiss_kappa(collection) == pytest.approx(0.3577, abs=5e-4)
    assert fleiss_kappa(remaining) == pytest.approx(0.5773, abs=5e-4)


def test_exclude_annotators_with_nobody_to_remove():
    collection = load_csv(SAMPLE_CSV)
    assert exclude_annotators(collection, set()).annotations == collection.annotations
