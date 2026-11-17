import random
from fractions import Fraction
from pathlib import Path

import pytest

from fleisskit.bad_actors import find_bad_actors
from fleisskit.disputes import (
    BAD_ACTOR_FLIP,
    BAD_ACTOR_NOISE,
    BAD_ACTOR_TAGS,
    CONFIDENT_DISSENT,
    CONFIDENT_DISSENT_MIN_VOTES,
    DEFAULT_WEAK_MAJORITY_CUTOFF,
    GOLD_KEY_CONFLICT,
    GOLD_MISMATCH,
    NO_MAJORITY,
    TAG_ORDER,
    TIE,
    WEAK_MAJORITY,
    find_disputes,
    format_fraction,
)
from fleisskit.io import load_csv
from fleisskit.metrics import majority_vote
from fleisskit.models import Annotation, AnnotationCollection

DATA = Path(__file__).parent / "data"
SAMPLE_CSV = DATA / "sample.csv"
OUTLIERS_CSV = DATA / "outliers.csv"
SPAM_KNOWN_CSV = DATA / "spam_known.csv"


def _ann(
    item: str,
    annotator: str,
    label: str | None,
    confidence: float | None = None,
    gold: str | None = None,
) -> Annotation:
    return Annotation(
        item_id=item,
        annotator_id=annotator,
        is_gold=gold is not None,
        label=label,
        confidence=confidence,
        gold_label=gold,
    )


def _item(labels: str, item: str = "i1", **kwargs) -> list[Annotation]:
    """One item; labels like 'aab' give annotators a0, a1, a2 labels a, a, b."""
    return [_ann(item, f"a{k}", label, **kwargs) for k, label in enumerate(labels)]


def _tags(annotations: list[Annotation], **kwargs) -> tuple[str, ...]:
    report = find_disputes(AnnotationCollection(annotations), **kwargs)
    assert len(report.items) <= 1
    return report.items[0].tags if report.items else ()


# --- The agreed order ---


def test_tag_order_is_the_agreed_severity_order():
    assert TAG_ORDER == (
        "GOLD_KEY_CONFLICT",
        "GOLD_MISMATCH",
        "TIE",
        "BAD_ACTOR_FLIP",
        "NO_MAJORITY",
        "WEAK_MAJORITY",
        "CONFIDENT_DISSENT",
        "BAD_ACTOR_NOISE",
    )
    assert BAD_ACTOR_TAGS == {BAD_ACTOR_FLIP, BAD_ACTOR_NOISE}


# --- Unanimous and undisputed items ---


@pytest.mark.parametrize("labels", ["aa", "aaa", "aaaaaaaa"])
def test_unanimous_items_are_not_disputed(labels):
    assert _tags(_item(labels)) == ()


def test_two_of_three_is_not_weak_by_default():
    """Default cutoff is exactly 2/3, and 2/3 is not below it."""
    assert DEFAULT_WEAK_MAJORITY_CUTOFF == Fraction(2, 3)
    assert _tags(_item("aab")) == ()
    assert _tags(_item("aaaabb")) == ()  # 4 of 6


# --- GOLD_KEY_CONFLICT ---


def test_gold_key_conflict_on_different_gold_labels():
    rows = [
        _ann("i1", "a0", "x", gold="x"),
        _ann("i1", "a1", "x", gold="y"),
        _ann("i1", "a2", "x", gold="x"),
    ]
    report = find_disputes(AnnotationCollection(rows))
    (item,) = report.items
    # GOLD_MISMATCH needs a single gold answer, so it doesn't fire here.
    assert item.tags == (GOLD_KEY_CONFLICT,)
    assert item.gold_labels == ("x", "y")
    assert item.reasons == ("gold labels disagree: 'x', 'y'",)


def test_gold_key_conflict_when_gold_on_some_rows_only():
    rows = [
        _ann("i1", "a0", "x", gold="x"),
        _ann("i1", "a1", "x"),
        _ann("i1", "a2", "x"),
    ]
    (item,) = find_disputes(AnnotationCollection(rows)).items
    assert item.tags == (GOLD_KEY_CONFLICT,)
    assert item.reasons == ("marked gold on 1 of 3 rows",)


def test_gold_key_conflict_needs_no_votes():
    """It is about the key, not the votes: an item with one label still gets it."""
    rows = [_ann("i1", "a0", "x", gold="x"), _ann("i1", "a1", None, gold="y")]
    report = find_disputes(AnnotationCollection(rows))
    assert report.items[0].tags == (GOLD_KEY_CONFLICT,)
    assert report.not_assessable == ("i1",)


# --- GOLD_MISMATCH ---


def test_gold_mismatch_when_gold_is_not_a_winner():
    rows = _item("yyx", gold="x")
    (item,) = find_disputes(AnnotationCollection(rows)).items
    assert item.tags == (GOLD_MISMATCH,)
    assert item.reasons == ("gold is 'x' (1 of 3 votes), votes favour 'y'",)


def test_gold_mismatch_counts_zero_votes_for_gold():
    rows = _item("yy", gold="x")
    (item,) = find_disputes(AnnotationCollection(rows)).items
    assert item.reasons == ("gold is 'x' (0 of 2 votes), votes favour 'y'",)


def test_gold_among_tied_winners_is_a_tie_not_a_mismatch():
    assert _tags(_item("xy", gold="x")) == (TIE,)
    assert _tags(_item("yz", gold="x")) == (GOLD_MISMATCH, TIE)


def test_gold_agreed_with_is_not_disputed():
    assert _tags(_item("xxy", gold="x")) == ()


# --- TIE / NO_MAJORITY / WEAK_MAJORITY (one vote shape each) ---


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        ("ab", (TIE,)),
        ("abc", (TIE,)),
        ("aabb", (TIE,)),
        ("aabbc", (TIE,)),
        ("aabc", (NO_MAJORITY,)),  # 2 of 4 = 1/2: not over half
        ("aaabbbc", (TIE,)),
        ("aaabbc", (NO_MAJORITY,)),  # 3 of 6
        ("aaabb", (WEAK_MAJORITY,)),  # 3 of 5 = 0.6
        ("aaaaabbb", (WEAK_MAJORITY,)),  # 5 of 8
        ("aaaabb", ()),  # 4 of 6 = 2/3 exactly
        ("aaaaaab", ()),
    ],
)
def test_vote_shape_rules(labels, expected):
    assert _tags(_item(labels)) == expected


def test_vote_shape_reasons():
    (tie,) = find_disputes(AnnotationCollection(_item("abc"))).items
    assert tie.reasons == ("tie between 'a', 'b', 'c' (1 vote each)",)
    (tie,) = find_disputes(AnnotationCollection(_item("aabb"))).items
    assert tie.reasons == ("tie between 'a', 'b' (2 votes each)",)
    (nm,) = find_disputes(AnnotationCollection(_item("aabc"))).items
    assert nm.reasons == ("'a' leads with 2 of 4 votes (50%)",)
    (weak,) = find_disputes(AnnotationCollection(_item("aaabb"))).items
    assert weak.reasons == ("'a' has 3 of 5 votes (60%), below 2/3",)


def test_tie_item_reports_votes_and_current_label():
    (item,) = find_disputes(AnnotationCollection(_item("bbaac"))).items
    assert item.votes == {"a": 2, "b": 2, "c": 1}
    assert item.winners == ("a", "b")
    # What majority_vote ships: the alphabetical tie-break.
    assert item.current_label == "a"
    assert (
        item.current_label == majority_vote(AnnotationCollection(_item("bbaac")))["i1"]
    )


# --- WEAK_MAJORITY cutoff ---


def test_cutoff_is_exact_so_067_flags_two_of_three():
    """0.67 is 67/100, just above 2/3, so 2-of-3 becomes weak."""
    assert _tags(_item("aab"), weak_majority_cutoff=0.67) == (WEAK_MAJORITY,)
    assert _tags(_item("aab"), weak_majority_cutoff="0.67") == (WEAK_MAJORITY,)
    assert _tags(_item("aab"), weak_majority_cutoff="2/3") == ()
    assert _tags(_item("aab"), weak_majority_cutoff=Fraction(2, 3)) == ()


def test_cutoff_of_one_flags_every_split_majority():
    assert _tags(_item("aaaaaaab"), weak_majority_cutoff=1) == (WEAK_MAJORITY,)
    assert _tags(_item("aaaaaaaa"), weak_majority_cutoff=1) == ()


def test_cutoff_is_reported():
    report = find_disputes(
        AnnotationCollection(_item("aaab")), weak_majority_cutoff=0.8
    )
    assert report.weak_majority_cutoff == Fraction(4, 5)
    assert report.items[0].reasons == ("'a' has 3 of 4 votes (75%), below 4/5",)


@pytest.mark.parametrize("bad", [0.5, "1/2", 0.3, 1.01, "3/2", "abc", "1/0", 0])
def test_cutoff_must_be_above_half_and_at_most_one(bad):
    with pytest.raises(ValueError, match="Weak majority cutoff"):
        find_disputes(AnnotationCollection(_item("ab")), weak_majority_cutoff=bad)


def test_format_fraction():
    assert format_fraction(Fraction(2, 3)) == "2/3"
    assert format_fraction(Fraction(67, 100)) == "0.67"
    assert format_fraction(Fraction(1)) == "1"


# --- CONFIDENT_DISSENT ---


def _confident_item(labels: str, confidences: list[float | None]) -> list[Annotation]:
    return [
        _ann("i1", f"a{k}", label, confidence=c)
        for k, (label, c) in enumerate(zip(labels, confidences, strict=True))
    ]


def test_confident_dissent_on_six_votes():
    """4-2 with both dissenters sure: the vote shape alone would pass it."""
    rows = _confident_item("aaaabb", [0.5, 0.5, 0.5, 0.5, 0.9, 0.8])
    (item,) = find_disputes(AnnotationCollection(rows)).items
    assert item.tags == (CONFIDENT_DISSENT,)
    assert item.reasons == ("2 annotators chose 'b' with confidence >= 0.8",)


def test_confident_dissent_needs_six_votes():
    """At 5 votes the same sure pair gives only WEAK_MAJORITY (3-2)."""
    assert CONFIDENT_DISSENT_MIN_VOTES == 6
    rows = _confident_item("aaabb", [0.5, 0.5, 0.5, 0.9, 0.9])
    assert _tags(rows) == (WEAK_MAJORITY,)


def test_confident_dissent_needs_two_sure_dissenters_on_one_label():
    # Only one sure dissenter.
    assert _tags(_confident_item("aaaabb", [0.9] * 4 + [0.9, 0.79])) == ()
    # Two sure dissenters, but on different labels.
    assert _tags(_confident_item("aaaabc", [0.5] * 4 + [0.9, 0.9])) == ()
    # Missing confidence never counts as sure.
    assert _tags(_confident_item("aaaabb", [0.5] * 4 + [0.9, None])) == ()
    # Sure annotators in the majority don't matter.
    assert _tags(_confident_item("aaaabb", [1.0] * 4 + [0.1, 0.1])) == ()


def test_confident_dissent_counts_losing_labels_in_a_tie():
    rows = _confident_item("aaabbbcc", [0.5] * 6 + [0.9, 0.95])
    assert _tags(rows) == (TIE, CONFIDENT_DISSENT)


def _shapes(n: int, parts: int = 3, largest: int | None = None):
    """Every way to split n votes over up to `parts` labels, largest first."""
    largest = n if largest is None else largest
    if n == 0:
        yield ()
        return
    if parts == 0:
        return
    for first in range(min(n, largest), 0, -1):
        for rest in _shapes(n - first, parts - 1, first):
            yield (first, *rest)


def _vote_rules_flag(shape: tuple[int, ...]) -> bool:
    n, top = sum(shape), shape[0]
    tied = len(shape) > 1 and shape[1] == top
    return tied or 2 * top <= n or Fraction(top, n) < DEFAULT_WEAK_MAJORITY_CUTOFF


def _dissent_possible(shape: tuple[int, ...]) -> bool:
    """A losing label with at least 2 votes can hold 2 sure dissenters."""
    return any(count >= 2 for count in shape if count < shape[0])


def test_six_is_the_smallest_item_where_confident_dissent_adds_anything():
    """
    The reasoning behind CONFIDENT_DISSENT_MIN_VOTES, checked on every split:
    below 5 votes the rule can't fire, at 5 it only fires where the vote
    rules already do, and at 6 it can flag an item nothing else flags.
    """
    adds_something = {
        n: [s for s in _shapes(n) if _dissent_possible(s) and not _vote_rules_flag(s)]
        for n in range(2, 9)
    }
    possible = {n: [s for s in _shapes(n) if _dissent_possible(s)] for n in range(2, 6)}
    assert possible[2] == possible[3] == possible[4] == []
    assert possible[5] == [(3, 2)]
    assert all(not adds_something[n] for n in range(2, 6))
    assert adds_something[6] == [(4, 2)]
    assert min(n for n, found in adds_something.items() if found) == (
        CONFIDENT_DISSENT_MIN_VOTES
    )


# --- Bad actor tags ---


def test_bad_actor_tags_absent_by_default():
    rows = _item("aabbb")  # a3, a4 say b
    report = find_disputes(AnnotationCollection(rows))
    assert report.used_bad_actors is False
    assert report.bad_actor_ids == ()
    assert BAD_ACTOR_FLIP not in report.tag_counts
    assert BAD_ACTOR_NOISE not in report.tag_counts
    assert list(report.tag_counts) == [t for t in TAG_ORDER if t not in BAD_ACTOR_TAGS]


def test_bad_actor_flip():
    """Three bad actors outvote two honest annotators."""
    rows = _item("aabbb")
    (item,) = find_disputes(
        AnnotationCollection(rows), bad_actor_ids={"a2", "a3", "a4"}
    ).items
    assert item.current_label == "b"
    assert item.votes == {"a": 2}
    assert item.all_votes == {"b": 3, "a": 2}
    # The honest votes are unanimous, so the disagreement is theirs alone too.
    assert item.tags == (BAD_ACTOR_FLIP, BAD_ACTOR_NOISE)
    assert item.reasons[0] == "'b' with bad actors, 'a' without them"


def test_bad_actor_flip_with_a_split_honest_vote():
    rows = _item("aaabbbbb")  # honest: a0-a2 a, a3-a4 b; bad: a5-a7 b
    (item,) = find_disputes(
        AnnotationCollection(rows), bad_actor_ids={"a5", "a6", "a7"}
    ).items
    assert item.votes == {"a": 3, "b": 2}
    assert item.current_label == "b"
    assert item.tags == (BAD_ACTOR_FLIP, WEAK_MAJORITY)
    assert item.severity == TAG_ORDER.index(BAD_ACTOR_FLIP) + 1


def test_tie_outranks_bad_actor_flip():
    rows = _item("abbb")  # honest a0 a, a1 b tie; bad a2, a3 b
    (item,) = find_disputes(
        AnnotationCollection(rows), bad_actor_ids={"a2", "a3"}
    ).items
    # 'b' is still a winner, so no flip: a tie with the current label in it.
    assert item.tags == (TIE,)

    rows = _item("abcc")  # honest a, b tie; bad actors push c
    (item,) = find_disputes(
        AnnotationCollection(rows), bad_actor_ids={"a2", "a3"}
    ).items
    assert item.tags == (TIE, BAD_ACTOR_FLIP)
    assert item.severity == 3


def test_bad_actor_noise():
    rows = _item("aaac")
    (item,) = find_disputes(AnnotationCollection(rows), bad_actor_ids={"a3"}).items
    assert item.tags == (BAD_ACTOR_NOISE,)
    assert item.reasons == ("unanimous 'a' without bad actors; they added 'c'",)


def test_bad_actors_agreeing_cause_no_tags():
    report = find_disputes(AnnotationCollection(_item("aaaa")), bad_actor_ids={"a3"})
    assert report.items == []
    assert report.tag_counts[BAD_ACTOR_FLIP] == report.tag_counts[BAD_ACTOR_NOISE] == 0


def test_empty_bad_actor_set_keeps_the_tags_but_changes_nothing():
    rows = _item("aab") + _item("abc", item="i2")
    with_empty = find_disputes(AnnotationCollection(rows), bad_actor_ids=set())
    without = find_disputes(AnnotationCollection(rows))
    assert with_empty.used_bad_actors is True
    assert with_empty.tag_counts[BAD_ACTOR_FLIP] == 0
    assert [i.tags for i in with_empty.items] == [i.tags for i in without.items]


def test_only_bad_actors_on_an_item_makes_it_not_assessable():
    rows = _item("ab") + _item("aa", item="i2")
    report = find_disputes(AnnotationCollection(rows), bad_actor_ids={"a0"})
    assert report.items == []
    assert report.not_assessable == ("i1", "i2")


# --- Not assessable ---


def test_items_with_under_two_votes_are_not_assessable():
    rows = [
        _ann("one", "a0", "x"),
        _ann("blank", "a0", None),
        _ann("blank", "a1", None),
        *_item("ab", item="two"),
    ]
    report = find_disputes(AnnotationCollection(rows))
    assert report.items_checked == 3
    assert report.not_assessable == ("blank", "one")
    assert [i.item_id for i in report.items] == ["two"]


def test_blank_labels_are_ignored():
    rows = [*_item("aab"), _ann("i1", "a9", None)]
    assert _tags(rows) == ()


def test_empty_collection():
    report = find_disputes(AnnotationCollection([]))
    assert report.items == []
    assert report.items_checked == 0
    assert report.not_assessable == ()


# --- Ordering and determinism ---


def test_items_sorted_by_severity_then_tag_count_then_id():
    rows = [
        *_item("aaabb", item="weak"),
        *_item("ab", item="tie_b"),
        *_item("ab", item="tie_a"),
        *_item("yz", item="gold_tie", gold="x"),
        *_item("aabc", item="nomaj"),
    ]
    report = find_disputes(AnnotationCollection(rows))
    assert [(i.item_id, i.severity) for i in report.items] == [
        ("gold_tie", 2),
        ("tie_a", 3),
        ("tie_b", 3),
        ("nomaj", 5),
        ("weak", 6),
    ]
    assert report.tag_counts == {
        GOLD_KEY_CONFLICT: 0,
        GOLD_MISMATCH: 1,
        TIE: 3,
        NO_MAJORITY: 1,
        WEAK_MAJORITY: 1,
        CONFIDENT_DISSENT: 0,
    }


def test_more_tags_first_within_a_severity():
    rows = [
        *_item("ab", item="a_plain_tie"),
        *_confident_item("aaabbbcc", [0.5] * 6 + [0.9, 0.9]),
    ]
    report = find_disputes(AnnotationCollection(rows))
    assert [i.item_id for i in report.items] == ["i1", "a_plain_tie"]


def test_input_order_does_not_matter():
    collection = load_csv(SPAM_KNOWN_CSV)
    bad = {b.annotator_id for b in find_bad_actors(collection)}
    shuffled = list(collection.annotations)
    random.Random(0).shuffle(shuffled)

    first = find_disputes(collection, bad_actor_ids=bad)
    second = find_disputes(AnnotationCollection(shuffled), bad_actor_ids=bad)
    assert first == second


# --- Fixtures ---


def test_sample_csv_has_three_three_way_ties():
    report = find_disputes(load_csv(SAMPLE_CSV))
    assert [(i.item_id, i.tags) for i in report.items] == [
        ("item_33", (TIE,)),
        ("item_35", (TIE,)),
        ("item_7", (TIE,)),
    ]
    assert report.items_checked == 50


def test_spam_known_with_and_without_bad_actors():
    collection = load_csv(SPAM_KNOWN_CSV)
    plain = find_disputes(collection)
    assert len(plain.items) == 103
    assert plain.tag_counts == {
        GOLD_KEY_CONFLICT: 0,
        GOLD_MISMATCH: 1,
        TIE: 11,
        NO_MAJORITY: 46,
        WEAK_MAJORITY: 46,
        CONFIDENT_DISSENT: 0,
    }

    bad = {b.annotator_id for b in find_bad_actors(collection)}
    report = find_disputes(collection, bad_actor_ids=bad)
    assert report.bad_actor_ids == (
        "biased_clicker",
        "constant_spammer",
        "random_clicker_fast",
    )
    assert len(report.items) == 63
    assert report.tag_counts == {
        GOLD_KEY_CONFLICT: 0,
        GOLD_MISMATCH: 1,
        TIE: 3,
        BAD_ACTOR_FLIP: 14,
        NO_MAJORITY: 18,
        WEAK_MAJORITY: 38,
        CONFIDENT_DISSENT: 0,
        BAD_ACTOR_NOISE: 4,
    }
    worst = report.items[0]
    assert worst.item_id == "item_0134"
    assert worst.tags == (GOLD_MISMATCH, WEAK_MAJORITY)
    assert worst.gold_labels == ("negative",)
    assert worst.votes == {"neutral": 5, "negative": 4}


def test_outliers_confident_dissent():
    collection = load_csv(OUTLIERS_CSV)
    bad = {b.annotator_id for b in find_bad_actors(collection)}
    report = find_disputes(collection, bad_actor_ids=bad)
    assert report.tag_counts[CONFIDENT_DISSENT] == 7
    assert report.tag_counts[BAD_ACTOR_NOISE] == 32
    assert report.tag_counts[BAD_ACTOR_FLIP] == 2
