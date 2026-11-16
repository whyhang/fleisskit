import random
from pathlib import Path

import pytest

from fleisskit.fleiss import (
    ConfidenceInterval,
    _fleiss_kappa_components,
    build_count_table,
    fleiss_kappa,
)
from fleisskit.generator import generate_synthetic_data
from fleisskit.influence import (
    AnnotatorInfluence,
    _robust_scores,
    _WeightedKappa,
    leave_one_annotator_out,
)
from fleisskit.io import load_csv
from fleisskit.models import Annotation, AnnotationCollection

# Checked-in fixture with known outliers, built once with random.Random(2024):
# 120 items, 3 labels, fully crossed by 10 annotators.
#   good_1 .. good_8  pick the item's true label 85% of the time
#   spammer           always answers "positive", in under 2 seconds
#   random_clicker    picks uniformly at random, ignoring the item
OUTLIERS_CSV = Path(__file__).parent / "data" / "outliers.csv"


def _make_ann(item_id: str, annotator_id: str, label: str | None) -> Annotation:
    return Annotation(
        item_id=item_id, annotator_id=annotator_id, is_gold=False, label=label
    )


def _collection(rows: dict[str, dict[str, str | None]]) -> AnnotationCollection:
    """{item_id: {annotator_id: label}} -> AnnotationCollection"""
    return AnnotationCollection(
        [
            _make_ann(item, annotator, label)
            for item, by_annotator in rows.items()
            for annotator, label in by_annotator.items()
        ]
    )


def _without(collection: AnnotationCollection, annotator_id: str):
    return AnnotationCollection(
        [a for a in collection.annotations if a.annotator_id != annotator_id]
    )


# A1 and A2 always agree; A3 contradicts them on the first two items.
HAND = _collection(
    {
        "i1": {"A1": "x", "A2": "x", "A3": "y"},
        "i2": {"A1": "y", "A2": "y", "A3": "x"},
        "i3": {"A1": "x", "A2": "x", "A3": "x"},
        "i4": {"A1": "y", "A2": "y", "A3": "y"},
    }
)


def test_hand_computed_example():
    """
    Full table rows [2,1], [1,2], [3,0], [0,3]:
      P_i = 1/3, 1/3, 1, 1 -> P_bar = 2/3; p_x = p_y = 1/2 -> P_bar_e = 1/2
      kappa = (2/3 - 1/2) / (1/2) = 1/3
    Without A3: A1 and A2 agree everywhere -> kappa = 1 (delta +2/3)
    Without A1 (or A2): agree on i3, i4 only -> P_bar = 1/2, P_bar_e = 1/2
      -> kappa = 0 (delta -1/3)
    """
    report = leave_one_annotator_out(HAND)

    assert report.full_kappa == pytest.approx(1 / 3)
    assert report.full_p_bar == pytest.approx(2 / 3)
    assert report.full_p_bar_e == pytest.approx(1 / 2)

    # Largest delta first; ties broken by annotator id.
    assert [r.annotator_id for r in report.annotators] == ["A3", "A1", "A2"]

    a3, a1, a2 = report.annotators
    assert a3.kappa_without == pytest.approx(1.0)
    assert a3.delta == pytest.approx(2 / 3)
    assert a3.p_bar_without == pytest.approx(1.0)
    assert a3.p_bar_e_without == pytest.approx(0.5)
    for r in (a1, a2):
        assert r.kappa_without == pytest.approx(0.0)
        assert r.delta == pytest.approx(-1 / 3)
        assert r.items_rated == 4
        assert r.error is None
        assert r.delta_ci is None  # calc_ci defaults to False


@pytest.mark.parametrize("seed", range(10))
def test_matches_recomputing_kappa_without_each_annotator(seed):
    """
    kappa_without must equal fleiss_kappa on the data with that annotator's
    rows dropped (a separate code path that rebuilds the table from scratch).
    """
    shape = random.Random(seed)
    collection, _ = generate_synthetic_data(
        num_items=shape.randint(10, 60),
        num_annotators=shape.randint(3, 6),
        labels=[f"L{i}" for i in range(shape.randint(2, 5))],
        accuracy=shape.uniform(0.3, 0.95),
        spammer_share=shape.choice([0.0, 0.2, 0.4]),
        seed=seed,
    )
    full = fleiss_kappa(collection)
    report = leave_one_annotator_out(collection)

    assert report.full_kappa == pytest.approx(full, abs=1e-12)
    for r in report.annotators:
        reduced = _without(collection, r.annotator_id)
        try:
            expected_kappa, expected_p, expected_pe = _fleiss_kappa_components(
                build_count_table(reduced)[0]
            )
        except ValueError:
            assert r.kappa_without is None and r.error
            continue
        assert r.kappa_without == pytest.approx(expected_kappa, abs=1e-12)
        assert r.delta == pytest.approx(expected_kappa - full, abs=1e-12)
        assert r.p_bar_without == pytest.approx(expected_p, abs=1e-12)
        assert r.p_bar_e_without == pytest.approx(expected_pe, abs=1e-12)


def test_label_used_only_by_removed_annotator():
    """A label only the removed annotator used drops out cleanly (zero column)."""
    collection = _collection(
        {
            "i1": {"A1": "x", "A2": "x", "A3": "z"},
            "i2": {"A1": "y", "A2": "y", "A3": "y"},
            "i3": {"A1": "x", "A2": "y", "A3": "x"},
        }
    )
    report = leave_one_annotator_out(collection)
    a3 = next(r for r in report.annotators if r.annotator_id == "A3")
    assert a3.kappa_without == pytest.approx(fleiss_kappa(_without(collection, "A3")))


def test_annotator_who_skipped_items_is_undefined():
    """
    Constant 3 ratings per item, but B, C and D each rated only 3 of 4 items.
    Removing any of them gives items 2 or 3 ratings, so Fleiss' kappa is
    undefined. A rated everything, so removing A is fine.
    """
    collection = _collection(
        {
            "i1": {"A": "x", "B": "x", "C": "y"},
            "i2": {"A": "y", "C": "y", "D": "y"},
            "i3": {"A": "x", "B": "x", "D": "x"},
            "i4": {"A": "y", "B": "x", "C": "y"},
        }
    )
    report = leave_one_annotator_out(collection)
    by_id = {r.annotator_id: r for r in report.annotators}

    assert by_id["A"].kappa_without == pytest.approx(
        fleiss_kappa(_without(collection, "A"))
    )
    assert by_id["B"].items_rated == 3
    for a in ("B", "C", "D"):
        assert by_id[a].kappa_without is None
        assert by_id[a].delta is None
        assert "of 4 items" in by_id[a].error
        assert "different numbers of ratings" in by_id[a].error

    # Defined entries first, undefined ones after (by annotator id).
    assert [r.annotator_id for r in report.annotators] == ["A", "B", "C", "D"]


def test_two_annotators_leave_too_few_ratings():
    collection = _collection(
        {"i1": {"A": "x", "B": "x"}, "i2": {"A": "y", "B": "x"}},
    )
    report = leave_one_annotator_out(collection)
    assert len(report.annotators) == 2
    for r in report.annotators:
        assert r.kappa_without is None
        assert "at least 2 ratings per item" in r.error


def test_no_variation_left_after_removal():
    """If the others all used one label, P_bar_e == 1 and kappa is undefined."""
    collection = _collection(
        {
            "i1": {"A1": "x", "A2": "x", "A3": "y"},
            "i2": {"A1": "x", "A2": "x", "A3": "x"},
            "i3": {"A1": "x", "A2": "x", "A3": "y"},
        }
    )
    report = leave_one_annotator_out(collection)
    by_id = {r.annotator_id: r for r in report.annotators}

    assert by_id["A3"].kappa_without is None
    assert "Chance agreement" in by_id["A3"].error
    assert by_id["A1"].kappa_without is not None


def test_full_data_kappa_undefined_raises():
    collection = _collection(
        {"i1": {"A": "x", "B": "x", "C": "y"}, "i2": {"A": "y", "B": "x"}},
    )
    with pytest.raises(ValueError, match="Varying number of ratings"):
        leave_one_annotator_out(collection)


def test_annotator_with_only_missing_labels_is_not_listed():
    """Missing labels don't enter the count table, so they can't affect kappa."""
    collection = AnnotationCollection(
        [*HAND.annotations, _make_ann("i1", "ghost", None)]
    )
    report = leave_one_annotator_out(collection)
    assert "ghost" not in {r.annotator_id for r in report.annotators}


# --- Bootstrap CI of delta ---


def test_weighted_kappa_with_unit_weights_matches_components():
    collection, _ = generate_synthetic_data(
        num_items=40, num_annotators=4, labels=["a", "b", "c"], seed=5
    )
    table, _ = build_count_table(collection)
    expected, _, _ = _fleiss_kappa_components(table)
    assert _WeightedKappa(table).kappa([1] * len(table)) == pytest.approx(expected)


def test_weighted_kappa_matches_expanded_table():
    """Weight w on a row is the same as repeating that row w times."""
    table = [[2, 1], [1, 2], [3, 0], [0, 3]]
    weights = [3, 0, 1, 0]
    expanded = [row for row, w in zip(table, weights, strict=True) for _ in range(w)]
    expected, _, _ = _fleiss_kappa_components(expanded)
    assert _WeightedKappa(table).kappa(weights) == pytest.approx(expected)
    # All weight on [3, 0] rows: one label gets every rating -> undefined.
    assert _WeightedKappa(table).kappa([0, 0, 4, 0]) is None


def test_spammer_is_flagged_by_ci():
    """
    One of five annotators always picks the same label. Removing the spammer
    raises kappa, with a CI entirely above 0. Removing any genuine annotator
    lowers it, with a CI entirely below 0.
    """
    collection, _ = generate_synthetic_data(
        num_items=200,
        num_annotators=5,
        labels=["A", "B", "C"],
        accuracy=0.75,
        spammer_share=0.2,
        seed=3,
    )
    report = leave_one_annotator_out(collection, calc_ci=True, ci_repeats=300)

    spammer = report.annotators[0]
    assert spammer.annotator_id == "annotator_0"  # generator's spammer
    assert spammer.delta > 0
    assert spammer.delta_ci.lower_bound > 0
    assert spammer.effect == "lowers_kappa"
    for r in report.annotators[1:]:
        assert r.effect == "raises_kappa"
        assert r.delta_ci.lower_bound <= r.delta <= r.delta_ci.upper_bound


def test_bonferroni_splits_alpha_across_annotators():
    collection, _ = generate_synthetic_data(
        num_items=60, num_annotators=4, labels=["a", "b", "c"], seed=11
    )
    plain = leave_one_annotator_out(
        collection, calc_ci=True, ci_repeats=400, ci_bonferroni=False
    )
    adjusted = leave_one_annotator_out(collection, calc_ci=True, ci_repeats=400)

    assert (plain.ci_family_alpha, plain.ci_num_comparisons) == (0.05, 1)
    assert (adjusted.ci_family_alpha, adjusted.ci_num_comparisons) == (0.05, 4)
    for p, a in zip(plain.annotators, adjusted.annotators, strict=True):
        assert p.delta_ci.alpha == pytest.approx(0.05)
        assert a.delta_ci.alpha == pytest.approx(0.05 / 4)
        # Same seed, so same resamples: the adjusted interval contains the plain one.
        assert a.delta_ci.lower_bound <= p.delta_ci.lower_bound
        assert a.delta_ci.upper_bound >= p.delta_ci.upper_bound


def test_bonferroni_counts_only_defined_annotators():
    """B, C and D have no defined delta, so only A's interval is tested (m = 1)."""
    collection = _collection(
        {
            "i1": {"A": "x", "B": "x", "C": "y"},
            "i2": {"A": "y", "C": "y", "D": "y"},
            "i3": {"A": "x", "B": "x", "D": "x"},
            "i4": {"A": "y", "B": "x", "C": "y"},
        }
    )
    report = leave_one_annotator_out(collection, calc_ci=True, ci_repeats=100)
    assert report.ci_num_comparisons == 1


def test_no_ci_fields_without_calc_ci():
    report = leave_one_annotator_out(HAND)
    assert report.ci_family_alpha is None
    assert report.ci_num_comparisons is None


def test_ci_is_deterministic_for_a_seed():
    collection, _ = generate_synthetic_data(
        num_items=30, num_annotators=4, labels=["a", "b"], seed=8
    )
    r1 = leave_one_annotator_out(collection, calc_ci=True, ci_repeats=200, ci_seed=1)
    r2 = leave_one_annotator_out(collection, calc_ci=True, ci_repeats=200, ci_seed=1)
    assert [a.delta_ci for a in r1.annotators] == [a.delta_ci for a in r2.annotators]


def test_ci_counts_skipped_resamples():
    """Small data: some resamples have only all-'x' items, so kappa is undefined."""
    collection = _collection(
        {
            "i1": {"A1": "x", "A2": "x", "A3": "y"},
            "i2": {"A1": "x", "A2": "x", "A3": "x"},
            "i3": {"A1": "y", "A2": "y", "A3": "y"},
        }
    )
    report = leave_one_annotator_out(collection, calc_ci=True, ci_repeats=500)
    for r in report.annotators:
        if r.kappa_without is None:
            assert r.delta_ci is None
        else:
            assert r.delta_ci is not None
            assert 0 < r.delta_ci.skipped_resamples < 500


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"ci_repeats": 0}, "ci_repeats"),
        ({"ci_alpha": 0.0}, "ci_alpha"),
        ({"ci_alpha": 1.0}, "ci_alpha"),
    ],
)
def test_invalid_ci_arguments_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        leave_one_annotator_out(HAND, calc_ci=True, **kwargs)


@pytest.mark.parametrize(
    ("lower", "upper", "effect"),
    [
        (0.01, 0.2, "lowers_kappa"),
        (-0.2, -0.01, "raises_kappa"),
        (-0.1, 0.1, "no_clear_effect"),
        (0.0, 0.1, "no_clear_effect"),
    ],
)
def test_effect_reads_the_ci(lower, upper, effect):
    info = AnnotatorInfluence(
        annotator_id="a",
        items_rated=1,
        delta_ci=ConfidenceInterval(lower, upper, "bootstrap", 0.05),
    )
    assert info.effect == effect


def test_effect_is_none_without_ci():
    assert AnnotatorInfluence(annotator_id="a", items_rated=1).effect is None


# --- Robust outlier score (modified z-score of the deltas) ---


def test_robust_scores_formula():
    """
    deltas [0.01, 0.02, 0.03, 0.04, 0.50] -> median 0.03,
    deviations [0.02, 0.01, 0, 0.01, 0.47] -> MAD 0.01,
    so score = 0.6745 * (delta - 0.03) / 0.01.
    """
    scores, median, mad = _robust_scores([0.01, 0.02, 0.03, 0.04, 0.50])

    assert (median, mad) == (0.03, pytest.approx(0.01))
    assert scores == pytest.approx([-1.349, -0.6745, 0.0, 0.6745, 31.7015], abs=1e-9)


@pytest.mark.parametrize(
    "deltas",
    [
        [0.0, 0.0, 0.0, 0.2],  # over half identical, one far away
        [0.4, 0.4, 0.4],  # all identical
        [0.25],  # a single annotator
    ],
)
def test_robust_scores_are_none_when_mad_is_zero(deltas):
    """A MAD of 0 would divide by zero: no scores at all, instead of a fallback."""
    scores, _, mad = _robust_scores(deltas)
    assert mad == 0.0
    assert scores is None


def test_mad_zero_flags_no_one_and_is_not_proven():
    """
    HAND's deltas are [+2/3, -1/3, -1/3]: the median is -1/3 and the deviations
    are [1, 0, 0], so the MAD is 0. A3 stands out, but the rule cannot score it,
    so nobody is flagged and every verdict is "not proven", whatever the threshold.
    """
    for threshold in (3.5, 0.001):
        report = leave_one_annotator_out(HAND, outlier_threshold=threshold)

        assert report.mad_is_zero is True
        assert report.delta_median == pytest.approx(-1 / 3)
        assert report.delta_mad == pytest.approx(0.0)
        for r in report.annotators:
            assert r.delta is not None
            assert r.robust_score is None
            assert r.is_outlier is False
            assert r.outlier_status == "not_proven"


def test_outlier_flag_uses_the_threshold():
    """Fixture scores are 16.21 (spammer) and 13.75 (random_clicker)."""
    collection = load_csv(OUTLIERS_CSV)

    default = leave_one_annotator_out(collection)
    assert default.mad_is_zero is False
    assert default.outlier_threshold == 3.5
    assert [r.annotator_id for r in default.annotators if r.is_outlier] == [
        "spammer",
        "random_clicker",
    ]

    strict = leave_one_annotator_out(collection, outlier_threshold=15.0)
    assert strict.outlier_threshold == 15.0
    assert [r.annotator_id for r in strict.annotators if r.is_outlier] == ["spammer"]
    by_id = {r.annotator_id: r for r in strict.annotators}
    assert by_id["spammer"].outlier_status == "outlier"
    assert by_id["random_clicker"].outlier_status == "not_flagged"


def test_scores_ignore_annotators_without_a_delta():
    """
    Undefined entries are left out of the median and MAD. Here only A has a
    delta, and the MAD of a single value is 0, so A is "not proven" while B, C
    and D have no verdict at all.
    """
    collection = _collection(
        {
            "i1": {"A": "x", "B": "x", "C": "y"},
            "i2": {"A": "y", "C": "y", "D": "y"},
            "i3": {"A": "x", "B": "x", "D": "x"},
            "i4": {"A": "y", "B": "x", "C": "y"},
        }
    )
    report = leave_one_annotator_out(collection)
    by_id = {r.annotator_id: r for r in report.annotators}

    assert report.delta_median == pytest.approx(by_id["A"].delta)
    assert report.mad_is_zero is True
    assert by_id["A"].outlier_status == "not_proven"
    for a in ("B", "C", "D"):
        assert by_id[a].robust_score is None
        assert by_id[a].is_outlier is False
        assert by_id[a].outlier_status is None


@pytest.mark.parametrize(
    ("delta", "score", "is_outlier", "status"),
    [
        (0.1, 4.0, True, "outlier"),
        (0.1, 1.0, False, "not_flagged"),
        (0.1, None, False, "not_proven"),
        (None, None, False, None),
    ],
)
def test_outlier_status(delta, score, is_outlier, status):
    info = AnnotatorInfluence(
        annotator_id="a",
        items_rated=1,
        delta=delta,
        robust_score=score,
        is_outlier=is_outlier,
    )
    assert info.outlier_status == status


def test_no_annotator_with_a_delta_leaves_median_and_mad_unset():
    collection = _collection(
        {"i1": {"A": "x", "B": "x"}, "i2": {"A": "y", "B": "x"}},
    )
    report = leave_one_annotator_out(collection)
    assert report.delta_median is None
    assert report.delta_mad is None
    assert report.mad_is_zero is False
    assert all(r.outlier_status is None for r in report.annotators)


# --- tests/data/outliers.csv fixture ---


def test_outliers_fixture_shape():
    """120 items x 10 annotators, fully crossed, 3 labels."""
    collection = load_csv(OUTLIERS_CSV)
    annotators = {a.annotator_id for a in collection.annotations}

    assert len(collection.annotations) == 1200
    assert len({a.item_id for a in collection.annotations}) == 120
    assert annotators == {
        *(f"good_{i}" for i in range(1, 9)),
        "spammer",
        "random_clicker",
    }
    assert {a.label for a in collection.annotations} == {
        "positive",
        "negative",
        "neutral",
    }


def test_outliers_fixture_flags_exactly_the_two_planted_outliers():
    """The spammer and the random clicker are flagged; the 8 good ones are not."""
    report = leave_one_annotator_out(load_csv(OUTLIERS_CSV))

    flagged = [r.annotator_id for r in report.annotators if r.is_outlier]
    assert flagged == ["spammer", "random_clicker"]

    by_id = {r.annotator_id: r for r in report.annotators}
    assert by_id["spammer"].robust_score > 3.5
    assert by_id["random_clicker"].robust_score > 3.5
    for i in range(1, 9):
        good = by_id[f"good_{i}"]
        assert good.delta < 0  # removing a good annotator lowers kappa
        assert abs(good.robust_score) < 3.5


def test_outliers_fixture_removing_both_outliers_raises_kappa():
    """Kappa on the fixture is well below what the 8 good annotators agree on."""
    collection = load_csv(OUTLIERS_CSV)
    clean = AnnotationCollection(
        [
            a
            for a in collection.annotations
            if a.annotator_id not in {"spammer", "random_clicker"}
        ]
    )
    report = leave_one_annotator_out(collection)

    assert report.full_kappa == pytest.approx(0.3577, abs=5e-4)
    assert fleiss_kappa(clean) == pytest.approx(0.5773, abs=5e-4)
    # One at a time understates the pair: each delta is well under the joint gain.
    for annotator_id in ("spammer", "random_clicker"):
        info = next(r for r in report.annotators if r.annotator_id == annotator_id)
        assert info.delta < fleiss_kappa(clean) - report.full_kappa


def test_outliers_fixture_outliers_are_flagged_with_ci_too():
    report = leave_one_annotator_out(
        load_csv(OUTLIERS_CSV), calc_ci=True, ci_repeats=300
    )
    for info in report.annotators[:2]:
        assert info.is_outlier
        assert info.effect == "lowers_kappa"
