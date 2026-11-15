import random

import pytest

from fleisskit.fleiss import (
    _fleiss_kappa_components,
    _per_category_kappa_components,
    build_count_table,
    fleiss_kappa,
    per_category_kappa,
)
from fleisskit.generator import generate_synthetic_data
from fleisskit.models import Annotation, AnnotationCollection


def _make_ann(item_id: str, annotator_id: str, label: str) -> Annotation:
    """Helper to quickly create annotations for tests."""
    return Annotation(
        item_id=str(item_id), annotator_id=annotator_id, is_gold=False, label=label
    )


def test_build_count_table():
    """Builds a correct 2D count matrix."""
    collection = AnnotationCollection(
        [
            _make_ann("item_1", "A1", "Pos"),
            _make_ann("item_1", "A2", "Neg"),
            _make_ann("item_2", "A1", "Pos"),
            _make_ann("item_2", "A2", "Pos"),
        ]
    )
    table, labels = build_count_table(collection)

    assert labels == ["Neg", "Pos"]
    assert table == [[1, 1], [0, 2]]


def test_fleiss_kappa_varying_ratings_raises_error():
    """Raises ValueError if items don't all have the same number of ratings."""
    collection = AnnotationCollection(
        [
            _make_ann("item_1", "A1", "Pos"),
            _make_ann("item_1", "A2", "Pos"),
            _make_ann("item_2", "A1", "Pos"),
        ]
    )
    with pytest.raises(ValueError, match="Varying number of ratings per item"):
        fleiss_kappa(collection)


WIKI_TABLE = [
    [0, 0, 0, 0, 14],
    [0, 2, 6, 4, 2],
    [0, 0, 3, 5, 6],
    [0, 3, 9, 2, 0],
    [2, 2, 8, 1, 1],
    [7, 7, 0, 0, 0],
    [3, 2, 6, 3, 0],
    [2, 5, 3, 2, 2],
    [6, 5, 2, 1, 0],
    [0, 2, 2, 3, 7],
]
WIKI_LABELS = ["1", "2", "3", "4", "5"]


def test_fleiss_kappa_wikipedia_example():
    """Exact worked example from the Fleiss' kappa Wikipedia article."""
    kappa, P_bar, P_bar_e = _fleiss_kappa_components(WIKI_TABLE)

    assert P_bar == pytest.approx(0.378, abs=0.001)
    assert P_bar_e == pytest.approx(0.213, abs=0.001)
    assert kappa == pytest.approx(0.210, abs=0.001)


# Per-category kappa tests


def test_per_category_kappa_wikipedia_example():
    """Per-label kappa on the Wikipedia table (values computed by hand from the formula)."""
    result = _per_category_kappa_components(WIKI_TABLE, WIKI_LABELS)

    assert result == pytest.approx(
        {
            "1": 157 / 780,
            "2": 0.0797,
            "3": 0.1716,
            "4": 0.0304,
            "5": 0.5077,
        },
        abs=1e-4,
    )


def test_per_category_kappa_weighted_average_equals_overall_kappa():
    """Overall kappa is the p_j*q_j-weighted average of the per-label kappas."""
    per_label = _per_category_kappa_components(WIKI_TABLE, WIKI_LABELS)
    overall, _, _ = _fleiss_kappa_components(WIKI_TABLE)

    N, n = len(WIKI_TABLE), sum(WIKI_TABLE[0])
    weights = {}
    for j, label in enumerate(WIKI_LABELS):
        p = sum(row[j] for row in WIKI_TABLE) / (N * n)
        weights[label] = p * (1 - p)

    weighted = sum(weights[lbl] * per_label[lbl] for lbl in WIKI_LABELS) / sum(
        weights.values()
    )
    assert weighted == pytest.approx(overall)


def test_per_category_kappa_perfect_agreement():
    """Every label gets kappa 1.0 when annotators always agree."""
    collection = AnnotationCollection(
        [
            _make_ann("item_1", "A1", "Pos"),
            _make_ann("item_1", "A2", "Pos"),
            _make_ann("item_2", "A1", "Neg"),
            _make_ann("item_2", "A2", "Neg"),
        ]
    )
    assert per_category_kappa(collection) == pytest.approx({"Neg": 1.0, "Pos": 1.0})


def test_per_category_kappa_uses_label_names_as_keys():
    """Picks out the weak label: 'Neu' is always mixed up, 'Pos' is consistent."""
    collection = AnnotationCollection(
        [
            _make_ann("item_1", "A1", "Pos"),
            _make_ann("item_1", "A2", "Pos"),
            _make_ann("item_2", "A1", "Neg"),
            _make_ann("item_2", "A2", "Neu"),
            _make_ann("item_3", "A1", "Neu"),
            _make_ann("item_3", "A2", "Neg"),
        ]
    )
    result = per_category_kappa(collection)

    assert set(result) == {"Neg", "Neu", "Pos"}
    assert result["Pos"] == pytest.approx(1.0)
    # Neg / Neu: p = 1/3, q = 2/3, disagreement = 1*1 + 1*1 = 2
    # kappa = 1 - 2 / (3 * 2 * 1 * 2/9) = -0.5
    assert result["Neg"] == pytest.approx(-0.5)
    assert result["Neu"] == pytest.approx(-0.5)


def test_per_category_kappa_skips_label_with_p_j_one():
    """A label used for every rating (p_j = 1) is skipped, not a division by zero."""
    collection = AnnotationCollection(
        [
            _make_ann("item_1", "A1", "Pos"),
            _make_ann("item_1", "A2", "Pos"),
            _make_ann("item_2", "A1", "Pos"),
            _make_ann("item_2", "A2", "Pos"),
        ]
    )
    assert per_category_kappa(collection) == {}


def test_per_category_kappa_skips_label_with_p_j_zero():
    """A label column that is never used (p_j = 0) is skipped."""
    table = [[2, 0, 0], [0, 2, 0], [1, 1, 0]]
    result = _per_category_kappa_components(table, ["A", "B", "Unused"])

    assert "Unused" not in result
    assert set(result) == {"A", "B"}


def test_per_category_kappa_returns_one_value_per_label():
    """
    Returns exactly k entries, one per label, keyed by label name.

    k is the number of distinct labels in the data. This holds whenever k >= 2:
    every used label then has 0 < p_j < 1, so none is skipped. (With a single
    label, p_j = 1 and the result is empty; see the p_j = 1 test above.)
    """
    labels = ["L0", "L1", "L2", "L3", "L4"]
    k = len(labels)
    collection, _ = generate_synthetic_data(
        num_items=100, num_annotators=4, labels=labels, accuracy=0.7, seed=7
    )
    # Precondition: every label was actually used, so k really is len(labels).
    assert {ann.label for ann in collection.annotations} == set(labels)

    result = per_category_kappa(collection)

    assert len(result) == k
    assert sorted(result) == labels
    assert all(isinstance(v, float) for v in result.values())


def test_per_category_kappa_varying_ratings_raises_error():
    """Same constant-n requirement as overall Fleiss' kappa."""
    collection = AnnotationCollection(
        [
            _make_ann("item_1", "A1", "Pos"),
            _make_ann("item_1", "A2", "Pos"),
            _make_ann("item_2", "A1", "Pos"),
        ]
    )
    with pytest.raises(ValueError, match="Varying number of ratings per item"):
        per_category_kappa(collection)


# Per-category kappa property tests on generated datasets


@pytest.mark.parametrize("seed", range(50))
def test_per_category_kappa_weighted_average_matches_overall_on_generated_data(seed):
    """
    On 50 seeded synthetic datasets, the p_j*q_j-weighted average of the per-label
    kappas must equal the overall Fleiss' kappa within 1e-9.

    The dataset shape (items, annotators, labels, accuracy, spammers) is also
    derived from the seed, so each run covers a different configuration.
    skip_rate stays 0 because Fleiss' kappa needs a constant number of ratings
    per item.
    """
    shape_rng = random.Random(seed)
    num_labels = shape_rng.randint(2, 6)
    collection, _ = generate_synthetic_data(
        num_items=shape_rng.randint(10, 200),
        num_annotators=shape_rng.randint(2, 10),
        labels=[f"L{i}" for i in range(num_labels)],
        accuracy=shape_rng.uniform(0.3, 0.95),
        spammer_share=shape_rng.choice([0.0, 0.2, 0.4]),
        skip_rate=0.0,
        seed=seed,
    )

    per_label = per_category_kappa(collection)
    overall = fleiss_kappa(collection)

    # Weights p_j * q_j, taken from the same count table the metrics use.
    table, labels = build_count_table(collection)
    total_ratings = len(table) * sum(table[0])
    weights = {}
    for j, label in enumerate(labels):
        p_j = sum(row[j] for row in table) / total_ratings
        weights[label] = p_j * (1.0 - p_j)

    # Skipped labels (p_j = 0 or 1) have zero weight, so leaving them out of
    # the sum is exact. Check that only those labels were dropped.
    assert set(per_label) == {lbl for lbl, w in weights.items() if w > 0}
    assert len(per_label) >= 2

    weighted = sum(weights[lbl] * k for lbl, k in per_label.items()) / sum(
        weights.values()
    )
    assert weighted == pytest.approx(overall, abs=1e-9)
