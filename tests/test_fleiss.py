import random
import time

import pytest

from fleisskit.fleiss import (
    _calculate_bootstrap_ci,
    _calculate_formula_ci,
    _fleiss_kappa_components,
    _per_category_kappa_components,
    analyze_fleiss,
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


# --- Per-category kappa tests ---


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
    """Returns exactly k entries, one per label, keyed by label name."""
    labels = ["L0", "L1", "L2", "L3", "L4"]
    k = len(labels)
    collection, _ = generate_synthetic_data(
        num_items=100, num_annotators=4, labels=labels, accuracy=0.7, seed=7
    )
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


@pytest.mark.parametrize("seed", range(15))
def test_per_category_kappa_weighted_average_matches_overall_on_generated_data(seed):
    """
    On seeded synthetic datasets, the p_j*q_j-weighted average of the per-label
    kappas must equal the overall Fleiss' kappa within 1e-9.
    """
    shape_rng = random.Random(seed)
    num_labels = shape_rng.randint(2, 6)
    collection, _ = generate_synthetic_data(
        num_items=shape_rng.randint(10, 50),
        num_annotators=shape_rng.randint(2, 5),
        labels=[f"L{i}" for i in range(num_labels)],
        accuracy=shape_rng.uniform(0.3, 0.95),
        spammer_share=shape_rng.choice([0.0, 0.2, 0.4]),
        skip_rate=0.0,
        seed=seed,
    )

    per_label = per_category_kappa(collection)
    overall = fleiss_kappa(collection)

    table, labels = build_count_table(collection)
    total_ratings = len(table) * sum(table[0])
    weights = {}
    for j, label in enumerate(labels):
        p_j = sum(row[j] for row in table) / total_ratings
        weights[label] = p_j * (1.0 - p_j)

    assert set(per_label) == {lbl for lbl, w in weights.items() if w > 0}

    weighted = sum(weights[lbl] * k for lbl, k in per_label.items()) / sum(
        weights.values()
    )
    assert weighted == pytest.approx(overall, abs=1e-9)


# --- Confidence Interval & Analysis Tests ---


def test_wikipedia_ci_contains_estimate():
    """Both CI methods must encompass the known Wikipedia estimate."""
    # 1. Overall Formula CI
    kappa, _, _ = _fleiss_kappa_components(WIKI_TABLE)
    ci_form = _calculate_formula_ci(WIKI_TABLE, kappa, 0.05)
    assert ci_form.lower_bound <= kappa <= ci_form.upper_bound

    # 2. Bootstrap CI
    ci_boot_overall, ci_boot_cats = _calculate_bootstrap_ci(
        WIKI_TABLE, WIKI_LABELS, repeats=500, seed=42, alpha=0.05
    )
    assert ci_boot_overall.lower_bound <= kappa <= ci_boot_overall.upper_bound

    # 3. Check per-category encompasses the estimates
    cat_estimates = _per_category_kappa_components(WIKI_TABLE, WIKI_LABELS)
    for lbl, est in cat_estimates.items():
        ci_cat = ci_boot_cats[lbl]
        assert ci_cat.lower_bound <= est <= ci_cat.upper_bound


def test_bootstrap_seed_determinism():
    """Identical seeds must yield identical bounds."""
    ci1_overall, _ = _calculate_bootstrap_ci(WIKI_TABLE, WIKI_LABELS, 100, 99, 0.05)
    ci2_overall, _ = _calculate_bootstrap_ci(WIKI_TABLE, WIKI_LABELS, 100, 99, 0.05)

    assert ci1_overall.lower_bound == ci2_overall.lower_bound
    assert ci1_overall.upper_bound == ci2_overall.upper_bound


def test_bootstrap_performance():
    """1000 resamples on 500 items must finish in well under 10 seconds."""
    collection, _ = generate_synthetic_data(
        num_items=500, num_annotators=5, labels=["A", "B", "C"], seed=42
    )

    start_time = time.time()
    analyze_fleiss(
        collection, calc_ci=True, ci_method="bootstrap", ci_repeats=1000, ci_seed=1
    )
    duration = time.time() - start_time

    assert duration < 10.0


def test_coverage_simulation():
    """
    Simulates datasets where the true kappa is known.
    The 95% interval should contain the true kappa in roughly ~95% of runs.
    """
    # Estimate true kappa on a massive dataset to establish ground truth
    massive_col, _ = generate_synthetic_data(
        num_items=5000, num_annotators=3, labels=["A", "B"], accuracy=0.75, seed=1
    )
    true_kappa = analyze_fleiss(massive_col).overall_kappa.estimate

    contains_true = 0
    num_datasets = 50

    for i in range(num_datasets):
        small_col, _ = generate_synthetic_data(
            num_items=40,
            num_annotators=3,
            labels=["A", "B"],
            accuracy=0.75,
            seed=i + 100,
        )
        report = analyze_fleiss(
            small_col, calc_ci=True, ci_method="bootstrap", ci_repeats=200, ci_seed=i
        )

        if (
            report.overall_kappa.ci.lower_bound
            <= true_kappa
            <= report.overall_kappa.ci.upper_bound
        ):
            contains_true += 1

    coverage_ratio = contains_true / num_datasets

    # Expect 95% coverage, allow tolerance (85% to 100%) due to small subset sizes in tests
    assert 0.85 <= coverage_ratio <= 1.0
