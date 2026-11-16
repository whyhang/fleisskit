from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from fleisskit.io import load_csv
from fleisskit.spam import DEFAULT_PARAMS, CombinerParams, combine_signals
from fleisskit.spam_tuning import (
    ARCHETYPES,
    SPAM_ARCHETYPES,
    START_PARAMS,
    VALIDATION_SEEDS,
    DetectionMetrics,
    LabelledDataset,
    _calibrate,
    _prepare,
    check_calibration,
    evaluate,
    fit_confidence_calibration,
    random_labelled_dataset,
    simulate_labelled_dataset,
    tune_combiner,
)

SPAM_KNOWN_CSV = Path(__file__).parent / "data" / "spam_known.csv"
SPAM_KNOWN_KINDS = ["honest"] * 6 + [
    "honest_fast",
    "honest_weak",
    "constant_spammer",
    "biased_clicker",
    "random_clicker",
    "random_clicker",
]
SPAM_KNOWN_IDS = [f"honest_{i}" for i in range(1, 7)] + [
    "fast_honest",
    "weak_honest",
    "constant_spammer",
    "biased_clicker",
    "random_clicker_fast",
    "random_clicker_slow",
]


def _key(a):
    return (a.item_id, a.annotator_id)


# --- Simulation ---


def test_simulation_is_deterministic_for_a_seed():
    kinds = ["honest", "random_clicker", "constant_spammer"]
    a = simulate_labelled_dataset(kinds, 30, ["x", "y"], seed=4, gold_share=0.2)
    b = simulate_labelled_dataset(kinds, 30, ["x", "y"], seed=4, gold_share=0.2)
    c = simulate_labelled_dataset(kinds, 30, ["x", "y"], seed=5, gold_share=0.2)

    assert a.collection.annotations == b.collection.annotations
    assert a.collection.annotations != c.collection.annotations


def test_simulation_roles_and_gold():
    kinds = ["honest", "honest_fast", "constant_spammer", "random_clicker"]
    ds = simulate_labelled_dataset(kinds, 50, ["x", "y", "z"], seed=1, gold_share=0.1)
    anns = ds.collection.annotations

    assert ds.archetypes == {
        "annotator_00": "honest",
        "annotator_01": "honest_fast",
        "annotator_02": "constant_spammer",
        "annotator_03": "random_clicker",
    }
    assert ds.spammers == {"annotator_02", "annotator_03"}
    assert len(anns) == 50 * 4
    gold = [a for a in anns if a.is_gold]
    assert len({a.item_id for a in gold}) == 5
    assert all(a.gold_label is not None for a in gold)
    # A constant spammer always gives one label.
    assert len({a.label for a in anns if a.annotator_id == "annotator_02"}) == 1


def test_simulation_sparse_design_and_skips():
    kinds = ["honest"] * 10
    ds = simulate_labelled_dataset(kinds, 100, ["x", "y"], seed=3, ratings_per_item=3)
    per_item = Counter(a.item_id for a in ds.collection.annotations)
    assert set(per_item.values()) == {3}

    skipped = simulate_labelled_dataset(kinds, 100, ["x", "y"], seed=3, skip_rate=0.5)
    assert 300 < len(skipped.collection.annotations) < 700


def test_simulation_without_durations():
    ds = simulate_labelled_dataset(
        ["honest"], 5, ["x", "y"], seed=1, with_durations=False
    )
    assert all(a.duration_sec is None for a in ds.collection.annotations)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"archetypes": ["nobody"]}, "Unknown archetypes"),
        ({"labels": ["x"]}, "at least 2 labels"),
        ({"annotator_ids": ["a", "b"]}, "one annotator id"),
        ({"ratings_per_item": 5}, "ratings_per_item"),
    ],
)
def test_simulation_rejects_bad_arguments(kwargs, match):
    args = {"archetypes": ["honest"], "num_items": 5, "labels": ["x", "y"], "seed": 1}
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        simulate_labelled_dataset(**args)


def test_random_datasets_vary_in_shape_and_are_reproducible():
    datasets = [random_labelled_dataset(s) for s in range(30)]
    again = random_labelled_dataset(7)

    assert again.collection.annotations == datasets[7].collection.annotations
    sizes = {len(ds.archetypes) for ds in datasets}
    assert min(sizes) >= 5 and max(sizes) <= 30 and len(sizes) > 5
    assert any(not ds.spammers for ds in datasets)
    assert any(ds.spammers for ds in datasets)
    kinds = {k for ds in datasets for k in ds.archetypes.values()}
    assert kinds == set(ARCHETYPES)


def test_labelled_dataset_explicit_spammers_win():
    ds = LabelledDataset(
        collection=simulate_labelled_dataset(["honest"], 5, ["x", "y"], 1).collection,
        archetypes={"annotator_00": "whatever"},
        spammers={"annotator_00"},
    )
    assert ds.spammers == {"annotator_00"}


def test_spam_known_fixture_matches_its_simulation():
    """tests/data/spam_known.csv is exactly this simulation, written to CSV."""
    ds = simulate_labelled_dataset(
        SPAM_KNOWN_KINDS,
        num_items=150,
        labels=["positive", "negative", "neutral"],
        seed=2,
        gold_share=0.1,
        annotator_ids=SPAM_KNOWN_IDS,
    )
    assert sorted(load_csv(SPAM_KNOWN_CSV).annotations, key=_key) == sorted(
        ds.collection.annotations, key=_key
    )


# --- Evaluation ---


def test_detection_metrics_formulas():
    m = DetectionMetrics(
        true_positives=8,
        false_positives=2,
        false_negatives=4,
        true_negatives=86,
        beta=0.5,
        by_archetype={},
    )
    assert m.precision == pytest.approx(0.8)
    assert m.recall == pytest.approx(8 / 12)
    assert m.false_positive_rate == pytest.approx(2 / 88)
    p, r = 0.8, 8 / 12
    assert m.f_beta == pytest.approx(1.25 * p * r / (0.25 * p + r))


def test_detection_metrics_with_nothing_to_find():
    m = DetectionMetrics(0, 0, 0, 10, 0.5, {})
    assert (m.precision, m.recall, m.false_positive_rate) == (1.0, 1.0, 0.0)


def test_evaluate_matches_combine_signals():
    """The tuner's fast path counts exactly what combine_signals flags."""
    datasets = [random_labelled_dataset(s) for s in range(10)]
    metrics = evaluate(DEFAULT_PARAMS, datasets)

    tp = fp = 0
    for row in _prepare(datasets, DEFAULT_PARAMS):
        flagged = combine_signals(row.signals, DEFAULT_PARAMS).is_spammer
        tp += flagged and row.is_spammer
        fp += flagged and not row.is_spammer
    assert (metrics.true_positives, metrics.false_positives) == (tp, fp)


# --- Tuning ---

SMALL_GRIDS = {
    "weight_grid": (0.0, 0.4, 0.7, 1.0),
    "ok_grid": (0.3, 0.6, 1.0),
    "threshold_grid": (0.4, 0.5, 0.6),
}


def test_tuning_improves_on_a_bad_start_and_is_deterministic():
    datasets = [random_labelled_dataset(s) for s in range(12)]
    # A bad start: only speed counts, so nothing can be flagged.
    bad = replace(
        START_PARAMS,
        agreement_weight=0.0,
        variety_weight=0.0,
        gold_weight=0.0,
    )

    first = tune_combiner(datasets, start=bad, **SMALL_GRIDS)
    second = tune_combiner(datasets, start=bad, **SMALL_GRIDS)

    assert evaluate(bad, datasets).f_beta == 0.0
    assert first.train_metrics.f_beta > 0.8
    assert first.params == second.params
    assert first.params.threshold in SMALL_GRIDS["threshold_grid"]
    assert first.params.min_labels == bad.min_labels  # policy is not tuned


def test_tuning_can_use_explicit_labels_on_real_style_data():
    """tune_combiner accepts any LabelledDataset, e.g. a checked-in file."""
    ds = LabelledDataset(
        collection=load_csv(SPAM_KNOWN_CSV),
        archetypes=dict(zip(SPAM_KNOWN_IDS, SPAM_KNOWN_KINDS, strict=True)),
    )
    result = tune_combiner([ds], **SMALL_GRIDS)
    assert isinstance(result.params, CombinerParams)
    assert result.train_metrics.precision == 1.0


def test_default_params_on_held_out_simulations():
    """
    A quick slice of the validation seeds (the full run is
    `python -m fleisskit.spam_tuning`). The defaults must stay precise.
    """
    datasets = [random_labelled_dataset(s) for s in list(VALIDATION_SEEDS)[:40]]
    m = evaluate(DEFAULT_PARAMS, datasets)

    assert m.precision >= 0.9
    assert m.false_positive_rate <= 0.01
    flagged, total = m.by_archetype["constant_spammer"]
    assert flagged / total >= 0.95
    assert all(k in m.by_archetype for k in SPAM_ARCHETYPES)


def test_tuning_drops_a_stale_calibration():
    """Re-tuned params no longer match the calibration fitted for the old ones."""
    datasets = [random_labelled_dataset(s) for s in range(6)]
    result = tune_combiner(datasets, start=DEFAULT_PARAMS, **SMALL_GRIDS)
    assert DEFAULT_PARAMS.calibration is not None
    assert result.params.calibration is None


# --- Confidence calibration ---


def test_calibrate_groups_pools_and_smooths():
    """
    Scores 0.50-0.59 (10 flagged): 6 spam; 0.60-0.69 (10): 10 spam.
    Groups of 10 stay apart (60% < 100%), then Jeffreys smoothing:
    (6 + 0.5) / 11 = 0.5909 and (10 + 0.5) / 11 = 0.9545.
    """
    flagged = [(0.50 + i / 100, i >= 4) for i in range(10)]
    flagged += [(0.60 + i / 100, True) for i in range(10)]
    cal = _calibrate(flagged, spam=16, total=100, min_group=10)

    assert cal.steps == (
        (0.50, pytest.approx(6.5 / 11)),
        (0.60, pytest.approx(10.5 / 11)),
    )
    assert cal.base_rate == pytest.approx(0.16)


def test_calibrate_merges_groups_that_would_make_confidence_fall():
    """A higher-scoring group with a lower spam rate is pooled with the one below."""
    flagged = [(0.5 + i / 100, True) for i in range(10)]  # 10/10 spam
    flagged += [(0.6 + i / 100, i >= 2) for i in range(10)]  # 8/10 spam
    cal = _calibrate(flagged, spam=18, total=50, min_group=10)

    assert cal.steps == ((0.5, pytest.approx(18.5 / 21)),)


def test_calibrate_merges_neighbouring_groups_with_equal_rates():
    """
    Two groups that are both 100% spam become one step, so the rate is based
    on all 20: (20 + 0.5) / 21 = 0.976, more than (10 + 0.5) / 11 = 0.955 each.
    """
    flagged = [(0.5 + i / 100, True) for i in range(20)]
    cal = _calibrate(flagged, spam=20, total=100, min_group=10)
    assert cal.steps == ((0.5, pytest.approx(20.5 / 21)),)


def test_calibrate_keeps_tied_scores_together():
    """30 annotators at exactly 1.0 never straddle a group boundary."""
    flagged = [(0.6, True)] * 5 + [(1.0, True)] * 30
    cal = _calibrate(flagged, spam=35, total=100, min_group=10)

    lows = [low for low, _ in cal.steps]
    assert len(lows) == len(set(lows))
    assert cal.steps == ((0.6, pytest.approx(35.5 / 36)),)


def test_calibrate_needs_enough_data():
    with pytest.raises(ValueError, match="at least 10 flagged"):
        _calibrate([(0.6, True)] * 9, spam=9, total=20, min_group=10)
    with pytest.raises(ValueError, match="both spammers and honest"):
        _calibrate([(0.6, True)] * 10, spam=10, total=10, min_group=10)


def test_fit_and_check_calibration_on_simulated_data():
    datasets = [random_labelled_dataset(s) for s in range(2000, 2060)]
    params = replace(DEFAULT_PARAMS, calibration=None)
    cal = fit_confidence_calibration(datasets, params, min_group=20)

    probs = [p for _, p in cal.steps]
    assert probs == sorted(probs)
    assert all(0.5 < p < 1.0 for p in probs)

    held_out = [random_labelled_dataset(s) for s in range(1000, 1030)]
    check = check_calibration(held_out, replace(params, calibration=cal))
    assert check.brier_score < 0.05
    assert sum(n for *_, n in check.steps) > 0


def test_check_calibration_needs_a_calibration():
    with pytest.raises(ValueError, match="no calibration"):
        check_calibration([], replace(DEFAULT_PARAMS, calibration=None))
