from dataclasses import replace
from pathlib import Path

import pytest

from fleisskit.io import load_csv
from fleisskit.models import Annotation, AnnotationCollection
from fleisskit.spam import (
    DEFAULT_PARAMS,
    MIN_GROUP_AGREEMENT,
    AnnotatorSignals,
    CombinerParams,
    ConfidenceCalibration,
    SpamVerdict,
    _suspicion,
    agreement_with_others,
    annotation_speed,
    combine_signals,
    compute_signals,
    detect_spammers,
    gold_accuracy,
    label_variety,
)

DATA = Path(__file__).parent / "data"

# Checked-in fixture with known roles. It is exactly
# simulate_labelled_dataset(SPAM_KNOWN_KINDS, num_items=150,
#     labels=["positive", "negative", "neutral"], seed=2, gold_share=0.1,
#     annotator_ids=SPAM_KNOWN_IDS)
# (see test_spam_known_fixture_matches_its_simulation in test_spam_tuning.py).
SPAM_KNOWN_CSV = DATA / "spam_known.csv"
OUTLIERS_CSV = DATA / "outliers.csv"


def _ann(
    item: str,
    annotator: str,
    label: str | None,
    seconds: float | None = None,
    gold: str | None = None,
) -> Annotation:
    return Annotation(
        item_id=item,
        annotator_id=annotator,
        is_gold=gold is not None,
        label=label,
        duration_sec=seconds,
        gold_label=gold,
    )


def _grid(rows: dict[str, str]) -> AnnotationCollection:
    """{annotator: "xyxy"} -> one label per item (i0, i1, ...), one char each."""
    return AnnotationCollection(
        [
            _ann(f"i{i}", annotator, label)
            for annotator, labels in rows.items()
            for i, label in enumerate(labels)
        ]
    )


def _signals(**overrides) -> AnnotatorSignals:
    """A typical, clean annotator; override fields to make them suspicious."""
    base = {
        "annotator_id": "a",
        "labels": 100,
        "agreement_kappa": 0.5,
        "agreement_items": 100,
        "median_seconds": 10.0,
        "entropy_bits": 1.5,
        "reference_entropy_bits": 1.5,
        "variety_evidence_items": 50.0,
        "gold_accuracy": None,
        "gold_kappa": None,
        "gold_items": 0,
        "agreement": 1.0,
        "speed": 1.0,
        "variety": 1.0,
    }
    base.update(overrides)
    return AnnotatorSignals(**base)


# --- Signal 1: agreement with others ---


def test_agreement_hand_computed():
    """
    A and B always agree; C always says x.
    For A, the others' majority per item is {x}, {x,y}, {x}, {x,y}:
      P_o = (1 + 0.5 + 1 + 0.5) / 4 = 0.75
      P_e = 0.5 * 0.75 (x) + 0.5 * 0.25 (y) = 0.5  ->  kappa = 0.5
    For C, the majority is A/B's label: P_o = 0.5, P_e = 1 * 0.5 -> kappa = 0.
    """
    result = agreement_with_others(_grid({"A": "xyxy", "B": "xyxy", "C": "xxxx"}))

    assert result["A"].kappa == pytest.approx(0.5)
    assert result["B"].kappa == pytest.approx(0.5)
    assert result["C"].kappa == pytest.approx(0.0)
    assert result["A"].items == 4


def test_agreement_constant_spammer_is_exactly_zero_even_on_the_common_label():
    """Always picking the label most items have still earns no agreement."""
    honest = "xxxxxxxyyz"
    collection = _grid({"h1": honest, "h2": honest, "h3": honest, "spam": "x" * 10})
    result = agreement_with_others(collection)

    assert result["spam"].kappa == pytest.approx(0.0, abs=1e-12)
    assert result["h1"].kappa == pytest.approx(1.0)


def test_agreement_needs_shared_items():
    collection = AnnotationCollection([_ann("i1", "A", "x"), _ann("i2", "B", "y")])
    result = agreement_with_others(collection)
    assert result["A"].kappa is None
    assert result["A"].items == 0


def test_agreement_undefined_when_chance_agreement_is_one():
    result = agreement_with_others(_grid({"A": "xxx", "B": "xxx"}))
    assert result["A"].kappa is None
    assert result["A"].items == 3


def test_agreement_ignores_missing_labels():
    collection = AnnotationCollection(
        [*_grid({"A": "xy", "B": "xy"}).annotations, _ann("i0", "C", None)]
    )
    result = agreement_with_others(collection)
    assert "C" not in result
    assert result["A"].items == 2


# --- Signal 2: speed ---


def test_speed_is_median_seconds_of_labelled_annotations():
    collection = AnnotationCollection(
        [
            _ann("i1", "A", "x", 2.0),
            _ann("i2", "A", "x", 4.0),
            _ann("i3", "A", "x", 300.0),  # one long break barely matters
            _ann("i4", "A", None, 0.1),  # skipped: not counted
            _ann("i1", "B", "x", None),  # no durations at all
        ]
    )
    assert annotation_speed(collection) == {"A": 4.0, "B": None}


# --- Signal 3: label variety ---


def test_variety_entropy_in_bits():
    """x,x,y,y is 1 bit; one label is 0 bits; 4 labels evenly is 2 bits."""
    result = label_variety(_grid({"two": "xxyy", "one": "xxxx", "four": "wxyz"}))

    assert result["two"].entropy_bits == pytest.approx(1.0)
    assert result["one"].entropy_bits == 0.0
    assert str(result["one"].entropy_bits) == "0.0"  # not -0.0
    assert result["four"].entropy_bits == pytest.approx(2.0)


def test_variety_reference_is_the_others_majority_on_the_same_items():
    """
    For `one`, the others' majority per item is {x,w}, {x}, {y}, {y,z}:
    x 1.5, y 1.5, w 0.5, z 0.5 out of 4 -> entropy of (3/8, 3/8, 1/8, 1/8).
    """
    result = label_variety(_grid({"two": "xxyy", "one": "xxxx", "four": "wxyz"}))
    expected = -2 * (3 / 8) * __import__("math").log2(3 / 8) - 2 * (1 / 8) * __import__(
        "math"
    ).log2(1 / 8)
    assert result["one"].reference_bits == pytest.approx(expected)


def test_variety_evidence_counts_items_where_the_majority_used_another_label():
    """`one` mostly says x; the others' majority is not x on i2 and i3 (plus
    half of i0, a w/x tie): 0.5 + 1 + 1 = 2.5 items."""
    result = label_variety(_grid({"two": "xxyy", "one": "xxxx", "four": "wxyz"}))
    assert result["one"].evidence_items == pytest.approx(2.5)


def test_variety_needs_evidence_before_it_is_used():
    """Fewer than MIN_VARIETY_EVIDENCE chances to differ: no variety signal."""
    few = compute_signals(_grid({"h1": "xxxxy", "h2": "xxxxy", "spam": "xxxxx"}))
    assert few["spam"].variety_evidence_items == pytest.approx(1.0)
    assert few["spam"].variety is None

    many = compute_signals(
        _grid({"h1": "xyxyxyxyxy", "h2": "xyxyxyxyxy", "spam": "xxxxxxxxxx"})
    )
    assert many["spam"].variety_evidence_items == pytest.approx(5.0)
    assert many["spam"].variety == 0.0


def test_variety_reference_is_none_without_shared_items():
    collection = AnnotationCollection([_ann("i1", "A", "x"), _ann("i2", "B", "y")])
    assert label_variety(collection)["A"].reference_bits is None


# --- Signal 4: gold accuracy ---


def test_gold_accuracy_and_kappa_hand_computed():
    """
    Gold x,x,y,y; answers x,y,y,y -> accuracy 3/4.
    P_e = 0.25 * 0.5 (x) + 0.75 * 0.5 (y) = 0.5 -> kappa = (0.75 - 0.5) / 0.5 = 0.5.
    """
    collection = AnnotationCollection(
        [
            _ann("g1", "A", "x", gold="x"),
            _ann("g2", "A", "y", gold="x"),
            _ann("g3", "A", "y", gold="y"),
            _ann("g4", "A", "y", gold="y"),
            _ann("i1", "A", "x"),  # not gold: ignored
            _ann("i1", "B", "x"),  # B saw no gold: left out
        ]
    )
    result = gold_accuracy(collection)

    assert set(result) == {"A"}
    assert result["A"].accuracy == pytest.approx(0.75)
    assert result["A"].kappa == pytest.approx(0.5)
    assert result["A"].items == 4


def test_gold_kappa_of_a_constant_answer_is_zero():
    collection = AnnotationCollection(
        [_ann(f"g{i}", "A", "x", gold=g) for i, g in enumerate("xxxy")]
    )
    result = gold_accuracy(collection)["A"]
    assert result.accuracy == pytest.approx(0.75)
    assert result.kappa == pytest.approx(0.0)


def test_gold_kappa_undefined_for_a_single_gold_item():
    result = gold_accuracy(AnnotationCollection([_ann("g1", "A", "x", gold="x")]))
    assert result["A"].accuracy == 1.0
    assert result["A"].kappa is None


# --- Signals relative to the group ---


def test_compute_signals_scales_against_the_group():
    collection = AnnotationCollection(
        [
            _ann(f"i{i}", annotator, label, seconds)
            for annotator, labels, seconds in [
                ("A", "xyzxyzxyzxyz", 10.0),
                ("B", "xyzxyzxyzxyz", 20.0),
                ("C", "xyzxyzxyzxyx", 30.0),
            ]
            for i, label in enumerate(labels)
        ]
    )
    signals = compute_signals(collection)
    kappas = {a: s.agreement_kappa for a, s in signals.items()}
    median_kappa = sorted(kappas.values())[1]

    for a, s in signals.items():
        assert s.agreement == pytest.approx(kappas[a] / median_kappa)
        assert s.variety_evidence_items >= 5
        assert s.variety == pytest.approx(s.entropy_bits / s.reference_entropy_bits)
        assert s.labels == 12
    assert signals["A"].speed == pytest.approx(0.5)  # 10 s vs median 20 s
    assert signals["C"].speed == pytest.approx(1.5)


def test_agreement_is_switched_off_when_the_group_barely_agrees():
    """Nobody agrees beyond chance, so low agreement isn't evidence against anyone."""
    collection = _grid({"A": "xyxyxyxy", "B": "xxyyxxyy", "C": "xyyxxyyx"})
    signals = compute_signals(collection)
    median = sorted(s.agreement_kappa for s in signals.values())[1]

    assert median < MIN_GROUP_AGREEMENT
    assert all(s.agreement is None for s in signals.values())
    assert all(s.agreement_kappa is not None for s in signals.values())


def test_no_durations_means_no_speed_signal():
    signals = compute_signals(_grid({"A": "xy", "B": "xy"}))
    assert signals["A"].speed is None
    assert signals["A"].median_seconds is None


# --- Combiner ---


@pytest.mark.parametrize(
    ("value", "ok", "expected"),
    [
        (-0.3, 0.5, 1.0),
        (0.0, 0.5, 1.0),
        (0.25, 0.5, 0.5),
        (0.5, 0.5, 0.0),
        (2.0, 0.5, 0.0),
    ],
)
def test_suspicion_ramp(value, ok, expected):
    assert _suspicion(value, ok) == pytest.approx(expected)


EVEN = CombinerParams(
    agreement_weight=0.5,
    speed_weight=0.5,
    variety_weight=0.5,
    gold_weight=0.5,
    agreement_ok=1.0,
    speed_ok=1.0,
    variety_ok=1.0,
    gold_ok=1.0,
    threshold=0.5,
)


def test_combiner_is_a_noisy_or_of_weighted_suspicions():
    """
    agreement 0.2 -> suspicion 0.8; variety 0.6 -> 0.4; speed 1.0 -> 0.
    score = 1 - (1 - 0.5*0.8) * (1 - 0.5*0.4) * (1 - 0) = 1 - 0.6 * 0.8 = 0.52
    """
    verdict = combine_signals(_signals(agreement=0.2, variety=0.6), EVEN)

    assert verdict.suspicion == pytest.approx(
        {"agreement": 0.8, "speed": 0.0, "variety": 0.4}
    )
    assert verdict.score == pytest.approx(0.52)
    assert verdict.is_spammer is True
    assert verdict.status == "spammer"
    assert verdict.reasons == ["agreement"]


def test_a_signal_that_does_not_fire_does_not_clear_the_annotator():
    """Adding a clean reading leaves the noisy-OR score unchanged."""
    without_gold = combine_signals(_signals(agreement=0.2, variety=0.6), EVEN)
    with_clean_gold = combine_signals(
        _signals(agreement=0.2, variety=0.6, gold_kappa=1.0, gold_items=20), EVEN
    )
    assert with_clean_gold.suspicion["gold"] == 0.0
    assert with_clean_gold.score == pytest.approx(without_gold.score)


def test_speed_alone_never_flags():
    params = CombinerParams(
        agreement_weight=0.5,
        speed_weight=1.0,
        variety_weight=0.5,
        gold_weight=0.5,
        agreement_ok=1.0,
        speed_ok=1.0,
        variety_ok=1.0,
        gold_ok=1.0,
        threshold=0.5,
    )
    verdict = combine_signals(_signals(speed=0.01), params)

    assert verdict.score == pytest.approx(0.99)
    assert verdict.reasons == ["speed"]
    assert verdict.is_spammer is False
    assert verdict.status == "ok"


def test_reasons_are_sorted_by_suspicion():
    verdict = combine_signals(_signals(agreement=0.4, variety=0.1, speed=0.3), EVEN)
    assert verdict.reasons == ["variety", "speed", "agreement"]


def test_too_few_labels_is_insufficient_data():
    verdict = combine_signals(_signals(labels=9, agreement=0.0, variety=0.0), EVEN)
    assert verdict.status == "insufficient_data"
    assert verdict.is_spammer is False
    assert verdict.score is None


def test_only_speed_available_is_insufficient_data():
    verdict = combine_signals(_signals(agreement=None, variety=None, speed=0.01), EVEN)
    assert verdict.status == "insufficient_data"
    assert verdict.is_spammer is False


def test_signals_need_enough_items_to_count():
    few_shared = combine_signals(_signals(agreement=0.0, agreement_items=9), EVEN)
    assert "agreement" not in few_shared.suspicion

    few_gold = combine_signals(_signals(gold_kappa=0.0, gold_items=4), EVEN)
    assert "gold" not in few_gold.suspicion
    enough_gold = combine_signals(_signals(gold_kappa=0.0, gold_items=5), EVEN)
    assert enough_gold.suspicion["gold"] == 1.0


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("agreement_weight", 1.5, "agreement_weight"),
        ("gold_weight", -0.1, "gold_weight"),
        ("speed_ok", 0.0, "speed_ok"),
        ("threshold", 0.0, "threshold"),
    ],
)
def test_combiner_params_are_validated(field, value, match):
    kwargs = {
        "agreement_weight": 0.5,
        "speed_weight": 0.5,
        "variety_weight": 0.5,
        "gold_weight": 0.5,
        "agreement_ok": 0.5,
        "speed_ok": 0.5,
        "variety_ok": 0.5,
        "gold_ok": 0.5,
        "threshold": 0.5,
        field: value,
    }
    with pytest.raises(ValueError, match=match):
        CombinerParams(**kwargs)


# --- The tuned defaults, on hand-made cases ---


def test_defaults_variety_alone_flags_a_one_label_annotator():
    verdict = combine_signals(_signals(variety=0.0))
    assert verdict.is_spammer
    assert verdict.reasons == ["variety"]


def test_defaults_low_agreement_alone_does_not_flag():
    """At-chance agreement alone could be a weak honest annotator (see README)."""
    verdict = combine_signals(_signals(agreement=0.0))
    assert verdict.score == pytest.approx(DEFAULT_PARAMS.agreement_weight)
    assert not verdict.is_spammer


def test_defaults_low_agreement_and_very_fast_flags():
    verdict = combine_signals(_signals(agreement=0.0, speed=0.05))
    assert verdict.is_spammer
    assert verdict.reasons == ["agreement", "speed"]


def test_defaults_fast_but_accurate_is_not_flagged():
    verdict = combine_signals(_signals(speed=0.05))
    assert not verdict.is_spammer


# --- detect_spammers end to end ---


def test_spam_known_fixture_flags_the_detectable_spammers():
    verdicts = detect_spammers(load_csv(SPAM_KNOWN_CSV))
    flagged = [v.annotator_id for v in verdicts if v.is_spammer]

    assert flagged == ["constant_spammer", "biased_clicker", "random_clicker_fast"]
    # The hard honest cases are not flagged.
    by_id = {v.annotator_id: v for v in verdicts}
    assert not by_id["fast_honest"].is_spammer
    assert not by_id["weak_honest"].is_spammer
    for i in range(1, 7):
        assert by_id[f"honest_{i}"].score < 0.1


def test_spam_known_fixture_slow_random_clicker_is_a_known_blind_spot():
    """
    A random clicker who works at a normal pace, with only 15 gold items, looks
    like a weak honest annotator to the tuned defaults. It scores close to the
    threshold but is not flagged. If this starts failing because the detector
    got better, update the README's limitations.
    """
    by_id = {v.annotator_id: v for v in detect_spammers(load_csv(SPAM_KNOWN_CSV))}
    slow = by_id["random_clicker_slow"]

    assert not slow.is_spammer
    assert 0.35 < slow.score < DEFAULT_PARAMS.threshold
    assert set(slow.reasons) == {"agreement", "gold"}


def test_outliers_fixture_flags_both_planted_outliers():
    """A fixture built by other code (no gold): both outliers, no good annotators."""
    verdicts = detect_spammers(load_csv(OUTLIERS_CSV))
    assert [v.annotator_id for v in verdicts if v.is_spammer] == [
        "spammer",
        "random_clicker",
    ]


def _one_sided_batch(batch_labels: str) -> AnnotationCollection:
    """
    40 items; h1..h4 rate all of them. `batch` only got items 0-19, which are
    all x, except that item 3 is ambiguous and three of the four others said y.
    """
    rows = []
    for i in range(40):
        truth = "x" if i < 20 else ("y" if i % 2 else "z")
        for n, a in enumerate(("h1", "h2", "h3", "h4")):
            label = "y" if (i == 3 and n < 3) else truth
            rows.append(_ann(f"i{i}", a, label, 10.0))
        if i < 20:
            rows.append(_ann(f"i{i}", "batch", batch_labels[i], 10.0))
    return AnnotationCollection(rows)


def test_honest_annotator_on_a_one_sided_batch_is_not_flagged():
    """
    `batch` said x on all 20 of their items, which is right for 19 of them.
    Compared with the whole dataset (3 labels) that looks like a constant
    spammer. Compared with the others on the same items it isn't evidence: the
    others' majority differed on only 1 item, so variety is not judged.
    """
    signals = compute_signals(_one_sided_batch("x" * 20))
    verdict = combine_signals(signals["batch"])

    assert signals["batch"].entropy_bits == 0.0
    assert signals["batch"].variety_evidence_items == pytest.approx(1.0)
    assert signals["batch"].variety is None
    assert not verdict.is_spammer


def test_detect_spammers_sorts_by_score_with_insufficient_data_last():
    collection = AnnotationCollection(
        [
            *load_csv(OUTLIERS_CSV).annotations,
            _ann("item_000", "newcomer", "positive", 5.0),
        ]
    )
    verdicts = detect_spammers(collection)
    scores = [v.score for v in verdicts if v.score is not None]

    assert scores == sorted(scores, reverse=True)
    assert verdicts[-1].annotator_id == "newcomer"
    assert verdicts[-1].status == "insufficient_data"


# --- Confidence for flagged annotators ---

CAL = ConfidenceCalibration(
    steps=((0.5, 0.8), (0.7, 0.95), (0.9, 0.999)), base_rate=0.2
)


@pytest.mark.parametrize(
    ("score", "expected"),
    [(0.5, 0.8), (0.69, 0.8), (0.7, 0.95), (0.95, 0.999), (1.0, 0.999), (0.4, 0.8)],
)
def test_calibration_looks_up_the_step_for_a_score(score, expected):
    """Each step covers scores from its lower bound up; below the first, the first."""
    assert CAL.confidence(score) == pytest.approx(expected)


def test_calibration_base_rate_shift_is_bayes_on_the_odds():
    """
    0.8 at a 20% calibration rate is odds 4. For a 5% base rate:
    4 * (0.05 / 0.95) / (0.2 / 0.8) = 0.8421 -> 0.8421 / 1.8421 = 0.4571.
    """
    assert CAL.confidence(0.5, base_rate=0.05) == pytest.approx(0.4571, abs=1e-4)
    assert CAL.confidence(0.5, base_rate=0.2) == pytest.approx(0.8)
    assert CAL.confidence(0.5, base_rate=0.5) > 0.8


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"steps": ()}, "at least one step"),
        ({"steps": ((0.7, 0.9), (0.5, 0.95))}, "sorted"),
        ({"steps": ((0.5, 1.0),)}, "between 0 and 1"),
        ({"base_rate": 0.0}, "base_rate"),
    ],
)
def test_calibration_is_validated(kwargs, match):
    args = {"steps": ((0.5, 0.9),), "base_rate": 0.1}
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        ConfidenceCalibration(**args)


def test_calibration_rejects_a_bad_base_rate_at_lookup():
    with pytest.raises(ValueError, match="base_rate"):
        CAL.confidence(0.5, base_rate=1.0)


@pytest.mark.parametrize(
    ("confidence", "percent"),
    [(None, None), (0.8921, 89), (0.9899, 98), (0.9992, 99), (0.99999, 99)],
)
def test_confidence_percent_rounds_down_and_never_reads_100(confidence, percent):
    verdict = SpamVerdict(
        annotator_id="a",
        is_spammer=confidence is not None,
        status="spammer",
        score=0.9,
        suspicion={},
        reasons=[],
        signals=_signals(),
        confidence=confidence,
    )
    assert verdict.confidence_percent == percent


def test_only_flagged_annotators_get_a_confidence():
    params = replace(EVEN, calibration=CAL)
    flagged = combine_signals(_signals(agreement=0.2, variety=0.6), params)
    clean = combine_signals(_signals(), params)

    assert flagged.is_spammer
    assert flagged.confidence == pytest.approx(CAL.confidence(flagged.score))
    assert not clean.is_spammer
    assert clean.confidence is None


def test_no_calibration_means_no_confidence():
    verdict = combine_signals(_signals(agreement=0.2, variety=0.6), EVEN)
    assert verdict.is_spammer
    assert verdict.confidence is None


def test_base_rate_is_passed_through_to_the_confidence():
    params = replace(EVEN, calibration=CAL)
    verdict = combine_signals(
        _signals(agreement=0.2, variety=0.6), params, base_rate=0.05
    )
    assert verdict.confidence == pytest.approx(CAL.confidence(verdict.score, 0.05))


def test_default_calibration_rises_with_score():
    steps = DEFAULT_PARAMS.calibration.steps
    assert [s for s, _ in steps] == sorted(s for s, _ in steps)
    probs = [p for _, p in steps]
    assert probs == sorted(probs)
    assert steps[0][0] >= DEFAULT_PARAMS.threshold


def test_spam_known_fixture_confidences():
    verdicts = detect_spammers(load_csv(SPAM_KNOWN_CSV))
    confidence = {
        v.annotator_id: v.confidence_percent for v in verdicts if v.is_spammer
    }
    assert confidence == {
        "constant_spammer": 99,
        "biased_clicker": 98,
        "random_clicker_fast": 98,
    }
