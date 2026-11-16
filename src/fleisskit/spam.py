"""
Spam annotator detection from four separate signals and one combiner.

Signals (each is its own function, computed for every annotator):

1. agreement_with_others: chance-corrected agreement with the majority vote of
   the other annotators on shared items (like Cohen's kappa against that
   majority). A spammer scores about 0, even one who always picks the most
   common label.
2. annotation_speed: median seconds per annotation.
3. label_variety: Shannon entropy (bits) of the annotator's labels, next to
   the entropy of the other annotators' majority labels on the same items.
   An annotator who always picks the same label scores 0. An honest
   annotator who was given a one-sided batch matches the others on it.
4. gold_accuracy: accuracy on gold items, plus Cohen's kappa against the gold
   labels (chance-corrected, so guessing the common label scores about 0).

compute_signals() runs all four and rescales them so 1.0 means "typical":
agreement and speed against the group median, and variety against the others
on the same items. That makes them comparable across tasks of different
difficulty and pace. Gold kappa is absolute, since the gold labels are the
truth.

combine_signals() runs once per annotator. It turns each available signal into
a suspicion between 0 and 1:

    suspicion = clip(1 - value / ok, 0, 1)

so a value at or above `ok` is no suspicion, and 0 (what a pure spammer scores
on every signal) is full suspicion. The suspicions are combined as a noisy-OR:

    score = 1 - product over signals of (1 - weight * suspicion)

Each signal can only add suspicion. A signal that doesn't fire never cancels
out one that does, which matters because each signal only catches some kinds of
spammer (a slow random clicker looks normal on speed and variety). The annotator
is flagged when the score reaches the threshold and at least one signal other
than speed gives a reason. The weights, `ok` points and threshold are fitted on
labelled data by fleisskit.spam_tuning (see DEFAULT_PARAMS).

The score is not a probability. Each flagged annotator also gets a confidence:
the share of flagged annotators with a similar score who really were spammers
in labelled calibration data (see ConfidenceCalibration).
"""

import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass

from fleisskit.models import AnnotationCollection

# Below this group median kappa the annotators barely agree with each other, so
# low agreement is not evidence against any one of them.
MIN_GROUP_AGREEMENT = 0.1

# Variety is only judged when the others' majority gave a label other than the
# annotator's most used one on at least this many of their items. On a
# one-sided batch an honest annotator and a constant spammer give the same
# labels, so there is no evidence either way.
MIN_VARIETY_EVIDENCE = 5.0

SIGNALS = ("agreement", "speed", "variety", "gold")


# --- Signal 1: agreement with others ---


@dataclass(frozen=True)
class AgreementSignal:
    """Chance-corrected agreement of one annotator with the others' majority."""

    # None if undefined (no shared items, or chance agreement is 1).
    kappa: float | None
    # Items this annotator shares with at least one other annotator.
    items: int


def agreement_with_others(
    collection: AnnotationCollection,
) -> dict[str, AgreementSignal]:
    """
    For annotator a, over the items where at least one other annotator also
    gave a label, let T_i be the label(s) most often given by the others on
    item i (a tie gives several):

        m_i(j) = 1/|T_i| if j is in T_i, else 0   (the others' majority)
        P_o    = mean over items of m_i(a's label)
        P_e    = sum over labels j of pi_a(j) * mean over items of m_i(j)
        kappa  = (P_o - P_e) / (1 - P_e)

    pi_a is a's own label distribution on those items. The majority of several
    annotators is much more reliable than any one of them, so honest
    annotators score clearly above 0 even when some peers are noisy. An
    annotator who always picks label c gets P_o = P_e, so kappa = 0 exactly,
    however common c is.
    """
    by_item: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for ann in collection.annotations:
        if ann.label is not None:
            by_item[ann.item_id].append((ann.annotator_id, ann.label))

    annotators: set[str] = set()
    obs_sum: dict[str, float] = defaultdict(float)
    items: dict[str, int] = defaultdict(int)
    own_counts: dict[str, Counter[str]] = defaultdict(Counter)
    majority_share: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )

    for entries in by_item.values():
        for annotator_id, label in entries:
            annotators.add(annotator_id)
            others = Counter(lbl for a, lbl in entries if a != annotator_id)
            if not others:
                continue
            top = max(others.values())
            majority = [lbl for lbl, count in others.items() if count == top]
            credit = 1.0 / len(majority)
            if label in majority:
                obs_sum[annotator_id] += credit
            items[annotator_id] += 1
            own_counts[annotator_id][label] += 1
            for lbl in majority:
                majority_share[annotator_id][lbl] += credit

    result: dict[str, AgreementSignal] = {}
    for annotator_id in sorted(annotators):
        n = items[annotator_id]
        if n == 0:
            result[annotator_id] = AgreementSignal(kappa=None, items=0)
            continue
        p_o = obs_sum[annotator_id] / n
        p_e = sum(
            (count / n) * (majority_share[annotator_id][lbl] / n)
            for lbl, count in own_counts[annotator_id].items()
        )
        kappa = None if p_e >= 1.0 - 1e-12 else (p_o - p_e) / (1.0 - p_e)
        result[annotator_id] = AgreementSignal(kappa=kappa, items=n)
    return result


# --- Signal 2: speed ---


def annotation_speed(collection: AnnotationCollection) -> dict[str, float | None]:
    """
    Median seconds per labelled annotation, per annotator. The median is used
    so one long break doesn't hide an otherwise rushed annotator. None if the
    annotator has no durations.
    """
    durations: dict[str, list[float]] = defaultdict(list)
    annotators: set[str] = set()
    for ann in collection.annotations:
        if ann.label is None:
            continue
        annotators.add(ann.annotator_id)
        if ann.duration_sec is not None:
            durations[ann.annotator_id].append(ann.duration_sec)

    return {
        a: statistics.median(durations[a]) if durations[a] else None
        for a in sorted(annotators)
    }


# --- Signal 3: label variety ---


@dataclass(frozen=True)
class VarietySignal:
    """How varied one annotator's labels are, next to the others on the same items."""

    # Shannon entropy in bits of the annotator's own labels.
    entropy_bits: float
    # Entropy of the others' majority labels on the items this annotator shares
    # with them (ties split evenly). None if they share no items.
    reference_bits: float | None
    # Items (ties split evenly) where the others' majority is not this
    # annotator's most used label: the chances they had to show variety.
    evidence_items: float = 0.0


def _entropy_bits(weights: dict[str, float]) -> float:
    total = sum(weights.values())
    entropy = -sum(
        (w / total) * math.log2(w / total) for w in weights.values() if w > 0
    )
    # A single label sums to -0.0; max(-0.0, 0.0) would keep the sign.
    return entropy if entropy > 0 else 0.0


def label_variety(collection: AnnotationCollection) -> dict[str, VarietySignal]:
    """
    Shannon entropy in bits of each annotator's label distribution:

        H = -sum over labels j of p(j) * log2(p(j))

    0 means one label for everything. Using k labels evenly gives log2(k).

    The reference is the same entropy, but for the others' majority label on
    each item this annotator shares with them. So an annotator who got a
    one-sided batch is compared with what that batch really contained, not
    with the whole dataset. evidence_items counts the items where that
    majority is not the annotator's most used label.
    """
    by_item: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for ann in collection.annotations:
        if ann.label is not None:
            by_item[ann.item_id].append((ann.annotator_id, ann.label))

    own: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    reference: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for entries in by_item.values():
        for annotator_id, label in entries:
            own[annotator_id][label] += 1.0
            others = Counter(lbl for a, lbl in entries if a != annotator_id)
            if not others:
                continue
            top = max(others.values())
            majority = [lbl for lbl, count in others.items() if count == top]
            for lbl in majority:
                reference[annotator_id][lbl] += 1.0 / len(majority)

    result: dict[str, VarietySignal] = {}
    for a in sorted(own):
        # Most used label; ties broken by name so the result is deterministic.
        top_label = min(own[a], key=lambda lbl: (-own[a][lbl], lbl))
        ref = reference.get(a)
        result[a] = VarietySignal(
            entropy_bits=_entropy_bits(own[a]),
            reference_bits=_entropy_bits(ref) if ref else None,
            evidence_items=(
                sum(w for lbl, w in ref.items() if lbl != top_label) if ref else 0.0
            ),
        )
    return result


# --- Signal 4: gold accuracy ---


@dataclass(frozen=True)
class GoldSignal:
    """How one annotator did on gold items."""

    accuracy: float
    # Cohen's kappa against the gold labels. None if chance agreement is 1
    # (e.g. a single gold item, or all gold items share one label that the
    # annotator also always used).
    kappa: float | None
    items: int


def gold_accuracy(collection: AnnotationCollection) -> dict[str, GoldSignal]:
    """
    For each annotator who labelled at least one gold item:

        accuracy = share of gold items where label == gold_label
        P_e      = sum over labels j of pi_a(j) * pi_gold(j)
        kappa    = (accuracy - P_e) / (1 - P_e)

    pi_a and pi_gold are the annotator's and the gold label distributions on
    those same gold items. Annotators with no gold items are left out.
    """
    pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for ann in collection.annotations:
        if ann.is_gold and ann.label is not None and ann.gold_label is not None:
            pairs[ann.annotator_id].append((ann.label, ann.gold_label))

    result: dict[str, GoldSignal] = {}
    for annotator_id in sorted(pairs):
        items = pairs[annotator_id]
        n = len(items)
        accuracy = sum(1 for label, gold in items if label == gold) / n
        own = Counter(label for label, _ in items)
        gold = Counter(g for _, g in items)
        p_e = sum((own[j] / n) * (gold[j] / n) for j in own)
        kappa = None if p_e >= 1.0 - 1e-12 else (accuracy - p_e) / (1.0 - p_e)
        result[annotator_id] = GoldSignal(accuracy=accuracy, kappa=kappa, items=n)
    return result


# --- Signals for each annotator, scaled against the group ---


@dataclass(frozen=True)
class AnnotatorSignals:
    """All four signals for one annotator, raw and relative to the group."""

    annotator_id: str
    labels: int
    # Raw values (None when not available).
    agreement_kappa: float | None
    agreement_items: int
    median_seconds: float | None
    entropy_bits: float
    reference_entropy_bits: float | None
    variety_evidence_items: float
    gold_accuracy: float | None
    gold_kappa: float | None
    gold_items: int
    # Relative values (1.0 = typical): agreement and speed against the group
    # median, variety against the others on the same items. None when not
    # available, e.g. no durations, or the group has no meaningful agreement.
    agreement: float | None
    speed: float | None
    variety: float | None


def _median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def compute_signals(collection: AnnotationCollection) -> dict[str, AnnotatorSignals]:
    """
    Runs every signal function and rescales them so 1.0 is "typical":

        agreement = kappa_a / median(kappa)        (None if median < 0.1)
        speed     = seconds_a / median(seconds)    (None if no durations)
        variety   = entropy_a / reference_entropy_a
                    (None unless the others' majority used a label other than
                    a's most used one on at least 5 of a's items)

    The medians are over annotators, so the spammers themselves cannot shift
    what counts as typical, as long as they are under half of the annotators.
    """
    agreement = agreement_with_others(collection)
    speed = annotation_speed(collection)
    variety = label_variety(collection)
    gold = gold_accuracy(collection)

    label_counts = Counter(a.annotator_id for a in collection.annotations if a.label)

    med_kappa = _median_or_none(
        [s.kappa for s in agreement.values() if s.kappa is not None]
    )
    med_seconds = _median_or_none([v for v in speed.values() if v is not None])

    result: dict[str, AnnotatorSignals] = {}
    for annotator_id in sorted(variety):
        agr = agreement.get(annotator_id, AgreementSignal(kappa=None, items=0))
        seconds = speed.get(annotator_id)
        var = variety[annotator_id]
        g = gold.get(annotator_id)

        rel_agreement = None
        if (
            agr.kappa is not None
            and med_kappa is not None
            and med_kappa >= MIN_GROUP_AGREEMENT
        ):
            rel_agreement = agr.kappa / med_kappa
        rel_speed = None
        if seconds is not None and med_seconds:
            rel_speed = seconds / med_seconds
        rel_variety = None
        if var.reference_bits and var.evidence_items >= MIN_VARIETY_EVIDENCE:
            rel_variety = var.entropy_bits / var.reference_bits

        result[annotator_id] = AnnotatorSignals(
            annotator_id=annotator_id,
            labels=label_counts[annotator_id],
            agreement_kappa=agr.kappa,
            agreement_items=agr.items,
            median_seconds=seconds,
            entropy_bits=var.entropy_bits,
            reference_entropy_bits=var.reference_bits,
            variety_evidence_items=var.evidence_items,
            gold_accuracy=g.accuracy if g else None,
            gold_kappa=g.kappa if g else None,
            gold_items=g.items if g else 0,
            agreement=rel_agreement,
            speed=rel_speed,
            variety=rel_variety,
        )
    return result


# --- Combiner ---


@dataclass(frozen=True)
class ConfidenceCalibration:
    """
    Maps a flagged annotator's score to the chance they really are a spammer.

    Fitted by fleisskit.spam_tuning.fit_confidence_calibration on labelled data:
    flagged annotators are grouped by score (at least 50 per group), groups are
    merged until the spam rate never falls as the score rises, and each rate is
    smoothed towards 50% by half a spammer and half an honest annotator
    (Jeffreys prior), so it is never exactly 0% or 100%.
    """

    # (lowest score in the group, share of spammers), by ascending score.
    steps: tuple[tuple[float, float], ...]
    # Share of spammers among all annotators in the calibration data. The
    # confidence assumes spammers are about this common.
    base_rate: float

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError("calibration needs at least one step")
        scores = [score for score, _ in self.steps]
        if scores != sorted(scores):
            raise ValueError("calibration steps must be sorted by score")
        if any(not 0.0 < prob < 1.0 for _, prob in self.steps):
            raise ValueError("calibration probabilities must be between 0 and 1")
        if not 0.0 < self.base_rate < 1.0:
            raise ValueError("base_rate must be between 0 and 1")

    def confidence(self, score: float, base_rate: float | None = None) -> float:
        """
        Chance that an annotator flagged with this score is a spammer.

        With base_rate, the answer is moved to a dataset where that share of
        annotators are spammers, by Bayes' rule on the odds:

            odds = odds_calibrated * (base_rate / (1 - base_rate))
                                   / (calibration_rate / (1 - calibration_rate))
        """
        prob = self.steps[0][1]
        for min_score, step_prob in self.steps:
            if score >= min_score:
                prob = step_prob
        if base_rate is None:
            return prob
        if not 0.0 < base_rate < 1.0:
            raise ValueError("base_rate must be between 0 and 1")
        odds = prob / (1.0 - prob)
        odds *= (base_rate / (1.0 - base_rate)) / (
            self.base_rate / (1.0 - self.base_rate)
        )
        return odds / (1.0 + odds)


@dataclass(frozen=True)
class CombinerParams:
    """
    Weights, `ok` points and flag threshold for combine_signals.

    For each signal, suspicion = clip(1 - value / ok, 0, 1). Weights are in
    [0, 1]: a weight is what a full-suspicion reading of that signal is worth
    on its own, and a weight of 0 switches the signal off.
    """

    agreement_weight: float
    speed_weight: float
    variety_weight: float
    gold_weight: float
    agreement_ok: float
    speed_ok: float
    variety_ok: float
    gold_ok: float
    threshold: float
    # Evidence needed before a signal (or the annotator) is judged at all.
    min_labels: int = 10
    min_agreement_items: int = 10
    min_gold_items: int = 5
    # Turns a flagged annotator's score into a confidence. Only valid for the
    # parameters it was fitted with, so re-tuned parameters start without one.
    calibration: ConfidenceCalibration | None = None

    def __post_init__(self) -> None:
        for signal in SIGNALS:
            if not 0.0 <= self.weight(signal) <= 1.0:
                raise ValueError(f"{signal}_weight must be between 0 and 1")
            if self.ok(signal) <= 0.0:
                raise ValueError(f"{signal}_ok must be greater than 0")
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")

    def weight(self, signal: str) -> float:
        return float(getattr(self, f"{signal}_weight"))

    def ok(self, signal: str) -> float:
        return float(getattr(self, f"{signal}_ok"))


# Fitted by `python -m fleisskit.spam_tuning` (F0.5, block coordinate descent)
# on 300 simulated datasets with known roles; see that module for the
# simulation. On 300 separate validation datasets: precision 0.979, recall
# 0.716, and 0.3% of honest annotators flagged.
#
# What the fitted numbers amount to:
# - variety (weight 1.0, ok 1.0) can flag alone once an annotator's label
#   variety is at most half of the others' on the same items.
# - agreement (0.4) cannot flag alone; with very fast work (speed under ~15%
#   of the median) it can.
# - speed (0.6) never flags alone (see combine_signals).
# - gold (0.1) barely counts. In the simulation most annotators have few or
#   no gold items. Re-tune with more weight on gold if you have plenty of it.
#
# The calibration was fitted on 600 further simulated datasets (14.8% of
# annotators spammers). On the validation datasets, predicted vs real spam
# rate per step: 89% vs 81% (31 flagged), 95% vs 96% (22), 99% vs 97% (105),
# 99.9% vs 99.7% (355); Brier score 0.019.
DEFAULT_PARAMS = CombinerParams(
    agreement_weight=0.4,
    speed_weight=0.6,
    variety_weight=1.0,
    gold_weight=0.1,
    agreement_ok=0.7,
    speed_ok=0.2,
    variety_ok=1.0,
    gold_ok=0.5,
    threshold=0.5,
    calibration=ConfidenceCalibration(
        steps=((0.5008, 0.8922), (0.5496, 0.951), (0.5914, 0.9861), (0.737, 0.9992)),
        base_rate=0.1481,
    ),
)


@dataclass(frozen=True)
class SpamVerdict:
    """Combiner output for one annotator."""

    annotator_id: str
    is_spammer: bool
    # "spammer", "ok", or "insufficient_data" (too few labels, or only speed
    # could be measured).
    status: str
    # Noisy-OR of the weighted suspicions, 0..1; None if nothing to judge.
    score: float | None
    # Suspicion (0..1) per signal that could be used.
    suspicion: dict[str, float]
    # Signals with suspicion >= 0.5, strongest first.
    reasons: list[str]
    signals: AnnotatorSignals
    # For flagged annotators: chance (0..1) they really are a spammer, from
    # the params' calibration. None if not flagged, or no calibration.
    confidence: float | None = None

    @property
    def confidence_percent(self) -> int | None:
        """
        Confidence as a whole percent, rounded down, so it never reads 100%:
        a calibration fitted on finite data can't justify certainty.
        """
        if self.confidence is None:
            return None
        return min(math.floor(self.confidence * 100), 99)


def _suspicion(value: float, ok: float) -> float:
    """1 at value <= 0, 0 at value >= ok, linear in between."""
    return min(1.0, max(0.0, 1.0 - value / ok))


def _available_values(
    signals: AnnotatorSignals, params: CombinerParams
) -> dict[str, float]:
    """The signal values that have enough evidence behind them to be used."""
    values: dict[str, float] = {}
    if (
        signals.agreement is not None
        and signals.agreement_items >= params.min_agreement_items
    ):
        values["agreement"] = signals.agreement
    if signals.speed is not None:
        values["speed"] = signals.speed
    if signals.variety is not None:
        values["variety"] = signals.variety
    if signals.gold_kappa is not None and signals.gold_items >= params.min_gold_items:
        values["gold"] = signals.gold_kappa
    return values


def _score(
    values: dict[str, float], params: CombinerParams
) -> tuple[float, dict[str, float], list[str], bool]:
    """
    The combiner's arithmetic on already-available signal values:
    (score, suspicion per signal, reasons, is_spammer). Shared by
    combine_signals and the tuner, so both always agree.
    """
    suspicion = {
        s: _suspicion(v, params.ok(s))
        for s, v in values.items()
        if params.weight(s) > 0
    }
    clear = 1.0
    for s, v in suspicion.items():
        clear *= 1.0 - params.weight(s) * v
    score = 1.0 - clear
    reasons = sorted(
        (s for s, v in suspicion.items() if v >= 0.5),
        key=lambda s: (-suspicion[s], SIGNALS.index(s)),
    )
    is_spammer = score >= params.threshold and any(r != "speed" for r in reasons)
    return score, suspicion, reasons, is_spammer


def _has_enough_data(
    signals: AnnotatorSignals, values: dict[str, float], params: CombinerParams
) -> bool:
    """Enough labels, and at least one usable signal besides speed."""
    usable = {s for s in values if params.weight(s) > 0}
    return signals.labels >= params.min_labels and bool(usable - {"speed"})


def combine_signals(
    signals: AnnotatorSignals,
    params: CombinerParams = DEFAULT_PARAMS,
    base_rate: float | None = None,
) -> SpamVerdict:
    """
    Turns one annotator's signals into a spam flag.

        score = 1 - product over available signals of (1 - w_s * suspicion_s)

    Flagged when score >= threshold and at least one signal other than speed
    has a suspicion of 0.5 or more. Being fast is never enough on its own:
    quick, careful annotators exist.

    The status is "insufficient_data" when the annotator has fewer than
    min_labels labels, or when speed is the only signal that can be measured.

    Flagged annotators get a confidence from params.calibration. Pass
    base_rate (the share of spammers you expect among your annotators) to
    adjust it when spam is more or less common than in the calibration data.
    """
    values = _available_values(signals, params)
    if not _has_enough_data(signals, values, params):
        return SpamVerdict(
            annotator_id=signals.annotator_id,
            is_spammer=False,
            status="insufficient_data",
            score=None,
            suspicion={},
            reasons=[],
            signals=signals,
        )

    score, suspicion, reasons, is_spammer = _score(values, params)
    confidence = None
    if is_spammer and params.calibration is not None:
        confidence = params.calibration.confidence(score, base_rate)
    return SpamVerdict(
        annotator_id=signals.annotator_id,
        is_spammer=is_spammer,
        status="spammer" if is_spammer else "ok",
        score=score,
        suspicion=suspicion,
        reasons=reasons,
        signals=signals,
        confidence=confidence,
    )


def detect_spammers(
    collection: AnnotationCollection,
    params: CombinerParams = DEFAULT_PARAMS,
    base_rate: float | None = None,
) -> list[SpamVerdict]:
    """
    Computes every signal, then runs the combiner for each annotator.
    base_rate adjusts the flagged annotators' confidence (see combine_signals).

    Sorted by score (highest first). Annotators with insufficient data come
    last. Ties are broken by annotator_id.
    """
    verdicts = [
        combine_signals(s, params, base_rate)
        for s in compute_signals(collection).values()
    ]
    verdicts.sort(key=lambda v: (v.score is None, -(v.score or 0.0), v.annotator_id))
    return verdicts
