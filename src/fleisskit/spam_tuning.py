"""
Tunes the spam combiner on labelled data (annotators whose role is known).

    python -m fleisskit.spam_tuning

This simulates datasets with known honest and spam annotators, fits
CombinerParams on the training seeds, fits the confidence calibration on
separate calibration seeds, and reports both on separate validation seeds.
The printed parameters are the ones in fleisskit.spam.DEFAULT_PARAMS.

If you have real annotators whose roles you know, pass them to
tune_combiner() as LabelledDataset objects instead of simulated ones.

Simulated annotator types:

    honest            skill 0.50-0.95, 6-40 s per item
    honest_fast       skill 0.50-0.95, 1.5-4 s per item (quick, not a spammer)
    honest_weak       skill 0.15-0.40, 6-40 s per item (poor, not a spammer)
    constant_spammer  always the same label
    random_clicker    uniformly random labels
    biased_clicker    random labels, but 60-90% on one favourite label

Spammers are fast (0.4-2.5 s) 60% of the time, otherwise normal speed.

Half the datasets are dense (5-15 annotators, everyone rates every item) and
half are sparse (a pool of 8-30 annotators, 3-5 of them per item), which is
how most crowdsourcing works. In sparse data the "majority of the others" is
only 2-4 people, so agreement is a weaker signal and gold matters more.
"Skill" s is the chance to rise above guessing on an item of difficulty d:
P(correct) = 1/k + s * (1 - d) * (1 - 1/k), with d drawn from 0-0.4 per item.
"""

import random
from dataclasses import dataclass, field, replace
from typing import Any

from fleisskit.models import Annotation, AnnotationCollection
from fleisskit.spam import (
    SIGNALS,
    AnnotatorSignals,
    CombinerParams,
    ConfidenceCalibration,
    _available_values,
    _has_enough_data,
    _score,
    compute_signals,
)

SPAM_ARCHETYPES = ("constant_spammer", "random_clicker", "biased_clicker")
HONEST_ARCHETYPES = ("honest", "honest_fast", "honest_weak")
ARCHETYPES = HONEST_ARCHETYPES + SPAM_ARCHETYPES

TRAIN_SEEDS = range(300)
VALIDATION_SEEDS = range(1000, 1300)
CALIBRATION_SEEDS = range(2000, 2600)

# Flagged annotators per group when fitting the confidence calibration.
MIN_CALIBRATION_GROUP = 50

# Neutral starting point for the search: every signal equal, halfway ramps.
START_PARAMS = CombinerParams(
    agreement_weight=0.5,
    speed_weight=0.5,
    variety_weight=0.5,
    gold_weight=0.5,
    agreement_ok=0.5,
    speed_ok=0.5,
    variety_ok=0.5,
    gold_ok=0.5,
    threshold=0.5,
)

WEIGHT_GRID = tuple(round(0.1 * i, 1) for i in range(11))  # 0.0..1.0
OK_GRID = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
THRESHOLD_GRID = tuple(round(0.3 + 0.05 * i, 2) for i in range(13))  # 0.30..0.90


@dataclass
class LabelledDataset:
    """Annotations plus the known role of every annotator."""

    collection: AnnotationCollection
    # annotator_id -> archetype (one of ARCHETYPES, or any name you like; names
    # in SPAM_ARCHETYPES count as spam unless `spammers` is given). Annotators
    # missing from this map count as honest when scoring.
    archetypes: dict[str, str]
    # Explicit spammer ids. Defaults to the annotators whose archetype is in
    # SPAM_ARCHETYPES.
    spammers: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if not self.spammers:
            self.spammers = {
                a for a, kind in self.archetypes.items() if kind in SPAM_ARCHETYPES
            }


# --- Simulation ---


def _dirichlet(rng: random.Random, alpha: float, k: int) -> list[float]:
    draws = [rng.gammavariate(alpha, 1.0) for _ in range(k)]
    total = sum(draws)
    return [d / total for d in draws]


def simulate_labelled_dataset(
    archetypes: list[str],
    num_items: int,
    labels: list[str],
    seed: int,
    gold_share: float = 0.0,
    skip_rate: float = 0.0,
    with_durations: bool = True,
    label_prior: list[float] | None = None,
    annotator_ids: list[str] | None = None,
    ratings_per_item: int | None = None,
) -> LabelledDataset:
    """
    Simulates one dataset with one annotator per entry in `archetypes`.

    Every annotator rates every item (or, with ratings_per_item, a random
    subset of that many annotators rates each item), except for pairs
    dropped at `skip_rate`.
    A `gold_share` of the items are gold: their annotations get is_gold=True
    and gold_label set to the true label. The same seed always gives the
    same data.
    """
    if len(labels) < 2:
        raise ValueError("Need at least 2 labels.")
    unknown = set(archetypes) - set(ARCHETYPES)
    if unknown:
        raise ValueError(f"Unknown archetypes: {sorted(unknown)}")
    if annotator_ids is None:
        annotator_ids = [f"annotator_{i:02d}" for i in range(len(archetypes))]
    if len(annotator_ids) != len(archetypes):
        raise ValueError("Need one annotator id per archetype.")
    if ratings_per_item is not None and not 1 <= ratings_per_item <= len(archetypes):
        raise ValueError(
            "ratings_per_item must be between 1 and the number of annotators."
        )

    rng = random.Random(seed)
    k = len(labels)
    chance = 1.0 / k
    prior = label_prior or [1.0 / k] * k

    item_ids = [f"item_{i:04d}" for i in range(num_items)]
    truth = {item: rng.choices(labels, weights=prior)[0] for item in item_ids}
    difficulty = {item: rng.uniform(0.0, 0.4) for item in item_ids}
    gold_items = set(rng.sample(item_ids, round(gold_share * num_items)))

    # Per-annotator traits.
    profiles = []
    for kind in archetypes:
        if kind == "honest_weak":
            skill = rng.uniform(0.15, 0.40)
        elif kind in HONEST_ARCHETYPES:
            skill = rng.uniform(0.50, 0.95)
        else:
            skill = 0.0

        if kind == "honest_fast":
            base_seconds = rng.uniform(1.5, 4.0)
        elif kind in SPAM_ARCHETYPES and rng.random() < 0.6:
            base_seconds = rng.uniform(0.4, 2.5)
        else:
            base_seconds = rng.uniform(6.0, 40.0)

        favourite = rng.choice(labels)
        favourite_share = rng.uniform(0.6, 0.9)
        profiles.append((kind, skill, base_seconds, favourite, favourite_share))

    everyone = list(range(len(archetypes)))
    annotations = []
    for item in item_ids:
        true_label = truth[item]
        is_gold = item in gold_items
        raters = (
            sorted(rng.sample(everyone, ratings_per_item))
            if ratings_per_item is not None
            else everyone
        )
        for index in raters:
            annotator_id = annotator_ids[index]
            kind, skill, base_seconds, fav, fav_share = profiles[index]
            if skip_rate and rng.random() < skip_rate:
                continue

            if kind == "constant_spammer":
                label = fav
            elif kind == "random_clicker":
                label = rng.choice(labels)
            elif kind == "biased_clicker":
                label = fav if rng.random() < fav_share else rng.choice(labels)
            else:
                p_correct = chance + skill * (1.0 - difficulty[item]) * (1.0 - chance)
                if rng.random() < p_correct:
                    label = true_label
                else:
                    label = rng.choice([lbl for lbl in labels if lbl != true_label])

            duration = (
                round(base_seconds * rng.lognormvariate(0.0, 0.35), 2)
                if with_durations
                else None
            )
            annotations.append(
                Annotation(
                    item_id=item,
                    annotator_id=annotator_id,
                    is_gold=is_gold,
                    label=label,
                    duration_sec=duration,
                    gold_label=true_label if is_gold else None,
                )
            )

    return LabelledDataset(
        collection=AnnotationCollection(annotations),
        archetypes=dict(zip(annotator_ids, archetypes, strict=True)),
    )


def random_labelled_dataset(seed: int) -> LabelledDataset:
    """
    A simulated dataset with a random shape, used for tuning and validation:
    dense (5-15 annotators rate everything, 40-250 items) or sparse (8-30
    annotators, 3-5 per item, 100-600 items), 2-5 labels, 0-30% spammers,
    0-20% gold items, sometimes skipped annotations, sometimes no durations,
    and a balanced or skewed label mix.
    """
    rng = random.Random(seed)
    sparse = rng.random() < 0.5
    num_annotators = rng.randint(8, 30) if sparse else rng.randint(5, 15)
    k = rng.randint(2, 5)
    labels = [f"L{i}" for i in range(k)]

    num_spam = round(rng.choice([0.0, 0.1, 0.2, 0.3]) * num_annotators)
    kinds = [rng.choice(SPAM_ARCHETYPES) for _ in range(num_spam)]
    for _ in range(num_annotators - num_spam):
        r = rng.random()
        kinds.append(
            "honest_fast" if r < 0.15 else "honest_weak" if r < 0.30 else "honest"
        )
    rng.shuffle(kinds)

    return simulate_labelled_dataset(
        archetypes=kinds,
        num_items=rng.randint(100, 600) if sparse else rng.randint(40, 250),
        ratings_per_item=rng.randint(3, 5) if sparse else None,
        labels=labels,
        seed=rng.randrange(2**31),
        gold_share=rng.choice([0.0, 0.0, 0.05, 0.1, 0.2]),
        skip_rate=rng.choice([0.0, 0.0, 0.1, 0.2]),
        with_durations=rng.random() < 0.8,
        label_prior=_dirichlet(rng, rng.choice([0.7, 1.5, 5.0]), k),
    )


# --- Evaluation ---


@dataclass
class DetectionMetrics:
    """How well a set of parameters separates known spammers from honest ones."""

    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    beta: float
    # archetype -> (flagged, total)
    by_archetype: dict[str, tuple[int, int]]

    @property
    def precision(self) -> float:
        flagged = self.true_positives + self.false_positives
        return self.true_positives / flagged if flagged else 1.0

    @property
    def recall(self) -> float:
        spam = self.true_positives + self.false_negatives
        return self.true_positives / spam if spam else 1.0

    @property
    def false_positive_rate(self) -> float:
        honest = self.false_positives + self.true_negatives
        return self.false_positives / honest if honest else 0.0

    @property
    def f_beta(self) -> float:
        """F-beta: beta < 1 weighs precision more, beta > 1 weighs recall more."""
        p, r, b2 = self.precision, self.recall, self.beta**2
        return (1 + b2) * p * r / (b2 * p + r) if p + r else 0.0


@dataclass
class _Row:
    """One annotator, reduced to what the combiner needs, computed once."""

    signals: AnnotatorSignals
    values: dict[str, float]
    is_spammer: bool
    archetype: str


def _prepare(datasets: list[LabelledDataset], params: CombinerParams) -> list[_Row]:
    # Which signals are available depends only on the min_* settings, which the
    # search does not change, so this is done once.
    rows = []
    for ds in datasets:
        for annotator_id, sig in compute_signals(ds.collection).items():
            rows.append(
                _Row(
                    signals=sig,
                    values=_available_values(sig, params),
                    is_spammer=annotator_id in ds.spammers,
                    archetype=ds.archetypes.get(annotator_id, "unknown"),
                )
            )
    return rows


def _evaluate_rows(
    params: CombinerParams, rows: list[_Row], beta: float
) -> DetectionMetrics:
    tp = fp = fn = tn = 0
    by_archetype: dict[str, list[int]] = {}
    for row in rows:
        flagged = (
            _has_enough_data(row.signals, row.values, params)
            and _score(row.values, params)[3]
        )
        counts = by_archetype.setdefault(row.archetype, [0, 0])
        counts[0] += flagged
        counts[1] += 1
        if row.is_spammer:
            tp += flagged
            fn += not flagged
        else:
            fp += flagged
            tn += not flagged
    return DetectionMetrics(
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
        beta=beta,
        by_archetype={k: (v[0], v[1]) for k, v in sorted(by_archetype.items())},
    )


def evaluate(
    params: CombinerParams, datasets: list[LabelledDataset], beta: float = 0.5
) -> DetectionMetrics:
    """Runs the combiner on labelled datasets and counts hits and misses."""
    return _evaluate_rows(params, _prepare(datasets, params), beta)


# --- Tuning ---


@dataclass
class TuningResult:
    params: CombinerParams
    train_metrics: DetectionMetrics
    rounds: int


def _blocks(
    weight_grid: tuple[float, ...],
    ok_grid: tuple[float, ...],
    threshold_grid: tuple[float, ...],
) -> list[list[dict[str, Any]]]:
    """
    The search moves: for each signal, every (weight, ok) pair at once, and
    then every threshold. A signal's weight and ok point work together (a
    higher weight is only useful with the right ramp), so they are searched
    jointly; searching them one at a time gets stuck.
    """
    blocks: list[list[dict[str, Any]]] = [
        [
            {f"{signal}_weight": w, f"{signal}_ok": ok}
            for w in weight_grid
            for ok in ok_grid
        ]
        for signal in SIGNALS
    ]
    blocks.append([{"threshold": t} for t in threshold_grid])
    return blocks


def tune_combiner(
    datasets: list[LabelledDataset],
    beta: float = 0.5,
    start: CombinerParams = START_PARAMS,
    weight_grid: tuple[float, ...] = WEIGHT_GRID,
    ok_grid: tuple[float, ...] = OK_GRID,
    threshold_grid: tuple[float, ...] = THRESHOLD_GRID,
    max_rounds: int = 20,
) -> TuningResult:
    """
    Fits CombinerParams to labelled data by block coordinate descent.

    Each round goes through the blocks in turn: one signal's (weight, ok)
    pair, searched jointly over the grids, and then the threshold. The
    min_* evidence settings are policy choices and are kept from `start`. Within a
    block it keeps the best move if it strictly improves F-beta. It stops when
    a whole round changes nothing. This is deterministic, and fast because
    the signals are computed only once.

    beta=0.5 weighs precision twice as much as recall: wrongly flagging an
    honest annotator is treated as worse than missing a spammer.
    """
    rows = _prepare(datasets, start)
    best = start
    best_score = _evaluate_rows(best, rows, beta).f_beta

    rounds = 0
    for rounds in range(1, max_rounds + 1):
        changed = False
        for block in _blocks(weight_grid, ok_grid, threshold_grid):
            for move in block:
                candidate = replace(best, **move)
                score = _evaluate_rows(candidate, rows, beta).f_beta
                if score > best_score + 1e-12:
                    best, best_score, changed = candidate, score, True
        if not changed:
            break

    # A calibration belongs to the params it was fitted for; new params need a
    # new one (fit_confidence_calibration).
    best = replace(best, calibration=None)
    return TuningResult(
        params=best,
        train_metrics=_evaluate_rows(best, rows, beta),
        rounds=rounds,
    )


# --- Confidence calibration ---


def _flagged_scores(
    datasets: list[LabelledDataset], params: CombinerParams
) -> tuple[list[tuple[float, bool]], int, int]:
    """(score, is_spammer) for every flagged annotator, plus spam and total counts."""
    flagged = []
    spam = total = 0
    for row in _prepare(datasets, params):
        total += 1
        spam += row.is_spammer
        if _has_enough_data(row.signals, row.values, params):
            score, _, _, is_flagged = _score(row.values, params)
            if is_flagged:
                flagged.append((score, row.is_spammer))
    return flagged, spam, total


def _calibrate(
    flagged: list[tuple[float, bool]], spam: int, total: int, min_group: int
) -> ConfidenceCalibration:
    """The calibration arithmetic on (score, is_spammer) pairs; see below."""
    if len(flagged) < min_group:
        raise ValueError(
            f"Need at least {min_group} flagged annotators to calibrate, "
            f"got {len(flagged)}."
        )
    if not 0 < spam < total:
        raise ValueError("Calibration data needs both spammers and honest annotators.")

    flagged = sorted(flagged)
    # [lowest score, spammers, annotators] per group.
    groups: list[list[float]] = []
    start = 0
    while start < len(flagged):
        end = min(start + min_group, len(flagged))
        # Don't split equal scores across groups.
        while end < len(flagged) and flagged[end][0] == flagged[end - 1][0]:
            end += 1
        chunk = flagged[start:end]
        spammers = sum(is_spam for _, is_spam in chunk)
        if groups and len(chunk) < min_group:
            groups[-1][1] += spammers
            groups[-1][2] += len(chunk)
        else:
            groups.append([chunk[0][0], spammers, len(chunk)])
        start = end

    pooled: list[list[float]] = []
    for group in groups:
        pooled.append(group)
        while len(pooled) >= 2 and (
            pooled[-2][1] / pooled[-2][2] >= pooled[-1][1] / pooled[-1][2]
        ):
            last = pooled.pop()
            pooled[-1][1] += last[1]
            pooled[-1][2] += last[2]

    return ConfidenceCalibration(
        steps=tuple((g[0], (g[1] + 0.5) / (g[2] + 1.0)) for g in pooled),
        base_rate=spam / total,
    )


def fit_confidence_calibration(
    datasets: list[LabelledDataset],
    params: CombinerParams,
    min_group: int = MIN_CALIBRATION_GROUP,
) -> ConfidenceCalibration:
    """
    Fits the map from a flagged annotator's score to the chance they are a
    spammer. Use datasets that were not used to tune `params`.

    1. Sort the flagged annotators by score and cut them into groups of at
       least `min_group`. Equal scores always stay in the same group.
    2. Merge neighbouring groups while a higher-scoring group's spam rate is
       not above the lower one's (pool adjacent violators), so confidence
       never falls as the score rises and equal rates share one step.
    3. Smooth each group's rate with a Jeffreys prior:
       (spammers + 0.5) / (annotators + 1). It is never exactly 0% or 100%.

    base_rate is the share of spammers among all annotators in `datasets`.
    """
    flagged, spam, total = _flagged_scores(datasets, params)
    return _calibrate(flagged, spam, total, min_group)


@dataclass
class CalibrationCheck:
    """Predicted confidence next to what really happened, per calibration step."""

    # (lowest score, predicted confidence, observed share of spammers, flagged)
    steps: list[tuple[float, float, float, int]]
    # Mean squared gap between confidence and the 0/1 truth (lower is better).
    brier_score: float


def check_calibration(
    datasets: list[LabelledDataset], params: CombinerParams
) -> CalibrationCheck:
    """Compares params.calibration with held-out labelled data."""
    if params.calibration is None:
        raise ValueError("params has no calibration to check")
    calibration = params.calibration
    flagged, _, _ = _flagged_scores(datasets, params)

    per_step: dict[float, list[int]] = {s: [0, 0] for s, _ in calibration.steps}
    squared_error = 0.0
    for score, is_spam in flagged:
        squared_error += (calibration.confidence(score) - is_spam) ** 2
        step = max(
            (s for s, _ in calibration.steps if score >= s),
            default=calibration.steps[0][0],
        )
        per_step[step][0] += is_spam
        per_step[step][1] += 1

    return CalibrationCheck(
        steps=[
            (s, prob, per_step[s][0] / per_step[s][1], per_step[s][1])
            for s, prob in calibration.steps
            if per_step[s][1]
        ],
        brier_score=squared_error / len(flagged) if flagged else 0.0,
    )


def _format_metrics(title: str, m: DetectionMetrics) -> list[str]:
    summary = (
        f"{title}: precision {m.precision:.3f}, recall {m.recall:.3f}, "
        f"F{m.beta:g} {m.f_beta:.3f}, false positive rate {m.false_positive_rate:.3f}"
    )
    lines = [summary]
    for kind, (flagged, total) in m.by_archetype.items():
        lines.append(
            f"  {kind:<17} flagged {flagged:>4} / {total:<4} ({flagged / total:.1%})"
        )
    return lines


def main() -> None:
    train = [random_labelled_dataset(s) for s in TRAIN_SEEDS]
    validation = [random_labelled_dataset(s) for s in VALIDATION_SEEDS]

    result = tune_combiner(train)
    print(f"Tuned in {result.rounds} rounds:")
    print(result.params)
    for line in _format_metrics("Train", result.train_metrics):
        print(line)
    for line in _format_metrics("Validation", evaluate(result.params, validation)):
        print(line)

    calibration_sets = [random_labelled_dataset(s) for s in CALIBRATION_SEEDS]
    calibration = fit_confidence_calibration(calibration_sets, result.params)
    print()
    print("Confidence calibration (rounded as in DEFAULT_PARAMS):")
    steps = tuple((round(low, 4), round(prob, 4)) for low, prob in calibration.steps)
    print(
        f"ConfidenceCalibration(steps={steps}, base_rate={calibration.base_rate:.4f})"
    )
    check = check_calibration(
        validation, replace(result.params, calibration=calibration)
    )
    print(f"Validation Brier score {check.brier_score:.4f}")
    for low, predicted, observed, n in check.steps:
        print(
            f"  score >= {low:.3f}: confidence {predicted:.1%}, "
            f"really spam {observed:.1%} of {n}"
        )


if __name__ == "__main__":
    main()
