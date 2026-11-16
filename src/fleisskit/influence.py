"""
Leave-one-annotator-out (LOAO) analysis for Fleiss' kappa.

For every annotator a, Fleiss' kappa is recomputed on the same items with a's
ratings removed:

    delta_a = kappa_without_a - kappa_full

- delta_a > 0: the data agrees better without a, so a pulls kappa down
  (a possible outlier).
- delta_a < 0: a agrees with the others more than average, so a holds kappa up.

Each "without a" count table is the full table minus a's one-hot rating in
every row. It only has a constant number of ratings per item (which Fleiss'
kappa needs) if a rated every item. So LOAO is only defined for annotators
who rated every item, and when at least 3 ratings per item remain before
removal.

Outliers are flagged with a modified z-score (Iglewicz & Hoaglin) over the
deltas, which uses the median and the median absolute deviation (MAD) instead
of the mean and standard deviation:

    MAD   = median(|delta_a - median(delta)|)
    score = 0.6745 * (delta_a - median(delta)) / MAD

0.6745 is the 0.75 quantile of the standard normal, so for normally spread
deltas the score reads like a standard z-score. Median and MAD are used
because the outliers we are looking for would themselves drag a mean and a
standard deviation towards themselves and hide the gap. An annotator is
flagged when their score is above outlier_threshold (3.5 by default).

If the MAD is 0 (more than half the deltas are identical), the score would
divide by zero. In that case nobody is scored or flagged, and every annotator
with a delta gets the verdict "not proven".

With calc_ci=True, each delta gets a paired percentile bootstrap interval.
Items are resampled with replacement, and kappa_full and every kappa_without_a
are computed on the same resample. This way the interval shows how much delta_a
would move under item sampling alone.

By default the intervals are Bonferroni-adjusted: with m annotators tested, each
interval uses alpha / m. Then the chance that any annotator is flagged when
all annotators are equally good stays at about alpha, rather than growing with
the number of annotators.
"""

import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from operator import mul

from fleisskit.fleiss import (
    ConfidenceInterval,
    _build_count_table_with_items,
    _fleiss_kappa_components,
    _percentile_ci,
)
from fleisskit.models import AnnotationCollection

# 0.75 quantile of the standard normal: scales the MAD to be comparable with a
# standard deviation, so the score reads like a z-score.
_MAD_SCALE = 0.6745
_DEFAULT_OUTLIER_THRESHOLD = 3.5


@dataclass
class AnnotatorInfluence:
    """Effect of removing a single annotator on Fleiss' kappa."""

    annotator_id: str
    items_rated: int
    kappa_without: float | None = None
    # kappa_without - kappa_full
    delta: float | None = None
    # Observed (P_bar) and chance (P_bar_e) agreement without this annotator.
    p_bar_without: float | None = None
    p_bar_e_without: float | None = None
    delta_ci: ConfidenceInterval | None = None
    # Modified z-score of delta against the other annotators' deltas. None when
    # delta is undefined, or when the MAD is 0 (see outlier_status).
    robust_score: float | None = None
    # True when robust_score is above the report's outlier_threshold.
    is_outlier: bool = False
    # Why kappa_without is undefined, if it is.
    error: str | None = None

    @property
    def outlier_status(self) -> str | None:
        """
        Outlier verdict for this annotator:
        - "outlier": score above the threshold
        - "not_flagged": scored, and at or below the threshold
        - "not_proven": has a delta but no score, because the MAD was 0
        - None: delta is undefined, so there is nothing to judge
        """
        if self.delta is None:
            return None
        if self.robust_score is None:
            return "not_proven"
        return "outlier" if self.is_outlier else "not_flagged"

    @property
    def effect(self) -> str | None:
        """
        Reads the delta confidence interval:
        - "lowers_kappa": whole CI above 0 (removing this annotator raises kappa)
        - "raises_kappa": whole CI below 0 (removing this annotator lowers kappa)
        - "no_clear_effect": CI contains 0
        - None: no CI was computed (or none could be)
        """
        if self.delta_ci is None:
            return None
        if self.delta_ci.lower_bound > 0:
            return "lowers_kappa"
        if self.delta_ci.upper_bound < 0:
            return "raises_kappa"
        return "no_clear_effect"


@dataclass
class LeaveOneOutReport:
    """Full-data Fleiss' kappa plus one LOAO entry per annotator."""

    full_kappa: float
    full_p_bar: float
    full_p_bar_e: float
    # Sorted by delta, largest first (the annotators pulling kappa down most come
    # first). Undefined entries come last. Ties are broken by annotator_id.
    annotators: list[AnnotatorInfluence]
    # Median and spread of the deltas that the robust scores are measured
    # against, and the threshold a score must pass to be flagged.
    delta_median: float | None = None
    delta_mad: float | None = None
    outlier_threshold: float = _DEFAULT_OUTLIER_THRESHOLD
    # True when the MAD was 0: no scores, nobody flagged, verdict "not proven".
    mad_is_zero: bool = False
    # Error rate for the whole set of intervals (the ci_alpha passed in), and
    # how many intervals it was split across (1 when not Bonferroni-adjusted).
    # Each interval's own alpha is ci_family_alpha / ci_num_comparisons.
    # Both are None when no CIs were computed.
    ci_family_alpha: float | None = None
    ci_num_comparisons: int | None = None


class _WeightedKappa:
    """
    Fast Fleiss' kappa for a fixed count table under item weights.

    A bootstrap resample is the same as giving item i a weight w_i (how many
    times it was drawn, with sum(w) == N). Each item's agreement P_i and its
    column counts are worked out once, so each resample only needs weighted
    sums. The formula is the same as _fleiss_kappa_components.
    """

    def __init__(self, table: list[list[int]]):
        self.N = len(table)
        self.n = sum(table[0])
        n = self.n
        self.agreement = [
            (sum(x * x for x in row) - n) / (n * (n - 1)) for row in table
        ]
        self.columns = [list(col) for col in zip(*table, strict=True)]

    def kappa(self, weights: list[int]) -> float | None:
        """Kappa for the weighted table, or None if chance agreement is 1."""
        total = self.N * self.n
        col_sums = [sum(map(mul, weights, col)) for col in self.columns]
        # Integer check: one label got every rating, so P_bar_e == 1.
        if max(col_sums) == total:
            return None
        p_bar = sum(map(mul, weights, self.agreement)) / self.N
        p_bar_e = sum((c / total) ** 2 for c in col_sums)
        return (p_bar - p_bar_e) / (1.0 - p_bar_e)


def _robust_scores(deltas: list[float]) -> tuple[list[float] | None, float, float]:
    """
    Modified z-scores for the deltas, plus the median and MAD they use.

    Scores are None when the MAD is 0: more than half the deltas are identical,
    so the score would divide by zero and nobody can be judged an outlier.
    """
    median = statistics.median(deltas)
    mad = statistics.median([abs(d - median) for d in deltas])
    if mad == 0:
        return None, median, mad
    return [_MAD_SCALE * (d - median) / mad for d in deltas], median, mad


def _bootstrap_delta_cis(
    full_table: list[list[int]],
    loo_tables: dict[str, list[list[int]]],
    repeats: int,
    seed: int,
    alpha: float,
) -> dict[str, ConfidenceInterval | None]:
    """
    Paired percentile bootstrap for every delta_a.

    Each resample draws N items with replacement. kappa_full and each
    kappa_without_a are computed on the same draw, and the difference is
    stored. A resample is skipped for an annotator if either kappa is
    undefined on it.
    """
    rng = random.Random(seed)
    N = len(full_table)
    full = _WeightedKappa(full_table)
    loo = {a: _WeightedKappa(t) for a, t in loo_tables.items()}

    deltas: dict[str, list[float]] = {a: [] for a in loo}
    skips: dict[str, int] = {a: 0 for a in loo}
    row_indices = range(N)

    for _ in range(repeats):
        weights = [0] * N
        for i in rng.choices(row_indices, k=N):
            weights[i] += 1

        k_full = full.kappa(weights)
        for a, wk in loo.items():
            k_without = wk.kappa(weights)
            if k_full is None or k_without is None:
                skips[a] += 1
            else:
                deltas[a].append(k_without - k_full)

    return {a: _percentile_ci(deltas[a], skips[a], alpha) for a in loo}


def leave_one_annotator_out(
    collection: AnnotationCollection,
    calc_ci: bool = False,
    ci_repeats: int = 1000,
    ci_seed: int = 42,
    ci_alpha: float = 0.05,
    ci_bonferroni: bool = True,
    outlier_threshold: float = _DEFAULT_OUTLIER_THRESHOLD,
) -> LeaveOneOutReport:
    """
    Recomputes Fleiss' kappa with each annotator left out in turn.

    Only annotators with at least one non-missing label are included, because
    the others do not affect kappa. An annotator's entry gets `error` set
    (instead of a kappa) if kappa without them is undefined. That happens if:
    - they did not rate every item (the item counts would differ),
    - fewer than 2 ratings per item would remain, or
    - the other annotators all used the same label (P_bar_e == 1).

    Every annotator with a defined delta gets a robust_score (a modified
    z-score of their delta against the other annotators' deltas) and is_outlier
    is set when that score is above outlier_threshold. If the MAD of the deltas
    is 0, nobody gets a score or a flag, and their outlier_status is
    "not_proven". The score needs the
    other annotators as a reference, so it is only meaningful with enough of
    them: with 3 or 4 annotators the MAD rests on one or two deviations.

    With calc_ci=True, every defined delta gets a paired bootstrap CI.
    ci_alpha is the error rate for the whole set of intervals. If ci_bonferroni
    is True (the default), each interval uses ci_alpha / m, where m is the number
    of annotators with a defined delta.

    Raises ValueError if Fleiss' kappa is undefined on the full data, since
    there is then no baseline to compare against.
    """
    if calc_ci:
        if ci_repeats < 1:
            raise ValueError(f"ci_repeats must be at least 1, got {ci_repeats}")
        if not 0.0 < ci_alpha < 1.0:
            raise ValueError(f"ci_alpha must be between 0 and 1, got {ci_alpha}")

    table, labels, item_ids = _build_count_table_with_items(collection)
    full_kappa, full_p_bar, full_p_bar_e = _fleiss_kappa_components(table)

    N = len(table)
    n = sum(table[0])
    row_of = {item_id: i for i, item_id in enumerate(item_ids)}
    col_of = {label: j for j, label in enumerate(labels)}

    # annotator -> the (row, column) cells their ratings add to
    cells_by_annotator: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for ann in collection.annotations:
        if ann.label is not None:
            cells_by_annotator[ann.annotator_id].append(
                (row_of[ann.item_id], col_of[ann.label])
            )

    results: list[AnnotatorInfluence] = []
    loo_tables: dict[str, list[list[int]]] = {}

    for annotator_id in sorted(cells_by_annotator):
        cells = cells_by_annotator[annotator_id]
        items_rated = len({row for row, _ in cells})
        info = AnnotatorInfluence(annotator_id=annotator_id, items_rated=items_rated)
        results.append(info)

        loo_table = [row.copy() for row in table]
        for row, col in cells:
            loo_table[row][col] -= 1

        if any(sum(row) != n - 1 for row in loo_table):
            info.error = (
                f"Rated {items_rated} of {N} items; without them, items have "
                "different numbers of ratings."
            )
            continue
        if n - 1 < 2:
            info.error = (
                "Fleiss' kappa requires at least 2 ratings per item; "
                f"only {n - 1} would remain."
            )
            continue

        try:
            kappa, p_bar, p_bar_e = _fleiss_kappa_components(loo_table)
        except ValueError as e:
            info.error = str(e)
            continue

        info.kappa_without = kappa
        info.delta = kappa - full_kappa
        info.p_bar_without = p_bar
        info.p_bar_e_without = p_bar_e
        loo_tables[annotator_id] = loo_table

    with_delta = [info for info in results if info.delta is not None]
    delta_median = delta_mad = None
    mad_is_zero = False
    if with_delta:
        scores, delta_median, delta_mad = _robust_scores(
            [info.delta for info in with_delta if info.delta is not None]
        )
        # MAD of 0: leave every score None and flag nobody ("not proven").
        mad_is_zero = scores is None
        if scores is not None:
            for info, score in zip(with_delta, scores, strict=True):
                info.robust_score = score
                info.is_outlier = score > outlier_threshold

    ci_num_comparisons = None
    if calc_ci and loo_tables:
        ci_num_comparisons = len(loo_tables) if ci_bonferroni else 1
        cis = _bootstrap_delta_cis(
            table,
            loo_tables,
            ci_repeats,
            ci_seed,
            ci_alpha / ci_num_comparisons,
        )
        for info in results:
            if info.annotator_id in cis:
                info.delta_ci = cis[info.annotator_id]

    results.sort(
        key=lambda r: (
            r.delta is None,
            -r.delta if r.delta is not None else 0.0,
            r.annotator_id,
        )
    )

    return LeaveOneOutReport(
        full_kappa=full_kappa,
        full_p_bar=full_p_bar,
        full_p_bar_e=full_p_bar_e,
        annotators=results,
        delta_median=delta_median,
        delta_mad=delta_mad,
        outlier_threshold=outlier_threshold,
        mad_is_zero=mad_is_zero,
        ci_family_alpha=ci_alpha if ci_num_comparisons is not None else None,
        ci_num_comparisons=ci_num_comparisons,
    )
