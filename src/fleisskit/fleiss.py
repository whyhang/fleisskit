import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass

from fleisskit.models import AnnotationCollection


@dataclass
class ConfidenceInterval:
    """
    Represents a confidence interval for an estimate.
    """

    lower_bound: float
    upper_bound: float
    method: str
    alpha: float
    skipped_resamples: int = 0


@dataclass
class KappaResult:
    """Holds a kappa estimate and its optional confidence interval."""

    estimate: float
    ci: ConfidenceInterval | None = None


@dataclass
class FleissReport:
    """Comprehensive result object for overall and per-category Fleiss' kappa."""

    overall_kappa: KappaResult
    per_category_kappa: dict[str, KappaResult]


def build_count_table(
    collection: AnnotationCollection,
) -> tuple[list[list[int]], list[str]]:
    """
    Builds a count table for Fleiss' kappa and related metrics.

    Returns a tuple of (count_table, label_list).
    - count_table: A 2D matrix where each row represents an item, and each column
      represents a label. Each cell contains n_ij (how many annotators gave
      that item that label).
    - label_list: The sorted list of unique labels corresponding to the columns.

    Raises ValueError if items have different numbers of ratings, as Fleiss'
    kappa requires a constant number of ratings per item.
    """
    item_labels: dict[str, list[str]] = defaultdict(list)
    unique_labels = set()

    for ann in collection.annotations:
        if ann.label is not None:
            item_labels[ann.item_id].append(ann.label)
            unique_labels.add(ann.label)

    if not item_labels:
        raise ValueError("No annotations with labels found.")

    labels_list = sorted(unique_labels)
    table = []
    expected_n = None

    # Sort keys for perfectly deterministic row ordering
    for item_id in sorted(item_labels.keys()):
        labels = item_labels[item_id]
        n_ratings = len(labels)

        if expected_n is None:
            expected_n = n_ratings
        elif n_ratings != expected_n:
            raise ValueError(
                f"Varying number of ratings per item. Found {n_ratings}, expected {expected_n}."
            )

        counts = Counter(labels)
        row = [counts[lbl] for lbl in labels_list]
        table.append(row)

    if expected_n is not None and expected_n < 2:
        raise ValueError("Fleiss' kappa requires at least 2 ratings per item.")

    return table, labels_list


def _fleiss_kappa_components(table: list[list[int]]) -> tuple[float, float, float]:
    """Internal helper to calculate kappa, P_bar, and P_bar_e from a count table."""
    N = len(table)
    k = len(table[0])
    n = sum(table[0])  # Total ratings per item

    # p_j: Each label's overall share
    p_j = [sum(row[j] for row in table) / (N * n) for j in range(k)]

    # P_i: Agreement on each item
    P_i = [(sum(x**2 for x in row) - n) / (n * (n - 1)) for row in table]

    # P_bar: Average agreement
    P_bar = sum(P_i) / N

    # P_bar_e: Chance agreement
    P_bar_e = sum(p**2 for p in p_j)

    if P_bar_e == 1.0:
        raise ValueError(
            "Chance agreement (P_bar_e) is 1.0 (division by zero). Data lacks variation."
        )

    # Kappa
    kappa = (P_bar - P_bar_e) / (1.0 - P_bar_e)

    return kappa, P_bar, P_bar_e


def fleiss_kappa(collection: AnnotationCollection) -> float:
    """
    Calculates Fleiss' kappa for multi-annotator agreement.

    Items must have the exact same number of ratings. If they vary,
    a ValueError is raised.
    """
    table, _ = build_count_table(collection)
    kappa, _, _ = _fleiss_kappa_components(table)
    return kappa


def _per_category_kappa_components(
    table: list[list[int]], labels: list[str]
) -> dict[str, float]:
    """
    Internal helper to calculate per-category kappa from a count table.

    For each label j:
        p_j = (total times j was used) / (N * n)
        q_j = 1 - p_j
        k_j = 1 - sum_i(n_ij * (n - n_ij)) / (N * n * (n - 1) * p_j * q_j)

    Labels with p_j == 0 (never used) or p_j == 1 (used for every rating) are
    skipped, because p_j * q_j == 0 and k_j is undefined.
    """
    N = len(table)
    n = sum(table[0])  # Total ratings per item
    total_ratings = N * n

    result: dict[str, float] = {}
    for j, label in enumerate(labels):
        # Integer count, so the p_j == 0 / p_j == 1 checks below are exact.
        label_total = sum(row[j] for row in table)
        if label_total == 0 or label_total == total_ratings:
            continue

        p_j = label_total / total_ratings
        q_j = 1.0 - p_j

        disagreement = sum(row[j] * (n - row[j]) for row in table)
        result[label] = 1.0 - disagreement / (N * n * (n - 1) * p_j * q_j)

    return result


def per_category_kappa(collection: AnnotationCollection) -> dict[str, float]:
    """
    Calculates Fleiss' kappa separately for each label (category).

    Shows which labels annotators agree on and which ones they struggle with.

    Returns a dict mapping label -> kappa_j. Labels where p_j is 0 or 1
    are left out, since kappa_j is undefined for them.

    Items must have the exact same number of ratings. If they vary,
    a ValueError is raised.
    """
    table, labels = build_count_table(collection)
    return _per_category_kappa_components(table, labels)


def _calculate_formula_ci(
    table: list[list[int]], estimate: float, alpha: float
) -> ConfidenceInterval:
    """
    Calculates the confidence interval using the analytical standard error formula.

    WARNING: This uses the standard error derived by Fleiss (1971) assuming the true
    kappa is strictly 0. It is meant for testing if agreement is better than chance,
    not for building accurate intervals around non-zero estimates.
    """
    N = len(table)
    k = len(table[0])
    n = sum(table[0])

    p_j = [sum(row[j] for row in table) / (N * n) for j in range(k)]
    sum_pj2 = sum(p**2 for p in p_j)
    sum_pj3 = sum(p**3 for p in p_j)

    numerator = sum_pj2 - (2 * n - 3) * (sum_pj2**2) + 2 * (n - 2) * sum_pj3
    denominator = N * n * (n - 1) * ((1.0 - sum_pj2) ** 2)
    var_k0 = (2.0 / denominator) * numerator

    se = math.sqrt(var_k0)
    z = 1.95996 if math.isclose(alpha, 0.05) else 1.96

    return ConfidenceInterval(
        lower_bound=estimate - (z * se),
        upper_bound=estimate + (z * se),
        method="formula_se_k0",
        alpha=alpha,
    )


def _calculate_bootstrap_ci(
    table: list[list[int]], labels: list[str], repeats: int, seed: int, alpha: float
) -> tuple[ConfidenceInterval | None, dict[str, ConfidenceInterval | None]]:
    """
    Calculates empirical confidence intervals using the bootstrap method.
    Resamples whole items (rows) with replacement.
    """
    rng = random.Random(seed)
    N = len(table)

    overall_kappas = []
    cat_kappas: dict[str, list[float]] = {lbl: [] for lbl in labels}

    overall_skips = 0
    cat_skips: dict[str, int] = {lbl: 0 for lbl in labels}

    for _ in range(repeats):
        resample = rng.choices(table, k=N)

        try:
            k_est, _, _ = _fleiss_kappa_components(resample)
            overall_kappas.append(k_est)
        except ValueError:
            overall_skips += 1

        c_ests = _per_category_kappa_components(resample, labels)
        for lbl in labels:
            if lbl in c_ests:
                cat_kappas[lbl].append(c_ests[lbl])
            else:
                cat_skips[lbl] += 1

    def get_ci(vals: list[float], skips: int) -> ConfidenceInterval | None:
        if not vals:
            return None
        vals.sort()
        lower_idx = int(len(vals) * (alpha / 2.0))
        upper_idx = min(int(len(vals) * (1.0 - (alpha / 2.0))), len(vals) - 1)

        return ConfidenceInterval(
            lower_bound=vals[lower_idx],
            upper_bound=vals[upper_idx],
            method="bootstrap",
            alpha=alpha,
            skipped_resamples=skips,
        )

    overall_ci = get_ci(overall_kappas, overall_skips)
    per_cat_cis = {lbl: get_ci(cat_kappas[lbl], cat_skips[lbl]) for lbl in labels}

    return overall_ci, per_cat_cis


def analyze_fleiss(
    collection: AnnotationCollection,
    calc_ci: bool = False,
    ci_method: str = "bootstrap",
    ci_repeats: int = 1000,
    ci_seed: int = 42,
    ci_alpha: float = 0.05,
) -> FleissReport:
    """
    Calculates overall Fleiss' kappa and per-category kappas, optionally with
    confidence intervals.
    """
    table, labels = build_count_table(collection)

    overall_est, _, _ = _fleiss_kappa_components(table)
    cat_ests = _per_category_kappa_components(table, labels)

    overall_ci = None
    cat_cis: dict[str, ConfidenceInterval | None] = {lbl: None for lbl in labels}

    if calc_ci:
        if ci_method == "bootstrap":
            overall_ci, cat_cis = _calculate_bootstrap_ci(
                table, labels, ci_repeats, ci_seed, ci_alpha
            )
        elif ci_method == "formula":
            overall_ci = _calculate_formula_ci(table, overall_est, ci_alpha)
        else:
            raise ValueError(f"Unknown CI method: {ci_method}")

    report = FleissReport(
        overall_kappa=KappaResult(estimate=overall_est, ci=overall_ci),
        per_category_kappa={},
    )

    for lbl, est in cat_ests.items():
        report.per_category_kappa[lbl] = KappaResult(estimate=est, ci=cat_cis.get(lbl))

    return report
