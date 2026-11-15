from collections import Counter, defaultdict

from fleisskit.models import AnnotationCollection


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
