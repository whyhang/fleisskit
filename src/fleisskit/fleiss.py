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
