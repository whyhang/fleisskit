from collections import Counter, defaultdict
from itertools import combinations

from fleisskit.models import AnnotationCollection


def percent_agreement(collection: AnnotationCollection) -> float:
    """
    Calculates the average percent agreement across items.
    For each item with at least 2 labels, this calculates the fraction of
    annotator pairs that agree, then averages those fractions across all items.
    Edge case scenario:
    - Items with fewer than 2 labels are skipped (no pairs can be formed).
    - If nothing has 2 labels, a ValueError is raised.
    """
    # Group labels by item
    item_labels: dict[str, list[str]] = defaultdict(list)
    for ann in collection.annotations:
        if ann.label is not None:
            item_labels[ann.item_id].append(ann.label)

    item_agreements: list[float] = []

    for labels in item_labels.values():
        if len(labels) < 2:
            continue

        # Generate all unique pairs of labels for this item
        pairs = list(combinations(labels, 2))

        # Count how many pairs agree
        agree_count = sum(1 for l1, l2 in pairs if l1 == l2)

        item_agreements.append(agree_count / len(pairs))

    if not item_agreements:
        raise ValueError(
            "No items found with at least 2 labels to calculate percent agreement."
        )

    return sum(item_agreements) / len(item_agreements)


def cohens_kappa(
    collection: AnnotationCollection, annotator1_id: str, annotator2_id: str
) -> float:
    """
    Calculates Cohen's kappa for two specific annotators.
    Measures agreement beyond chance between two annotators, using only items
    that both of them labeled.
    Edge case handling:
    - If they share no items, raises a ValueError.
    - If chance agreement (p_e) is 1.0 (e.g., both annotators only ever used a single,
      identical label), raises a ValueError because the formula divides by zero.
    """
    # Extract labels for each annotator mapping item_id -> label
    a1_labels = {
        ann.item_id: ann.label
        for ann in collection.annotations
        if ann.annotator_id == annotator1_id and ann.label is not None
    }
    a2_labels = {
        ann.item_id: ann.label
        for ann in collection.annotations
        if ann.annotator_id == annotator2_id and ann.label is not None
    }

    # Get shared items as a set
    shared_items: set[str] = set(a1_labels.keys()).intersection(set(a2_labels.keys()))

    # Edge case handling
    if not shared_items:
        raise ValueError(
            f"Annotators '{annotator1_id}' and '{annotator2_id}' have no shared items."
        )

    len_shared = len(shared_items)
    agree_count = 0

    # Track label frequencies for chance agreement calculation
    a1_counts: dict[str, int] = defaultdict(int)
    a2_counts: dict[str, int] = defaultdict(int)

    for item_id in shared_items:
        label1 = a1_labels[item_id]
        label2 = a2_labels[item_id]

        if label1 == label2:
            agree_count += 1

        a1_counts[label1] += 1
        a2_counts[label2] += 1

    # 1. Observed agreement (p_o)
    agreement_observed = agree_count / len_shared
    # 2. Chance agreement (p_e)
    all_unique_labels = set(a1_counts.keys()).union(set(a2_counts.keys()))
    agreement_chance = 0.0

    for label in all_unique_labels:
        share1 = a1_counts[label] / len_shared
        share2 = a2_counts[label] / len_shared
        agreement_chance += share1 * share2

    # 3. Kappa calculation
    if agreement_chance == 1.0:
        raise ValueError(
            "Chance agreement (p_e) is 1.0 (division by zero). Data lacks variation."
        )

    return (agreement_observed - agreement_chance) / (1.0 - agreement_chance)


def majority_vote(collection: AnnotationCollection) -> dict[str, str]:
    """
    Calculates the majority vote label for each item.

    For each item, returns the label chosen by the most annotators.

    Tie-breaking rule:
    If there is a tie (e.g., label 'B' has 2 votes and label 'A' has 2 votes),
    the function chooses the alphabetically first label (so it returns 'A').

    Returns:
        A dictionary mapping item_id to the chosen majority label.
    """
    # Group labels by item
    item_labels: dict[str, list[str]] = defaultdict(list)
    for ann in collection.annotations:
        if ann.label is not None:
            item_labels[ann.item_id].append(ann.label)

    results: dict[str, str] = {}

    for item_id, labels in item_labels.items():
        if not labels:
            continue

        counts = Counter(labels)
        max_votes = max(counts.values())

        # Gather all labels that got the maximum number of votes
        tied_labels = [label for label, count in counts.items() if count == max_votes]

        # Sort alphabetically and pick the first one
        tied_labels.sort()
        results[item_id] = tied_labels[0]

    return results
