import pytest

from fleisskit.metrics import cohens_kappa, majority_vote, percent_agreement
from fleisskit.models import Annotation, AnnotationCollection


# --- Helper to quickly create annotations for tests ---
def _make_ann(item_id: str, annotator_id: str, label: str) -> Annotation:
    return Annotation(
        item_id=str(item_id), annotator_id=annotator_id, is_gold=False, label=label
    )


# Percent Agreement Tests


def test_percent_agreement_simple_example():
    """3 annotators label one item A, A, B. The pairs are A-A, A-B, A-B, so 1/3."""
    collection = AnnotationCollection(
        [
            _make_ann("item_1", "A1", "A"),
            _make_ann("item_1", "A2", "A"),
            _make_ann("item_1", "A3", "B"),
        ]
    )
    assert percent_agreement(collection) == pytest.approx(1.0 / 3.0)


def test_percent_agreement_skips_singletons():
    """Items with fewer than 2 labels should be skipped, averaging only the rest."""
    collection = AnnotationCollection(
        [
            # Item 1: only 1 label (should be skipped)
            _make_ann("item_1", "A1", "A"),
            # Item 2: 2 labels, complete agreement (1.0)
            _make_ann("item_2", "A1", "A"),
            _make_ann("item_2", "A2", "A"),
        ]
    )
    assert percent_agreement(collection) == pytest.approx(1.0)


def test_percent_agreement_raises_error_if_no_qualifying_items():
    """Raises ValueError if no items have >= 2 labels."""
    collection = AnnotationCollection(
        [
            _make_ann("item_1", "A1", "A"),
            _make_ann("item_2", "A1", "B"),
        ]
    )
    with pytest.raises(ValueError, match="No items found with at least 2 labels"):
        percent_agreement(collection)


# Cohen's Kappa Tests


def test_cohens_kappa_wikipedia_example():
    """
    Example
    - A=Yes, B=Yes: 20
    - A=Yes, B=No: 5
    - A=No,  B=Yes: 10
    - A=No,  B=No: 15
    Result: p_o = 0.7, p_e = 0.5, kappa = 0.4
    """
    anns = []
    # 20 Yes / Yes
    for i in range(20):
        anns.extend(
            [_make_ann(f"YY_{i}", "A", "Yes"), _make_ann(f"YY_{i}", "B", "Yes")]
        )
    # 5 Yes / No
    for i in range(5):
        anns.extend([_make_ann(f"YN_{i}", "A", "Yes"), _make_ann(f"YN_{i}", "B", "No")])
    # 10 No / Yes
    for i in range(10):
        anns.extend([_make_ann(f"NY_{i}", "A", "No"), _make_ann(f"NY_{i}", "B", "Yes")])
    # 15 No / No
    for i in range(15):
        anns.extend([_make_ann(f"NN_{i}", "A", "No"), _make_ann(f"NN_{i}", "B", "No")])

    collection = AnnotationCollection(anns)

    assert cohens_kappa(collection, "A", "B") == pytest.approx(0.4)


def test_cohens_kappa_raises_error_no_shared_items():
    """Raises ValueError if annotators share no items."""
    collection = AnnotationCollection(
        [
            _make_ann("item_1", "A", "Yes"),
            _make_ann("item_2", "B", "No"),
        ]
    )
    with pytest.raises(ValueError, match="have no shared items"):
        cohens_kappa(collection, "A", "B")


def test_cohens_kappa_raises_error_divide_by_zero():
    """Raises ValueError if p_e is 1.0 (both annotators only used the exact same label everywhere)."""
    collection = AnnotationCollection(
        [
            _make_ann("item_1", "A", "Yes"),
            _make_ann("item_1", "B", "Yes"),
            _make_ann("item_2", "A", "Yes"),
            _make_ann("item_2", "B", "Yes"),
        ]
    )
    with pytest.raises(ValueError, match="division by zero"):
        cohens_kappa(collection, "A", "B")


# Majority Vote Tests


def test_majority_vote_clear_winner():
    """A clear majority label is selected."""
    collection = AnnotationCollection(
        [
            _make_ann("item_1", "A1", "Positive"),
            _make_ann("item_1", "A2", "Positive"),
            _make_ann("item_1", "A3", "Negative"),
        ]
    )
    result = majority_vote(collection)
    assert result["item_1"] == "Positive"


def test_majority_vote_tie_breaker():
    """Ties are broken alphabetically (A_Label comes before B_Label)."""
    collection = AnnotationCollection(
        [
            _make_ann("item_1", "A1", "B_Label"),
            _make_ann("item_1", "A2", "B_Label"),
            _make_ann("item_1", "A3", "A_Label"),
            _make_ann("item_1", "A4", "A_Label"),
            _make_ann("item_1", "A5", "C_Label"),  # Just 1 vote
        ]
    )
    result = majority_vote(collection)
    # A_Label and B_Label both have 2 votes. A_Label wins alphabetically.
    assert result["item_1"] == "A_Label"


def test_majority_vote_handles_multiple_items():
    """Correctly processes multiple items independently."""
    collection = AnnotationCollection(
        [
            _make_ann("item_1", "A1", "Yes"),
            _make_ann("item_1", "A2", "Yes"),
            _make_ann("item_2", "A1", "No"),
            _make_ann("item_2", "A2", "Maybe"),
            _make_ann("item_2", "A3", "Maybe"),
        ]
    )
    result = majority_vote(collection)
    assert result["item_1"] == "Yes"
    assert result["item_2"] == "Maybe"
