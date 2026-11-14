import pytest

from fleisskit.fleiss import _fleiss_kappa_components, build_count_table, fleiss_kappa
from fleisskit.models import Annotation, AnnotationCollection


def _make_ann(item_id: str, annotator_id: str, label: str) -> Annotation:
    """Helper to quickly create annotations for tests."""
    return Annotation(
        item_id=str(item_id), annotator_id=annotator_id, is_gold=False, label=label
    )


def test_build_count_table():
    """Builds a correct 2D count matrix."""
    collection = AnnotationCollection(
        [
            _make_ann("item_1", "A1", "Pos"),
            _make_ann("item_1", "A2", "Neg"),
            _make_ann("item_2", "A1", "Pos"),
            _make_ann("item_2", "A2", "Pos"),
        ]
    )
    table, labels = build_count_table(collection)

    assert labels == ["Neg", "Pos"]
    assert table == [[1, 1], [0, 2]]


def test_fleiss_kappa_varying_ratings_raises_error():
    """Raises ValueError if items don't all have the same number of ratings."""
    collection = AnnotationCollection(
        [
            _make_ann("item_1", "A1", "Pos"),
            _make_ann("item_1", "A2", "Pos"),
            _make_ann("item_2", "A1", "Pos"),
        ]
    )
    with pytest.raises(ValueError, match="Varying number of ratings per item"):
        fleiss_kappa(collection)


def test_fleiss_kappa_wikipedia_example():
    """Exact worked example from the Fleiss' kappa Wikipedia article."""
    wiki_table = [
        [0, 0, 0, 0, 14],
        [0, 2, 6, 4, 2],
        [0, 0, 3, 5, 6],
        [0, 3, 9, 2, 0],
        [2, 2, 8, 1, 1],
        [7, 7, 0, 0, 0],
        [3, 2, 6, 3, 0],
        [2, 5, 3, 2, 2],
        [6, 5, 2, 1, 0],
        [0, 2, 2, 3, 7],
    ]

    kappa, P_bar, P_bar_e = _fleiss_kappa_components(wiki_table)

    assert P_bar == pytest.approx(0.378, abs=0.001)
    assert P_bar_e == pytest.approx(0.213, abs=0.001)
    assert kappa == pytest.approx(0.210, abs=0.001)
