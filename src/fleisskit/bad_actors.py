"""
Pulls the two annotator checks together into one list of bad actors.

An annotator is a bad actor when either check objects to them:

- the spam check (fleisskit.spam): their labels look like they were not
  produced by doing the task, and
- the leave-one-annotator-out outlier score (fleisskit.influence): Fleiss'
  kappa rises noticeably when their ratings are dropped.

The two usually agree, but not always. The spam check works on any design and
can name what is wrong (low agreement, one label for everything, too fast,
gold answers). The outlier score needs a constant number of ratings per item,
and only says that someone is out of line with the rest of this group. Taking
either keeps the cases one check can see and the other cannot.

Detection always runs on the full data: an annotator can only be judged
against their peers while those peers are still there. exclude_annotators()
then gives a collection without them, for a report on what is left.
"""

from dataclasses import dataclass

from fleisskit.influence import (
    _DEFAULT_OUTLIER_THRESHOLD,
    AnnotatorInfluence,
    leave_one_annotator_out,
)
from fleisskit.models import AnnotationCollection
from fleisskit.spam import DEFAULT_PARAMS, CombinerParams, SpamVerdict, detect_spammers

SPAM_CHECK = "spam_check"
LEAVE_ONE_OUT = "leave_one_out"


@dataclass(frozen=True)
class BadActor:
    """One annotator, and why each check objected to them."""

    annotator_id: str
    # Which checks flagged them: SPAM_CHECK and/or LEAVE_ONE_OUT.
    sources: tuple[str, ...]
    # One readable line per source, in the same order.
    reasons: tuple[str, ...]
    # The underlying results, when that check flagged them.
    spam: SpamVerdict | None = None
    influence: AnnotatorInfluence | None = None


def _spam_reason(verdict: SpamVerdict) -> str:
    why = ", ".join(verdict.reasons)
    if verdict.confidence_percent is not None:
        return f"spam check ({verdict.confidence_percent}% confident): {why}"
    return f"spam check: {why}"


def _influence_reason(info: AnnotatorInfluence) -> str:
    assert info.robust_score is not None and info.delta is not None
    return (
        f"leave-one-out outlier (score {info.robust_score:.1f}): kappa "
        f"{info.delta:+.4f} without them"
    )


def find_bad_actors(
    collection: AnnotationCollection,
    spam_params: CombinerParams = DEFAULT_PARAMS,
    base_rate: float | None = None,
    outlier_threshold: float = _DEFAULT_OUTLIER_THRESHOLD,
) -> list[BadActor]:
    """
    Runs both checks on the whole collection and returns everyone either one
    flagged, worst first (by spam score, then by outlier score).

    The leave-one-out part is skipped when Fleiss' kappa is undefined for the
    data (for example when items have different numbers of ratings); the spam
    check still runs.
    """
    spam_by_id = {
        v.annotator_id: v
        for v in detect_spammers(collection, spam_params, base_rate)
        if v.is_spammer
    }

    influence_by_id: dict[str, AnnotatorInfluence] = {}
    try:
        report = leave_one_annotator_out(
            collection, outlier_threshold=outlier_threshold
        )
    except ValueError:
        pass  # kappa undefined for this data: only the spam check applies
    else:
        influence_by_id = {r.annotator_id: r for r in report.annotators if r.is_outlier}

    bad_actors = []
    for annotator_id in sorted(set(spam_by_id) | set(influence_by_id)):
        spam = spam_by_id.get(annotator_id)
        influence = influence_by_id.get(annotator_id)
        sources = []
        reasons = []
        if spam is not None:
            sources.append(SPAM_CHECK)
            reasons.append(_spam_reason(spam))
        if influence is not None:
            sources.append(LEAVE_ONE_OUT)
            reasons.append(_influence_reason(influence))
        bad_actors.append(
            BadActor(
                annotator_id=annotator_id,
                sources=tuple(sources),
                reasons=tuple(reasons),
                spam=spam,
                influence=influence,
            )
        )

    bad_actors.sort(
        key=lambda b: (
            -(b.spam.score or 0.0 if b.spam else 0.0),
            -(b.influence.robust_score or 0.0 if b.influence else 0.0),
            b.annotator_id,
        )
    )
    return bad_actors


def exclude_annotators(
    collection: AnnotationCollection, annotator_ids: set[str]
) -> AnnotationCollection:
    """A collection without those annotators' annotations."""
    return AnnotationCollection(
        [a for a in collection.annotations if a.annotator_id not in annotator_ids]
    )
