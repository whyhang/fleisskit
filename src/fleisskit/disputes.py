"""
Finds disputed items: items whose labels conflict, and says why with tags.

Each item gets every tag whose rule fires. Its severity is its worst tag.
Tags, most severe first:

1. GOLD_KEY_CONFLICT  the item's own rows disagree on the gold answer: two
                      different gold labels, or gold on some rows only.
2. GOLD_MISMATCH      a gold item whose gold label is not a winner of the vote.
3. TIE                two or more labels share the top vote count, so the
                      majority vote is decided by alphabetical order.
4. BAD_ACTOR_FLIP     the majority vote over all labels is not a winner once
                      bad actors' labels are removed.
5. NO_MAJORITY        one winner, but with half of the votes or fewer.
6. WEAK_MAJORITY      one winner with over half the votes, but a share below
                      the cutoff (default 2/3).
7. CONFIDENT_DISSENT  on an item with at least 6 votes, a losing label was
                      chosen by at least 2 annotators with confidence >= 0.8.
8. BAD_ACTOR_NOISE    unanimous once bad actors are removed, but not with
                      them: the whole disagreement came from bad actors.

BAD_ACTOR_FLIP and BAD_ACTOR_NOISE exist only when bad actors are supplied
(bad_actor_ids). Without them every annotator is trusted, those two tags are
left out of the report, and "votes" below means all labels.

Definitions:

- votes: per item, how many trusted annotators gave each label (blank labels
  are ignored). With bad actors supplied, their labels are not counted.
- winners: the label(s) with the most votes.
- current label: what fleisskit.metrics.majority_vote gives for the item, from
  all labels, ties broken alphabetically.
- Items with fewer than 2 votes are "not assessable": only GOLD_KEY_CONFLICT
  can be checked for them. They are listed separately in the report.

Everything is deterministic: vote comparisons use whole numbers and exact
fractions, the only decimal comparison is confidence >= 0.8, nothing is random,
and the input order of the annotations does not matter.

Why CONFIDENT_DISSENT needs at least 6 votes: two dissenters can only share a
losing label when the winner has at least 3 votes, so below 5 votes the rule
can never fire. At exactly 5 the only such split is 3-2, which WEAK_MAJORITY
already flags. 6 is the smallest item size where it can catch a dispute the
vote counts alone miss (4-2).
"""

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction

from fleisskit.models import Annotation, AnnotationCollection

GOLD_KEY_CONFLICT = "GOLD_KEY_CONFLICT"
GOLD_MISMATCH = "GOLD_MISMATCH"
TIE = "TIE"
BAD_ACTOR_FLIP = "BAD_ACTOR_FLIP"
NO_MAJORITY = "NO_MAJORITY"
WEAK_MAJORITY = "WEAK_MAJORITY"
CONFIDENT_DISSENT = "CONFIDENT_DISSENT"
BAD_ACTOR_NOISE = "BAD_ACTOR_NOISE"

# Most severe first. An item's severity is the 1-based position of its worst tag.
TAG_ORDER: tuple[str, ...] = (
    GOLD_KEY_CONFLICT,
    GOLD_MISMATCH,
    TIE,
    BAD_ACTOR_FLIP,
    NO_MAJORITY,
    WEAK_MAJORITY,
    CONFIDENT_DISSENT,
    BAD_ACTOR_NOISE,
)
# Only exist when bad actors are supplied.
BAD_ACTOR_TAGS = frozenset({BAD_ACTOR_FLIP, BAD_ACTOR_NOISE})

# Vote-based rules need at least this many votes on an item.
MIN_VOTES = 2
# WEAK_MAJORITY: the winner's share of the votes is below this.
DEFAULT_WEAK_MAJORITY_CUTOFF = Fraction(2, 3)
# CONFIDENT_DISSENT: at least this many votes on the item (see module docstring),
CONFIDENT_DISSENT_MIN_VOTES = 6
# ... and this many annotators on one losing label,
CONFIDENT_DISSENT_MIN_DISSENTERS = 2
# ... each with at least this self-reported confidence.
CONFIDENT_DISSENT_MIN_CONFIDENCE = 0.8


@dataclass(frozen=True)
class DisputedItem:
    """One disputed item, its tags (worst first) and the evidence."""

    item_id: str
    # Tags in TAG_ORDER order, most severe first.
    tags: tuple[str, ...]
    # One readable reason per tag, in the same order.
    reasons: tuple[str, ...]
    # 1-based position of the worst tag in TAG_ORDER (1 = most severe).
    severity: int
    # Label -> count over trusted annotators (all annotators when bad actors
    # are not used). These are the votes the rules use.
    votes: dict[str, int]
    # Label -> count over every annotator, bad actors included.
    all_votes: dict[str, int]
    # Label(s) with the most votes, sorted.
    winners: tuple[str, ...]
    # majority_vote over all labels (ties go to the alphabetically first).
    # None if the item has no labels at all.
    current_label: str | None
    # Distinct gold labels given for the item, sorted (empty if not gold).
    gold_labels: tuple[str, ...]


@dataclass(frozen=True)
class ItemVotes:
    """The votes on an item the rules could not judge (fewer than MIN_VOTES)."""

    item_id: str
    # As in DisputedItem.
    votes: dict[str, int]
    all_votes: dict[str, int]
    winners: tuple[str, ...]
    current_label: str | None
    gold_labels: tuple[str, ...]


@dataclass(frozen=True)
class DisputeReport:
    """Every disputed item, worst first, plus what the rules ran with."""

    # Sorted by severity, then most tags first, then item id.
    items: list[DisputedItem]
    items_checked: int
    # Items with fewer than MIN_VOTES votes: the vote rules could not run.
    not_assessable: tuple[str, ...]
    # The tags this run could produce (bad actor tags only when used), in
    # severity order, with how many items got each one.
    tag_counts: dict[str, int]
    weak_majority_cutoff: Fraction
    used_bad_actors: bool
    # The annotators whose labels were set aside (empty when not used).
    bad_actor_ids: tuple[str, ...]
    # The votes on each not_assessable item, in the same order.
    not_assessable_votes: tuple[ItemVotes, ...] = ()


def to_fraction(value: Fraction | float | str) -> Fraction:
    """
    An exact fraction for a cutoff. Floats are read by their shortest decimal
    form, so 0.67 means 67/100 exactly. Strings may be decimals or "2/3".
    """
    if isinstance(value, Fraction):
        return value
    if isinstance(value, float):
        return Fraction(repr(value))
    return Fraction(value)


def _validate_cutoff(value: Fraction | float | str) -> Fraction:
    try:
        cutoff = to_fraction(value)
    except (ValueError, ZeroDivisionError) as e:
        raise ValueError(f"Weak majority cutoff must be a number, got {value!r}") from e
    if not Fraction(1, 2) < cutoff <= 1:
        raise ValueError(
            "Weak majority cutoff must be above 1/2 and at most 1, "
            f"got {format_fraction(cutoff)}"
        )
    return cutoff


def format_fraction(value: Fraction) -> str:
    """2/3 for simple fractions, else a decimal (67/100 -> 0.67)."""
    if value.denominator == 1:
        return str(value.numerator)
    if value.denominator <= 12:
        return f"{value.numerator}/{value.denominator}"
    return f"{float(value):g}"


def format_votes(votes: dict[str, int]) -> str:
    """'neutral 5, negative 4', in the dict's order (most votes first); '-' if none."""
    return ", ".join(f"{label} {count}" for label, count in votes.items()) or "-"


def _count(annotations: Iterable[Annotation]) -> dict[str, int]:
    counts = Counter(a.label for a in annotations if a.label is not None)
    # Most votes first, then alphabetical: a stable, readable order.
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _winners(votes: dict[str, int]) -> tuple[str, ...]:
    if not votes:
        return ()
    top = max(votes.values())
    return tuple(sorted(label for label, n in votes.items() if n == top))


def _share(part: int, whole: int) -> str:
    return f"{part} of {whole} votes ({part / whole:.0%})"


def _quoted(labels: Iterable[str]) -> str:
    return ", ".join(f"'{label}'" for label in labels)


def _check_item(
    item_id: str,
    rows: list[Annotation],
    bad: frozenset[str] | None,
    cutoff: Fraction,
) -> tuple[list[tuple[str, str]], DisputedItem | None, ItemVotes | None]:
    """
    Runs every rule on one item. Returns the (tag, reason) pairs, the item
    (None if no tag fired) and, if the vote rules could not run, its votes.
    """
    trusted = rows if bad is None else [r for r in rows if r.annotator_id not in bad]
    all_votes = _count(rows)
    votes = _count(trusted)
    n = sum(votes.values())
    winners = _winners(votes)
    all_winners = _winners(all_votes)
    current = all_winners[0] if all_winners else None

    gold_rows = [r for r in rows if r.is_gold]
    gold_labels = tuple(sorted({r.gold_label for r in gold_rows if r.gold_label}))

    found: dict[str, str] = {}

    # 1. The answer key contradicts itself.
    if len(gold_labels) > 1:
        found[GOLD_KEY_CONFLICT] = f"gold labels disagree: {_quoted(gold_labels)}"
    elif gold_rows and len(gold_rows) < len(rows):
        found[GOLD_KEY_CONFLICT] = (
            f"marked gold on {len(gold_rows)} of {len(rows)} rows"
        )

    assessable = n >= MIN_VOTES
    if assessable:
        top = votes[winners[0]]
        tied = len(winners) > 1

        # 2. The votes reject the (single) gold answer.
        if len(gold_labels) == 1 and gold_labels[0] not in winners:
            found[GOLD_MISMATCH] = (
                f"gold is '{gold_labels[0]}' ({votes.get(gold_labels[0], 0)} of "
                f"{n} votes), votes favour {_quoted(winners)}"
            )

        # 3. Majority vote is down to alphabetical order.
        if tied:
            each = "vote" if top == 1 else "votes"
            found[TIE] = f"tie between {_quoted(winners)} ({top} {each} each)"

        # 4. Bad actors changed the label.
        if bad is not None and current is not None and current not in winners:
            found[BAD_ACTOR_FLIP] = (
                f"'{current}' with bad actors, {_quoted(winners)} without them"
            )

        # 5-6. The shape of a single winner's lead.
        if not tied:
            if 2 * top <= n:
                found[NO_MAJORITY] = f"'{winners[0]}' leads with {_share(top, n)}"
            elif Fraction(top, n) < cutoff:
                found[WEAK_MAJORITY] = (
                    f"'{winners[0]}' has {_share(top, n)}, "
                    f"below {format_fraction(cutoff)}"
                )

        # 7. A sure minority, on a big enough item.
        if n >= CONFIDENT_DISSENT_MIN_VOTES:
            sure = Counter(
                r.label
                for r in trusted
                if r.label is not None
                and r.label not in winners
                and r.confidence is not None
                and r.confidence >= CONFIDENT_DISSENT_MIN_CONFIDENCE
            )
            dissent = sorted(
                (label, k)
                for label, k in sure.items()
                if k >= CONFIDENT_DISSENT_MIN_DISSENTERS
            )
            if dissent:
                found[CONFIDENT_DISSENT] = "; ".join(
                    f"{k} annotators chose '{label}' with confidence >= "
                    f"{CONFIDENT_DISSENT_MIN_CONFIDENCE:g}"
                    for label, k in dissent
                )

        # 8. Only the bad actors disagreed.
        if bad is not None and len(votes) == 1 and len(all_votes) > 1:
            others = [label for label in all_votes if label not in votes]
            found[BAD_ACTOR_NOISE] = (
                f"unanimous '{winners[0]}' without bad actors; they added "
                f"{_quoted(others)}"
            )

    unjudged = (
        None
        if assessable
        else ItemVotes(
            item_id=item_id,
            votes=votes,
            all_votes=all_votes,
            winners=winners,
            current_label=current,
            gold_labels=gold_labels,
        )
    )
    pairs = [(tag, found[tag]) for tag in TAG_ORDER if tag in found]
    if not pairs:
        return pairs, None, unjudged
    item = DisputedItem(
        item_id=item_id,
        tags=tuple(tag for tag, _ in pairs),
        reasons=tuple(reason for _, reason in pairs),
        severity=TAG_ORDER.index(pairs[0][0]) + 1,
        votes=votes,
        all_votes=all_votes,
        winners=winners,
        current_label=current,
        gold_labels=gold_labels,
    )
    return pairs, item, unjudged


def find_disputes(
    collection: AnnotationCollection,
    weak_majority_cutoff: Fraction | float | str = DEFAULT_WEAK_MAJORITY_CUTOFF,
    bad_actor_ids: Iterable[str] | None = None,
) -> DisputeReport:
    """
    Tags every disputed item in the collection (see the module docstring for
    the rules) and returns them worst first.

    weak_majority_cutoff: WEAK_MAJORITY fires when the winner's share of the
        votes is below this. Must be above 1/2 and at most 1. Taken exactly:
        0.67 means 67/100, so a 2-of-3 vote (0.666...) counts as weak; pass
        Fraction(2, 3) or "2/3" (the default) to keep 2-of-3 out.
    bad_actor_ids: annotators whose labels should not count (for example
        from fleisskit.bad_actors.find_bad_actors, run on this same
        collection). None (the default) trusts everyone and leaves out
        BAD_ACTOR_FLIP and BAD_ACTOR_NOISE. An empty set keeps those tags,
        which then never fire.

    Raises ValueError for a cutoff outside (1/2, 1].
    """
    cutoff = _validate_cutoff(weak_majority_cutoff)
    bad = None if bad_actor_ids is None else frozenset(bad_actor_ids)

    by_item: dict[str, list[Annotation]] = defaultdict(list)
    for ann in collection.annotations:
        by_item[ann.item_id].append(ann)

    available = [t for t in TAG_ORDER if bad is not None or t not in BAD_ACTOR_TAGS]
    tag_counts = dict.fromkeys(available, 0)
    items: list[DisputedItem] = []
    not_assessable: list[ItemVotes] = []

    for item_id in sorted(by_item):
        pairs, item, unjudged = _check_item(item_id, by_item[item_id], bad, cutoff)
        if unjudged is not None:
            not_assessable.append(unjudged)
        for tag, _ in pairs:
            tag_counts[tag] += 1
        if item is not None:
            items.append(item)

    items.sort(key=lambda it: (it.severity, -len(it.tags), it.item_id))
    return DisputeReport(
        items=items,
        items_checked=len(by_item),
        not_assessable=tuple(v.item_id for v in not_assessable),
        tag_counts=tag_counts,
        weak_majority_cutoff=cutoff,
        used_bad_actors=bad is not None,
        bad_actor_ids=tuple(sorted(bad)) if bad is not None else (),
        not_assessable_votes=tuple(not_assessable),
    )
