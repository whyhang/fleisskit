import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from fleisskit.bad_actors import BadActor, exclude_annotators, find_bad_actors
from fleisskit.fleiss import fleiss_kappa, per_category_kappa
from fleisskit.generator import generate_synthetic_data
from fleisskit.influence import (
    _DEFAULT_OUTLIER_THRESHOLD,
    LeaveOneOutReport,
    leave_one_annotator_out,
)
from fleisskit.io import DataValidationError, load_csv, load_json
from fleisskit.metrics import majority_vote, percent_agreement
from fleisskit.spam import DEFAULT_PARAMS, SpamVerdict, detect_spammers


def _format_kappa_table(per_label: dict[str, float]) -> list[str]:
    """
    Formats per-category kappa as an aligned text table, one row per label.

    Labels are sorted alphabetically. Columns widen to fit the longest label or
    value. Kappa values are right-aligned so decimal points (and minus signs)
    line up.
    """
    rows = [(label, f"{kappa:.4f}") for label, kappa in sorted(per_label.items())]
    label_w = max(len("Label"), *(len(label) for label, _ in rows))
    kappa_w = max(len("Kappa"), *(len(value) for _, value in rows))

    lines = [
        f"  {'Label':<{label_w}} | {'Kappa':>{kappa_w}}",
        f"  {'-' * label_w}-+-{'-' * kappa_w}",
    ]
    lines += [f"  {label:<{label_w}} | {value:>{kappa_w}}" for label, value in rows]
    return lines


_OUTLIER_TEXT = {
    "outlier": "yes",
    "not_flagged": "",
    "not_proven": "not proven",
    None: "",
}

_EFFECT_TEXT = {
    "lowers_kappa": "lowers kappa",
    "raises_kappa": "raises kappa",
    "no_clear_effect": "no clear effect",
    None: "n/a",
}


# Below this many scored annotators the MAD rests on one or two deviations.
_MIN_RELIABLE_SCORED = 5


def _loo_score_notes(loo: LeaveOneOutReport) -> list[str]:
    """Explains the robust score, and warns when its spread estimate is shaky."""
    if loo.delta_median is None or loo.delta_mad is None:
        return []

    if loo.mad_is_zero:
        not_proven_note = (
            "  Score: not computed because the MAD of Delta is 0 (more than half "
            "the deltas are identical), so no one is flagged: not proven"
        )
        return [not_proven_note]

    score_note = (
        "  Score: 0.6745 * (Delta - median) / MAD, flagged above "
        f"{loo.outlier_threshold:g} "
        f"(median {loo.delta_median:+.4f}, MAD {loo.delta_mad:.4f})"
    )
    notes = [score_note]
    scored = sum(1 for info in loo.annotators if info.robust_score is not None)
    if scored < _MIN_RELIABLE_SCORED:
        notes.append(
            f"  Only {scored} annotator(s) scored: with fewer than "
            f"{_MIN_RELIABLE_SCORED} the median and MAD are unstable, so treat "
            "the scores as a hint only"
        )
    return notes


def _format_loo_table(loo: LeaveOneOutReport) -> list[str]:
    """
    Formats the leave-one-annotator-out result as an aligned text table.

    Rows keep the report's order (largest delta first). Delta is
    kappa_without - kappa_full, so a positive delta means the annotator pulls
    kappa down. Score is the modified z-score of Delta, and annotators above
    the threshold are marked in the Outlier column. The bootstrap CI columns
    are only added when the intervals were computed. Undefined rows show "N/A"
    and the reason.
    """
    with_ci = loo.ci_num_comparisons is not None
    family_alpha = loo.ci_family_alpha if loo.ci_family_alpha is not None else 0.05

    header = ["Annotator", "Items", "Kappa w/o", "Delta", "Score", "Outlier"]
    # Left-align text columns, right-align numeric ones.
    align = ["<", ">", ">", ">", ">", "<"]
    if with_ci:
        header += [f"{(1 - family_alpha) * 100:g}% CI of Delta", "Effect"]
        align += ["<", "<"]

    rows: list[tuple[str, ...]] = []
    notes: list[str] = []
    for info in loo.annotators:
        if info.kappa_without is None or info.delta is None:
            blanks = ("",) * (len(header) - 3)
            rows.append((info.annotator_id, str(info.items_rated), "N/A", *blanks))
            notes.append(f"  {info.annotator_id}: {info.error}")
            continue

        cells = [
            info.annotator_id,
            str(info.items_rated),
            f"{info.kappa_without:.4f}",
            f"{info.delta:+.4f}",
            f"{info.robust_score:.2f}" if info.robust_score is not None else "",
            _OUTLIER_TEXT[info.outlier_status],
        ]
        if with_ci:
            ci = info.delta_ci
            cells.append(
                f"[{ci.lower_bound:+.4f}, {ci.upper_bound:+.4f}]"
                if ci is not None
                else ""
            )
            cells.append(_EFFECT_TEXT[info.effect])
        rows.append(tuple(cells))

    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]

    def fmt(cells: tuple[str, ...]) -> str:
        parts = [f"{c:{a}{w}}" for c, a, w in zip(cells, align, widths, strict=True)]
        return ("  " + " | ".join(parts)).rstrip()

    lines = [fmt(tuple(header)), "  " + "-+-".join("-" * w for w in widths)]
    lines += [fmt(r) for r in rows]
    lines += _loo_score_notes(loo)

    m = loo.ci_num_comparisons
    if m is not None and m > 1:
        per_interval = (1 - family_alpha / m) * 100
        lines.append(
            f"  CI: paired bootstrap, Bonferroni-adjusted for {m} annotators "
            f"(each interval at {per_interval:.2f}%)"
        )
    elif m == 1:
        lines.append("  CI: paired bootstrap")
    return lines + notes


def _loo_to_json(loo: LeaveOneOutReport) -> dict:
    """JSON-ready dict for the leave-one-annotator-out result."""
    return {
        "full_kappa": loo.full_kappa,
        "delta_median": loo.delta_median,
        "delta_mad": loo.delta_mad,
        "outlier_threshold": loo.outlier_threshold,
        "mad_is_zero": loo.mad_is_zero,
        "ci_family_alpha": loo.ci_family_alpha,
        "ci_num_comparisons": loo.ci_num_comparisons,
        "annotators": [
            {
                "annotator_id": info.annotator_id,
                "items_rated": info.items_rated,
                "kappa_without": info.kappa_without,
                "delta": info.delta,
                "p_bar_without": info.p_bar_without,
                "p_bar_e_without": info.p_bar_e_without,
                "robust_score": info.robust_score,
                "is_outlier": info.is_outlier,
                "outlier_status": info.outlier_status,
                "delta_ci": (
                    {
                        "lower": info.delta_ci.lower_bound,
                        "upper": info.delta_ci.upper_bound,
                        "alpha": info.delta_ci.alpha,
                        "method": info.delta_ci.method,
                        "skipped_resamples": info.delta_ci.skipped_resamples,
                    }
                    if info.delta_ci is not None
                    else None
                ),
                "effect": info.effect,
                "error": info.error,
            }
            for info in loo.annotators
        ],
    }


def _fmt_optional(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "-"


def _format_spam_table(
    verdicts: list[SpamVerdict], base_rate: float | None = None
) -> list[str]:
    """
    Formats the spam check as an aligned text table, highest score first.

    Agreement, speed and variety are relative to the group (1.00 = typical).
    Gold is kappa against the gold labels, with the number of gold items.
    "-" means the signal could not be measured. Confidence (flagged
    annotators only) is the calibrated chance they really are a spammer.
    """
    header = (
        "Annotator", "Labels", "Agreement", "Speed", "Variety", "Gold",
        "Score", "Spammer", "Confidence", "Reasons",
    )  # fmt: skip
    align = ("<", ">", ">", ">", ">", ">", ">", "<", ">", "<")
    spammer_text = {"spammer": "yes", "ok": "", "insufficient_data": "n/a"}

    rows = []
    for v in verdicts:
        s = v.signals
        gold = (
            f"{s.gold_kappa:.2f} ({s.gold_items})"
            if s.gold_kappa is not None and s.gold_items
            else "-"
        )
        rows.append(
            (
                v.annotator_id,
                str(s.labels),
                _fmt_optional(s.agreement),
                _fmt_optional(s.speed),
                _fmt_optional(s.variety),
                gold,
                _fmt_optional(v.score),
                spammer_text[v.status],
                f"{v.confidence_percent}%" if v.confidence_percent is not None else "",
                ", ".join(v.reasons),
            )
        )

    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]

    def fmt(cells: tuple[str, ...]) -> str:
        parts = [f"{c:{a}{w}}" for c, a, w in zip(cells, align, widths, strict=True)]
        return ("  " + " | ".join(parts)).rstrip()

    lines = [fmt(header), "  " + "-+-".join("-" * w for w in widths)]
    lines += [fmt(r) for r in rows]
    lines.append(
        "  Agreement, Speed and Variety are relative (1.00 = typical); Gold is kappa "
        "against gold labels (items)"
    )
    lines.append(
        f"  Flagged at Score >= {DEFAULT_PARAMS.threshold:g}, with at least one "
        "reason besides speed; n/a = too little data to judge"
    )
    calibration = DEFAULT_PARAMS.calibration
    if calibration is not None:
        rate = base_rate if base_rate is not None else calibration.base_rate
        source = "--spam-base-rate" if base_rate is not None else "calibration data"
        lines.append(
            "  Confidence: chance a flagged annotator really is a spammer, "
            f"assuming {rate:.0%} of annotators are ({source})"
        )
    return lines


def _spam_to_json(verdicts: list[SpamVerdict]) -> list[dict[str, Any]]:
    """JSON-ready list for the spam check."""
    return [
        {
            "annotator_id": v.annotator_id,
            "status": v.status,
            "is_spammer": v.is_spammer,
            "score": v.score,
            "confidence": v.confidence,
            "confidence_percent": v.confidence_percent,
            "reasons": v.reasons,
            "suspicion": v.suspicion,
            "signals": {
                "labels": v.signals.labels,
                "agreement": v.signals.agreement,
                "agreement_kappa": v.signals.agreement_kappa,
                "agreement_items": v.signals.agreement_items,
                "speed": v.signals.speed,
                "median_seconds": v.signals.median_seconds,
                "variety": v.signals.variety,
                "entropy_bits": v.signals.entropy_bits,
                "reference_entropy_bits": v.signals.reference_entropy_bits,
                "gold_kappa": v.signals.gold_kappa,
                "gold_accuracy": v.signals.gold_accuracy,
                "gold_items": v.signals.gold_items,
            },
        }
        for v in verdicts
    ]


def _format_bad_actors(
    bad_actors: list[BadActor],
    excluded: bool,
    kappa_before: float | None,
    kappa_after: float | None,
) -> list[str]:
    """
    Tabulates the bad actors: annotator id and reason, worst first.

    Someone flagged by both checks gets one row per reason, with the id
    repeated, so every row stands alone (and grepping an id finds all of
    their reasons). When they were excluded, ends with Fleiss' kappa before
    and after, so the effect of dropping them is visible.
    """
    if not bad_actors:
        return ["Bad Actors: None"]

    headline = f"Bad Actors ({len(bad_actors)}"
    headline += ", excluded from this report):" if excluded else "):"

    rows = [(bad.annotator_id, reason) for bad in bad_actors for reason in bad.reasons]
    id_width = max(len("Annotator"), *(len(a) for a, _ in rows))
    reason_width = max(len("Reason"), *(len(r) for _, r in rows))
    lines = [
        headline,
        f"  {'Annotator':<{id_width}} | Reason",
        f"  {'-' * id_width}-+-{'-' * reason_width}",
    ]
    lines += [f"  {a:<{id_width}} | {r}" for a, r in rows]

    if excluded and kappa_before is not None and kappa_after is not None:
        lines.append(
            f"  Fleiss' Kappa: {kappa_before:.4f} with them, {kappa_after:.4f} without"
        )
    return lines


def _bad_actors_to_json(
    bad_actors: list[BadActor],
    excluded: bool,
    kappa_before: float | None,
) -> dict[str, Any]:
    """JSON-ready block for the bad actors."""
    return {
        "excluded": excluded,
        # Only set when they were excluded: the kappa the report would show
        # with them left in.
        "fleiss_kappa_with_bad_actors": kappa_before if excluded else None,
        "annotators": [
            {
                "annotator_id": bad.annotator_id,
                "sources": list(bad.sources),
                "reasons": list(bad.reasons),
                "spam_score": bad.spam.score if bad.spam else None,
                "spam_confidence_percent": (
                    bad.spam.confidence_percent if bad.spam else None
                ),
                "spam_reasons": list(bad.spam.reasons) if bad.spam else [],
                "outlier_score": (
                    bad.influence.robust_score if bad.influence else None
                ),
                "kappa_delta": bad.influence.delta if bad.influence else None,
            }
            for bad in bad_actors
        ],
    }


def run_report(
    file_path: Path,
    output_format: str,
    leave_one_out: bool = False,
    leave_one_out_ci: bool = False,
    outlier_threshold: float = _DEFAULT_OUTLIER_THRESHOLD,
    spam_check: bool = False,
    spam_base_rate: float | None = None,
    list_bad_actors: bool = False,
    exclude_bad_actors: bool = False,
) -> int:
    """report command logic"""
    try:
        if not file_path.exists():
            print(f"Error: File not found: {file_path}", file=sys.stderr)
            return 1

        # Route loader based on file extension
        if file_path.suffix.lower() == ".json":
            collection = load_json(file_path)
        else:
            collection = load_csv(file_path)

        # Bad actors are found on the whole dataset: an annotator can only be
        # judged against their peers while those peers are still in the data.
        bad_actors: list[BadActor] | None = None
        excluded = False
        kappa_before = None
        if list_bad_actors or exclude_bad_actors:
            bad_actors = find_bad_actors(
                collection,
                base_rate=spam_base_rate,
                outlier_threshold=outlier_threshold,
            )

        if exclude_bad_actors and bad_actors:
            remaining = exclude_annotators(
                collection, {bad.annotator_id for bad in bad_actors}
            )
            if not any(a.label is not None for a in remaining.annotations):
                names = ", ".join(bad.annotator_id for bad in bad_actors)
                print(
                    "Error: every annotator with labels is a bad actor "
                    f"({names}), so there is nothing left to report on.",
                    file=sys.stderr,
                )
                return 1
            try:
                kappa_before = fleiss_kappa(collection)
            except ValueError:
                kappa_before = None
            # Everything below is about the data without them.
            collection = remaining
            excluded = True

        # Calculate summaries
        items_count = len({a.item_id for a in collection.annotations})
        annotators_count = len({a.annotator_id for a in collection.annotations})
        labels_count = sum(1 for a in collection.annotations if a.label is not None)

        try:
            pa = percent_agreement(collection)
        except ValueError:
            pa = None

        mv = majority_vote(collection)
        mv_counts = dict(Counter(mv.values()))

        # Calculate Fleiss' Kappa
        fk_error = None
        try:
            fk = fleiss_kappa(collection)
        except ValueError as e:
            fk = None
            fk_error = str(e)

        # Calculate per-category (per-label) Fleiss' kappa
        pck_error = None
        try:
            pck = per_category_kappa(collection)
        except ValueError as e:
            pck = None
            pck_error = str(e)

        # Leave-one-annotator-out analysis (opt-in: it runs a bootstrap)
        loo: LeaveOneOutReport | None = None
        loo_error = None
        if leave_one_out or leave_one_out_ci:
            try:
                loo = leave_one_annotator_out(
                    collection,
                    calc_ci=leave_one_out_ci,
                    outlier_threshold=outlier_threshold,
                )
            except ValueError as e:
                loo_error = str(e)

        # Spam check (opt-in). Works with any design, no constant ratings needed.
        spam = (
            detect_spammers(collection, base_rate=spam_base_rate)
            if spam_check
            else None
        )

        # Output results to stdout
        if output_format == "json":
            output_data: dict[str, Any] = {
                "items": items_count,
                "annotators": annotators_count,
                "labels_provided": labels_count,
                "percent_agreement": pa,
                "majority_vote_distribution": mv_counts,
                "fleiss_kappa": fk,
                "per_category_kappa": pck,
            }
            if bad_actors is not None:
                output_data["bad_actors"] = _bad_actors_to_json(
                    bad_actors, excluded, kappa_before
                )
            if leave_one_out or leave_one_out_ci:
                output_data["leave_one_annotator_out"] = (
                    _loo_to_json(loo) if loo is not None else None
                )
            if spam is not None:
                output_data["spam_check"] = _spam_to_json(spam)
            print(json.dumps(output_data, indent=2))
        else:
            print(f"Report for {file_path.name}")
            print("-" * (11 + len(file_path.name)))
            print(f"Items: {items_count}")
            print(f"Annotators: {annotators_count}")
            print(f"Labels Provided: {labels_count}")

            if pa is not None:
                print(f"Percent Agreement: {pa:.2%}")
            else:
                print("Percent Agreement: N/A (not enough overlapping labels)")

            print("Majority Vote Distribution:")
            if mv_counts:
                for label, count in sorted(mv_counts.items()):
                    print(f"  - {label}: {count}")
            else:
                print("  None")

            # Print Fleiss' Kappa Output
            if fk is not None:
                print(f"Fleiss' Kappa: {fk:.4f}")
            else:
                print(f"Fleiss' Kappa: N/A ({fk_error})")

            # Print per-category kappa output
            if pck is not None:
                print("Per-Category Kappa:")
                if pck:
                    for line in _format_kappa_table(pck):
                        print(line)
                else:
                    print("  None (labels with p_j of 0 or 1 are skipped)")
            else:
                print(f"Per-Category Kappa: N/A ({pck_error})")

            if leave_one_out or leave_one_out_ci:
                if loo is not None:
                    print("Leave-One-Annotator-Out:")
                    for line in _format_loo_table(loo):
                        print(line)
                else:
                    print(f"Leave-One-Annotator-Out: N/A ({loo_error})")

            if spam is not None:
                print("Spam Check:")
                for line in _format_spam_table(spam, spam_base_rate):
                    print(line)

            if bad_actors is not None:
                for line in _format_bad_actors(bad_actors, excluded, kappa_before, fk):
                    print(line)

    except (DataValidationError, ValueError, TypeError) as e:
        # data problems exit with code 1
        print(f"Data error:\n{e}", file=sys.stderr)
        return 1

    # Exits with code 0 on success
    return 0


def _positive_int(value: str) -> int:
    """argparse type: a whole number >= 1."""
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number, got '{value}'")
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {number}")
    return number


def _positive_float(value: str) -> float:
    """argparse type: a number greater than 0."""
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, got '{value}'")
    if number <= 0:
        raise argparse.ArgumentTypeError(f"must be greater than 0, got {value}")
    return number


def _open_fraction(value: str) -> float:
    """argparse type: a number strictly between 0 and 1."""
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a number between 0 and 1, got '{value}'"
        )
    if not 0.0 < number < 1.0:
        raise argparse.ArgumentTypeError(
            f"must be greater than 0 and less than 1, got {value}"
        )
    return number


def _fraction(value: str) -> float:
    """argparse type: a number between 0 and 1 (inclusive)."""
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a number between 0 and 1, got '{value}'"
        )
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError(f"must be between 0 and 1, got {value}")
    return number


def _label_list(value: str) -> list[str]:
    """argparse type: at least 2 distinct, non-empty, comma-separated labels."""
    labels = [part.strip() for part in value.split(",")]
    if any(not label for label in labels):
        raise argparse.ArgumentTypeError(f"labels must not be empty, got '{value}'")
    if len(set(labels)) != len(labels):
        raise argparse.ArgumentTypeError(f"labels must be unique, got '{value}'")
    if len(labels) < 2:
        raise argparse.ArgumentTypeError(
            f"need at least 2 comma-separated labels, got '{value}'"
        )
    return labels


def run_generate(
    output_path: Path,
    num_items: int,
    num_annotators: int,
    labels: list[str],
    accuracy: float,
    spammer_share: float,
    skip_rate: float,
    seed: int,
    force: bool,
) -> int:
    """generate command logic"""
    # Format follows the file extension, like the report command's loader.
    suffix = output_path.suffix.lower()
    if suffix not in (".csv", ".json"):
        print(
            f"Error: Output file must end in .csv or .json, got: {output_path.name}",
            file=sys.stderr,
        )
        return 1

    if output_path.exists() and not force:
        print(
            f"Error: File already exists: {output_path} (use --force to overwrite)",
            file=sys.stderr,
        )
        return 1

    try:
        collection, _ = generate_synthetic_data(
            num_items=num_items,
            num_annotators=num_annotators,
            labels=labels,
            accuracy=accuracy,
            spammer_share=spammer_share,
            skip_rate=skip_rate,
            seed=seed,
            output_csv=output_path if suffix == ".csv" else None,
            output_json=output_path if suffix == ".json" else None,
        )
    except OSError as e:
        print(f"Error: Could not write {output_path}: {e}", file=sys.stderr)
        return 1

    print(
        f"Wrote {len(collection.annotations)} annotations "
        f"({num_items} items x {num_annotators} annotators, "
        f"{len(labels)} labels) to {output_path}"
    )
    return 0


def main(args: list[str] | None = None) -> int:
    """
    Main entry point for the CLI.
    Accepts args list (for testing) or reads sys.argv directly.
    """
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="fleisskit", description="Fleisskit annotation analysis tools."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Report Subcommand
    report_parser = subparsers.add_parser(
        "report", help="Generate a report from an annotation file"
    )
    report_parser.add_argument("file", type=Path, help="Path to the CSV or JSON file")
    report_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    report_parser.add_argument(
        "--leave-one-out",
        action="store_true",
        help="Recompute Fleiss' kappa with each annotator left out and flag "
        "outliers by the modified z-score of the change",
    )
    report_parser.add_argument(
        "--leave-one-out-ci",
        action="store_true",
        help="As --leave-one-out, plus a Bonferroni-adjusted 95%% bootstrap CI "
        "for each change (1000 resamples, seed 42; slower)",
    )
    report_parser.add_argument(
        "--list-bad-actors",
        action="store_true",
        help="List annotators flagged by the spam check or as leave-one-out "
        "outliers, with the reason, at the end of the report",
    )
    report_parser.add_argument(
        "--exclude-bad-actors",
        action="store_true",
        help="Report on the data with those annotators' labels removed, and "
        "list them (they are still found using the whole dataset)",
    )
    report_parser.add_argument(
        "--spam-check",
        action="store_true",
        help="Flag likely spammers from agreement with others, speed, label "
        "variety and gold accuracy",
    )
    report_parser.add_argument(
        "--spam-base-rate",
        type=_open_fraction,
        default=None,
        help="Share of annotators you expect to be spammers (0-1, exclusive). "
        "Adjusts the --spam-check confidence (default: the calibration data's rate)",
    )
    report_parser.add_argument(
        "--outlier-threshold",
        type=_positive_float,
        default=_DEFAULT_OUTLIER_THRESHOLD,
        help="Score above which an annotator is flagged as an outlier "
        f"(default: {_DEFAULT_OUTLIER_THRESHOLD:g})",
    )

    # Generate Subcommand
    generate_parser = subparsers.add_parser(
        "generate", help="Generate a synthetic annotation file (CSV or JSON)"
    )
    generate_parser.add_argument(
        "output",
        type=Path,
        help="Path of the file to write; the format follows the .csv / .json extension",
    )
    generate_parser.add_argument(
        "--items", type=_positive_int, default=50, help="Number of items (default: 50)"
    )
    generate_parser.add_argument(
        "--annotators",
        type=_positive_int,
        default=3,
        help="Number of annotators; each rates every item (default: 3)",
    )
    generate_parser.add_argument(
        "--labels",
        type=_label_list,
        default=["positive", "negative", "neutral"],
        help="Comma-separated labels (default: positive,negative,neutral)",
    )
    generate_parser.add_argument(
        "--accuracy",
        type=_fraction,
        default=0.8,
        help="Chance a normal annotator picks the true label (default: 0.8)",
    )
    generate_parser.add_argument(
        "--spammer-share",
        type=_fraction,
        default=0.0,
        help="Share of annotators who always pick one label (default: 0.0)",
    )
    generate_parser.add_argument(
        "--skip-rate",
        type=_fraction,
        default=0.0,
        help="Share of annotations randomly dropped (default: 0.0). "
        "Fleiss' kappa needs 0, since it requires equal ratings per item",
    )
    generate_parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    generate_parser.add_argument(
        "--force", action="store_true", help="Overwrite the output file if it exists"
    )

    try:
        parsed_args = parser.parse_args(args)
    except SystemExit as e:
        # automatically calls sys.exit(2) on bad usage.
        # catching it here so tests can verify without crashing
        return e.code if isinstance(e.code, int) else 2

    if parsed_args.command == "report":
        return run_report(
            parsed_args.file,
            parsed_args.format,
            parsed_args.leave_one_out,
            parsed_args.leave_one_out_ci,
            parsed_args.outlier_threshold,
            parsed_args.spam_check,
            parsed_args.spam_base_rate,
            parsed_args.list_bad_actors,
            parsed_args.exclude_bad_actors,
        )

    if parsed_args.command == "generate":
        return run_generate(
            output_path=parsed_args.output,
            num_items=parsed_args.items,
            num_annotators=parsed_args.annotators,
            labels=parsed_args.labels,
            accuracy=parsed_args.accuracy,
            spammer_share=parsed_args.spammer_share,
            skip_rate=parsed_args.skip_rate,
            seed=parsed_args.seed,
            force=parsed_args.force,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
