import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from fleisskit.fleiss import fleiss_kappa, per_category_kappa
from fleisskit.generator import generate_synthetic_data
from fleisskit.io import DataValidationError, load_csv, load_json
from fleisskit.metrics import majority_vote, percent_agreement


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


def run_report(file_path: Path, output_format: str) -> int:
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

        # Output results to stdout
        if output_format == "json":
            output_data = {
                "items": items_count,
                "annotators": annotators_count,
                "labels_provided": labels_count,
                "percent_agreement": pa,
                "majority_vote_distribution": mv_counts,
                "fleiss_kappa": fk,
                "per_category_kappa": pck,
            }
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
        return run_report(parsed_args.file, parsed_args.format)

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
