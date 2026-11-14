import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from fleisskit.io import DataValidationError, load_csv, load_json
from fleisskit.metrics import majority_vote, percent_agreement


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
            pa = None  # Handle only 1 reviewe per label

        mv = majority_vote(collection)
        mv_counts = dict(Counter(mv.values()))

        # results to stdout
        if output_format == "json":
            output_data = {
                "items": items_count,
                "annotators": annotators_count,
                "labels_provided": labels_count,
                "percent_agreement": pa,
                "majority_vote_distribution": mv_counts,
                "fleiss_kappa": None,  # sneak peek
            }
            print(json.dumps(output_data, indent=2))
        else:
            print(f"Report for {file_path.name}")
            print("-" * (11 + len(file_path.name)))  # center aligner
            print(f"Items: {items_count}")
            print(f"Annotators: {annotators_count}")
            print(f"Labels Provided: {labels_count}")

            if pa is not None:
                print(f"Percent Agreement: {pa:.2%}")
            else:
                print("Percent Agreement: N/A (not enough overlapping labels)")

            print("Majority Vote Distribution:")
            if mv_counts:
                # consistency - alphabetical sort
                for label, count in sorted(mv_counts.items()):
                    print(f"  - {label}: {count}")
            else:
                print("  None")

            print("Fleiss' Kappa: Not yet implemented")

    except (DataValidationError, ValueError, TypeError) as e:
        # data problems exit with code 1
        print(f"Data error:\n{e}", file=sys.stderr)
        return 1

    # Exits with code 0 on success
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

    try:
        parsed_args = parser.parse_args(args)
    except SystemExit as e:
        # automatically calls sys.exit(2) on bad usage.
        # catching it here so tests can verify without crashing
        return e.code if isinstance(e.code, int) else 2

    if parsed_args.command == "report":
        return run_report(parsed_args.file, parsed_args.format)

    return 0


if __name__ == "__main__":
    sys.exit(main())
