import json

from fleisskit.cli import main


def test_cli_report_text(tmp_path, capsys):
    """clean file loads and prints a human-readable result summary."""
    csv_file = tmp_path / "valid.csv"
    csv_file.write_text(
        "item_id,annotator_id,is_gold,label\n"
        "item_1,user_1,false,positive\n"
        "item_1,user_2,false,positive\n"
        "item_2,user_1,false,negative\n",
        encoding="utf-8",
    )

    exit_code = main(["report", str(csv_file)])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Report for valid.csv" in captured.out
    assert "Items: 2" in captured.out
    assert "Annotators: 2" in captured.out
    assert "Labels Provided: 3" in captured.out
    assert "Percent Agreement: 100.00%" in captured.out
    assert "- positive: 1" in captured.out
    assert "- negative: 1" in captured.out
    assert "Fleiss' Kappa: N/A (Varying number of ratings per item." in captured.out
    assert captured.err == ""  # Stderr MUST be clean


def test_cli_report_json(tmp_path, capsys):
    """JSON format prints valid JSON to stdout"""
    csv_file = tmp_path / "valid.csv"
    csv_file.write_text(
        "item_id,annotator_id,is_gold,label\n"
        "item_1,user_1,false,yes\n"
        "item_1,user_2,false,yes\n",
        encoding="utf-8",
    )

    exit_code = main(["report", str(csv_file), "--format", "json"])

    assert exit_code == 0
    captured = capsys.readouterr()

    # This will fail if the output is not strictly valid JSON
    data = json.loads(captured.out)

    assert data["items"] == 1
    assert data["annotators"] == 2
    assert data["labels_provided"] == 2
    assert data["percent_agreement"] == 1.0
    assert data["majority_vote_distribution"] == {"yes": 1}
    assert "fleiss_kappa" in data
    assert captured.err == ""


def test_cli_missing_file(capsys):
    """A missing file exits with code 1 => stderr."""
    exit_code = main(["report", "does_not_exist_file.csv"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert captured.out == ""  # Stdout MUST be empty
    assert "Error: File not found" in captured.err


def test_cli_broken_file(tmp_path, capsys):
    """Data validation errors exit with code 1"""
    csv_file = tmp_path / "broken.csv"
    csv_file.write_text(
        "item_id,annotator_id,is_gold,label\n"
        "1,A,true,yes\n",  # is_gold but missing gold_label
        encoding="utf-8",
    )

    exit_code = main(["report", str(csv_file)])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Data error" in captured.err
    assert "gold_label is required" in captured.err


def test_cli_wrong_usage(capsys):
    """No file argument triggers argparse's code 2."""
    exit_code = main(["report"])
    assert exit_code == 2
    captured = capsys.readouterr()

    assert "the following arguments are required: file" in captured.err
