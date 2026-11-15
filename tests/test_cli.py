import json
from pathlib import Path

import pytest

from fleisskit.cli import main
from fleisskit.io import load_csv, load_json

SAMPLE_CSV = Path(__file__).parent / "data" / "sample.csv"


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
    assert (
        "Per-Category Kappa: N/A (Varying number of ratings per item." in captured.out
    )
    assert captured.err == ""  # Stderr MUST be clean


def test_cli_report_text_per_category_kappa(tmp_path, capsys):
    """With constant ratings per item, the report shows a per-label kappa table."""
    csv_file = tmp_path / "valid.csv"
    csv_file.write_text(
        "item_id,annotator_id,is_gold,label\n"
        "item_1,user_1,false,positive\n"
        "item_1,user_2,false,positive\n"
        "item_2,user_1,false,negative\n"
        "item_2,user_2,false,neutral\n"
        "item_3,user_1,false,neutral\n"
        "item_3,user_2,false,negative\n",
        encoding="utf-8",
    )

    exit_code = main(["report", str(csv_file)])
    assert exit_code == 0
    captured = capsys.readouterr()
    # positive is used consistently; negative/neutral are always mixed up.
    # Rows are sorted by label; columns are padded so values line up.
    expected_table = (
        "Per-Category Kappa:\n"
        "  Label    |   Kappa\n"
        "  ---------+--------\n"
        "  negative | -0.5000\n"
        "  neutral  | -0.5000\n"
        "  positive |  1.0000\n"
    )
    assert expected_table in captured.out
    assert captured.err == ""


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
    # The only label has p_j = 1, so it is skipped -> empty dict, not an error.
    assert data["per_category_kappa"] == {}
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


# Generate command tests


def test_cli_generate_defaults_reproduce_sample_file(tmp_path, capsys):
    """Default options recreate tests/data/sample.csv byte-for-byte (used by the README)."""
    out = tmp_path / "sample.csv"

    exit_code = main(["generate", str(out)])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Wrote 150 annotations (50 items x 3 annotators, 3 labels)" in captured.out
    assert captured.err == ""
    assert out.read_bytes() == SAMPLE_CSV.read_bytes()


def test_cli_generate_json_gives_same_report_as_csv(tmp_path, capsys):
    """A generated .json file is valid JSON and reports identically to the .csv twin."""
    csv_out, json_out = tmp_path / "data.csv", tmp_path / "data.json"
    assert main(["generate", str(csv_out), "--seed", "7"]) == 0
    assert main(["generate", str(json_out), "--seed", "7"]) == 0

    assert isinstance(json.loads(json_out.read_text(encoding="utf-8")), list)
    assert len(load_json(json_out).annotations) == 150
    capsys.readouterr()

    assert main(["report", str(csv_out), "--format", "json"]) == 0
    csv_report = json.loads(capsys.readouterr().out)
    assert main(["report", str(json_out), "--format", "json"]) == 0
    json_report = json.loads(capsys.readouterr().out)

    assert csv_report == json_report
    assert csv_report["per_category_kappa"]  # kappa is computable on generated data


def test_cli_generate_custom_options(tmp_path, capsys):
    """Items, annotators, labels and skip rate are passed through to the generator."""
    out = tmp_path / "custom.csv"

    exit_code = main(
        [
            "generate",
            str(out),
            "--items", "10",
            "--annotators", "4",
            "--labels", "cat, dog",
            "--skip-rate", "0.5",
            "--seed", "1",
        ]
    )  # fmt: skip

    assert exit_code == 0
    collection = load_csv(out)
    assert len(collection.annotations) == 20  # 10 * 4, half skipped
    assert {a.item_id for a in collection.annotations} <= {
        f"item_{i}" for i in range(10)
    }
    assert {a.label for a in collection.annotations} <= {"cat", "dog"}
    assert "Wrote 20 annotations (10 items x 4 annotators, 2 labels)" in (
        capsys.readouterr().out
    )


def test_cli_generate_refuses_to_overwrite_without_force(tmp_path, capsys):
    """An existing file is left untouched unless --force is given."""
    out = tmp_path / "existing.csv"
    out.write_text("do not clobber me", encoding="utf-8")

    assert main(["generate", str(out)]) == 1
    captured = capsys.readouterr()
    assert "File already exists" in captured.err
    assert "--force" in captured.err
    assert captured.out == ""
    assert out.read_text(encoding="utf-8") == "do not clobber me"

    assert main(["generate", str(out), "--force"]) == 0
    assert len(load_csv(out).annotations) == 150


def test_cli_generate_rejects_unknown_extension(tmp_path, capsys):
    """Only .csv and .json are accepted; nothing is written otherwise."""
    out = tmp_path / "data.txt"

    assert main(["generate", str(out)]) == 1
    captured = capsys.readouterr()
    assert "must end in .csv or .json" in captured.err
    assert captured.out == ""
    assert not out.exists()


def test_cli_generate_missing_directory(tmp_path, capsys):
    """A path in a missing directory gives a clean error, not a traceback."""
    out = tmp_path / "no_such_dir" / "data.csv"

    assert main(["generate", str(out)]) == 1
    captured = capsys.readouterr()
    assert "Could not write" in captured.err
    assert captured.out == ""


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("--items", "0", "must be at least 1"),
        ("--annotators", "x", "expected a whole number"),
        ("--accuracy", "1.5", "must be between 0 and 1"),
        ("--skip-rate", "-0.1", "must be between 0 and 1"),
        ("--spammer-share", "nan", "must be between 0 and 1"),
        ("--labels", "only_one", "need at least 2"),
        ("--labels", "a,,b", "must not be empty"),
        ("--labels", "a,a", "must be unique"),
    ],
)
def test_cli_generate_invalid_options(tmp_path, capsys, option, value, message):
    """Bad option values are usage errors (exit 2) and write nothing."""
    out = tmp_path / "bad.csv"

    assert main(["generate", str(out), option, value]) == 2
    captured = capsys.readouterr()
    assert message in captured.err
    assert not out.exists()
