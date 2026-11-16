import json
from pathlib import Path

import pytest

from fleisskit.cli import main
from fleisskit.io import load_csv, load_json

SAMPLE_CSV = Path(__file__).parent / "data" / "sample.csv"
OUTLIERS_CSV = Path(__file__).parent / "data" / "outliers.csv"
SPAM_KNOWN_CSV = Path(__file__).parent / "data" / "spam_known.csv"


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


# Leave-one-annotator-out tests

# A1 and A2 always agree; A3 contradicts them on the first two items.
# Full kappa = 1/3; without A3 -> 1.0; without A1 or A2 -> 0.0.
LOO_CSV = (
    "item_id,annotator_id,is_gold,label\n"
    "i1,A1,false,x\ni1,A2,false,x\ni1,A3,false,y\n"
    "i2,A1,false,y\ni2,A2,false,y\ni2,A3,false,x\n"
    "i3,A1,false,x\ni3,A2,false,x\ni3,A3,false,x\n"
    "i4,A1,false,y\ni4,A2,false,y\ni4,A3,false,y\n"
)


def test_cli_report_leave_one_out_text(tmp_path, capsys):
    """--leave-one-out adds a table sorted by delta, largest first, no bootstrap."""
    csv_file = tmp_path / "loo.csv"
    csv_file.write_text(LOO_CSV, encoding="utf-8")

    exit_code = main(["report", str(csv_file), "--leave-one-out"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    out = captured.out
    assert "Fleiss' Kappa: 0.3333" in out
    assert "Leave-One-Annotator-Out:" in out
    table = out.split("Leave-One-Annotator-Out:\n")[1]
    # Deltas are [+2/3, -1/3, -1/3], so the MAD is 0: nobody is scored or
    # flagged, and every verdict is "not proven".
    assert table == (
        "  Annotator | Items | Kappa w/o |   Delta | Score | Outlier\n"
        "  ----------+-------+-----------+---------+-------+-----------\n"
        "  A3        |     4 |    1.0000 | +0.6667 |       | not proven\n"
        "  A1        |     4 |    0.0000 | -0.3333 |       | not proven\n"
        "  A2        |     4 |    0.0000 | -0.3333 |       | not proven\n"
        "  Score: not computed because the MAD of Delta is 0 (more than half the "
        "deltas are identical), so no one is flagged: not proven\n"
    )
    # No CI columns unless they were asked for.
    assert "CI of Delta" not in out
    assert "paired bootstrap" not in out


def test_cli_report_leave_one_out_flags_outliers(capsys):
    """The fixture's planted spammer and random clicker are marked in the table."""
    exit_code = main(["report", str(OUTLIERS_CSV), "--leave-one-out"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    lines = captured.out.split("Leave-One-Annotator-Out:\n")[1].splitlines()

    assert lines[0].endswith("| Score | Outlier")
    assert lines[2].startswith("  spammer        |   120 |")
    assert lines[2].endswith("| yes")
    assert lines[3].startswith("  random_clicker |   120 |")
    assert lines[3].endswith("| yes")
    # The 8 good annotators are listed without a flag.
    assert [line.split("|")[0].strip() for line in lines[4:12]] == [
        f"good_{i}" for i in (1, 8, 4, 2, 7, 6, 5, 3)
    ]
    assert not any(line.rstrip().endswith("| yes") for line in lines[4:12])
    assert "Only" not in captured.out  # 10 annotators, so no reliability warning


def test_cli_report_outlier_threshold(capsys):
    """--outlier-threshold moves the bar: at 15 only the spammer (16.21) is flagged."""
    args = ["report", str(OUTLIERS_CSV), "--leave-one-out", "--outlier-threshold"]

    assert main([*args, "15"]) == 0
    out = capsys.readouterr().out
    assert "  spammer        |   120 |    0.4571 | +0.0994 | 16.21 | yes" in out
    assert "  random_clicker |   120 |    0.4391 | +0.0814 | 13.75 |\n" in out
    assert "flagged above 15 " in out


def test_cli_report_mad_zero_ignores_threshold(tmp_path, capsys):
    """Even a tiny threshold flags nobody when the MAD is 0."""
    csv_file = tmp_path / "loo.csv"
    csv_file.write_text(LOO_CSV, encoding="utf-8")

    args = ["report", str(csv_file), "--leave-one-out", "--outlier-threshold"]
    assert main([*args, "0.001"]) == 0
    out = capsys.readouterr().out
    assert "| yes" not in out
    assert out.count("| not proven") == 3


def test_cli_report_rejects_bad_outlier_threshold(tmp_path, capsys):
    csv_file = tmp_path / "loo.csv"
    csv_file.write_text(LOO_CSV, encoding="utf-8")

    exit_code = main(
        ["report", str(csv_file), "--leave-one-out", "--outlier-threshold", "0"]
    )

    assert exit_code == 2
    assert "must be greater than 0" in capsys.readouterr().err


def test_cli_report_leave_one_out_ci_adds_interval_columns(tmp_path, capsys):
    """--leave-one-out-ci keeps the score columns and adds the bootstrap ones."""
    csv_file = tmp_path / "loo.csv"
    csv_file.write_text(LOO_CSV, encoding="utf-8")

    exit_code = main(["report", str(csv_file), "--leave-one-out-ci"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    lines = captured.out.split("Leave-One-Annotator-Out:\n")[1].splitlines()
    assert lines[0] == (
        "  Annotator | Items | Kappa w/o |   Delta | Score | Outlier    "
        "| 95% CI of Delta    | Effect"
    )
    assert lines[2].startswith(
        "  A3        |     4 |    1.0000 | +0.6667 |       | not proven | ["
    )
    assert "[" in lines[2]
    assert lines[-1] == (
        "  CI: paired bootstrap, Bonferroni-adjusted for 3 annotators "
        "(each interval at 98.33%)"
    )


def test_cli_report_without_flag_has_no_leave_one_out(tmp_path, capsys):
    """The analysis is opt-in; the default text and JSON output are unchanged."""
    csv_file = tmp_path / "loo.csv"
    csv_file.write_text(LOO_CSV, encoding="utf-8")

    assert main(["report", str(csv_file)]) == 0
    assert "Leave-One-Annotator-Out" not in capsys.readouterr().out

    assert main(["report", str(csv_file), "--format", "json"]) == 0
    assert "leave_one_annotator_out" not in json.loads(capsys.readouterr().out)


def test_cli_report_leave_one_out_json(tmp_path, capsys):
    csv_file = tmp_path / "loo.csv"
    csv_file.write_text(LOO_CSV, encoding="utf-8")

    exit_code = main(["report", str(csv_file), "--format", "json", "--leave-one-out"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    loo = json.loads(captured.out)["leave_one_annotator_out"]
    assert loo["full_kappa"] == pytest.approx(1 / 3)
    assert [a["annotator_id"] for a in loo["annotators"]] == ["A3", "A1", "A2"]

    assert loo["delta_median"] == pytest.approx(-1 / 3)
    assert loo["delta_mad"] == pytest.approx(0.0)
    assert loo["outlier_threshold"] == 3.5
    assert loo["mad_is_zero"] is True

    a3 = loo["annotators"][0]
    assert a3["items_rated"] == 4
    assert a3["kappa_without"] == pytest.approx(1.0)
    assert a3["delta"] == pytest.approx(2 / 3)
    assert a3["p_bar_without"] == pytest.approx(1.0)
    assert a3["p_bar_e_without"] == pytest.approx(0.5)
    # MAD is 0: no score, no flag, "not proven".
    assert a3["robust_score"] is None
    assert a3["is_outlier"] is False
    assert a3["outlier_status"] == "not_proven"
    assert a3["error"] is None
    # No bootstrap was run, so there are no intervals.
    assert a3["delta_ci"] is None
    assert a3["effect"] is None
    assert loo["ci_family_alpha"] is None
    assert loo["ci_num_comparisons"] is None


def test_cli_report_leave_one_out_ci_json(tmp_path, capsys):
    csv_file = tmp_path / "loo.csv"
    csv_file.write_text(LOO_CSV, encoding="utf-8")

    exit_code = main(
        ["report", str(csv_file), "--format", "json", "--leave-one-out-ci"]
    )

    assert exit_code == 0
    loo = json.loads(capsys.readouterr().out)["leave_one_annotator_out"]
    # Bonferroni: 3 annotators share the 5% error rate.
    assert loo["ci_family_alpha"] == 0.05
    assert loo["ci_num_comparisons"] == 3

    a3 = loo["annotators"][0]
    assert a3["delta_ci"]["alpha"] == pytest.approx(0.05 / 3)
    assert a3["delta_ci"]["method"] == "bootstrap"
    assert a3["delta_ci"]["lower"] <= a3["delta_ci"]["upper"]
    assert a3["effect"] in {"lowers_kappa", "raises_kappa", "no_clear_effect"}


def test_cli_report_leave_one_out_json_flags_outliers(capsys):
    """The fixture's outliers come through as is_outlier in JSON."""
    assert (
        main(["report", str(OUTLIERS_CSV), "--format", "json", "--leave-one-out"]) == 0
    )
    loo = json.loads(capsys.readouterr().out)["leave_one_annotator_out"]

    flagged = [a["annotator_id"] for a in loo["annotators"] if a["is_outlier"]]
    assert flagged == ["spammer", "random_clicker"]
    assert loo["annotators"][0]["robust_score"] > 3.5
    assert loo["mad_is_zero"] is False
    statuses = {a["annotator_id"]: a["outlier_status"] for a in loo["annotators"]}
    assert statuses["spammer"] == "outlier"
    assert statuses["good_1"] == "not_flagged"


def test_cli_report_leave_one_out_undefined_rows(tmp_path, capsys):
    """Annotators whose removal leaves kappa undefined show N/A and the reason."""
    csv_file = tmp_path / "partial.csv"
    csv_file.write_text(
        "item_id,annotator_id,is_gold,label\n"
        "i1,A,false,x\ni1,B,false,x\ni1,C,false,y\n"
        "i2,A,false,y\ni2,C,false,y\ni2,D,false,y\n"
        "i3,A,false,x\ni3,B,false,x\ni3,D,false,x\n"
        "i4,A,false,y\ni4,B,false,x\ni4,C,false,y\n",
        encoding="utf-8",
    )

    exit_code = main(["report", str(csv_file), "--leave-one-out"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "  B         |     3 |       N/A |" in out
    assert (
        "  B: Rated 3 of 4 items; without them, items have different numbers "
        "of ratings." in out
    )


def test_cli_report_leave_one_out_when_kappa_undefined(tmp_path, capsys):
    """Varying ratings per item: the section shows N/A instead of failing."""
    csv_file = tmp_path / "varying.csv"
    csv_file.write_text(
        "item_id,annotator_id,is_gold,label\n"
        "i1,A,false,x\ni1,B,false,x\ni1,C,false,y\n"
        "i2,A,false,y\ni2,B,false,y\n",
        encoding="utf-8",
    )

    assert main(["report", str(csv_file), "--leave-one-out"]) == 0
    out = capsys.readouterr().out
    assert "Leave-One-Annotator-Out: N/A (Varying number of ratings per item." in out

    assert main(["report", str(csv_file), "--format", "json", "--leave-one-out"]) == 0
    assert json.loads(capsys.readouterr().out)["leave_one_annotator_out"] is None


# Spam check tests


def test_cli_report_spam_check_text(capsys):
    """--spam-check adds a table, highest score first, with flags and reasons."""
    exit_code = main(["report", str(SPAM_KNOWN_CSV), "--spam-check"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    lines = captured.out.split("Spam Check:\n")[1].splitlines()

    assert lines[0] == (
        "  Annotator           | Labels | Agreement | Speed | Variety |       Gold "
        "| Score | Spammer | Confidence | Reasons"
    )
    assert lines[2] == (
        "  constant_spammer    |    150 |      0.00 |  1.06 |    0.00 |  0.00 (15) "
        "|  1.00 | yes     |        99% | agreement, variety, gold"
    )
    flagged = [line.split("|")[0].strip() for line in lines[2:14] if "| yes " in line]
    assert flagged == ["constant_spammer", "biased_clicker", "random_clicker_fast"]
    # Only flagged annotators get a confidence.
    confidences = [line.split("|")[8].strip() for line in lines[2:14]]
    assert confidences == ["99%", "98%", "98%"] + [""] * 9
    assert lines[-3].startswith("  Agreement, Speed and Variety are relative")
    assert lines[-2].startswith("  Flagged at Score >= 0.5")
    assert lines[-1] == (
        "  Confidence: chance a flagged annotator really is a spammer, assuming "
        "15% of annotators are (calibration data)"
    )


def test_cli_report_spam_check_json(capsys):
    exit_code = main(
        ["report", str(SPAM_KNOWN_CSV), "--format", "json", "--spam-check"]
    )

    assert exit_code == 0
    spam = json.loads(capsys.readouterr().out)["spam_check"]
    assert len(spam) == 12

    top = spam[0]
    assert top["annotator_id"] == "constant_spammer"
    assert top["status"] == "spammer"
    assert top["is_spammer"] is True
    assert top["score"] == pytest.approx(1.0)
    assert top["confidence"] == pytest.approx(0.9992)
    assert top["confidence_percent"] == 99
    assert top["reasons"] == ["agreement", "variety", "gold"]
    assert set(top["suspicion"]) == {"agreement", "speed", "variety", "gold"}
    assert top["signals"]["gold_items"] == 15
    assert top["signals"]["entropy_bits"] == 0.0
    assert [a["annotator_id"] for a in spam if a["is_spammer"]] == [
        "constant_spammer",
        "biased_clicker",
        "random_clicker_fast",
    ]
    for a in spam:
        assert (a["confidence"] is not None) == a["is_spammer"]
        assert (a["confidence_percent"] is not None) == a["is_spammer"]


def test_cli_report_without_spam_check_flag(capsys):
    assert main(["report", str(SPAM_KNOWN_CSV), "--format", "json"]) == 0
    assert "spam_check" not in json.loads(capsys.readouterr().out)

    assert main(["report", str(SPAM_KNOWN_CSV)]) == 0
    assert "Spam Check" not in capsys.readouterr().out


def test_cli_report_spam_check_works_without_constant_ratings(tmp_path, capsys):
    """Unlike Fleiss' kappa, the spam check doesn't need equal ratings per item."""
    rows = ["item_id,annotator_id,is_gold,label"]
    for i in range(30):
        for a in ("A", "B", "C"):
            rows.append(f"i{i},{a},false,{'x' if i % 2 else 'y'}")
        if i % 3:
            rows.append(f"i{i},D,false,x")
    csv_file = tmp_path / "uneven.csv"
    csv_file.write_text("\n".join(rows) + "\n", encoding="utf-8")

    assert main(["report", str(csv_file), "--spam-check"]) == 0
    out = capsys.readouterr().out
    assert "Fleiss' Kappa: N/A (Varying number of ratings per item." in out
    spam_lines = out.split("Spam Check:\n")[1].splitlines()
    assert spam_lines[2].startswith("  D ")
    assert "| yes " in spam_lines[2]


def test_cli_report_spam_base_rate_lowers_confidence(capsys):
    """Fewer expected spammers means less confidence in each flag."""
    args = ["report", str(SPAM_KNOWN_CSV), "--format", "json", "--spam-check"]

    assert main(args) == 0
    default = {
        a["annotator_id"]: a for a in json.loads(capsys.readouterr().out)["spam_check"]
    }
    assert main([*args, "--spam-base-rate", "0.02"]) == 0
    rare = {
        a["annotator_id"]: a for a in json.loads(capsys.readouterr().out)["spam_check"]
    }

    assert (
        rare["biased_clicker"]["confidence"] < default["biased_clicker"]["confidence"]
    )
    assert rare["biased_clicker"]["confidence_percent"] == 89
    # The flags themselves don't change.
    assert [a for a in rare if rare[a]["is_spammer"]] == [
        a for a in default if default[a]["is_spammer"]
    ]

    assert (
        main(
            ["report", str(SPAM_KNOWN_CSV), "--spam-check", "--spam-base-rate", "0.02"]
        )
        == 0
    )
    assert "assuming 2% of annotators are (--spam-base-rate)" in capsys.readouterr().out


@pytest.mark.parametrize("value", ["0", "1", "1.5", "abc"])
def test_cli_report_rejects_bad_spam_base_rate(value, capsys):
    exit_code = main(
        ["report", str(SPAM_KNOWN_CSV), "--spam-check", "--spam-base-rate", value]
    )
    assert exit_code == 2
    assert "--spam-base-rate" in capsys.readouterr().err


# Bad actor tests


def test_cli_report_list_bad_actors(capsys):
    """
    --list-bad-actors ends the report with a table of who and why, metrics
    unchanged. Someone flagged by both checks gets a row per reason.
    """
    assert main(["report", str(OUTLIERS_CSV)]) == 0
    plain = capsys.readouterr().out

    assert main(["report", str(OUTLIERS_CSV), "--list-bad-actors"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    listed = captured.out

    # The report itself is untouched; the list is appended.
    assert listed.startswith(plain)
    assert listed[len(plain) :] == (
        "Bad Actors (2):\n"
        "  Annotator      | Reason\n"
        "  ---------------+---------------------------------------------------------------\n"
        "  spammer        | spam check (99% confident): agreement, variety, speed\n"
        "  spammer        | leave-one-out outlier (score 16.2): kappa +0.0994 without them\n"
        "  random_clicker | spam check (89% confident): agreement\n"
        "  random_clicker | leave-one-out outlier (score 13.8): kappa +0.0814 without them\n"
    )


def test_cli_report_exclude_bad_actors(capsys):
    """--exclude-bad-actors reports on what is left, and says what it dropped."""
    assert main(["report", str(OUTLIERS_CSV), "--exclude-bad-actors"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    out = captured.out

    # 10 annotators and kappa 0.3577 with everyone; 8 and 0.5773 without.
    assert "Annotators: 8" in out
    assert "Items: 120" in out
    assert "Labels Provided: 960" in out
    assert "Fleiss' Kappa: 0.5773" in out
    assert "Bad Actors (2, excluded from this report):" in out
    assert (
        "  spammer        | spam check (99% confident): agreement, variety, speed"
        in out
    )
    assert "  Fleiss' Kappa: 0.3577 with them, 0.5773 without" in out


def test_cli_report_exclude_bad_actors_recomputes_the_other_sections(capsys):
    """The leave-one-out and spam sections describe the cleaned data too."""
    exit_code = main(
        [
            "report",
            str(OUTLIERS_CSV),
            "--exclude-bad-actors",
            "--leave-one-out",
            "--spam-check",
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    loo = out.split("Leave-One-Annotator-Out:\n")[1].split("Spam Check:")[0]
    spam = out.split("Spam Check:\n")[1].split("Bad Actors")[0]

    def annotators(table: str) -> set[str]:
        return {
            line.split("|")[0].strip()
            for line in table.splitlines()[2:]
            if "|" in line and not line.startswith("  -")
        }

    # Both sections now describe the 8 good annotators only.
    assert annotators(loo) >= {f"good_{i}" for i in range(1, 9)}
    assert not annotators(loo) & {"spammer", "random_clicker"}
    assert not annotators(spam) & {"spammer", "random_clicker"}
    assert "| yes " not in spam
    assert "Bad Actors (2, excluded from this report):" in out


def test_cli_report_no_bad_actors(capsys):
    """Clean data says so, and excluding changes nothing."""
    assert main(["report", str(SAMPLE_CSV), "--list-bad-actors"]) == 0
    assert capsys.readouterr().out.endswith("Bad Actors: None\n")

    assert main(["report", str(SAMPLE_CSV), "--exclude-bad-actors"]) == 0
    excluded = capsys.readouterr().out
    assert main(["report", str(SAMPLE_CSV), "--list-bad-actors"]) == 0
    assert excluded == capsys.readouterr().out


def test_cli_report_exclude_everyone_is_an_error(tmp_path, capsys):
    """Three annotators, each stuck on their own label: nothing would be left."""
    rows = ["item_id,annotator_id,is_gold,label"]
    for i in range(40):
        for annotator, label in (("A", "x"), ("B", "y"), ("C", "z")):
            rows.append(f"i{i},{annotator},false,{label}")
    csv_file = tmp_path / "allbad.csv"
    csv_file.write_text("\n".join(rows) + "\n", encoding="utf-8")

    exit_code = main(["report", str(csv_file), "--exclude-bad-actors"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "every annotator with labels is a bad actor (A, B, C)" in captured.err

    # Listing them still works.
    assert main(["report", str(csv_file), "--list-bad-actors"]) == 0
    table = capsys.readouterr().out.split("Bad Actors (3):\n")[1].splitlines()
    assert [row.split("|")[0].strip() for row in table[2:]] == ["A", "B", "C"]


def test_cli_report_bad_actors_json(capsys):
    assert (
        main(["report", str(OUTLIERS_CSV), "--format", "json", "--exclude-bad-actors"])
        == 0
    )
    data = json.loads(capsys.readouterr().out)

    assert data["annotators"] == 8
    assert data["fleiss_kappa"] == pytest.approx(0.5773, abs=5e-4)
    bad = data["bad_actors"]
    assert bad["excluded"] is True
    assert bad["fleiss_kappa_with_bad_actors"] == pytest.approx(0.3577, abs=5e-4)

    spammer = bad["annotators"][0]
    assert spammer["annotator_id"] == "spammer"
    assert spammer["sources"] == ["spam_check", "leave_one_out"]
    assert len(spammer["reasons"]) == 2
    assert spammer["spam_confidence_percent"] == 99
    assert spammer["spam_reasons"] == ["agreement", "variety", "speed"]
    assert spammer["outlier_score"] == pytest.approx(16.2, abs=0.1)
    assert spammer["kappa_delta"] == pytest.approx(0.0994, abs=5e-4)


def test_cli_report_list_bad_actors_json_keeps_everyone(capsys):
    assert (
        main(["report", str(OUTLIERS_CSV), "--format", "json", "--list-bad-actors"])
        == 0
    )
    data = json.loads(capsys.readouterr().out)

    assert data["annotators"] == 10
    assert data["fleiss_kappa"] == pytest.approx(0.3577, abs=5e-4)
    assert data["bad_actors"]["excluded"] is False
    assert data["bad_actors"]["fleiss_kappa_with_bad_actors"] is None
    assert len(data["bad_actors"]["annotators"]) == 2


def test_cli_report_without_bad_actor_flags(capsys):
    assert main(["report", str(OUTLIERS_CSV), "--format", "json"]) == 0
    assert "bad_actors" not in json.loads(capsys.readouterr().out)

    assert main(["report", str(OUTLIERS_CSV)]) == 0
    assert "Bad Actors" not in capsys.readouterr().out


def test_cli_report_bad_actors_respects_spam_base_rate(capsys):
    """The confidence in the reason follows --spam-base-rate."""
    assert (
        main(
            [
                "report",
                str(OUTLIERS_CSV),
                "--list-bad-actors",
                "--spam-base-rate",
                "0.02",
            ]
        )
        == 0
    )
    assert "spam check (49% confident): agreement\n" in capsys.readouterr().out


def test_cli_report_bad_actor_table_columns_fit_the_longest_entries(capsys):
    """Widths follow the longest id and reason; one row per bad actor here."""
    assert main(["report", str(SPAM_KNOWN_CSV), "--list-bad-actors"]) == 0
    table = capsys.readouterr().out.split("Bad Actors (3):\n")[1].splitlines()

    assert table == [
        "  Annotator           | Reason",
        "  --------------------+-----------------------------------------------------",
        "  constant_spammer    | spam check (99% confident): agreement, variety, gold",
        "  biased_clicker      | spam check (98% confident): agreement, gold",
        "  random_clicker_fast | spam check (98% confident): agreement, gold, speed",
    ]


def test_cli_report_bad_actors_follow_outlier_threshold(capsys):
    """--outlier-threshold moves the leave-one-out bar for bad actors too."""
    args = ["report", str(OUTLIERS_CSV), "--format", "json", "--list-bad-actors"]

    assert main(args) == 0
    default = json.loads(capsys.readouterr().out)["bad_actors"]["annotators"]
    assert main([*args, "--outlier-threshold", "0.1"]) == 0
    strict = json.loads(capsys.readouterr().out)["bad_actors"]["annotators"]

    assert len(default) == 2
    extra = {a["annotator_id"] for a in strict} - {a["annotator_id"] for a in default}
    assert extra == {"good_1", "good_8"}  # scores 0.17 and 0.12
    assert all(
        a["sources"] == ["leave_one_out"] for a in strict if a["annotator_id"] in extra
    )
