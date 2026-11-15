# FLeisskit

Quality of labels remains a challenge in data annotation world. Fleisskit is a library and CLI tool for processing and analysing annotated data. It can load from JSON or CSV records. It validates the data and calculates inter-rater agreement metrics.

## Installation

If you are using `uv` (recommended), install the package in editable mode:

    uv pip install -e .

Alternatively, using standard pip:

    pip install .

## Usage

You can generate a summary report directly from the command line by pointing the `report` command at your CSV or JSON annotation file:

    uv run fleisskit report tests/data/sample.csv

**Output:**

    Report for sample.csv
    ---------------------
    Items: 50
    Annotators: 3
    Labels Provided: 150
    Percent Agreement: 62.00%
    Majority Vote Distribution:
      - negative: 13
      - neutral: 14
      - positive: 23
    Fleiss' Kappa: 0.4108
    Per-Category Kappa:
      Label    |  Kappa
      ---------+-------
      negative | 0.3864
      neutral  | 0.4048
      positive | 0.4351

You can also output the results in strict JSON format for use in other data pipelines:

    uv run fleisskit report tests/data/sample.csv --format json

### Generating sample data

The `generate` command writes a synthetic annotation file you can try the `report` command on. The format follows the file extension (`.csv` or `.json`):

    uv run fleisskit generate my_sample.csv
    uv run fleisskit generate my_sample.json

The defaults (50 items, 3 annotators, labels `positive,negative,neutral`, seed 42) recreate `tests/data/sample.csv` exactly. Every generator setting can be changed:

    uv run fleisskit generate data.json --items 200 --annotators 5 --labels cat,dog,bird --accuracy 0.7 --spammer-share 0.2 --seed 7

| Option | Default | Meaning |
|---|---|---|
| `--items` | 50 | Number of items |
| `--annotators` | 3 | Number of annotators (each rates every item) |
| `--labels` | `positive,negative,neutral` | Comma-separated labels (at least 2) |
| `--accuracy` | 0.8 | Chance a normal annotator picks the true label |
| `--spammer-share` | 0.0 | Share of annotators who always pick the same label |
| `--skip-rate` | 0.0 | Share of annotations randomly dropped |
| `--seed` | 42 | Random seed; the same seed always gives the same file |
| `--force` | off | Overwrite the output file if it already exists |

Keep `--skip-rate` at 0 if you want Fleiss' kappa in the report: skipped annotations give items different numbers of ratings, and the report will then show Fleiss' kappa as N/A.

## Metrics Included

*   **Percent Agreement**: Calculates the average fraction of agreeing annotator pairs
*   **Cohen's Kappa**: Measures agreement beyond chance between two specific annotators
*   **Majority Vote**: Determines the most frequent label for each item
*   **Fleiss' Kappa**: Measures agreement across multiple annotators
*   **Per-Category Kappa**: Breaks Fleiss' kappa down per label, showing which labels annotators struggle with

## Limitations

*   **Fleiss' Kappa**: Needs the exact same number of ratings for every single item
*   **Per-Category Kappa**: Same constant-ratings requirement; labels used for none or all of the ratings are skipped (their kappa is undefined)
*   **Percent Agreement**: Items with fewer than 2 labels are automatically skipped 
*   **Cohen's Kappa**: Fails if the two specified annotators share no common items

## Running Tests

To run the complete test suite

    uv run pytest