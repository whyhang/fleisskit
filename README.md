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
    Percent Agreement: 85.33%
    Majority Vote Distribution:
      - negative: 12
      - neutral: 8
      - positive: 30

You can also output the results in strict JSON format for use in other data pipelines:

    uv run fleisskit report tests/data/sample.csv --format json

## Metrics Included

*   **Percent Agreement**: Calculates the average fraction of agreeing annotator pairs
*   **Cohen's Kappa**: Measures agreement beyond chance between two specific annotators
*   **Majority Vote**: Determines the most frequent label for each item
*   **Fleiss' Kappa**: [TODO] Measures agreement across multiple annotators.

## Limitations

*   **Fleiss' Kappa**: Needs the exact same number of ratings for every single item
*   **Percent Agreement**: Items with fewer than 2 labels are automatically skipped 
*   **Cohen's Kappa**: Fails if the two specified annotators share no common items

## Running Tests

To run the complete test suite

    uv run pytest