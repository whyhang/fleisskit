# Fleisskit

A library and CLI for checking the quality of annotated data. It loads CSV or JSON annotations, validates them, and measures inter-rater agreement. It can also find annotators who are dragging that agreement down, and report on the data without them.

## Installation

    uv pip install -e .      # or: pip install .

## Usage

    uv run fleisskit report tests/data/sample.csv

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

Options for `report`. Each section is added to the end of the report, and to the JSON under the key shown:

| Flag | Adds | JSON key |
|---|---|---|
| `--format json` | the whole report as JSON | |
| `--leave-one-out` | kappa without each annotator, with outliers flagged | `leave_one_annotator_out` |
| `--leave-one-out-ci` | the same, plus a bootstrap CI per annotator (slower) | `leave_one_annotator_out` |
| `--outlier-threshold N` | the bar for a leave-one-out outlier (default 3.5) | |
| `--spam-check` | a spam verdict per annotator, with confidence | `spam_check` |
| `--spam-base-rate P` | the expected share of spammers, which adjusts the confidence | |
| `--list-bad-actors` | a table of bad actors and why | `bad_actors` |
| `--exclude-bad-actors` | the whole report on the data without bad actors, plus the table | `bad_actors` |

**Metrics:**
- **Percent agreement:** the mean share of agreeing annotator pairs per item.
- **Majority vote:** ties go to the alphabetically first label.
- **Fleiss' kappa:** agreement across all annotators, corrected for chance.
- **Per-category kappa:** Fleiss' kappa per label. The overall kappa is their p(1−p)-weighted mean.
- **Cohen's kappa:** for a pair of annotators, available from Python.

## Leave-one-annotator-out

One outlier annotator can drag kappa down for everyone. This recomputes kappa with each annotator removed in turn:

    uv run fleisskit report tests/data/outliers.csv --leave-one-out

    Leave-One-Annotator-Out:
      Annotator      | Items | Kappa w/o |   Delta | Score | Outlier
      ---------------+-------+-----------+---------+-------+--------
      spammer        |   120 |    0.4571 | +0.0994 | 16.21 | yes
      random_clicker |   120 |    0.4391 | +0.0814 | 13.75 | yes
      ...  (8 good annotators, Delta -0.02 to -0.04, Score -2.28 to 0.17)
      Score: 0.6745 * (Delta - median) / MAD, flagged above 3.5 (median -0.0197, MAD 0.0050)

    Delta = kappa without the annotator - kappa with everyone      (> 0: they pull kappa down)
    MAD   = median(|Delta - median(Delta)|)
    Score = 0.6745 * (Delta - median(Delta)) / MAD                  (flagged when > 3.5)

- **Why median and MAD?** The outliers being hunted would drag a mean and standard deviation towards themselves, hiding the gap. The median and MAD ignore them. The factor 0.6745 makes the score read like a z-score.
- **Only high scores are flagged.** A very negative score means the annotator agrees with the group *more* than most.
- **If the MAD is 0, no one is flagged.** This happens when more than half the deltas are identical. The score would divide by zero, so every annotator shows `not proven` (JSON: `"outlier_status": "not_proven"`, `"mad_is_zero": true`).
- **`--leave-one-out-ci`** adds a paired bootstrap interval for each Delta. It uses 1,000 item resamples (seed 42), computing kappa with and without the annotator on the same items. The intervals are Bonferroni-adjusted, so the 95% covers all annotators at once. **Effect** reads the interval: `lowers kappa` if it is all above 0, `raises kappa` if all below, otherwise `no clear effect`. "Raises" is relative to this group: with an outlier present, everyone else looks good by comparison.
- **It needs enough annotators.** Below 5, the MAD rests on one or two numbers, and the report warns about this. Even with enough, on clean simulated data this rule flags someone in about 1 run in 10.
- **JSON extras:** each annotator's observed (`p_bar_without`) and chance (`p_bar_e_without`) agreement, so you can see which half of kappa moved.

## Spam check

Finds annotators who aren't really doing the task, using four separate signals and one combiner. Unlike Fleiss' kappa, it works even when items have different numbers of ratings.

    uv run fleisskit report tests/data/spam_known.csv --spam-check

    Spam Check:
      Annotator           | Labels | Agreement | Speed | Variety |       Gold | Score | Spammer | Confidence | Reasons
      --------------------+--------+-----------+-------+---------+------------+-------+---------+------------+-------------------------
      constant_spammer    |    150 |      0.00 |  1.06 |    0.00 |  0.00 (15) |  1.00 | yes     |        99% | agreement, variety, gold
      biased_clicker      |    150 |      0.01 |  0.11 |    0.76 |  0.04 (15) |  0.70 | yes     |        98% | agreement, gold
      random_clicker_fast |    150 |     -0.09 |  0.09 |    0.99 | -0.15 (15) |  0.63 | yes     |        98% | agreement, gold, speed
      random_clicker_slow |    150 |      0.07 |  1.84 |    1.00 | -0.07 (15) |  0.43 |         |            | gold, agreement
      ...  (8 honest annotators, including a fast one and a weak one, Score 0.00 to 0.17)

**The four signals.** Each is its own function in `fleisskit.spam`:

| Signal | Measures | Compared with |
|---|---|---|
| `agreement_with_others` | kappa against the **majority vote of the other annotators** on shared items | the group median (1.00 = typical) |
| `annotation_speed` | median seconds per annotation | the group median |
| `label_variety` | entropy of the annotator's labels | the others' majority labels **on the same items** |
| `gold_accuracy` | kappa against `gold_label` on gold items | nothing: absolute |

- **Chance-corrected.** Agreement and gold are kappas, so always picking the most common label scores 0, not "70% right".
- **Relative.** A hard task lowers everyone's agreement and long items slow everyone down, so agreement and speed are compared with the group.
- **Fair to one-sided batches.** An honest annotator given nearly all-positive items isn't penalised for answering "positive". Variety is only judged once the others' majority used another label on at least 5 of the annotator's items.

**The combiner** runs once per annotator:

    suspicion = clip(1 - value / ok, 0, 1)                         (0 = what a pure spammer scores)
    score     = 1 - product over signals of (1 - weight * suspicion)

- **A noisy-OR.** Each signal can only *add* suspicion, since each catches only some kinds of spammer. With a weighted average, a slow random clicker's normal speed would cancel out its damning agreement.
- **When someone is flagged:** at `score >= 0.5`, with at least one reason besides speed. Being quick is never enough on its own.
- **When someone isn't judged:** under 10 labels, or only speed measurable, gives `n/a`.

**Confidence.** The score is not a probability. Each flagged annotator also gets the chance they really are a spammer, learned from labelled data:
- **How it was fitted:** on 600 simulated datasets not used for tuning. Flagged annotators were grouped by score (at least 50 per group), groups were merged until confidence rises with score, and each group's rate was smoothed as (spammers + 0.5) / (annotators + 1).
- **How it's shown:** rounded down and capped at 99%.
- **How it held up on validation data:** predicted 89.2 / 95.1 / 98.6 / 99.9% against observed 80.6 / 95.5 / 97.1 / 99.7% (31, 22, 105 and 355 flagged). The lowest step runs high, within noise.
- **It assumes about 15% of annotators are spammers, like the calibration data.** `--spam-base-rate 0.02` rescales it by Bayes' rule on the odds: `biased_clicker` drops from 98% to 89%. The flags don't change.

**Tuning.** The weights, `ok` points and threshold in `DEFAULT_PARAMS` were fitted on known data. `uv run python -m fleisskit.spam_tuning` reproduces them and the calibration.
- **The simulation:** honest annotators (some fast, some weak: the cases that must *not* be flagged) alongside constant spammers, random clickers and biased clickers. Spammers are fast 60% of the time. Datasets are dense or sparse (3–5 ratings per item), with 0–30% spammers, 0–20% gold items, missing durations and skewed labels.
- **The search:** block coordinate descent, searching each signal's weight and `ok` point together. It maximises F0.5 on 300 training datasets; F0.5 counts precision twice as much as recall, because wrongly calling an honest annotator a spammer is worse than missing one.
- **Results on 300 validation datasets:** 99% of constant spammers, 93% of biased clickers and 25% of random clickers flagged, against 0.3% of honest annotators. Precision 0.979, recall 0.716.

Re-tuning for a different objective moves along the trade-off:

| Objective | Precision | Recall | Honest flagged | Random clickers caught |
|---|---|---|---|---|
| **F0.5 (default)** | 0.979 | 0.716 | 11 / 3877 | 61 / 242 |
| F1 | 0.948 | 0.779 | 30 / 3877 | 102 / 242 |
| F2 | 0.735 | 0.896 | 227 / 3877 | 182 / 242 |

## Bad actors

A bad actor is an annotator flagged by **either** the spam check or the leave-one-out outlier score. The spam check works on any data and says what is wrong. The outlier score needs the same number of ratings per item, and says someone is out of line with this group. `--list-bad-actors` appends the table; `--exclude-bad-actors` reports on the data without them and ends with the same table:

    uv run fleisskit report tests/data/outliers.csv --exclude-bad-actors

    ...
    Fleiss' Kappa: 0.5773
    ...
    Bad Actors (2, excluded from this report):
      Annotator      | Reason
      ---------------+---------------------------------------------------------------
      spammer        | spam check (99% confident): agreement, variety, speed
      spammer        | leave-one-out outlier (score 16.2): kappa +0.0994 without them
      random_clicker | spam check (89% confident): agreement
      random_clicker | leave-one-out outlier (score 13.8): kappa +0.0814 without them
      Fleiss' Kappa: 0.3577 with them, 0.5773 without

- **Found on the whole dataset, then removed.** An annotator can only be judged against their peers while those peers are still there.
- **Everything above the table describes the remaining data,** including any `--leave-one-out` and `--spam-check` sections.
- **Rows:** someone flagged by both checks gets a row per reason.
- **If every annotator with labels is a bad actor,** excluding leaves nothing, so it exits 1 with a message. `--list-bad-actors` still works.
- **`--spam-base-rate` flows into the quoted confidence.** At 2%, `random_clicker` reads `49% confident`: still flagged, but more likely honest than not.
- **JSON:** `bad_actors` holds `excluded`, `fleiss_kappa_with_bad_actors` (set when they were excluded), and per annotator `sources`, `reasons`, `spam_confidence_percent`, `outlier_score` and `kappa_delta`.

## Python API

    from fleisskit.io import load_csv
    from fleisskit.fleiss import analyze_fleiss          # kappa, per-category kappa, optional CIs
    from fleisskit.influence import leave_one_annotator_out
    from fleisskit.spam import detect_spammers
    from fleisskit.bad_actors import find_bad_actors, exclude_annotators

    collection = load_csv("data.csv")
    bad = find_bad_actors(collection, base_rate=0.05)
    clean = exclude_annotators(collection, {b.annotator_id for b in bad})

If you know some annotators' roles, you can tune and calibrate the spam check on them:

    from dataclasses import replace
    from fleisskit.spam_tuning import LabelledDataset, fit_confidence_calibration, tune_combiner

    result = tune_combiner([LabelledDataset(collection, archetypes={"ann_7": "constant_spammer"})])
    calibration = fit_confidence_calibration(other_labelled_datasets, result.params)
    verdicts = detect_spammers(collection, replace(result.params, calibration=calibration))

- **Unlisted annotators count as honest.**
- **Re-tuned parameters have no confidence** until calibrated on *other* labelled data, with at least 50 flagged annotators.

## Sample data

`generate` writes a synthetic file; the format follows the extension. The defaults recreate `tests/data/sample.csv` exactly:

    uv run fleisskit generate data.json --items 200 --annotators 5 --labels cat,dog,bird --accuracy 0.7 --spammer-share 0.2 --seed 7

| Option | Default | Meaning |
|---|---|---|
| `--items` / `--annotators` | 50 / 3 | every annotator rates every item |
| `--labels` | `positive,negative,neutral` | at least 2, comma-separated |
| `--accuracy` | 0.8 | chance a normal annotator picks the true label |
| `--spammer-share` | 0.0 | share of annotators who always pick one label |
| `--skip-rate` | 0.0 | share of annotations dropped (over 0, Fleiss' kappa becomes N/A) |
| `--seed` / `--force` | 42 / off | reproducibility / overwrite an existing file |

Test fixtures:

| File | Contents |
|---|---|
| `sample.csv` | 50 items, 3 honest annotators (the `generate` defaults) |
| `outliers.csv` | 120 items, 8 good annotators, a constant spammer and a random clicker |
| `spam_known.csv` | 150 items (15 gold), 12 annotators named by their true role. It is exactly `simulate_labelled_dataset(..., seed=2)` (see `tests/test_spam_tuning.py`) |

## Limitations

- **Fleiss' and per-category kappa need the same number of ratings on every item.** Per-category kappa skips labels used for none or all ratings. Percent agreement skips items with fewer than 2 labels. Cohen's kappa needs shared items.
- **Leave-one-out needs each annotator to have rated every item, with at least 3 ratings per item;** otherwise it shows N/A. Outliers who agree with each other partly mask each other.
- **Both checks measure agreement, not correctness.** An expert who rightly disagrees with a sloppy majority looks like an outlier. Both also assume an honest majority: if half the annotators are spammers, "typical" is no longer honest.
- **Slow random clickers are the spam check's blind spot.** Without much gold, they look like weak but honest annotators (`random_clicker_slow` above scores 0.43, unflagged). Gold items fix this. Gold gets only weight 0.1 because most simulated annotators have little gold, so re-tune if you have plenty.
- **The tuning and confidence come from simulated annotators.** A 98% means "98% in a world like the simulation", and spam strategies it doesn't model (e.g. copying a neighbour) may go uncaught.
- **A flag is a prompt to review someone's work, not proof.**

## Running tests

    uv run pytest
