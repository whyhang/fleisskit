import csv
import random
from pathlib import Path
from typing import TypedDict

from fleisskit.models import Annotation, AnnotationCollection


class AnnotatorProfile(TypedDict):
    is_spammer: bool
    spammer_label: str | None


def generate_synthetic_data(
    num_items: int,
    num_annotators: int,
    labels: list[str],
    accuracy: float = 0.8,
    spammer_share: float = 0.0,
    skip_rate: float = 0.0,
    seed: int = 42,
    output_csv: str | Path | None = None,
) -> tuple[AnnotationCollection, dict[str, str]]:
    """
    Generates realistic synthetic annotations

    Constraints:
    - Each item gets a true label.
    - Normal annotators apply the true label based on `accuracy`, or a random wrong one.
    - Spammers always pick the same label extremely fast (ignores accuracy).
    - Annotations are randomly dropped based on `skip_rate`.
    - ALL randomness is governed strictly by the provided seed.
    """
    if len(labels) < 2:
        raise ValueError("Must provide at least 2 labels to generate meaningful data.")

    rng = random.Random(seed)

    # asign true labels
    true_labels: dict[str, str] = {}
    for i in range(num_items):
        item_id = f"item_{i}"
        true_labels[item_id] = rng.choice(labels)

    # Define annotator profiles (genuine /  spammers)
    num_spammers = int(num_annotators * spammer_share)
    annotator_profiles: dict[str, AnnotatorProfile] = {}

    for i in range(num_annotators):
        a_id = f"annotator_{i}"
        is_spammer = i < num_spammers

        # Spammers pick one random label and stick to it permanently
        spammer_label = rng.choice(labels) if is_spammer else None

        annotator_profiles[a_id] = {
            "is_spammer": is_spammer,
            "spammer_label": spammer_label,
        }
    # calculate exact pairs to drop according to skiprate
    all_pairs = [
        (item_id, a_id) for item_id in true_labels for a_id in annotator_profiles
    ]
    num_to_drop = int(len(all_pairs) * skip_rate)
    pairs_to_drop = set(rng.sample(all_pairs, num_to_drop))

    # Generate Annotations
    annotations = []

    for item_id, t_label in true_labels.items():
        for a_id, profile in annotator_profiles.items():
            # Apply skip logic
            if (item_id, a_id) in pairs_to_drop:
                continue

            is_spammer = profile["is_spammer"]

            # Spammer behavior vs Normal behavior
            if is_spammer:
                chosen_label = profile["spammer_label"]
                duration = rng.uniform(0.1, 2.2)  # Spammers are fast
            else:
                duration = rng.uniform(5.0, 45.0)  # Normal annotator duration

                # Accuracy check
                if rng.random() < accuracy:
                    chosen_label = t_label
                else:
                    # choosing a wrong label
                    wrong_labels = [L for L in labels if L != t_label]
                    chosen_label = rng.choice(wrong_labels)

            ann = Annotation(
                item_id=item_id,
                annotator_id=a_id,
                is_gold=False,
                label=chosen_label,
                confidence=rng.uniform(0.4, 1.0),
                duration_sec=duration,
            )
            annotations.append(ann)

    collection = AnnotationCollection(annotations)

    # save to CSV [optional]
    if output_csv:
        path = Path(output_csv)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "item_id",
                    "annotator_id",
                    "is_gold",
                    "label",
                    "confidence",
                    "duration_sec",
                    "annotated_at",
                    "batch_id",
                    "gold_label",
                ]
            )
            for ann in collection.annotations:
                writer.writerow(
                    [
                        ann.item_id,
                        ann.annotator_id,
                        str(ann.is_gold).lower(),
                        ann.label or "",
                        round(ann.confidence, 4) if ann.confidence else "",
                        round(ann.duration_sec, 2) if ann.duration_sec else "",
                        ann.annotated_at.isoformat() if ann.annotated_at else "",
                        ann.batch_id or "",
                        ann.gold_label or "",
                    ]
                )

    return collection, true_labels
