from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Annotation:
    # Required fields
    item_id: str
    annotator_id: str
    is_gold: bool
    label: str | None = None
    confidence: float | None = None
    duration_in_seconds: float | None = None
    annotated_at: datetime | None = None
    batch_id: str | None = None
    gold_label: str | None = None

    def __post_init__(self):
        # Missing => None
        for field_name, value in [
            ("label", self.label),
            ("batch_id", self.batch_id),
            ("gold_label", self.gold_label),
        ]:
            if value == "":
                raise ValueError(
                    f"{field_name} cannot be an empty string; use None if missing."
                )

        # Validate confidence score
        if self.confidence is not None and (not (0.0 <= self.confidence <= 1.0)):
            raise ValueError(
                f"Confidence must be between 0 and 1, got {self.confidence}"
            )

        # Validate duration
        if self.duration_in_seconds is not None and self.duration_in_seconds < 0:
            raise ValueError(
                f"Duration cannot be negative, got {self.duration_in_seconds}"
            )

        # Validate gold_label is present when the is_gold is True
        if self.is_gold and self.gold_label is None:
            raise ValueError("gold_label is required when is_gold is True")


class AnnotationCollection:
    """Collection of Annotation class defined above"""

    def __init__(self, annotations: list[Annotation]):
        self.annotations = annotations

    def get_by_annotator(self, annotator_id: str) -> list[Annotation]:
        return [ann for ann in self.annotations if ann.annotator_id == annotator_id]
