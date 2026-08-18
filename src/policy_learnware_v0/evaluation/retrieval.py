"""Offline oracle-labelled retrieval metrics; never imported by the selector."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Iterable

from ..reuse.selector import SelectionResult


@dataclass(frozen=True)
class RetrievalTrial:
    expected_opaque_id: str
    selection: SelectionResult

    @property
    def correct(self) -> bool:
        return self.expected_opaque_id == self.selection.selected_opaque_id


@dataclass(frozen=True)
class RetrievalMetrics:
    trial_count: int
    correct_count: int
    accuracy: float
    mean_selected_distance: float
    mean_margin_to_second: float | None


def summarize_retrieval(trials: Iterable[RetrievalTrial]) -> RetrievalMetrics:
    values = tuple(trials)
    if not values:
        raise ValueError("at least one retrieval trial is required")
    selected_distances = [trial.selection.sorted_distances[0].distance for trial in values]
    margins = [
        trial.selection.sorted_distances[1].distance
        - trial.selection.sorted_distances[0].distance
        for trial in values
        if len(trial.selection.sorted_distances) > 1
    ]
    correct_count = sum(trial.correct for trial in values)
    return RetrievalMetrics(
        trial_count=len(values),
        correct_count=correct_count,
        accuracy=correct_count / len(values),
        mean_selected_distance=fmean(selected_distances),
        mean_margin_to_second=fmean(margins) if margins else None,
    )
