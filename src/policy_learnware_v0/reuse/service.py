"""Thin target-side service that stops at selection, before deployment."""

from __future__ import annotations

from typing import Any

from ..pool.learnware import LearnwarePool
from .selector import NearestSpecSelector, SelectionResult


class ReuseService:
    """Keep retrieval and deployment as two explicit phases."""

    def __init__(self, pool: LearnwarePool) -> None:
        self._selector = NearestSpecSelector(pool)

    def select_from_empirical_kme(
        self,
        empirical_kme: Any,
        *,
        target_dataset_digest: str,
        probe_episode_count: int,
        probe_steps: int,
    ) -> SelectionResult:
        return self._selector.select(
            empirical_kme,
            target_dataset_digest=target_dataset_digest,
            probe_episode_count=probe_episode_count,
            probe_steps=probe_steps,
        )
