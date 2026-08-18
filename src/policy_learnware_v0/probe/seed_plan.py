"""Stable, domain-separated seed derivation for every data split."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Mapping

from ..hashing import canonical_json_bytes


SEED_PLAN_SCHEMA = "policy-learnware.seed-plan.v0"
SEED_MODULUS = 2**31 - 1
SPLIT_NAMESPACES = frozenset(
    {
        "encoder_train",
        "encoder_validation",
        "kernel_calibration",
        "separability_calibration",
        "source_taskspec",
        "championization",
        "target_query",
        "final_return",
    }
)


@dataclass(frozen=True)
class EpisodeSeeds:
    namespace: str
    task_index: int
    episode_index: int
    reset_seed: int
    probe_seed: int
    bank_index: int = 0

    @property
    def collision_key(self) -> tuple[int, int, int]:
        return (self.task_index, self.reset_seed, self.probe_seed)


@dataclass(frozen=True)
class SeedPlan:
    project_seed: int
    schema: str = SEED_PLAN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SEED_PLAN_SCHEMA:
            raise ValueError(f"unsupported seed plan schema: {self.schema!r}")
        if isinstance(self.project_seed, bool) or not isinstance(self.project_seed, int):
            raise TypeError("project_seed must be an integer")
        if not 0 <= self.project_seed < 2**63:
            raise ValueError("project_seed must lie in [0, 2**63)")

    def derive(
        self,
        namespace: str,
        task_index: int,
        episode_index: int,
        *,
        stream: str,
        bank_index: int = 0,
    ) -> int:
        if namespace not in SPLIT_NAMESPACES:
            raise ValueError(f"unknown seed namespace: {namespace!r}")
        if isinstance(task_index, bool) or not isinstance(task_index, int) or task_index < 0:
            raise ValueError("task_index must be a non-negative integer")
        if (
            isinstance(episode_index, bool)
            or not isinstance(episode_index, int)
            or episode_index < 0
        ):
            raise ValueError("episode_index must be a non-negative integer")
        if stream not in {"environment_reset", "probe_action", "policy_action"}:
            raise ValueError(f"unknown seed stream: {stream!r}")
        if (
            isinstance(bank_index, bool)
            or not isinstance(bank_index, int)
            or bank_index < 0
        ):
            raise ValueError("bank_index must be a non-negative integer")
        if namespace != "target_query" and bank_index != 0:
            raise ValueError("only target_query may use a non-zero bank_index")
        payload = canonical_json_bytes(
            {
                "schema": self.schema,
                "project_seed": self.project_seed,
                "namespace": namespace,
                "task_index": task_index,
                "episode_index": episode_index,
                "bank_index": bank_index,
                "stream": stream,
            }
        )
        raw = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        return raw % SEED_MODULUS

    def episode(
        self,
        namespace: str,
        task_index: int,
        episode_index: int,
        *,
        bank_index: int = 0,
    ) -> EpisodeSeeds:
        return EpisodeSeeds(
            namespace=namespace,
            task_index=task_index,
            episode_index=episode_index,
            reset_seed=self.derive(
                namespace,
                task_index,
                episode_index,
                stream="environment_reset",
                bank_index=bank_index,
            ),
            probe_seed=self.derive(
                namespace,
                task_index,
                episode_index,
                stream="probe_action",
                bank_index=bank_index,
            ),
            bank_index=bank_index,
        )

    def episodes(
        self,
        namespace: str,
        task_index: int,
        episode_ids: Iterable[int],
        *,
        bank_index: int = 0,
    ) -> tuple[EpisodeSeeds, ...]:
        return tuple(
            self.episode(
                namespace, task_index, index, bank_index=bank_index
            )
            for index in episode_ids
        )


def assert_seed_records_disjoint(
    records_by_split: Mapping[str, Iterable[EpisodeSeeds]],
) -> None:
    """Reject any exact (task, reset, probe) overlap across split names."""

    owner: dict[tuple[int, int, int], str] = {}
    for split, records in records_by_split.items():
        for record in records:
            key = record.collision_key
            previous = owner.get(key)
            if previous is not None and previous != split:
                raise ValueError(
                    f"seed overlap between {previous!r} and {split!r}: {key}"
                )
            owner[key] = split
