"""Paired, domain-separated seed contracts for the v0.1 experiment."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

from ..hashing import canonical_json_bytes


V01_SEED_PLAN_SCHEMA = "policy-learnware.v01-seed-plan.v0"
SEED_MODULUS = 2**31 - 1


def _index(value: int, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{where} must be a non-negative integer")
    return value


def _name(value: str, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where} must be a non-empty string")
    return value


@dataclass(frozen=True)
class ProbeEpisodeSeeds:
    task: str
    bank: int
    episode_index: int
    reset_seed: int
    probe_seed: int
    namespace: str = "v01_probe"

    @property
    def collision_key(self) -> tuple[int, int]:
        return (self.reset_seed, self.probe_seed)


@dataclass(frozen=True)
class OracleEpisodeSeeds:
    task: str
    candidate_id: str
    episode_index: int
    reset_seed: int
    policy_seed: int
    namespace: str = "v01_oracle"

    @property
    def collision_key(self) -> tuple[int, int]:
        return (self.reset_seed, self.policy_seed)


@dataclass(frozen=True)
class Gate0EpisodeSeeds:
    task: str
    episode_index: int
    reset_seed: int
    action_seed: int
    namespace: str = "v01_gate0"

    @property
    def collision_key(self) -> tuple[int, int]:
        return (self.reset_seed, self.action_seed)


@dataclass(frozen=True)
class V01SeedPlan:
    project_seed: int
    schema: str = V01_SEED_PLAN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != V01_SEED_PLAN_SCHEMA:
            raise ValueError(f"unsupported v0.1 seed schema: {self.schema!r}")
        if isinstance(self.project_seed, bool) or not isinstance(self.project_seed, int):
            raise TypeError("project_seed must be an integer")
        if not 0 <= self.project_seed < 2**63:
            raise ValueError("project_seed must lie in [0, 2**63)")

    def derive(self, namespace: str, *, fields: Mapping[str, object], stream: str) -> int:
        namespace = _name(namespace, "namespace")
        stream = _name(stream, "stream")
        if namespace not in {
            "v01_probe", "v01_oracle", "v01_gate0", "v01_bootstrap", "v01_report"
        }:
            raise ValueError(f"unknown v0.1 seed namespace: {namespace!r}")
        payload = canonical_json_bytes(
            {
                "schema": self.schema,
                "project_seed": self.project_seed,
                "namespace": namespace,
                "fields": dict(fields),
                "stream": stream,
            }
        )
        raw = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        return raw % SEED_MODULUS

    def probe_episode(self, task: str, bank: int, episode_index: int) -> ProbeEpisodeSeeds:
        """No variant/factor argument by design: streams pair across variants."""

        task = _name(task, "task")
        bank = _index(bank, "bank")
        episode_index = _index(episode_index, "episode_index")
        fields = {"task": task, "bank": bank, "episode": episode_index}
        return ProbeEpisodeSeeds(
            task=task,
            bank=bank,
            episode_index=episode_index,
            reset_seed=self.derive("v01_probe", fields=fields, stream="environment_reset"),
            probe_seed=self.derive("v01_probe", fields=fields, stream="probe_action"),
        )

    def oracle_episode(
        self, task: str, candidate_id: str, episode_index: int
    ) -> OracleEpisodeSeeds:
        """No variant/factor argument, but candidate identity is mandatory."""

        task = _name(task, "task")
        candidate_id = _name(candidate_id, "candidate_id")
        episode_index = _index(episode_index, "episode_index")
        fields = {"task": task, "candidate_id": candidate_id, "episode": episode_index}
        return OracleEpisodeSeeds(
            task=task,
            candidate_id=candidate_id,
            episode_index=episode_index,
            reset_seed=self.derive("v01_oracle", fields=fields, stream="environment_reset"),
            policy_seed=self.derive("v01_oracle", fields=fields, stream="policy_action"),
        )

    def gate0_episode(self, task: str, episode_index: int) -> Gate0EpisodeSeeds:
        """Same Gate-0 action tensor is consumed by nominal and shifted variants."""

        task = _name(task, "task")
        episode_index = _index(episode_index, "episode_index")
        fields = {"task": task, "episode": episode_index}
        return Gate0EpisodeSeeds(
            task=task,
            episode_index=episode_index,
            reset_seed=self.derive("v01_gate0", fields=fields, stream="environment_reset"),
            action_seed=self.derive("v01_gate0", fields=fields, stream="probe_action"),
        )

    def bootstrap_seed(self, run_id: str, task: str, statistic_family: str) -> int:
        fields = {
            "run_id": _name(run_id, "run_id"),
            "task": _name(task, "task"),
            "statistic_family": _name(statistic_family, "statistic_family"),
        }
        return self.derive("v01_bootstrap", fields=fields, stream="resample")

    def report_seed(self, run_id: str, report_operation: str) -> int:
        fields = {
            "run_id": _name(run_id, "run_id"),
            "report_operation": _name(report_operation, "report_operation"),
        }
        return self.derive("v01_report", fields=fields, stream="monte_carlo")

    def probe_episodes(
        self, task: str, bank: int, episode_indices: Iterable[int]
    ) -> tuple[ProbeEpisodeSeeds, ...]:
        return tuple(self.probe_episode(task, bank, index) for index in episode_indices)

    def oracle_episodes(
        self, task: str, candidate_id: str, episode_indices: Iterable[int]
    ) -> tuple[OracleEpisodeSeeds, ...]:
        return tuple(
            self.oracle_episode(task, candidate_id, index) for index in episode_indices
        )


def assert_v01_seed_records_disjoint(
    records_by_domain: Mapping[
        str, Iterable[ProbeEpisodeSeeds | OracleEpisodeSeeds | Gate0EpisodeSeeds]
    ],
) -> None:
    """Reject observed cross-domain pair collisions.

    Intentional pairing across variants is represented by reusing one record,
    not by enumerating the same record once per variant.
    """

    owners: dict[tuple[int, int], str] = {}
    for domain, records in records_by_domain.items():
        _name(domain, "domain")
        for record in records:
            key = record.collision_key
            previous = owners.get(key)
            if previous is not None:
                raise ValueError(
                    "observed v0.1 seed collision between "
                    f"{previous!r} and {domain!r}: {key}"
                )
            owners[key] = domain


def assert_no_base_seed_pair_collision(
    v01_records: Iterable[ProbeEpisodeSeeds | OracleEpisodeSeeds | Gate0EpisodeSeeds],
    base_seed_pairs: Iterable[tuple[int, int]],
) -> None:
    base = set(base_seed_pairs)
    for record in v01_records:
        if record.collision_key in base:
            raise ValueError(f"v0.1 seed pair collides with a known base-v0 pair: {record.collision_key}")


def collect_known_base_seed_pairs(base_run_dir: str | Path) -> frozenset[tuple[int, int]]:
    """Collect every persisted v0 reset/action pair from authoritative JSON.

    The v0 run stores probe, championization and deployment seeds under a few
    versioned key pairs.  Recursing over the dataset/policy trees makes this
    audit independent of optional raw NPZ synchronization while still binding
    it to the completed formal artifacts.
    """

    root = Path(base_run_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"base run directory does not exist: {root}")
    key_pairs = (
        ("reset_seeds", "probe_seeds"),
        ("reset_seeds", "policy_seeds"),
        ("evaluation_reset_seeds", "evaluation_policy_seeds"),
    )
    result: set[tuple[int, int]] = set()

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            for left_name, right_name in key_pairs:
                if left_name not in value or right_name not in value:
                    continue
                left = value[left_name]
                right = value[right_name]
                if not isinstance(left, list) or not isinstance(right, list):
                    raise ValueError(
                        f"base seed fields {left_name}/{right_name} must be lists"
                    )
                if len(left) != len(right):
                    raise ValueError(
                        f"base seed fields {left_name}/{right_name} are misaligned"
                    )
                for first, second in zip(left, right, strict=True):
                    if (
                        isinstance(first, bool)
                        or isinstance(second, bool)
                        or not isinstance(first, int)
                        or not isinstance(second, int)
                        or first < 0
                        or second < 0
                    ):
                        raise ValueError("base seed records must be nonnegative integers")
                    result.add((int(first), int(second)))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for directory in (root / "datasets", root / "policy"):
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*.json")):
            try:
                visit(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(f"cannot inspect base seed artifact {path}: {error}") from error
    if not result:
        raise ValueError("completed base run exposes no auditable seed pairs")
    return frozenset(result)


__all__ = [
    "Gate0EpisodeSeeds", "OracleEpisodeSeeds", "ProbeEpisodeSeeds", "SEED_MODULUS",
    "V01SeedPlan", "V01_SEED_PLAN_SCHEMA", "assert_no_base_seed_pair_collision",
    "assert_v01_seed_records_disjoint", "collect_known_base_seed_pairs",
]
