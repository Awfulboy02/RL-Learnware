"""Strict, immutable configuration contract for the v0.1 shift diagnostic.

The v0.1 YAML is a research protocol, not a bag of command-line defaults.  This
module therefore rejects unknown and missing keys, normalises every sequence to
an immutable tuple, and exposes separate typed projections for measurement,
oracle and analysis identities.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import yaml

from ..hashing import canonicalize, sha256_json


V01_EXPERIMENT_CONFIG_SCHEMA = "policy-learnware.v01-experiment-config.v0"
APPROVED_TASKS = frozenset({"WalkerWalk", "FingerTurnEasy"})
APPROVED_SHIFT_ID = "global_nonzero_dof_damping_scale"
APPROVED_GRID = (0.5, 0.75, 1.0, 1.5, 2.0)
APPROVED_FORMAL_CONFIG_DIGEST = (
    "8966a1f38bd4e4f46c54e6fdb5a8ac004a33fb2eb512187731bced2485ebc2d1"
)
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


class V01ConfigError(ValueError):
    """The v0.1 experiment contract is incomplete or has drifted."""


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise V01ConfigError(f"{where} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise V01ConfigError(f"{where} keys must be strings")
    return value


def _strict(data: Mapping[str, Any], fields: set[str], where: str) -> None:
    missing = fields - set(data)
    unknown = set(data) - fields
    if missing or unknown:
        raise V01ConfigError(
            f"invalid {where} keys; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _string(value: Any, where: str, *, safe: bool = False) -> str:
    if not isinstance(value, str) or not value:
        raise V01ConfigError(f"{where} must be a non-empty string")
    if safe and not _SAFE_ID.fullmatch(value):
        raise V01ConfigError(f"{where} is not a path-safe identifier")
    return value


def _bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise V01ConfigError(f"{where} must be boolean")
    return value


def _int(value: Any, where: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise V01ConfigError(f"{where} must be an integer >= {minimum}")
    return value


def _float(value: Any, where: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise V01ConfigError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise V01ConfigError(f"{where} must be finite")
    if minimum is not None and result < minimum:
        raise V01ConfigError(f"{where} must be >= {minimum}")
    return result


def _strings(value: Any, where: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise V01ConfigError(f"{where} must be a sequence of strings")
    result = tuple(_string(item, f"{where}[]") for item in value)
    if not allow_empty and not result:
        raise V01ConfigError(f"{where} cannot be empty")
    if len(set(result)) != len(result):
        raise V01ConfigError(f"{where} contains duplicates")
    return result


def _ints(value: Any, where: str, *, minimum: int = 0) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise V01ConfigError(f"{where} must be an integer sequence")
    return tuple(_int(item, f"{where}[]", minimum=minimum) for item in value)


def _hex_digest(value: Any, where: str) -> str:
    digest = _string(value, where)
    if len(digest) != 64:
        raise V01ConfigError(f"{where} must be a SHA-256 hex digest")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise V01ConfigError(f"{where} must be a SHA-256 hex digest") from exc
    return digest.lower()


def _exact(value: Any, expected: Any, where: str) -> Any:
    if value != expected:
        raise V01ConfigError(f"{where} must be {expected!r}, got {value!r}")
    return value


@dataclass(frozen=True)
class BaseConfig:
    pool_id: str
    expected_protocol_id: str
    expected_protocol_draft_hash: str
    checkpoint_outer: int
    actual_environment_steps: int
    candidates_per_task: int

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BaseConfig":
        data = _mapping(value, "base")
        fields = {
            "pool_id", "expected_protocol_id", "expected_protocol_draft_hash",
            "checkpoint_outer", "actual_environment_steps", "candidates_per_task",
        }
        _strict(data, fields, "base")
        candidates = _int(data["candidates_per_task"], "base.candidates_per_task", minimum=1)
        _exact(candidates, 10, "base.candidates_per_task")
        return cls(
            pool_id=_string(data["pool_id"], "base.pool_id", safe=True),
            expected_protocol_id=_hex_digest(data["expected_protocol_id"], "base.expected_protocol_id"),
            expected_protocol_draft_hash=_hex_digest(
                data["expected_protocol_draft_hash"], "base.expected_protocol_draft_hash"
            ),
            checkpoint_outer=_int(data["checkpoint_outer"], "base.checkpoint_outer", minimum=0),
            actual_environment_steps=_int(
                data["actual_environment_steps"], "base.actual_environment_steps", minimum=1
            ),
            candidates_per_task=candidates,
        )


@dataclass(frozen=True)
class TasksConfig:
    infrastructure: tuple[str, ...]
    confirmatory: tuple[str, ...]

    @property
    def all(self) -> tuple[str, ...]:
        return self.infrastructure + self.confirmatory

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TasksConfig":
        data = _mapping(value, "tasks")
        _strict(data, {"infrastructure", "confirmatory"}, "tasks")
        infrastructure = _strings(data["infrastructure"], "tasks.infrastructure")
        confirmatory = _strings(data["confirmatory"], "tasks.confirmatory", allow_empty=True)
        all_tasks = infrastructure + confirmatory
        if len(set(all_tasks)) != len(all_tasks):
            raise V01ConfigError("infrastructure and confirmatory tasks must be disjoint")
        illegal = set(all_tasks) - APPROVED_TASKS
        if illegal:
            raise V01ConfigError(f"unapproved v0.1 tasks: {sorted(illegal)}")
        if "WalkerWalk" not in infrastructure:
            raise V01ConfigError("WalkerWalk must remain the infrastructure task")
        if confirmatory and confirmatory != ("FingerTurnEasy",):
            raise V01ConfigError("the only approved confirmatory task is FingerTurnEasy")
        return cls(infrastructure=infrastructure, confirmatory=confirmatory)


@dataclass(frozen=True)
class ShiftConfig:
    shift_id: str
    nominal_factor: float
    diagnostic_grid: tuple[float, ...]
    static_within_episode: bool
    allow_cli_override: bool

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ShiftConfig":
        data = _mapping(value, "shift")
        fields = {
            "shift_id", "nominal_factor", "diagnostic_grid",
            "static_within_episode", "allow_cli_override",
        }
        _strict(data, fields, "shift")
        shift_id = _string(data["shift_id"], "shift.shift_id")
        _exact(shift_id, APPROVED_SHIFT_ID, "shift.shift_id")
        nominal = _float(data["nominal_factor"], "shift.nominal_factor", minimum=0.0)
        _exact(nominal, 1.0, "shift.nominal_factor")
        raw_grid = data["diagnostic_grid"]
        if isinstance(raw_grid, (str, bytes)) or not isinstance(raw_grid, Sequence):
            raise V01ConfigError("shift.diagnostic_grid must be a sequence")
        grid = tuple(_float(item, "shift.diagnostic_grid[]", minimum=0.0) for item in raw_grid)
        if any(item <= 0.0 for item in grid):
            raise V01ConfigError("shift factors must be positive")
        if len(set(grid)) != len(grid) or tuple(sorted(grid)) != grid:
            raise V01ConfigError("shift.diagnostic_grid must be unique and increasing")
        _exact(grid, APPROVED_GRID, "shift.diagnostic_grid")
        _exact(_bool(data["static_within_episode"], "shift.static_within_episode"), True,
               "shift.static_within_episode")
        _exact(_bool(data["allow_cli_override"], "shift.allow_cli_override"), False,
               "shift.allow_cli_override")
        return cls(shift_id, nominal, grid, True, False)


@dataclass(frozen=True)
class EvidenceConfig:
    shifted_random_probe: str
    shifted_candidate_rollout: str
    selector_may_read_context: bool
    selector_may_read_oracle: bool

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceConfig":
        data = _mapping(value, "evidence")
        fields = {
            "shifted_random_probe", "shifted_candidate_rollout",
            "selector_may_read_context", "selector_may_read_oracle",
        }
        _strict(data, fields, "evidence")
        probe = _string(data["shifted_random_probe"], "evidence.shifted_random_probe")
        rollout = _string(data["shifted_candidate_rollout"], "evidence.shifted_candidate_rollout")
        _exact(probe, "allowed_for_diagnostic", "evidence.shifted_random_probe")
        _exact(rollout, "oracle_only", "evidence.shifted_candidate_rollout")
        context = _bool(data["selector_may_read_context"], "evidence.selector_may_read_context")
        oracle = _bool(data["selector_may_read_oracle"], "evidence.selector_may_read_oracle")
        _exact(context, False, "evidence.selector_may_read_context")
        _exact(oracle, False, "evidence.selector_may_read_oracle")
        return cls(probe, rollout, context, oracle)


@dataclass(frozen=True)
class ReturnContractConfig:
    horizon: int
    discount: float
    primary: str
    save_raw_episodic_sum: bool
    early_termination: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReturnContractConfig":
        data = _mapping(value, "return_contract")
        fields = {"horizon", "discount", "primary", "save_raw_episodic_sum", "early_termination"}
        _strict(data, fields, "return_contract")
        horizon = _int(data["horizon"], "return_contract.horizon", minimum=1)
        _exact(horizon, 1000, "return_contract.horizon")
        discount = _float(data["discount"], "return_contract.discount", minimum=0.0)
        _exact(discount, 1.0, "return_contract.discount")
        primary = _string(data["primary"], "return_contract.primary")
        _exact(primary, "undiscounted_per_step_mean", "return_contract.primary")
        raw = _bool(data["save_raw_episodic_sum"], "return_contract.save_raw_episodic_sum")
        _exact(raw, True, "return_contract.save_raw_episodic_sum")
        early = _string(data["early_termination"], "return_contract.early_termination")
        _exact(early, "fail_gate0", "return_contract.early_termination")
        return cls(horizon, discount, primary, raw, early)


@dataclass(frozen=True)
class ProbeConfig:
    banks: int
    max_episodes_per_bank: int
    prefix_grid: tuple[int, ...]
    gate_b_unreduced_prefix: int
    sparse_within_bank_pairs: tuple[tuple[int, int], ...]
    between_pairing: str
    max_prefix_role: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProbeConfig":
        data = _mapping(value, "probe")
        fields = {
            "banks", "max_episodes_per_bank", "prefix_grid", "gate_b_unreduced_prefix",
            "sparse_within_bank_pairs", "between_pairing", "max_prefix_role",
        }
        _strict(data, fields, "probe")
        banks = _int(data["banks"], "probe.banks", minimum=2)
        maximum = _int(data["max_episodes_per_bank"], "probe.max_episodes_per_bank", minimum=1)
        grid = _ints(data["prefix_grid"], "probe.prefix_grid", minimum=1)
        if not grid or tuple(sorted(set(grid))) != grid or grid[-1] != maximum:
            raise V01ConfigError("probe.prefix_grid must be unique, increasing, and end at max episodes")
        gate_prefix = _int(data["gate_b_unreduced_prefix"], "probe.gate_b_unreduced_prefix", minimum=1)
        if gate_prefix not in grid:
            raise V01ConfigError("probe.gate_b_unreduced_prefix must be in prefix_grid")
        raw_pairs = data["sparse_within_bank_pairs"]
        if isinstance(raw_pairs, (str, bytes)) or not isinstance(raw_pairs, Sequence):
            raise V01ConfigError("probe.sparse_within_bank_pairs must be a sequence")
        pairs: list[tuple[int, int]] = []
        used: set[int] = set()
        for index, raw in enumerate(raw_pairs):
            pair = _ints(raw, f"probe.sparse_within_bank_pairs[{index}]", minimum=0)
            if len(pair) != 2 or pair[0] == pair[1] or any(item >= banks for item in pair):
                raise V01ConfigError("within-bank pairs must contain two distinct valid banks")
            if pair[0] >= pair[1] or used.intersection(pair):
                raise V01ConfigError("within-bank pairs must be ordered and disjoint")
            used.update(pair)
            pairs.append((pair[0], pair[1]))
        if not pairs:
            raise V01ConfigError("at least one sparse within-bank pair is required")
        between = _string(data["between_pairing"], "probe.between_pairing")
        role = _string(data["max_prefix_role"], "probe.max_prefix_role")
        _exact(between, "same_bank_nominal_to_each_shift", "probe.between_pairing")
        _exact(role, "six_source_rkme_routing", "probe.max_prefix_role")
        return cls(banks, maximum, grid, gate_prefix, tuple(pairs), between, role)


@dataclass(frozen=True)
class OracleConfig:
    episodes_per_candidate_variant: int
    paired_across_variants: bool
    paired_across_candidates: bool
    require_golden_parity: bool
    require_compiled_parity: bool

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OracleConfig":
        data = _mapping(value, "oracle")
        fields = {
            "episodes_per_candidate_variant", "paired_across_variants",
            "paired_across_candidates", "require_golden_parity", "require_compiled_parity",
        }
        _strict(data, fields, "oracle")
        episodes = _int(data["episodes_per_candidate_variant"], "oracle.episodes_per_candidate_variant", minimum=1)
        parsed = {
            key: _bool(data[key], f"oracle.{key}")
            for key in fields - {"episodes_per_candidate_variant"}
        }
        for key, expected in {
            "paired_across_variants": True,
            "paired_across_candidates": False,
            "require_golden_parity": True,
            "require_compiled_parity": True,
        }.items():
            _exact(parsed[key], expected, f"oracle.{key}")
        return cls(episodes_per_candidate_variant=episodes, **parsed)


@dataclass(frozen=True)
class StatisticsConfig:
    competence_alpha: float
    minimum_material_effect: float
    minimum_sensitivity_heterogeneity: float
    confidence_level: float
    bootstrap_method: str
    p_value_method: str
    bootstrap_resamples: int
    multiplicity: str
    holm_population: str
    holm_families: tuple[str, ...]
    top1_bootstrap_probability: float
    gate_c_resampling: str
    gate_c_minimum_finite_bootstrap_fraction: float

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StatisticsConfig":
        data = _mapping(value, "statistics")
        fields = {
            "competence_alpha", "minimum_material_effect", "minimum_sensitivity_heterogeneity",
            "confidence_level", "bootstrap_method", "p_value_method", "bootstrap_resamples",
            "multiplicity", "holm_population", "holm_families", "top1_bootstrap_probability",
            "gate_c_resampling", "gate_c_minimum_finite_bootstrap_fraction",
        }
        _strict(data, fields, "statistics")
        numbers = {
            name: _float(data[name], f"statistics.{name}", minimum=0.0)
            for name in {
                "competence_alpha", "minimum_material_effect", "minimum_sensitivity_heterogeneity",
                "confidence_level", "top1_bootstrap_probability",
                "gate_c_minimum_finite_bootstrap_fraction",
            }
        }
        for name in {"competence_alpha", "confidence_level", "top1_bootstrap_probability", "gate_c_minimum_finite_bootstrap_fraction"}:
            if numbers[name] > 1.0:
                raise V01ConfigError(f"statistics.{name} must lie in [0, 1]")
        strings = {
            name: _string(data[name], f"statistics.{name}")
            for name in {
                "bootstrap_method", "p_value_method", "multiplicity",
                "holm_population", "gate_c_resampling",
            }
        }
        expected = {
            "bootstrap_method": "contract_aware_percentile",
            "p_value_method": "centered_bootstrap",
            "multiplicity": "holm_bonferroni_per_task_and_test_family",
            "holm_population": "all_10_candidates_before_competence_filter",
            "gate_c_resampling": "nested_probe_banks_and_oracle_episodes",
        }
        for name, required in expected.items():
            _exact(strings[name], required, f"statistics.{name}")
        families = _strings(data["holm_families"], "statistics.holm_families")
        _exact(families, ("material", "heterogeneity", "ranking_gap"), "statistics.holm_families")
        return cls(
            **numbers,
            **strings,
            bootstrap_resamples=_int(data["bootstrap_resamples"], "statistics.bootstrap_resamples", minimum=1),
            holm_families=families,
        )


@dataclass(frozen=True)
class IdentityGateConfig:
    audit_episodes: int
    non_nominal_finite_episodes_per_factor: int
    non_nominal_action_source: str
    audit_seed_namespace: str
    model_digest_exact: bool
    flag_and_seed_exact: bool
    trajectory_atol: float
    trajectory_rtol: float
    return_atol: float
    return_atol_metric: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "IdentityGateConfig":
        data = _mapping(value, "gates.identity")
        fields = {
            "audit_episodes", "non_nominal_finite_episodes_per_factor", "non_nominal_action_source",
            "audit_seed_namespace", "model_digest_exact", "flag_and_seed_exact",
            "trajectory_atol", "trajectory_rtol", "return_atol", "return_atol_metric",
        }
        _strict(data, fields, "gates.identity")
        result = cls(
            audit_episodes=_int(data["audit_episodes"], "gates.identity.audit_episodes", minimum=1),
            non_nominal_finite_episodes_per_factor=_int(data["non_nominal_finite_episodes_per_factor"], "gates.identity.non_nominal_finite_episodes_per_factor", minimum=1),
            non_nominal_action_source=_string(data["non_nominal_action_source"], "gates.identity.non_nominal_action_source"),
            audit_seed_namespace=_string(data["audit_seed_namespace"], "gates.identity.audit_seed_namespace"),
            model_digest_exact=_bool(data["model_digest_exact"], "gates.identity.model_digest_exact"),
            flag_and_seed_exact=_bool(data["flag_and_seed_exact"], "gates.identity.flag_and_seed_exact"),
            trajectory_atol=_float(data["trajectory_atol"], "gates.identity.trajectory_atol", minimum=0.0),
            trajectory_rtol=_float(data["trajectory_rtol"], "gates.identity.trajectory_rtol", minimum=0.0),
            return_atol=_float(data["return_atol"], "gates.identity.return_atol", minimum=0.0),
            return_atol_metric=_string(data["return_atol_metric"], "gates.identity.return_atol_metric"),
        )
        _exact(result.non_nominal_action_source, "v01_gaussian_random_probe", "gates.identity.non_nominal_action_source")
        _exact(result.audit_seed_namespace, "v01_gate0", "gates.identity.audit_seed_namespace")
        _exact(result.model_digest_exact, True, "gates.identity.model_digest_exact")
        _exact(result.flag_and_seed_exact, True, "gates.identity.flag_and_seed_exact")
        _exact(result.return_atol_metric, "undiscounted_per_step_mean", "gates.identity.return_atol_metric")
        return result


@dataclass(frozen=True)
class TaskSpecGateConfig:
    minimum_between_within_ratio: float
    minimum_severity_spearman: float
    severity_excludes_nominal_self_point: bool
    undefined_spearman: str
    minimum_max_prefix_routing_accuracy: float
    numerical_zero_tolerance: float

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskSpecGateConfig":
        data = _mapping(value, "gates.taskspec")
        fields = {
            "minimum_between_within_ratio", "minimum_severity_spearman",
            "severity_excludes_nominal_self_point", "undefined_spearman",
            "minimum_max_prefix_routing_accuracy", "numerical_zero_tolerance",
        }
        _strict(data, fields, "gates.taskspec")
        result = cls(
            minimum_between_within_ratio=_float(data["minimum_between_within_ratio"], "gates.taskspec.minimum_between_within_ratio", minimum=0.0),
            minimum_severity_spearman=_float(data["minimum_severity_spearman"], "gates.taskspec.minimum_severity_spearman", minimum=-1.0),
            severity_excludes_nominal_self_point=_bool(data["severity_excludes_nominal_self_point"], "gates.taskspec.severity_excludes_nominal_self_point"),
            undefined_spearman=_string(data["undefined_spearman"], "gates.taskspec.undefined_spearman"),
            minimum_max_prefix_routing_accuracy=_float(data["minimum_max_prefix_routing_accuracy"], "gates.taskspec.minimum_max_prefix_routing_accuracy", minimum=0.0),
            numerical_zero_tolerance=_float(data["numerical_zero_tolerance"], "gates.taskspec.numerical_zero_tolerance", minimum=0.0),
        )
        if result.minimum_severity_spearman > 1.0 or result.minimum_max_prefix_routing_accuracy > 1.0:
            raise V01ConfigError("TaskSpec correlation/accuracy thresholds cannot exceed 1")
        _exact(result.severity_excludes_nominal_self_point, True, "gates.taskspec.severity_excludes_nominal_self_point")
        _exact(result.undefined_spearman, "fail_with_null_reason", "gates.taskspec.undefined_spearman")
        return result


@dataclass(frozen=True)
class LeakageGateConfig:
    forbidden_measurement_fields: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LeakageGateConfig":
        data = _mapping(value, "gates.leakage")
        _strict(data, {"forbidden_measurement_fields"}, "gates.leakage")
        fields = _strings(data["forbidden_measurement_fields"], "gates.leakage.forbidden_measurement_fields")
        required = {
            "factor", "d_theta", "policy_id", "return", "task", "candidate_id",
            "environment_instance_digest", "base_model_digest", "shifted_model_digest",
        }
        if not required.issubset(fields):
            raise V01ConfigError("gates.leakage omits mandatory forbidden fields")
        return cls(fields)


@dataclass(frozen=True)
class GatesConfig:
    identity: IdentityGateConfig
    taskspec: TaskSpecGateConfig
    leakage: LeakageGateConfig

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GatesConfig":
        data = _mapping(value, "gates")
        _strict(data, {"identity", "taskspec", "leakage"}, "gates")
        return cls(
            identity=IdentityGateConfig.from_dict(data["identity"]),
            taskspec=TaskSpecGateConfig.from_dict(data["taskspec"]),
            leakage=LeakageGateConfig.from_dict(data["leakage"]),
        )


@dataclass(frozen=True)
class FailurePolicyConfig:
    gate0_or_gate_d: str
    gate_a: str
    gate_b: str
    automatic_range_or_pool_expansion: bool

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FailurePolicyConfig":
        data = _mapping(value, "failure_policy")
        fields = {"gate0_or_gate_d", "gate_a", "gate_b", "automatic_range_or_pool_expansion"}
        _strict(data, fields, "failure_policy")
        result = cls(
            gate0_or_gate_d=_string(data["gate0_or_gate_d"], "failure_policy.gate0_or_gate_d"),
            gate_a=_string(data["gate_a"], "failure_policy.gate_a"),
            gate_b=_string(data["gate_b"], "failure_policy.gate_b"),
            automatic_range_or_pool_expansion=_bool(data["automatic_range_or_pool_expansion"], "failure_policy.automatic_range_or_pool_expansion"),
        )
        _exact(result.gate0_or_gate_d, "block_formal_run", "failure_policy.gate0_or_gate_d")
        _exact(result.gate_a, "report_no_go_pool", "failure_policy.gate_a")
        _exact(result.gate_b, "report_no_go_taskspec", "failure_policy.gate_b")
        _exact(result.automatic_range_or_pool_expansion, False, "failure_policy.automatic_range_or_pool_expansion")
        return result


@dataclass(frozen=True)
class V01ExperimentConfig:
    schema: str
    experiment_id: str
    project_seed: int
    base: BaseConfig
    tasks: TasksConfig
    shift: ShiftConfig
    evidence: EvidenceConfig
    return_contract: ReturnContractConfig
    probe: ProbeConfig
    oracle: OracleConfig
    statistics: StatisticsConfig
    gates: GatesConfig
    failure_policy: FailurePolicyConfig

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "V01ExperimentConfig":
        data = _mapping(value, "v0.1 config")
        fields = {
            "schema", "experiment_id", "project_seed", "base", "tasks", "shift",
            "evidence", "return_contract", "probe", "oracle", "statistics", "gates",
            "failure_policy",
        }
        _strict(data, fields, "v0.1 config")
        schema = _string(data["schema"], "schema")
        _exact(schema, V01_EXPERIMENT_CONFIG_SCHEMA, "schema")
        config = cls(
            schema=schema,
            experiment_id=_string(data["experiment_id"], "experiment_id", safe=True),
            project_seed=_int(data["project_seed"], "project_seed", minimum=0),
            base=BaseConfig.from_dict(data["base"]),
            tasks=TasksConfig.from_dict(data["tasks"]),
            shift=ShiftConfig.from_dict(data["shift"]),
            evidence=EvidenceConfig.from_dict(data["evidence"]),
            return_contract=ReturnContractConfig.from_dict(data["return_contract"]),
            probe=ProbeConfig.from_dict(data["probe"]),
            oracle=OracleConfig.from_dict(data["oracle"]),
            statistics=StatisticsConfig.from_dict(data["statistics"]),
            gates=GatesConfig.from_dict(data["gates"]),
            failure_policy=FailurePolicyConfig.from_dict(data["failure_policy"]),
        )
        if config.project_seed >= 2**63:
            raise V01ConfigError("project_seed must lie in [0, 2**63)")
        return config

    def to_dict(self) -> dict[str, Any]:
        return canonicalize(asdict(self))

    @property
    def config_digest(self) -> str:
        return sha256_json(self.to_dict())

    def measurement_projection(self) -> dict[str, Any]:
        """Candidate-independent part of the approved experiment contract."""

        return canonicalize(
            {
                "schema": "policy-learnware.v01-measurement-config-projection.v0",
                "base": {
                    "expected_protocol_id": self.base.expected_protocol_id,
                    "expected_protocol_draft_hash": self.base.expected_protocol_draft_hash,
                },
                "tasks": asdict(self.tasks),
                "shift": asdict(self.shift),
                "evidence": {"shifted_random_probe": self.evidence.shifted_random_probe},
                "return_contract": asdict(self.return_contract),
                "probe": asdict(self.probe),
                "measurement_gates": {
                    "identity": asdict(self.gates.identity),
                    "taskspec": asdict(self.gates.taskspec),
                    "leakage": asdict(self.gates.leakage),
                },
            }
        )

    def oracle_projection(self) -> dict[str, Any]:
        return canonicalize(
            {
                "schema": "policy-learnware.v01-oracle-config-projection.v0",
                "base": asdict(self.base),
                "tasks": asdict(self.tasks),
                "shift": asdict(self.shift),
                "return_contract": asdict(self.return_contract),
                "oracle": asdict(self.oracle),
                "identity_gate": asdict(self.gates.identity),
            }
        )

    def analysis_projection(self) -> dict[str, Any]:
        return canonicalize(
            {
                "schema": "policy-learnware.v01-analysis-config-projection.v0",
                "statistics": asdict(self.statistics),
                "gates": asdict(self.gates),
                "failure_policy": asdict(self.failure_policy),
            }
        )

    @property
    def measurement_config_digest(self) -> str:
        return sha256_json(self.measurement_projection())

    @property
    def oracle_config_digest(self) -> str:
        return sha256_json(self.oracle_projection())

    @property
    def analysis_config_digest(self) -> str:
        return sha256_json(self.analysis_projection())


def load_v01_experiment_config(path: str | Path) -> V01ExperimentConfig:
    config_path = Path(path)
    try:
        value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise V01ConfigError(f"cannot load v0.1 config {config_path}: {exc}") from exc
    return V01ExperimentConfig.from_dict(_mapping(value, "v0.1 config"))


__all__ = [
    "APPROVED_FORMAL_CONFIG_DIGEST", "APPROVED_GRID", "APPROVED_SHIFT_ID",
    "APPROVED_TASKS", "BaseConfig",
    "EvidenceConfig", "FailurePolicyConfig", "GatesConfig", "IdentityGateConfig",
    "LeakageGateConfig", "OracleConfig", "ProbeConfig", "ReturnContractConfig",
    "ShiftConfig", "StatisticsConfig", "TaskSpecGateConfig", "TasksConfig",
    "V01ConfigError", "V01ExperimentConfig", "V01_EXPERIMENT_CONFIG_SCHEMA",
    "load_v01_experiment_config",
]
