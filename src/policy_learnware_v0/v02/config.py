"""Strict configuration contract for the v0.2 source-anchor experiment.

Two loaders are intentionally exposed:

``load_v02_config_draft`` records an RFC containing explicit review markers;
``load_v02_formal_config`` accepts only a complete v0.2 freeze-ready protocol.  The
formal path never supplies defaults for scientific choices and rejects null,
TBD, unknown keys, split overlap, invalid factor roles, and duplicated nominal
anchors.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

import yaml

from ..hashing import canonicalize, sha256_json


V02_EXPERIMENT_CONFIG_SCHEMA = "policy-learnware.v02-experiment-config.v0"
V02_PROTOCOL_FAMILY_ID = "continuous-vector-mdp-v02"
V02_STAGES = frozenset({"audit_smoke", "development_discovery", "v02_freeze_ready"})
FORMAL_V02_TASKS = frozenset(
    {
        "CartpoleSwingup",
        "CheetahRun",
        "FingerTurnEasy",
        "FishSwim",
        "ReacherEasy",
        "WalkerWalk",
    }
)
FORMAL_V02_METHOD_IDS = frozenset(
    {"B0", "B1", "B2", "B3a", "B3b", "B4a", "B4b", "A-Env", "M02/B5"}
)
FACTOR_ROLES = frozenset(
    {"source", "development", "confirmatory_heldout", "safety_exact_reference"}
)
TARGET_REGIMES = frozenset(
    {
        "safety_exact",
        "heldout_interpolation",
        "heldout_extrapolation",
        "market_ood_boundary",
    }
)
REVIEW_MARKERS = frozenset({"TBD", "REVIEW_REQUIRED", "[REVIEW REQUIRED]"})
COMPETENCE_MODES = frozenset({"OBSERVE", "ENFORCE"})
CompetenceMode = Literal["OBSERVE", "ENFORCE"]
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.:/-]+$")
_OPAQUE_TARGET_ID = re.compile(r"^v02q-[0-9a-f]{32}$")
_TOP_LEVEL_FIELDS = {
    "schema",
    "experiment_id",
    "stage",
    "protocol_family_id",
    "tasks",
    "dynamics_axes",
    "source_factors",
    "development_targets",
    "confirmatory_targets",
    "safety_exact_targets",
    "primary_algorithm",
    "training_steps",
    "training_seeds",
    "checkpoint_rule",
    "source_eval_episodes",
    "competence_floor",
    "source_championization",
    "probe_protocol_id",
    "probe_prefixes",
    "encoder_eval_prefixes",
    "representation_ids",
    "method_ids",
    "primary_endpoint",
    "noninferiority_margin",
    "minimum_effect",
    "bootstrap_plan",
    "multiple_testing_plan",
    "artifact_root",
}
_OPTIONAL_TOP_LEVEL_FIELDS = frozenset({"source_championization"})


class V02ConfigError(ValueError):
    """The v0.2 protocol is incomplete, ambiguous, or internally inconsistent."""


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise V02ConfigError(f"{where} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise V02ConfigError(f"{where} keys must be strings")
    return value


def _strict(value: Mapping[str, Any], expected: set[str], where: str) -> None:
    data = _mapping(value, where)
    missing = expected - set(data)
    unknown = set(data) - expected
    if missing or unknown:
        raise V02ConfigError(
            f"invalid {where} keys; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _strict_top_level(value: Mapping[str, Any], where: str) -> None:
    """Validate the additive v0.2 top-level schema.

    ``source_championization`` is optional only so existing audit/development
    configs retain their exact serialized form and digest.  The executable
    formal loader requires it after parsing ``stage`` below.
    """

    data = _mapping(value, where)
    required = _TOP_LEVEL_FIELDS - set(_OPTIONAL_TOP_LEVEL_FIELDS)
    missing = required - set(data)
    unknown = set(data) - _TOP_LEVEL_FIELDS
    if missing or unknown:
        raise V02ConfigError(
            f"invalid {where} keys; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _is_review_marker(value: Any) -> bool:
    return isinstance(value, str) and value.strip().upper() in REVIEW_MARKERS


def _string(value: Any, where: str, *, safe: bool = False) -> str:
    if value is None or _is_review_marker(value):
        raise V02ConfigError(f"{where} remains unresolved")
    if not isinstance(value, str) or not value:
        raise V02ConfigError(f"{where} must be a non-empty string")
    if safe and not _SAFE_ID.fullmatch(value):
        raise V02ConfigError(f"{where} is not a safe identifier")
    return value


def _digest(value: Any, where: str) -> str:
    result = _string(value, where).lower()
    if len(result) != 64:
        raise V02ConfigError(f"{where} must be a SHA-256 hex digest")
    try:
        int(result, 16)
    except ValueError as exc:
        raise V02ConfigError(f"{where} must be a SHA-256 hex digest") from exc
    return result


def _bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise V02ConfigError(f"{where} must be boolean")
    return value


def _int(value: Any, where: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise V02ConfigError(f"{where} must be an integer >= {minimum}")
    return value


def _float(value: Any, where: str, *, minimum: float | None = None) -> float:
    if value is None or _is_review_marker(value):
        raise V02ConfigError(f"{where} remains unresolved")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise V02ConfigError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise V02ConfigError(f"{where} must be finite")
    if minimum is not None and result < minimum:
        raise V02ConfigError(f"{where} must be >= {minimum}")
    return result


def _sequence(value: Any, where: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise V02ConfigError(f"{where} must be a sequence")
    return value


def _strings(value: Any, where: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    result = tuple(
        _string(item, f"{where}[]", safe=True) for item in _sequence(value, where)
    )
    if not allow_empty and not result:
        raise V02ConfigError(f"{where} cannot be empty")
    if len(set(result)) != len(result):
        raise V02ConfigError(f"{where} contains duplicates")
    return result


def _ints(value: Any, where: str, *, minimum: int = 0) -> tuple[int, ...]:
    result = tuple(_int(item, f"{where}[]", minimum=minimum) for item in _sequence(value, where))
    if not result:
        raise V02ConfigError(f"{where} cannot be empty")
    if len(set(result)) != len(result):
        raise V02ConfigError(f"{where} contains duplicates")
    return result


def _deep_freeze(value: Any) -> Any:
    canonical = canonicalize(value)
    if isinstance(canonical, Mapping):
        return MappingProxyType({key: _deep_freeze(item) for key, item in canonical.items()})
    if isinstance(canonical, list):
        return tuple(_deep_freeze(item) for item in canonical)
    return canonical


def _find_unresolved(value: Any, path: str = "$") -> tuple[str, ...]:
    found: list[str] = []
    nullable_record_fields = (
        ".axis_binding_digest",
        ".source_anchor_ref",
        ".axis_id",
    )
    if value is None and not path.endswith(nullable_record_fields):
        found.append(path)
    elif _is_review_marker(value):
        found.append(path)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            found.extend(_find_unresolved(item, f"{path}.{key}"))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            found.extend(_find_unresolved(item, f"{path}[{index}]"))
    return tuple(found)


@dataclass(frozen=True)
class AxisConfig:
    axis_id: str
    operator_id: str
    operator_digest: str
    leaf_allowlist: tuple[str, ...]
    static_within_episode: bool

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], where: str) -> "AxisConfig":
        fields = {
            "axis_id", "operator_id", "operator_digest", "leaf_allowlist",
            "static_within_episode",
        }
        data = _mapping(value, where)
        _strict(data, fields, where)
        leaves = _strings(data["leaf_allowlist"], f"{where}.leaf_allowlist")
        if not all(leaf.startswith("_mjx_model.") for leaf in leaves):
            raise V02ConfigError(f"{where}.leaf_allowlist must contain exact model leaves")
        static = _bool(data["static_within_episode"], f"{where}.static_within_episode")
        if not static:
            raise V02ConfigError("v0.2 supports only episode-static dynamics shifts")
        return cls(
            axis_id=_string(data["axis_id"], f"{where}.axis_id", safe=True),
            operator_id=_string(data["operator_id"], f"{where}.operator_id", safe=True),
            operator_digest=_digest(data["operator_digest"], f"{where}.operator_digest"),
            leaf_allowlist=leaves,
            static_within_episode=True,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis_id": self.axis_id,
            "operator_id": self.operator_id,
            "operator_digest": self.operator_digest,
            "leaf_allowlist": list(self.leaf_allowlist),
            "static_within_episode": self.static_within_episode,
        }


@dataclass(frozen=True)
class SourceFactorConfig:
    factor_id: str
    value: float
    roles: tuple[str, ...]
    source_anchor_id: str
    axis_binding_digest: str | None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], where: str) -> "SourceFactorConfig":
        fields = {"factor_id", "value", "roles", "source_anchor_id", "axis_binding_digest"}
        data = _mapping(value, where)
        _strict(data, fields, where)
        factor_value = _float(data["value"], f"{where}.value", minimum=0.0)
        if factor_value <= 0.0:
            raise V02ConfigError(f"{where}.value must be positive")
        roles = _strings(data["roles"], f"{where}.roles")
        illegal = set(roles) - FACTOR_ROLES
        if illegal:
            raise V02ConfigError(f"{where}.roles contains unknown roles: {sorted(illegal)}")
        if set(roles) != {"source"}:
            raise V02ConfigError("source_factors must carry exactly the source role")
        binding_value = data["axis_binding_digest"]
        binding = None if binding_value is None else _digest(
            binding_value, f"{where}.axis_binding_digest"
        )
        is_nominal = math.isclose(factor_value, 1.0, rel_tol=0.0, abs_tol=0.0)
        if is_nominal != (binding is None):
            raise V02ConfigError(
                f"{where}.axis_binding_digest must be null exactly for factor 1.0"
            )
        return cls(
            factor_id=_string(data["factor_id"], f"{where}.factor_id", safe=True),
            value=factor_value,
            roles=roles,
            source_anchor_id=_digest(data["source_anchor_id"], f"{where}.source_anchor_id"),
            axis_binding_digest=binding,
        )

    @property
    def is_nominal(self) -> bool:
        return self.axis_binding_digest is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_id": self.factor_id,
            "value": self.value,
            "roles": list(self.roles),
            "source_anchor_id": self.source_anchor_id,
            "axis_binding_digest": self.axis_binding_digest,
        }


@dataclass(frozen=True)
class TargetFactorConfig:
    target_id: str
    task_id: str
    axis_id: str | None
    factor_id: str
    factor_value: float
    roles: tuple[str, ...]
    regime: str
    source_anchor_ref: str | None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], where: str) -> "TargetFactorConfig":
        fields = {
            "target_id", "task_id", "axis_id", "factor_id", "factor_value", "roles",
            "regime", "source_anchor_ref",
        }
        data = _mapping(value, where)
        _strict(data, fields, where)
        target_id = _string(data["target_id"], f"{where}.target_id")
        if not _OPAQUE_TARGET_ID.fullmatch(target_id):
            raise V02ConfigError(
                f"{where}.target_id must be v02q- followed by 128-bit lowercase hex"
            )
        axis = data["axis_id"]
        if axis is not None:
            axis = _string(axis, f"{where}.axis_id", safe=True)
        factor_value = _float(data["factor_value"], f"{where}.factor_value", minimum=0.0)
        if factor_value <= 0.0:
            raise V02ConfigError(f"{where}.factor_value must be positive")
        roles = _strings(data["roles"], f"{where}.roles")
        illegal = set(roles) - FACTOR_ROLES
        if illegal:
            raise V02ConfigError(f"{where}.roles contains unknown roles: {sorted(illegal)}")
        regime = _string(data["regime"], f"{where}.regime", safe=True)
        if regime not in TARGET_REGIMES:
            raise V02ConfigError(f"{where}.regime is not registered")
        source_value = data["source_anchor_ref"]
        source_ref = None if source_value is None else _digest(
            source_value, f"{where}.source_anchor_ref"
        )
        if regime == "safety_exact" and source_ref is None:
            raise V02ConfigError("safety_exact targets require source_anchor_ref")
        if regime != "safety_exact" and source_ref is not None:
            raise V02ConfigError("held-out targets cannot carry source_anchor_ref")
        return cls(
            target_id=target_id,
            task_id=_string(data["task_id"], f"{where}.task_id", safe=True),
            axis_id=axis,
            factor_id=_string(data["factor_id"], f"{where}.factor_id", safe=True),
            factor_value=factor_value,
            roles=roles,
            regime=regime,
            source_anchor_ref=source_ref,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "task_id": self.task_id,
            "axis_id": self.axis_id,
            "factor_id": self.factor_id,
            "factor_value": self.factor_value,
            "roles": list(self.roles),
            "regime": self.regime,
            "source_anchor_ref": self.source_anchor_ref,
        }


@dataclass(frozen=True)
class SourceEvaluationConfig:
    selection_episodes: int
    attestation_episodes: int

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourceEvaluationConfig":
        data = _mapping(value, "source_eval_episodes")
        _strict(data, {"selection", "attestation"}, "source_eval_episodes")
        return cls(
            selection_episodes=_int(data["selection"], "source_eval_episodes.selection", minimum=1),
            attestation_episodes=_int(
                data["attestation"], "source_eval_episodes.attestation", minimum=1
            ),
        )

    def to_dict(self) -> dict[str, int]:
        return {"selection": self.selection_episodes, "attestation": self.attestation_episodes}


@dataclass(frozen=True)
class SourceChampionizationConfig:
    """Reviewed source-only champion selection and admission statistics."""

    mean_tolerance: float
    lcb_z: float
    competence_mode: CompetenceMode

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourceChampionizationConfig":
        data = _mapping(value, "source_championization")
        _strict(
            data,
            {"mean_tolerance", "lcb_z", "competence_mode"},
            "source_championization",
        )
        mode = _string(
            data["competence_mode"],
            "source_championization.competence_mode",
            safe=True,
        )
        if mode not in COMPETENCE_MODES:
            raise V02ConfigError(
                "source_championization.competence_mode must be OBSERVE or ENFORCE"
            )
        return cls(
            mean_tolerance=_float(
                data["mean_tolerance"],
                "source_championization.mean_tolerance",
                minimum=0.0,
            ),
            lcb_z=_float(
                data["lcb_z"],
                "source_championization.lcb_z",
                minimum=0.0,
            ),
            competence_mode=mode,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_tolerance": self.mean_tolerance,
            "lcb_z": self.lcb_z,
            "competence_mode": self.competence_mode,
        }


@dataclass(frozen=True)
class BootstrapPlan:
    resamples: int
    confidence: float
    hierarchy: tuple[str, ...]
    method: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BootstrapPlan":
        data = _mapping(value, "bootstrap_plan")
        _strict(data, {"resamples", "confidence", "hierarchy", "method"}, "bootstrap_plan")
        confidence = _float(data["confidence"], "bootstrap_plan.confidence", minimum=0.0)
        if not 0.0 < confidence < 1.0:
            raise V02ConfigError("bootstrap_plan.confidence must lie in (0, 1)")
        return cls(
            resamples=_int(data["resamples"], "bootstrap_plan.resamples", minimum=1),
            confidence=confidence,
            hierarchy=_strings(data["hierarchy"], "bootstrap_plan.hierarchy"),
            method=_string(data["method"], "bootstrap_plan.method", safe=True),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "resamples": self.resamples,
            "confidence": self.confidence,
            "hierarchy": list(self.hierarchy),
            "method": self.method,
        }


@dataclass(frozen=True)
class MultipleTestingPlan:
    simultaneous_interval: str
    p_value_adjustment: str
    alpha: float
    families: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MultipleTestingPlan":
        data = _mapping(value, "multiple_testing_plan")
        fields = {"simultaneous_interval", "p_value_adjustment", "alpha", "families"}
        _strict(data, fields, "multiple_testing_plan")
        alpha = _float(data["alpha"], "multiple_testing_plan.alpha", minimum=0.0)
        if not 0.0 < alpha < 1.0:
            raise V02ConfigError("multiple_testing_plan.alpha must lie in (0, 1)")
        return cls(
            simultaneous_interval=_string(
                data["simultaneous_interval"],
                "multiple_testing_plan.simultaneous_interval",
                safe=True,
            ),
            p_value_adjustment=_string(
                data["p_value_adjustment"],
                "multiple_testing_plan.p_value_adjustment",
                safe=True,
            ),
            alpha=alpha,
            families=_strings(data["families"], "multiple_testing_plan.families"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "simultaneous_interval": self.simultaneous_interval,
            "p_value_adjustment": self.p_value_adjustment,
            "alpha": self.alpha,
            "families": list(self.families),
        }


def _parse_targets(value: Any, where: str) -> tuple[TargetFactorConfig, ...]:
    return tuple(
        TargetFactorConfig.from_dict(item, f"{where}[{index}]")
        for index, item in enumerate(_sequence(value, where))
    )


@dataclass(frozen=True)
class V02ExperimentConfig:
    experiment_id: str
    stage: str
    protocol_family_id: str
    tasks: tuple[str, ...]
    dynamics_axes: Mapping[str, tuple[AxisConfig, ...]]
    source_factors: Mapping[str, Mapping[str, tuple[SourceFactorConfig, ...]]]
    development_targets: tuple[TargetFactorConfig, ...]
    confirmatory_targets: tuple[TargetFactorConfig, ...]
    safety_exact_targets: tuple[TargetFactorConfig, ...]
    primary_algorithm: str
    training_steps: int
    training_seeds: tuple[int, ...]
    checkpoint_rule: str
    source_eval_episodes: SourceEvaluationConfig
    competence_floor: Mapping[str, float]
    source_championization: SourceChampionizationConfig | None
    probe_protocol_id: str
    probe_prefixes: tuple[int, ...]
    encoder_eval_prefixes: tuple[int, ...]
    representation_ids: tuple[str, ...]
    method_ids: tuple[str, ...]
    primary_endpoint: str
    noninferiority_margin: float
    minimum_effect: float
    bootstrap_plan: BootstrapPlan
    multiple_testing_plan: MultipleTestingPlan
    artifact_root: str
    schema: str = V02_EXPERIMENT_CONFIG_SCHEMA

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "V02ExperimentConfig":
        data = _mapping(value, "v0.2 config")
        _strict_top_level(data, "v0.2 config")
        if data["schema"] != V02_EXPERIMENT_CONFIG_SCHEMA:
            raise V02ConfigError(f"unsupported v0.2 config schema: {data['schema']!r}")
        if _find_unresolved(data):
            # Legitimate nullable anchor/reference fields are checked below.  All
            # text review markers remain forbidden regardless of their location.
            markers = tuple(
                path for path in _find_marker_paths(data)
            )
            if markers:
                raise V02ConfigError(f"unresolved REVIEW/TBD values: {list(markers)}")

        experiment_id = _string(data["experiment_id"], "experiment_id", safe=True)
        stage = _string(data["stage"], "stage", safe=True)
        if stage not in V02_STAGES:
            raise V02ConfigError(f"unsupported v0.2 stage: {stage!r}")
        if stage == "v02_freeze_ready" and "source_championization" not in data:
            raise V02ConfigError(
                "v02_freeze_ready requires reviewed source_championization"
            )
        family = _string(data["protocol_family_id"], "protocol_family_id", safe=True)
        if family != V02_PROTOCOL_FAMILY_ID:
            raise V02ConfigError(f"protocol_family_id must be {V02_PROTOCOL_FAMILY_ID!r}")
        tasks = _strings(data["tasks"], "tasks")
        if stage == "v02_freeze_ready" and set(tasks) != FORMAL_V02_TASKS:
            raise V02ConfigError(
                "v0.2 formal scope requires the exact registered six-task set"
            )

        raw_axes = _mapping(data["dynamics_axes"], "dynamics_axes")
        if set(raw_axes) != set(tasks):
            raise V02ConfigError("dynamics_axes keys must exactly match tasks")
        axes: dict[str, tuple[AxisConfig, ...]] = {}
        for task in tasks:
            parsed = tuple(
                AxisConfig.from_dict(item, f"dynamics_axes.{task}[{index}]")
                for index, item in enumerate(_sequence(raw_axes[task], f"dynamics_axes.{task}"))
            )
            allowed_axis_count = {2} if stage == "v02_freeze_ready" else {1, 2}
            if len(parsed) not in allowed_axis_count or len({axis.axis_id for axis in parsed}) != len(parsed):
                expected = "exactly two" if stage == "v02_freeze_ready" else "one or two"
                raise V02ConfigError(f"{task} must register {expected} distinct axes")
            axes[task] = parsed

        raw_factors = _mapping(data["source_factors"], "source_factors")
        if set(raw_factors) != set(tasks):
            raise V02ConfigError("source_factors keys must exactly match tasks")
        factors: dict[str, Mapping[str, tuple[SourceFactorConfig, ...]]] = {}
        all_anchor_owners: dict[str, str] = {}
        for task in tasks:
            task_factors = _mapping(raw_factors[task], f"source_factors.{task}")
            axis_ids = {axis.axis_id for axis in axes[task]}
            if set(task_factors) != axis_ids:
                raise V02ConfigError(f"source_factors.{task} keys must exactly match its axes")
            parsed_by_axis: dict[str, tuple[SourceFactorConfig, ...]] = {}
            nominal_anchor_ids: set[str] = set()
            task_anchor_ids: set[str] = set()
            for axis_id in sorted(axis_ids):
                parsed = tuple(
                    SourceFactorConfig.from_dict(
                        item, f"source_factors.{task}.{axis_id}[{index}]"
                    )
                    for index, item in enumerate(
                        _sequence(task_factors[axis_id], f"source_factors.{task}.{axis_id}")
                    )
                )
                if stage == "v02_freeze_ready" and len(parsed) != 3:
                    raise V02ConfigError(f"{task}/{axis_id} must have exactly three source factors")
                if stage != "v02_freeze_ready" and not 2 <= len(parsed) <= 3:
                    raise V02ConfigError(
                        f"{task}/{axis_id} discovery requires nominal plus one or two endpoints"
                    )
                if len({item.factor_id for item in parsed}) != len(parsed):
                    raise V02ConfigError(f"{task}/{axis_id} factor IDs must be unique")
                values = tuple(item.value for item in parsed)
                if tuple(sorted(values)) != values or len(set(values)) != len(values):
                    raise V02ConfigError(f"{task}/{axis_id} factors must be unique and increasing")
                nominal = tuple(item for item in parsed if item.is_nominal)
                if len(nominal) != 1:
                    raise V02ConfigError(f"{task}/{axis_id} must contain exactly one factor 1.0")
                nominal_anchor_ids.add(nominal[0].source_anchor_id)
                task_anchor_ids.update(item.source_anchor_id for item in parsed)
                parsed_by_axis[axis_id] = parsed
            if len(nominal_anchor_ids) != 1:
                raise V02ConfigError(f"{task} axes must share one canonical nominal anchor")
            expected_anchor_count = 1 + sum(
                len(parsed_by_axis[axis_id]) - 1 for axis_id in parsed_by_axis
            )
            if len(task_anchor_ids) != expected_anchor_count:
                raise V02ConfigError(
                    f"{task} must resolve to {expected_anchor_count} unique source anchors"
                )
            for anchor_id in task_anchor_ids:
                owner = all_anchor_owners.setdefault(anchor_id, task)
                if owner != task:
                    raise V02ConfigError("source anchor IDs cannot be shared across tasks")
            factors[task] = MappingProxyType(parsed_by_axis)

        development = _parse_targets(data["development_targets"], "development_targets")
        confirmatory = _parse_targets(data["confirmatory_targets"], "confirmatory_targets")
        safety = _parse_targets(data["safety_exact_targets"], "safety_exact_targets")
        _validate_target_splits(
            tasks=tasks,
            axes=axes,
            source_factors=factors,
            development=development,
            confirmatory=confirmatory,
            safety=safety,
        )
        if stage == "v02_freeze_ready":
            if not development:
                raise V02ConfigError("v02_freeze_ready requires frozen development contexts")
            if confirmatory or safety:
                raise V02ConfigError(
                    "v02_freeze_ready must not instantiate Paper-I sealed or safety target IDs"
                )

        algorithm = _string(data["primary_algorithm"], "primary_algorithm", safe=True)
        if algorithm not in {"PPO", "FPO"}:
            raise V02ConfigError("primary_algorithm must be exactly PPO or FPO")
        seeds = _ints(data["training_seeds"], "training_seeds", minimum=0)
        if stage == "v02_freeze_ready" and len(seeds) != 3:
            raise V02ConfigError("formal training requires exactly three distinct seeds")
        if stage == "development_discovery" and len(seeds) != 1:
            raise V02ConfigError("development discovery requires exactly one training seed")
        source_eval = SourceEvaluationConfig.from_dict(data["source_eval_episodes"])
        source_championization = (
            None
            if "source_championization" not in data
            else SourceChampionizationConfig.from_dict(data["source_championization"])
        )

        raw_floor = _mapping(data["competence_floor"], "competence_floor")
        if set(raw_floor) != set(tasks):
            raise V02ConfigError("competence_floor keys must exactly match tasks")
        floor: dict[str, float] = {}
        for task in tasks:
            score = _float(raw_floor[task], f"competence_floor.{task}", minimum=0.0)
            if score > 1.0:
                raise V02ConfigError("competence floors must lie in [0, 1]")
            floor[task] = score

        probe_prefixes = _ints(data["probe_prefixes"], "probe_prefixes", minimum=1)
        encoder_prefixes = _ints(
            data["encoder_eval_prefixes"], "encoder_eval_prefixes", minimum=1
        )
        if tuple(sorted(probe_prefixes)) != probe_prefixes:
            raise V02ConfigError("probe_prefixes must be increasing")
        if tuple(sorted(encoder_prefixes)) != encoder_prefixes:
            raise V02ConfigError("encoder_eval_prefixes must be increasing")
        if not set(probe_prefixes).issubset(encoder_prefixes):
            raise V02ConfigError("encoder_eval_prefixes must include all primary probe prefixes")
        representations = _strings(data["representation_ids"], "representation_ids")
        if "raw_transition_v02" not in representations or not any(
            "corro" in item.lower() for item in representations
        ):
            raise V02ConfigError("representation_ids require raw and at least one CORRO-style entry")
        methods = _strings(data["method_ids"], "method_ids")
        if stage == "v02_freeze_ready" and set(methods) != FORMAL_V02_METHOD_IDS:
            raise V02ConfigError(
                "v0.2 formal scope requires B0, B1, B2, B3a, B3b, B4a, "
                "B4b, A-Env, and M02/B5 exactly"
            )

        artifact_root = _string(data["artifact_root"], "artifact_root")
        root_path = Path(artifact_root)
        if not root_path.is_absolute() or ".." in root_path.parts:
            raise V02ConfigError("artifact_root must be an absolute traversal-free path")

        result = cls(
            schema=V02_EXPERIMENT_CONFIG_SCHEMA,
            experiment_id=experiment_id,
            stage=stage,
            protocol_family_id=family,
            tasks=tasks,
            dynamics_axes=MappingProxyType(axes),
            source_factors=MappingProxyType(factors),
            development_targets=development,
            confirmatory_targets=confirmatory,
            safety_exact_targets=safety,
            primary_algorithm=algorithm,
            training_steps=_int(data["training_steps"], "training_steps", minimum=1),
            training_seeds=seeds,
            checkpoint_rule=_string(data["checkpoint_rule"], "checkpoint_rule", safe=True),
            source_eval_episodes=source_eval,
            competence_floor=MappingProxyType(floor),
            source_championization=source_championization,
            probe_protocol_id=_digest(data["probe_protocol_id"], "probe_protocol_id"),
            probe_prefixes=probe_prefixes,
            encoder_eval_prefixes=encoder_prefixes,
            representation_ids=representations,
            method_ids=methods,
            primary_endpoint=_string(data["primary_endpoint"], "primary_endpoint", safe=True),
            noninferiority_margin=_float(
                data["noninferiority_margin"], "noninferiority_margin", minimum=0.0
            ),
            minimum_effect=_float(data["minimum_effect"], "minimum_effect", minimum=0.0),
            bootstrap_plan=BootstrapPlan.from_dict(data["bootstrap_plan"]),
            multiple_testing_plan=MultipleTestingPlan.from_dict(data["multiple_testing_plan"]),
            artifact_root=artifact_root,
        )
        return result

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "experiment_id": self.experiment_id,
            "stage": self.stage,
            "protocol_family_id": self.protocol_family_id,
            "tasks": self.tasks,
            "dynamics_axes": {
                task: [axis.to_dict() for axis in self.dynamics_axes[task]] for task in self.tasks
            },
            "source_factors": {
                task: {
                    axis.axis_id: [
                        factor.to_dict() for factor in self.source_factors[task][axis.axis_id]
                    ]
                    for axis in self.dynamics_axes[task]
                }
                for task in self.tasks
            },
            "development_targets": [target.to_dict() for target in self.development_targets],
            "confirmatory_targets": [target.to_dict() for target in self.confirmatory_targets],
            "safety_exact_targets": [target.to_dict() for target in self.safety_exact_targets],
            "primary_algorithm": self.primary_algorithm,
            "training_steps": self.training_steps,
            "training_seeds": self.training_seeds,
            "checkpoint_rule": self.checkpoint_rule,
            "source_eval_episodes": self.source_eval_episodes.to_dict(),
            "competence_floor": dict(self.competence_floor),
            "probe_protocol_id": self.probe_protocol_id,
            "probe_prefixes": self.probe_prefixes,
            "encoder_eval_prefixes": self.encoder_eval_prefixes,
            "representation_ids": self.representation_ids,
            "method_ids": self.method_ids,
            "primary_endpoint": self.primary_endpoint,
            "noninferiority_margin": self.noninferiority_margin,
            "minimum_effect": self.minimum_effect,
            "bootstrap_plan": self.bootstrap_plan.to_dict(),
            "multiple_testing_plan": self.multiple_testing_plan.to_dict(),
            "artifact_root": self.artifact_root,
        }
        if self.source_championization is not None:
            payload["source_championization"] = self.source_championization.to_dict()
        return canonicalize(payload)

    @property
    def config_digest(self) -> str:
        return sha256_json(self.to_dict())

    @property
    def source_anchor_to_task(self) -> Mapping[str, str]:
        """Canonical ownership map for the deduplicated source-anchor universe."""

        owners: dict[str, str] = {}
        for task in self.tasks:
            for axis in self.dynamics_axes[task]:
                for factor in self.source_factors[task][axis.axis_id]:
                    owner = owners.setdefault(factor.source_anchor_id, task)
                    if owner != task:  # defensive: construction already rejects this
                        raise V02ConfigError("source anchor IDs cannot be shared across tasks")
        return MappingProxyType(dict(sorted(owners.items())))

    @property
    def source_anchor_ids(self) -> tuple[str, ...]:
        """All reviewed source anchors, with the shared nominal deduplicated."""

        return tuple(self.source_anchor_to_task)

    @property
    def source_competence_floor_by_anchor(self) -> Mapping[str, float]:
        """Project task-level reviewed floors onto their exact source anchors."""

        return MappingProxyType(
            {
                anchor: float(self.competence_floor[task])
                for anchor, task in self.source_anchor_to_task.items()
            }
        )

    @property
    def benchmark_projection(self) -> Mapping[str, Any]:
        payload = self.to_dict()
        keys = {
            "protocol_family_id", "tasks", "dynamics_axes", "source_factors",
            "development_targets", "confirmatory_targets", "safety_exact_targets",
        }
        return _deep_freeze({key: payload[key] for key in keys})

    @property
    def training_projection(self) -> Mapping[str, Any]:
        payload = self.to_dict()
        keys = {
            "primary_algorithm", "training_steps", "training_seeds", "checkpoint_rule",
            "source_eval_episodes", "competence_floor",
        }
        if self.source_championization is not None:
            keys.add("source_championization")
        return _deep_freeze({key: payload[key] for key in keys})

    @property
    def probe_projection(self) -> Mapping[str, Any]:
        return _deep_freeze({
            "probe_protocol_id": self.probe_protocol_id,
            "probe_prefixes": self.probe_prefixes,
            "encoder_eval_prefixes": self.encoder_eval_prefixes,
        })

    @property
    def analysis_projection(self) -> Mapping[str, Any]:
        return _deep_freeze({
            "method_ids": self.method_ids,
            "primary_endpoint": self.primary_endpoint,
            "noninferiority_margin": self.noninferiority_margin,
            "minimum_effect": self.minimum_effect,
            "bootstrap_plan": self.bootstrap_plan.to_dict(),
            "multiple_testing_plan": self.multiple_testing_plan.to_dict(),
        })


def _find_marker_paths(value: Any, path: str = "$") -> tuple[str, ...]:
    found: list[str] = []
    if _is_review_marker(value):
        found.append(path)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            found.extend(_find_marker_paths(item, f"{path}.{key}"))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            found.extend(_find_marker_paths(item, f"{path}[{index}]"))
    return tuple(found)


def _validate_target_splits(
    *,
    tasks: tuple[str, ...],
    axes: Mapping[str, tuple[AxisConfig, ...]],
    source_factors: Mapping[str, Mapping[str, tuple[SourceFactorConfig, ...]]],
    development: tuple[TargetFactorConfig, ...],
    confirmatory: tuple[TargetFactorConfig, ...],
    safety: tuple[TargetFactorConfig, ...],
) -> None:
    all_targets = development + confirmatory + safety
    target_ids = [target.target_id for target in all_targets]
    if len(set(target_ids)) != len(target_ids):
        raise V02ConfigError("target IDs must be unique across every split")
    axis_ids = {task: {axis.axis_id for axis in axes[task]} for task in tasks}
    source_by_anchor: dict[str, list[tuple[str, str, SourceFactorConfig]]] = {}
    source_value_keys: set[tuple[str, str, float]] = set()
    source_id_keys: set[tuple[str, str, str]] = set()
    for task in tasks:
        for axis_id in axis_ids[task]:
            for factor in source_factors[task][axis_id]:
                source_by_anchor.setdefault(factor.source_anchor_id, []).append(
                    (task, axis_id, factor)
                )
                source_value_keys.add((task, axis_id, factor.value))
                source_id_keys.add((task, axis_id, factor.factor_id))

    expected = (
        (development, {"development"}, {"heldout_interpolation", "heldout_extrapolation"}),
        (
            confirmatory,
            {"confirmatory_heldout"},
            {"heldout_interpolation", "heldout_extrapolation", "market_ood_boundary"},
        ),
        (safety, {"safety_exact_reference"}, {"safety_exact"}),
    )
    for group, roles, regimes in expected:
        for target in group:
            if target.task_id not in axis_ids:
                raise V02ConfigError(f"target {target.target_id} references unknown task")
            if set(target.roles) != roles:
                raise V02ConfigError(
                    f"target {target.target_id} has invalid role for its split"
                )
            if target.regime not in regimes:
                raise V02ConfigError(
                    f"target {target.target_id} has invalid regime for its split"
                )
            if target.regime != "safety_exact":
                if target.axis_id not in axis_ids[target.task_id]:
                    raise V02ConfigError(f"target {target.target_id} references unknown axis")

    development_values = {(item.task_id, item.axis_id, item.factor_value) for item in development}
    confirmatory_values = {(item.task_id, item.axis_id, item.factor_value) for item in confirmatory}
    development_ids = {(item.task_id, item.axis_id, item.factor_id) for item in development}
    confirmatory_ids = {(item.task_id, item.axis_id, item.factor_id) for item in confirmatory}
    if source_value_keys & development_values or source_id_keys & development_ids:
        raise V02ConfigError("development held-out factors overlap source factors")
    if source_value_keys & confirmatory_values or source_id_keys & confirmatory_ids:
        raise V02ConfigError("confirmatory held-out factors overlap source factors")
    if development_values & confirmatory_values or development_ids & confirmatory_ids:
        raise V02ConfigError("development and confirmatory held-out factors overlap")
    if len(development_values) != len(development) or len(confirmatory_values) != len(confirmatory):
        raise V02ConfigError("held-out splits contain duplicate task-axis-factor contexts")

    for target in safety:
        matches = source_by_anchor.get(target.source_anchor_ref or "", [])
        if not matches or any(task != target.task_id for task, _, _ in matches):
            raise V02ConfigError("safety_exact source_anchor_ref is not a source anchor for its task")
        nominal = all(factor.is_nominal for _, _, factor in matches)
        if nominal:
            if target.axis_id is not None or target.factor_value != 1.0:
                raise V02ConfigError("nominal safety_exact must use axis_id=null and factor 1.0")
        else:
            if len(matches) != 1:
                raise V02ConfigError("nonnominal source anchors must bind one axis")
            _, axis_id, factor = matches[0]
            if (
                target.axis_id != axis_id
                or target.factor_id != factor.factor_id
                or target.factor_value != factor.value
            ):
                raise V02ConfigError("safety_exact factor does not match source anchor")


@dataclass(frozen=True)
class V02ConfigDraft:
    """Immutable, non-executable record of an RFC with explicit review markers."""

    payload: Mapping[str, Any]
    unresolved_fields: tuple[str, ...]
    config_digest: str

    def __post_init__(self) -> None:
        canonical = canonicalize(self.payload)
        expected_digest = sha256_json(canonical)
        actual_digest = _digest(self.config_digest, "config_digest")
        if actual_digest != expected_digest:
            raise V02ConfigError("config_digest does not match draft payload")
        expected_unresolved = _find_unresolved(canonical)
        if tuple(self.unresolved_fields) != expected_unresolved:
            raise V02ConfigError("unresolved_fields does not match draft payload")
        object.__setattr__(self, "payload", _deep_freeze(canonical))
        object.__setattr__(self, "unresolved_fields", expected_unresolved)
        object.__setattr__(self, "config_digest", actual_digest)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "V02ConfigDraft":
        data = _mapping(value, "v0.2 config draft")
        _strict_top_level(data, "v0.2 config draft")
        if data["schema"] != V02_EXPERIMENT_CONFIG_SCHEMA:
            raise V02ConfigError(f"unsupported v0.2 config schema: {data['schema']!r}")
        if not _is_review_marker(data["stage"]) and data["stage"] not in V02_STAGES:
            raise V02ConfigError(f"unsupported v0.2 stage: {data['stage']!r}")
        family = data["protocol_family_id"]
        if not _is_review_marker(family) and family != V02_PROTOCOL_FAMILY_ID:
            raise V02ConfigError(f"protocol_family_id must be {V02_PROTOCOL_FAMILY_ID!r}")
        canonical = canonicalize(data)
        return cls(
            payload=_deep_freeze(canonical),
            unresolved_fields=_find_unresolved(canonical),
            config_digest=sha256_json(canonical),
        )

    def to_dict(self) -> dict[str, Any]:
        return canonicalize(self.payload)


def _load_yaml(path: str | Path) -> Mapping[str, Any]:
    source = Path(path)
    try:
        value = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise V02ConfigError(f"cannot load v0.2 config {source}: {exc}") from exc
    return _mapping(value, f"v0.2 config {source}")


def load_v02_config_draft(path: str | Path) -> V02ConfigDraft:
    return V02ConfigDraft.from_dict(_load_yaml(path))


def load_v02_experiment_config(path: str | Path) -> V02ExperimentConfig:
    """Load any complete v0.2 stage without supplying scientific defaults."""

    return V02ExperimentConfig.from_dict(_load_yaml(path))


def load_v02_formal_config(path: str | Path) -> V02ExperimentConfig:
    """Load an executable freeze-ready config, failing closed on every draft marker."""

    config = load_v02_experiment_config(path)
    if config.stage != "v02_freeze_ready":
        raise V02ConfigError("formal loader accepts only stage=v02_freeze_ready")
    return config


__all__ = [
    "AxisConfig",
    "BootstrapPlan",
    "COMPETENCE_MODES",
    "CompetenceMode",
    "FACTOR_ROLES",
    "FORMAL_V02_METHOD_IDS",
    "FORMAL_V02_TASKS",
    "MultipleTestingPlan",
    "REVIEW_MARKERS",
    "SourceEvaluationConfig",
    "SourceChampionizationConfig",
    "SourceFactorConfig",
    "TargetFactorConfig",
    "V02ConfigDraft",
    "V02ConfigError",
    "V02ExperimentConfig",
    "V02_EXPERIMENT_CONFIG_SCHEMA",
    "V02_PROTOCOL_FAMILY_ID",
    "load_v02_config_draft",
    "load_v02_experiment_config",
    "load_v02_formal_config",
]
