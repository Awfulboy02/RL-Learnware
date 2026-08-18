"""Strict, immutable configuration objects for ProtocolV0.

The YAML files are research inputs, so this loader deliberately fails on
unknown or missing keys instead of silently accepting protocol drift.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .hashing import sha256_json


PROTOCOL_DRAFT_SCHEMA = "policy-learnware.protocol-draft.v0"


class ConfigError(ValueError):
    """Raised when a protocol draft is incomplete or internally inconsistent."""


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{where} must be a mapping, got {type(value).__name__}")
    if not all(isinstance(key, str) for key in value):
        raise ConfigError(f"{where} keys must all be strings")
    return value


def _strict_keys(
    data: Mapping[str, Any],
    *,
    required: set[str],
    where: str,
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    missing = required - set(data)
    unknown = set(data) - required - optional
    if missing:
        raise ConfigError(f"{where} is missing keys: {sorted(missing)}")
    if unknown:
        raise ConfigError(f"{where} has unknown keys: {sorted(unknown)}")


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{where} must be a positive integer")
    return value


def _nonnegative_float(value: Any, where: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{where} must be finite")
    if (positive and result <= 0.0) or (not positive and result < 0.0):
        qualifier = "positive" if positive else "non-negative"
        raise ConfigError(f"{where} must be {qualifier}")
    return result


def _unit_interval_float(value: Any, where: str) -> float:
    result = _nonnegative_float(value, where)
    if result > 1.0:
        raise ConfigError(f"{where} must lie in [0, 1]")
    return result


def _nonempty_string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{where} must be a non-empty string")
    return value


def _string_tuple(value: Any, where: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigError(f"{where} must be a sequence of strings")
    result = tuple(_nonempty_string(item, f"{where}[]") for item in value)
    if not result:
        raise ConfigError(f"{where} cannot be empty")
    if len(set(result)) != len(result):
        raise ConfigError(f"{where} contains duplicates")
    return result


@dataclass(frozen=True)
class RuntimeConfig:
    python: str
    fpo_root: str
    fpo_commit: str
    reproduction_root: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RuntimeConfig":
        data = _mapping(value, "runtime")
        fields = {"python", "fpo_root", "fpo_commit", "reproduction_root"}
        _strict_keys(data, required=fields, where="runtime")
        return cls(**{key: _nonempty_string(data[key], f"runtime.{key}") for key in fields})


@dataclass(frozen=True)
class EnvironmentConfig:
    backend: str
    tasks: tuple[str, ...]
    horizon: int
    action_repeat: int
    max_observation_dim: int
    max_action_dim: int

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EnvironmentConfig":
        data = _mapping(value, "environment")
        fields = {
            "backend",
            "tasks",
            "horizon",
            "action_repeat",
            "max_observation_dim",
            "max_action_dim",
        }
        _strict_keys(data, required=fields, where="environment")
        return cls(
            backend=_nonempty_string(data["backend"], "environment.backend"),
            tasks=_string_tuple(data["tasks"], "environment.tasks"),
            horizon=_positive_int(data["horizon"], "environment.horizon"),
            action_repeat=_positive_int(
                data["action_repeat"], "environment.action_repeat"
            ),
            max_observation_dim=_positive_int(
                data["max_observation_dim"], "environment.max_observation_dim"
            ),
            max_action_dim=_positive_int(
                data["max_action_dim"], "environment.max_action_dim"
            ),
        )


@dataclass(frozen=True)
class ProbeConfig:
    type: str
    sigma: float
    action_low: float
    action_high: float
    rng_backend: str = "jax_threefry_full_episode_v0"

    def __post_init__(self) -> None:
        if self.type != "clipped_gaussian":
            raise ConfigError(f"unsupported probe.type: {self.type!r}")
        for name in ("sigma", "action_low", "action_high"):
            if not math.isfinite(float(getattr(self, name))):
                raise ConfigError(f"probe.{name} must be finite")
        if self.sigma <= 0:
            raise ConfigError("probe.sigma must be positive")
        if not self.action_low < self.action_high:
            raise ConfigError("probe.action_low must be less than probe.action_high")
        if self.rng_backend != "jax_threefry_full_episode_v0":
            raise ConfigError(
                "probe.rng_backend must be jax_threefry_full_episode_v0 in v0"
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProbeConfig":
        data = _mapping(value, "probe")
        fields = {"type", "sigma", "action_low", "action_high", "rng_backend"}
        _strict_keys(data, required=fields, where="probe")
        probe_type = _nonempty_string(data["type"], "probe.type")
        if probe_type != "clipped_gaussian":
            raise ConfigError(f"unsupported probe.type: {probe_type!r}")
        low = float(data["action_low"])
        high = float(data["action_high"])
        if not low < high:
            raise ConfigError("probe.action_low must be less than probe.action_high")
        return cls(
            type=probe_type,
            sigma=_nonnegative_float(data["sigma"], "probe.sigma", positive=True),
            action_low=low,
            action_high=high,
            rng_backend=_nonempty_string(
                data["rng_backend"], "probe.rng_backend"
            ),
        )


@dataclass(frozen=True)
class EpisodesConfig:
    encoder_train_per_task: int
    encoder_validation_per_task: int
    kernel_calibration_per_task: int
    separability_calibration_per_task: int
    source_taskspec_per_task: int
    target_query_max_per_task: int
    target_query_prefix_grid: tuple[int, ...]
    target_query_banks: int
    championization_per_candidate: int
    final_return_per_task: int

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EpisodesConfig":
        data = _mapping(value, "episodes")
        fields = {
            "encoder_train_per_task",
            "encoder_validation_per_task",
            "kernel_calibration_per_task",
            "separability_calibration_per_task",
            "source_taskspec_per_task",
            "target_query_max_per_task",
            "target_query_prefix_grid",
            "target_query_banks",
            "championization_per_candidate",
            "final_return_per_task",
        }
        _strict_keys(data, required=fields, where="episodes")
        scalar_fields = fields - {"target_query_prefix_grid"}
        parsed = {
            key: _positive_int(data[key], f"episodes.{key}") for key in scalar_fields
        }
        grid_raw = data["target_query_prefix_grid"]
        if isinstance(grid_raw, (str, bytes)) or not isinstance(grid_raw, Sequence):
            raise ConfigError("episodes.target_query_prefix_grid must be a sequence")
        grid = tuple(
            _positive_int(item, "episodes.target_query_prefix_grid[]")
            for item in grid_raw
        )
        if not grid or tuple(sorted(set(grid))) != grid:
            raise ConfigError(
                "episodes.target_query_prefix_grid must be non-empty, unique, and increasing"
            )
        if grid[-1] > parsed["target_query_max_per_task"]:
            raise ConfigError(
                "target_query_prefix_grid exceeds target_query_max_per_task"
            )
        return cls(target_query_prefix_grid=grid, **parsed)


@dataclass(frozen=True)
class NormalizationConfig:
    fit_split: str
    observation: str
    action: str
    reward: str
    std_floor: float
    include_next_observation: bool

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "NormalizationConfig":
        data = _mapping(value, "normalization")
        fields = {
            "fit_split",
            "observation",
            "action",
            "reward",
            "std_floor",
            "include_next_observation",
        }
        _strict_keys(data, required=fields, where="normalization")
        fit_split = _nonempty_string(data["fit_split"], "normalization.fit_split")
        if fit_split != "encoder_train":
            raise ConfigError("normalization.fit_split must be encoder_train in v0")
        expected_modes = {
            "observation": "task_balanced_valid_slot_mean_std",
            "action": "identity_after_env_normalization",
            "reward": "task_balanced_global_mean_std",
        }
        for name, expected in expected_modes.items():
            if data[name] != expected:
                raise ConfigError(f"normalization.{name} must be {expected} in v0")
        if data["include_next_observation"] is not True:
            raise ConfigError("normalization.include_next_observation must be true in v0")
        return cls(
            fit_split=fit_split,
            observation=_nonempty_string(
                data["observation"], "normalization.observation"
            ),
            action=_nonempty_string(data["action"], "normalization.action"),
            reward=_nonempty_string(data["reward"], "normalization.reward"),
            std_floor=_nonnegative_float(
                data["std_floor"], "normalization.std_floor", positive=True
            ),
            include_next_observation=True,
        )


@dataclass(frozen=True)
class EncoderConfig:
    framework: str
    hidden_dims: tuple[int, ...]
    latent_dim: int
    activation: str
    l2_normalize_output: bool
    objective: str
    temperature: float
    batch_size: int
    train_steps: int
    learning_rate: float
    weight_decay: float
    validation_interval: int
    checkpoint_metric: str
    checkpoint_tie_break: str
    seed: int
    validation_batches: int

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EncoderConfig":
        data = _mapping(value, "encoder")
        fields = {
            "framework",
            "hidden_dims",
            "latent_dim",
            "activation",
            "l2_normalize_output",
            "objective",
            "temperature",
            "batch_size",
            "train_steps",
            "learning_rate",
            "weight_decay",
            "validation_interval",
            "checkpoint_metric",
            "checkpoint_tie_break",
            "seed",
            "validation_batches",
        }
        _strict_keys(data, required=fields, where="encoder")
        hidden_raw = data["hidden_dims"]
        if isinstance(hidden_raw, (str, bytes)) or not isinstance(hidden_raw, Sequence):
            raise ConfigError("encoder.hidden_dims must be a sequence")
        hidden_dims = tuple(
            _positive_int(item, "encoder.hidden_dims[]") for item in hidden_raw
        )
        if not hidden_dims:
            raise ConfigError("encoder.hidden_dims cannot be empty")
        if not isinstance(data["l2_normalize_output"], bool):
            raise ConfigError("encoder.l2_normalize_output must be boolean")
        if data["framework"] != "flax":
            raise ConfigError("encoder.framework must be flax in v0")
        if data["activation"] != "relu":
            raise ConfigError("encoder.activation must be relu in v0")
        if data["objective"] != "supervised_contrastive":
            raise ConfigError("encoder.objective must be supervised_contrastive in v0")
        if data["l2_normalize_output"] is not True:
            raise ConfigError("encoder.l2_normalize_output must be true in v0")
        checkpoint_metric = _nonempty_string(
            data["checkpoint_metric"], "encoder.checkpoint_metric"
        )
        checkpoint_tie_break = _nonempty_string(
            data["checkpoint_tie_break"], "encoder.checkpoint_tie_break"
        )
        if checkpoint_metric != "validation_supcon_loss":
            raise ConfigError(
                "encoder.checkpoint_metric must be validation_supcon_loss in v0"
            )
        if checkpoint_tie_break != "earliest_step":
            raise ConfigError(
                "encoder.checkpoint_tie_break must be earliest_step in v0"
            )
        return cls(
            framework=_nonempty_string(data["framework"], "encoder.framework"),
            hidden_dims=hidden_dims,
            latent_dim=_positive_int(data["latent_dim"], "encoder.latent_dim"),
            activation=_nonempty_string(data["activation"], "encoder.activation"),
            l2_normalize_output=data["l2_normalize_output"],
            objective=_nonempty_string(data["objective"], "encoder.objective"),
            temperature=_nonnegative_float(
                data["temperature"], "encoder.temperature", positive=True
            ),
            batch_size=_positive_int(data["batch_size"], "encoder.batch_size"),
            train_steps=_positive_int(data["train_steps"], "encoder.train_steps"),
            learning_rate=_nonnegative_float(
                data["learning_rate"], "encoder.learning_rate", positive=True
            ),
            weight_decay=_nonnegative_float(
                data["weight_decay"], "encoder.weight_decay"
            ),
            validation_interval=_positive_int(
                data["validation_interval"], "encoder.validation_interval"
            ),
            checkpoint_metric=checkpoint_metric,
            checkpoint_tie_break=checkpoint_tie_break,
            seed=int(data["seed"]),
            validation_batches=_positive_int(
                data["validation_batches"], "encoder.validation_batches"
            ),
        )


@dataclass(frozen=True)
class KernelConfig:
    type: str
    bandwidth: str
    calibration_pairs: int
    seed: int
    pair_sampling: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "KernelConfig":
        data = _mapping(value, "kernel")
        fields = {"type", "bandwidth", "calibration_pairs", "seed", "pair_sampling"}
        _strict_keys(data, required=fields, where="kernel")
        kernel_type = _nonempty_string(data["type"], "kernel.type")
        if kernel_type != "gaussian":
            raise ConfigError(f"unsupported kernel.type: {kernel_type!r}")
        if data["bandwidth"] != "source_balanced_median":
            raise ConfigError("kernel.bandwidth must be source_balanced_median in v0")
        if data["pair_sampling"] != "uniform_task_episode_transition":
            raise ConfigError(
                "kernel.pair_sampling must be uniform_task_episode_transition in v0"
            )
        return cls(
            type=kernel_type,
            bandwidth=_nonempty_string(data["bandwidth"], "kernel.bandwidth"),
            calibration_pairs=_positive_int(
                data["calibration_pairs"], "kernel.calibration_pairs"
            ),
            seed=int(data["seed"]),
            pair_sampling=str(data["pair_sampling"]),
        )


@dataclass(frozen=True)
class ReducerConfig:
    objective: str
    support_budget: int
    init: str
    support_steps: int
    learning_rate: float
    pinv_rcond: float
    ridge: float
    reconstruction_tolerance: float
    reconstruction_tolerance_metric: str
    kmeans_steps: int
    optimizer_backend: str
    negative_tolerance: float

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReducerConfig":
        data = _mapping(value, "reducer")
        fields = {
            "objective",
            "support_budget",
            "init",
            "support_steps",
            "learning_rate",
            "pinv_rcond",
            "ridge",
            "reconstruction_tolerance",
            "reconstruction_tolerance_metric",
            "kmeans_steps",
            "optimizer_backend",
            "negative_tolerance",
        }
        _strict_keys(data, required=fields, where="reducer")
        objective = _nonempty_string(data["objective"], "reducer.objective")
        tolerance_metric = _nonempty_string(
            data["reconstruction_tolerance_metric"],
            "reducer.reconstruction_tolerance_metric",
        )
        if objective != "weighted_kme_ridge":
            raise ConfigError("reducer.objective must be weighted_kme_ridge in v0")
        if tolerance_metric != "rkhs_norm":
            raise ConfigError(
                "reducer.reconstruction_tolerance_metric must be rkhs_norm in v0"
            )
        if data["init"] != "weighted_kmeans":
            raise ConfigError("reducer.init must be weighted_kmeans in v0")
        if data["optimizer_backend"] not in {"jax", "numpy"}:
            raise ConfigError("reducer.optimizer_backend must be jax or numpy")
        return cls(
            objective=objective,
            support_budget=_positive_int(
                data["support_budget"], "reducer.support_budget"
            ),
            init=_nonempty_string(data["init"], "reducer.init"),
            support_steps=_positive_int(
                data["support_steps"], "reducer.support_steps"
            ),
            learning_rate=_nonnegative_float(
                data["learning_rate"], "reducer.learning_rate", positive=True
            ),
            pinv_rcond=_nonnegative_float(
                data["pinv_rcond"], "reducer.pinv_rcond", positive=True
            ),
            ridge=_nonnegative_float(data["ridge"], "reducer.ridge"),
            reconstruction_tolerance=_nonnegative_float(
                data["reconstruction_tolerance"],
                "reducer.reconstruction_tolerance",
                positive=True,
            ),
            reconstruction_tolerance_metric=tolerance_metric,
            kmeans_steps=_positive_int(data["kmeans_steps"], "reducer.kmeans_steps"),
            optimizer_backend=str(data["optimizer_backend"]),
            negative_tolerance=_nonnegative_float(
                data["negative_tolerance"], "reducer.negative_tolerance"
            ),
        )


@dataclass(frozen=True)
class SelectorConfig:
    negative_tolerance: float
    tie_break: str
    fallback: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SelectorConfig":
        data = _mapping(value, "selector")
        fields = {"negative_tolerance", "tie_break", "fallback"}
        _strict_keys(data, required=fields, where="selector")
        if data["tie_break"] != "opaque_id_lexical":
            raise ConfigError("selector.tie_break must be opaque_id_lexical in v0")
        if data["fallback"] != "none":
            raise ConfigError("selector.fallback must be none in v0")
        return cls(
            negative_tolerance=_nonnegative_float(
                data["negative_tolerance"], "selector.negative_tolerance"
            ),
            tie_break=str(data["tie_break"]),
            fallback=str(data["fallback"]),
        )


@dataclass(frozen=True)
class UnreducedGateConfig:
    minimum_between_within_ratio: float
    minimum_absolute_margin: float
    minimum_split_retrieval_accuracy: float

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "UnreducedGateConfig":
        data = _mapping(value, "gates.unreduced")
        fields = {
            "minimum_between_within_ratio",
            "minimum_absolute_margin",
            "minimum_split_retrieval_accuracy",
        }
        _strict_keys(data, required=fields, where="gates.unreduced")
        ratio = _nonnegative_float(
            data["minimum_between_within_ratio"],
            "gates.unreduced.minimum_between_within_ratio",
            positive=True,
        )
        if ratio < 1.0:
            raise ConfigError(
                "gates.unreduced.minimum_between_within_ratio must be at least 1"
            )
        return cls(
            minimum_between_within_ratio=ratio,
            minimum_absolute_margin=_nonnegative_float(
                data["minimum_absolute_margin"],
                "gates.unreduced.minimum_absolute_margin",
                positive=True,
            ),
            minimum_split_retrieval_accuracy=_unit_interval_float(
                data["minimum_split_retrieval_accuracy"],
                "gates.unreduced.minimum_split_retrieval_accuracy",
            ),
        )


@dataclass(frozen=True)
class RankingGateConfig:
    minimum_top1_agreement: float

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RankingGateConfig":
        data = _mapping(value, "gates.reduced_unreduced_ranking")
        fields = {"minimum_top1_agreement"}
        _strict_keys(
            data,
            required=fields,
            where="gates.reduced_unreduced_ranking",
        )
        return cls(
            minimum_top1_agreement=_unit_interval_float(
                data["minimum_top1_agreement"],
                "gates.reduced_unreduced_ranking.minimum_top1_agreement",
            )
        )


@dataclass(frozen=True)
class RetrievalGateConfig:
    minimum_max_prefix_accuracy: float

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RetrievalGateConfig":
        data = _mapping(value, "gates.retrieval")
        fields = {"minimum_max_prefix_accuracy"}
        _strict_keys(data, required=fields, where="gates.retrieval")
        return cls(
            minimum_max_prefix_accuracy=_unit_interval_float(
                data["minimum_max_prefix_accuracy"],
                "gates.retrieval.minimum_max_prefix_accuracy",
            )
        )


@dataclass(frozen=True)
class DeploymentGateConfig:
    minimum_correct_retrieval_deployability_rate: float

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DeploymentGateConfig":
        data = _mapping(value, "gates.deployment")
        fields = {"minimum_correct_retrieval_deployability_rate"}
        _strict_keys(data, required=fields, where="gates.deployment")
        return cls(
            minimum_correct_retrieval_deployability_rate=_unit_interval_float(
                data["minimum_correct_retrieval_deployability_rate"],
                "gates.deployment.minimum_correct_retrieval_deployability_rate",
            )
        )


@dataclass(frozen=True)
class GatesConfig:
    unreduced: UnreducedGateConfig
    reduced_unreduced_ranking: RankingGateConfig
    retrieval: RetrievalGateConfig
    deployment: DeploymentGateConfig

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GatesConfig":
        data = _mapping(value, "gates")
        fields = {
            "unreduced",
            "reduced_unreduced_ranking",
            "retrieval",
            "deployment",
        }
        _strict_keys(data, required=fields, where="gates")
        return cls(
            unreduced=UnreducedGateConfig.from_dict(data["unreduced"]),
            reduced_unreduced_ranking=RankingGateConfig.from_dict(
                data["reduced_unreduced_ranking"]
            ),
            retrieval=RetrievalGateConfig.from_dict(data["retrieval"]),
            deployment=DeploymentGateConfig.from_dict(data["deployment"]),
        )


@dataclass(frozen=True)
class PolicyConfig:
    parity_atol: float
    parity_rtol: float
    golden_sample_count: int
    golden_parity_on_load: bool
    require_runtime_commit_match: bool
    verify_module_origin: bool
    dependency_mode: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PolicyConfig":
        data = _mapping(value, "policy")
        fields = {
            "parity_atol",
            "parity_rtol",
            "golden_sample_count",
            "golden_parity_on_load",
            "require_runtime_commit_match",
            "verify_module_origin",
            "dependency_mode",
        }
        _strict_keys(data, required=fields, where="policy")
        for flag in (
            "golden_parity_on_load",
            "require_runtime_commit_match",
            "verify_module_origin",
        ):
            if not isinstance(data[flag], bool):
                raise ConfigError(f"policy.{flag} must be boolean")
            if data[flag] is not True:
                raise ConfigError(f"policy.{flag} must be true in v0")
        golden_sample_count = _positive_int(
            data["golden_sample_count"], "policy.golden_sample_count"
        )
        if golden_sample_count != 8:
            raise ConfigError("policy.golden_sample_count must be 8 in v0")
        if data["dependency_mode"] != "locked_upstream_source":
            raise ConfigError(
                "policy.dependency_mode must be locked_upstream_source in v0"
            )
        return cls(
            parity_atol=_nonnegative_float(data["parity_atol"], "policy.parity_atol"),
            parity_rtol=_nonnegative_float(data["parity_rtol"], "policy.parity_rtol"),
            golden_sample_count=golden_sample_count,
            golden_parity_on_load=data["golden_parity_on_load"],
            require_runtime_commit_match=data["require_runtime_commit_match"],
            verify_module_origin=data["verify_module_origin"],
            dependency_mode=str(data["dependency_mode"]),
        )


@dataclass(frozen=True)
class PoolConfig:
    pool_id: str
    checkpoint_outer: int
    actual_environment_steps: int
    candidates_per_task: int

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PoolConfig":
        data = _mapping(value, "pool")
        fields = {
            "pool_id",
            "checkpoint_outer",
            "actual_environment_steps",
            "candidates_per_task",
        }
        _strict_keys(data, required=fields, where="pool")
        candidates_per_task = _positive_int(
            data["candidates_per_task"], "pool.candidates_per_task"
        )
        if candidates_per_task != 10:
            raise ConfigError(
                "pool.candidates_per_task must be 10 (PPO/FPO x seeds 0..4) in v0"
            )
        return cls(
            pool_id=_nonempty_string(data["pool_id"], "pool.pool_id"),
            checkpoint_outer=_positive_int(
                data["checkpoint_outer"], "pool.checkpoint_outer"
            ),
            actual_environment_steps=_positive_int(
                data["actual_environment_steps"], "pool.actual_environment_steps"
            ),
            candidates_per_task=candidates_per_task,
        )


@dataclass(frozen=True)
class ProtocolDraft:
    schema: str
    project_seed: int
    runtime: RuntimeConfig
    environment: EnvironmentConfig
    probe: ProbeConfig
    episodes: EpisodesConfig
    normalization: NormalizationConfig
    encoder: EncoderConfig
    kernel: KernelConfig
    reducer: ReducerConfig
    pool: PoolConfig
    selector: SelectorConfig
    gates: GatesConfig
    policy: PolicyConfig

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProtocolDraft":
        data = _mapping(value, "protocol")
        fields = {
            "schema",
            "project_seed",
            "runtime",
            "environment",
            "probe",
            "episodes",
            "normalization",
            "encoder",
            "kernel",
            "reducer",
            "pool",
            "selector",
            "gates",
            "policy",
        }
        _strict_keys(data, required=fields, where="protocol")
        schema = _nonempty_string(data["schema"], "schema")
        if schema != PROTOCOL_DRAFT_SCHEMA:
            raise ConfigError(
                f"unsupported protocol schema {schema!r}; expected {PROTOCOL_DRAFT_SCHEMA!r}"
            )
        project_seed = data["project_seed"]
        if isinstance(project_seed, bool) or not isinstance(project_seed, int):
            raise ConfigError("project_seed must be an integer")
        if not 0 <= project_seed < 2**63:
            raise ConfigError("project_seed must lie in [0, 2**63)")
        result = cls(
            schema=schema,
            project_seed=project_seed,
            runtime=RuntimeConfig.from_dict(data["runtime"]),
            environment=EnvironmentConfig.from_dict(data["environment"]),
            probe=ProbeConfig.from_dict(data["probe"]),
            episodes=EpisodesConfig.from_dict(data["episodes"]),
            normalization=NormalizationConfig.from_dict(data["normalization"]),
            encoder=EncoderConfig.from_dict(data["encoder"]),
            kernel=KernelConfig.from_dict(data["kernel"]),
            reducer=ReducerConfig.from_dict(data["reducer"]),
            pool=PoolConfig.from_dict(data["pool"]),
            selector=SelectorConfig.from_dict(data["selector"]),
            gates=GatesConfig.from_dict(data["gates"]),
            policy=PolicyConfig.from_dict(data["policy"]),
        )
        result.validate_cross_fields()
        return result

    def validate_cross_fields(self) -> None:
        if self.encoder.batch_size < 2 * len(self.environment.tasks):
            raise ConfigError(
                "encoder.batch_size must permit at least two events per task"
            )
        if self.episodes.encoder_train_per_task < 2:
            raise ConfigError("encoder_train_per_task must be at least two")
        if self.episodes.encoder_validation_per_task < 2:
            raise ConfigError("encoder_validation_per_task must be at least two")
        if self.episodes.separability_calibration_per_task < 2:
            raise ConfigError(
                "separability_calibration_per_task must be at least two"
            )
        if self.encoder.seed != self.project_seed or self.kernel.seed != self.project_seed:
            raise ConfigError("encoder.seed and kernel.seed must equal project_seed in v0")
        if self.pool.checkpoint_outer != 6:
            raise ConfigError("v0 main pool is fixed to checkpoint outer 6")
        if self.pool.actual_environment_steps != 5_898_240:
            raise ConfigError("outer 6 must map to 5,898,240 environment steps")

    @property
    def effective_task_balanced_batch_size(self) -> int:
        """Largest complete per-task-balanced batch not above the request."""

        task_count = len(self.environment.tasks)
        return (self.encoder.batch_size // task_count) * task_count

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def draft_hash(self) -> str:
        return sha256_json(self.to_dict())


def load_protocol_draft(path: str | Path) -> ProtocolDraft:
    """Load a strict ProtocolV0 draft from YAML without mutating it."""

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency failure message
        raise RuntimeError("PyYAML is required to load protocol YAML") from exc

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return ProtocolDraft.from_dict(_mapping(data, str(config_path)))
