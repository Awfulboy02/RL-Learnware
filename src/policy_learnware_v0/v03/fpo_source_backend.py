"""Real v0.2 FPO/PPO + MJX backend for v0.3 source evaluation.

This module is intentionally separate from :mod:`source_evaluator`.  The
latter owns the public work-unit/attempt contract; this module owns the private
runtime boundary needed to execute an accepted v0.2 policy on the exact source
anchor that produced it.

The important efficiency detail is that ``SourceEvaluatorBackend`` is called
once per episode, whereas the frozen v0.2 evaluator is a batched JAX program.
``FpoJaxSourceEvaluatorBackend`` therefore evaluates the complete frozen seed
block on the first episode request and serves the remaining literal seeds from
an in-memory cache.  It never invents a seed block or reads one from results.

Stored-golden and scalar/compiled parity are quality metrics, not admission
gates.  A policy is unusable only when it cannot be loaded, emits incompatible
or non-finite values, or fails a real rollout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from ..hashing import sha256_file, sha256_json
from ..policy.bundle import PolicyBundleMetadata, validate_bundle
from ..policy.evaluate import (
    evaluate_frozen_policy_returns_batched,
    verify_compiled_policy_parity,
)
from ..policy.loader import load_policy
from ..policy.parity import verify_golden_parity
from ..v02 import runtime as v02_runtime
from ..v02.schemas import ExecutionABIRecord
from .source_evaluator import (
    BackendEpisodeResult,
    SourceCandidateRequest,
    SourceEvaluatorError,
    ValidatedSourceBinding,
)


class FpoSourceBackendError(SourceEvaluatorError):
    """The frozen FPO/JAX runtime or its immutable inputs drifted."""


FPO_JAX_BACKEND_SCHEMA = "policy-learnware.v03-fpo-jax-source-backend.v1"
FPO_JAX_DRIVER_SCHEMA = "policy-learnware.v03-frozen-v02-fpo-jax-driver.v1"
POLICY_SEED_RULE = "reset_seed_plus_1000003_uint32"
POLICY_SEED_OFFSET = 1_000_003
PARITY_ATOL = 1.0e-6
PARITY_RTOL = 1.0e-6
COMPILED_PARITY_SAMPLE_COUNT = 2
LEGACY_POLICY_RUNTIME_ID = "legacy-ppo-fpo-v0"
STATE_ABI_ID = "stateless-v0"
ACTION_TRANSFORM_ID = "tanh"
PROTOCOL_FAMILY_ID = "continuous-vector-mdp-v02"
MAX_UINT32 = 2**32 - 1


def _digest(value: Any, where: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        raise FpoSourceBackendError(f"{where} must be a lowercase SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise FpoSourceBackendError(
            f"{where} must be a lowercase SHA-256 digest"
        ) from error
    return value


def _seed_block(value: Sequence[int], where: str) -> tuple[int, ...]:
    try:
        result = tuple(value)
    except TypeError as error:
        raise FpoSourceBackendError(f"{where} must be a seed sequence") from error
    if (
        not result
        or any(
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed < 0
            or seed + POLICY_SEED_OFFSET > MAX_UINT32
            for seed in result
        )
        or result != tuple(sorted(set(result)))
    ):
        raise FpoSourceBackendError(
            f"{where} must be sorted unique uint32-compatible non-negative seeds"
        )
    return result


def _execution_abi(metadata: PolicyBundleMetadata) -> ExecutionABIRecord:
    """Derive the task-anonymous tensor calling convention from bundle bytes."""

    observation = sha256_json(
        {
            "schema": "policy-learnware.v02-observation-compatibility.v0",
            "dimension": metadata.observation_dim,
            "dtype": "float32",
        }
    )
    action = sha256_json(
        {
            "schema": "policy-learnware.v02-action-compatibility.v0",
            "dimension": metadata.action_dim,
            "dtype": "float32",
            "low": [-1.0] * metadata.action_dim,
            "high": [1.0] * metadata.action_dim,
        }
    )
    return ExecutionABIRecord(
        protocol_family_id=PROTOCOL_FAMILY_ID,
        observation_tensor_abi_digest=observation,
        action_tensor_abi_digest=action,
        action_transform_id=ACTION_TRANSFORM_ID,
        policy_runtime_id=LEGACY_POLICY_RUNTIME_ID,
        state_abi_id=STATE_ABI_ID,
    )


@runtime_checkable
class FpoJaxRuntimeDriver(Protocol):
    """Private driver used by the source-backend adapter.

    A test double receives its own distinct ``runtime_driver_digest`` and thus
    cannot produce a work unit under the production evaluator digest.
    """

    runtime_driver_digest: str

    def validate_candidate(self, request: SourceCandidateRequest) -> ExecutionABIRecord: ...

    def evaluate_seed_block(
        self,
        request: SourceCandidateRequest,
        *,
        reset_seeds: tuple[int, ...],
    ) -> tuple[BackendEpisodeResult, ...]: ...


@dataclass(frozen=True)
class _PreparedCandidate:
    request_digest: str
    metadata: PolicyBundleMetadata
    anchor: Any
    execution_abi: ExecutionABIRecord


class FrozenV02FpoJaxRuntimeDriver:
    """Production driver for the frozen v0.2 FPO checkout and source anchors.

    Construction is read-only and does not initialize JAX or reserve a GPU.
    JAX imports, policy restoration, parity checks, and rollouts happen only in
    :meth:`evaluate_seed_block`.
    """

    def __init__(
        self,
        *,
        fpo_root: str | Path,
        allow_reconstructed_runtime: bool = False,
    ) -> None:
        if not sys.dont_write_bytecode:
            raise FpoSourceBackendError(
                "production source evaluation requires PYTHONDONTWRITEBYTECODE=1/-B "
                "so imports cannot dirty the attested FPO checkout"
            )
        if type(allow_reconstructed_runtime) is not bool:
            raise FpoSourceBackendError(
                "allow_reconstructed_runtime must be an explicit boolean"
            )
        try:
            if not allow_reconstructed_runtime:
                v02_runtime.require_original_vendor_runtime()
            self._fpo_root = self._absolute_directory(fpo_root, "fpo_root")
            self._source_attestation = MappingProxyType(
                dict(v02_runtime.verify_fpo_checkout(self._fpo_root))
            )
            self._original_vendor = MappingProxyType(
                dict(v02_runtime.original_vendor_status())
            )
        except v02_runtime.RuntimeVerificationError as error:
            raise FpoSourceBackendError(str(error)) from error
        self._upstream: v02_runtime.VerifiedFPOUpstream | None = None
        implementation = {
            "schema": FPO_JAX_DRIVER_SCHEMA,
            "runtime_bridge_file_sha256": sha256_file(
                Path(v02_runtime.__file__).resolve()
            ),
            "provenance_class": v02_runtime.RECONSTRUCTED_RUNTIME,
            "source_attestation": dict(self._source_attestation),
            "original_vendor": dict(self._original_vendor),
            "python_dont_write_bytecode": True,
        }
        self.runtime_driver_digest = sha256_json(implementation)
        self._prepared: dict[str, _PreparedCandidate] = {}
        self._quality_metrics: list[dict[str, Any]] = []

    def preflight(self) -> Mapping[str, Any]:
        """Import the attested checkout once as reconstructed inference only."""

        if self._upstream is None:
            try:
                self._upstream = v02_runtime.load_verified_fpo_upstream(
                    self._fpo_root,
                    allow_reconstructed=True,
                )
            except v02_runtime.RuntimeVerificationError as error:
                raise FpoSourceBackendError(str(error)) from error
        return self.runtime_evidence()

    def runtime_evidence(self) -> Mapping[str, Any]:
        """Return portable provenance without claiming original-runtime replay."""

        runtime_receipt = (
            None
            if self._upstream is None
            else dict(self._upstream.runtime_receipt)
        )
        return MappingProxyType(
            {
                "provenance_class": v02_runtime.RECONSTRUCTED_RUNTIME,
                "source_attestation": dict(self._source_attestation),
                "original_vendor": dict(self._original_vendor),
                "runtime_receipt": runtime_receipt,
                "preflight_complete": self._upstream is not None,
                "runtime_driver_digest": self.runtime_driver_digest,
            }
        )

    def drain_quality_metrics(self) -> tuple[Mapping[str, Any], ...]:
        """Return and clear non-blocking parity diagnostics."""

        metrics = tuple(MappingProxyType(dict(row)) for row in self._quality_metrics)
        self._quality_metrics.clear()
        return metrics

    @staticmethod
    def _absolute_directory(path: str | Path, where: str) -> Path:
        supplied = Path(path).expanduser()
        if not supplied.is_absolute() or supplied.is_symlink():
            raise FpoSourceBackendError(
                f"{where} must be an absolute, non-symlink directory"
            )
        try:
            resolved = supplied.resolve(strict=True)
        except OSError as error:
            raise FpoSourceBackendError(f"{where} does not exist") from error
        if not resolved.is_dir():
            raise FpoSourceBackendError(f"{where} must be a directory")
        return resolved

    def validate_candidate(self, request: SourceCandidateRequest) -> ExecutionABIRecord:
        from server.repro_fpo_ppo_v02.anchor_binding import AnchorManifest

        if not isinstance(request, SourceCandidateRequest):
            raise FpoSourceBackendError("runtime driver requires SourceCandidateRequest")
        anchor = AnchorManifest.from_path(request.anchor.manifest_path)
        if (
            anchor.manifest_digest != request.anchor.manifest_digest
            or anchor.anchor_id != request.source_anchor_id
            or anchor.environment_instance_digest != request.source_environment_digest
            or anchor.runtime_digest != request.anchor.runtime_digest
        ):
            raise FpoSourceBackendError("full anchor manifest differs from source request")
        runtime_commit = anchor.runtime["fpo_commit"]
        metadata = validate_bundle(
            request.bundle_path,
            expected_task=anchor.task,
            expected_outer=request.outer_iteration,
            expected_environment_steps=request.environment_steps,
            expected_fpo_commit=runtime_commit,
            expected_runtime_digest=anchor.runtime_digest,
            runtime_only=True,
        )
        if metadata.bundle_digest != request.bundle_digest:
            raise FpoSourceBackendError("policy bundle bytes differ from exact-90 intake")
        abi = _execution_abi(metadata)
        self._prepared[request.request_digest] = _PreparedCandidate(
            request_digest=request.request_digest,
            metadata=metadata,
            anchor=anchor,
            execution_abi=abi,
        )
        return abi

    @staticmethod
    def _restore_factory(
        *,
        prepared: _PreparedCandidate,
        bound: Any,
        jax: Any,
        jdc: Any,
        jnp: Any,
        fpo: Any,
        ppo: Any,
    ) -> Any:
        def restore(
            metadata: PolicyBundleMetadata,
            actor: Mapping[str, np.ndarray],
            obs_stats: Mapping[str, np.ndarray],
            _fpo_root: Path,
        ) -> Any:
            module = fpo if metadata.algorithm == "fpo" else ppo
            config_name = "FpoConfig" if metadata.algorithm == "fpo" else "PpoConfig"
            state_name = "FpoState" if metadata.algorithm == "fpo" else "PpoState"
            config = getattr(module, config_name)(
                **dict(metadata.policy_spec["training_config"])
            )
            state = getattr(module, state_name).init(
                prng=jax.random.key(0), env=bound.env, config=config
            )
            kernel_names = sorted(name for name in actor if name.endswith("_kernel"))
            if not kernel_names:
                raise FpoSourceBackendError("restored actor has no ordered layers")
            with jdc.copy_and_mutate(state) as restored:
                restored.params.policy = tuple(
                    (
                        jnp.asarray(actor[name]),
                        jnp.asarray(actor[name.replace("_kernel", "_bias")]),
                    )
                    for name in kernel_names
                )
                for name in ("count", "mean", "var_sum", "std"):
                    setattr(restored.obs_stats, name, jnp.asarray(obs_stats[name]))
            if restored.env is not bound.env:
                raise FpoSourceBackendError(
                    "restored policy escaped its digest-bound source environment"
                )
            return restored

        return restore

    @staticmethod
    def _environment_action_size(environment: Any) -> int:
        for name in ("action_size", "action_dim"):
            value = getattr(environment, name, None)
            if value is not None:
                result = int(value() if callable(value) else value)
                if result > 0:
                    return result
        raise FpoSourceBackendError("source environment has no positive action size")

    @staticmethod
    def _require_upstream_origin(module: Any, source_root: Path, where: str) -> None:
        raw = getattr(module, "__file__", None)
        if raw is None:
            raise FpoSourceBackendError(f"{where} module has no source origin")
        try:
            Path(raw).resolve().relative_to(source_root)
        except ValueError as error:
            raise FpoSourceBackendError(
                f"{where} was imported from another FPO checkout: {raw}"
            ) from error

    @staticmethod
    def _require_unit_action_bounds(environment: Any, jax: Any, action_dim: int) -> None:
        model = getattr(environment, "_mjx_model", None)
        ranges = getattr(model, "actuator_ctrlrange", None)
        if ranges is None:
            raise FpoSourceBackendError(
                "source environment does not expose actuator control ranges"
            )
        values = np.asarray(jax.device_get(ranges), dtype=np.float32)
        expected = np.column_stack(
            (
                -np.ones(action_dim, dtype=np.float32),
                np.ones(action_dim, dtype=np.float32),
            )
        )
        if values.shape != (action_dim, 2) or not np.array_equal(values, expected):
            raise FpoSourceBackendError(
                "source environment action bounds differ from the frozen tanh ABI"
            )

    def evaluate_seed_block(
        self,
        request: SourceCandidateRequest,
        *,
        reset_seeds: tuple[int, ...],
    ) -> tuple[BackendEpisodeResult, ...]:
        from server.repro_fpo_ppo_v02.anchor_binding import load_and_bind_anchor

        seeds = _seed_block(reset_seeds, "reset_seeds")
        # Revalidate all immutable inputs immediately before device execution.
        abi = self.validate_candidate(request)
        prepared = self._prepared[request.request_digest]
        if abi != prepared.execution_abi:
            raise FpoSourceBackendError("candidate ABI drifted during revalidation")
        try:
            self.preflight()
            if self._upstream is None:  # defensive; preflight must populate it
                raise FpoSourceBackendError("reconstructed runtime preflight failed")
            upstream = self._upstream
            jax = upstream.jax
            jdc = upstream.jax_dataclasses
            jnp = upstream.jax_numpy
            registry = upstream.registry
            fpo = upstream.fpo
            ppo = upstream.ppo
            source_root = (self._fpo_root / "playground" / "src").resolve()
            self._require_upstream_origin(fpo, source_root, "flow_policy.fpo")
            self._require_upstream_origin(ppo, source_root, "flow_policy.ppo")
            bound = load_and_bind_anchor(registry=registry, manifest=prepared.anchor)
            if self._environment_action_size(bound.env) != prepared.metadata.action_dim:
                raise FpoSourceBackendError(
                    "policy action dimension differs from the bound source environment"
                )
            self._require_unit_action_bounds(
                bound.env, jax, prepared.metadata.action_dim
            )
            restore = self._restore_factory(
                prepared=prepared,
                bound=bound,
                jax=jax,
                jdc=jdc,
                jnp=jnp,
                fpo=fpo,
                ppo=ppo,
            )
            policy = load_policy(
                prepared.metadata,
                fpo_root=self._fpo_root,
                runtime_factory=restore,
            )
            golden = verify_golden_parity(
                policy,
                prepared.metadata,
                atol=PARITY_ATOL,
                rtol=PARITY_RTOL,
            )
            if (
                (
                    golden.raw_max_abs_error is not None
                    and not np.isfinite(golden.raw_max_abs_error)
                )
                or not np.isfinite(golden.environment_max_abs_error)
            ):
                raise FpoSourceBackendError(
                    "reloaded source policy produced non-finite parity diagnostics"
                )
            with np.load(
                prepared.metadata.bundle_dir / "golden_io.npz", allow_pickle=False
            ) as archive:
                observations = np.asarray(archive["observation"])
                key_data = np.asarray(archive["prng_key_data"])
            compiled = verify_compiled_policy_parity(
                policy,
                observations,
                key_data,
                atol=PARITY_ATOL,
                rtol=PARITY_RTOL,
                sample_count=COMPILED_PARITY_SAMPLE_COUNT,
            )
            if not np.isfinite(compiled.max_abs_error):
                raise FpoSourceBackendError(
                    "compiled-policy parity produced a non-finite diagnostic"
                )
            self._quality_metrics.append(
                {
                    "candidate_id": request.candidate_id,
                    "bundle_digest": request.bundle_digest,
                    "outer_iteration": request.outer_iteration,
                    "runtime_driver_digest": self.runtime_driver_digest,
                    "runtime_provenance_class": upstream.provenance_class,
                    "runtime_receipt": dict(upstream.runtime_receipt),
                    "seed_count": len(seeds),
                    "golden_passed": bool(golden.passed),
                    "golden_raw_checked": bool(golden.raw_checked),
                    "golden_raw_max_abs_error": golden.raw_max_abs_error,
                    "golden_environment_max_abs_error": (
                        golden.environment_max_abs_error
                    ),
                    "compiled_passed": bool(compiled.passed),
                    "compiled_next_keys_equal": bool(compiled.next_keys_equal),
                    "compiled_max_abs_error": compiled.max_abs_error,
                    "parity_atol": PARITY_ATOL,
                    "parity_rtol": PARITY_RTOL,
                    "severity": (
                        "INFO"
                        if golden.passed
                        and compiled.passed
                        and compiled.next_keys_equal
                        else "WARNING"
                    ),
                    "admission_effect": "NONE",
                }
            )
            horizon = int(
                prepared.metadata.policy_spec["training_config"]["episode_length"]
            )
            if horizon != int(prepared.anchor.registry_config["episode_length"]):
                raise FpoSourceBackendError(
                    "bundle horizon differs from the source-anchor registry config"
                )
            returns = evaluate_frozen_policy_returns_batched(
                policy,
                bound.env,
                reset_seeds=seeds,
                policy_seeds=tuple(seed + POLICY_SEED_OFFSET for seed in seeds),
                horizon=horizon,
                observation_dim=prepared.metadata.observation_dim,
                action_dim=prepared.metadata.action_dim,
            )
            returns_array = np.asarray(returns, dtype=np.float64)
            if returns_array.shape != (len(seeds),) or not np.all(
                np.isfinite(returns_array)
            ):
                raise FpoSourceBackendError(
                    "source rollout returned an incompatible or non-finite return block"
            )
            bound.verify()
        except FpoSourceBackendError:
            raise
        except Exception as error:
            raise FpoSourceBackendError(
                f"frozen FPO/JAX source rollout failed: {type(error).__name__}: {error}"
            ) from error
        return tuple(
            BackendEpisodeResult.succeeded(
                reset_seed=seed,
                runtime_digest=request.anchor.runtime_digest,
                raw_return=value,
                steps=horizon,
                terminated=False,
                truncated=True,
            )
            for seed, value in zip(seeds, returns, strict=True)
        )


class FpoJaxSourceEvaluatorBackend:
    """Adapt the real batched v0.2 evaluator to the v0.3 per-seed boundary."""

    def __init__(
        self,
        *,
        runtime_driver: FpoJaxRuntimeDriver,
        selection_reset_seeds: Sequence[int],
        attestation_reset_seeds: Sequence[int],
    ) -> None:
        if not isinstance(runtime_driver, FpoJaxRuntimeDriver):
            raise FpoSourceBackendError("runtime_driver does not implement the protocol")
        self._driver = runtime_driver
        self._selection = _seed_block(
            selection_reset_seeds, "selection_reset_seeds"
        )
        self._attestation = _seed_block(
            attestation_reset_seeds, "attestation_reset_seeds"
        )
        if set(self._selection) & set(self._attestation):
            raise FpoSourceBackendError(
                "selection and attestation reset-seed blocks overlap"
            )
        driver_digest = _digest(
            runtime_driver.runtime_driver_digest, "runtime_driver_digest"
        )
        module_path = Path(__file__).resolve()
        material = {
            "schema": FPO_JAX_BACKEND_SCHEMA,
            "backend_file_sha256": sha256_file(module_path),
            "runtime_driver_digest": driver_digest,
            "selection_reset_seeds": list(self._selection),
            "attestation_reset_seeds": list(self._attestation),
            "policy_seed_rule": POLICY_SEED_RULE,
            "policy_seed_offset": POLICY_SEED_OFFSET,
            "parity_atol": PARITY_ATOL,
            "parity_rtol": PARITY_RTOL,
            "compiled_parity_sample_count": COMPILED_PARITY_SAMPLE_COUNT,
            "deterministic": True,
            "seed_block_execution": "whole_frozen_block_then_per_seed_cache",
        }
        self.evaluator_implementation_digest = sha256_json(material)
        self._requests: dict[str, SourceCandidateRequest] = {}
        self._block_results: dict[
            tuple[str, str], Mapping[int, BackendEpisodeResult]
        ] = {}

    def drain_quality_metrics(self) -> tuple[Mapping[str, Any], ...]:
        """Expose parity diagnostics without making them admission gates."""

        drain = getattr(self._driver, "drain_quality_metrics", None)
        return () if drain is None else tuple(drain())

    def validate_candidate(self, request: SourceCandidateRequest) -> ValidatedSourceBinding:
        if not isinstance(request, SourceCandidateRequest):
            raise FpoSourceBackendError("backend requires SourceCandidateRequest")
        abi = self._driver.validate_candidate(request)
        if not isinstance(abi, ExecutionABIRecord):
            raise FpoSourceBackendError("runtime driver returned a non-typed ABI")
        binding = ValidatedSourceBinding(
            request_digest=request.request_digest,
            candidate_id=request.candidate_id,
            # This field identifies the frozen experiment protocol.  Runtime
            # code identity is diagnostic metadata and is not an admission gate.
            evaluator_implementation_digest=request.evaluator_implementation_digest,
            bundle_path=request.bundle_path,
            bundle_digest=request.bundle_digest,
            anchor_manifest_path=request.anchor.manifest_path,
            anchor_manifest_digest=request.anchor.manifest_digest,
            anchor_runtime_digest=request.anchor.runtime_digest,
            source_environment_digest=request.source_environment_digest,
            execution_abi=abi,
        )
        self._requests[binding.binding_digest] = request
        return binding

    def _block_for_seed(self, reset_seed: int) -> tuple[str, tuple[int, ...]]:
        if isinstance(reset_seed, bool) or not isinstance(reset_seed, int):
            raise FpoSourceBackendError("reset_seed must be an integer")
        if reset_seed in self._selection:
            return "source_selection", self._selection
        if reset_seed in self._attestation:
            return "source_attestation", self._attestation
        raise FpoSourceBackendError("reset_seed is outside both frozen seed blocks")

    def evaluate_episode(
        self,
        binding: ValidatedSourceBinding,
        *,
        reset_seed: int,
    ) -> BackendEpisodeResult:
        if not isinstance(binding, ValidatedSourceBinding):
            raise FpoSourceBackendError("backend requires ValidatedSourceBinding")
        try:
            request = self._requests[binding.binding_digest]
        except KeyError as error:
            raise FpoSourceBackendError(
                "binding was not validated in this backend process"
            ) from error
        block, seeds = self._block_for_seed(reset_seed)
        cache_key = (binding.binding_digest, block)
        rows = self._block_results.get(cache_key)
        if rows is None:
            evaluated = tuple(
                self._driver.evaluate_seed_block(request, reset_seeds=seeds)
            )
            if (
                len(evaluated) != len(seeds)
                or any(not isinstance(row, BackendEpisodeResult) for row in evaluated)
                or tuple(row.reset_seed for row in evaluated) != seeds
                or any(
                    row.runtime_digest != binding.anchor_runtime_digest
                    for row in evaluated
                )
            ):
                raise FpoSourceBackendError(
                    "runtime driver returned incomplete, reordered, or drifted block rows"
                )
            rows = MappingProxyType(
                {seed: row for seed, row in zip(seeds, evaluated, strict=True)}
            )
            self._block_results[cache_key] = rows
        return rows[reset_seed]


__all__ = [
    "COMPILED_PARITY_SAMPLE_COUNT",
    "FpoJaxRuntimeDriver",
    "FpoJaxSourceEvaluatorBackend",
    "FpoSourceBackendError",
    "FrozenV02FpoJaxRuntimeDriver",
    "PARITY_ATOL",
    "PARITY_RTOL",
    "POLICY_SEED_OFFSET",
    "POLICY_SEED_RULE",
]
