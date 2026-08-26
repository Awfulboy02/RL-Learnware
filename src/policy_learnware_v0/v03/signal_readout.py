"""Reviewed selection and aggregate publication of formal signal readouts.

The 39-cell atlas intentionally does not prescribe which numeric cells receive
prefix curves or dynamics-axis diagnostics.  ``FormalSignalReadoutPlan`` makes
that human decision explicit and digest-bound without inventing new scientific
coverage.  ``FormalSignalReadoutBundle`` then joins exact readout coverage to
one already-valid formal atlas and exposes an aggregate-only publication.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from ..hashing import sha256_json
from .dynamics_axis import (
    DynamicsAxisDiagnostics,
    DynamicsAxisRegistry,
    DynamicsPublicQueryJoin,
    build_dynamics_axis_diagnostics,
)
from .preflight import PublicQueryPlan
from .signal_atlas import FormalSignalAtlasAuthorization, SignalAtlasRun
from .signal_contrasts import (
    SignalContrastGateEvaluation,
    SignalContrastPlan,
    SignalMaterialityThresholds,
    build_signal_contrast_plan,
    pair_control_evidence_set_digest,
    signal_metric_record_set_digest,
)
from .signal_matrix import SignalMatrixPlan, build_signal_matrix_plan
from .signal_metrics import SignalMetricRecord
from .signal_prefix import SignalPrefixRun, SignalPrefixSchedule
from .signal_runtime import FORMAL_MODE, SignalExecutionProtocol


FORMAL_SIGNAL_READOUT_PLAN_SCHEMA = (
    "policy-learnware.v03-formal-signal-readout-plan.v0"
)
FORMAL_SIGNAL_READOUT_BUNDLE_SCHEMA = (
    "policy-learnware.v03-formal-signal-readout-bundle.v0"
)
PUBLIC_SIGNAL_READOUT_BUNDLE_SCHEMA = (
    "policy-learnware.v03-public-signal-readout-bundle.v0"
)

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_PRIVATE_PUBLIC_KEYS = frozenset(
    {
        "rows",
        "expected_source_by_query",
        "query_diagnostics",
        "dynamics_context_by_opaque_query_id",
        "bank_id",
        "query_bank_id",
        "source_bank_id",
        "task_id",
        "query_task_id",
        "source_task_id",
        "context_id",
        "query_context_id",
        "source_context_id",
        "goal_contract_id",
        "dynamics_context_id",
        "nearest_source_bank_ids",
        "selected_source_bank_id",
    }
)


class SignalReadoutError(ValueError):
    """A reviewed readout selection or joined publication is invalid."""


def _digest(value: Any, where: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise SignalReadoutError(f"{where} must be a lowercase SHA-256 digest")
    return value


def _work_schedule(
    *, historical_seed: int
) -> Mapping[str, tuple[str, int | None]]:
    signal_plan = build_signal_matrix_plan()
    contrast_plan = build_signal_contrast_plan(historical_seed=historical_seed)
    result: dict[str, tuple[str, int | None]] = {}
    for work_key in contrast_plan.expected_numeric_work_keys:
        matches = tuple(
            cell
            for cell in signal_plan.numeric_cells
            if work_key.startswith(cell.cell_id.replace("::", "--") + "--seed-")
        )
        if len(matches) != 1:  # pragma: no cover - canonical matrix invariant
            raise SignalReadoutError("cannot resolve canonical signal work key")
        seed_token = work_key.rsplit("--seed-", 1)[1]
        result[work_key] = (
            matches[0].cell_id,
            None if seed_token == "NONE" else int(seed_token),
        )
    if len(result) != 79:  # pragma: no cover - protects future schedule drift
        raise SignalReadoutError("formal signal schedule must contain 79 work keys")
    return MappingProxyType(dict(sorted(result.items())))


def _work_keys(value: Sequence[str], where: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise SignalReadoutError(f"{where} must be a sequence of work keys")
    result = tuple(sorted(value))
    if (
        not result
        or any(not isinstance(item, str) or not item for item in result)
        or len(set(result)) != len(result)
    ):
        raise SignalReadoutError(f"{where} must be non-empty and unique")
    return result


def _raw_membership(record: SignalMetricRecord) -> frozenset[tuple[str, ...]]:
    if not isinstance(record, SignalMetricRecord):
        raise SignalReadoutError("raw-membership join requires a metric record")
    return frozenset(
        (
            row.query_bank_id,
            row.query_receipt_digest,
            row.query_raw_dataset_digest,
            row.source_bank_id,
            row.source_receipt_digest,
            row.source_raw_dataset_digest,
        )
        for row in record.rows
    )


@dataclass(frozen=True)
class FormalSignalReadoutPlan:
    signal_matrix_digest: str
    signal_execution_protocol_digest: str
    signal_identity_registry_digest: str
    signal_historical_seed: int
    review_decisions_digest: str
    prefix_work_keys: tuple[str, ...]
    dynamics_work_keys: tuple[str, ...]
    formal_signal_prefix_schedule_digest: str
    dynamics_axis_registry_digest: str
    public_query_plan_digest: str
    signal_contrast_plan_digest: str
    signal_materiality_threshold_digest: str
    attribution_gate_evidence_digest: str
    plan_digest: str | None = None
    schema: str = FORMAL_SIGNAL_READOUT_PLAN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FORMAL_SIGNAL_READOUT_PLAN_SCHEMA:
            raise SignalReadoutError("unsupported FormalSignalReadoutPlan schema")
        for name in (
            "signal_matrix_digest",
            "signal_execution_protocol_digest",
            "signal_identity_registry_digest",
            "review_decisions_digest",
            "formal_signal_prefix_schedule_digest",
            "dynamics_axis_registry_digest",
            "public_query_plan_digest",
            "signal_contrast_plan_digest",
            "signal_materiality_threshold_digest",
            "attribution_gate_evidence_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if (
            isinstance(self.signal_historical_seed, bool)
            or not isinstance(self.signal_historical_seed, int)
            or self.signal_historical_seed < 0
        ):
            raise SignalReadoutError("signal_historical_seed is invalid")
        canonical_matrix = build_signal_matrix_plan()
        canonical_contrasts = build_signal_contrast_plan(
            historical_seed=self.signal_historical_seed
        )
        if (
            self.signal_matrix_digest != canonical_matrix.plan_digest
            or self.signal_contrast_plan_digest != canonical_contrasts.plan_digest
        ):
            raise SignalReadoutError(
                "readout plan differs from the canonical matrix/contrast schedule"
            )
        expected_work = set(canonical_contrasts.expected_numeric_work_keys)
        prefix = _work_keys(self.prefix_work_keys, "prefix_work_keys")
        dynamics = _work_keys(self.dynamics_work_keys, "dynamics_work_keys")
        if not set(prefix) <= expected_work or not set(dynamics) <= expected_work:
            raise SignalReadoutError(
                "readout work keys must be numeric members of the exact 79-work schedule"
            )
        object.__setattr__(self, "prefix_work_keys", prefix)
        object.__setattr__(self, "dynamics_work_keys", dynamics)
        expected = sha256_json(self._payload_without_digest())
        if self.plan_digest is None:
            object.__setattr__(self, "plan_digest", expected)
        elif _digest(self.plan_digest, "plan_digest") != expected:
            raise SignalReadoutError("formal signal readout plan digest mismatch")

    @classmethod
    def create(
        cls,
        *,
        signal_plan: SignalMatrixPlan,
        signal_execution_protocol: SignalExecutionProtocol,
        prefix_work_keys: Sequence[str],
        dynamics_work_keys: Sequence[str],
        review_decisions_digest: str,
        prefix_schedule: SignalPrefixSchedule,
        dynamics_axis_registry: DynamicsAxisRegistry,
        public_query_plan: PublicQueryPlan,
        contrast_plan: SignalContrastPlan,
        materiality_thresholds: SignalMaterialityThresholds,
        attribution_gate_evidence_digest: str,
    ) -> "FormalSignalReadoutPlan":
        if not isinstance(signal_plan, SignalMatrixPlan) or not isinstance(
            signal_execution_protocol, SignalExecutionProtocol
        ):
            raise SignalReadoutError("readout plan requires typed signal protocols")
        if signal_execution_protocol.execution_mode != FORMAL_MODE:
            raise SignalReadoutError("formal readout plan requires FORMAL execution")
        if (
            signal_execution_protocol.plan_digest != signal_plan.plan_digest
            or signal_execution_protocol.plan_digest
            != build_signal_matrix_plan().plan_digest
        ):
            raise SignalReadoutError("signal execution belongs to another matrix")
        if not isinstance(prefix_schedule, SignalPrefixSchedule) or (
            prefix_schedule.scope != "FORMAL"
        ):
            raise SignalReadoutError("readout plan requires the formal prefix schedule")
        if not isinstance(dynamics_axis_registry, DynamicsAxisRegistry):
            raise SignalReadoutError("readout plan requires a dynamics registry")
        if not isinstance(public_query_plan, PublicQueryPlan):
            raise SignalReadoutError("readout plan requires a public query plan")
        if not isinstance(contrast_plan, SignalContrastPlan) or (
            contrast_plan.historical_seed
            != signal_execution_protocol.historical_seed
        ):
            raise SignalReadoutError("contrast plan differs from signal execution")
        if not isinstance(materiality_thresholds, SignalMaterialityThresholds) or (
            materiality_thresholds.contrast_plan_digest != contrast_plan.plan_digest
        ):
            raise SignalReadoutError("materiality thresholds belong to another contrast plan")
        review_digest = _digest(review_decisions_digest, "review_decisions_digest")
        if materiality_thresholds.review_decision_digest != review_digest:
            raise SignalReadoutError("thresholds differ from the reviewed decisions")
        return cls(
            signal_matrix_digest=str(signal_plan.plan_digest),
            signal_execution_protocol_digest=str(
                signal_execution_protocol.protocol_digest
            ),
            signal_identity_registry_digest=(
                signal_execution_protocol.identity_registry_digest
            ),
            signal_historical_seed=signal_execution_protocol.historical_seed,
            review_decisions_digest=review_digest,
            prefix_work_keys=tuple(prefix_work_keys),
            dynamics_work_keys=tuple(dynamics_work_keys),
            formal_signal_prefix_schedule_digest=str(prefix_schedule.schedule_digest),
            dynamics_axis_registry_digest=str(dynamics_axis_registry.registry_digest),
            public_query_plan_digest=str(public_query_plan.plan_digest),
            signal_contrast_plan_digest=str(contrast_plan.plan_digest),
            signal_materiality_threshold_digest=str(
                materiality_thresholds.threshold_digest
            ),
            attribution_gate_evidence_digest=attribution_gate_evidence_digest,
        )

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "signal_matrix_digest": self.signal_matrix_digest,
            "signal_execution_protocol_digest": self.signal_execution_protocol_digest,
            "signal_identity_registry_digest": self.signal_identity_registry_digest,
            "signal_historical_seed": self.signal_historical_seed,
            "review_decisions_digest": self.review_decisions_digest,
            "prefix_work_keys": list(self.prefix_work_keys),
            "dynamics_work_keys": list(self.dynamics_work_keys),
            "formal_signal_prefix_schedule_digest": (
                self.formal_signal_prefix_schedule_digest
            ),
            "dynamics_axis_registry_digest": self.dynamics_axis_registry_digest,
            "public_query_plan_digest": self.public_query_plan_digest,
            "signal_contrast_plan_digest": self.signal_contrast_plan_digest,
            "signal_materiality_threshold_digest": (
                self.signal_materiality_threshold_digest
            ),
            "attribution_gate_evidence_digest": (
                self.attribution_gate_evidence_digest
            ),
            "selection_authority": "EXTERNAL_HUMAN_REVIEW",
            "work_keys_are_exact_79_numeric_subset": True,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "plan_digest": self.plan_digest}


def _assert_public_projection_has_no_private_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        leaked = _PRIVATE_PUBLIC_KEYS & set(value)
        if leaked:
            raise SignalReadoutError(
                f"public signal readout leaks private fields: {sorted(leaked)}"
            )
        for item in value.values():
            _assert_public_projection_has_no_private_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_public_projection_has_no_private_keys(item)


def _verified_public_projection(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SignalReadoutError(f"{where} public projection must be a mapping")
    payload = dict(value)
    supplied = _digest(
        payload.pop("public_projection_digest", None),
        f"{where} public_projection_digest",
    )
    if supplied != sha256_json(payload):
        raise SignalReadoutError(f"{where} public projection digest mismatch")
    complete = {**payload, "public_projection_digest": supplied}
    _assert_public_projection_has_no_private_keys(complete)
    return MappingProxyType(complete)


@dataclass(frozen=True)
class FormalSignalReadoutBundle:
    plan: FormalSignalReadoutPlan
    atlas_run: SignalAtlasRun
    prefix_runs: Mapping[str, SignalPrefixRun]
    dynamics_diagnostics: Mapping[str, DynamicsAxisDiagnostics]
    dynamics_axis_registry: DynamicsAxisRegistry
    public_query_plan: PublicQueryPlan
    dynamics_public_query_join: DynamicsPublicQueryJoin
    contrast_gate_evaluation: SignalContrastGateEvaluation
    pair_control_evidence_digests: Mapping[str, Sequence[str]]
    attribution_gate_evidence_digest: str
    bundle_digest: str | None = None
    schema: str = FORMAL_SIGNAL_READOUT_BUNDLE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FORMAL_SIGNAL_READOUT_BUNDLE_SCHEMA:
            raise SignalReadoutError("unsupported FormalSignalReadoutBundle schema")
        if not isinstance(self.plan, FormalSignalReadoutPlan):
            raise SignalReadoutError("readout bundle requires a typed plan")
        if not isinstance(self.atlas_run, SignalAtlasRun):
            raise SignalReadoutError("readout bundle requires a typed SignalAtlasRun")
        authorization = self.atlas_run.formal_authorization
        if not isinstance(authorization, FormalSignalAtlasAuthorization):
            raise SignalReadoutError("atlas run lacks formal authorization")
        try:
            authorization.validate_signal_readout_plan(self.plan)
        except Exception as error:
            raise SignalReadoutError(str(error)) from error
        if (
            self.atlas_run.plan.plan_digest != self.plan.signal_matrix_digest
            or self.atlas_run.execution_protocol.protocol_digest
            != self.plan.signal_execution_protocol_digest
            or self.atlas_run.identity_registry.registry_digest
            != self.plan.signal_identity_registry_digest
            or self.atlas_run.execution_protocol.historical_seed
            != self.plan.signal_historical_seed
        ):
            raise SignalReadoutError("atlas run differs from the reviewed readout plan")
        schedule = _work_schedule(historical_seed=self.plan.signal_historical_seed)

        prefix_runs = dict(sorted(self.prefix_runs.items()))
        if set(prefix_runs) != set(self.plan.prefix_work_keys) or not all(
            isinstance(item, SignalPrefixRun) for item in prefix_runs.values()
        ):
            raise SignalReadoutError("prefix runs differ from exact reviewed coverage")
        if len({item.run_digest for item in prefix_runs.values()}) != len(prefix_runs):
            raise SignalReadoutError("prefix work keys cannot alias one run")
        for work_key, run in prefix_runs.items():
            cell_id, seed = schedule[work_key]
            protocol = run.execution_protocol
            atlas_metric = self.atlas_run.cell_runs[work_key].metric_record
            max_prefix_metric = run.points[-1].metric_record
            if (
                protocol.signal_execution_mode != FORMAL_MODE
                or run.formal_authorization_digest != authorization.authorization_digest
                or protocol.signal_execution_protocol_digest
                != self.plan.signal_execution_protocol_digest
                or protocol.plan_digest != self.plan.signal_matrix_digest
                or protocol.identity_registry_digest
                != self.plan.signal_identity_registry_digest
                or protocol.prefix_schedule.schedule_digest
                != self.plan.formal_signal_prefix_schedule_digest
                or protocol.cell_id != cell_id
                or any(
                    point.metric_record.cell_id != cell_id
                    or point.metric_record.representation_seed != seed
                    for point in run.points
                )
                or max_prefix_metric.source_index_digest
                != atlas_metric.source_index_digest
                or max_prefix_metric.expected_source_by_query
                != atlas_metric.expected_source_by_query
                or _raw_membership(max_prefix_metric)
                != _raw_membership(atlas_metric)
            ):
                raise SignalReadoutError(
                    f"prefix run differs from reviewed work key {work_key!r}"
                )

        if not isinstance(self.dynamics_axis_registry, DynamicsAxisRegistry) or (
            self.dynamics_axis_registry.registry_digest
            != self.plan.dynamics_axis_registry_digest
        ):
            raise SignalReadoutError("dynamics registry differs from readout plan")
        diagnostics = dict(sorted(self.dynamics_diagnostics.items()))
        if set(diagnostics) != set(self.plan.dynamics_work_keys) or not all(
            isinstance(item, DynamicsAxisDiagnostics)
            for item in diagnostics.values()
        ):
            raise SignalReadoutError(
                "dynamics diagnostics differ from exact reviewed coverage"
            )
        if len({item.diagnostics_digest for item in diagnostics.values()}) != len(
            diagnostics
        ):
            raise SignalReadoutError("dynamics work keys cannot alias one diagnostic")
        for work_key, diagnostic in diagnostics.items():
            metric_record = self.atlas_run.cell_runs[work_key].metric_record
            expected = build_dynamics_axis_diagnostics(
                metric_record=metric_record,
                registry=self.dynamics_axis_registry,
                execution_mode=FORMAL_MODE,
                signal_plan_digest=self.plan.signal_matrix_digest,
                signal_execution_protocol_digest=(
                    self.plan.signal_execution_protocol_digest
                ),
                identity_registry_digest=self.plan.signal_identity_registry_digest,
                formal_authorization=authorization,
            )
            if diagnostic.to_dict() != expected.to_dict():
                raise SignalReadoutError(
                    f"dynamics diagnostic differs from atlas work key {work_key!r}"
                )

        if not isinstance(self.public_query_plan, PublicQueryPlan) or (
            self.public_query_plan.plan_digest != self.plan.public_query_plan_digest
        ):
            raise SignalReadoutError("public query plan differs from readout plan")
        if not isinstance(self.dynamics_public_query_join, DynamicsPublicQueryJoin):
            raise SignalReadoutError("readout bundle requires a private dynamics join")
        try:
            self.dynamics_public_query_join.validate(
                public_query_plan=self.public_query_plan,
                registry=self.dynamics_axis_registry,
            )
            authorization.validate_dynamics_public_query_join(
                self.dynamics_public_query_join
            )
        except Exception as error:
            raise SignalReadoutError(str(error)) from error

        if not isinstance(
            self.contrast_gate_evaluation, SignalContrastGateEvaluation
        ):
            raise SignalReadoutError("readout bundle requires a contrast gate")
        contrast_plan = build_signal_contrast_plan(
            historical_seed=self.plan.signal_historical_seed
        )
        atlas_records = {
            key: run.metric_record for key, run in self.atlas_run.cell_runs.items()
        }
        metric_set_digest = signal_metric_record_set_digest(
            contrast_plan, atlas_records
        )
        pair_set_digest = pair_control_evidence_set_digest(
            contrast_plan, self.pair_control_evidence_digests
        )
        gate = self.contrast_gate_evaluation
        if (
            gate.formal_atlas_authorization_digest
            != authorization.authorization_digest
            or gate.contrast_plan_digest != self.plan.signal_contrast_plan_digest
            or gate.threshold_digest
            != self.plan.signal_materiality_threshold_digest
            or gate.metric_record_set_digest != metric_set_digest
            or gate.pair_control_evidence_set_digest != pair_set_digest
        ):
            raise SignalReadoutError("contrast gate differs from atlas/readout freeze")
        attribution = _digest(
            self.attribution_gate_evidence_digest,
            "attribution_gate_evidence_digest",
        )
        if attribution != self.plan.attribution_gate_evidence_digest:
            raise SignalReadoutError(
                "external attribution gate evidence differs from readout plan"
            )
        object.__setattr__(self, "prefix_runs", MappingProxyType(prefix_runs))
        object.__setattr__(
            self, "dynamics_diagnostics", MappingProxyType(diagnostics)
        )
        # Bind both the private typed run digests and their aggregate-only
        # projections.  A later publisher therefore cannot silently swap the
        # public view while retaining the same private bundle manifest.
        _verified_public_projection(self.atlas_run.to_public_dict(), "atlas")
        for work_key, run in self.prefix_runs.items():
            _verified_public_projection(
                run.to_public_dict(), f"prefix_runs[{work_key}]"
            )
        for work_key, diagnostic in self.dynamics_diagnostics.items():
            _verified_public_projection(
                diagnostic.to_public_dict(), f"dynamics_diagnostics[{work_key}]"
            )
        _verified_public_projection(
            self.dynamics_public_query_join.to_public_dict(),
            "dynamics_public_query_join",
        )
        _verified_public_projection(
            self.contrast_gate_evaluation.to_public_dict(), "contrast_gate"
        )
        expected_digest = sha256_json(self._payload_without_digest())
        if self.bundle_digest is None:
            object.__setattr__(self, "bundle_digest", expected_digest)
        elif _digest(self.bundle_digest, "bundle_digest") != expected_digest:
            raise SignalReadoutError("formal signal readout bundle digest mismatch")

    @property
    def formal_authorization_digest(self) -> str:
        return str(self.atlas_run.formal_authorization.authorization_digest)

    @property
    def pair_control_evidence_set_digest(self) -> str:
        return self.contrast_gate_evaluation.pair_control_evidence_set_digest

    def _payload_without_digest(self) -> dict[str, Any]:
        atlas_public = _verified_public_projection(
            self.atlas_run.to_public_dict(), "atlas"
        )
        return {
            "schema": self.schema,
            "readout_plan_digest": self.plan.plan_digest,
            "freeze_manifest_digest": (
                self.atlas_run.formal_authorization.freeze_manifest.freeze_manifest_digest
            ),
            "formal_authorization_digest": self.formal_authorization_digest,
            "atlas_run_digest": self.atlas_run.run_digest,
            "atlas_public_projection_digest": atlas_public[
                "public_projection_digest"
            ],
            "prefix_run_digests": {
                key: run.run_digest for key, run in self.prefix_runs.items()
            },
            "prefix_public_projection_digests": {
                key: _verified_public_projection(
                    run.to_public_dict(), f"prefix_runs[{key}]"
                )["public_projection_digest"]
                for key, run in self.prefix_runs.items()
            },
            "dynamics_diagnostic_digests": {
                key: item.diagnostics_digest
                for key, item in self.dynamics_diagnostics.items()
            },
            "dynamics_public_projection_digests": {
                key: _verified_public_projection(
                    item.to_public_dict(), f"dynamics_diagnostics[{key}]"
                )["public_projection_digest"]
                for key, item in self.dynamics_diagnostics.items()
            },
            "dynamics_public_query_join_digest": (
                self.dynamics_public_query_join.join_digest
            ),
            "dynamics_query_join_public_projection_digest": (
                _verified_public_projection(
                    self.dynamics_public_query_join.to_public_dict(),
                    "dynamics_public_query_join",
                )["public_projection_digest"]
            ),
            "contrast_gate_evaluation_digest": (
                self.contrast_gate_evaluation.evaluation_digest
            ),
            "contrast_gate_public_projection_digest": (
                _verified_public_projection(
                    self.contrast_gate_evaluation.to_public_dict(),
                    "contrast_gate",
                )["public_projection_digest"]
            ),
            "pair_control_evidence_set_digest": (
                self.pair_control_evidence_set_digest
            ),
            "attribution_gate_evidence_digest": (
                self.attribution_gate_evidence_digest
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "bundle_digest": self.bundle_digest}

    def to_public_dict(self) -> dict[str, Any]:
        atlas_public = _verified_public_projection(
            self.atlas_run.to_public_dict(), "atlas"
        )
        prefix_public = {
            key: _verified_public_projection(
                run.to_public_dict(), f"prefix_runs[{key}]"
            )
            for key, run in self.prefix_runs.items()
        }
        dynamics_public = {
            key: _verified_public_projection(
                item.to_public_dict(), f"dynamics_diagnostics[{key}]"
            )
            for key, item in self.dynamics_diagnostics.items()
        }
        join_public = _verified_public_projection(
            self.dynamics_public_query_join.to_public_dict(),
            "dynamics_public_query_join",
        )
        contrast_public = _verified_public_projection(
            self.contrast_gate_evaluation.to_public_dict(), "contrast_gate"
        )
        payload = {
            "schema": PUBLIC_SIGNAL_READOUT_BUNDLE_SCHEMA,
            "readout_plan_digest": self.plan.plan_digest,
            "freeze_manifest_digest": (
                self.atlas_run.formal_authorization.freeze_manifest.freeze_manifest_digest
            ),
            "formal_authorization_digest": self.formal_authorization_digest,
            "atlas": dict(atlas_public),
            "prefix_readouts": {
                key: dict(value) for key, value in prefix_public.items()
            },
            "dynamics_readouts": {
                key: dict(value) for key, value in dynamics_public.items()
            },
            "dynamics_query_join": dict(join_public),
            "contrast_gate": dict(contrast_public),
            "pair_control_evidence_set_digest": (
                self.pair_control_evidence_set_digest
            ),
            "pair_control_evidence_count": sum(
                len(tuple(values))
                for values in self.pair_control_evidence_digests.values()
            ),
            "attribution_gate_evidence_digest": (
                self.attribution_gate_evidence_digest
            ),
            "private_bank_task_context_and_alias_rows_withheld": True,
            "private_bundle_digest": self.bundle_digest,
        }
        _assert_public_projection_has_no_private_keys(payload)
        return {**payload, "public_projection_digest": sha256_json(payload)}


__all__ = [
    "FORMAL_SIGNAL_READOUT_BUNDLE_SCHEMA",
    "FORMAL_SIGNAL_READOUT_PLAN_SCHEMA",
    "PUBLIC_SIGNAL_READOUT_BUNDLE_SCHEMA",
    "FormalSignalReadoutBundle",
    "FormalSignalReadoutPlan",
    "SignalReadoutError",
]
