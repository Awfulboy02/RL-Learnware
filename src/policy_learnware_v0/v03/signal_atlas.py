"""Coverage-complete orchestration for the frozen v0.3 signal atlas."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

from ..hashing import sha256_json
from .contracts import QUERY_EMPIRICAL_PROTOCOL_ID, derive_reducer_digest
from .condition_plan import ConditionPlanError
from .dynamics_axis import DynamicsAxisRegistry, DynamicsPublicQueryJoin
from .signal_matrix import (
    C_RF_SHUFFLED_NEXT,
    SignalCell,
    SignalCellRecord,
    SignalMatrixLedger,
    SignalMatrixPlan,
    build_optimization_fit_jobs,
)
from .signal_runtime import (
    RepresentedBank,
    SignalCellRun,
    SignalExecutionProtocol,
    SignalIdentityRegistry,
    fit_source_kernel_protocol,
    run_signal_cell,
)
from .preflight import ExecutionCheckpoint, PreExperimentFreezeManifest
from .representation_ladder import (
    R5L_SUPERVISED_LINEAR,
    R5_VIEW_SPECIFIC_CORRO_REFIT,
    FormalTrainedRepresentationReceipt,
)
from .source_fit import (
    DATA_FITTED_REPRESENTATION_IDS,
    FormalSourceFitBatch,
    SourceFitProvenanceError,
)
from .signal_prefix import SignalPrefixSchedule
from .signal_contrasts import SignalContrastPlan, build_signal_contrast_plan


SIGNAL_CELL_WORK_ITEM_SCHEMA = "policy-learnware.v03-signal-cell-work-item.v0"
SIGNAL_ATLAS_RUN_SCHEMA = "policy-learnware.v03-signal-atlas-run.v0"
FORMAL_SIGNAL_ATLAS_AUTHORIZATION_SCHEMA = (
    "policy-learnware.v03-formal-signal-atlas-authorization.v5"
)

SignalExecutionMode = Literal["DEVELOPMENT_SMOKE", "FORMAL"]
DEVELOPMENT_SMOKE_MODE: SignalExecutionMode = "DEVELOPMENT_SMOKE"
FORMAL_MODE: SignalExecutionMode = "FORMAL"


class SignalAtlasError(ValueError):
    """A signal-atlas work plan or result coverage is incomplete."""


def _digest(value: Any, where: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        raise SignalAtlasError(f"{where} must be a lowercase SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise SignalAtlasError(f"{where} must be a lowercase SHA-256 digest") from error
    return value


def validate_formal_atlas_fit_schedule_bindings(
    *,
    plan: SignalMatrixPlan,
    source_membership_digests: Sequence[str],
    fit_job_receipts: Mapping[str, FormalTrainedRepresentationReceipt],
) -> tuple[str, str]:
    """Require the complete 45-fit atlas to share one rows/schedule freeze."""

    if not isinstance(plan, SignalMatrixPlan):
        raise SignalAtlasError("formal fit binding requires SignalMatrixPlan")
    memberships = {
        _digest(item, "source_membership_digest")
        for item in source_membership_digests
    }
    if len(memberships) != 1:
        raise SignalAtlasError(
            "formal atlas fit jobs must share one source-row membership"
        )
    expected_jobs = {
        job.job_id: job for job in build_optimization_fit_jobs(plan)
    }
    receipts = dict(fit_job_receipts)
    if set(receipts) != set(expected_jobs) or not all(
        isinstance(item, FormalTrainedRepresentationReceipt)
        for item in receipts.values()
    ):
        raise SignalAtlasError(
            "formal atlas receipts must cover the exact 45-fit schedule"
        )
    schedules = set()
    for job_id, receipt in receipts.items():
        job = expected_jobs[job_id]
        if (
            receipt.representation_id != job.representation_id
            or receipt.formal_fit_job_digest != job.job_digest
        ):
            raise SignalAtlasError(
                "formal checkpoint receipt belongs to another fit job"
            )
        schedules.add(receipt.formal_source_fit_schedule_digest)
    if len(schedules) != 1:
        raise SignalAtlasError(
            "formal atlas R5/R5L checkpoints must share one source-fit schedule"
        )
    return next(iter(schedules)), next(iter(memberships))


@dataclass(frozen=True)
class FormalSignalAtlasAuthorization:
    """Join an externally reviewed freeze to one exact formal atlas protocol.

    The v0.3 CLI may publish an engineering freeze but cannot mint review
    authority.  A coverage-complete run therefore needs this typed join to an
    already-authorized manifest supplied by the external Paper-I owner.
    """

    freeze_manifest: PreExperimentFreezeManifest
    plan_digest: str
    signal_contrast_plan_digest: str
    signal_materiality_threshold_digest: str
    formal_signal_readout_plan_digest: str
    identity_registry_digest: str
    execution_protocol_digest: str
    representation_plan_digest: str
    condition_plan_digest: str
    formal_source_fit_schedule_digest: str
    formal_source_membership_digest: str
    execution_plan_digest: str
    work_item_graph_digest: str
    formal_signal_prefix_schedule_digest: str
    dynamics_axis_registry_digest: str
    data_role_manifest_digest: str
    canonicalizer_registry_digest: str
    source_reduced_query_empirical_protocol_digest: str
    authorization_digest: str | None = None
    schema: str = FORMAL_SIGNAL_ATLAS_AUTHORIZATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FORMAL_SIGNAL_ATLAS_AUTHORIZATION_SCHEMA:
            raise SignalAtlasError("unsupported formal signal-atlas authorization")
        if not isinstance(self.freeze_manifest, PreExperimentFreezeManifest):
            raise SignalAtlasError("formal authorization requires a typed freeze")
        if not self.freeze_manifest.formal_run_authorized:
            raise SignalAtlasError(
                "signal atlas requires externally verified review authority"
            )
        for name in (
            "plan_digest",
            "signal_contrast_plan_digest",
            "signal_materiality_threshold_digest",
            "formal_signal_readout_plan_digest",
            "identity_registry_digest",
            "execution_protocol_digest",
            "representation_plan_digest",
            "condition_plan_digest",
            "formal_source_fit_schedule_digest",
            "formal_source_membership_digest",
            "execution_plan_digest",
            "work_item_graph_digest",
            "formal_signal_prefix_schedule_digest",
            "dynamics_axis_registry_digest",
            "data_role_manifest_digest",
            "canonicalizer_registry_digest",
            "source_reduced_query_empirical_protocol_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        expected_freeze = (
            self.freeze_manifest.signal_matrix_digest,
            self.freeze_manifest.signal_contrast_plan_digest,
            self.freeze_manifest.signal_materiality_threshold_digest,
            self.freeze_manifest.formal_signal_readout_plan_digest,
            self.freeze_manifest.signal_identity_registry_digest,
            self.freeze_manifest.signal_execution_protocol_digest,
            self.freeze_manifest.representation_plan_digest,
            self.freeze_manifest.condition_plan_digest,
            self.freeze_manifest.formal_source_fit_schedule_digest,
            self.freeze_manifest.formal_source_membership_digest,
            self.freeze_manifest.signal_work_item_graph_digest,
            self.freeze_manifest.formal_signal_prefix_schedule_digest,
            self.freeze_manifest.dynamics_axis_registry_digest,
            self.freeze_manifest.data_role_manifest_digest,
            self.freeze_manifest.canonicalizer_registry_digest,
            self.freeze_manifest.source_reduced_query_empirical_protocol_digest,
        )
        observed = (
            self.plan_digest,
            self.signal_contrast_plan_digest,
            self.signal_materiality_threshold_digest,
            self.formal_signal_readout_plan_digest,
            self.identity_registry_digest,
            self.execution_protocol_digest,
            self.representation_plan_digest,
            self.condition_plan_digest,
            self.formal_source_fit_schedule_digest,
            self.formal_source_membership_digest,
            self.work_item_graph_digest,
            self.formal_signal_prefix_schedule_digest,
            self.dynamics_axis_registry_digest,
            self.data_role_manifest_digest,
            self.canonicalizer_registry_digest,
            self.source_reduced_query_empirical_protocol_digest,
        )
        if observed != expected_freeze:
            raise SignalAtlasError(
                "formal atlas protocols differ from the externally reviewed freeze"
            )
        if self.formal_signal_prefix_schedule_digest != (
            SignalPrefixSchedule.formal().schedule_digest
        ):
            raise SignalAtlasError(
                "formal authorization requires the exact 1/2/4/8/16/32/64 prefix schedule"
            )
        expected = sha256_json(self._payload_without_digest())
        if self.authorization_digest is None:
            object.__setattr__(self, "authorization_digest", expected)
        elif _digest(self.authorization_digest, "authorization_digest") != expected:
            raise SignalAtlasError("formal signal-atlas authorization digest mismatch")

    @classmethod
    def bind(
        cls,
        freeze_manifest: PreExperimentFreezeManifest,
        *,
        plan: SignalMatrixPlan,
        execution_protocol: SignalExecutionProtocol,
        identity_registry: SignalIdentityRegistry,
        dynamics_axis_registry: DynamicsAxisRegistry,
        work_item_digests: Mapping[str, str],
    ) -> "FormalSignalAtlasAuthorization":
        if not isinstance(plan, SignalMatrixPlan) or not isinstance(
            execution_protocol, SignalExecutionProtocol
        ) or not isinstance(identity_registry, SignalIdentityRegistry):
            raise SignalAtlasError(
                "formal authorization requires typed plan/execution/identity inputs"
            )
        if execution_protocol.execution_mode != FORMAL_MODE:
            raise SignalAtlasError("formal authorization cannot bind a development protocol")
        if not isinstance(dynamics_axis_registry, DynamicsAxisRegistry):
            raise SignalAtlasError(
                "formal authorization requires a typed dynamics-axis registry"
            )
        return cls(
            freeze_manifest=freeze_manifest,
            plan_digest=str(plan.plan_digest),
            signal_contrast_plan_digest=str(
                build_signal_contrast_plan(
                    historical_seed=execution_protocol.historical_seed
                ).plan_digest
            ),
            signal_materiality_threshold_digest=(
                freeze_manifest.signal_materiality_threshold_digest
            ),
            formal_signal_readout_plan_digest=(
                freeze_manifest.formal_signal_readout_plan_digest
            ),
            identity_registry_digest=str(identity_registry.registry_digest),
            execution_protocol_digest=str(execution_protocol.protocol_digest),
            representation_plan_digest=str(
                execution_protocol.representation_plan.plan_digest
            ),
            condition_plan_digest=str(execution_protocol.condition_plan.plan_digest),
            formal_source_fit_schedule_digest=(
                freeze_manifest.formal_source_fit_schedule_digest
            ),
            formal_source_membership_digest=(
                freeze_manifest.formal_source_membership_digest
            ),
            execution_plan_digest=signal_execution_plan_digest(
                plan, execution_protocol
            ),
            work_item_graph_digest=signal_work_item_graph_digest(
                plan, execution_protocol, work_item_digests
            ),
            formal_signal_prefix_schedule_digest=str(
                SignalPrefixSchedule.formal().schedule_digest
            ),
            dynamics_axis_registry_digest=str(
                dynamics_axis_registry.registry_digest
            ),
            data_role_manifest_digest=freeze_manifest.data_role_manifest_digest,
            canonicalizer_registry_digest=(
                freeze_manifest.canonicalizer_registry_digest
            ),
            source_reduced_query_empirical_protocol_digest=(
                signal_asymmetric_kme_protocol_digest(execution_protocol)
            ),
        )

    def validate(
        self,
        *,
        plan: SignalMatrixPlan,
        execution_protocol: SignalExecutionProtocol,
        identity_registry: SignalIdentityRegistry,
    ) -> None:
        observed = (
            plan.plan_digest,
            identity_registry.registry_digest,
            execution_protocol.protocol_digest,
            execution_protocol.representation_plan.plan_digest,
            execution_protocol.condition_plan.plan_digest,
            signal_execution_plan_digest(plan, execution_protocol),
            signal_asymmetric_kme_protocol_digest(execution_protocol),
        )
        expected = (
            self.plan_digest,
            self.identity_registry_digest,
            self.execution_protocol_digest,
            self.representation_plan_digest,
            self.condition_plan_digest,
            self.execution_plan_digest,
            self.source_reduced_query_empirical_protocol_digest,
        )
        if observed != expected or execution_protocol.execution_mode != FORMAL_MODE:
            raise SignalAtlasError(
                "formal authorization belongs to another execution freeze"
            )

    def validate_work_items(
        self,
        *,
        plan: SignalMatrixPlan,
        execution_protocol: SignalExecutionProtocol,
        work_item_digests: Mapping[str, str],
    ) -> None:
        observed = signal_work_item_graph_digest(
            plan, execution_protocol, work_item_digests
        )
        if (
            observed != self.work_item_graph_digest
            or observed != self.freeze_manifest.signal_work_item_graph_digest
        ):
            raise SignalAtlasError(
                "formal work-item contents differ from the externally reviewed freeze"
            )

    def validate_signal_prefix_schedule(
        self, schedule: SignalPrefixSchedule
    ) -> None:
        if not isinstance(schedule, SignalPrefixSchedule):
            raise SignalAtlasError("formal prefix readout requires typed schedule")
        if (
            schedule.scope != "FORMAL"
            or schedule.schedule_digest
            != self.formal_signal_prefix_schedule_digest
        ):
            raise SignalAtlasError(
                "prefix schedule differs from the externally reviewed freeze"
            )

    def validate_signal_contrast_plan(self, plan: SignalContrastPlan) -> None:
        if not isinstance(plan, SignalContrastPlan):
            raise SignalAtlasError("formal contrast gate requires a typed plan")
        if (
            plan.plan_digest != self.signal_contrast_plan_digest
            or plan.plan_digest != self.freeze_manifest.signal_contrast_plan_digest
        ):
            raise SignalAtlasError(
                "signal contrast plan differs from the externally reviewed freeze"
            )

    def validate_signal_readout_plan(self, plan: Any) -> None:
        from .signal_readout import FormalSignalReadoutPlan

        if not isinstance(plan, FormalSignalReadoutPlan):
            raise SignalAtlasError("formal readout requires a typed plan")
        if (
            plan.plan_digest != self.formal_signal_readout_plan_digest
            or plan.plan_digest
            != self.freeze_manifest.formal_signal_readout_plan_digest
            or plan.signal_execution_protocol_digest
            != self.execution_protocol_digest
            or plan.signal_identity_registry_digest
            != self.identity_registry_digest
            or plan.review_decisions_digest
            != self.freeze_manifest.review_decisions_digest
            or plan.formal_signal_prefix_schedule_digest
            != self.formal_signal_prefix_schedule_digest
            or plan.dynamics_axis_registry_digest
            != self.dynamics_axis_registry_digest
            or plan.signal_contrast_plan_digest
            != self.signal_contrast_plan_digest
            or plan.signal_materiality_threshold_digest
            != self.signal_materiality_threshold_digest
            or plan.public_query_plan_digest
            != self.freeze_manifest.public_query_plan_digest
        ):
            raise SignalAtlasError(
                "signal readout plan differs from the externally reviewed freeze"
            )

    def validate_dynamics_axis_registry(
        self, registry: DynamicsAxisRegistry
    ) -> None:
        if not isinstance(registry, DynamicsAxisRegistry):
            raise SignalAtlasError(
                "formal dynamics readout requires typed axis registry"
            )
        if registry.registry_digest != self.dynamics_axis_registry_digest:
            raise SignalAtlasError(
                "dynamics-axis registry differs from the externally reviewed freeze"
            )

    def validate_dynamics_public_query_join(
        self, join: DynamicsPublicQueryJoin
    ) -> None:
        if not isinstance(join, DynamicsPublicQueryJoin):
            raise SignalAtlasError(
                "formal dynamics readout requires typed public-query join"
            )
        if (
            join.public_query_plan_digest
            != self.freeze_manifest.public_query_plan_digest
            or join.dynamics_axis_registry_digest
            != self.dynamics_axis_registry_digest
        ):
            raise SignalAtlasError(
                "dynamics public-query join differs from the reviewed freeze"
            )

    def validate_runtime_bindings(
        self, work_items: Sequence["SignalCellWorkItem"]
    ) -> None:
        items = tuple(work_items)
        if not items or not all(isinstance(item, SignalCellWorkItem) for item in items):
            raise SignalAtlasError("formal runtime binding requires typed work items")
        source_fit_roles = {
            item.formal_source_fit.authority.data_role_manifest_digest
            for item in items
            if item.formal_source_fit is not None
        }
        if source_fit_roles != {self.data_role_manifest_digest}:
            raise SignalAtlasError(
                "formal source-fit data roles differ from the reviewed freeze"
            )
        source_memberships = tuple(
            item.formal_source_fit.authority.source_membership_digest
            for item in items
            if item.formal_source_fit is not None
        )
        receipts_by_job_id = {}
        canonical_fit_jobs = {
            (job.condition_id, job.representation_id, job.seed): job
            for job in build_optimization_fit_jobs(items[0].plan)
        }
        for item in items:
            if item.cell.representation_id not in {
                R5_VIEW_SPECIFIC_CORRO_REFIT,
                R5L_SUPERVISED_LINEAR,
            }:
                continue
            receipt = item.source_banks[0].formal_fit_receipt
            if not isinstance(receipt, FormalTrainedRepresentationReceipt):
                raise SignalAtlasError(
                    "formal atlas R5/R5L work item lacks checkpoint receipt"
                )
            key = (
                item.cell.condition_id,
                item.cell.representation_id,
                item.evaluation_seed,
            )
            try:
                fit_job = canonical_fit_jobs[key]
            except KeyError as error:
                raise SignalAtlasError(
                    "formal atlas R5/R5L work item is absent from 45-fit schedule"
                ) from error
            receipts_by_job_id[fit_job.job_id] = receipt
        schedule_digest, membership_digest = validate_formal_atlas_fit_schedule_bindings(
            plan=items[0].plan,
            source_membership_digests=source_memberships,
            fit_job_receipts=receipts_by_job_id,
        )
        if (
            schedule_digest != self.formal_source_fit_schedule_digest
            or membership_digest != self.formal_source_membership_digest
        ):
            raise SignalAtlasError(
                "formal atlas source schedule/membership differs from reviewed freeze"
            )
        observed_canonicalizer = signal_canonicalizer_registry_digest(items)
        if observed_canonicalizer != self.canonicalizer_registry_digest:
            raise SignalAtlasError(
                "formal canonicalizer/shape registry differs from the reviewed freeze"
            )

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "freeze_manifest_digest": self.freeze_manifest.freeze_manifest_digest,
            "review_authority_receipt_digest": (
                self.freeze_manifest.review_authority_receipt_digest
            ),
            "plan_digest": self.plan_digest,
            "signal_contrast_plan_digest": self.signal_contrast_plan_digest,
            "signal_materiality_threshold_digest": (
                self.signal_materiality_threshold_digest
            ),
            "formal_signal_readout_plan_digest": (
                self.formal_signal_readout_plan_digest
            ),
            "identity_registry_digest": self.identity_registry_digest,
            "execution_protocol_digest": self.execution_protocol_digest,
            "representation_plan_digest": self.representation_plan_digest,
            "condition_plan_digest": self.condition_plan_digest,
            "formal_source_fit_schedule_digest": (
                self.formal_source_fit_schedule_digest
            ),
            "formal_source_membership_digest": (
                self.formal_source_membership_digest
            ),
            "execution_plan_digest": self.execution_plan_digest,
            "work_item_graph_digest": self.work_item_graph_digest,
            "formal_signal_prefix_schedule_digest": (
                self.formal_signal_prefix_schedule_digest
            ),
            "dynamics_axis_registry_digest": self.dynamics_axis_registry_digest,
            "data_role_manifest_digest": self.data_role_manifest_digest,
            "canonicalizer_registry_digest": self.canonicalizer_registry_digest,
            "source_reduced_query_empirical_protocol_digest": (
                self.source_reduced_query_empirical_protocol_digest
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._payload_without_digest(),
            "authorization_digest": self.authorization_digest,
        }


@dataclass(frozen=True)
class SignalCellWorkItem:
    plan: SignalMatrixPlan
    cell: SignalCell
    source_banks: tuple[RepresentedBank, ...]
    query_banks: tuple[RepresentedBank, ...]
    expected_source_by_query: Mapping[str, str]
    identity_registry: SignalIdentityRegistry
    execution_protocol: SignalExecutionProtocol
    evaluation_seed: int | None
    execution_mode: SignalExecutionMode = DEVELOPMENT_SMOKE_MODE
    formal_source_fit: FormalSourceFitBatch | None = None
    work_item_digest: str | None = None
    schema: str = SIGNAL_CELL_WORK_ITEM_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SIGNAL_CELL_WORK_ITEM_SCHEMA:
            raise SignalAtlasError("unsupported SignalCellWorkItem schema")
        if not isinstance(self.plan, SignalMatrixPlan) or not isinstance(
            self.cell, SignalCell
        ):
            raise SignalAtlasError("work item requires typed plan/cell")
        try:
            frozen = self.plan.cell(self.cell.cell_id)
        except Exception as error:
            raise SignalAtlasError("work-item cell is absent from plan") from error
        if frozen.to_dict() != self.cell.to_dict() or self.cell.applicability != "NUMERIC":
            raise SignalAtlasError("work item must bind one numeric frozen cell")
        if not isinstance(self.identity_registry, SignalIdentityRegistry) or not isinstance(
            self.execution_protocol, SignalExecutionProtocol
        ):
            raise SignalAtlasError(
                "work item requires frozen identity/execution protocols"
            )
        if (
            self.execution_protocol.plan_digest != self.plan.plan_digest
            or self.execution_protocol.identity_registry_digest
            != self.identity_registry.registry_digest
            or self.execution_protocol.measurement_protocol_digest
            != self.identity_registry.measurement_protocol_digest
        ):
            raise SignalAtlasError(
                "work-item execution protocol differs from plan/identity freeze"
            )
        sources = tuple(self.source_banks)
        queries = tuple(self.query_banks)
        if len(sources) < 2 or not queries or not all(
            isinstance(item, RepresentedBank) for item in (*sources, *queries)
        ):
            raise SignalAtlasError(
                "work item requires at least two typed sources and one typed query"
            )
        source_ids = tuple(item.feature_bank.receipt.bank_id for item in sources)
        query_ids = tuple(item.feature_bank.receipt.bank_id for item in queries)
        if len(set(source_ids)) != len(source_ids) or len(set(query_ids)) != len(query_ids):
            raise SignalAtlasError("work item contains duplicate bank IDs")
        for bank in (*sources, *queries):
            self.identity_registry.validate_feature_bank(bank.feature_bank)
            try:
                self.execution_protocol.condition_plan.validate_feature_bank(
                    bank.feature_bank
                )
            except ConditionPlanError as error:
                raise SignalAtlasError(str(error)) from error
        if self.cell.condition_id == C_RF_SHUFFLED_NEXT and any(
            bank.feature_bank.condition_audit_passed is not True
            or bank.feature_bank.condition_result_digest is None
            or bank.feature_bank.condition_audit_digest is None
            for bank in (*sources, *queries)
        ):
            raise SignalAtlasError(
                "formal C_RF_SHUFFLED_NEXT work item requires typed passed audits"
            )
        if self.execution_mode not in {DEVELOPMENT_SMOKE_MODE, FORMAL_MODE}:
            raise SignalAtlasError("signal work item execution_mode is invalid")
        if self.execution_mode != self.execution_protocol.execution_mode:
            raise SignalAtlasError(
                "work-item execution mode differs from its frozen execution protocol"
            )
        manifests = tuple(item.representation_manifest for item in (*sources, *queries))
        coordinate_digests = {item.coordinate_digest for item in manifests}
        representation_ids = {item.representation_id for item in manifests}
        if len(coordinate_digests) != 1 or representation_ids != {
            self.cell.representation_id
        }:
            raise SignalAtlasError(
                "work item banks must share the frozen cell representation coordinate"
            )
        manifest = manifests[0]
        if self.formal_source_fit is not None:
            if not isinstance(self.formal_source_fit, FormalSourceFitBatch):
                raise SignalAtlasError("formal_source_fit must be a typed source-fit batch")
            if manifest.representation_id not in DATA_FITTED_REPRESENTATION_IDS:
                raise SignalAtlasError(
                    "formal_source_fit applies only to data-fitted R2/R5/R5L"
                )
            authority = self.formal_source_fit.authority
            for source_fit_bank in (
                *self.formal_source_fit.train_feature_banks,
                *self.formal_source_fit.validation_feature_banks,
            ):
                self.identity_registry.validate_feature_bank(source_fit_bank)
            canonicalizers = {
                item.feature_bank.receipt.canonicalizer_digest
                for item in (*sources, *queries)
            }
            measurements = {
                item.feature_bank.identity.measurement_protocol_digest
                for item in (*sources, *queries)
            }
            if authority.condition_id != self.cell.condition_id:
                raise SignalAtlasError(
                    "formal source-fit condition differs from the signal cell"
                )
            if canonicalizers != {authority.canonicalizer_digest}:
                raise SignalAtlasError(
                    "formal source-fit canonicalizer differs from work-item banks"
                )
            if measurements != {authority.measurement_protocol_digest}:
                raise SignalAtlasError(
                    "formal source-fit measurement protocol differs from work-item banks"
                )
            try:
                self.formal_source_fit.require_condition_plan(
                    self.execution_protocol.condition_plan
                )
                self.formal_source_fit.require_manifest_binding(manifest)
            except SourceFitProvenanceError as error:
                raise SignalAtlasError(str(error)) from error
            if (
                self.execution_mode == FORMAL_MODE
                and manifest.representation_id
                in {R5_VIEW_SPECIFIC_CORRO_REFIT, R5L_SUPERVISED_LINEAR}
            ):
                receipts = tuple(
                    item.formal_fit_receipt for item in (*sources, *queries)
                )
                if any(item is None for item in receipts) or len(
                    {
                        item.receipt_digest
                        for item in receipts
                        if isinstance(item, FormalTrainedRepresentationReceipt)
                    }
                ) != 1:
                    raise SignalAtlasError(
                        "formal R5/R5L work item requires one shared checkpoint receipt"
                    )
                receipt = receipts[0]
                if not isinstance(receipt, FormalTrainedRepresentationReceipt):
                    raise SignalAtlasError(
                        "formal R5/R5L work item checkpoint receipt is invalid"
                    )
                if (
                    receipt.formal_source_fit_batch_digest
                    != self.formal_source_fit.batch_digest
                    or receipt.representation_execution_plan_digest
                    != self.execution_protocol.representation_plan.plan_digest
                ):
                    raise SignalAtlasError(
                        "formal checkpoint receipt differs from source-fit/representation freeze"
                    )
        if (
            self.execution_mode == FORMAL_MODE
            and manifest.representation_id in DATA_FITTED_REPRESENTATION_IDS
            and self.formal_source_fit is None
        ):
            raise SignalAtlasError(
                "formal R2/R5/R5L work item requires typed source-fit provenance"
            )
        expected = dict(sorted(self.expected_source_by_query.items()))
        if set(expected) != set(query_ids) or any(
            source_id not in set(source_ids) for source_id in expected.values()
        ):
            raise SignalAtlasError(
                "work item expected-source mapping does not cover query/source banks"
            )
        if self.evaluation_seed is not None and (
            isinstance(self.evaluation_seed, bool)
            or not isinstance(self.evaluation_seed, int)
            or self.evaluation_seed < 0
        ):
            raise SignalAtlasError("evaluation_seed is invalid")
        if self.evaluation_seed not in self.execution_protocol.expected_evaluation_seeds(
            self.cell
        ):
            raise SignalAtlasError(
                "evaluation seed is absent from the frozen cell schedule"
            )
        if {item.seed for item in manifests} != {self.evaluation_seed}:
            raise SignalAtlasError(
                "represented-bank seeds differ from the work-item evaluation seed"
            )
        object.__setattr__(self, "source_banks", sources)
        object.__setattr__(self, "query_banks", queries)
        object.__setattr__(self, "expected_source_by_query", MappingProxyType(expected))
        expected_digest = sha256_json(self._payload_without_digest())
        if self.work_item_digest is None:
            object.__setattr__(self, "work_item_digest", expected_digest)
        elif _digest(self.work_item_digest, "work_item_digest") != expected_digest:
            raise SignalAtlasError("work_item_digest does not match inputs")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "plan_digest": self.plan.plan_digest,
            "cell_digest": self.cell.cell_digest,
            "source_represented_bank_digests": sorted(
                str(item.represented_bank_digest) for item in self.source_banks
            ),
            "query_represented_bank_digests": sorted(
                str(item.represented_bank_digest) for item in self.query_banks
            ),
            "expected_source_by_query": dict(self.expected_source_by_query),
            "identity_registry_digest": self.identity_registry.registry_digest,
            "execution_protocol_digest": self.execution_protocol.protocol_digest,
            "evaluation_seed": self.evaluation_seed,
            "execution_mode": self.execution_mode,
            "formal_source_fit_batch_digest": (
                None
                if self.formal_source_fit is None
                else self.formal_source_fit.batch_digest
            ),
        }

    @property
    def work_key(self) -> str:
        return signal_work_key(self.cell.cell_id, self.evaluation_seed)

    def execute(self) -> SignalCellRun:
        return run_signal_cell(
            plan=self.plan,
            cell=self.cell,
            source_banks=self.source_banks,
            query_banks=self.query_banks,
            expected_source_by_query=self.expected_source_by_query,
            identity_registry=self.identity_registry,
            execution_protocol=self.execution_protocol,
            source_fit_provenance_digest=(
                None
                if self.formal_source_fit is None
                else self.formal_source_fit.batch_digest
            ),
            work_item_digest=self.work_item_digest,
        )

    def validate_run(self, run: SignalCellRun) -> None:
        """Validate a checkpoint-restored result against this exact work item."""

        _validate_cell_run_against_work_item(run, self)


def _validate_cell_run_against_work_item(
    run: SignalCellRun, item: SignalCellWorkItem
) -> None:
    """Join a restored numeric result to every material work-item input.

    ``SignalCellRun`` is intentionally serializable for crash recovery.  Its
    own constructor validates internal digests, while this join prevents a
    syntactically valid run from being attached to a different bank set,
    source-fit authority, taxonomy, representation coordinate, or query
    schedule when a complete atlas is assembled.
    """

    if not isinstance(run, SignalCellRun) or not isinstance(
        item, SignalCellWorkItem
    ):
        raise SignalAtlasError("cell/work join requires typed inputs")
    expected_provenance = (
        None
        if item.formal_source_fit is None
        else str(item.formal_source_fit.batch_digest)
    )
    if (
        run.work_item_digest != item.work_item_digest
        or run.plan_digest != item.plan.plan_digest
        or run.cell_id != item.cell.cell_id
        or run.cell_digest != item.cell.cell_digest
        or run.execution_protocol_digest
        != item.execution_protocol.protocol_digest
        or run.execution_mode != item.execution_mode
        or run.evaluation_seed != item.evaluation_seed
        or run.source_fit_provenance_digest != expected_provenance
    ):
        raise SignalAtlasError(
            "cell run identity/provenance differs from its frozen work item"
        )

    sources = {
        bank.feature_bank.receipt.bank_id: bank for bank in item.source_banks
    }
    queries = {
        bank.feature_bank.receipt.bank_id: bank for bank in item.query_banks
    }
    if any(
        bank.feature_bank.receipt.data_role != "source_reference_spec"
        for bank in sources.values()
    ) or any(
        bank.feature_bank.receipt.data_role
        not in {"development_query", "confirmatory_query"}
        for bank in queries.values()
    ):
        raise SignalAtlasError("cell work item carries invalid source/query roles")

    # Recomputing the source-only bandwidth is cheap relative to RKME
    # reduction and catches a hand-assembled kernel with a changed bank set or
    # calibration value without repeating the expensive distance stage.
    expected_kernel = fit_source_kernel_protocol(
        tuple(sources.values()), execution_protocol=item.execution_protocol
    )
    if run.kernel_protocol.to_dict() != expected_kernel.to_dict():
        raise SignalAtlasError(
            "cell run kernel differs from its frozen source-only calibration"
        )

    manifest = item.source_banks[0].representation_manifest
    metric = run.metric_record
    expected_query_manifest = sha256_json(
        {
            "schema": "policy-learnware.v03-signal-query-manifest.v0",
            "query_bank_digests": sorted(
                str(bank.represented_bank_digest) for bank in queries.values()
            ),
            "expected_source_by_query": dict(
                sorted(item.expected_source_by_query.items())
            ),
        }
    )
    if (
        metric.cell_id != item.cell.cell_id
        or metric.view_or_condition_id != item.cell.condition_id
        or metric.representation_id != item.cell.representation_id
        or metric.representation_coordinate_digest != manifest.coordinate_digest
        or metric.representation_seed != item.evaluation_seed
        or metric.query_manifest_digest != expected_query_manifest
        or dict(metric.expected_source_by_query)
        != dict(item.expected_source_by_query)
        or set(run.query_run_digests) != set(queries)
    ):
        raise SignalAtlasError(
            "cell run metric/query schedule differs from its frozen work item"
        )

    expected_pairs = {
        (query_id, source_id)
        for query_id in queries
        for source_id in sources
    }
    observed_pairs = {
        (row.query_bank_id, row.source_bank_id) for row in metric.rows
    }
    if observed_pairs != expected_pairs:
        raise SignalAtlasError(
            "cell run distance rows do not cover the frozen query/source matrix"
        )
    for row in metric.rows:
        query = queries[row.query_bank_id].feature_bank
        source = sources[row.source_bank_id].feature_bank
        expected_metadata = (
            query.receipt.receipt_digest,
            source.receipt.receipt_digest,
            query.receipt.raw_dataset_digest,
            source.receipt.raw_dataset_digest,
            query.receipt.task_private_id,
            source.receipt.task_private_id,
            query.identity.context_id,
            source.identity.context_id,
            query.identity.embodiment_id,
            source.identity.embodiment_id,
            query.identity.abi_contract_id,
            source.identity.abi_contract_id,
            query.identity.goal_contract_id,
            source.identity.goal_contract_id,
            query.identity.dynamics_context_id,
            source.identity.dynamics_context_id,
            query.identity.equivalence_class_id,
            source.identity.equivalence_class_id,
        )
        observed_metadata = (
            row.query_receipt_digest,
            row.source_receipt_digest,
            row.query_raw_dataset_digest,
            row.source_raw_dataset_digest,
            row.query_task_id,
            row.source_task_id,
            row.query_context_id,
            row.source_context_id,
            row.query_embodiment_id,
            row.source_embodiment_id,
            row.query_abi_contract_id,
            row.source_abi_contract_id,
            row.query_goal_contract_id,
            row.source_goal_contract_id,
            row.query_dynamics_context_id,
            row.source_dynamics_context_id,
            row.query_equivalence_class_id,
            row.source_equivalence_class_id,
        )
        if observed_metadata != expected_metadata:
            raise SignalAtlasError(
                "cell run distance-row identity differs from frozen bank metadata"
            )
    # Geometry cannot be reconstructed from distance rows alone.  Recompute it
    # from the exact work-item arrays before admitting a restored artifact.
    from .signal_diagnostics import build_signal_cell_diagnostics

    expected_diagnostics = build_signal_cell_diagnostics(
        source_banks=tuple(sources.values()),
        query_banks=tuple(queries.values()),
        metric_record=metric,
    )
    if (
        run.diagnostics.to_private_dict()
        != expected_diagnostics.to_private_dict()
    ):
        raise SignalAtlasError(
            "cell diagnostics differ from frozen represented-bank arrays"
        )


def build_rf_control_audit_summary(
    work_items: Mapping[str, SignalCellWorkItem],
) -> Mapping[str, Mapping[str, Any]]:
    """Build an aggregate-only C_RF audit projection from typed work items."""

    records: dict[str, Mapping[str, Any]] = {}
    for work_key, item in sorted(work_items.items()):
        if not isinstance(item, SignalCellWorkItem):
            raise SignalAtlasError("control audit summary requires typed work items")
        if item.cell.condition_id != C_RF_SHUFFLED_NEXT:
            continue
        feature_banks = tuple(
            bank.feature_bank for bank in (*item.source_banks, *item.query_banks)
        )
        if any(
            bank.condition_audit_passed is not True
            or bank.condition_result_digest is None
            or bank.condition_audit_digest is None
            for bank in feature_banks
        ):
            raise SignalAtlasError(
                "public C_RF_SHUFFLED_NEXT audit summary requires passed typed evidence"
            )
        transforms = {bank.condition_transform_digest for bank in feature_banks}
        if len(transforms) != 1:
            raise SignalAtlasError(
                "C_RF_SHUFFLED_NEXT work item mixes transform protocols"
            )
        records[work_key] = MappingProxyType(
            {
                "condition_id": C_RF_SHUFFLED_NEXT,
                "evaluation_seed": item.evaluation_seed,
                "condition_transform_digest": next(iter(transforms)),
                "audited_bank_count": len(feature_banks),
                "result_set_digest": sha256_json(
                    sorted(
                        str(bank.condition_result_digest) for bank in feature_banks
                    )
                ),
                "marginal_audit_set_digest": sha256_json(
                    sorted(
                        str(bank.condition_audit_digest) for bank in feature_banks
                    )
                ),
                "all_marginal_audits_passed": True,
                "private_bank_membership_withheld": True,
            }
        )
    return MappingProxyType(records)


@dataclass(frozen=True)
class SignalAtlasRun:
    plan: SignalMatrixPlan
    execution_protocol: SignalExecutionProtocol
    identity_registry: SignalIdentityRegistry
    formal_authorization: FormalSignalAtlasAuthorization
    work_items: Mapping[str, SignalCellWorkItem]
    cell_runs: Mapping[str, SignalCellRun]
    ledger: SignalMatrixLedger
    run_digest: str | None = None
    schema: str = SIGNAL_ATLAS_RUN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SIGNAL_ATLAS_RUN_SCHEMA:
            raise SignalAtlasError("unsupported SignalAtlasRun schema")
        if not isinstance(self.plan, SignalMatrixPlan) or not isinstance(
            self.ledger, SignalMatrixLedger
        ):
            raise SignalAtlasError("atlas run requires typed plan/ledger")
        if not isinstance(
            self.execution_protocol, SignalExecutionProtocol
        ) or not isinstance(self.identity_registry, SignalIdentityRegistry):
            raise SignalAtlasError(
                "atlas run requires typed execution and identity protocols"
            )
        if not isinstance(
            self.formal_authorization, FormalSignalAtlasAuthorization
        ):
            raise SignalAtlasError(
                "atlas run requires externally reviewed formal authorization"
            )
        self.formal_authorization.validate(
            plan=self.plan,
            execution_protocol=self.execution_protocol,
            identity_registry=self.identity_registry,
        )
        if self.execution_protocol.execution_mode != FORMAL_MODE:
            raise SignalAtlasError("a coverage-complete signal atlas must be formal")
        if (
            self.execution_protocol.plan_digest != self.plan.plan_digest
            or self.execution_protocol.identity_registry_digest
            != self.identity_registry.registry_digest
            or self.execution_protocol.measurement_protocol_digest
            != self.identity_registry.measurement_protocol_digest
        ):
            raise SignalAtlasError(
                "atlas execution/identity protocols differ from the frozen plan"
            )
        if self.ledger.plan.plan_digest != self.plan.plan_digest:
            raise SignalAtlasError("atlas ledger is bound to another plan")
        items = dict(sorted(self.work_items.items()))
        if not all(isinstance(item, SignalCellWorkItem) for item in items.values()):
            raise SignalAtlasError("atlas work_items must be typed")
        work = {
            work_key: str(item.work_item_digest)
            for work_key, item in items.items()
        }
        runs = dict(sorted(self.cell_runs.items()))
        expected_work = expected_signal_work_keys(self.plan, self.execution_protocol)
        if set(items) != expected_work or set(runs) != expected_work:
            raise SignalAtlasError(
                "atlas work/run coverage differs from the complete frozen schedule"
            )
        self.formal_authorization.validate_work_items(
            plan=self.plan,
            execution_protocol=self.execution_protocol,
            work_item_digests=work,
        )
        self.formal_authorization.validate_runtime_bindings(tuple(items.values()))
        for work_key, item in items.items():
            if (
                item.work_key != work_key
                or item.plan.plan_digest != self.plan.plan_digest
                or item.execution_protocol.protocol_digest
                != self.execution_protocol.protocol_digest
                or item.identity_registry.registry_digest
                != self.identity_registry.registry_digest
                or item.execution_mode != FORMAL_MODE
            ):
                raise SignalAtlasError(
                    "atlas work item differs from the frozen formal protocol"
                )
        for work_key, run in runs.items():
            if not isinstance(run, SignalCellRun):
                raise SignalAtlasError("atlas cell_runs must be typed")
            cell = self.plan.cell(run.cell_id)
            if (
                run.plan_digest != self.plan.plan_digest
                or run.cell_digest != cell.cell_digest
                or run.execution_protocol_digest
                != self.execution_protocol.protocol_digest
                or signal_work_key(run.cell_id, run.evaluation_seed) != work_key
                or run.execution_mode != FORMAL_MODE
                or run.work_item_digest != work[work_key]
            ):
                raise SignalAtlasError("atlas cell run differs from frozen plan")
            items[work_key].validate_run(run)
        expected_ledger = _build_signal_ledger(
            self.plan, self.execution_protocol, runs
        )
        if self.ledger.to_dict() != expected_ledger.to_dict():
            raise SignalAtlasError(
                "atlas ledger is not the exact aggregation of its seed runs"
            )
        object.__setattr__(self, "work_items", MappingProxyType(items))
        object.__setattr__(self, "cell_runs", MappingProxyType(runs))
        expected = sha256_json(self._payload_without_digest())
        if self.run_digest is None:
            object.__setattr__(self, "run_digest", expected)
        elif _digest(self.run_digest, "run_digest") != expected:
            raise SignalAtlasError("signal atlas run digest mismatch")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "plan_digest": self.plan.plan_digest,
            "execution_protocol_digest": self.execution_protocol.protocol_digest,
            "identity_registry_digest": self.identity_registry.registry_digest,
            "formal_authorization_digest": (
                self.formal_authorization.authorization_digest
            ),
            "work_item_digests": dict(self.work_item_digests),
            "cell_run_digests": {
                cell_id: run.run_digest for cell_id, run in self.cell_runs.items()
            },
            "ledger_digest": self.ledger.ledger_digest,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "run_digest": self.run_digest}

    @property
    def work_item_digests(self) -> Mapping[str, str]:
        return MappingProxyType(
            {
                work_key: str(item.work_item_digest)
                for work_key, item in self.work_items.items()
            }
        )

    def to_public_dict(self) -> dict[str, Any]:
        """Publish aggregate metrics and opaque digests, never private distance rows."""

        control_audits = build_rf_control_audit_summary(self.work_items)
        payload = {
            "schema": "policy-learnware.v03-public-signal-atlas.v0",
            "plan_digest": self.plan.plan_digest,
            "execution_protocol_digest": self.execution_protocol.protocol_digest,
            "identity_registry_digest": self.identity_registry.registry_digest,
            "formal_authorization_digest": (
                self.formal_authorization.authorization_digest
            ),
            "freeze_manifest_digest": (
                self.formal_authorization.freeze_manifest.freeze_manifest_digest
            ),
            "logical_cell_records": [
                record.to_dict() for record in self.ledger.records
            ],
            "seed_metric_records": {
                key: run.metric_record.to_public_dict()
                for key, run in self.cell_runs.items()
            },
            "seed_diagnostic_records": {
                key: run.diagnostics.to_public_dict()
                for key, run in self.cell_runs.items()
            },
            "control_audit_records": {
                key: dict(value) for key, value in control_audits.items()
            },
            "private_distance_rows_withheld": True,
            "private_run_digest": self.run_digest,
        }
        return {**payload, "public_projection_digest": sha256_json(payload)}


def run_signal_atlas(
    plan: SignalMatrixPlan,
    execution_protocol: SignalExecutionProtocol,
    identity_registry: SignalIdentityRegistry,
    work_items: Sequence[SignalCellWorkItem],
    *,
    formal_authorization: FormalSignalAtlasAuthorization | None = None,
) -> SignalAtlasRun:
    """Execute every required seed instance and aggregate one logical cell row."""

    by_key = _validate_formal_atlas_inputs(
        plan=plan,
        execution_protocol=execution_protocol,
        identity_registry=identity_registry,
        work_items=work_items,
        formal_authorization=formal_authorization,
    )
    runs = {work_key: by_key[work_key].execute() for work_key in sorted(by_key)}
    return assemble_signal_atlas(
        plan=plan,
        execution_protocol=execution_protocol,
        identity_registry=identity_registry,
        work_items=tuple(by_key.values()),
        cell_runs=runs,
        formal_authorization=formal_authorization,
    )


def _validate_formal_atlas_inputs(
    *,
    plan: SignalMatrixPlan,
    execution_protocol: SignalExecutionProtocol,
    identity_registry: SignalIdentityRegistry,
    work_items: Sequence[SignalCellWorkItem],
    formal_authorization: FormalSignalAtlasAuthorization | None,
) -> dict[str, SignalCellWorkItem]:
    """Validate the frozen formal graph without starting any numeric work."""

    if not isinstance(plan, SignalMatrixPlan):
        raise SignalAtlasError("atlas execution requires SignalMatrixPlan")
    if not isinstance(execution_protocol, SignalExecutionProtocol) or not isinstance(
        identity_registry, SignalIdentityRegistry
    ):
        raise SignalAtlasError("atlas requires frozen execution/identity protocols")
    if (
        execution_protocol.plan_digest != plan.plan_digest
        or execution_protocol.identity_registry_digest
        != identity_registry.registry_digest
    ):
        raise SignalAtlasError("atlas protocol differs from plan/identity freeze")
    if execution_protocol.execution_mode != FORMAL_MODE:
        raise SignalAtlasError(
            "coverage-complete signal atlas requires a FORMAL execution protocol"
        )
    if not isinstance(formal_authorization, FormalSignalAtlasAuthorization):
        raise SignalAtlasError(
            "coverage-complete signal atlas requires formal authorization"
        )
    formal_authorization.validate(
        plan=plan,
        execution_protocol=execution_protocol,
        identity_registry=identity_registry,
    )
    items = tuple(work_items)
    if not all(isinstance(item, SignalCellWorkItem) for item in items):
        raise SignalAtlasError("atlas work_items must be typed")
    by_key = {item.work_key: item for item in items}
    expected_keys = expected_signal_work_keys(plan, execution_protocol)
    if len(by_key) != len(items) or set(by_key) != expected_keys:
        raise SignalAtlasError(
            "atlas requires exact numeric-cell and representation-seed coverage"
        )
    if any(
        item.plan.plan_digest != plan.plan_digest
        or item.execution_protocol.protocol_digest
        != execution_protocol.protocol_digest
        or item.identity_registry.registry_digest != identity_registry.registry_digest
        for item in items
    ):
        raise SignalAtlasError("atlas work item belongs to another formal freeze")
    if any(item.execution_mode != FORMAL_MODE for item in items):
        raise SignalAtlasError(
            "coverage-complete signal atlas accepts formal work items only"
        )
    formal_authorization.validate_work_items(
        plan=plan,
        execution_protocol=execution_protocol,
        work_item_digests={
            work_key: str(item.work_item_digest)
            for work_key, item in by_key.items()
        },
    )
    formal_authorization.validate_runtime_bindings(tuple(by_key.values()))
    return by_key


def assemble_signal_atlas(
    *,
    plan: SignalMatrixPlan,
    execution_protocol: SignalExecutionProtocol,
    identity_registry: SignalIdentityRegistry,
    work_items: Sequence[SignalCellWorkItem],
    cell_runs: Mapping[str, SignalCellRun],
    formal_authorization: FormalSignalAtlasAuthorization | None,
) -> SignalAtlasRun:
    """Assemble checkpoint-restored runs without executing a cell twice."""

    by_key = _validate_formal_atlas_inputs(
        plan=plan,
        execution_protocol=execution_protocol,
        identity_registry=identity_registry,
        work_items=work_items,
        formal_authorization=formal_authorization,
    )
    runs = dict(sorted(cell_runs.items()))
    if set(runs) != set(by_key) or not all(
        isinstance(run, SignalCellRun) for run in runs.values()
    ):
        raise SignalAtlasError(
            "restored cell runs must exactly cover the frozen work graph"
        )
    if any(
        run.work_item_digest != by_key[work_key].work_item_digest
        or run.execution_mode != FORMAL_MODE
        for work_key, run in runs.items()
    ):
        raise SignalAtlasError(
            "restored cell run differs from its frozen formal work item"
        )
    ledger = _build_signal_ledger(plan, execution_protocol, runs)
    return SignalAtlasRun(
        plan=plan,
        execution_protocol=execution_protocol,
        identity_registry=identity_registry,
        formal_authorization=formal_authorization,
        work_items=by_key,
        cell_runs=runs,
        ledger=ledger,
    )


def _build_signal_ledger(
    plan: SignalMatrixPlan,
    execution_protocol: SignalExecutionProtocol,
    runs: Mapping[str, SignalCellRun],
) -> SignalMatrixLedger:
    """Deterministically reconstruct the 39-row logical ledger from seed runs."""

    if not isinstance(plan, SignalMatrixPlan) or not isinstance(
        execution_protocol, SignalExecutionProtocol
    ):
        raise SignalAtlasError("ledger aggregation requires typed plan/protocol")
    if execution_protocol.plan_digest != plan.plan_digest:
        raise SignalAtlasError("ledger protocol belongs to another signal plan")
    typed_runs = tuple(runs.values())
    if not all(isinstance(run, SignalCellRun) for run in typed_runs):
        raise SignalAtlasError("ledger aggregation requires typed signal-cell runs")
    records = []
    for cell in plan.cells:
        if cell.applicability == "STRUCTURAL_NA":
            records.append(
                SignalCellRecord(
                    plan_digest=str(plan.plan_digest),
                    cell_id=cell.cell_id,
                    cell_digest=str(cell.cell_digest),
                    status="STRUCTURAL_NA",
                    metrics=None,
                    numeric_artifact_digest=None,
                )
            )
        else:
            seed_runs = tuple(run for run in typed_runs if run.cell_id == cell.cell_id)
            if {run.evaluation_seed for run in seed_runs} != set(
                execution_protocol.expected_evaluation_seeds(cell)
            ):
                raise SignalAtlasError("cell run seed coverage is incomplete")
            metric_key_sets = {
                tuple(sorted((run.metric_record.metric_values or {}).keys()))
                for run in seed_runs
            }
            if len(metric_key_sets) != 1:
                raise SignalAtlasError("seed runs expose inconsistent metric sets")
            metric_names = next(iter(metric_key_sets))
            aggregate_metrics = {
                name: float(
                    sum(
                        float((run.metric_record.metric_values or {})[name])
                        for run in seed_runs
                    )
                    / len(seed_runs)
                )
                for name in metric_names
            }
            records.append(
                SignalCellRecord(
                    plan_digest=str(plan.plan_digest),
                    cell_id=cell.cell_id,
                    cell_digest=str(cell.cell_digest),
                    status="COMPUTED",
                    metrics=aggregate_metrics,
                    numeric_artifact_digest=sha256_json(
                        {
                            "schema": "policy-learnware.v03-seed-aggregate.v0",
                            "cell_digest": cell.cell_digest,
                            "seed_run_digests": sorted(
                                str(run.run_digest) for run in seed_runs
                            ),
                        }
                    ),
                )
            )
    return SignalMatrixLedger(plan=plan, records=tuple(records))


def signal_work_key(cell_id: str, seed: int | None) -> str:
    if not isinstance(cell_id, str) or not cell_id:
        raise SignalAtlasError("work key requires a cell ID")
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
    ):
        raise SignalAtlasError("work key seed is invalid")
    safe_cell_id = cell_id.replace("::", "--")
    return f"{safe_cell_id}--seed-{'NONE' if seed is None else seed}"


def expected_signal_work_keys(
    plan: SignalMatrixPlan, protocol: SignalExecutionProtocol
) -> frozenset[str]:
    if not isinstance(plan, SignalMatrixPlan) or not isinstance(
        protocol, SignalExecutionProtocol
    ):
        raise SignalAtlasError("work-key schedule requires typed plan/protocol")
    if protocol.plan_digest != plan.plan_digest:
        raise SignalAtlasError("execution protocol belongs to another signal plan")
    return frozenset(
        signal_work_key(cell.cell_id, seed)
        for cell in plan.numeric_cells
        for seed in protocol.expected_evaluation_seeds(cell)
    )


def signal_execution_plan_digest(
    plan: SignalMatrixPlan, protocol: SignalExecutionProtocol
) -> str:
    """Digest the exact resumable work graph for one frozen protocol."""

    keys = expected_signal_work_keys(plan, protocol)
    return sha256_json(
        {
            "schema": "policy-learnware.v03-signal-execution-plan.v0",
            "plan_digest": plan.plan_digest,
            "execution_protocol_digest": protocol.protocol_digest,
            "work_keys": sorted(keys),
        }
    )


def signal_work_item_graph_digest(
    plan: SignalMatrixPlan,
    protocol: SignalExecutionProtocol,
    work_item_digests: Mapping[str, str],
) -> str:
    """Bind the reviewed 79-key schedule to exact banks and source-fit inputs."""

    if not isinstance(work_item_digests, Mapping):
        raise SignalAtlasError("work-item graph requires a digest mapping")
    expected_keys = expected_signal_work_keys(plan, protocol)
    frozen = {
        work_key: _digest(digest, f"work_item_digests[{work_key}]")
        for work_key, digest in sorted(work_item_digests.items())
    }
    if set(frozen) != expected_keys:
        raise SignalAtlasError(
            "work-item graph must exactly cover the frozen 79-key schedule"
        )
    return sha256_json(
        {
            "schema": "policy-learnware.v03-signal-work-item-graph.v0",
            "plan_digest": plan.plan_digest,
            "execution_protocol_digest": protocol.protocol_digest,
            "work_item_digests": frozen,
        }
    )


def signal_asymmetric_kme_protocol_digest(
    protocol: SignalExecutionProtocol,
) -> str:
    """Project the numeric protocol onto the source-reduced/query-empirical contract."""

    if not isinstance(protocol, SignalExecutionProtocol):
        raise SignalAtlasError("asymmetric KME digest requires execution protocol")
    return sha256_json(
        {
            "schema": "policy-learnware.v03-source-reduced-query-empirical.v0",
            "source_spec_role": "SOURCE_REDUCED",
            "query_spec_role": "QUERY_EMPIRICAL",
            "query_protocol_id": QUERY_EMPIRICAL_PROTOCOL_ID,
            "query_support_reduction": False,
            "source_reducer_digest": derive_reducer_digest(
                protocol.reducer_config
            ),
            "kernel_bandwidth_fit_scope": "SOURCE_ONLY",
            "measurement_protocol_digest": protocol.measurement_protocol_digest,
            "pair_budget": protocol.pair_budget,
            "bandwidth_seed": protocol.bandwidth_seed,
            "block_size": protocol.block_size,
        }
    )


def signal_canonicalizer_registry_digest(
    work_items: Sequence[SignalCellWorkItem],
) -> str:
    """Digest the one canonicalizer/shape/normalizer coordinate used by all work."""

    items = tuple(work_items)
    if not items or not all(isinstance(item, SignalCellWorkItem) for item in items):
        raise SignalAtlasError("canonicalizer binding requires typed work items")
    coordinates = set()
    for item in items:
        feature_banks = [
            bank.feature_bank for bank in (*item.source_banks, *item.query_banks)
        ]
        if item.formal_source_fit is not None:
            feature_banks.extend(item.formal_source_fit.train_feature_banks)
            feature_banks.extend(item.formal_source_fit.validation_feature_banks)
        coordinates.update(
            (
                bank.receipt.canonicalizer_digest,
                bank.receipt.native_shape_registry_digest,
                bank.receipt.normalizer_digest,
            )
            for bank in feature_banks
        )
    if len(coordinates) != 1:
        raise SignalAtlasError(
            "formal atlas cannot mix canonicalizer/shape/normalizer coordinates"
        )
    canonicalizer, registry, normalizer = next(iter(coordinates))
    return sha256_json(
        {
            "schema": "policy-learnware.v03-canonicalizer-registry-binding.v0",
            "canonicalizer_digest": canonicalizer,
            "native_shape_registry_digest": registry,
            "normalizer_digest": normalizer,
        }
    )


def initialize_signal_execution_checkpoint(
    plan: SignalMatrixPlan, protocol: SignalExecutionProtocol
) -> ExecutionCheckpoint:
    """Create the durable 79-item resume graph before any large work starts."""

    keys = expected_signal_work_keys(plan, protocol)
    execution_plan_digest = signal_execution_plan_digest(plan, protocol)
    return ExecutionCheckpoint(
        execution_plan_digest=execution_plan_digest,
        work_item_states={key: "PENDING" for key in sorted(keys)},
        completed_artifact_digests={},
        attempt=0,
    )


__all__ = [
    "FORMAL_SIGNAL_ATLAS_AUTHORIZATION_SCHEMA",
    "SIGNAL_ATLAS_RUN_SCHEMA",
    "SIGNAL_CELL_WORK_ITEM_SCHEMA",
    "DEVELOPMENT_SMOKE_MODE",
    "FORMAL_MODE",
    "FormalSignalAtlasAuthorization",
    "SignalExecutionMode",
    "SignalAtlasError",
    "SignalAtlasRun",
    "SignalCellWorkItem",
    "assemble_signal_atlas",
    "build_rf_control_audit_summary",
    "expected_signal_work_keys",
    "initialize_signal_execution_checkpoint",
    "run_signal_atlas",
    "signal_execution_plan_digest",
    "signal_asymmetric_kme_protocol_digest",
    "signal_canonicalizer_registry_digest",
    "signal_work_item_graph_digest",
    "signal_work_key",
    "validate_formal_atlas_fit_schedule_bindings",
]
