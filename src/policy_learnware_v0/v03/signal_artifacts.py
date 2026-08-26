"""Durable, private-by-default publication for v0.3 signal-atlas work.

The large signal matrix is deliberately executed as independently recoverable
work items.  A completed item is not merely marked complete: its checkpoint
stores the SHA-256 of the exact immutable JSON bytes that contain the complete
``SignalCellRun`` (including the private distance rows needed for recompute).

Checkpoint files are themselves immutable and versioned by content digest.
Callers retain the returned path + byte digest as the resume token.  On a fresh
process, every COMPLETE artifact is read through the digest-verifying artifact
reader and reconstructed as a typed ``SignalCellRun`` before any pending work
is resumed.  RUNNING/FAILED work is reset to PENDING; an unreferenced artifact
left by a crash is never trusted and a retry may reuse it only when its bytes
are exactly identical.

Private work/checkpoint artifacts live in ``signal_atlas_private``.  The sole
joint/public publication method accepts only ``SignalAtlasRun.to_public_dict``
and rejects distance rows and task taxonomy recursively.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from ..hashing import sha256_json
from .artifacts import V03ArtifactError, V03ArtifactLayout
from .preflight import ExecutionCheckpoint
from .signal_atlas import (
    FormalSignalAtlasAuthorization,
    SignalAtlasRun,
    signal_work_key,
)
from .signal_controls import (
    ExactRepeatDistanceResult,
    PairControlEvaluation,
    PairControlPlan,
)
from .signal_metrics import SignalDistanceRow, SignalMetricRecord
from .signal_diagnostics import SignalCellDiagnostics
from .signal_runtime import (
    SignalCellRun,
    SignalIdentityRegistry,
    SourceKernelProtocol,
)


PRIVATE_SIGNAL_CELL_ARTIFACT_SCHEMA = (
    "policy-learnware.v03-private-signal-cell-artifact.v0"
)
PUBLIC_SIGNAL_ATLAS_ARTIFACT_SCHEMA = (
    "policy-learnware.v03-public-signal-atlas.v0"
)
SIGNAL_CHECKPOINT_PUBLICATION_SCHEMA = (
    "policy-learnware.v03-signal-checkpoint-publication.v0"
)
FORMAL_PAIR_CONTROL_AUTHORIZATION_SCHEMA = (
    "policy-learnware.v03-formal-pair-control-authorization.v0"
)
PRIVATE_PAIR_CONTROL_EVALUATION_ARTIFACT_SCHEMA = (
    "policy-learnware.v03-private-pair-control-evaluation-artifact.v0"
)
PUBLIC_PAIR_CONTROL_PANEL_ARTIFACT_SCHEMA = (
    "policy-learnware.v03-public-pair-control-panel.v0"
)
PAIR_CONTROL_PUBLICATION_SCHEMA = (
    "policy-learnware.v03-pair-control-publication.v0"
)

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_PRIVATE_PUBLIC_KEYS = frozenset(
    {
        "rows",
        "distance_rows",
        "expected_source_by_query",
        "query_bank_id",
        "source_bank_id",
        "query_receipt_digest",
        "source_receipt_digest",
        "query_raw_dataset_digest",
        "source_raw_dataset_digest",
        "query_task_id",
        "source_task_id",
        "query_context_id",
        "source_context_id",
        "query_embodiment_id",
        "source_embodiment_id",
        "query_abi_contract_id",
        "source_abi_contract_id",
        "query_goal_contract_id",
        "source_goal_contract_id",
        "query_dynamics_context_id",
        "source_dynamics_context_id",
        "query_equivalence_class_id",
        "source_equivalence_class_id",
        "pair_id",
        "left_bank_id",
        "right_bank_id",
        "left_receipt_digest",
        "right_receipt_digest",
        "left_feature_bank_digest",
        "right_feature_bank_digest",
        "left_bank_reference",
        "right_bank_reference",
        "source_metric_record",
        "pair_metric_record",
        "direct_row",
        "bank_id",
        "represented_bank_digest",
        "data_role",
        "bank_geometries",
        "confusion_records",
        "true_identity",
        "predicted_identity",
    }
)


class SignalArtifactError(ValueError):
    """A work artifact, checkpoint transition, or public projection is invalid."""


def _digest(value: Any, where: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise SignalArtifactError(f"{where} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class FormalPairControlAuthorization:
    """Bind a private pair-control preregistration to the reviewed atlas freeze.

    Pair controls are a separate panel, not additional rows in the 39-cell
    matrix.  They nevertheless share the atlas measurement protocol and may
    run formally only when the reviewed T-P4-03 contract digest is the exact
    private ``PairControlPlan`` digest.
    """

    atlas_authorization: FormalSignalAtlasAuthorization
    pair_control_plan_digest: str
    measurement_protocol_digest: str
    hard_todo_evidence_digest: str
    authorization_digest: str | None = None
    schema: str = FORMAL_PAIR_CONTROL_AUTHORIZATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FORMAL_PAIR_CONTROL_AUTHORIZATION_SCHEMA:
            raise SignalArtifactError(
                "unsupported formal pair-control authorization"
            )
        if not isinstance(
            self.atlas_authorization, FormalSignalAtlasAuthorization
        ):
            raise SignalArtifactError(
                "pair-control authorization requires atlas authorization"
            )
        for name in (
            "pair_control_plan_digest",
            "measurement_protocol_digest",
            "hard_todo_evidence_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        evidence = {
            item.todo_id: item
            for item in self.atlas_authorization.freeze_manifest.hard_todo_evidence
        }.get("T-P4-03")
        if evidence is None:
            raise SignalArtifactError(
                "formal freeze is missing T-P4-03 pair-control evidence"
            )
        if (
            evidence.contract_digest != self.pair_control_plan_digest
            or evidence.evidence_digest != self.hard_todo_evidence_digest
        ):
            raise SignalArtifactError(
                "pair-control plan differs from reviewed T-P4-03 evidence"
            )
        expected = sha256_json(self._payload_without_digest())
        if self.authorization_digest is None:
            object.__setattr__(self, "authorization_digest", expected)
        elif _digest(self.authorization_digest, "authorization_digest") != expected:
            raise SignalArtifactError(
                "formal pair-control authorization digest mismatch"
            )

    @classmethod
    def bind(
        cls,
        atlas_authorization: FormalSignalAtlasAuthorization,
        *,
        plan: PairControlPlan,
        identity_registry: SignalIdentityRegistry,
    ) -> "FormalPairControlAuthorization":
        if not isinstance(plan, PairControlPlan) or not isinstance(
            identity_registry, SignalIdentityRegistry
        ):
            raise SignalArtifactError(
                "pair authorization requires typed plan and identity registry"
            )
        if not isinstance(
            atlas_authorization, FormalSignalAtlasAuthorization
        ) or (
            atlas_authorization.identity_registry_digest
            != identity_registry.registry_digest
        ):
            raise SignalArtifactError(
                "pair authorization belongs to another atlas identity registry"
            )
        if (
            plan.measurement_protocol_digest
            != identity_registry.measurement_protocol_digest
        ):
            raise SignalArtifactError(
                "pair-control plan differs from atlas measurement protocol"
            )
        identities = {item.bank_id: item for item in identity_registry.identities}
        for contract in plan.contracts:
            for reference in (contract.left, contract.right):
                identity = identities.get(reference.bank_id)
                if identity is None or (
                    identity.task_private_id != reference.registered_task_id
                    or identity.goal_contract_id != reference.goal_contract_id
                    or identity.context_id != reference.context_id
                    or identity.measurement_protocol_digest
                    != reference.measurement_protocol_digest
                    or identity.probe_seed_digest != reference.probe_seed_digest
                ):
                    raise SignalArtifactError(
                        "pair-control reference is absent from atlas identity registry"
                    )
        evidence = {
            item.todo_id: item
            for item in atlas_authorization.freeze_manifest.hard_todo_evidence
        }.get("T-P4-03")
        if evidence is None or evidence.contract_digest != plan.plan_digest:
            raise SignalArtifactError(
                "reviewed T-P4-03 evidence does not freeze this pair plan"
            )
        return cls(
            atlas_authorization=atlas_authorization,
            pair_control_plan_digest=plan.plan_digest,
            measurement_protocol_digest=plan.measurement_protocol_digest,
            hard_todo_evidence_digest=evidence.evidence_digest,
        )

    def validate(
        self,
        *,
        plan: PairControlPlan,
        identity_registry: SignalIdentityRegistry,
    ) -> None:
        if (
            not isinstance(plan, PairControlPlan)
            or not isinstance(identity_registry, SignalIdentityRegistry)
            or plan.plan_digest != self.pair_control_plan_digest
            or plan.measurement_protocol_digest
            != self.measurement_protocol_digest
            or identity_registry.registry_digest
            != self.atlas_authorization.identity_registry_digest
            or identity_registry.measurement_protocol_digest
            != self.measurement_protocol_digest
        ):
            raise SignalArtifactError(
                "pair-control authorization belongs to another formal protocol"
            )

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "formal_atlas_authorization_digest": (
                self.atlas_authorization.authorization_digest
            ),
            "freeze_manifest_digest": (
                self.atlas_authorization.freeze_manifest.freeze_manifest_digest
            ),
            "pair_control_plan_digest": self.pair_control_plan_digest,
            "measurement_protocol_digest": self.measurement_protocol_digest,
            "hard_todo_evidence_digest": self.hard_todo_evidence_digest,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "authorization_digest": self.authorization_digest}


def _strict(value: Any, expected: set[str], where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        actual = set(value) if isinstance(value, Mapping) else type(value).__name__
        raise SignalArtifactError(
            f"{where} fields differ: expected {sorted(expected)!r}, got {actual!r}"
        )
    return value


def _safe_work_id(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 240
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) is None
    ):
        raise SignalArtifactError(f"unsafe signal work ID: {value!r}")
    return value


def transition_execution_checkpoint(
    checkpoint: ExecutionCheckpoint,
    work_id: str,
    target_state: str,
    *,
    completed_artifact_digest: str | None = None,
) -> ExecutionCheckpoint:
    """Apply one legal immutable state transition.

    Legal edges are ``PENDING -> RUNNING -> COMPLETE|FAILED``.  Recovery uses
    ``ExecutionCheckpoint.resume`` to reset RUNNING/FAILED back to PENDING.
    COMPLETE is terminal and its exact artifact-byte digest is immutable.
    """

    if not isinstance(checkpoint, ExecutionCheckpoint):
        raise SignalArtifactError("checkpoint transition requires ExecutionCheckpoint")
    work_id = _safe_work_id(work_id)
    if work_id not in checkpoint.work_item_states:
        raise SignalArtifactError(f"unknown signal work item: {work_id}")
    current = checkpoint.work_item_states[work_id]
    legal = {
        ("PENDING", "RUNNING"),
        ("RUNNING", "COMPLETE"),
        ("RUNNING", "FAILED"),
    }
    if (current, target_state) not in legal:
        raise SignalArtifactError(
            f"illegal signal work transition: {current} -> {target_state}"
        )
    artifacts = dict(checkpoint.completed_artifact_digests)
    if target_state == "COMPLETE":
        if completed_artifact_digest is None:
            raise SignalArtifactError(
                "COMPLETE transition requires the actual artifact byte digest"
            )
        artifacts[work_id] = _digest(
            completed_artifact_digest,
            f"completed artifact digest for {work_id}",
        )
    elif completed_artifact_digest is not None:
        raise SignalArtifactError(
            "only a COMPLETE transition may bind an artifact digest"
        )
    states = dict(checkpoint.work_item_states)
    states[work_id] = target_state
    return ExecutionCheckpoint(
        execution_plan_digest=checkpoint.execution_plan_digest,
        work_item_states=states,
        completed_artifact_digests=artifacts,
        attempt=checkpoint.attempt,
    )


def serialize_signal_cell_run(
    *,
    work_id: str,
    work_item_digest: str,
    run: SignalCellRun,
) -> dict[str, Any]:
    """Create the lossless private payload for one completed cell run."""

    work_id = _safe_work_id(work_id)
    work_item_digest = _digest(work_item_digest, "work_item_digest")
    if not isinstance(run, SignalCellRun):
        raise SignalArtifactError("signal work executor must return SignalCellRun")
    if signal_work_key(run.cell_id, run.evaluation_seed) != work_id:
        raise SignalArtifactError("SignalCellRun identity differs from the work ID")
    run_work_digest = getattr(run, "work_item_digest", None)
    if run_work_digest is not None and run_work_digest != work_item_digest:
        raise SignalArtifactError(
            "SignalCellRun is bound to another SignalCellWorkItem"
        )
    body = {
        "schema": PRIVATE_SIGNAL_CELL_ARTIFACT_SCHEMA,
        "work_id": work_id,
        "work_item_digest": work_item_digest,
        "run": run.to_dict(),
        "kernel_protocol": run.kernel_protocol.to_dict(),
        "metric_record": run.metric_record.to_dict(),
        "diagnostics": run.diagnostics.to_private_dict(),
    }
    return {**body, "artifact_payload_digest": sha256_json(body)}


def _deserialize_kernel(value: Any) -> SourceKernelProtocol:
    expected = set(SourceKernelProtocol.__dataclass_fields__) | {
        "fit_scope",
        "estimator",
    }
    data = _strict(value, expected, "private kernel protocol")
    if data["fit_scope"] != "SOURCE_ONLY":
        raise SignalArtifactError("kernel artifact is not source-only")
    if data["estimator"] != "median-positive-euclidean-distance":
        raise SignalArtifactError("kernel artifact estimator differs from v0.3")
    return SourceKernelProtocol(
        **{
            name: data[name]
            for name in SourceKernelProtocol.__dataclass_fields__
        }
    )


def _deserialize_metric(value: Any) -> SignalMetricRecord:
    expected = {
        "schema",
        "cell_id",
        "view_or_condition_id",
        "representation_id",
        "representation_coordinate_digest",
        "representation_seed",
        "source_index_digest",
        "query_manifest_digest",
        "rows",
        "expected_source_by_query",
        "metric_values",
    }
    data = _strict(value, expected, "private signal metric")
    if not isinstance(data["rows"], list):
        raise SignalArtifactError("private signal metric rows must be a list")
    row_fields = set(SignalDistanceRow.__dataclass_fields__)
    rows = tuple(
        SignalDistanceRow(
            **dict(_strict(row, row_fields, "private signal distance row"))
        )
        for row in data["rows"]
    )
    return SignalMetricRecord(
        schema=data["schema"],
        cell_id=data["cell_id"],
        view_or_condition_id=data["view_or_condition_id"],
        representation_id=data["representation_id"],
        representation_coordinate_digest=data[
            "representation_coordinate_digest"
        ],
        representation_seed=data["representation_seed"],
        source_index_digest=data["source_index_digest"],
        query_manifest_digest=data["query_manifest_digest"],
        rows=rows,
        expected_source_by_query=data["expected_source_by_query"],
        metric_values=data["metric_values"],
    )


def deserialize_signal_cell_run(
    value: Any,
    *,
    expected_work_id: str,
    expected_work_item_digest: str,
    expected_plan_digest: str,
    expected_execution_protocol_digest: str,
    expected_execution_mode: str,
) -> SignalCellRun:
    """Reconstruct and revalidate a complete typed run from private JSON."""

    expected_work_id = _safe_work_id(expected_work_id)
    expected_work_item_digest = _digest(
        expected_work_item_digest, "expected_work_item_digest"
    )
    expected_plan_digest = _digest(expected_plan_digest, "expected_plan_digest")
    expected_execution_protocol_digest = _digest(
        expected_execution_protocol_digest,
        "expected_execution_protocol_digest",
    )
    fields = {
        "schema",
        "work_id",
        "work_item_digest",
        "run",
        "kernel_protocol",
        "metric_record",
        "diagnostics",
        "artifact_payload_digest",
    }
    data = _strict(value, fields, "private signal-cell artifact")
    if data["schema"] != PRIVATE_SIGNAL_CELL_ARTIFACT_SCHEMA:
        raise SignalArtifactError("unsupported private signal-cell artifact schema")
    body = {name: data[name] for name in fields - {"artifact_payload_digest"}}
    if _digest(data["artifact_payload_digest"], "artifact_payload_digest") != sha256_json(
        body
    ):
        raise SignalArtifactError("private signal-cell payload digest mismatch")
    if data["work_id"] != expected_work_id:
        raise SignalArtifactError("private artifact belongs to another work item")
    if data["work_item_digest"] != expected_work_item_digest:
        raise SignalArtifactError("private artifact work-item digest mismatch")

    kernel = _deserialize_kernel(data["kernel_protocol"])
    metric = _deserialize_metric(data["metric_record"])
    try:
        diagnostics = SignalCellDiagnostics.from_private_dict(data["diagnostics"])
    except Exception as error:
        raise SignalArtifactError(f"private signal diagnostics are invalid: {error}") from error
    run_data = data["run"]
    run_fields = set(SignalCellRun.__dataclass_fields__)
    projected_fields = (run_fields - {"kernel_protocol", "metric_record", "diagnostics"}) | {
        "kernel_protocol_digest",
        "metric_record_digest",
        "diagnostics_digest",
    }
    run_data = _strict(run_data, projected_fields, "private SignalCellRun")
    if run_data["kernel_protocol_digest"] != kernel.protocol_digest:
        raise SignalArtifactError("nested kernel protocol digest mismatch")
    if run_data["metric_record_digest"] != metric.record_digest:
        raise SignalArtifactError("nested metric record digest mismatch")
    if run_data["diagnostics_digest"] != diagnostics.diagnostics_digest:
        raise SignalArtifactError("nested diagnostics digest mismatch")
    kwargs = {
        name: run_data[name]
        for name in run_fields - {"kernel_protocol", "metric_record", "diagnostics"}
    }
    run = SignalCellRun(
        **kwargs,
        kernel_protocol=kernel,
        metric_record=metric,
        diagnostics=diagnostics,
    )
    if run.to_dict() != dict(run_data):
        raise SignalArtifactError("reconstructed SignalCellRun projection differs")
    if run.plan_digest != expected_plan_digest:
        raise SignalArtifactError("SignalCellRun belongs to another matrix plan")
    if run.execution_protocol_digest != expected_execution_protocol_digest:
        raise SignalArtifactError("SignalCellRun belongs to another execution protocol")
    if signal_work_key(run.cell_id, run.evaluation_seed) != expected_work_id:
        raise SignalArtifactError("SignalCellRun identity differs from the work ID")
    run_work_digest = getattr(run, "work_item_digest", None)
    if run_work_digest is not None and run_work_digest != expected_work_item_digest:
        raise SignalArtifactError("SignalCellRun work-item binding mismatch")
    run_mode = getattr(run, "execution_mode", expected_execution_mode)
    if run_mode != expected_execution_mode:
        raise SignalArtifactError("SignalCellRun execution mode mismatch")
    return run


@dataclass(frozen=True)
class PublishedSignalCheckpoint:
    """External resume token for one immutable checkpoint artifact."""

    checkpoint: ExecutionCheckpoint
    path: Path
    artifact_sha256: str
    schema: str = SIGNAL_CHECKPOINT_PUBLICATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SIGNAL_CHECKPOINT_PUBLICATION_SCHEMA:
            raise SignalArtifactError("unsupported checkpoint publication schema")
        if not isinstance(self.checkpoint, ExecutionCheckpoint):
            raise SignalArtifactError("publication requires ExecutionCheckpoint")
        object.__setattr__(self, "path", Path(self.path).resolve())
        object.__setattr__(
            self,
            "artifact_sha256",
            _digest(self.artifact_sha256, "checkpoint artifact_sha256"),
        )


@dataclass(frozen=True)
class PublishedPairControlEvaluation:
    """Byte-exact private publication token for one evaluated pair."""

    pair_digest: str
    evaluation_digest: str
    path: Path
    artifact_sha256: str
    schema: str = PAIR_CONTROL_PUBLICATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PAIR_CONTROL_PUBLICATION_SCHEMA:
            raise SignalArtifactError(
                "unsupported pair-control publication schema"
            )
        for name in ("pair_digest", "evaluation_digest", "artifact_sha256"):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        object.__setattr__(self, "path", Path(self.path).resolve())


class PairControlArtifactRunner:
    """Publish the pair panel separately from the 39-cell signal atlas."""

    def __init__(
        self,
        *,
        layout: V03ArtifactLayout,
        plan: PairControlPlan,
        identity_registry: SignalIdentityRegistry,
        formal_authorization: FormalPairControlAuthorization,
    ) -> None:
        if not isinstance(layout, V03ArtifactLayout) or layout.namespace != "joint":
            raise SignalArtifactError(
                "formal pair controls require the joint artifact namespace"
            )
        if not isinstance(plan, PairControlPlan) or not isinstance(
            identity_registry, SignalIdentityRegistry
        ):
            raise SignalArtifactError(
                "pair-control runner requires typed plan and identity registry"
            )
        if not isinstance(formal_authorization, FormalPairControlAuthorization):
            raise SignalArtifactError(
                "pair-control runner requires reviewed pair authorization"
            )
        formal_authorization.validate(
            plan=plan, identity_registry=identity_registry
        )
        self.layout = layout
        self.plan = plan
        self.identity_registry = identity_registry
        self.formal_authorization = formal_authorization

    def _private_path(self, pair_digest: str) -> Path:
        return self.layout.artifact(
            "signal_atlas_private",
            "pair_controls",
            f"pair-{_digest(pair_digest, 'pair_digest')}.json",
        )

    @staticmethod
    def _contract(evaluation: Any) -> Any:
        return getattr(evaluation, "contract", None)

    @staticmethod
    def _evaluation_digest(evaluation: Any) -> str:
        value = (
            evaluation.evaluation_digest
            if isinstance(evaluation, PairControlEvaluation)
            else getattr(evaluation, "result_digest", None)
        )
        return _digest(value, "pair formal-result digest")

    @staticmethod
    def _public_result(evaluation: Any) -> Mapping[str, Any]:
        if isinstance(evaluation, PairControlEvaluation):
            return evaluation.result.to_public_dict()
        if isinstance(evaluation, ExactRepeatDistanceResult):
            return evaluation.to_public_dict()
        raise SignalArtifactError("unknown pair-control formal result")

    def _validate_evaluation(self, evaluation: Any) -> None:
        if not isinstance(
            evaluation, (PairControlEvaluation, ExactRepeatDistanceResult)
        ):
            raise SignalArtifactError(
                "pair-control publication requires a typed evaluation"
            )
        contract = self._contract(evaluation)
        try:
            frozen = self.plan.contract(str(contract.pair_digest))
        except Exception as error:
            raise SignalArtifactError(
                "pair evaluation is absent from the frozen plan"
            ) from error
        if frozen.to_dict() != contract.to_dict():
            raise SignalArtifactError(
                "pair evaluation differs from the frozen contract"
            )
        if (
            evaluation.membership.measurement_protocol_digest
            != self.formal_authorization.measurement_protocol_digest
        ):
            raise SignalArtifactError(
                "pair evaluation differs from the frozen measurement protocol"
            )
        identities = {
            item.bank_id: item for item in self.identity_registry.identities
        }
        expected_receipts = (
            identities[contract.left.bank_id].receipt_digest,
            identities[contract.right.bank_id].receipt_digest,
        )
        if expected_receipts != (
            evaluation.membership.left_receipt_digest,
            evaluation.membership.right_receipt_digest,
        ):
            raise SignalArtifactError(
                "pair membership receipts differ from atlas identity registry"
            )

    def _private_payload(
        self, evaluation: Any
    ) -> dict[str, Any]:
        body = {
            "schema": PRIVATE_PAIR_CONTROL_EVALUATION_ARTIFACT_SCHEMA,
            "pair_control_plan_digest": self.plan.plan_digest,
            "formal_pair_control_authorization_digest": (
                self.formal_authorization.authorization_digest
            ),
            "evaluation": evaluation.to_private_dict(),
        }
        return {**body, "artifact_payload_digest": sha256_json(body)}

    def publish_private_evaluation(
        self, evaluation: Any, *, resume: bool = False
    ) -> PublishedPairControlEvaluation:
        self._validate_evaluation(evaluation)
        pair_digest = str(self._contract(evaluation).pair_digest)
        path = self._private_path(pair_digest)
        artifact_sha256 = self.layout.writer(
            "signal_atlas_private"
        ).publish_json(path, self._private_payload(evaluation), resume=resume)
        return PublishedPairControlEvaluation(
            pair_digest=pair_digest,
            evaluation_digest=self._evaluation_digest(evaluation),
            path=path,
            artifact_sha256=artifact_sha256,
        )

    def _verify_private_publication(
        self,
        evaluation: Any,
        publication: PublishedPairControlEvaluation,
    ) -> None:
        if not isinstance(publication, PublishedPairControlEvaluation):
            raise SignalArtifactError(
                "public pair panel requires typed private publications"
            )
        if (
            publication.pair_digest != self._contract(evaluation).pair_digest
            or publication.evaluation_digest != self._evaluation_digest(evaluation)
            or publication.path != self._private_path(publication.pair_digest).resolve()
        ):
            raise SignalArtifactError(
                "private pair publication belongs to another evaluation"
            )
        try:
            value = self.layout.reader("signal_atlas_private").load_json(
                publication.path,
                expected_sha256=publication.artifact_sha256,
            )
        except V03ArtifactError as error:
            raise SignalArtifactError(
                "private pair publication failed byte verification"
            ) from error
        if value != self._private_payload(evaluation):
            raise SignalArtifactError(
                "private pair publication differs from typed evaluation"
            )

    def _validate_against_atlas_run(
        self, evaluation: Any, atlas_run: SignalAtlasRun
    ) -> None:
        if not isinstance(atlas_run, SignalAtlasRun) or (
            atlas_run.formal_authorization.authorization_digest
            != self.formal_authorization.atlas_authorization.authorization_digest
        ):
            raise SignalArtifactError(
                "pair panel requires the authorized complete SignalAtlasRun"
            )
        metric_record = (
            evaluation.source_metric_record
            if isinstance(evaluation, PairControlEvaluation)
            else evaluation.metric_record
        )
        candidates = tuple(
            (work_key, atlas_run.work_items[work_key], run)
            for work_key, run in atlas_run.cell_runs.items()
            if run.metric_record.record_digest == metric_record.record_digest
        )
        if len(candidates) != 1:
            raise SignalArtifactError(
                "pair result is not uniquely backed by an authorized atlas cell"
            )
        _work_key, work_item, run = candidates[0]
        work_item.validate_run(run)
        if isinstance(evaluation, ExactRepeatDistanceResult) and (
            evaluation.signal_cell_run_digest != run.run_digest
            or evaluation.kernel_protocol_digest
            != run.kernel_protocol.protocol_digest
            or evaluation.query_run_digest
            != run.query_run_digests.get(evaluation.contract.left.bank_id)
        ):
            raise SignalArtifactError(
                "exact-repeat provenance differs from authorized atlas run"
            )
        query_features = {
            bank.feature_bank.receipt.bank_id: bank.feature_bank
            for bank in work_item.query_banks
        }
        source_features = {
            bank.feature_bank.receipt.bank_id: bank.feature_bank
            for bank in work_item.source_banks
        }
        if isinstance(evaluation, PairControlEvaluation):
            left = query_features.get(evaluation.contract.left.bank_id)
            right = query_features.get(evaluation.contract.right.bank_id)
        else:
            left = query_features.get(evaluation.contract.left.bank_id)
            right = source_features.get(evaluation.contract.right.bank_id)
        if left is None or right is None or (
            left.receipt.receipt_digest
            != evaluation.membership.left_receipt_digest
            or right.receipt.receipt_digest
            != evaluation.membership.right_receipt_digest
            or left.feature_bank_digest
            != evaluation.membership.left_feature_bank_digest
            or right.feature_bank_digest
            != evaluation.membership.right_feature_bank_digest
        ):
            raise SignalArtifactError(
                "pair membership/feature banks differ from authorized atlas work item"
            )

    @staticmethod
    def _reject_private_public_payload(value: Any) -> None:
        SignalAtlasArtifactRunner._reject_private_public_payload(value)

    def publish_public_panel(
        self,
        *,
        atlas_run: SignalAtlasRun,
        evaluations: Sequence[Any],
        publications: Sequence[PublishedPairControlEvaluation],
        resume: bool = False,
    ) -> tuple[Path, str]:
        """Publish aggregates only after every frozen pair is byte verified."""

        if not all(
            isinstance(item, (PairControlEvaluation, ExactRepeatDistanceResult))
            for item in evaluations
        ):
            raise SignalArtifactError(
                "public pair panel requires typed formal results"
            )
        evaluation_by_pair = {
            str(self._contract(item).pair_digest): item for item in evaluations
        }
        publication_by_pair = {item.pair_digest: item for item in publications}
        expected_pairs = {
            str(item.pair_digest) for item in self.plan.contracts
        }
        if (
            len(evaluation_by_pair) != len(tuple(evaluations))
            or len(publication_by_pair) != len(tuple(publications))
            or set(evaluation_by_pair) != expected_pairs
            or set(publication_by_pair) != expected_pairs
        ):
            raise SignalArtifactError(
                "public pair panel requires exact frozen pair coverage"
            )
        for pair_digest in sorted(expected_pairs):
            evaluation = evaluation_by_pair[pair_digest]
            self._validate_evaluation(evaluation)
            self._validate_against_atlas_run(evaluation, atlas_run)
            self._verify_private_publication(
                evaluation, publication_by_pair[pair_digest]
            )
        private_panel_digest = sha256_json(
            {
                "schema": "policy-learnware.v03-private-pair-control-panel.v0",
                "pair_control_plan_digest": self.plan.plan_digest,
                "evaluations": {
                    pair_digest: {
                        "evaluation_digest": evaluation_by_pair[
                            pair_digest
                        ].evaluation_digest
                        if isinstance(
                            evaluation_by_pair[pair_digest], PairControlEvaluation
                        )
                        else evaluation_by_pair[pair_digest].result_digest,
                        "artifact_sha256": publication_by_pair[
                            pair_digest
                        ].artifact_sha256,
                    }
                    for pair_digest in sorted(expected_pairs)
                },
            }
        )
        body = {
            "schema": PUBLIC_PAIR_CONTROL_PANEL_ARTIFACT_SCHEMA,
            "pair_control_plan_digest": self.plan.plan_digest,
            "formal_pair_control_authorization_digest": (
                self.formal_authorization.authorization_digest
            ),
            "formal_atlas_authorization_digest": (
                self.formal_authorization.atlas_authorization.authorization_digest
            ),
            "freeze_manifest_digest": (
                self.formal_authorization.atlas_authorization.freeze_manifest.freeze_manifest_digest
            ),
            "control_results": [
                self._public_result(evaluation_by_pair[pair_digest])
                for pair_digest in sorted(expected_pairs)
            ],
            "private_pair_membership_withheld": True,
            "private_panel_digest": private_panel_digest,
        }
        payload = {**body, "public_projection_digest": sha256_json(body)}
        self._reject_private_public_payload(payload)
        destination = self.layout.artifact(
            "pair_controls", "public", "pair_control_panel.json"
        )
        artifact_sha256 = self.layout._authorized_pair_control_writer().publish_json(
            destination, payload, resume=resume
        )
        return destination, artifact_sha256


@dataclass(frozen=True)
class SignalAtlasProgress:
    checkpoint_publication: PublishedSignalCheckpoint
    completed_runs: Mapping[str, SignalCellRun]

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint_publication, PublishedSignalCheckpoint):
            raise SignalArtifactError("progress requires a checkpoint publication")
        runs = dict(sorted(self.completed_runs.items()))
        complete = {
            work_id
            for work_id, state in self.checkpoint_publication.checkpoint.work_item_states.items()
            if state == "COMPLETE"
        }
        if set(runs) != complete or not all(
            isinstance(run, SignalCellRun) for run in runs.values()
        ):
            raise SignalArtifactError(
                "progress runs must exactly cover checkpoint COMPLETE items"
            )
        object.__setattr__(self, "completed_runs", MappingProxyType(runs))


class SignalAtlasExecutionInterrupted(SignalArtifactError):
    """Executor/publisher failure with the last durable FAILED checkpoint."""

    def __init__(
        self,
        work_id: str,
        checkpoint_publication: PublishedSignalCheckpoint,
        completed_runs: Mapping[str, SignalCellRun],
    ) -> None:
        super().__init__(f"signal work item failed durably: {work_id}")
        self.work_id = work_id
        self.checkpoint_publication = checkpoint_publication
        self.completed_runs = MappingProxyType(dict(completed_runs))


class SignalAtlasArtifactRunner:
    """Immutable per-item publisher and fresh-process recovery coordinator."""

    def __init__(
        self,
        *,
        layout: V03ArtifactLayout,
        execution_plan_digest: str,
        plan_digest: str,
        execution_protocol_digest: str,
        expected_work_item_digests: Mapping[str, str],
        execution_mode: str,
        formal_authorization: FormalSignalAtlasAuthorization | None = None,
    ) -> None:
        if not isinstance(layout, V03ArtifactLayout):
            raise SignalArtifactError("signal runner requires V03ArtifactLayout")
        if execution_mode not in {"DEVELOPMENT_SMOKE", "FORMAL"}:
            raise SignalArtifactError("signal runner execution mode is invalid")
        if layout.namespace == "joint" and execution_mode != "FORMAL":
            raise SignalArtifactError("joint signal artifacts require FORMAL mode")
        if layout.namespace == "development" and execution_mode != "DEVELOPMENT_SMOKE":
            raise SignalArtifactError(
                "formal signal artifacts must use the joint namespace"
            )
        work = {
            _safe_work_id(work_id): _digest(digest, f"work digest for {work_id}")
            for work_id, digest in sorted(expected_work_item_digests.items())
        }
        if not work:
            raise SignalArtifactError("signal runner requires expected work items")
        if execution_mode == "FORMAL":
            if not isinstance(
                formal_authorization, FormalSignalAtlasAuthorization
            ):
                raise SignalArtifactError(
                    "formal signal runner requires external atlas authorization"
                )
            observed_execution_plan = sha256_json(
                {
                    "schema": "policy-learnware.v03-signal-execution-plan.v0",
                    "plan_digest": plan_digest,
                    "execution_protocol_digest": execution_protocol_digest,
                    "work_keys": sorted(work),
                }
            )
            if (
                formal_authorization.plan_digest != plan_digest
                or formal_authorization.execution_protocol_digest
                != execution_protocol_digest
                or formal_authorization.execution_plan_digest
                != execution_plan_digest
                or formal_authorization.execution_plan_digest
                != observed_execution_plan
            ):
                raise SignalArtifactError(
                    "formal signal runner differs from its reviewed work graph"
                )
            # The key-only execution plan is insufficient: bind each key to
            # the reviewed work-item digest (banks, mapping and source fit).
            observed_work_graph = sha256_json(
                {
                    "schema": "policy-learnware.v03-signal-work-item-graph.v0",
                    "plan_digest": plan_digest,
                    "execution_protocol_digest": execution_protocol_digest,
                    "work_item_digests": work,
                }
            )
            if (
                observed_work_graph != formal_authorization.work_item_graph_digest
                or observed_work_graph
                != formal_authorization.freeze_manifest.signal_work_item_graph_digest
            ):
                raise SignalArtifactError(
                    "formal signal runner work-item contents differ from reviewed freeze"
                )
        elif formal_authorization is not None:
            raise SignalArtifactError(
                "development signal runner cannot carry formal authorization"
            )
        self.layout = layout
        self.execution_plan_digest = _digest(
            execution_plan_digest, "execution_plan_digest"
        )
        self.plan_digest = _digest(plan_digest, "plan_digest")
        self.execution_protocol_digest = _digest(
            execution_protocol_digest, "execution_protocol_digest"
        )
        self.expected_work_item_digests = MappingProxyType(work)
        self.execution_mode = execution_mode
        self.formal_authorization = formal_authorization

    def _work_path(self, work_id: str) -> Path:
        return self.layout.artifact(
            "signal_atlas_private", "work_items", f"{_safe_work_id(work_id)}.json"
        )

    def _checkpoint_path(self, checkpoint: ExecutionCheckpoint) -> Path:
        return self.layout.artifact(
            "signal_atlas_private",
            "checkpoints",
            f"checkpoint-{checkpoint.checkpoint_digest}.json",
        )

    def _validate_checkpoint(self, checkpoint: ExecutionCheckpoint) -> None:
        if not isinstance(checkpoint, ExecutionCheckpoint):
            raise SignalArtifactError("signal runner requires ExecutionCheckpoint")
        if checkpoint.execution_plan_digest != self.execution_plan_digest:
            raise SignalArtifactError("checkpoint belongs to another execution plan")
        if set(checkpoint.work_item_states) != set(self.expected_work_item_digests):
            raise SignalArtifactError("checkpoint work coverage differs from runner")

    def publish_checkpoint(
        self, checkpoint: ExecutionCheckpoint
    ) -> PublishedSignalCheckpoint:
        self._validate_checkpoint(checkpoint)
        path = self._checkpoint_path(checkpoint)
        artifact_sha256 = self.layout.writer("signal_atlas_private").publish_json(
            path, checkpoint.to_dict(), resume=True
        )
        return PublishedSignalCheckpoint(
            checkpoint=checkpoint,
            path=path,
            artifact_sha256=artifact_sha256,
        )

    def start(self, checkpoint: ExecutionCheckpoint) -> SignalAtlasProgress:
        """Publish an all-PENDING initial graph before executing any work."""

        self._validate_checkpoint(checkpoint)
        if set(checkpoint.work_item_states.values()) != {"PENDING"}:
            raise SignalArtifactError("initial signal checkpoint must be all PENDING")
        if checkpoint.completed_artifact_digests:
            raise SignalArtifactError("initial signal checkpoint cannot bind outputs")
        return SignalAtlasProgress(self.publish_checkpoint(checkpoint), {})

    def _load_complete_run(
        self, work_id: str, artifact_sha256: str
    ) -> SignalCellRun:
        path = self._work_path(work_id)
        try:
            value = self.layout.reader("signal_atlas_private").load_json(
                path, expected_sha256=artifact_sha256
            )
        except V03ArtifactError as error:
            raise SignalArtifactError(
                f"completed signal artifact failed byte verification: {work_id}"
            ) from error
        return deserialize_signal_cell_run(
            value,
            expected_work_id=work_id,
            expected_work_item_digest=self.expected_work_item_digests[work_id],
            expected_plan_digest=self.plan_digest,
            expected_execution_protocol_digest=self.execution_protocol_digest,
            expected_execution_mode=self.execution_mode,
        )

    def resume(
        self,
        checkpoint_path: str | Path,
        *,
        expected_checkpoint_sha256: str,
    ) -> SignalAtlasProgress:
        """Verify a resume token and reconstruct every completed typed run.

        Missing or modified COMPLETE artifacts fail closed.  RUNNING and FAILED
        states are reset to PENDING in a newly published checkpoint.  Files not
        referenced by a COMPLETE state are not trusted or loaded; on retry the
        immutable writer accepts them only if the newly computed bytes match.
        """

        try:
            value = self.layout.reader("signal_atlas_private").load_json(
                checkpoint_path,
                expected_sha256=_digest(
                    expected_checkpoint_sha256,
                    "expected_checkpoint_sha256",
                ),
            )
        except V03ArtifactError as error:
            raise SignalArtifactError("signal checkpoint failed byte verification") from error
        try:
            checkpoint = ExecutionCheckpoint.from_dict(value)
        except Exception as error:
            raise SignalArtifactError("signal checkpoint failed typed validation") from error
        self._validate_checkpoint(checkpoint)
        completed = {
            work_id: self._load_complete_run(work_id, artifact_sha256)
            for work_id, artifact_sha256 in checkpoint.completed_artifact_digests.items()
        }
        if any(
            state != "COMPLETE" for state in checkpoint.work_item_states.values()
        ):
            checkpoint = checkpoint.resume()
            publication = self.publish_checkpoint(checkpoint)
        else:
            publication = PublishedSignalCheckpoint(
                checkpoint=checkpoint,
                path=Path(checkpoint_path),
                artifact_sha256=expected_checkpoint_sha256,
            )
        return SignalAtlasProgress(publication, completed)

    def run_remaining(
        self,
        progress: SignalAtlasProgress,
        executor: Callable[[str], SignalCellRun],
    ) -> SignalAtlasProgress:
        """Execute pending items, atomically publishing each before COMPLETE."""

        if not isinstance(progress, SignalAtlasProgress) or not callable(executor):
            raise SignalArtifactError("run_remaining requires progress and executor")
        checkpoint = progress.checkpoint_publication.checkpoint
        self._validate_checkpoint(checkpoint)
        if any(state == "RUNNING" for state in checkpoint.work_item_states.values()):
            raise SignalArtifactError("resume/reset RUNNING items before execution")
        completed = dict(progress.completed_runs)
        for work_id in sorted(self.expected_work_item_digests):
            if checkpoint.work_item_states[work_id] == "COMPLETE":
                continue
            if checkpoint.work_item_states[work_id] != "PENDING":
                raise SignalArtifactError("resume/reset FAILED items before execution")
            checkpoint = transition_execution_checkpoint(
                checkpoint, work_id, "RUNNING"
            )
            self.publish_checkpoint(checkpoint)
            try:
                run = executor(work_id)
                payload = serialize_signal_cell_run(
                    work_id=work_id,
                    work_item_digest=self.expected_work_item_digests[work_id],
                    run=run,
                )
                artifact_sha256 = self.layout.writer(
                    "signal_atlas_private"
                ).publish_json(self._work_path(work_id), payload, resume=True)
                # Round-trip the exact bytes before they can become COMPLETE.
                verified_run = self._load_complete_run(work_id, artifact_sha256)
            except Exception as error:
                checkpoint = transition_execution_checkpoint(
                    checkpoint, work_id, "FAILED"
                )
                publication = self.publish_checkpoint(checkpoint)
                raise SignalAtlasExecutionInterrupted(
                    work_id, publication, completed
                ) from error
            checkpoint = transition_execution_checkpoint(
                checkpoint,
                work_id,
                "COMPLETE",
                completed_artifact_digest=artifact_sha256,
            )
            completed[work_id] = verified_run
            publication = self.publish_checkpoint(checkpoint)
            progress = SignalAtlasProgress(publication, completed)
        return progress

    @staticmethod
    def _reject_private_public_payload(value: Any) -> None:
        if isinstance(value, Mapping):
            leaked = set(value) & _PRIVATE_PUBLIC_KEYS
            if leaked:
                raise SignalArtifactError(
                    f"public signal atlas leaks private fields: {sorted(leaked)!r}"
                )
            for item in value.values():
                SignalAtlasArtifactRunner._reject_private_public_payload(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                SignalAtlasArtifactRunner._reject_private_public_payload(item)

    def publish_public_atlas(
        self,
        *,
        atlas_run: Any,
        checkpoint: ExecutionCheckpoint,
        resume: bool = False,
    ) -> tuple[Path, str]:
        """Publish only the aggregate public projection of a complete formal run."""

        if self.layout.namespace != "joint" or self.execution_mode != "FORMAL":
            raise SignalArtifactError(
                "public signal atlas publication requires a formal joint runner"
            )
        self._validate_checkpoint(checkpoint)
        if set(checkpoint.work_item_states.values()) != {"COMPLETE"}:
            raise SignalArtifactError(
                "public signal atlas cannot publish before every work item completes"
            )
        # A syntactically COMPLETE checkpoint is insufficient.  Re-read every
        # checkpoint-bound private artifact by its exact byte SHA and rebuild
        # the typed run before accepting any aggregate projection.
        verified_runs = {
            work_id: self._load_complete_run(work_id, artifact_sha256)
            for work_id, artifact_sha256 in checkpoint.completed_artifact_digests.items()
        }
        if not isinstance(atlas_run, SignalAtlasRun):
            raise SignalArtifactError(
                "formal publication requires a coverage-complete typed SignalAtlasRun"
            )
        if (
            self.formal_authorization is None
            or atlas_run.formal_authorization.authorization_digest
            != self.formal_authorization.authorization_digest
        ):
            raise SignalArtifactError(
                "atlas run differs from the externally reviewed authorization"
            )
        atlas_work = getattr(atlas_run, "work_item_digests", None)
        if dict(atlas_work or {}) != dict(self.expected_work_item_digests):
            raise SignalArtifactError("atlas run work coverage differs from publisher")
        atlas_cell_runs = getattr(atlas_run, "cell_runs", None)
        if atlas_cell_runs is not None:
            if set(atlas_cell_runs) != set(verified_runs) or any(
                atlas_cell_runs[work_id].run_digest != run.run_digest
                for work_id, run in verified_runs.items()
            ):
                raise SignalArtifactError(
                    "atlas run differs from checkpoint-verified cell artifacts"
                )
        payload = atlas_run.to_public_dict()
        if not isinstance(payload, Mapping):
            raise SignalArtifactError("atlas public projection must be a mapping")
        if payload.get("schema") != PUBLIC_SIGNAL_ATLAS_ARTIFACT_SCHEMA:
            raise SignalArtifactError("unsupported public signal-atlas projection")
        supplied_digest = payload.get("public_projection_digest")
        body = dict(payload)
        body.pop("public_projection_digest", None)
        if _digest(supplied_digest, "public_projection_digest") != sha256_json(body):
            raise SignalArtifactError("public signal-atlas projection digest mismatch")
        self._reject_private_public_payload(payload)
        destination = self.layout.artifact(
            "signal_atlas", "public", "signal_atlas.json"
        )
        artifact_sha256 = self.layout._authorized_signal_atlas_writer().publish_json(
            destination, payload, resume=resume
        )
        return destination, artifact_sha256


__all__ = [
    "FORMAL_PAIR_CONTROL_AUTHORIZATION_SCHEMA",
    "FormalPairControlAuthorization",
    "PAIR_CONTROL_PUBLICATION_SCHEMA",
    "PRIVATE_PAIR_CONTROL_EVALUATION_ARTIFACT_SCHEMA",
    "PRIVATE_SIGNAL_CELL_ARTIFACT_SCHEMA",
    "PUBLIC_PAIR_CONTROL_PANEL_ARTIFACT_SCHEMA",
    "PUBLIC_SIGNAL_ATLAS_ARTIFACT_SCHEMA",
    "PairControlArtifactRunner",
    "PublishedPairControlEvaluation",
    "SIGNAL_CHECKPOINT_PUBLICATION_SCHEMA",
    "PublishedSignalCheckpoint",
    "SignalArtifactError",
    "SignalAtlasArtifactRunner",
    "SignalAtlasExecutionInterrupted",
    "SignalAtlasProgress",
    "deserialize_signal_cell_run",
    "serialize_signal_cell_run",
    "transition_execution_checkpoint",
]
