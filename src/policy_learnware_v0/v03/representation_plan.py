"""Frozen architecture/objective schedule for the v0.3 representation ladder."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from ..hashing import sha256_json
from .corro_trainers import (
    CorroOptimizationConfig,
    FormalCorroTrainerContract,
    TASK_SUPCON_OBJECTIVE_DIGEST,
)
from .representation_ladder import (
    R0_PADDED_RAW,
    R1_FIXED_RANDOM_LINEAR,
    R2_SOURCE_PCA_WHITEN,
    R3_MATCHED_RANDOM_MLP,
    R5L_SUPERVISED_LINEAR,
    R5_VIEW_SPECIFIC_CORRO_REFIT,
    R_HIST_RANDOM_TANH,
    RepresentationManifest,
    TrainingRequest,
)
from .signal_controls import HistoricalRandomTanhSpec
from .signal_matrix import SignalMatrixPlan


REPRESENTATION_EXECUTION_PLAN_SCHEMA = (
    "policy-learnware.v03-representation-execution-plan.v0"
)


class RepresentationPlanError(ValueError):
    """A representation architecture, objective or seed drifted from the freeze."""


def _digest(value: Any, where: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        raise RepresentationPlanError(f"{where} must be a lowercase SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise RepresentationPlanError(
            f"{where} must be a lowercase SHA-256 digest"
        ) from error
    return value


@dataclass(frozen=True)
class RepresentationExecutionPlan:
    signal_matrix_digest: str
    shared_output_dim: int
    hidden_dims: tuple[int, int]
    pca_whiten: bool
    pca_epsilon: float
    objective_digest: str
    optimization: CorroOptimizationConfig
    historical_seed: int
    historical_output_dim: int
    historical_protocol_digest: str
    historical_checkpoint_digest: str
    plan_digest: str | None = None
    schema: str = REPRESENTATION_EXECUTION_PLAN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != REPRESENTATION_EXECUTION_PLAN_SCHEMA:
            raise RepresentationPlanError("unsupported representation execution plan")
        for name in (
            "signal_matrix_digest",
            "objective_digest",
            "historical_protocol_digest",
            "historical_checkpoint_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if self.shared_output_dim not in {32, 64}:
            raise RepresentationPlanError("shared output dimension must be 32 or 64")
        if (
            not isinstance(self.hidden_dims, tuple)
            or len(self.hidden_dims) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in self.hidden_dims)
        ):
            raise RepresentationPlanError("hidden_dims must contain two positive widths")
        if self.hidden_dims != (256, 256):
            raise RepresentationPlanError("v0.3 CORRO-style hidden widths are fixed at 256×256")
        if self.pca_whiten is not True:
            raise RepresentationPlanError("formal R2 must enable whitening")
        if (
            isinstance(self.pca_epsilon, bool)
            or not isinstance(self.pca_epsilon, (int, float))
            or not math.isfinite(float(self.pca_epsilon))
            or float(self.pca_epsilon) <= 0.0
        ):
            raise RepresentationPlanError("pca_epsilon must be finite and positive")
        object.__setattr__(self, "pca_epsilon", float(self.pca_epsilon))
        if self.objective_digest != TASK_SUPCON_OBJECTIVE_DIGEST:
            raise RepresentationPlanError("R5/R5L objective must be frozen task-SupCon")
        if not isinstance(self.optimization, CorroOptimizationConfig):
            raise RepresentationPlanError("optimization must be CorroOptimizationConfig")
        for name in ("historical_seed", "historical_output_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RepresentationPlanError(f"{name} is invalid")
        if self.historical_output_dim <= 0:
            raise RepresentationPlanError("historical_output_dim must be positive")
        expected = sha256_json(self._payload_without_digest())
        if self.plan_digest is None:
            object.__setattr__(self, "plan_digest", expected)
        elif _digest(self.plan_digest, "plan_digest") != expected:
            raise RepresentationPlanError("representation plan digest mismatch")

    @classmethod
    def create(
        cls,
        *,
        signal_plan: SignalMatrixPlan,
        historical_spec: HistoricalRandomTanhSpec,
        shared_output_dim: int = 32,
        hidden_dims: tuple[int, int] = (256, 256),
        pca_epsilon: float = 1.0e-12,
        optimization: CorroOptimizationConfig | None = None,
    ) -> "RepresentationExecutionPlan":
        if not isinstance(signal_plan, SignalMatrixPlan):
            raise RepresentationPlanError("representation plan requires signal plan")
        if not isinstance(historical_spec, HistoricalRandomTanhSpec):
            raise RepresentationPlanError("representation plan requires historical spec")
        return cls(
            signal_matrix_digest=str(signal_plan.plan_digest),
            shared_output_dim=shared_output_dim,
            hidden_dims=hidden_dims,
            pca_whiten=True,
            pca_epsilon=pca_epsilon,
            objective_digest=TASK_SUPCON_OBJECTIVE_DIGEST,
            optimization=optimization or CorroOptimizationConfig(),
            historical_seed=historical_spec.seed,
            historical_output_dim=historical_spec.output_dim,
            historical_protocol_digest=str(
                historical_spec.representation_protocol_digest
            ),
            historical_checkpoint_digest=str(historical_spec.checkpoint_digest),
        )

    @property
    def optimization_digest(self) -> str:
        return self.optimization.optimization_digest

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "signal_matrix_digest": self.signal_matrix_digest,
            "shared_output_dim": self.shared_output_dim,
            "hidden_dims": list(self.hidden_dims),
            "r1_distribution": "normal(0,1/sqrt(input_dim))",
            "r1_bias": False,
            "r1_nonlinearity": None,
            "pca_solver": "numpy.linalg.svd(full_matrices=false)",
            "pca_sign_convention": "largest-absolute-loading-positive",
            "pca_whiten": self.pca_whiten,
            "pca_epsilon": self.pca_epsilon,
            "activation": "relu",
            "l2_normalize_output": True,
            "objective_digest": self.objective_digest,
            "optimization_digest": self.optimization_digest,
            "representation_seeds": [0, 1, 2],
            "historical_seed": self.historical_seed,
            "historical_output_dim": self.historical_output_dim,
            "historical_protocol_digest": self.historical_protocol_digest,
            "historical_checkpoint_digest": self.historical_checkpoint_digest,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "plan_digest": self.plan_digest}

    def validate_training_request(self, request: TrainingRequest) -> None:
        if not isinstance(request, TrainingRequest):
            raise RepresentationPlanError("training request must be typed")
        if request.output_dim != self.shared_output_dim:
            raise RepresentationPlanError("training request output dimension drifted")
        if request.objective_digest != self.objective_digest:
            raise RepresentationPlanError("training request objective drifted")
        if request.seed not in {0, 1, 2}:
            raise RepresentationPlanError("training request seed is not preregistered")
        if request.representation_id == R5_VIEW_SPECIFIC_CORRO_REFIT:
            if request.hidden_dims != self.hidden_dims or request.activation != "relu":
                raise RepresentationPlanError("R5 architecture drifted")
        elif request.representation_id == R5L_SUPERVISED_LINEAR:
            if request.hidden_dims or request.activation is not None:
                raise RepresentationPlanError("R5L architecture drifted")
        else:
            raise RepresentationPlanError("request is not an R5/R5L training request")

    def validate_optimization_config(
        self, optimization: CorroOptimizationConfig
    ) -> None:
        """Require a trainer/restore call to use the optimization freeze verbatim."""

        if not isinstance(optimization, CorroOptimizationConfig):
            raise RepresentationPlanError(
                "optimization config must be CorroOptimizationConfig"
            )
        if optimization.optimization_digest != self.optimization_digest:
            raise RepresentationPlanError("optimization config drifted from freeze")

    def validate_formal_trainer_contract(
        self,
        contract: FormalCorroTrainerContract,
        request: TrainingRequest,
    ) -> None:
        """Require formal R5/R5L publication to use the frozen trainer route."""

        if not isinstance(contract, FormalCorroTrainerContract):
            raise RepresentationPlanError(
                "formal fit requires FormalCorroTrainerContract"
            )
        self.validate_training_request(request)
        if (
            contract.training_request_digest != request.request_digest
            or contract.representation_id != request.representation_id
        ):
            raise RepresentationPlanError(
                "formal trainer request differs from representation freeze"
            )
        if contract.optimization_digest != self.optimization_digest:
            raise RepresentationPlanError(
                "formal trainer optimization differs from representation freeze"
            )
        if contract.objective_digest != self.objective_digest:
            raise RepresentationPlanError(
                "formal trainer objective differs from representation freeze"
            )

    def validate_manifest(self, manifest: RepresentationManifest) -> None:
        if not isinstance(manifest, RepresentationManifest):
            raise RepresentationPlanError("representation manifest must be typed")
        representation_id = manifest.representation_id
        if representation_id == R0_PADDED_RAW:
            expected_protocol = sha256_json(
                {
                    "schema": "policy-learnware.v03-r0-identity-protocol.v0",
                    "operation": "identity",
                }
            )
            valid = (
                manifest.output_dim == manifest.input_dim
                and manifest.seed is None
                and manifest.protocol_digest == expected_protocol
            )
        elif representation_id == R1_FIXED_RANDOM_LINEAR:
            expected_protocol = sha256_json(
                {
                    "schema": "policy-learnware.v03-r1-random-linear-protocol.v0",
                    "distribution": "normal(0,1/sqrt(input_dim))",
                    "bias": False,
                    "nonlinearity": None,
                    "input_dim": manifest.input_dim,
                    "output_dim": self.shared_output_dim,
                    "seed": manifest.seed,
                }
            )
            valid = (
                manifest.output_dim == self.shared_output_dim
                and manifest.seed in {0, 1, 2}
                and manifest.protocol_digest == expected_protocol
            )
        elif representation_id == R2_SOURCE_PCA_WHITEN:
            expected_protocol = sha256_json(
                {
                    "schema": "policy-learnware.v03-r2-pca-protocol.v0",
                    "solver": "numpy.linalg.svd(full_matrices=false)",
                    "sign_convention": "largest-absolute-loading-positive",
                    "whiten": True,
                    "epsilon": self.pca_epsilon,
                    "output_dim": self.shared_output_dim,
                }
            )
            valid = (
                manifest.output_dim == self.shared_output_dim
                and manifest.seed is None
                and manifest.protocol_digest == expected_protocol
            )
        elif representation_id == R3_MATCHED_RANDOM_MLP:
            expected_protocol = sha256_json(
                {
                    "schema": "policy-learnware.v03-r3-random-mlp-protocol.v0",
                    "hidden_dims": list(self.hidden_dims),
                    "activation": "relu",
                    "l2_normalize_output": True,
                    "initializer": "he-normal-zero-bias",
                    "seed": manifest.seed,
                }
            )
            valid = (
                manifest.output_dim == self.shared_output_dim
                and manifest.seed in {0, 1, 2}
                and manifest.protocol_digest == expected_protocol
            )
        elif representation_id in {
            R5_VIEW_SPECIFIC_CORRO_REFIT,
            R5L_SUPERVISED_LINEAR,
        }:
            if (
                manifest.output_dim != self.shared_output_dim
                or manifest.seed not in {0, 1, 2}
            ):
                raise RepresentationPlanError(
                    f"representation manifest drifted from freeze: {representation_id}"
                )
            request = TrainingRequest(
                representation_id=representation_id,
                input_dim=manifest.input_dim,
                output_dim=self.shared_output_dim,
                hidden_dims=(
                    self.hidden_dims
                    if representation_id == R5_VIEW_SPECIFIC_CORRO_REFIT
                    else ()
                ),
                activation=(
                    "relu"
                    if representation_id == R5_VIEW_SPECIFIC_CORRO_REFIT
                    else None
                ),
                l2_normalize_output=True,
                objective_digest=self.objective_digest,
                seed=manifest.seed,
            )
            self.validate_training_request(request)
            valid = manifest.protocol_digest == request.request_digest
        elif representation_id == R_HIST_RANDOM_TANH:
            valid = (
                manifest.output_dim == self.historical_output_dim
                and manifest.seed == self.historical_seed
                and manifest.protocol_digest == self.historical_protocol_digest
                and manifest.checkpoint_digest == self.historical_checkpoint_digest
            )
        else:
            raise RepresentationPlanError(
                f"representation is outside the v0.3 signal plan: {representation_id}"
            )
        if not valid:
            raise RepresentationPlanError(
                f"representation manifest drifted from freeze: {representation_id}"
            )


__all__ = [
    "REPRESENTATION_EXECUTION_PLAN_SCHEMA",
    "RepresentationExecutionPlan",
    "RepresentationPlanError",
]
